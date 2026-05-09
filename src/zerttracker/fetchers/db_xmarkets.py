"""DB X-markets scraper.

Versucht die offizielle DB-X-markets-Produktsuche für Express-Zertifikate
abzufragen. Da DB Server-side rendering verwendet (Stand Mai 2026), parsen wir
HTML mit BeautifulSoup. Der Scraper ist defensiv: wenn ein Selektor nicht
matcht, gibt er logisch leere Ergebnisse zurück statt zu crashen.

Dies ist die EINZIGE Datenquelle für reale DB-Express-Zertifikatsmetadaten in
diesem Projekt. yfinance liefert nur Underlying-Kurse, ECB nur Makrodaten.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from zerttracker.models import CertificateType, ExpressCertificate, ObservationDate

logger = logging.getLogger(__name__)


BASE_URL = "https://www.xmarkets.db.com"
SEARCH_PATHS = [
    "/DE/Produkt_Uebersicht/Express-Zertifikate_Klass?pstate=AllActive",
    "/DE/Produkt_Uebersicht/Express-Zertifikate_Memory?pstate=AllActive",
    "/DE/Produkt_Uebersicht/Zertifikate?pstate=AllActive",
]
DETAIL_PATH_TEMPLATES = [
    "/DE/Produkt_Detail/{isin}",
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


@dataclass
class ScrapeStats:
    listing_url: Optional[str] = None
    listing_status: Optional[int] = None
    products_found_in_listing: int = 0
    detail_pages_attempted: int = 0
    detail_pages_parsed: int = 0
    parse_errors: list[str] = None

    def __post_init__(self):
        if self.parse_errors is None:
            self.parse_errors = []


def fetch_db_xmarkets(
    *, max_products: int = 200, sleep_between: float = 0.6, verbose: bool = True
) -> tuple[list[ExpressCertificate], ScrapeStats]:
    """Scrape Express-Zertifikate von DB X-markets.

    Returns: (Liste an Certificates, ScrapeStats für Debugging).
    Stats sind bewusst Teil des Returns, damit man bei Problemen sehen kann,
    ob die Listing-URL überhaupt erreicht wurde.
    """
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    stats = ScrapeStats()

    isins: list[str] = []
    seen: set[str] = set()
    for path in SEARCH_PATHS:
        url = urljoin(BASE_URL, path)
        try:
            r = session.get(url, timeout=20, allow_redirects=True)
        except requests.RequestException as exc:
            logger.info("Listing %s failed: %s", url, exc)
            continue
        stats.listing_status = r.status_code
        stats.listing_url = url
        if r.status_code != 200 or len(r.text) < 1000:
            continue
        new = [x for x in _extract_isins_from_listing(r.text) if x not in seen]
        for x in new:
            seen.add(x)
        isins.extend(new)
        logger.info("Listing %s -> %d neue ISINs (gesamt %d)", path, len(new), len(isins))
    stats.products_found_in_listing = len(isins)

    if not isins:
        logger.warning("Listing erreicht (200) aber keine ISINs erkannt — Regex-Problem oder andere Seitenstruktur.")
        return [], stats


    isins = isins[:max_products]
    certs: list[ExpressCertificate] = []
    for i, isin in enumerate(isins):
        stats.detail_pages_attempted += 1
        try:
            cert = _fetch_detail(session, isin)
        except Exception as exc:
            stats.parse_errors.append(f"{isin}: {type(exc).__name__}: {exc}")
            cert = None
        if cert is not None:
            certs.append(cert)
            stats.detail_pages_parsed += 1
        if i < len(isins) - 1:
            time.sleep(sleep_between)

    return certs, stats


def _fetch_first_working(
    session: requests.Session, paths: list[str], stats: ScrapeStats
) -> tuple[Optional[str], Optional[str]]:
    for path in paths:
        url = urljoin(BASE_URL, path)
        try:
            r = session.get(url, timeout=20, allow_redirects=True)
        except requests.RequestException as exc:
            stats.parse_errors.append(f"GET {url}: {exc}")
            continue
        stats.listing_status = r.status_code
        stats.listing_url = url
        if r.status_code == 200 and len(r.text) > 1000:
            return r.text, url
    return None, None


def _extract_isins_from_listing(html: str) -> list[str]:
    pattern = re.compile(r"\bDE000[A-Z0-9]{7}\b")
    seen: set[str] = set()
    out: list[str] = []
    for m in pattern.finditer(html):
        isin = m.group(0)
        if isin not in seen:
            seen.add(isin)
            out.append(isin)
    return out


def _fetch_detail(session: requests.Session, isin: str) -> Optional[ExpressCertificate]:
    for tmpl in DETAIL_PATH_TEMPLATES:
        url = urljoin(BASE_URL, tmpl.format(isin=isin))
        try:
            r = session.get(url, timeout=20, allow_redirects=True)
        except requests.RequestException as exc:
            logger.debug("Detail %s failed: %s", url, exc)
            continue
        if r.status_code != 200 or len(r.text) < 500:
            continue
        cert = _parse_detail_html(isin, r.text)
        if cert is not None:
            return cert
    return None


def _extract_column_table(soup: BeautifulSoup) -> dict[str, str]:
    """Parse <td class="column0">Label</td><td class="column1">Value</td> pairs."""
    out: dict[str, str] = {}
    for label_cell in soup.select("td.column0"):
        label = label_cell.get_text(" ", strip=True)
        value_cell = label_cell.find_next_sibling("td")
        value = value_cell.get_text(" ", strip=True) if value_cell else ""
        if label and label not in out:
            out[label] = value
    return out


def _extract_observation_rows(soup: BeautifulSoup) -> list[list[str]]:
    """Find the table under <H2>Beobachtungstage</H2> and return rows as lists of strings."""
    header = None
    for h in soup.find_all(["h2", "H2", "h3", "h1"]):
        if "Beobachtungstag" in h.get_text():
            header = h
            break
    if header is None:
        return []
    container = header
    table = None
    for _ in range(6):
        container = container.find_parent() or container
        table = container.find("table")
        if table:
            break
    if table is None:
        return []
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    return rows


def _parse_detail_html(isin: str, html: str) -> Optional[ExpressCertificate]:
    soup = BeautifulSoup(html, "lxml")
    fields = _extract_column_table(soup)

    def f(*keys: str) -> Optional[str]:
        for k in keys:
            for actual_k, v in fields.items():
                if k.lower() in actual_k.lower() and v and v != "-":
                    return v
        return None

    title_text = _extract_title(soup) or ""
    has_memory = "memory" in title_text.lower() or "memory" in html.lower()
    name = title_text.split(" | ")[1] if " | " in title_text else (title_text or f"DB Express {isin}")

    underlying_name = "?"
    m = re.search(r"Basiswert\s*\(([^)]+)\)", html)
    if m:
        underlying_name = m.group(1).strip()
    underlying_name = f(*[k for k in ("Basiswert", "Underlying")]) or underlying_name

    wkn = f("WKN")
    issue = _to_date(f("Emissionstag", "Erster Handelstag"))
    maturity = _to_date(f("Laufzeit", "Fälligkeit", "Letzter Bewertungstag"))
    initial = _to_float(f("Basisreferenzstand", "Anfänglicher Referenzpreis", "Startwert", "Erstes Fixing"))
    bid = _to_float(f("Geld", "Bid"))
    ask = _to_float(f("Brief", "Ask"))
    last = _to_float(f("Letzter Kurs", "Letzter"))

    observations = _observations_from_rows(_extract_observation_rows(soup), initial)
    nominal = initial if initial else 100.0

    if not observations or initial is None or maturity is None or issue is None:
        logger.debug("Skip %s: obs=%d init=%s issue=%s mat=%s",
                     isin, len(observations), initial, issue, maturity)
        return None

    knockin = observations[-1].autocall_level

    underlying_ticker = _guess_ticker(underlying_name)

    try:
        return ExpressCertificate(
            isin=isin,
            wkn=wkn,
            name=name[:120],
            issuer="Deutsche Bank",
            cert_type=CertificateType.EXPRESS_MEMORY if has_memory else CertificateType.EXPRESS,
            underlying_name=underlying_name,
            underlying_ticker=underlying_ticker,
            initial_fixing=initial,
            issue_date=issue,
            maturity_date=maturity,
            observations=observations,
            knock_in_barrier=knockin,
            nominal=nominal,
            currency="EUR",
            bid=bid,
            ask=ask,
            last=last,
            has_memory=has_memory,
        )
    except Exception as exc:
        logger.debug("Validation failed for %s: %s", isin, exc)
        return None


def _observations_from_rows(rows: list[list[str]], initial: Optional[float]) -> list[ObservationDate]:
    out: list[ObservationDate] = []
    if not rows or not initial or initial <= 0:
        return out
    for cells in rows[1:]:
        if len(cells) < 2:
            continue
        d = _to_date(cells[0])
        if d is None:
            continue
        threshold_eur = _to_float(cells[1])
        payout_eur = _to_float(cells[2]) if len(cells) > 2 else None
        if threshold_eur is None:
            continue
        autocall_level = threshold_eur / initial
        coupon_level = autocall_level
        coupon_amount = (payout_eur - initial) if payout_eur is not None else 0.0
        out.append(ObservationDate(
            date=d,
            autocall_level=autocall_level,
            coupon_level=coupon_level,
            coupon_amount=max(0.0, coupon_amount),
        ))
    return out


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else None


_GERMAN_TICKER_HINTS = {
    "sap": "SAP.DE",
    "allianz": "ALV.DE",
    "siemens": "SIE.DE",
    "bmw": "BMW.DE",
    "basf": "BAS.DE",
    "volkswagen": "VOW3.DE",
    "deutsche bank": "DBK.DE",
    "bayer": "BAYN.DE",
    "munich re": "MUV2.DE",
    "münchener rück": "MUV2.DE",
    "mercedes": "MBG.DE",
    "porsche": "P911.DE",
    "infineon": "IFX.DE",
    "deutsche telekom": "DTE.DE",
    "telekom": "DTE.DE",
    "rwe": "RWE.DE",
    "e.on": "EOAN.DE",
    "adidas": "ADS.DE",
    "puma": "PUM.DE",
    "lufthansa": "LHA.DE",
    "fresenius": "FRE.DE",
    "henkel": "HEN3.DE",
    "covestro": "1COV.DE",
    "continental": "CON.DE",
    "daimler truck": "DTG.DE",
    "deutsche post": "DHL.DE",
    "dhl": "DHL.DE",
    "merck": "MRK.DE",
    "qiagen": "QIA.DE",
    "vonovia": "VNA.DE",
    "deutsche boerse": "DB1.DE",
    "deutsche börse": "DB1.DE",
    "commerzbank": "CBK.DE",
    "ing": "INGA.AS",
    "bnp": "BNP.PA",
    "santander": "SAN.MC",
    "nestl": "NESN.SW",
    "novartis": "NOVN.SW",
    "roche": "ROG.SW",
    "lvmh": "MC.PA",
    "loreal": "OR.PA",
    "l'oreal": "OR.PA",
    "airbus": "AIR.PA",
    "totalenergies": "TTE.PA",
    "sanofi": "SAN.PA",
    "siemens energy": "ENR.DE",
    "siemens healthineers": "SHL.DE",
    "deutsche wohnen": "DWNI.DE",
    "delivery hero": "DHER.DE",
    "zalando": "ZAL.DE",
    "hugo boss": "BOSS.DE",
    "thyssenkrupp": "TKA.DE",
    "k+s": "SDF.DE",
    "ms ci": "URTH",
    "msci world": "URTH",
    "dax": "^GDAXI",
    "euro stoxx 50": "^STOXX50E",
    "eurostoxx 50": "^STOXX50E",
    "stoxx europe 600": "^STOXX",
    "s&p 500": "^GSPC",
    "nasdaq": "^IXIC",
}


def _guess_ticker(name: str) -> str:
    n = name.lower()
    for key, ticker in _GERMAN_TICKER_HINTS.items():
        if key in n:
            return ticker
    return ""


def _to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = re.sub(r"[€$%]|EUR|USD|CHF|GBP", "", s, flags=re.I).strip()
    s = re.sub(r"[\s ]+", "", s)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _to_pct(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    f = _to_float(s)
    if f is None:
        return None
    return f / 100.0 if f > 1.5 else f


def _to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None
