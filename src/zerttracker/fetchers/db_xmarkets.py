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

    listing_html, listing_url = _fetch_first_working(session, SEARCH_PATHS, stats)
    if listing_html is None:
        logger.warning("Konnte keine Listing-Seite erreichen. Stats: %s", stats)
        return [], stats

    isins = _extract_isins_from_listing(listing_html)
    stats.products_found_in_listing = len(isins)
    if verbose:
        logger.info("Listing geladen (%s) — %d ISINs gefunden", listing_url, len(isins))

    if not isins:
        logger.warning("Listing erreicht (200) aber keine ISINs erkannt — Regex-Problem oder andere Seitenstruktur.")
        return [], stats

    if isins:
        first = isins[0]
        logger.info("=== PROBE first detail page %s ===", first)
        try:
            r = session.get(urljoin(BASE_URL, f"/DE/Produkt_Detail/{first}"), timeout=20)
            logger.info("Detail status=%d len=%d", r.status_code, len(r.text))
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "lxml")
                title = soup.find("title")
                logger.info("Detail title: %s", title.get_text(strip=True) if title else "MISSING")
                for needle in ["Basiswert", "Barriere", "Beobachtungstag", "Anfänglicher Referenzpreis",
                               "Letzter Bewertungstag", "Emissionstag", "Tilgungslevel", "Memory", first]:
                    if needle in r.text:
                        idx = r.text.find(needle)
                        logger.info("  contains %r at %d: %r", needle, idx,
                                    r.text[max(0, idx-30):idx+200])
                    else:
                        logger.info("  MISSING %r", needle)
        except Exception as exc:
            logger.warning("First-detail probe crashed: %s", exc)

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


def _parse_detail_html(isin: str, html: str) -> Optional[ExpressCertificate]:
    soup = BeautifulSoup(html, "lxml")

    def get_field(*labels: str) -> Optional[str]:
        for lbl in labels:
            el = soup.find(string=re.compile(rf"\b{re.escape(lbl)}\b", re.I))
            if not el:
                continue
            parent = el.find_parent(["tr", "div", "li", "dt", "th"])
            if parent is None:
                continue
            sibling = parent.find_next_sibling()
            if sibling and sibling.get_text(strip=True):
                return sibling.get_text(strip=True)
            cells = parent.find_all(["td", "dd", "span"])
            for c in cells:
                txt = c.get_text(strip=True)
                if txt and txt != lbl:
                    return txt
        return None

    name = get_field("Name", "Produktname") or _extract_title(soup) or f"DB Express {isin}"
    wkn = get_field("WKN")
    underlying_name = get_field("Basiswert", "Underlying") or "?"
    initial = _to_float(get_field("Anfänglicher Referenzpreis", "Startwert", "Initial Fixing"))
    knockin = _to_pct(get_field("Barriere", "Knock-In", "Sicherheitsschwelle"))
    issue = _to_date(get_field("Emissionstag", "Erster Handelstag", "Issue Date"))
    maturity = _to_date(get_field("Laufzeitende", "Fälligkeit", "Maturity"))
    bid = _to_float(get_field("Geld", "Bid"))
    ask = _to_float(get_field("Brief", "Ask"))
    last = _to_float(get_field("Letzter", "Last"))
    has_memory = bool(re.search(r"memory", html, re.I))
    nominal = _to_float(get_field("Nominalbetrag", "Nominal")) or 1000.0

    observations = _extract_observations(soup, initial)

    if not observations or initial is None or maturity is None or issue is None or knockin is None:
        return None

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


def _extract_observations(soup: BeautifulSoup, initial: Optional[float]) -> list[ObservationDate]:
    out: list[ObservationDate] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers or not any("beobachtung" in h or "termin" in h or "observation" in h for h in headers):
            continue
        for row in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            d = _to_date(cells[0])
            if d is None:
                continue
            ac = 1.0
            cl = 1.0
            ca = 0.0
            for c in cells[1:]:
                f = _to_pct(c)
                if f is not None and 0.3 <= f <= 1.5:
                    if ac == 1.0:
                        ac = f
                    else:
                        cl = f
                    continue
                v = _to_float(c)
                if v is not None and v > 1.5:
                    ca = v
            out.append(ObservationDate(date=d, autocall_level=ac, coupon_level=cl, coupon_amount=ca))
    return out


def _extract_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find(["h1", "h2"])
    return h1.get_text(strip=True) if h1 else None


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
    s = s.strip().replace(" ", "").replace(" ", "")
    s = re.sub(r"[€$%]", "", s)
    s = s.replace(".", "").replace(",", ".") if "," in s else s
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
