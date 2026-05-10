"""Boerse Frankfurt API Client fuer Bid/Ask/Last von Zertifikaten.

Diese API ist undokumentiert aber stabil; wir setzen primaer auf den
"price_information"-Endpoint, der pro ISIN Last/Bid/Ask liefert.

Anti-Bot-Auth (signed headers) ist optional und nur noetig wenn die
unauthenticated Variante 401/403 liefert. Erst probieren ohne, dann mit.

Wenn alles fehlschlaegt: leere Antwort, Pipeline laeuft mit fehlenden
Marktpreisen weiter.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.boerse-frankfurt.de/v1/data"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Origin": "https://www.boerse-frankfurt.de",
    "Referer": "https://www.boerse-frankfurt.de/",
}

# Hardcoded fallback salt; rotiert seitens Boerse Frankfurt regelmaessig.
# Wenn Anfragen mit signing fehlschlagen, ist meist dieser Salt veraltet.
# Aktualisierung: aus main JS bundle holen, regex auf 'salt' o.ae.
_KNOWN_SALT = "Wq2&AzN8gJ"


@dataclass
class Quote:
    isin: str
    last: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: Optional[str] = None


def _signed_headers(url: str) -> dict:
    now = datetime.now(timezone.utc)
    client_date = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    trace_input = (client_date + url + _KNOWN_SALT).encode("utf-8")
    x_traceid = hashlib.md5(trace_input).hexdigest()
    local_str = datetime.now().strftime("%Y%m%d%H%M")
    x_security = hashlib.md5(local_str.encode("utf-8")).hexdigest()
    return {
        "client-date": client_date,
        "x-client-traceid": x_traceid,
        "x-security": x_security,
    }


def _try_get(session: requests.Session, url: str, *, with_signing: bool) -> Optional[dict]:
    headers = dict(DEFAULT_HEADERS)
    if with_signing:
        headers.update(_signed_headers(url))
    try:
        r = session.get(url, headers=headers, timeout=10)
    except requests.RequestException as exc:
        logger.debug("BF GET %s failed: %s", url, exc)
        return None
    if r.status_code != 200:
        logger.debug("BF GET %s -> %d", url, r.status_code)
        return None
    try:
        return r.json()
    except Exception:
        return None


def fetch_quote(isin: str, *, session: Optional[requests.Session] = None) -> Optional[Quote]:
    """Holt Last/Bid/Ask fuer eine ISIN. None wenn nicht verfuegbar."""
    if session is None:
        session = requests.Session()
    url = f"{BASE_URL}/price_information?isin={isin}&mic=XFRA"

    data = _try_get(session, url, with_signing=False)
    if data is None:
        data = _try_get(session, url, with_signing=True)
    if data is None:
        return None

    last = _safe_float(data.get("lastPrice"))
    bid = _safe_float(data.get("bidPrice")) or _safe_float(data.get("bid"))
    ask = _safe_float(data.get("askPrice")) or _safe_float(data.get("ask"))
    ts = data.get("timestampLastPrice") or data.get("timestamp")
    if last is None and bid is None and ask is None:
        return None
    return Quote(isin=isin, last=last, bid=bid, ask=ask, timestamp=ts)


def fetch_quotes(isins: list[str]) -> dict[str, Quote]:
    session = requests.Session()
    out: dict[str, Quote] = {}
    for isin in isins:
        q = fetch_quote(isin, session=session)
        if q is not None:
            out[isin] = q
    return out


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
