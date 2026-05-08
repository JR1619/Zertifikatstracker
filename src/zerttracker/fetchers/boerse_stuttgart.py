from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

from zerttracker.models import CertificateType, ExpressCertificate, ObservationDate

logger = logging.getLogger(__name__)


BOERSE_STUTTGART_SEARCH = "https://api.boerse-stuttgart.de/v1/public/instruments/search"
BOERSE_STUTTGART_DETAIL = "https://api.boerse-stuttgart.de/v1/public/instruments"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ZertTracker/0.1; +https://example.invalid)",
    "Accept": "application/json",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


@dataclass
class FetchResult:
    certificates: list[ExpressCertificate]
    source: str
    fetched_at: datetime


class BoerseStuttgartFetcher:
    """Free fetcher hitting the public boerse-stuttgart.de search.

    Endpoints are reverse-engineered from the public website and may change.
    Falls back to a CSV-based loader so the tool stays usable without network
    access or when the API breaks.
    """

    def __init__(self, session: Optional[requests.Session] = None, timeout: float = 15.0):
        self.session = session or requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        self.timeout = timeout

    def search_express_db(self, page_size: int = 100, max_pages: int = 5) -> list[dict]:
        results: list[dict] = []
        for page in range(max_pages):
            params = {
                "type": "certificate",
                "subType": "express",
                "issuer": "Deutsche Bank",
                "tradable": "true",
                "page": page,
                "pageSize": page_size,
            }
            try:
                r = self.session.get(BOERSE_STUTTGART_SEARCH, params=params, timeout=self.timeout)
                r.raise_for_status()
                payload = r.json()
            except (requests.RequestException, ValueError) as exc:
                logger.warning("BS search page %s failed: %s", page, exc)
                break

            items = payload.get("instruments") or payload.get("results") or []
            if not items:
                break
            results.extend(items)
            if len(items) < page_size:
                break
            time.sleep(0.4)
        return results

    def fetch_detail(self, isin: str) -> Optional[dict]:
        try:
            r = self.session.get(f"{BOERSE_STUTTGART_DETAIL}/{isin}", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("Detail fetch failed for %s: %s", isin, exc)
            return None

    def fetch(self) -> FetchResult:
        certificates: list[ExpressCertificate] = []
        try:
            raw = self.search_express_db()
        except Exception as exc:
            logger.exception("Boerse Stuttgart search failed: %s", exc)
            raw = []

        for item in raw:
            try:
                cert = _parse_summary(item)
                if cert is not None:
                    certificates.append(cert)
            except Exception as exc:
                logger.debug("Skipping unparseable item: %s", exc)

        return FetchResult(certificates=certificates, source="boerse-stuttgart", fetched_at=datetime.utcnow())


def _parse_summary(item: dict) -> Optional[ExpressCertificate]:
    isin = item.get("isin")
    name = item.get("name") or item.get("title")
    if not isin or not name:
        return None

    underlying = item.get("underlying") or {}
    obs_raw: Iterable[dict] = item.get("observations") or []
    observations = []
    for o in obs_raw:
        try:
            observations.append(
                ObservationDate(
                    date=_parse_date(o.get("date") or o.get("observationDate")),
                    autocall_level=float(o.get("autocallLevel", o.get("autocallBarrier", 1.0))),
                    coupon_level=float(o.get("couponLevel", o.get("couponBarrier", 1.0))),
                    coupon_amount=float(o.get("couponAmount", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue

    if not observations:
        return None

    issue = _parse_date(item.get("issueDate"))
    maturity = _parse_date(item.get("maturityDate") or item.get("expiryDate"))
    if issue is None or maturity is None:
        return None

    return ExpressCertificate(
        isin=isin,
        wkn=item.get("wkn"),
        name=name,
        issuer=item.get("issuer", "Deutsche Bank"),
        cert_type=CertificateType(item.get("subType", "express")),
        underlying_name=underlying.get("name", "?"),
        underlying_ticker=underlying.get("ticker") or underlying.get("symbol") or "",
        underlying_isin=underlying.get("isin"),
        initial_fixing=float(item.get("initialFixing", 0.0)),
        issue_date=issue,
        maturity_date=maturity,
        observations=observations,
        knock_in_barrier=float(item.get("knockInBarrier", item.get("barrier", 0.6))),
        nominal=float(item.get("nominal", 1000.0)),
        currency=item.get("currency", "EUR"),
        bid=_safe_float(item.get("bid")),
        ask=_safe_float(item.get("ask")),
        last=_safe_float(item.get("last")),
        has_memory=bool(item.get("memoryFeature", False)),
    )


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(v, fmt).date()
        except (TypeError, ValueError):
            continue
    return None


def load_csv(path: Path) -> FetchResult:
    """CSV-Fallback. Eine Zeile pro Beobachtungstermin.

    Erforderliche Spalten:
        isin, wkn, name, underlying_name, underlying_ticker,
        initial_fixing, issue_date, maturity_date,
        knock_in_barrier, nominal, currency,
        observation_date, autocall_level, coupon_level, coupon_amount,
        bid, ask, last, has_memory
    """
    df = pd.read_csv(path, parse_dates=["issue_date", "maturity_date", "observation_date"])
    certs: list[ExpressCertificate] = []
    for isin, group in df.groupby("isin", sort=False):
        first = group.iloc[0]
        observations = [
            ObservationDate(
                date=row["observation_date"].date(),
                autocall_level=float(row["autocall_level"]),
                coupon_level=float(row["coupon_level"]),
                coupon_amount=float(row["coupon_amount"]),
            )
            for _, row in group.sort_values("observation_date").iterrows()
        ]
        certs.append(
            ExpressCertificate(
                isin=str(isin),
                wkn=str(first.get("wkn", "")) or None,
                name=str(first["name"]),
                issuer=str(first.get("issuer", "Deutsche Bank")),
                cert_type=CertificateType(first.get("cert_type", "express")),
                underlying_name=str(first["underlying_name"]),
                underlying_ticker=str(first["underlying_ticker"]),
                underlying_isin=str(first.get("underlying_isin", "")) or None,
                initial_fixing=float(first["initial_fixing"]),
                issue_date=first["issue_date"].date(),
                maturity_date=first["maturity_date"].date(),
                observations=observations,
                knock_in_barrier=float(first["knock_in_barrier"]),
                nominal=float(first.get("nominal", 1000.0)),
                currency=str(first.get("currency", "EUR")),
                bid=_safe_float(first.get("bid")),
                ask=_safe_float(first.get("ask")),
                last=_safe_float(first.get("last")),
                has_memory=bool(first.get("has_memory", False)),
            )
        )
    return FetchResult(certificates=certs, source=f"csv:{path.name}", fetched_at=datetime.utcnow())


def fetch_with_fallback(csv_fallback: Optional[Path] = None) -> FetchResult:
    fetcher = BoerseStuttgartFetcher()
    result = fetcher.fetch()
    if result.certificates:
        return result
    if csv_fallback is not None and csv_fallback.exists():
        logger.info("Using CSV fallback at %s", csv_fallback)
        return load_csv(csv_fallback)
    return result
