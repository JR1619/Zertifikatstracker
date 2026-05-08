from __future__ import annotations

import logging
from datetime import date
from io import StringIO
from typing import Optional

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

from zerttracker.models import MacroSnapshot

logger = logging.getLogger(__name__)


ECB_BASE = "https://data-api.ecb.europa.eu/service/data"


def _ecb_last(series_key: str) -> Optional[float]:
    url = f"{ECB_BASE}/{series_key}?format=csvdata&lastNObservations=1"
    try:
        r = requests.get(url, timeout=10, headers={"Accept": "text/csv"})
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if "OBS_VALUE" in df.columns and len(df) > 0:
            return float(df["OBS_VALUE"].iloc[-1])
    except Exception as exc:
        logger.debug("ECB fetch failed for %s: %s", series_key, exc)
    return None


def _yahoo_last(ticker: str) -> Optional[float]:
    if yf is None:
        return None
    try:
        h = yf.Ticker(ticker).history(period="1mo")
        if h.empty:
            return None
        return float(h["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.debug("Yahoo fetch failed for %s: %s", ticker, exc)
        return None


def fetch_macro() -> MacroSnapshot:
    rf_3m = _ecb_last("FM/D.U2.EUR.RT.MM.EURIBOR3MD_.HSTA")
    rf_10y = _ecb_last("FM/D.U2.EUR.4F.BB.U2_10Y.YLD")
    if rf_10y is None:
        rf_10y = _yahoo_last("^TNX")
        if rf_10y is not None:
            rf_10y = rf_10y / 100.0 * 100.0
    if rf_3m is None:
        rf_3m = 3.0
    if rf_10y is None:
        rf_10y = 2.5

    rf_3m = rf_3m / 100.0 if rf_3m > 1 else rf_3m
    rf_10y = rf_10y / 100.0 if rf_10y > 1 else rf_10y

    vstoxx = _yahoo_last("^V2TX") or _yahoo_last("^VSTX")
    inflation = _ecb_last("ICP/M.U2.N.000000.4.ANR")

    return MacroSnapshot(
        as_of=date.today(),
        risk_free_eur_3m=rf_3m,
        risk_free_eur_10y=rf_10y,
        yield_curve_slope=rf_10y - rf_3m,
        vstoxx=vstoxx,
        eur_inflation_yoy=inflation,
    )
