from __future__ import annotations

import io
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

from zerttracker.models import UnderlyingMetrics

logger = logging.getLogger(__name__)


_STOOQ_INDEX_MAP = {
    "^STOXX50E": "^stx50",
    "^GDAXI": "^dax",
    "^GSPC": "^spx",
    "^IXIC": "^ndx",
    "^FTSE": "^ftm",
}


def fetch_underlying(ticker: str, *, max_retries: int = 3) -> Optional[UnderlyingMetrics]:
    if not ticker:
        return None

    for attempt in range(max_retries):
        result = _fetch_yfinance(ticker)
        if result is not None:
            return result
        if attempt < max_retries - 1:
            time.sleep(0.8 * (2 ** attempt))

    logger.info("yfinance exhausted for %s, falling back to stooq", ticker)
    return _fetch_stooq(ticker)


def _fetch_yfinance(ticker: str) -> Optional[UnderlyingMetrics]:
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5y", auto_adjust=True)
        if hist.empty:
            return None
        close = hist["Close"].dropna()
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        return _build_metrics(ticker, close, info)
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        return None


def _fetch_stooq(ticker: str) -> Optional[UnderlyingMetrics]:
    stooq_ticker = _to_stooq_ticker(ticker)
    if not stooq_ticker:
        return None
    url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
    try:
        resp = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (zerttracker)"},
        )
        if resp.status_code != 200 or not resp.text:
            return None
        text = resp.text.strip()
        if text.lower().startswith("no data") or "Date" not in text.split("\n", 1)[0]:
            return None
        df = pd.read_csv(io.StringIO(text))
        if df.empty or "Close" not in df.columns:
            return None
        df = df.dropna(subset=["Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        close = df.set_index("Date")["Close"].sort_index()
        close = close.tail(252 * 5)
        if close.empty:
            return None
        return _build_metrics(ticker, close, {})
    except Exception as exc:
        logger.warning("stooq failed for %s: %s", ticker, exc)
        return None


def _to_stooq_ticker(yahoo_ticker: str) -> Optional[str]:
    if yahoo_ticker in _STOOQ_INDEX_MAP:
        return _STOOQ_INDEX_MAP[yahoo_ticker]
    if yahoo_ticker.startswith("^"):
        return None
    return yahoo_ticker.lower()


def _build_metrics(ticker: str, close: pd.Series, info: dict) -> UnderlyingMetrics:
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) >= 30:
        hv_1y = float(log_returns.tail(252).std() * np.sqrt(252))
        rv_3m = float(log_returns.tail(63).std() * np.sqrt(252))
    else:
        hv_1y = float(log_returns.std() * np.sqrt(252)) if len(log_returns) > 1 else 0.25
        rv_3m = None

    spot = float(close.iloc[-1])
    max_dd = _max_drawdown(close)
    ret_1y = float(close.iloc[-1] / close.iloc[-min(252, len(close))] - 1.0)

    return UnderlyingMetrics(
        ticker=ticker,
        spot=spot,
        historical_vol_1y=hv_1y,
        realized_vol_3m=rv_3m,
        dividend_yield=_safe(info.get("dividendYield")),
        beta=_safe(info.get("beta")),
        pe_ratio=_safe(info.get("trailingPE")),
        forward_pe=_safe(info.get("forwardPE")),
        pb_ratio=_safe(info.get("priceToBook")),
        market_cap=_safe(info.get("marketCap")),
        sector=info.get("sector"),
        max_drawdown_5y=max_dd,
        return_1y=ret_1y,
    )


def _safe(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _max_drawdown(close: pd.Series) -> float:
    cummax = close.cummax()
    dd = close / cummax - 1.0
    return float(dd.min())
