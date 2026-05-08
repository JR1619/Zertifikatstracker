from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from zerttracker.models import UnderlyingMetrics

logger = logging.getLogger(__name__)


def fetch_underlying(ticker: str) -> Optional[UnderlyingMetrics]:
    if yf is None:
        logger.warning("yfinance not available")
        return None
    if not ticker:
        return None

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5y", auto_adjust=True)
        if hist.empty:
            return None

        close = hist["Close"].dropna()
        log_returns = np.log(close / close.shift(1)).dropna()
        hv_1y = float(log_returns.tail(252).std() * np.sqrt(252)) if len(log_returns) >= 30 else float(log_returns.std() * np.sqrt(252))
        rv_3m = float(log_returns.tail(63).std() * np.sqrt(252)) if len(log_returns) >= 30 else None

        spot = float(close.iloc[-1])
        max_dd = _max_drawdown(close)
        ret_1y = float(close.iloc[-1] / close.iloc[-min(252, len(close))] - 1.0)

        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

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
    except Exception as exc:
        logger.warning("Underlying fetch failed for %s: %s", ticker, exc)
        return None


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
