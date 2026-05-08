from __future__ import annotations

from datetime import date

import numpy as np

from zerttracker.models import ExpressCertificate, MacroSnapshot, UnderlyingMetrics
from zerttracker.pricing.monte_carlo import MCResult


def score_value(market_price: float | None, fair_value: float, nominal: float) -> float:
    if market_price is None or market_price <= 0:
        return 50.0
    discount = (fair_value - market_price) / nominal
    return float(np.clip(50.0 + discount * 500.0, 0.0, 100.0))


def score_risk_reward(mc: MCResult, cert: ExpressCertificate, market_price: float | None) -> float:
    base = market_price if market_price else cert.nominal
    if base <= 0:
        return 0.0

    horizon = max(mc.expected_holding_period_years, 0.25)
    expected_return_pa = (mc.expected_payoff_undiscounted / base) ** (1.0 / horizon) - 1.0
    risk_penalty = mc.prob_capital_loss * mc.expected_loss_given_breach
    raw = expected_return_pa - 1.5 * risk_penalty
    return float(np.clip(50.0 + raw * 400.0, 0.0, 100.0))


def score_underlying(u: UnderlyingMetrics) -> float:
    components: list[float] = []

    if u.pe_ratio and 0 < u.pe_ratio < 100:
        components.append(_band(u.pe_ratio, ideal=15, low=8, high=30))
    if u.forward_pe and 0 < u.forward_pe < 100:
        components.append(_band(u.forward_pe, ideal=14, low=8, high=28))
    if u.pb_ratio and u.pb_ratio > 0:
        components.append(_band(u.pb_ratio, ideal=2.0, low=0.8, high=6.0))
    if u.dividend_yield is not None:
        dy = u.dividend_yield * (100.0 if u.dividend_yield < 1 else 1.0)
        components.append(float(np.clip(40 + dy * 6, 0, 100)))
    if u.beta is not None:
        components.append(float(np.clip(100 - abs(u.beta - 0.9) * 50, 0, 100)))
    if u.return_1y is not None:
        components.append(float(np.clip(50 + u.return_1y * 100, 0, 100)))
    if u.max_drawdown_5y is not None:
        components.append(float(np.clip(100 + u.max_drawdown_5y * 100, 0, 100)))
    if u.historical_vol_1y:
        components.append(float(np.clip(100 - (u.historical_vol_1y - 0.15) * 200, 0, 100)))

    if not components:
        return 50.0
    return float(np.mean(components))


def score_macro(m: MacroSnapshot) -> float:
    components: list[float] = []
    components.append(float(np.clip(50 + m.yield_curve_slope * 1000, 0, 100)))
    if m.vstoxx is not None:
        components.append(float(np.clip(100 - (m.vstoxx - 18) * 3, 0, 100)))
    if m.eur_inflation_yoy is not None:
        infl = m.eur_inflation_yoy if m.eur_inflation_yoy > 1 else m.eur_inflation_yoy * 100
        components.append(float(np.clip(100 - abs(infl - 2.0) * 15, 0, 100)))
    return float(np.mean(components))


def combined_score(value: float, risk_reward: float, underlying: float, macro: float) -> float:
    return float(0.35 * risk_reward + 0.25 * value + 0.25 * underlying + 0.15 * macro)


def expected_return_pa(mc: MCResult, market_price: float | None, nominal: float) -> float:
    base = market_price if market_price else nominal
    if base <= 0:
        return 0.0
    horizon = max(mc.expected_holding_period_years, 0.25)
    return float((mc.expected_payoff_undiscounted / base) ** (1.0 / horizon) - 1.0)


def days_until_next_observation(cert: ExpressCertificate, today: date | None = None) -> int | None:
    today = today or date.today()
    upcoming = [o.date for o in cert.observations if o.date > today]
    if not upcoming:
        return None
    return (min(upcoming) - today).days


def _band(value: float, ideal: float, low: float, high: float) -> float:
    if value <= low or value >= high:
        return 20.0
    if value <= ideal:
        return float(np.clip(20 + (value - low) / (ideal - low) * 80, 0, 100))
    return float(np.clip(100 - (value - ideal) / (high - ideal) * 80, 0, 100))
