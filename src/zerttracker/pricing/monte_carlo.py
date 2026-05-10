"""Monte-Carlo pricer fuer Express-Zertifikate.

Aenderungen gegenueber V1:
- Antithetic variates (Halbierung der MC-Varianz fuer symmetrische Payoffs)
- 50k Pfade default
- Diskontierung ueber pricing.discount.SwapCurve + Issuer-Spread (vorher pure r)
- Vola via pricing.vol.pricing_vol (realized + 3pp), vorher nur realized
- Output: zusaetzlich mc_standard_error, ci_95_low/high, cvar_95
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np

from zerttracker.models import ExpressCertificate, MacroSnapshot, UnderlyingMetrics
from zerttracker.pricing.discount import (
    SwapCurve,
    issuer_spread_bp,
    make_swap_curve,
)
from zerttracker.pricing.vol import pricing_vol


@dataclass
class MCResult:
    fair_value: float
    expected_payoff: float
    expected_payoff_undiscounted: float
    expected_holding_period_years: float
    prob_autocall_first: float
    prob_full_coupons: float
    prob_capital_loss: float
    prob_barrier_breach: float
    expected_loss_given_breach: float
    mc_standard_error: float
    ci_95_low: float
    ci_95_high: float
    cvar_95: float
    sigma_used: float
    n_paths: int


def price_express(
    cert: ExpressCertificate,
    underlying: UnderlyingMetrics,
    macro: MacroSnapshot,
    *,
    valuation_date: Optional[date] = None,
    n_paths: int = 50_000,
    steps_per_year: int = 252,
    seed: Optional[int] = 42,
    physical_drift: Optional[float] = None,
    swap_curve: Optional[SwapCurve] = None,
    issuer_spread_override_bp: Optional[float] = None,
    vol_premium: Optional[float] = None,
) -> MCResult:
    """Monte-Carlo-Pricing fuer Express-Zertifikate.

    physical_drift: wenn None -> risk-neutral (drift = r aus Kurve), sonst real-world.
    swap_curve: wenn None -> aus Macro-Snapshot konstruiert.
    issuer_spread_override_bp: wenn None -> aus YAML.
    vol_premium: wenn None -> Default 3pp.
    """
    valuation_date = valuation_date or date.today()
    rng = np.random.default_rng(seed)
    n_paths = int(n_paths)
    if n_paths % 2 != 0:
        n_paths += 1

    future_obs = [o for o in cert.observations if o.date > valuation_date]
    if not future_obs:
        return _trivial_result(cert)

    if swap_curve is None:
        swap_curve = make_swap_curve(macro.risk_free_eur_3m, macro.risk_free_eur_10y)
    if issuer_spread_override_bp is None:
        issuer_spread_override_bp = issuer_spread_bp(cert.issuer)

    realized_vol = float(underlying.historical_vol_1y or 0.20)
    sigma = pricing_vol(realized_vol, premium=vol_premium if vol_premium is not None else 0.03)

    q = float(underlying.dividend_yield or 0.0)

    s0 = float(underlying.spot)
    initial = float(cert.initial_fixing)
    nominal = cert.nominal
    knock_in_abs = cert.knock_in_barrier * initial

    times = np.array([(o.date - valuation_date).days / 365.0 for o in future_obs])
    if (times <= 0).any():
        times = np.maximum(times, 1e-6)

    discount_factors = _make_discount_factors(times, swap_curve, issuer_spread_override_bp)

    n_obs = len(future_obs)
    obs_levels = np.empty((n_paths, n_obs))
    barrier_breached = np.zeros(n_paths, dtype=bool)
    s_curr = np.full(n_paths, s0)
    prev_t = 0.0

    half = n_paths // 2

    for i, t in enumerate(times):
        dt = t - prev_t
        n_steps = max(1, int(np.ceil(dt * steps_per_year)))
        step_dt = dt / n_steps
        for _ in range(n_steps):
            z_half = rng.standard_normal(half)
            z = np.concatenate([z_half, -z_half])
            if physical_drift is not None:
                drift_pa = physical_drift
            else:
                drift_pa = swap_curve.zero_rate(t)
            s_curr = s_curr * np.exp(
                (drift_pa - q - 0.5 * sigma ** 2) * step_dt
                + sigma * np.sqrt(step_dt) * z
            )
            barrier_breached |= s_curr < knock_in_abs
        obs_levels[:, i] = s_curr
        prev_t = t

    discounted_payoff = np.zeros(n_paths)
    undiscounted_payoff = np.zeros(n_paths)
    autocall_time = np.full(n_paths, np.nan)
    coupons_received = np.zeros(n_paths)
    undiscounted_coupons = np.zeros(n_paths)
    autocalled_first = np.zeros(n_paths, dtype=bool)
    active = np.ones(n_paths, dtype=bool)
    deferred_coupons = np.zeros(n_paths)
    total_coupon_periods = sum(1 for o in future_obs if o.coupon_amount > 0)
    coupon_periods_paid = np.zeros(n_paths, dtype=int)

    for i, obs in enumerate(future_obs):
        t = times[i]
        df = discount_factors[i]

        autocall_abs = obs.autocall_level * initial
        coupon_abs = obs.coupon_level * initial

        if active.any():
            level = obs_levels[:, i]
            pays_coupon = active & (level >= coupon_abs) & (obs.coupon_amount > 0)
            if cert.has_memory:
                paid_now = obs.coupon_amount + np.where(pays_coupon, deferred_coupons, 0.0)
                coupons_received[pays_coupon] += df * paid_now[pays_coupon]
                undiscounted_coupons[pays_coupon] += paid_now[pays_coupon]
                deferred_coupons[pays_coupon] = 0.0
                missed = active & ~pays_coupon & (obs.coupon_amount > 0)
                deferred_coupons[missed] += obs.coupon_amount
            else:
                coupons_received[pays_coupon] += df * obs.coupon_amount
                undiscounted_coupons[pays_coupon] += obs.coupon_amount
            coupon_periods_paid[pays_coupon] += 1

            triggers = active & (level >= autocall_abs)
            if i < n_obs - 1:
                discounted_payoff[triggers] += df * nominal
                undiscounted_payoff[triggers] += nominal
                autocall_time[triggers] = t
                if i == 0:
                    autocalled_first[triggers] = True
                active &= ~triggers

        if i == n_obs - 1 and active.any():
            level = obs_levels[active, i]
            redemption = np.where(
                (level >= autocall_abs) | ((level >= initial) & ~barrier_breached[active]),
                nominal,
                nominal * (level / initial),
            )
            discounted_payoff_active = df * redemption
            idx_active = np.where(active)[0]
            discounted_payoff[idx_active] += discounted_payoff_active
            undiscounted_payoff[idx_active] += redemption
            autocall_time[idx_active] = t
            active[:] = False

    discounted_total = discounted_payoff + coupons_received
    undiscounted_total = undiscounted_payoff + undiscounted_coupons

    fair_value = float(np.mean(discounted_total))
    mc_se = float(np.std(discounted_total, ddof=1) / np.sqrt(n_paths))
    ci_lo = fair_value - 1.96 * mc_se
    ci_hi = fair_value + 1.96 * mc_se
    expected_payoff_undiscounted = float(np.mean(undiscounted_total))
    expected_horizon = float(np.nanmean(autocall_time)) if not np.all(np.isnan(autocall_time)) else float(times[-1])

    sorted_payoff = np.sort(discounted_total)
    cutoff = max(1, int(0.05 * n_paths))
    cvar_95 = float(np.mean(sorted_payoff[:cutoff]))

    if total_coupon_periods > 0:
        full_coupons = coupon_periods_paid == total_coupon_periods
    else:
        full_coupons = np.ones(n_paths, dtype=bool)

    redeemed_at_loss = undiscounted_payoff < nominal
    prob_capital_loss = float(np.mean(redeemed_at_loss))
    prob_barrier_breach = float(np.mean(barrier_breached))
    if barrier_breached.any():
        breached_payoff = undiscounted_payoff[barrier_breached]
        expected_loss = float(np.mean(np.maximum(0.0, nominal - breached_payoff)) / nominal)
    else:
        expected_loss = 0.0

    prob_autocall_first = float(np.mean(autocalled_first))
    prob_full_coupons = float(np.mean(full_coupons))

    return MCResult(
        fair_value=fair_value,
        expected_payoff=fair_value,
        expected_payoff_undiscounted=expected_payoff_undiscounted,
        expected_holding_period_years=expected_horizon,
        prob_autocall_first=prob_autocall_first,
        prob_full_coupons=prob_full_coupons,
        prob_capital_loss=prob_capital_loss,
        prob_barrier_breach=prob_barrier_breach,
        expected_loss_given_breach=expected_loss,
        mc_standard_error=mc_se,
        ci_95_low=ci_lo,
        ci_95_high=ci_hi,
        cvar_95=cvar_95,
        sigma_used=sigma,
        n_paths=n_paths,
    )


def _make_discount_factors(times, swap, issuer_spread_bp_value):
    rates = np.array([swap.zero_rate(float(t)) for t in times])
    return np.exp(-(rates + issuer_spread_bp_value / 10_000.0) * times)


def estimate_physical_drift(underlying: UnderlyingMetrics) -> float:
    """Bayesian-style blend: 60% long-run equity baseline + 40% recent 1y return."""
    baseline = 0.06
    if underlying.return_1y is not None and -0.40 <= underlying.return_1y <= 0.60:
        mu = 0.6 * baseline + 0.4 * underlying.return_1y
    else:
        mu = baseline
    return float(np.clip(mu, -0.03, 0.12))


def _trivial_result(cert: ExpressCertificate) -> MCResult:
    return MCResult(
        fair_value=cert.nominal,
        expected_payoff=cert.nominal,
        expected_payoff_undiscounted=cert.nominal,
        expected_holding_period_years=0.0,
        prob_autocall_first=0.0,
        prob_full_coupons=0.0,
        prob_capital_loss=0.0,
        prob_barrier_breach=0.0,
        expected_loss_given_breach=0.0,
        mc_standard_error=0.0,
        ci_95_low=cert.nominal,
        ci_95_high=cert.nominal,
        cvar_95=cert.nominal,
        sigma_used=0.0,
        n_paths=0,
    )
