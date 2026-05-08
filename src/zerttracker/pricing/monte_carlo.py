from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np

from zerttracker.models import ExpressCertificate, MacroSnapshot, UnderlyingMetrics


@dataclass
class MCResult:
    fair_value: float
    expected_payoff: float
    expected_holding_period_years: float
    prob_autocall_first: float
    prob_full_coupons: float
    prob_capital_loss: float
    prob_barrier_breach: float
    expected_loss_given_breach: float


def price_express(
    cert: ExpressCertificate,
    underlying: UnderlyingMetrics,
    macro: MacroSnapshot,
    *,
    valuation_date: Optional[date] = None,
    n_paths: int = 20000,
    steps_per_year: int = 252,
    seed: Optional[int] = 42,
) -> MCResult:
    """Risk-neutral Monte-Carlo for express certificates.

    Underlying simulated as GBM with drift (r - q) and historical vol as
    proxy for implied vol (no options chain to avoid paid feeds).
    """
    valuation_date = valuation_date or date.today()
    rng = np.random.default_rng(seed)

    future_obs = [o for o in cert.observations if o.date > valuation_date]
    if not future_obs:
        return _trivial_result(cert)

    r = float(macro.risk_free_eur_3m if cert.maturity_date.year - valuation_date.year < 1 else macro.risk_free_eur_10y)
    q = float(underlying.dividend_yield or 0.0)
    sigma = float(underlying.historical_vol_1y or 0.20)
    sigma = max(0.05, min(sigma, 1.0))

    s0 = float(underlying.spot)
    initial = float(cert.initial_fixing)

    times = np.array([(o.date - valuation_date).days / 365.0 for o in future_obs])
    if (times <= 0).any():
        times = np.maximum(times, 1e-6)

    n_obs = len(future_obs)
    obs_levels = s0 * np.ones((n_paths, n_obs))
    prev_t = 0.0
    s_curr = np.full(n_paths, s0)

    barrier_breached = np.zeros(n_paths, dtype=bool)
    knock_in_abs = cert.knock_in_barrier * initial

    for i, t in enumerate(times):
        dt = t - prev_t
        n_steps = max(1, int(np.ceil(dt * steps_per_year)))
        step_dt = dt / n_steps
        for _ in range(n_steps):
            z = rng.standard_normal(n_paths)
            s_curr = s_curr * np.exp((r - q - 0.5 * sigma ** 2) * step_dt + sigma * np.sqrt(step_dt) * z)
            barrier_breached |= s_curr < knock_in_abs
        obs_levels[:, i] = s_curr
        prev_t = t

    nominal = cert.nominal
    payoff = np.zeros(n_paths)
    discounted_payoff = np.zeros(n_paths)
    autocall_time = np.full(n_paths, np.nan)
    coupons_received = np.zeros(n_paths)
    full_coupons = np.zeros(n_paths, dtype=bool)
    autocalled_first = np.zeros(n_paths, dtype=bool)

    active = np.ones(n_paths, dtype=bool)
    deferred_coupons = np.zeros(n_paths)
    total_coupon_periods = sum(1 for o in future_obs if o.coupon_amount > 0)
    coupon_periods_paid = np.zeros(n_paths, dtype=int)

    for i, obs in enumerate(future_obs):
        t = times[i]
        df = np.exp(-r * t)

        autocall_abs = obs.autocall_level * initial
        coupon_abs = obs.coupon_level * initial

        if active.any():
            level = obs_levels[:, i]
            pays_coupon = active & (level >= coupon_abs) & (obs.coupon_amount > 0)
            if cert.has_memory:
                memory_pay = pays_coupon
                paid_now = obs.coupon_amount + np.where(memory_pay, deferred_coupons, 0.0)
                coupons_received[memory_pay] += df * paid_now[memory_pay]
                deferred_coupons[memory_pay] = 0.0
                missed = active & ~pays_coupon & (obs.coupon_amount > 0)
                deferred_coupons[missed] += obs.coupon_amount
            else:
                coupons_received[pays_coupon] += df * obs.coupon_amount
            coupon_periods_paid[pays_coupon] += 1

            triggers = active & (level >= autocall_abs)
            if i < n_obs - 1:
                discounted_payoff[triggers] += df * nominal
                autocall_time[triggers] = t
                if i == 0:
                    autocalled_first[triggers] = triggers[triggers]
                active &= ~triggers

        if i == n_obs - 1 and active.any():
            level = obs_levels[active, i]
            df_final = df
            redemption = np.where(
                (level >= initial) | (~barrier_breached[active]),
                nominal,
                nominal * (level / initial),
            )
            discounted_payoff_active = df_final * redemption
            idx_active = np.where(active)[0]
            discounted_payoff[idx_active] += discounted_payoff_active
            autocall_time[idx_active] = t
            active[:] = False

    discounted_total = discounted_payoff + coupons_received
    fair_value = float(np.mean(discounted_total))
    expected_payoff = float(np.mean(discounted_payoff + coupons_received))
    expected_horizon = float(np.nanmean(autocall_time))

    if total_coupon_periods > 0:
        full_coupons = coupon_periods_paid == total_coupon_periods
    else:
        full_coupons = np.ones(n_paths, dtype=bool)

    redeemed_at_loss = discounted_payoff < (nominal * np.exp(-r * np.where(np.isnan(autocall_time), times[-1], autocall_time)))
    prob_capital_loss = float(np.mean(redeemed_at_loss))
    prob_barrier_breach = float(np.mean(barrier_breached))
    if barrier_breached.any():
        breached_payoff = discounted_payoff[barrier_breached] / np.exp(-r * np.where(np.isnan(autocall_time[barrier_breached]), times[-1], autocall_time[barrier_breached]))
        expected_loss = float(np.mean(np.maximum(0.0, nominal - breached_payoff)) / nominal)
    else:
        expected_loss = 0.0

    prob_autocall_first = float(np.mean(autocalled_first))
    prob_full_coupons = float(np.mean(full_coupons))

    return MCResult(
        fair_value=fair_value,
        expected_payoff=expected_payoff,
        expected_holding_period_years=expected_horizon,
        prob_autocall_first=prob_autocall_first,
        prob_full_coupons=prob_full_coupons,
        prob_capital_loss=prob_capital_loss,
        prob_barrier_breach=prob_barrier_breach,
        expected_loss_given_breach=expected_loss,
    )


def _trivial_result(cert: ExpressCertificate) -> MCResult:
    return MCResult(
        fair_value=cert.nominal,
        expected_payoff=cert.nominal,
        expected_holding_period_years=0.0,
        prob_autocall_first=0.0,
        prob_full_coupons=0.0,
        prob_capital_loss=0.0,
        prob_barrier_breach=0.0,
        expected_loss_given_breach=0.0,
    )
