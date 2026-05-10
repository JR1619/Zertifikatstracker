"""Sanity-Tests fuer den Black-Scholes Pricer und MC-Konvergenz."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerttracker.pricing.black_scholes import bs_call, bs_put, mc_european_call


def test_bs_call_atm_no_div():
    """ATM Call mit S=K=100, T=1, r=0.05, sigma=0.2 -> bekannter Referenzwert ~10.45."""
    price = bs_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
    assert abs(price - 10.4506) < 0.01


def test_bs_put_call_parity():
    """C - P = S*exp(-qT) - K*exp(-rT)."""
    S, K, T, r, sigma, q = 110.0, 100.0, 0.5, 0.04, 0.25, 0.02
    c = bs_call(S, K, T, r, sigma, q)
    p = bs_put(S, K, T, r, sigma, q)
    parity = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs((c - p) - parity) < 1e-6


def test_bs_call_zero_T():
    assert bs_call(120, 100, 0, 0.05, 0.2) == 20.0


def test_mc_call_matches_bs():
    """MC-Sampler muss den BS-Call bis auf 2-Sigma treffen."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = bs_call(S, K, T, r, sigma)
    mc, se = mc_european_call(S, K, T, r, sigma, n_paths=100_000, seed=7)
    assert abs(mc - bs) < 3 * se, f"MC={mc:.4f} BS={bs:.4f} SE={se:.4f}"


def test_mc_antithetic_estimate_close_to_bs():
    """Antithetic-MC sollte ATM-Call praezise treffen."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = bs_call(S, K, T, r, sigma)
    mc, _ = mc_european_call(S, K, T, r, sigma, n_paths=50_000, seed=11, antithetic=True)
    assert abs(mc - bs) < 0.10
