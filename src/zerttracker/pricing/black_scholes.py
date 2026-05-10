"""Closed-form Black-Scholes pricers — used as analytical reference for MC tests.

Not used in production scoring (Express-Zertifikate sind path-dependent), nur
zum Sanity-Check der MC-Engine: ein European Call/Put unter GBM mit konstanter
Vola muss von der MC bis auf Standardfehler getroffen werden.
"""
from __future__ import annotations

import math

from scipy.stats import norm


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1))


def mc_european_call(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
    *, n_paths: int = 50_000, seed: int = 42, antithetic: bool = True,
) -> tuple[float, float]:
    """Returns (mean, standard_error) of a European call MC under GBM.

    Used purely to validate the MC sampler against the BS closed form.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)
    S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    payoff = np.maximum(S_T - K, 0.0) * math.exp(-r * T)
    return float(payoff.mean()), float(payoff.std(ddof=1) / math.sqrt(len(payoff)))
