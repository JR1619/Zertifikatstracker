"""Discounting for fair-value pricing.

Aufbau:
- SwapCurve: lin-interpolierte EUR-Swap-Kurve (Stützstellen 3M, 1Y, 2Y, 5Y, 10Y).
  Aktuell speisen wir nur 3M und 10Y aus ECB; Zwischenpunkte werden interpoliert.
- IssuerSpread: konstanter Aufschlag pro Emittent (in bp), aus YAML geladen.
- discount_factor(t, curve, spread) gibt den Pure-Discount-Faktor inkl. Spread.

Warum: bisher wurde mit `np.exp(-r_3m * t)` ODER `np.exp(-r_10y * t)` diskontiert
(je nach Zertifikat-Laufzeit), ohne Spread und ohne Term-Struktur. Das war ok
für Heuristik, aber nicht fuer Fair-Value-Pricing.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent / "config"


@dataclass
class SwapCurve:
    tenors_y: tuple[float, ...]
    rates: tuple[float, ...]

    def zero_rate(self, t_years: float) -> float:
        if t_years <= self.tenors_y[0]:
            return float(self.rates[0])
        if t_years >= self.tenors_y[-1]:
            return float(self.rates[-1])
        return float(np.interp(t_years, self.tenors_y, self.rates))


def make_swap_curve(rate_3m: float, rate_10y: float) -> SwapCurve:
    """Aus 3M und 10Y eine plausible Term-Struktur konstruieren.

    Zwischenpunkte 1Y, 2Y, 5Y werden lin interpoliert auf log(t)-Achse, was
    naeher an realer Kurvenform ist als pure linear-in-t.
    """
    if rate_10y >= rate_3m:
        rate_1y = rate_3m + 0.30 * (rate_10y - rate_3m)
        rate_2y = rate_3m + 0.50 * (rate_10y - rate_3m)
        rate_5y = rate_3m + 0.80 * (rate_10y - rate_3m)
    else:
        rate_1y = rate_3m + 0.40 * (rate_10y - rate_3m)
        rate_2y = rate_3m + 0.65 * (rate_10y - rate_3m)
        rate_5y = rate_3m + 0.85 * (rate_10y - rate_3m)
    return SwapCurve(
        tenors_y=(0.25, 1.0, 2.0, 5.0, 10.0),
        rates=(rate_3m, rate_1y, rate_2y, rate_5y, rate_10y),
    )


def load_issuer_spreads(path: Optional[Path] = None) -> dict[str, dict]:
    p = path or (CONFIG_DIR / "issuer_spreads.yaml")
    if not p.exists():
        logger.warning("issuer_spreads.yaml nicht gefunden bei %s, default 0bp.", p)
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def issuer_spread_bp(issuer: str, spreads: Optional[dict] = None) -> float:
    if spreads is None:
        spreads = load_issuer_spreads()
    entry = spreads.get(issuer)
    if not entry:
        return 0.0
    return float(entry.get("spread_bp", 0.0))


def discount_factor(
    t_years: float,
    swap: SwapCurve,
    issuer_spread_bp_value: float = 0.0,
) -> float:
    """Continuous discount factor with issuer spread."""
    if t_years <= 0:
        return 1.0
    r = swap.zero_rate(t_years) + issuer_spread_bp_value / 10_000.0
    return math.exp(-r * t_years)


def discount_factors_array(
    times_years,
    swap: SwapCurve,
    issuer_spread_bp_value: float = 0.0,
):
    """Vectorized discount factors for an array of times."""
    times = np.asarray(times_years, dtype=float)
    rates = np.array([swap.zero_rate(float(t)) for t in times])
    return np.exp(-(rates + issuer_spread_bp_value / 10_000.0) * times)
