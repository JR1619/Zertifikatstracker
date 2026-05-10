"""Sub-Score-Komponenten auf absoluter Skala (0..100).

Jeder Sub-Score ist auf kalibrierte Schwellen gemappt: "bad" -> 0, "good" -> 100,
linear dazwischen. Schwellen liegen in `config/scoring.yaml` und sind dort
anpassbar.

Vier Komponenten:
- score_value:           Fair Value vs Markt/Nominal -- "ist es preiswert?"
- score_expected_return: erwartete annualisierte Rendite
- score_risk:            (invertiertes) Verlustrisiko
- score_risk_adjusted:   risk-adjusted return = exp_return / |CVaR-Loss|
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from zerttracker.models import ExpressCertificate
from zerttracker.pricing.monte_carlo import MCResult

CONFIG_PATH = Path(__file__).parent.parent / "config" / "scoring.yaml"


@dataclass
class ScoringConfig:
    weights: dict[str, float]
    thresholds: dict[str, dict]
    vol_risk_premium: float = 0.03

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ScoringConfig":
        p = path or CONFIG_PATH
        with open(p) as f:
            data = yaml.safe_load(f)
        return cls(
            weights=data.get("weights", {}),
            thresholds=data.get("thresholds", {}),
            vol_risk_premium=float(data.get("vol_risk_premium", 0.03)),
        )


def linear_score(x: float, bad: float, good: float, invert: bool = False) -> float:
    """Map x to 0..100. bad -> 0, good -> 100, linearly clipped.

    Die `invert`-Flag in der YAML ist ein Hinweis fuer den Leser, hat aber
    keinen Effekt: solange bad/good korrekt benannt sind (bad ist immer der
    schlechte Schwellwert), funktioniert die Formel auch wenn bad > good.
    """
    if good == bad:
        return 50.0
    t = (x - bad) / (good - bad)
    return float(max(0.0, min(100.0, t * 100.0)))


def score_value(market_price: Optional[float], fair_value: float, nominal: float, cfg: ScoringConfig) -> float:
    base = market_price if (market_price and market_price > 0) else nominal
    if base <= 0:
        return 50.0
    ratio = fair_value / base
    th = cfg.thresholds["value_ratio"]
    return linear_score(ratio, th["bad"], th["good"], th.get("invert", False))


def score_expected_return(exp_return_pa: float, cfg: ScoringConfig) -> float:
    th = cfg.thresholds["expected_return_pa"]
    return linear_score(exp_return_pa, th["bad"], th["good"], th.get("invert", False))


def score_risk(prob_capital_loss: float, cfg: ScoringConfig) -> float:
    th = cfg.thresholds["prob_capital_loss"]
    return linear_score(prob_capital_loss, th["bad"], th["good"], th.get("invert", False))


def score_risk_adjusted(exp_return_pa: float, cvar_95: float, nominal: float, cfg: ScoringConfig) -> float:
    """Erwartete Rendite p.a. dividiert durch CVaR-basierten Verlust pro Nominal.

    cvar_95 ist Mean of worst 5% diskontierter Payoffs (in EUR pro Nominal).
    cvar_loss = max(0, 1 - cvar_95 / nominal) -- wie viel man im Schwanz verliert.
    """
    cvar_loss = max(0.01, 1.0 - cvar_95 / max(nominal, 1e-6))
    ratio = exp_return_pa / cvar_loss
    th = cfg.thresholds["risk_adjusted_return"]
    return linear_score(ratio, th["bad"], th["good"], th.get("invert", False))


def composite(value: float, ret: float, risk: float, risk_adj: float, cfg: ScoringConfig) -> float:
    w = cfg.weights
    total = (
        value * w.get("value", 0.25)
        + ret * w.get("expected_return", 0.30)
        + risk * w.get("risk", 0.25)
        + risk_adj * w.get("risk_adjusted", 0.20)
    )
    return float(max(0.0, min(100.0, total)))


def expected_return_pa_from_mc(mc: MCResult, market_price: Optional[float], nominal: float) -> float:
    base = market_price if (market_price and market_price > 0) else nominal
    if base <= 0:
        return 0.0
    horizon = max(mc.expected_holding_period_years, 0.25)
    raw = (mc.expected_payoff_undiscounted / base) ** (1.0 / horizon) - 1.0
    return float(max(-0.99, min(5.0, raw)))
