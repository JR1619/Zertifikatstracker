"""Tests fuer die Score-Komponenten."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerttracker.scoring.components import (
    ScoringConfig,
    composite,
    linear_score,
    score_expected_return,
    score_risk,
    score_risk_adjusted,
    score_value,
)


def test_linear_score_endpoints():
    assert linear_score(0.0, bad=0.0, good=1.0) == 0.0
    assert linear_score(1.0, bad=0.0, good=1.0) == 100.0
    assert linear_score(0.5, bad=0.0, good=1.0) == 50.0


def test_linear_score_decreasing():
    """Wenn bad>good, wird invertiert: hohe Werte bei "bad" geben niedrige Scores."""
    assert linear_score(1.0, bad=1.0, good=0.0) == 0.0
    assert linear_score(0.0, bad=1.0, good=0.0) == 100.0
    assert linear_score(0.5, bad=1.0, good=0.0) == 50.0


def test_linear_score_clipping():
    assert linear_score(-100, bad=0.0, good=1.0) == 0.0
    assert linear_score(999, bad=0.0, good=1.0) == 100.0


def test_score_value_at_par():
    cfg = ScoringConfig.load()
    s = score_value(market_price=100.0, fair_value=100.0, nominal=100.0, cfg=cfg)
    assert 0 < s < 100  # zwischen den Schwellen


def test_score_risk_inversion():
    cfg = ScoringConfig.load()
    high_loss = score_risk(0.70, cfg)
    low_loss = score_risk(0.10, cfg)
    assert low_loss > high_loss


def test_composite_weighted_sum():
    cfg = ScoringConfig.load()
    total = composite(value=100, ret=100, risk=100, risk_adj=100, cfg=cfg)
    assert 99 < total <= 100  # alle 100 -> ~100
    total = composite(value=0, ret=0, risk=0, risk_adj=0, cfg=cfg)
    assert total == 0.0
