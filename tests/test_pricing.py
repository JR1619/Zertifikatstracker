from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerttracker.models import (
    CertificateType,
    ExpressCertificate,
    MacroSnapshot,
    ObservationDate,
    UnderlyingMetrics,
)
from zerttracker.pricing.monte_carlo import price_express


def _make_cert(initial: float = 100.0) -> ExpressCertificate:
    today = date.today()
    obs = [
        ObservationDate(date=today + timedelta(days=365 * (i + 1)), autocall_level=1.0, coupon_level=0.7, coupon_amount=70.0)
        for i in range(4)
    ]
    return ExpressCertificate(
        isin="DE0000000001",
        wkn="TEST01",
        name="Test Express",
        cert_type=CertificateType.EXPRESS_MEMORY,
        underlying_name="Test",
        underlying_ticker="TEST",
        initial_fixing=initial,
        issue_date=today,
        maturity_date=today + timedelta(days=365 * 4),
        observations=obs,
        knock_in_barrier=0.6,
        nominal=1000.0,
        has_memory=True,
    )


def test_pricing_runs():
    cert = _make_cert(initial=100.0)
    underlying = UnderlyingMetrics(ticker="TEST", spot=100.0, historical_vol_1y=0.25, dividend_yield=0.02)
    macro = MacroSnapshot(as_of=date.today(), risk_free_eur_3m=0.03, risk_free_eur_10y=0.025, yield_curve_slope=-0.005)

    result = price_express(cert, underlying, macro, n_paths=2000, seed=1)

    assert 700 < result.fair_value < 1300
    assert 0.0 <= result.prob_capital_loss <= 1.0
    assert 0.0 <= result.prob_autocall_first <= 1.0
    assert 0.0 <= result.prob_barrier_breach <= 1.0
    assert result.expected_holding_period_years > 0


def test_deep_itm_spot_likely_autocalls():
    cert = _make_cert(initial=100.0)
    underlying = UnderlyingMetrics(ticker="TEST", spot=140.0, historical_vol_1y=0.20, dividend_yield=0.0)
    macro = MacroSnapshot(as_of=date.today(), risk_free_eur_3m=0.02, risk_free_eur_10y=0.025, yield_curve_slope=0.005)

    result = price_express(cert, underlying, macro, n_paths=2000, seed=2)
    assert result.prob_autocall_first > 0.5


def test_deep_otm_spot_high_loss_risk():
    cert = _make_cert(initial=100.0)
    underlying = UnderlyingMetrics(ticker="TEST", spot=55.0, historical_vol_1y=0.40, dividend_yield=0.0)
    macro = MacroSnapshot(as_of=date.today(), risk_free_eur_3m=0.03, risk_free_eur_10y=0.030, yield_curve_slope=0.0)

    result = price_express(cert, underlying, macro, n_paths=2000, seed=3)
    assert result.prob_barrier_breach > 0.5
