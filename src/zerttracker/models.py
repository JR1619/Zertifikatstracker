from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CertificateType(str, Enum):
    EXPRESS = "express"
    EXPRESS_MEMORY = "express_memory"
    EXPRESS_RELAX = "express_relax"
    EXPRESS_PLUS = "express_plus"


class ObservationDate(BaseModel):
    date: date
    autocall_level: float = Field(description="Autocall trigger as fraction of initial fixing, e.g. 1.0")
    coupon_level: float = Field(description="Coupon trigger as fraction of initial fixing")
    coupon_amount: float = Field(description="Coupon paid this period in absolute currency units (e.g. 5.50 for EUR 5.50)")


class ExpressCertificate(BaseModel):
    isin: str
    wkn: Optional[str] = None
    name: str
    issuer: str = "Deutsche Bank"
    cert_type: CertificateType = CertificateType.EXPRESS

    underlying_name: str
    underlying_ticker: str
    underlying_isin: Optional[str] = None
    initial_fixing: float

    issue_date: date
    maturity_date: date
    observations: list[ObservationDate]

    knock_in_barrier: float = Field(description="Capital protection barrier as fraction of initial fixing, e.g. 0.6")
    nominal: float = Field(default=1000.0, description="Nominal value in currency units")
    currency: str = "EUR"

    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None

    has_memory: bool = False
    is_worst_of: bool = False

    @field_validator("knock_in_barrier")
    @classmethod
    def _check_barrier(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("knock_in_barrier must be in (0, 1]")
        return v

    @property
    def mid_price(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.last


class UnderlyingMetrics(BaseModel):
    ticker: str
    spot: float
    historical_vol_1y: float
    realized_vol_3m: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    max_drawdown_5y: Optional[float] = None
    return_1y: Optional[float] = None


class MacroSnapshot(BaseModel):
    as_of: date
    risk_free_eur_3m: float = 0.0
    risk_free_eur_10y: float = 0.0
    yield_curve_slope: float = 0.0
    vstoxx: Optional[float] = None
    eur_inflation_yoy: Optional[float] = None


class CertificateAnalysis(BaseModel):
    certificate: ExpressCertificate
    underlying: UnderlyingMetrics
    macro: MacroSnapshot

    fair_value: float
    market_price: Optional[float]
    expected_return_pa: float
    expected_holding_period_years: float
    prob_autocall_first: float
    prob_full_coupons: float
    prob_capital_loss: float
    prob_barrier_breach: float
    expected_loss_given_breach: float

    score_value: float
    score_risk_reward: float
    score_underlying: float
    score_macro: float
    score_total: float

    score_expected_return: float = 0.0
    score_risk: float = 0.0
    score_risk_adjusted: float = 0.0
    cvar_95: float = 0.0
    mc_standard_error: float = 0.0
    ci_95_low: float = 0.0
    ci_95_high: float = 0.0
    sigma_used: float = 0.0

    notes: list[str] = Field(default_factory=list)
