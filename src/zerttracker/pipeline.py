"""Pipeline orchestration.

POLICY: Dieses Projekt darf KEINE kostenpflichtigen externen Dienste nutzen.
Erlaubt: Börse Stuttgart Public-Endpoints, yfinance (Yahoo), ECB SDW.
Verboten: Anthropic/OpenAI, Alpha Vantage, Refinitiv, Bloomberg, alles mit API-Key/Subscription.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from zerttracker.fetchers import (
    boerse_stuttgart,
    db_xmarkets,
    macro as macro_fetch,
    underlyings,
    user_certificates,
)
from zerttracker.models import (
    CertificateAnalysis,
    ExpressCertificate,
    MacroSnapshot,
    UnderlyingMetrics,
)
from zerttracker.pricing.monte_carlo import (
    MCResult,
    estimate_physical_drift,
    price_express,
)
from zerttracker.scoring.heuristic import (
    score_macro,
    score_underlying,
)
from zerttracker.scoring.components import (
    ScoringConfig,
    composite,
    expected_return_pa_from_mc,
    score_expected_return,
    score_risk,
    score_risk_adjusted,
    score_value,
)

logger = logging.getLogger(__name__)


def analyze_certificate(
    cert: ExpressCertificate,
    macro: MacroSnapshot,
    underlying: Optional[UnderlyingMetrics] = None,
    *,
    valuation_date: Optional[date] = None,
    n_paths: int = 50000,
    scoring_cfg: Optional[ScoringConfig] = None,
) -> Optional[CertificateAnalysis]:
    if underlying is None:
        underlying = underlyings.fetch_underlying(cert.underlying_ticker)
    if underlying is None:
        logger.warning("Skipping %s: no underlying data for ticker=%r name=%r",
                       cert.isin, cert.underlying_ticker, cert.underlying_name)
        return None

    if scoring_cfg is None:
        scoring_cfg = ScoringConfig.load()

    mc_rn = price_express(
        cert, underlying, macro, valuation_date=valuation_date, n_paths=n_paths,
        vol_premium=scoring_cfg.vol_risk_premium,
    )
    mu_phys = estimate_physical_drift(underlying)
    mc_real = price_express(
        cert, underlying, macro,
        valuation_date=valuation_date, n_paths=n_paths,
        physical_drift=mu_phys, seed=43,
        vol_premium=scoring_cfg.vol_risk_premium,
    )

    market_price = cert.mid_price
    nominal = cert.nominal

    exp_ret_pa = expected_return_pa_from_mc(mc_real, market_price, nominal)

    s_value = score_value(market_price, mc_rn.fair_value, nominal, scoring_cfg)
    s_exp_ret = score_expected_return(exp_ret_pa, scoring_cfg)
    s_risk = score_risk(mc_real.prob_capital_loss, scoring_cfg)
    s_risk_adj = score_risk_adjusted(exp_ret_pa, mc_real.cvar_95, nominal, scoring_cfg)
    s_total = composite(s_value, s_exp_ret, s_risk, s_risk_adj, scoring_cfg)

    s_und = score_underlying(underlying)
    s_macro = score_macro(macro)

    notes: list[str] = []
    if market_price and market_price > mc_rn.fair_value * 1.02:
        notes.append("Marktpreis > Fair Value (>2% Aufschlag)")
    if mc_real.prob_capital_loss > 0.20:
        notes.append("Hohe Verlustwahrscheinlichkeit (>20%)")
    if mc_real.prob_autocall_first > 0.6:
        notes.append("Sehr wahrscheinliche frühe Rückzahlung")
    if underlying.historical_vol_1y and underlying.historical_vol_1y > 0.40:
        notes.append("Underlying mit hoher Volatilität (>40%)")
    if cert.is_worst_of:
        notes.append("Worst-of Struktur — höheres Risiko")
    if mc_rn.mc_standard_error / max(mc_rn.fair_value, 1e-6) > 0.05:
        notes.append(f"MC-Standardfehler hoch ({mc_rn.mc_standard_error:.2f} EUR)")

    return CertificateAnalysis(
        certificate=cert,
        underlying=underlying,
        macro=macro,
        fair_value=mc_rn.fair_value,
        market_price=market_price,
        expected_return_pa=exp_ret_pa,
        expected_holding_period_years=mc_real.expected_holding_period_years,
        prob_autocall_first=mc_real.prob_autocall_first,
        prob_full_coupons=mc_real.prob_full_coupons,
        prob_capital_loss=mc_real.prob_capital_loss,
        prob_barrier_breach=mc_real.prob_barrier_breach,
        expected_loss_given_breach=mc_real.expected_loss_given_breach,
        score_value=s_value,
        score_risk_reward=s_risk_adj,  # legacy field, jetzt risk-adjusted
        score_underlying=s_und,
        score_macro=s_macro,
        score_total=s_total,
        score_expected_return=s_exp_ret,
        score_risk=s_risk,
        score_risk_adjusted=s_risk_adj,
        cvar_95=mc_rn.cvar_95,
        mc_standard_error=mc_rn.mc_standard_error,
        ci_95_low=mc_rn.ci_95_low,
        ci_95_high=mc_rn.ci_95_high,
        sigma_used=mc_rn.sigma_used,
        notes=notes,
    )


def run_weekly(
    csv_fallback: Optional[Path] = None,
    output_path: Optional[Path] = None,
    n_paths: int = 50000,
) -> pd.DataFrame:
    certs: list[ExpressCertificate] = []
    source = "none"

    logger.info("Trying primary source: DB X-markets ...")
    try:
        xm_certs, xm_stats = db_xmarkets.fetch_db_xmarkets(max_products=200)
        logger.info(
            "DB X-markets stats: listing_url=%s status=%s isins_in_listing=%d details_attempted=%d details_parsed=%d errors=%d",
            xm_stats.listing_url, xm_stats.listing_status,
            xm_stats.products_found_in_listing, xm_stats.detail_pages_attempted,
            xm_stats.detail_pages_parsed, len(xm_stats.parse_errors),
        )
        for err in xm_stats.parse_errors[:5]:
            logger.info("  parse_error: %s", err)
        if xm_certs:
            certs = xm_certs
            source = "db-xmarkets"
    except Exception as exc:
        logger.warning("DB X-markets crashed: %s", exc)

    if not certs:
        user_certs = user_certificates.load_user_certificates()
        if user_certs:
            certs = user_certs
            source = "user-watchlist"
            logger.info("Loaded %d certificates from user watchlist", len(certs))

    if not certs and csv_fallback is not None and csv_fallback.exists():
        logger.info("Falling back to CSV sample at %s ...", csv_fallback)
        result = boerse_stuttgart.load_csv(csv_fallback)
        certs = result.certificates
        source = result.source

    logger.info("Fetched %d certificates from %s", len(certs), source)

    macro = macro_fetch.fetch_macro()
    logger.info("Macro snapshot: %s", macro.model_dump())

    rows: list[dict] = []
    for cert in certs:
        analysis = analyze_certificate(cert, macro, n_paths=n_paths)
        if analysis is None:
            continue
        rows.append(_flatten(analysis))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score_total", ascending=False).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved %d analyses to %s", len(df), output_path)

    return df


def _flatten(a: CertificateAnalysis) -> dict:
    return {
        "isin": a.certificate.isin,
        "wkn": a.certificate.wkn,
        "name": a.certificate.name,
        "underlying": a.certificate.underlying_name,
        "ticker": a.certificate.underlying_ticker,
        "type": a.certificate.cert_type.value,
        "memory": a.certificate.has_memory,
        "worst_of": a.certificate.is_worst_of,
        "maturity": a.certificate.maturity_date,
        "knock_in": a.certificate.knock_in_barrier,
        "spot": a.underlying.spot,
        "initial_fixing": a.certificate.initial_fixing,
        "barrier_distance_pct": (a.underlying.spot / (a.certificate.knock_in_barrier * a.certificate.initial_fixing) - 1.0),
        "vol_1y": a.underlying.historical_vol_1y,
        "sigma_used": a.sigma_used,
        "pe": a.underlying.pe_ratio,
        "div_yield": a.underlying.dividend_yield,
        "market_price": a.market_price,
        "fair_value": a.fair_value,
        "fair_value_ci_low": a.ci_95_low,
        "fair_value_ci_high": a.ci_95_high,
        "mc_se": a.mc_standard_error,
        "cvar_95": a.cvar_95,
        "exp_return_pa": a.expected_return_pa,
        "exp_horizon_y": a.expected_holding_period_years,
        "p_autocall_first": a.prob_autocall_first,
        "p_full_coupons": a.prob_full_coupons,
        "p_capital_loss": a.prob_capital_loss,
        "p_barrier_breach": a.prob_barrier_breach,
        "score_value": a.score_value,
        "score_expected_return": a.score_expected_return,
        "score_risk": a.score_risk,
        "score_risk_adjusted": a.score_risk_adjusted,
        "score_underlying": a.score_underlying,
        "score_macro": a.score_macro,
        "score_total": a.score_total,
        "notes": "; ".join(a.notes),
    }
