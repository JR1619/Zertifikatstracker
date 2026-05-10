"""Volatility model for fair-value pricing.

Wir benutzen historische realisierte Vola plus einen konstanten Aufschlag fuer
die Vola-Risikopraemie (~3 Prozentpunkte). Das ist eine empirisch gut belegte
Approximation der impliziten Vola: implied = realized + ~2-4pp im Schnitt
fuer Aktien-Underlyings.

Begruendung: implizite Vola ist die fuer Pricing relevante Groesse, wir haben
aber keine Bezahl-Optionsfeeds. Der Aufschlag bringt das Modell in die Naehe
der echten Bewertung -- fuer Express-Zertifikate, die typische OTM-Puts
implizit short sind, ist die Risikopraemie wichtig, sonst werden sie chronisch
unterbewertet.

Der Wert ist parametrisierbar (in scoring.yaml), Default 0.03.
"""
from __future__ import annotations

DEFAULT_VOL_RISK_PREMIUM = 0.03


def pricing_vol(realized_vol: float, premium: float = DEFAULT_VOL_RISK_PREMIUM) -> float:
    """Convert realized vol to pricing vol via constant risk premium add-on."""
    if realized_vol is None or realized_vol <= 0:
        realized_vol = 0.20
    sigma = realized_vol + premium
    return float(max(0.05, min(sigma, 1.0)))
