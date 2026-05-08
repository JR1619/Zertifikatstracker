"""Streamlit-Seite zum Hinzufuegen/Bearbeiten von Express-Zertifikaten.

DB X-markets blockt Bot-Zugriff per Cloudflare, deshalb pflegt der Nutzer seine
Watchlist hier manuell. Daten landen in data/user_certificates.json und werden
beim naechsten Pipeline-Run analysiert.
"""
from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from zerttracker.fetchers.user_certificates import (
    DEFAULT_STORE_PATH,
    delete_user_certificate,
    load_user_certificates,
    upsert_user_certificate,
)
from zerttracker.models import CertificateType, ExpressCertificate, ObservationDate

st.set_page_config(page_title="Zertifikat hinzufuegen", layout="wide")
st.title("Zertifikat hinzufügen / bearbeiten")
st.caption(
    f"Wird gespeichert in `{DEFAULT_STORE_PATH.relative_to(ROOT)}` und beim "
    "naechsten Pipeline-Lauf analysiert."
)

UNDERLYING_PRESETS = {
    "EuroStoxx 50 (^STOXX50E)": "^STOXX50E",
    "DAX (^GDAXI)": "^GDAXI",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq 100 (^NDX)": "^NDX",
    "SAP (SAP.DE)": "SAP.DE",
    "Allianz (ALV.DE)": "ALV.DE",
    "Siemens (SIE.DE)": "SIE.DE",
    "BMW (BMW.DE)": "BMW.DE",
    "BASF (BAS.DE)": "BAS.DE",
    "Volkswagen Vz (VOW3.DE)": "VOW3.DE",
    "Deutsche Bank (DBK.DE)": "DBK.DE",
    "Bayer (BAYN.DE)": "BAYN.DE",
    "Mercedes (MBG.DE)": "MBG.DE",
    "Munich Re (MUV2.DE)": "MUV2.DE",
    "Adidas (ADS.DE)": "ADS.DE",
    "Anderes (manuell)": "",
}

existing = load_user_certificates()
existing_isins = [c.isin for c in existing]

mode_col, _ = st.columns([1, 3])
mode = mode_col.radio(
    "Modus",
    ["Neu", "Bestehendes bearbeiten"],
    horizontal=True,
    disabled=not existing_isins,
)

prefill: ExpressCertificate | None = None
if mode == "Bestehendes bearbeiten" and existing_isins:
    sel = st.selectbox("Zertifikat", existing_isins, format_func=lambda i: f"{i} — {next(c.name for c in existing if c.isin == i)}")
    prefill = next(c for c in existing if c.isin == sel)


def _default(field: str, fallback):
    if prefill is None:
        return fallback
    return getattr(prefill, field, fallback)


with st.form("cert_form", clear_on_submit=False):
    st.subheader("Stammdaten")
    c1, c2, c3 = st.columns(3)
    isin = c1.text_input("ISIN *", value=_default("isin", ""), max_chars=12, help="z.B. DE000DB9XXXX")
    wkn = c2.text_input("WKN", value=_default("wkn", "") or "")
    name = c3.text_input("Name *", value=_default("name", ""), max_chars=120)

    c4, c5, c6 = st.columns(3)
    issuer = c4.text_input("Emittent", value=_default("issuer", "Deutsche Bank"))
    cert_type_default = _default("cert_type", CertificateType.EXPRESS)
    cert_type_str = c5.selectbox(
        "Typ",
        [t.value for t in CertificateType],
        index=[t.value for t in CertificateType].index(cert_type_default.value if hasattr(cert_type_default, "value") else cert_type_default),
    )
    has_memory = c6.checkbox("Memory-Coupon", value=_default("has_memory", False))

    st.subheader("Underlying")
    u1, u2, u3 = st.columns(3)
    underlying_name = u1.text_input("Name Underlying *", value=_default("underlying_name", ""))
    preset_label = u2.selectbox("Yahoo-Ticker (Preset)", list(UNDERLYING_PRESETS.keys()), index=len(UNDERLYING_PRESETS) - 1)
    ticker_preset = UNDERLYING_PRESETS[preset_label]
    underlying_ticker = u3.text_input("Ticker (Yahoo) *", value=ticker_preset or _default("underlying_ticker", ""), help="Notwendig fuer Kursabfrage via yfinance")

    st.subheader("Struktur")
    s1, s2, s3, s4 = st.columns(4)
    initial_fixing = s1.number_input("Initial Fixing *", value=float(_default("initial_fixing", 0.0) or 0.0), step=1.0, format="%.4f")
    knock_in_pct = s2.number_input("Knock-In Barriere (%) *", value=float(_default("knock_in_barrier", 0.6) or 0.6) * 100.0, min_value=10.0, max_value=100.0, step=1.0, help="z.B. 60 fuer 60%")
    nominal = s3.number_input("Nominal", value=float(_default("nominal", 1000.0) or 1000.0), step=100.0)
    currency = s4.text_input("Waehrung", value=_default("currency", "EUR"), max_chars=3)

    s5, s6 = st.columns(2)
    issue_date_val = s5.date_input("Emissionstag *", value=_default("issue_date", date.today()))
    maturity_date_val = s6.date_input("Endfaelligkeit *", value=_default("maturity_date", date.today() + timedelta(days=365 * 4)))

    st.subheader("Aktuelle Marktpreise (optional)")
    p1, p2, p3 = st.columns(3)
    bid = p1.number_input("Bid", value=float(_default("bid", 0.0) or 0.0), step=0.1, format="%.2f")
    ask = p2.number_input("Ask", value=float(_default("ask", 0.0) or 0.0), step=0.1, format="%.2f")
    last = p3.number_input("Letzter", value=float(_default("last", 0.0) or 0.0), step=0.1, format="%.2f")

    st.subheader("Beobachtungstage")
    st.caption(
        "Eine Zeile pro Beobachtungstag, in chronologischer Reihenfolge. "
        "Levels in % vom Initial Fixing (z.B. 100 fuer 100%). Coupon-Betrag in Waehrung."
    )
    obs_default = pd.DataFrame(
        [
            {
                "Datum": o.date,
                "Autocall-Level (%)": o.autocall_level * 100.0,
                "Coupon-Level (%)": o.coupon_level * 100.0,
                "Coupon-Betrag": o.coupon_amount,
            }
            for o in (prefill.observations if prefill else [])
        ]
        or [
            {
                "Datum": date.today() + timedelta(days=365),
                "Autocall-Level (%)": 100.0,
                "Coupon-Level (%)": 100.0,
                "Coupon-Betrag": 50.0,
            }
        ]
    )
    obs_df = st.data_editor(
        obs_default,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Datum": st.column_config.DateColumn(required=True),
            "Autocall-Level (%)": st.column_config.NumberColumn(min_value=0.0, max_value=200.0, step=1.0, format="%.2f"),
            "Coupon-Level (%)": st.column_config.NumberColumn(min_value=0.0, max_value=200.0, step=1.0, format="%.2f"),
            "Coupon-Betrag": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="%.2f"),
        },
        key="obs_editor",
    )

    submit_col, _ = st.columns([1, 4])
    submitted = submit_col.form_submit_button("Speichern", type="primary")


def _build_cert() -> ExpressCertificate:
    obs: list[ObservationDate] = []
    for _, row in obs_df.iterrows():
        if pd.isna(row["Datum"]):
            continue
        obs.append(
            ObservationDate(
                date=row["Datum"] if isinstance(row["Datum"], date) else pd.to_datetime(row["Datum"]).date(),
                autocall_level=float(row["Autocall-Level (%)"]) / 100.0,
                coupon_level=float(row["Coupon-Level (%)"]) / 100.0,
                coupon_amount=float(row["Coupon-Betrag"]),
            )
        )
    obs.sort(key=lambda o: o.date)
    return ExpressCertificate(
        isin=isin.strip().upper(),
        wkn=wkn.strip() or None,
        name=name.strip(),
        issuer=issuer.strip(),
        cert_type=CertificateType(cert_type_str),
        underlying_name=underlying_name.strip(),
        underlying_ticker=underlying_ticker.strip(),
        initial_fixing=float(initial_fixing),
        issue_date=issue_date_val,
        maturity_date=maturity_date_val,
        observations=obs,
        knock_in_barrier=knock_in_pct / 100.0,
        nominal=float(nominal),
        currency=currency.strip().upper(),
        bid=bid if bid > 0 else None,
        ask=ask if ask > 0 else None,
        last=last if last > 0 else None,
        has_memory=has_memory,
    )


if submitted:
    try:
        cert = _build_cert()
        if not cert.observations:
            st.error("Mindestens ein Beobachtungstag erforderlich.")
        elif not isin or not name or not underlying_name or not underlying_ticker:
            st.error("Pflichtfelder mit * ausfuellen.")
        elif initial_fixing <= 0:
            st.error("Initial Fixing muss > 0 sein.")
        else:
            upsert_user_certificate(cert)
            st.success(f"Gespeichert: {cert.isin} — {cert.name}")
            st.cache_data.clear()
    except Exception as exc:
        st.error(f"Fehler beim Speichern: {type(exc).__name__}: {exc}")
        st.code(traceback.format_exc())

st.divider()
st.subheader("Watchlist")
existing = load_user_certificates()
if not existing:
    st.info("Noch keine Zertifikate hinterlegt.")
else:
    df_view = pd.DataFrame(
        [
            {
                "ISIN": c.isin,
                "Name": c.name,
                "Underlying": c.underlying_name,
                "Initial": c.initial_fixing,
                "Knock-In": f"{c.knock_in_barrier * 100:.0f}%",
                "Faelligkeit": c.maturity_date,
                "# Termine": len(c.observations),
                "Memory": "Ja" if c.has_memory else "Nein",
            }
            for c in existing
        ]
    )
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    del_isin = st.selectbox("Loeschen", [""] + [c.isin for c in existing])
    if del_isin and st.button("Loeschen bestaetigen", type="secondary"):
        delete_user_certificate(del_isin)
        st.success(f"Geloescht: {del_isin}")
        st.rerun()
