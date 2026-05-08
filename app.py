from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from zerttracker.pipeline import run_weekly

CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PATH = CACHE_DIR / "latest.parquet"
SAMPLE_CSV = ROOT / "src" / "zerttracker" / "data" / "sample_certificates.csv"


st.set_page_config(page_title="DB Express-Zertifikate Tracker", layout="wide")
st.title("Deutsche Bank Express-Zertifikate · Wöchentlicher Tracker")
st.caption("Quantitative Bewertung über Monte-Carlo + Heuristik. Datenquellen: Börse Stuttgart, Yahoo Finance, ECB.")

with st.sidebar:
    st.header("Steuerung")
    use_cache = st.checkbox("Letztes Ergebnis aus Cache laden", value=True)
    n_paths = st.slider("Monte-Carlo Pfade", 2000, 50000, 10000, step=2000)
    run_now = st.button("Jetzt aktualisieren", type="primary")
    st.divider()
    st.caption("CSV-Fallback")
    csv_path_str = st.text_input("Pfad zur Fallback-CSV", value=str(SAMPLE_CSV))


@st.cache_data(show_spinner=False)
def _load_latest(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def _load_or_run(force: bool, csv_path: Path, n_paths: int) -> tuple[pd.DataFrame, str]:
    if not force and use_cache and LATEST_PATH.exists():
        df = _load_latest(str(LATEST_PATH))
        return df, f"Cache ({datetime.fromtimestamp(LATEST_PATH.stat().st_mtime):%Y-%m-%d %H:%M})"
    with st.spinner("Lade Zertifikate, Underlyings und Makro-Daten..."):
        df = run_weekly(csv_fallback=csv_path, output_path=LATEST_PATH, n_paths=n_paths)
    _load_latest.clear()
    return df, "frisch berechnet"


csv_path = Path(csv_path_str)
df, source_label = _load_or_run(run_now, csv_path, n_paths)

if df.empty:
    st.warning(
        "Keine Daten vorhanden. Versuche „Jetzt aktualisieren". "
        "Wenn Börse Stuttgart blockt, lege eine Fallback-CSV unter dem angezeigten Pfad ab."
    )
    st.stop()

st.success(f"{len(df)} Zertifikate · Quelle: {source_label}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Score Median", f"{df['score_total'].median():.0f}")
col2.metric("Top-Score", f"{df['score_total'].max():.0f}")
col3.metric("Ø Erw. Rendite p.a.", f"{df['exp_return_pa'].mean()*100:.1f}%")
col4.metric("Ø Verlust-W'keit", f"{df['p_capital_loss'].mean()*100:.1f}%")
col5.metric("Ø Autocall-W'keit (1. Termin)", f"{df['p_autocall_first'].mean()*100:.1f}%")

st.divider()

with st.expander("Filter", expanded=True):
    fc1, fc2, fc3, fc4 = st.columns(4)
    underlyings = ["alle", *sorted(df["underlying"].dropna().unique().tolist())]
    sel_underlying = fc1.selectbox("Underlying", underlyings)
    min_score = fc2.slider("Min. Score", 0, 100, 50)
    max_loss = fc3.slider("Max. Verlust-W'keit (%)", 0, 100, 30)
    min_return = fc4.slider("Min. Erw. Rendite p.a. (%)", -10, 30, 0)

mask = (
    (df["score_total"] >= min_score)
    & (df["p_capital_loss"] * 100 <= max_loss)
    & (df["exp_return_pa"] * 100 >= min_return)
)
if sel_underlying != "alle":
    mask &= df["underlying"] == sel_underlying

filtered = df[mask].copy()
st.subheader(f"Ranking ({len(filtered)} Treffer)")

display_cols = [
    "isin", "wkn", "name", "underlying", "maturity",
    "market_price", "fair_value", "exp_return_pa", "exp_horizon_y",
    "p_autocall_first", "p_full_coupons", "p_capital_loss", "p_barrier_breach",
    "score_total", "score_risk_reward", "score_value", "score_underlying", "score_macro",
    "notes",
]
fmt = {
    "exp_return_pa": "{:.2%}",
    "exp_horizon_y": "{:.2f}",
    "p_autocall_first": "{:.1%}",
    "p_full_coupons": "{:.1%}",
    "p_capital_loss": "{:.1%}",
    "p_barrier_breach": "{:.1%}",
    "score_total": "{:.0f}",
    "score_risk_reward": "{:.0f}",
    "score_value": "{:.0f}",
    "score_underlying": "{:.0f}",
    "score_macro": "{:.0f}",
    "market_price": "{:.2f}",
    "fair_value": "{:.2f}",
}
st.dataframe(
    filtered[display_cols].style.format(fmt),
    use_container_width=True,
    height=420,
)

st.divider()
st.subheader("Visualisierungen")
vc1, vc2 = st.columns(2)
with vc1:
    fig = px.scatter(
        filtered,
        x="exp_return_pa",
        y="p_capital_loss",
        size="score_total",
        color="underlying",
        hover_data=["isin", "name", "score_total"],
        labels={"exp_return_pa": "Erw. Rendite p.a.", "p_capital_loss": "Verlust-W'keit"},
        title="Risiko vs. Chance",
    )
    fig.update_layout(xaxis_tickformat=".1%", yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

with vc2:
    score_breakdown = (
        filtered.head(15)
        .melt(
            id_vars=["isin", "name"],
            value_vars=["score_value", "score_risk_reward", "score_underlying", "score_macro"],
            var_name="Komponente",
            value_name="Score",
        )
    )
    fig2 = px.bar(
        score_breakdown,
        x="isin",
        y="Score",
        color="Komponente",
        title="Score-Komponenten Top 15",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("Detail-Ansicht")
sel_isin = st.selectbox("Zertifikat auswählen", filtered["isin"].tolist())
if sel_isin:
    row = filtered[filtered["isin"] == sel_isin].iloc[0]
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown("**Stammdaten**")
        st.write({
            "Name": row["name"],
            "WKN": row.get("wkn"),
            "Underlying": row["underlying"],
            "Typ": row["type"],
            "Memory": bool(row["memory"]),
            "Fälligkeit": row["maturity"],
            "Knock-in Barriere": f"{row['knock_in']*100:.0f}% des Initial",
        })
    with dc2:
        st.markdown("**Bewertung**")
        st.write({
            "Marktpreis": row["market_price"],
            "Fair Value (MC)": round(row["fair_value"], 2),
            "Erw. Rendite p.a.": f"{row['exp_return_pa']*100:.2f}%",
            "Erw. Haltedauer": f"{row['exp_horizon_y']:.2f} J.",
            "Score Total": int(row["score_total"]),
        })
    with dc3:
        st.markdown("**Wahrscheinlichkeiten**")
        st.write({
            "Autocall am 1. Termin": f"{row['p_autocall_first']*100:.1f}%",
            "Volle Kupons": f"{row['p_full_coupons']*100:.1f}%",
            "Kapitalverlust": f"{row['p_capital_loss']*100:.1f}%",
            "Barrierenbruch": f"{row['p_barrier_breach']*100:.1f}%",
        })
    if row.get("notes"):
        st.warning(row["notes"])
