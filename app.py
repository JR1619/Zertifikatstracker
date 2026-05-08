from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="DB Express-Zertifikate Tracker", layout="wide")

try:
    import plotly.express as px
    from zerttracker.pipeline import run_weekly
except Exception as exc:
    st.error(f"Import-Fehler beim App-Start: `{type(exc).__name__}: {exc}`")
    st.code(traceback.format_exc())
    st.stop()

CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PATH = CACHE_DIR / "latest.parquet"
SAMPLE_CSV = ROOT / "src" / "zerttracker" / "data" / "sample_certificates.csv"

IS_CLOUD = str(ROOT).startswith("/mount/") or os.environ.get("STREAMLIT_RUNTIME_HOST") is not None

st.title("Deutsche Bank Express-Zertifikate · Wöchentlicher Tracker")
st.caption(
    "Quantitative Bewertung über Monte-Carlo + Heuristik. Datenquellen: "
    "**Watchlist (manuell)** als primäre Quelle, Yahoo Finance für Underlying-Kurse, ECB für Makro. "
    "Zertifikate hinzufuegen/bearbeiten ueber die Seite 'Zertifikat hinzufuegen' (Sidebar)."
)

if LATEST_PATH.exists():
    mtime = datetime.fromtimestamp(LATEST_PATH.stat().st_mtime)
    age_days = (datetime.now() - mtime).days
    badge = "🟢" if age_days < 8 else ("🟡" if age_days < 15 else "🔴")
    st.info(f"{badge} **Datenstand:** {mtime:%d.%m.%Y %H:%M} · Updates wöchentlich (montags) per GitHub Action.")
elif IS_CLOUD:
    st.warning("Noch keine Daten im Repo. Wartet auf den ersten GitHub-Action-Lauf oder lade unten eine CSV hoch.")

with st.sidebar:
    st.header("Steuerung")
    use_cache = st.checkbox("Letztes Ergebnis aus Cache laden", value=True)
    n_paths = st.slider("Monte-Carlo Pfade", 2000, 50000, 10000, step=2000)

    if IS_CLOUD:
        st.button("Jetzt aktualisieren", disabled=True, help="Auf Streamlit Cloud nicht verfügbar (IPs werden geblockt). Daten werden via GitHub Action wöchentlich aktualisiert.")
        run_now = False
    else:
        run_now = st.button("Jetzt aktualisieren", type="primary")

    st.divider()
    st.caption("CSV-Fallback")
    csv_path_str = st.text_input("Pfad zur Fallback-CSV", value=str(SAMPLE_CSV))
    uploaded = st.file_uploader("oder eigene Zertifikate-CSV hochladen", type=["csv"])

    st.divider()
    st.caption(
        "**Methodik:** Fair Value risk-neutral (Monte-Carlo). "
        "Erwartete Rendite & Wahrscheinlichkeiten unter physischer Drift "
        "(Bayesian-Blend: 60% Equity-Baseline 6%, 40% 1J-Rendite Underlying)."
    )


@st.cache_data(show_spinner=False)
def _load_latest(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def _resolve_csv_path() -> Path:
    if uploaded is not None:
        tmp = CACHE_DIR / "uploaded_certificates.csv"
        tmp.write_bytes(uploaded.getvalue())
        return tmp
    return Path(csv_path_str)


def _load_or_run(force: bool, csv_path: Path, n_paths: int) -> tuple[pd.DataFrame, str]:
    if not force and use_cache and LATEST_PATH.exists():
        mtime = LATEST_PATH.stat().st_mtime
        df = _load_latest(str(LATEST_PATH), mtime)
        return df, f"Cache ({datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M})"
    with st.spinner("Lade Zertifikate, Underlyings und Makro-Daten..."):
        df = run_weekly(csv_fallback=csv_path, output_path=LATEST_PATH, n_paths=n_paths)
    _load_latest.clear()
    return df, "frisch berechnet"


csv_path = _resolve_csv_path()
try:
    df, source_label = _load_or_run(run_now, csv_path, n_paths)
except Exception as exc:
    st.error(f"Fehler beim Laden/Berechnen: `{type(exc).__name__}: {exc}`")
    st.code(traceback.format_exc())
    st.stop()

if df.empty:
    if IS_CLOUD:
        st.error(
            "Keine Daten verfügbar. Auf Streamlit Cloud kann nicht live gescraped werden — "
            "lass die GitHub Action laufen (Repo → Actions → 'Weekly data update' → Run workflow), "
            "oder lade in der Seitenleiste eine eigene CSV hoch."
        )
    else:
        st.warning(
            'Keine Daten vorhanden. Klick "Jetzt aktualisieren" oder hinterlege eine Fallback-CSV.'
        )
    st.stop()

st.success(f"{len(df)} Zertifikate · Quelle: {source_label}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Score Median", f"{df['score_total'].median():.0f}")
col2.metric("Top-Score", f"{df['score_total'].max():.0f}")
col3.metric("Ø Erw. Rendite p.a. (real)", f"{df['exp_return_pa'].mean()*100:.1f}%")
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
            "Fair Value (MC, risk-neutral)": round(row["fair_value"], 2),
            "Erw. Rendite p.a. (real)": f"{row['exp_return_pa']*100:.2f}%",
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
