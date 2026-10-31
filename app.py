"""Sales outlook for a Singapore retail or F&B sector.

    streamlit run app.py

Reads results/ (built by run.py) and data/series.csv. No model runs in the app.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BLUE, BAND, MUTED = "#2a78d6", "rgba(42,120,214,0.18)", "#52514e"

st.set_page_config(page_title="SG sales outlook", layout="wide")


@st.cache_data
def load():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    history = pd.read_csv(ROOT / "data" / "series.csv", parse_dates=["month"])
    forecasts = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["month"])
    per_series = pd.read_csv(RESULTS / "per_series.csv", index_col=0)
    return summary, history, forecasts, per_series


summary, history, forecasts, per_series = load()
labels = summary["labels"]
chosen = summary["chosen_model"]

st.title("Singapore sales outlook, next 12 months")
st.caption(
    f"SingStat sales indices to {summary['data_through'][:7]} (2025 = 100). "
    "Use it to plan stock, staff and cash. The model and its range come from a backtest of "
    f"{summary['forecasts_scored_excl_covid']:,} past forecasts; see the README for how it was tested."
)

series_names = sorted(history["series"].unique())
default = series_names.index("Food & beverage: Cafes") if "Food & beverage: Cafes" in series_names else 0
left, right = st.columns([2, 1])
with left:
    sector = st.selectbox("Your sector", series_names, index=default)
with right:
    model = st.selectbox("Model", list(labels), index=list(labels).index(chosen), format_func=lambda m: labels[m])

h = history[history["series"] == sector].sort_values("month")
h = h[h["month"] >= h["month"].max() - pd.DateOffset(years=5)]
f = forecasts[(forecasts["series"] == sector) & (forecasts["model"] == model)].sort_values("month")

fig = go.Figure()
fig.add_trace(go.Scatter(x=pd.concat([f["month"], f["month"][::-1]]),
                         y=pd.concat([f["high_80"], f["low_80"][::-1]]),
                         fill="toself", fillcolor=BAND, line={"width": 0}, hoverinfo="skip",
                         name="80% range"))
fig.add_trace(go.Scatter(x=h["month"], y=h["index"], mode="lines", line={"color": MUTED, "width": 2},
                         name="Actual", hovertemplate="%{x|%b %Y}: %{y:.1f}<extra></extra>"))
fig.add_trace(go.Scatter(x=f["month"], y=f["forecast"], mode="lines+markers",
                         line={"color": BLUE, "width": 2}, marker={"size": 8}, name="Forecast",
                         hovertemplate="%{x|%b %Y}: %{y:.1f}<extra></extra>"))
fig.update_layout(height=420, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                  yaxis_title="Sales index (2025 = 100)", hovermode="x unified",
                  legend={"orientation": "h", "y": 1.08})
st.plotly_chart(fig, width="stretch")

last_year = h.set_index("month")["index"]
table = f[["month", "forecast", "low_80", "high_80"]].copy()
table["vs same month last year"] = [
    f"{row.forecast / last_year.get(row.month - pd.DateOffset(years=1), float('nan')) - 1:+.1%}"
    for row in table.itertuples()
]
table["month"] = table["month"].dt.strftime("%b %Y")
table = table.rename(columns={"forecast": "forecast", "low_80": "low (80%)", "high_80": "high (80%)"})
st.dataframe(table.round(1), hide_index=True, width="stretch")

err = per_series.loc[sector] if sector in per_series.index else None
if err is not None:
    st.markdown(
        f"**How wrong is it usually?** In the backtest, outside COVID, *{labels[model]}* missed this "
        f"sector's actual sales by **{err[model]:.1%}** on average, against **{err['seasonal_naive']:.1%}** "
        "for simply repeating the same month last year."
    )
st.caption("The 80% range: in past forecasts at the same distance ahead, 8 in 10 actuals fell inside it.")
