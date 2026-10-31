"""Draw the README charts from results/.  python scripts/charts.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, DOCS = ROOT / "results", ROOT / "docs"
COLOURS = {"seasonal_naive": "#2a78d6", "naive_plus_growth": "#eb6834", "ets": "#1baf7a", "lightgbm": "#eda100"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def error_by_horizon(summary: dict) -> None:
    by_h = pd.read_csv(RESULTS / "by_horizon.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ends = []
    for model, colour in COLOURS.items():
        ax.plot(by_h.index, by_h[model], color=colour, linewidth=2, marker="o", markersize=4)
        ends.append((by_h[model].iloc[-1], model))
    # direct labels at the right edge, nudged apart so they never overlap
    # Spread labels at least `gap` apart, keeping their order and centring the group on the data.
    ends.sort()
    gap = 0.0042
    ys = [v for v, _ in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (sum(v for v, _ in ends) - sum(ys)) / len(ys)
    ys = [y + shift for y in ys]
    for i in range(1, len(ys)):  # re-enforce the gap after centring
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (value, model), y in zip(ends, ys):
        ax.annotate(summary["labels"][model], (12, value), xytext=(12.4, y), textcoords="data",
                    va="center", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8})
    ax.set_xlim(0.5, 17.5)
    ax.set_xticks(range(1, 13))
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
    ax.set_xlabel("Months ahead")
    ax.set_ylabel("Average error")
    ax.set_title("How far off each model is, by months ahead", loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, f"{summary['forecasts_scored_excl_covid']:,} backtest forecasts, 22 SingStat series, "
             f"origins {summary['origins'][0][:7]} to {summary['origins'][1][:7]}, COVID months excluded.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(DOCS / "error_by_horizon.png", dpi=150)
    plt.close(fig)


def example_forecast(summary: dict, series: str = "Food & beverage: Cafes") -> None:
    hist = pd.read_csv(ROOT / "data" / "series.csv", parse_dates=["month"])
    hist = hist[(hist["series"] == series) & (hist["month"] >= hist["month"].max() - pd.DateOffset(years=4))]
    fc = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["month"])
    fc = fc[(fc["series"] == series) & (fc["model"] == summary["chosen_model"])]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(fc["month"], fc["low_80"], fc["high_80"], color=COLOURS["seasonal_naive"], alpha=0.18,
                    linewidth=0)
    ax.plot(hist["month"], hist["index"], color=MUTED, linewidth=2)
    ax.plot(fc["month"], fc["forecast"], color=COLOURS["seasonal_naive"], linewidth=2, marker="o", markersize=4)
    ax.annotate("Actual", (hist["month"].iloc[0], hist["index"].iloc[0]), xytext=(-4, 10),
                textcoords="offset points", fontsize=9, color=INK)
    top = max(hist["index"].max(), fc["high_80"].max())
    ax.set_ylim(None, top * 1.08)
    ax.annotate("Forecast. Shaded: the range 8 in 10\npast forecasts this far ahead fell in",
                (fc["month"].iloc[2], fc["high_80"].iloc[2]),
                xytext=(hist["month"].iloc[-22], top * 1.035), textcoords="data",
                fontsize=9, color=INK, va="center",
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_ylabel("Sales index (2025 = 100)")
    ax.set_title(f"{series.split(': ')[1]}: next 12 months ({summary['labels'][summary['chosen_model']]})",
                 loc="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, f"SingStat F&B Services Index, current prices, data to {summary['data_through'][:7]}.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(DOCS / "example_cafes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    DOCS.mkdir(exist_ok=True)
    s = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    error_by_horizon(s)
    example_forecast(s)
