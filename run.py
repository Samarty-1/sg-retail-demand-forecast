"""Backtest every model, check the uncertainty ranges, and forecast the next 12 months.

    python run.py            # ~10 minutes; writes results/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.backtest import load_panel, run, summarise
from forecast.intervals import LEVEL, calibration, quantiles_by_horizon
from forecast.models import H, LABELS, MODELS, GlobalGBM, ets, naive_plus_growth, seasonal_naive

RESULTS = Path(__file__).resolve().parent / "results"


def choose_model(headline: pd.DataFrame) -> str:
    """The protocol's rule: the lowest headline error among models that beat the
    baseline; seasonal naive if none does."""
    winners = headline[headline["beats_baseline"].astype(bool)]
    return str(winners["error_excl_covid"].idxmin()) if len(winners) else "seasonal_naive"


def forecast_next_year(panel: pd.DataFrame, bands: dict[str, pd.DataFrame]) -> pd.DataFrame:
    gbm = GlobalGBM(panel).fit(cutoff=len(panel) - 1)
    t = len(panel) - 1
    future = pd.date_range(panel.index[-1] + pd.DateOffset(months=1), periods=H, freq="MS")
    rows = []
    for s, name in enumerate(panel.columns):
        hist = panel[name].dropna().to_numpy()
        preds = {"seasonal_naive": seasonal_naive(hist), "naive_plus_growth": naive_plus_growth(hist),
                 "ets": ets(hist), "lightgbm": gbm.predict(s, t)}
        for model in MODELS:
            q = bands[model]
            for h in range(1, H + 1):
                rows.append({"series": name, "model": model, "month": future[h - 1].date(), "h": h,
                             "forecast": round(preds[model][h - 1], 3),
                             "low_80": round(preds[model][h - 1] * q.loc[h, "low"], 3),
                             "high_80": round(preds[model][h - 1] * q.loc[h, "high"], 3)})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    RESULTS.mkdir(exist_ok=True)
    panel = load_panel()

    bt = run(panel)
    bt.to_parquet(RESULTS / "backtest.parquet", index=False)
    s = summarise(bt)
    for name, table in s.items():
        table.round(4).to_csv(RESULTS / f"{name}.csv")
    chosen = choose_model(s["headline"])

    calib = {m: calibration(bt, m) for m in MODELS}
    bands = {m: quantiles_by_horizon(bt, m) for m in MODELS}
    pd.concat(bands, names=["model", "h"]).round(4).to_csv(RESULTS / "range_multipliers.csv")

    fc = forecast_next_year(panel, bands)
    fc.to_csv(RESULTS / "forecasts.csv", index=False)

    summary = {
        "data_through": str(panel.index[-1].date()),
        "series": int(panel.shape[1]),
        "forecasts_scored": int(len(bt)),
        "forecasts_scored_excl_covid": int((~bt["covid"]).sum()),
        "origins": [str(bt["origin"].min().date()), str(bt["origin"].max().date())],
        "chosen_model": chosen,
        "headline": {m: {k: (None if pd.isna(v) else (bool(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 4)))
                         for k, v in row.items()} for m, row in s["headline"].iterrows()},
        "calibration_80pct": {m: {"coverage": round(c["coverage"], 3), "n": c["n"]} for m, c in calib.items()},
        "labels": LABELS,
        "range_level": LEVEL,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(s["by_horizon"].round(4).to_string())
    print(s["by_season"].round(4).to_string())


if __name__ == "__main__":
    main()
