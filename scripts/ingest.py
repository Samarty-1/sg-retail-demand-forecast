"""Download SingStat's monthly sales indices into data/series.csv.  python scripts/ingest.py

Retail Sales Index (M602121) and Food & Beverage Services Index (M602131), both at
current prices (what tills actually take in), 2025 = 100. The CSV is committed so the
backtest and tests run offline and reproduce exactly; re-run this to update it.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "series.csv"
API = "https://tablebuilder.singstat.gov.sg/api/table/tabledata"
TABLES = {"M602121": "Retail", "M602131": "Food & beverage"}
PAGE = 3000  # the API caps each response at this many data points

log = logging.getLogger("ingest")


def get(params: dict, table: str) -> dict:
    for attempt in range(6):
        try:
            r = requests.get(f"{API}/{table}", params=params, timeout=60,
                             headers={"User-Agent": "sg-retail-demand-forecast"})
            r.raise_for_status()
            return r.json()["Data"]
        except (requests.RequestException, KeyError, ValueError) as e:
            if attempt == 5:
                raise
            log.warning("retrying %s: %s", table, e)
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def fetch(table: str) -> tuple[pd.DataFrame, str]:
    rows, offset, updated = [], 0, ""
    while True:
        d = get({"limit": PAGE, "offset": offset}, table)
        updated = d["dataLastUpdated"]
        got = 0
        for series in d["row"]:
            for point in series["columns"]:
                got += 1
                if point["value"] in ("", "-", "na", None):
                    continue
                rows.append({"series": series["rowText"].strip(),
                             "month": datetime.strptime(point["key"], "%Y %b").date(),
                             "index": float(point["value"])})
        if got < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(rows), updated


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    frames = []
    for table, group in TABLES.items():
        df, updated = fetch(table)
        df.insert(0, "group", group)
        df.insert(0, "table", table)
        frames.append(df)
        log.info("%s: %d series, %d points, updated %s", table, df["series"].nunique(), len(df), updated)
    out = pd.concat(frames, ignore_index=True)
    # Two tables both have a "Total" row: make every series name unique.
    out["series"] = out["group"] + ": " + out["series"]
    dupes = out.duplicated(["series", "month"]).sum()
    if dupes:
        raise SystemExit(f"{dupes} duplicate (series, month) rows")
    # Months as integers (year*12 + month): consecutive months differ by exactly 1.
    stamp = pd.to_datetime(out["month"])
    out["_m"] = stamp.dt.year * 12 + stamp.dt.month
    gaps = out.sort_values("_m").groupby("series")["_m"].apply(lambda m: int((m.diff().dropna() != 1).sum()))
    out = out.drop(columns="_m")
    if gaps.any():
        raise SystemExit(f"series with missing months: {gaps[gaps > 0].to_dict()}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values(["series", "month"]).to_csv(OUT, index=False)
    log.info("wrote %s (%d rows)", OUT, len(out))


if __name__ == "__main__":
    main()
