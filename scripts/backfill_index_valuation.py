"""Backfill index P/E, P/B, dividend yield, close and TRI (1999+), India VIX and
the 10y G-sec yield into DuckDB.

    python scripts/backfill_index_valuation.py                 # download + load
    python scripts/backfill_index_valuation.py --load-only     # re-load the cache
    python scripts/backfill_index_valuation.py --verify        # cross-source checks

Raw pulls are cached under data/index_valuation_raw/ as one parquet per
(index, year) so a re-run only fetches what is missing. The current year is
always re-fetched. An empty year is cached as a ".none" marker (the index did
not exist yet); delete it to force a retry.

--verify compares three independent copies of the same numbers:
  1. niftyindices history endpoint (this backfill)
  2. index_data (captured day by day from niftyindices' daily snapshot CSVs)
  3. NSE's own /api/historicalOR/indicesYield (2016+)
and reports how many sessions disagree by more than rounding.
"""
import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.ingestion.index_valuation_fetcher import (VALUATION_INDICES,
                                                   IndexValuationFetcher)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "index_valuation_raw")
FIRST_YEAR = 1999


def _p(name: str, year: int) -> str:
    return os.path.join(RAW, f"{name.replace(' ', '_')}_{year}.parquet")


def download(indices: list[str], first: int = FIRST_YEAR) -> None:
    os.makedirs(RAW, exist_ok=True)
    f = IndexValuationFetcher()
    this_year = date.today().year
    for name in indices:
        got = 0
        for y in range(first, this_year + 1):
            p = _p(name, y)
            if y < this_year and (os.path.exists(p) or os.path.exists(p + ".none")):
                continue
            df = f.fetch_year(name, y)
            if df.empty:
                if y < this_year:
                    open(p + ".none", "w").close()
                continue
            df.to_parquet(p, index=False)
            got += len(df)
        print(f"{name:22s} +{got:,} rows", flush=True)

    vix = []
    for y in range(2008, this_year + 1):
        v = f.fetch_vix(date(y, 1, 1), min(date(y, 12, 31), date.today()))
        vix.append(v)
    vix = pd.concat(vix, ignore_index=True).drop_duplicates("trade_date")
    vix.to_parquet(os.path.join(RAW, "_vix.parquet"), index=False)
    print(f"VIX {len(vix):,} sessions from {vix['trade_date'].min()}")

    g = f.fetch_gsec_monthly()
    g.to_parquet(os.path.join(RAW, "_gsec.parquet"), index=False)
    print(f"G-sec {len(g):,} source-months")


def load() -> None:
    from src.data.repository import (upsert_gsec_yield, upsert_index_valuation,
                                     upsert_india_vix)
    from src.data.schema import initialize_schema
    initialize_schema()
    parts = [pd.read_parquet(os.path.join(RAW, fn)) for fn in sorted(os.listdir(RAW))
             if fn.endswith(".parquet") and not fn.startswith("_")]
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df.drop_duplicates(["trade_date", "index_name"], keep="last")
    print("index_valuation rows:", upsert_index_valuation(df))
    vp = os.path.join(RAW, "_vix.parquet")
    if os.path.exists(vp):
        print("india_vix_history rows:", upsert_india_vix(pd.read_parquet(vp)))
    gp = os.path.join(RAW, "_gsec.parquet")
    if os.path.exists(gp):
        print("gsec_yield_monthly rows:", upsert_gsec_yield(pd.read_parquet(gp)))


def verify() -> dict:
    """Cross-source agreement. Rounding is 2 dp, so > 0.015 is a real disagreement."""
    from src.data.repository import query_dataframe
    out: dict = {}
    hist = query_dataframe("SELECT * FROM index_valuation")
    snap = query_dataframe("""SELECT trade_date, index_name, pe_ratio pe, pb_ratio pb,
                                     div_yield dy, close_val FROM index_data""")
    for d in (hist, snap):
        d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date
    m = hist.merge(snap, on=["trade_date", "index_name"], suffixes=("", "_snap"))
    for c in ("pe", "pb", "dy", "close_val"):
        tol = 0.015 if c != "close_val" else 0.06
        both = m[[c, c + "_snap"]].dropna()
        bad = (both[c] - both[c + "_snap"]).abs() > tol
        out[f"snapshot_{c}"] = {"compared": int(len(both)), "disagree": int(bad.sum())}
    bad_rows = m[(m["pe"] - m["pe_snap"]).abs() > 0.015]
    out["snapshot_pe_worst"] = (bad_rows.assign(d=(bad_rows["pe"] - bad_rows["pe_snap"]).abs())
                                .nlargest(8, "d")[["trade_date", "index_name", "pe", "pe_snap"]]
                                .astype(str).to_dict("records"))

    f = IndexValuationFetcher()
    nse = []
    for y in range(2016, date.today().year + 1):
        n = f.fetch_nse_yield("NIFTY 50", date(y, 1, 1), min(date(y, 12, 31), date.today()))
        nse.append(n)
    nse = pd.concat(nse, ignore_index=True)
    nse["index_name"] = "Nifty 50"
    nse["trade_date"] = pd.to_datetime(nse["trade_date"]).dt.date
    m2 = hist.merge(nse, on=["trade_date", "index_name"], suffixes=("", "_nse"))
    for c in ("pe", "pb", "dy"):
        both = m2[[c, c + "_nse"]].dropna()
        out[f"nse_{c}"] = {"compared": int(len(both)),
                           "disagree": int(((both[c] - both[c + "_nse"]).abs() > 0.015).sum())}
    only_hist = set(hist.loc[(hist.index_name == "Nifty 50")
                             & (pd.to_datetime(hist.trade_date) >= "2016-01-01"), "trade_date"])
    out["nse_sessions_missing_from_history"] = sorted(map(str, set(nse.trade_date) - only_hist))[:20]
    out["history_sessions_missing_from_nse"] = sorted(map(str, only_hist - set(nse.trade_date)))[:20]
    print(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--index", action="append", help="endpoint name, e.g. 'NIFTY 50'")
    a = ap.parse_args()
    if a.verify:
        verify()
        sys.exit(0)
    if not a.load_only:
        download(a.index or list(VALUATION_INDICES), a.first_year)
    if not a.download_only:
        load()
