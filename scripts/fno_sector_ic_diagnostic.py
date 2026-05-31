"""
READ-ONLY: does the SECTOR F&O aggregate predict forward sector returns?

Phase-3 validation gate before the F&O overlay is ever allowed to touch the
sector score. Walk-forward, no leakage: the F&O factor on day D uses only D's OI
and D-vs-(D-1) OI change; forward returns look D+1..D+h.

Factors tested (from get_sector_fno_aggregate, recomputed per historical date):
  fno_avg_score   — mean per-stock futures OI-price score across the sector
                    (+2 Long Buildup ... -2 Short Buildup)
  fno_net_score   — sum of the same (size-weighted by # F&O stocks)
  sector_pcr      — sector put/call OI ratio (contrarian: high PCR -> bullish)
  pcr_contra      — -(sector_pcr) so "higher = more bullish", comparable sign

Benchmarks on the SAME rows (does F&O ADD anything over what we already have?):
  dv5d            — the delivery 5-day flow factor (existing edge)
  rs_2w           — relative strength vs Nifty (the strongest existing factor)

Post-expiry days (futures OI suppressed market-wide) are reported separately for
the futures-score factors, since those days carry no futures signal by design.

Metrics per (factor, horizon): IC (mean daily Spearman), ICIR, t, decile spread.
Nothing is written. Safe to delete.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.factor_ic_diagnostic import load_panel, load_nifty, build_factors, evaluate, HORIZONS  # noqa: E402
from src.data.repository import query_dataframe                                                      # noqa: E402
from src.analytics.fno_stocks import get_sector_fno_aggregate                                         # noqa: E402

MAX_DATES = 90   # most recent N FNO dates (recomputing the aggregate per date is heavy)


def _fno_dates() -> list[date]:
    df = query_dataframe(
        "SELECT DISTINCT trade_date FROM fno_bhavcopy "
        "WHERE instrument IN ('FUTSTK','OPTSTK') ORDER BY trade_date DESC LIMIT ?",
        [MAX_DATES],
    )
    ds = [d.date() if hasattr(d, "date") else d for d in df["trade_date"]]
    return sorted(ds)


def build_fno_factor_panel(dates: list[date]) -> pd.DataFrame:
    """One row per (trade_date, sector) with the F&O factors, recomputed per date."""
    rows = []
    for i, d in enumerate(dates):
        try:
            agg = get_sector_fno_aggregate(d)
        except Exception as exc:
            print(f"  {d}: ERROR {exc!r}")
            continue
        if agg.empty:
            continue
        agg = agg.copy()
        agg["trade_date"] = pd.Timestamp(d)
        rows.append(agg[["trade_date", "sector", "fno_avg_score", "fno_net_score",
                         "sector_pcr", "post_expiry"]])
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(dates)} dates processed")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    dates = _fno_dates()
    print(f"F&O dates: {len(dates)} ({dates[0]} -> {dates[-1]})")

    # Forward sector returns (reuse the validated harness)
    print("Building forward-return panel...")
    fwd = build_factors(load_panel(), load_nifty())   # trade_date=Timestamp, sector, fwd_h, dv5d, rs_2w
    fwd_keep = ["trade_date", "sector", "dv5d", "rs_2w"] + [f"fwd_{h}" for h in HORIZONS]
    fwd = fwd[[c for c in fwd_keep if c in fwd.columns]]

    print("Recomputing sector F&O aggregate across history (heavy)...")
    fno = build_fno_factor_panel(dates)
    if fno.empty:
        print("No F&O factor rows — abort."); return
    fno["pcr_contra"] = -fno["sector_pcr"]   # higher = more bullish (contrarian)

    m = fno.merge(fwd, on=["trade_date", "sector"], how="inner")
    print(f"merged rows: {len(m)}  ({m['trade_date'].nunique()} days x ~{m['sector'].nunique()} sectors)\n")

    # Split: non-post-expiry rows (futures signal valid) vs all rows
    m_fut = m[~m["post_expiry"].astype(bool)]

    def block(df, factors, title, note=""):
        print("=" * 96)
        print(title + (f"   [{note}]" if note else ""))
        print(f"rows={len(df)}  days={df['trade_date'].nunique()}")
        print("=" * 96)
        head = f"{'factor':<14}"
        for h in HORIZONS:
            head += f"  h={h:<2d} IC/ICIR/spr%   "
        print(head); print("-" * 96)
        for f in factors:
            if f not in df.columns:
                continue
            line = f"{f:<14}"
            for h in HORIZONS:
                r = evaluate(df, f, f"fwd_{h}")
                line += (f"  {r['ic']:+.3f}/{r['icir']:+.2f}/{r['spr']:+.2f}".ljust(19)
                         if r else f"  {'—':<17}")
            print(line)
        print()

    # Futures-score factors only make sense off post-expiry days
    block(m_fut, ["fno_avg_score", "fno_net_score"],
          "F&O FUTURES-SCORE  → forward sector return", "post-expiry days excluded")
    # PCR works every day
    block(m, ["pcr_contra"],
          "F&O OPTIONS (PCR, contrarian)  → forward sector return", "all days")
    # Benchmarks on the SAME merged rows — does F&O beat / add to these?
    block(m_fut, ["dv5d", "rs_2w"],
          "BENCHMARKS on the same rows (existing factors)", "post-expiry days excluded")

    # Headline verdict @ short horizons (F&O is a short-term read)
    print("=" * 96); print("VERDICT — is the F&O futures score tradeable?"); print("=" * 96)
    for h in [1, 3, 5]:
        a = evaluate(m_fut, "fno_avg_score", f"fwd_{h}")
        d = evaluate(m_fut, "dv5d", f"fwd_{h}")
        if a and d:
            verdict = ("ADDS edge" if a["ic"] > 0.03 and a["icir"] > 0.2 else
                       "WEAK / no edge")
            print(f"  h={h}: fno_avg_score IC={a['ic']:+.3f} (IR {a['icir']:+.2f}, t {a['t']:+.1f})"
                  f"  vs dv5d IC={d['ic']:+.3f}   -> {verdict}")
    print("\nDIAGNOSTIC_DONE")


if __name__ == "__main__":
    main()
