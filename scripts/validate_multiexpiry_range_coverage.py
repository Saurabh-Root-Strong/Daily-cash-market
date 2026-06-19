"""
validate_multiexpiry_range_coverage.py  —  are the WEEKLY / MONTHLY bands calibrated?

The Index-Prediction page shows two forward "to-expiry" bands beside the daily 1σ range:
  • WEEKLY  range (wk_*) — spot ± move to the next weekly expiry, from 0.6×ATM straddle
                           + 0.4×σ√T, bounded by that expiry's OI walls.
  • MONTHLY range (mn_*) — NIFTY only, same method off the MONTHLY chain.
Unlike the daily band (validated ~68% by validate_range_coverage.py), these two have
NEVER been checked against realised outcomes. This script does that.

prediction_log does NOT store the wk_/mn_ bands or the close AT expiry, so we REPLAY:
re-run the live engine as-of every historical F&O day (same path as backtest_signals.py,
no look-ahead), read wk_range_low/high + wk_expiry (and mn_* for NIFTY), then look up the
index's OHLC over the holding window (td → expiry) and measure containment.

Two honest coverage definitions, both reported:
  • TERMINAL — did the CLOSE at expiry land inside the band? (direct analogue of the
               daily validator's close-containment; the right test for "where will it
               settle by expiry".)
  • PATH     — did the index NEVER leave the band intraday between today and expiry?
               (max high ≤ range_high AND min low ≥ range_low — the test that matters
               for a stop sitting at the band edge.) Path coverage is always ≤ terminal.

Reports per index + pooled: terminal & path coverage, the half-width multiplier k needed
to hit the ~68% terminal target (recalibration knob), and average band half-width.

Usage:  python scripts/validate_multiexpiry_range_coverage.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root → import src.*

from src.data.repository import query_dataframe
from src.analytics.index_prediction import get_index_predictions

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_IDX = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank",
        "FINNIFTY": "Nifty Financial Services", "MIDCPNIFTY": "Nifty Midcap Select"}
CLAIM = 0.68
SEED = 7


def _boot_ci(mask: np.ndarray, reps: int, rng) -> tuple[float, float]:
    n = len(mask)
    if n == 0:
        return (float("nan"), float("nan"))
    bs = np.array([mask[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    # All index OHLC once, indexed by (index_name) → date-sorted frame.
    ohlc = query_dataframe(
        "select index_name, trade_date, high_val, low_val, close_val from index_data "
        "where close_val is not null order by index_name, trade_date"
    )
    ohlc["trade_date"] = pd.to_datetime(ohlc["trade_date"]).dt.date
    by_idx = {name: g.reset_index(drop=True) for name, g in ohlc.groupby("index_name")}
    last_data_date = ohlc["trade_date"].max()

    dates = [d.date() if hasattr(d, "date") else d for d in query_dataframe(
        "select distinct trade_date from fno_bhavcopy order by trade_date"
    )["trade_date"].tolist()]
    print(f"[harness] {len(dates)} F&O days: {dates[0]} .. {dates[-1]}", flush=True)

    rows = []
    for i, td in enumerate(dates):
        try:
            preds = get_index_predictions(td, persist=False)
        except Exception as e:
            print(f"  ! {td} compute failed: {e}", flush=True); continue
        for p in preds:
            if not getattr(p, "data_available", True) or p.spot_close is None:
                continue
            g = by_idx.get(_IDX[p.fno_symbol])
            if g is None:
                continue
            for kind, lo, hi, exp in (
                ("weekly", p.wk_range_low, p.wk_range_high, p.wk_expiry),
                ("monthly", p.mn_range_low, p.mn_range_high, p.mn_expiry),
            ):
                if lo is None or hi is None or exp is None:
                    continue
                if exp > last_data_date:
                    continue  # not yet resolved — no settled outcome
                # holding window: strictly after td, up to and including the expiry date
                win = g[(g["trade_date"] > td) & (g["trade_date"] <= exp)]
                if win.empty:
                    continue
                term_close = float(win.iloc[-1]["close_val"])      # close at (or just before) expiry
                win_hi = float(win["high_val"].max())
                win_lo = float(win["low_val"].min())
                half = (hi - lo) / 2.0
                rows.append({
                    "symbol": p.fno_symbol, "kind": kind, "date": td, "expiry": exp,
                    "spot": p.spot_close, "lo": lo, "hi": hi, "half": half,
                    "term_contained": (lo <= term_close <= hi),
                    "path_contained": (win_lo >= lo and win_hi <= hi),
                    "u_term": abs(term_close - p.spot_close) / half if half > 0 else np.nan,
                })
        if (i + 1) % 25 == 0:
            print(f"  .. {i+1}/{len(dates)} days", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No resolved weekly/monthly ranges to score."); return
    df.to_parquet("multiexpiry_range_coverage.parquet")

    rng = np.random.default_rng(SEED)
    for kind in ("weekly", "monthly"):
        dk = df[df["kind"] == kind]
        if dk.empty:
            continue
        print(f"\n{'='*82}\n{kind.upper()} to-expiry range coverage  "
              f"(claim: band contains the close ~{int(CLAIM*100)}% by expiry)\n{'='*82}")
        print(f"{'index':11} {'n':>4} {'TERMINAL cover':>22}  {'PATH cover':>11}  "
              f"{f'k→{int(CLAIM*100)}%':>6}  {'half_pts':>8}")
        print("-" * 82)
        syms = [s for s in _IDX if not dk[dk["symbol"] == s].empty]
        for sym in syms + ["POOLED"]:
            d = dk if sym == "POOLED" else dk[dk["symbol"] == sym]
            if len(d) < 10:
                continue
            term = d["term_contained"].to_numpy().astype(float)
            path = d["path_contained"].to_numpy().astype(float)
            lo_ci, hi_ci = _boot_ci(term, 2000, rng)
            k = float(np.nanquantile(d["u_term"].to_numpy(), CLAIM))  # half-width multiple → 68%
            flag = "OK" if (lo_ci <= CLAIM <= hi_ci) else ("WIDE" if term.mean() > CLAIM else "TIGHT")
            print(f"{sym:11} {len(d):>4} {100*term.mean():>9.1f}% [{100*lo_ci:>4.1f},{100*hi_ci:>4.1f}]  "
                  f"{100*path.mean():>9.1f}%  {k:>6.2f}  {d['half'].mean():>8.0f}  {flag}")

    # ── Out-of-sample calibration check (guards against in-sample overfit) ──
    # Fit the half-width multiplier k on the first 70% of dates, measure HELD-OUT
    # terminal coverage at that k vs at the current band (k=1). A "TIGHT" in-sample
    # flag that disappears OOS is regime-fit noise, NOT a real miscalibration — exactly
    # the trap that rejected a 75% daily target. NOTE: daily rows are autocorrelated
    # (many snapshots per expiry), so the INDEPENDENT sample is ~n/5 weekly / ~n/20
    # monthly expiries — treat thin-sample OOS coverage with caution.
    print("\nOUT-OF-SAMPLE  (fit k on first 70% of dates → held-out terminal coverage):")
    print(f"{'kind':8} {'index':11} {'n_te':>5} {'k(train)':>8} {'cov_te@k':>9} {'cov_te@1×':>9}")
    for kind in ("weekly", "monthly"):
        dk = df[df["kind"] == kind]
        for sym in [s for s in _IDX if not dk[dk["symbol"] == s].empty] + ["POOLED"]:
            d = dk if sym == "POOLED" else dk[dk["symbol"] == sym]
            dts = np.sort(d["date"].unique())
            if len(dts) < 40:
                continue
            cut = dts[int(len(dts) * 0.7)]
            tr, te = d[d["date"] < cut], d[d["date"] >= cut]
            if len(tr) < 20 or len(te) < 20:
                continue
            k = float(np.nanquantile(tr["u_term"].to_numpy(), CLAIM))
            cov_k = float((te["u_term"].to_numpy() <= k).mean())
            cov_1 = float((te["u_term"].to_numpy() <= 1).mean())
            print(f"{kind:8} {sym:11} {len(te):>5} {k:>8.2f} {100*cov_k:>8.1f}% {100*cov_1:>8.1f}%")

    print("\nRead:")
    print("  • TERMINAL = close at expiry inside the band (the 'where will it settle' test).")
    print("  • PATH = price never left the band before expiry (the stop-at-the-edge test); ≤ terminal.")
    print(f"  • k = the half-width multiplier that WOULD hit ~{int(CLAIM*100)}% terminal: k>1 → band too")
    print("    tight (widen by k); k<1 → too wide. flag: OK = 68% in CI · WIDE/ TIGHT otherwise.")
    print("  • The straddle-blend band ≈ a 'typical move' (E|move|) band, so terminal coverage")
    print("    below 68% is EXPECTED — this quantifies by how much, per index, to recalibrate.")


if __name__ == "__main__":
    main()
