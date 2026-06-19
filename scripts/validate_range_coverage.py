"""
validate_range_coverage.py  —  is the cockpit's next-day RANGE actually calibrated?

The Index-Prediction page shows a daily expected-move range (spot ± expected_move_pts,
from VIX/√252 blended with realized 20D vol) and asserts in move_basis:
    "~75% of days close within ±expected_move pts".
That 75% is an UNVERIFIED string. Direction on this page is already known to be a
coin-flip (hit-rate ~50%); the RANGE is the one output that can be honest — but only
if its stated coverage matches reality. This script measures it.

Source: prediction_log (logged daily, outcomes filled after the fact) already stores
spot_close, range_low, range_high, expected_move_pts, and actual_return (next-day %).
So we audit coverage directly — no engine replay:

    next_close = spot_close * (1 + actual_return/100)
    contained  = range_low <= next_close <= range_high
    u          = |next_close - spot_close| / expected_move_pts   # move in EM units

Reports, per index and pooled:
  • realized close-containment of the displayed range (the band is ±1 EM) vs the 75% claim
  • the EM-multiplier k needed to hit 75% / 80% target coverage (recalibration knob)
  • directional centering bias (is the band drifting off-center?)
with a bootstrap CI on the coverage proportion.

Usage:  python scripts/validate_range_coverage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
CLAIM = 0.68          # honest target: the band is a ~1σ band (68%), not the old "75%"
SEED = 7
ORDER = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]


def _boot_ci(mask: np.ndarray, reps: int, rng) -> tuple[float, float]:
    n = len(mask)
    bs = np.array([mask[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _report(name: str, d, rng, reps: int = 2000) -> None:
    contained = d["contained"].to_numpy()
    u = d["u"].to_numpy()
    signed = d["signed"].to_numpy()
    cover = contained.mean()
    lo, hi = _boot_ci(contained, reps, rng)
    k75 = float(np.quantile(u, CLAIM))          # EM-multiple that contains 75%
    k80 = float(np.quantile(u, 0.80))
    bias = float(np.mean(signed))               # >0 = closes land above center on avg
    flag = "OK" if (lo <= CLAIM <= hi) else ("WIDE" if cover > CLAIM else "TIGHT")
    print(f"{name:11} {len(d):>4} {100*cover:>6.1f}% [{100*lo:>4.1f},{100*hi:>4.1f}]   "
          f"{k75:>5.2f}   {k80:>5.2f}   {bias:>+5.2f}   {flag}")


def _oos(df, rng, target: float = CLAIM, frac: float = 0.7) -> None:
    """Fit the EM-multiplier on the first `frac` of dates, measure HELD-OUT coverage.
    Guards against the in-sample overfit a static multiplier invites — a multiplier
    that hits `target` in-sample but drifts far from it here is regime-fit, not real.
    (This is the check that rejected a 75% target and kept only the MIDCAP widening.)"""
    print("\nOut-of-sample check  (fit k on first 70% of dates → coverage on last 30%):")
    print(f"{'index':11} {'k(train)':>8} {'cover_te@k':>11} {'cover_te@1×':>12}")
    for sym in ORDER + ["POOLED"]:
        d = df if sym == "POOLED" else df[df.fno_symbol == sym]
        dts = np.sort(d.trade_date.unique())
        if len(dts) < 40:
            continue
        cut = dts[int(len(dts) * frac)]
        tr, te = d[d.trade_date < cut], d[d.trade_date >= cut]
        if len(tr) < 20 or len(te) < 20:
            continue
        k = float(np.quantile(tr["u"].to_numpy(), target))
        cov_k = float((te["u"].to_numpy() <= k).mean())
        cov_1 = float((te["u"].to_numpy() <= 1).mean())
        print(f"{sym:11} {k:>8.2f} {100*cov_k:>10.1f}% {100*cov_1:>11.1f}%")


def main() -> None:
    if not DB.exists():
        print(f"DB not found: {DB}"); return
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "select trade_date, fno_symbol, spot_close, range_low, range_high, "
        "       expected_move_pts, actual_return "
        "from prediction_log "
        "where outcome_filled and actual_return is not null and spot_close is not null "
        "  and range_low is not null and range_high is not null and expected_move_pts > 0"
    ).df()
    if df.empty:
        print("No filled predictions with range data."); return

    df["next_close"] = df["spot_close"] * (1 + df["actual_return"] / 100.0)
    df["contained"] = (df["next_close"] >= df["range_low"]) & (df["next_close"] <= df["range_high"])
    df["u"] = (df["next_close"] - df["spot_close"]).abs() / df["expected_move_pts"]
    # signed centering: where did the close land vs the band center, in EM units?
    center = (df["range_low"] + df["range_high"]) / 2.0
    df["signed"] = (df["next_close"] - center) / df["expected_move_pts"]

    rng = np.random.default_rng(SEED)
    print(f"\nNext-day RANGE coverage audit  (claim: ~{100*CLAIM:.0f}% close within ±1 expected-move)")
    print(f"prediction_log: {len(df)} filled outcomes, {df.trade_date.min()} → {df.trade_date.max()}")
    print("=" * 78)
    print(f"{'index':11} {'n':>4} {'cover@1×EM':>17}   {f'k→{int(CLAIM*100)}%':>5}   {'k→80%':>5}   {'bias':>5}   flag")
    print("-" * 78)
    for sym in ORDER:
        d = df[df.fno_symbol == sym]
        if len(d) >= 20:
            _report(sym, d, rng)
    print("-" * 78)
    _report("POOLED", df, rng)

    _oos(df, rng)

    cover = df["contained"].mean()
    print("\nRead:")
    print(f"  • These rows are the RAW ±1σ band from logged history. A per-index calibration "
          f"(_EM_CALIBRATION in index_prediction.py) now widens the band toward a consistent ~68%;")
    print(f"    the effect appears as new calibrated predictions accumulate in prediction_log.")
    print(f"  • Pooled raw coverage {100*cover:.1f}% vs the honest ~{100*CLAIM:.0f}% (1σ) target. "
          f"MIDCAP was the worst (vol most underestimated by India-VIX).")
    print("  • flag: OK = 68% inside CI · WIDE = covers more · TIGHT = covers less.")


if __name__ == "__main__":
    main()
