"""
backtest_verdict_outcomes.py  —  how does the PRUNED index-F&O engine actually score?

After the 2026-06-19 refocus (engine scoped to pure index-F&O: multi-expiry option
OI/vol/premium, futures near/next/far, FII/DII/Pro/Client positions + derivative
turnover; cash/fundamental/constituent signals pruned out of the vote), this replays
the LIVE engine as-of every historical F&O day (no look-ahead, same path as
signal_ic_diagnostic.py) and scores its VERDICT against the next-day index move.

For each (date, index) it captures the engine's call — UP / DOWN / SIDEWAYS, the
confidence, the composite, the ±1σ range and the sideways band — then looks up the
index's ACTUAL next-day return and tallies:

  Directional calls (UP/DOWN)
    sign_hit   = P(sign(next_ret) == call)                       vs 50% coin-flip
    strict_hit = call resolved beyond the sideways band in the called direction
  Sideways calls
    in_band    = |next_ret| <= sideways-band%   (the call said "no trend")
                 reported against the UNCONDITIONAL base rate of in-band days, so we
                 can see whether SIDEWAYS skill beats just always-saying-flat.
  Range (the trustworthy product, regardless of arrow)
    cover      = next close lands inside [range_low, range_high]   (~68% target)

  Combined "was the verdict right?"  — UP & ret>+band  |  DOWN & ret<-band  |
                 SIDEWAYS & |ret|<=band   → the single succeed/fail/sideways tally.

Read-only: store_prediction is monkeypatched to a no-op. Saves verdict_outcomes.parquet.

Usage:  python scripts/backtest_verdict_outcomes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Neutralize DB writes BEFORE importing the engine.
import src.analytics.memory_engine as _mem
_mem.store_prediction = lambda *a, **k: None

from src.analytics.index_prediction import get_index_predictions, _INDEX_MAP  # noqa: E402
from src.data.repository import query_dataframe                                # noqa: E402

_IDX_NAME = {s: idx for s, (_, idx) in _INDEX_MAP.items()}
ORDER = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
SEED = 7


def _next_day_returns(index_name: str) -> dict:
    df = query_dataframe(
        "SELECT trade_date, pct_chg FROM index_data "
        "WHERE index_name = ? AND pct_chg IS NOT NULL ORDER BY trade_date",
        [index_name],
    )
    if df.empty:
        return {}
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    rows = df.to_dict("records")
    return {rows[i]["trade_date"]: float(rows[i + 1]["pct_chg"]) for i in range(len(rows) - 1)}


def _boot_ci(mask: np.ndarray, rng, reps: int = 2000) -> tuple[float, float]:
    n = len(mask)
    if n == 0:
        return (float("nan"), float("nan"))
    bs = np.array([mask[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    dates_df = query_dataframe(
        "SELECT DISTINCT trade_date FROM fno_bhavcopy "
        "WHERE instrument IN ('FUTIDX','OPTIDX') ORDER BY trade_date"
    )
    if dates_df.empty:
        print("No FNO dates."); return
    dates = [d.date() if hasattr(d, "date") else d for d in dates_df["trade_date"]]
    dates = dates[:-1] if len(dates) > 1 else dates       # last day has no next-day outcome
    print(f"[harness] replaying {len(dates)} F&O days "
          f"({dates[0]} -> {dates[-1]}) x {len(_INDEX_MAP)} indices ...", flush=True)

    nxt = {idx: _next_day_returns(idx) for idx in _IDX_NAME.values()}

    rows = []
    for i, td in enumerate(dates):
        try:
            preds = get_index_predictions(td)
        except Exception as e:
            print(f"  ! {td} ERROR {e}", flush=True); continue
        for p in preds:
            if not p.data_available or p.spot_close is None:
                continue
            ret = nxt.get(_IDX_NAME.get(p.fno_symbol), {}).get(td)
            if ret is None:
                continue
            spot = float(p.spot_close)
            band_pts = p.sideways_band_pts if p.sideways_band_pts else (
                (p.expected_move_pts or 0.0) * 0.40)
            band_pct = (band_pts / spot * 100.0) if spot else 0.0
            next_close = spot * (1 + ret / 100.0)
            covered = (p.range_low is not None and p.range_high is not None
                       and p.range_low <= next_close <= p.range_high)
            # actual regime: did it move beyond the band, and which way?
            if ret > band_pct:
                actual = "UP"
            elif ret < -band_pct:
                actual = "DOWN"
            else:
                actual = "FLAT"
            v = p.direction                                   # UP / DOWN / SIDEWAYS
            if v in ("UP", "DOWN"):
                sign_hit = (ret > 0) if v == "UP" else (ret < 0)
                strict_hit = (actual == v)
                correct = strict_hit
            else:                                             # SIDEWAYS
                sign_hit = np.nan
                strict_hit = np.nan
                correct = (actual == "FLAT")
            rows.append({
                "symbol": p.fno_symbol, "date": td, "verdict": v,
                "confidence": p.confidence, "composite": float(p.composite_score),
                "ret": ret, "band_pct": band_pct, "in_band": abs(ret) <= band_pct,
                "actual": actual, "sign_hit": sign_hit, "strict_hit": strict_hit,
                "covered": bool(covered), "correct": bool(correct),
            })
        if (i + 1) % 25 == 0:
            print(f"  .. {i+1}/{len(dates)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No scored predictions."); return
    df.to_parquet("verdict_outcomes.parquet")
    rng = np.random.default_rng(SEED)

    # ── 1. VERDICT MIX + combined correctness ────────────────────────────────
    print(f"\n{'='*92}\nVERDICT OUTCOME SCORECARD  (pruned index-F&O engine, walk-forward, "
          f"n={len(df)} calls)\n{'='*92}")
    print(f"{'index':11} {'n':>4} {'UP':>4} {'DN':>4} {'SW':>4} "
          f"{'correct%':>9} {'[95% CI]':>14} {'range_cov%':>11} {'base_flat%':>10}")
    print("-" * 92)
    for sym in ORDER + ["POOLED"]:
        d = df if sym == "POOLED" else df[df.symbol == sym]
        if len(d) < 10:
            continue
        cor = d["correct"].to_numpy().astype(float)
        lo, hi = _boot_ci(cor, rng)
        nU = (d.verdict == "UP").sum(); nD = (d.verdict == "DOWN").sum()
        nS = (d.verdict == "SIDEWAYS").sum()
        base_flat = (d["actual"] == "FLAT").mean()
        print(f"{sym:11} {len(d):>4} {nU:>4} {nD:>4} {nS:>4} "
              f"{100*cor.mean():>8.1f}% [{100*lo:>4.1f},{100*hi:>4.1f}] "
              f"{100*d['covered'].mean():>10.1f}% {100*base_flat:>9.1f}%")

    # ── 2. DIRECTIONAL calls only (vs 50% coin-flip) ─────────────────────────
    dd = df[df.verdict.isin(["UP", "DOWN"])]
    print(f"\n{'-'*92}\nDIRECTIONAL CALLS ONLY (UP/DOWN) — does the arrow beat a coin-flip?\n{'-'*92}")
    print(f"{'index':11} {'n_dir':>6} {'sign_hit%':>10} {'[95% CI]':>14} "
          f"{'strict_hit%':>12} {'avg_ret_signed':>15}")
    print("-" * 92)
    for sym in ORDER + ["POOLED"]:
        d = dd if sym == "POOLED" else dd[dd.symbol == sym]
        if len(d) < 8:
            continue
        sh = d["sign_hit"].to_numpy().astype(float)
        lo, hi = _boot_ci(sh, rng)
        # signed: return in the called direction (UP→+ret, DOWN→−ret)
        signed = np.where(d.verdict == "UP", d.ret, -d.ret)
        print(f"{sym:11} {len(d):>6} {100*sh.mean():>9.1f}% [{100*lo:>4.1f},{100*hi:>4.1f}] "
              f"{100*d['strict_hit'].mean():>11.1f}% {signed.mean():>+14.3f}%")

    # ── 3. SIDEWAYS calls (vs base rate of flat days) ────────────────────────
    ds = df[df.verdict == "SIDEWAYS"]
    print(f"\n{'-'*92}\nSIDEWAYS CALLS — did 'no trend' actually stay in-band, vs the base flat rate?\n{'-'*92}")
    print(f"{'index':11} {'n_sw':>5} {'in_band%':>9} {'base_flat%':>11} {'edge_pp':>8}")
    print("-" * 92)
    for sym in ORDER + ["POOLED"]:
        d = ds if sym == "POOLED" else ds[ds.symbol == sym]
        base = df if sym == "POOLED" else df[df.symbol == sym]
        if len(d) < 8:
            continue
        ib = d["in_band"].mean()
        bf = base["in_band"].mean()
        print(f"{sym:11} {len(d):>5} {100*ib:>8.1f}% {100*bf:>10.1f}% {100*(ib-bf):>+7.1f}")

    # ── 4. composite IC + confidence stratification ──────────────────────────
    print(f"\n{'-'*92}\nCOMPOSITE IC + CONFIDENCE STRATIFICATION (directional skill, signed)\n{'-'*92}")
    cs, cr = df["composite"].to_numpy(), df["ret"].to_numpy()
    ic = pd.Series(cs).corr(pd.Series(cr), method="spearman")
    print(f"  pooled composite→next-ret Spearman IC = {ic:+.3f}  (n={len(df)})")
    for conf in ("HIGH", "MEDIUM", "LOW"):
        d = dd[dd.confidence == conf]
        if len(d) < 8:
            print(f"    {conf:7}: n={len(d):>4} (too few)"); continue
        print(f"    {conf:7}: n={len(d):>4}  sign_hit={100*d['sign_hit'].mean():>5.1f}%  "
              f"strict={100*d['strict_hit'].mean():>5.1f}%")

    print("\nRead:")
    print("  • correct% = combined verdict accuracy (UP/DN resolved beyond band in the called")
    print("    way, OR SIDEWAYS stayed in-band). Compare to base_flat% — most days ARE flat,")
    print("    so a high correct% driven by SIDEWAYS is the engine agreeing with the base rate.")
    print("  • sign_hit% is the honest directional test; ~50% = coin-flip (the known no-edge).")
    print("  • range_cov% is the trustworthy product — aim ~68%.")
    print("\nVERDICT_BACKTEST_DONE")


if __name__ == "__main__":
    main()
