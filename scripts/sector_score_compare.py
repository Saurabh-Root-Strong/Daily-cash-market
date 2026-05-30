"""
READ-ONLY: compare the CURRENT sector composite score (min-max blend) against a
NEW rank-based, RS-aware candidate — on real data, walk-forward.

Reuses build_factors() from factor_ic_diagnostic so the factor engineering is
identical. Each trading day we cross-sectionally score every sector with both
methods, then measure how well each score predicts FORWARD sector returns:

  IC    = mean daily Spearman(score, forward return)
  ICIR  = mean(IC)/std(IC)
  spr%  = avg(top-tercile fwd return − bottom-tercile fwd return)

CURRENT   = min-max(dv5d)*30 + min-max(dv_ratio)*20 + min-max(z)*10
            + min-max(price_mom_2w)*10        (breadth & slope omitted: breadth
            needs per-stock data not in this panel; slope has ~0 IC. The 20%
            breadth + 10% slope are dropped from BOTH scores so the comparison
            isolates normalization + RS, apples-to-apples.)

NEW       = cross-sectional rank blend, RS-aware, slope dropped:
            rank(dv5d)*0.30 + rank(dv_ratio)*0.20 + rank(rs_2w)*0.30
            + rank(rank_pct)*0.10 + rank(z_log)*0.10
            (momentum/RS up-weighted because the factor IC study showed it is the
            single strongest predictor; delivery kept as the differentiated edge.)

Nothing is written. Safe to delete.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.factor_ic_diagnostic import (   # noqa: E402
    load_panel, load_nifty, build_factors, evaluate, HORIZONS,
)


def _xs_minmax(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-day cross-sectional min-max to [0,1] (mirrors live _normalize, but XS)."""
    def f(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    return df.groupby("trade_date")[col].transform(f)


def _xs_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """Per-day cross-sectional percentile rank in [0,1] (distribution-free)."""
    return df.groupby("trade_date")[col].rank(pct=True)


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── CURRENT-style: min-max blend (breadth+slope dropped from both) ─────────
    # reweight the live 30/20/10/10 (of the kept four) to sum to 1.0
    cur = (
        _xs_minmax(df, "dv5d")         * 0.30 +
        _xs_minmax(df, "dv_ratio")     * 0.20 +
        _xs_minmax(df, "z_score")      * 0.10 +
        _xs_minmax(df, "price_mom_2w") * 0.10
    ) / 0.70
    df["score_current"] = cur

    # ── NEW: cross-sectional rank, RS-aware, slope dropped ─────────────────────
    new = (
        _xs_rank(df, "dv5d")      * 0.30 +   # sustained delivery (differentiated edge)
        _xs_rank(df, "dv_ratio")  * 0.20 +   # today's delivery freshness
        _xs_rank(df, "rs_2w")     * 0.30 +   # relative strength vs Nifty (was discarded!)
        _xs_rank(df, "rank_pct")  * 0.10 +   # robust delivery percentile
        _xs_rank(df, "z_log")     * 0.10     # log-z (fat-tail-safe abnormality)
    )
    df["score_new"] = new

    # Variants to see which lever matters
    df["score_new_nors"] = (        # new method but WITHOUT RS (isolate RS value)
        _xs_rank(df, "dv5d")     * 0.40 +
        _xs_rank(df, "dv_ratio") * 0.30 +
        _xs_rank(df, "rank_pct") * 0.15 +
        _xs_rank(df, "z_log")    * 0.15
    )
    df["score_rs_only"] = _xs_rank(df, "rs_2w")     # pure RS benchmark
    df["score_dv5d_only"] = _xs_rank(df, "dv5d")    # pure delivery benchmark
    return df


def main() -> None:
    panel = load_panel()
    nifty = load_nifty()
    print(f"panel: {panel['sector'].nunique()} sectors | {panel['trade_date'].nunique()} days "
          f"| {panel['trade_date'].min().date()} → {panel['trade_date'].max().date()}")
    df = build_factors(panel, nifty)
    df = build_scores(df)

    scores = ["score_current", "score_new", "score_new_nors",
              "score_rs_only", "score_dv5d_only"]

    print("\n" + "=" * 92)
    print("COMPOSITE SCORE → forward-return IC / ICIR / tercile-spread%  (higher = better)")
    print("=" * 92)
    head = f"{'score':<18}"
    for h in HORIZONS:
        head += f"  h={h:<2d} IC/IR/spr      "
    print(head)
    print("-" * 92)
    for sc in scores:
        line = f"{sc:<18}"
        for h in HORIZONS:
            r = evaluate(df, sc, f"fwd_{h}")
            if r:
                line += f"  {r['ic']:+.3f}/{r['icir']:+.2f}/{r['spr']:+.2f}".ljust(19)
            else:
                line += f"  {'—':<17}"
        print(line)

    # Headline verdict at the swing horizons that matter (10d, 20d)
    print("\n" + "=" * 92)
    print("VERDICT (does NEW beat CURRENT?)")
    print("=" * 92)
    for h in [5, 10, 20]:
        cur = evaluate(df, "score_current", f"fwd_{h}")
        new = evaluate(df, "score_new", f"fwd_{h}")
        if cur and new:
            d_ic  = new["ic"] - cur["ic"]
            d_spr = new["spr"] - cur["spr"]
            win = "NEW wins" if new["ic"] > cur["ic"] else "current wins"
            print(f"  h={h:<2d}:  current IC {cur['ic']:+.3f} (spr {cur['spr']:+.2f}%)  "
                  f"vs  NEW IC {new['ic']:+.3f} (spr {new['spr']:+.2f}%)   "
                  f"ΔIC {d_ic:+.3f}  Δspr {d_spr:+.2f}%  → {win}")
    print("\nCOMPARE_DONE")


if __name__ == "__main__":
    main()
