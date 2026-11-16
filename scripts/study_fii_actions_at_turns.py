"""
What was FII DOING (buildup / unwinding / covering) just before and just after Nifty's
swing tops and bottoms -- in index futures and in index options?

Futures sides:  L = index-futures longs,  S = index-futures shorts.
Options sides:  L = calls bought + puts written   (bullish exposure)
                S = calls written + puts bought   (bearish exposure)
Each side's change is taken over 5 sessions as a % of that side's own 60-session
average (counts are not comparable across lot-size changes; percentages are), and the
bigger move names the action:
   LONG/BULLISH BUILD  (L up)       SHORT/BEARISH BUILD (S up)
   LONG/BULLISH UNWIND (L down)     SHORT/BEARISH COVER (S down)
Turns: 3% zigzag on Nifty closes. "Before" = the 5 sessions ending ON the turn day,
"after" = the 5 sessions starting the day after. Compared with the same action's share
of ALL days, with a circular-shift permutation for the biggest cells.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                          # noqa: BLE001
    pass

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from src.analytics import big_players as bp, positioning_cycle as pc   # noqa: E402

AS = date(2026, 11, 13)
RNG = np.random.default_rng(11)
NAMES = {"LB": "build L", "SB": "build S", "LU": "unwind L", "SC": "cover S", "FLAT": "flat"}


def states(L: pd.Series, S: pd.Series) -> pd.Series:
    sl = L.diff(5) / L.rolling(60, min_periods=20).mean() * 100
    ss = S.diff(5) / S.rolling(60, min_periods=20).mean() * 100
    return pd.Series([bp._flow_state(a, b) for a, b in zip(sl.values, ss.values)], index=L.index)


def main() -> None:
    wide, px, exp = bp._load(AS)
    f = wide["FII"].dropna(subset=["fut_idx_long"])
    idx = f.index
    c = px["close_val"].reindex(idx).ffill().values
    books = {
        "FUTURES": (f["fut_idx_long"], f["fut_idx_short"]),
        "OPTIONS": (f["opt_idx_call_long"] + f["opt_idx_put_short"], f["opt_idx_call_short"] + f["opt_idx_put_long"]),
    }
    zz = [(i, k) for i, k in pc._zigzag(c, 0.03) if 10 <= i < len(c) - 6]
    for book, (L, S) in books.items():
        st_ = states(L.astype(float), S.astype(float)).values
        valid = st_ != "FLAT"
        base = {k: 100 * np.mean(st_[valid] == k) for k in ("LB", "SB", "LU", "SC")}
        print(f"\n===== {book}   all days: " + "  ".join(f"{NAMES[k]} {v:.0f}%" for k, v in base.items()))
        for kind in ("TOP", "BOTTOM"):
            ti = np.array([i for i, k in zz if k == kind])
            for when, off in (("5 sessions BEFORE (ending on turn)", 0), ("5 sessions AFTER", 5)):
                lab = st_[ti + off]
                row, ps = [], []
                for k in ("LB", "SB", "LU", "SC"):
                    obs = 100 * np.mean(lab == k)
                    null = []
                    for _ in range(2000):
                        sh = (ti + off + RNG.integers(30, len(st_) - 30)) % len(st_)
                        null.append(100 * np.mean(st_[sh] == k))
                    null = np.array(null)
                    p = min(1.0, min((null >= obs).mean(), (null <= obs).mean()) * 2)
                    row.append(f"{NAMES[k]} {obs:4.0f}% (base {base[k]:3.0f}, p {p:.3f})")
                print(f"  {kind:<6} {when:<36} n={len(ti)}  " + " | ".join(row))
    print("\n4 actions x 2 windows x 2 turn kinds x 2 books = 32 tests; Bonferroni p < 0.0016")


if __name__ == "__main__":
    main()
