"""
Weekly (5-day) index outlook — a MEAN-REVERSION bias, validated and distinct from
the next-day engine.

WHY MEAN-REVERSION (not trend)
------------------------------
Walk-forward across all four indices (point-in-time, leakage-free) shows the 5-day
forward index move is reliably MEAN-REVERTING: how *extended* price is (position in
its 20-day range + RSI) has a consistent NEGATIVE rank-IC vs the forward 5-day
return (~ -0.11, same sign on every index). i.e. overbought/extended → pulls back;
oversold/below-VWAP → bounces. The oversold tercile beat the overbought tercile by
~+0.5%/week.

This is the OPPOSITE of the next-day directional engine (where trend/momentum
signals like raw VWAP/RSI have no 1-day edge). So this is a separate, contrarian
*weekly* lens — modest but real (IC ~0.11, ~52% sign-hit). Treat it as a tilt, not
a precise forecast.

EXTENSION SCORE  = mean( position-in-20D-range[0..1] , RSI(14)/100[0..1] )  → 0..1
  high (>=0.65) = overbought / extended  → weekly bias DOWN (fade)
  low  (<=0.35) = oversold                → weekly bias UP   (bounce)
Point-in-time: only data <= as_of_date is used.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_weekly_outlook"]

# (display label, index_data name)
_INDICES = [
    ("Nifty 50",     "Nifty 50"),
    ("Bank Nifty",   "Nifty Bank"),
    ("Fin Nifty",    "Nifty Financial Services"),
    ("Midcap Nifty", "Nifty Midcap Select"),
]

_OVERSOLD   = 0.35   # extension <= this → oversold (weekly UP bias)
_OVERBOUGHT = 0.65   # extension >= this → overbought (weekly DOWN bias)
# Validated edge (walk-forward, 5d): oversold tercile +0.36%, overbought -0.14%.
_REVERT_UP   = 0.36
_REVERT_DOWN = -0.14
VALIDATION = "5d rank-IC -0.11 (consistent across 4 indices) · oversold−overbought spread +0.5%/wk · ~52% sign-hit"


def _extension_series(h: pd.DataFrame) -> Optional[pd.DataFrame]:
    if h is None or len(h) < 25:
        return None
    c  = h["close_val"].astype(float)
    hi = h["high_val"].astype(float)
    lo = h["low_val"].astype(float)
    rng_hi = hi.rolling(20).max()
    rng_lo = lo.rolling(20).min()
    pos = (c - rng_lo) / (rng_hi - rng_lo).replace(0, np.nan)          # 0..1 in 20D range
    d = c.diff(); g = d.clip(lower=0); l = (-d).clip(lower=0)
    rsi = 100.0 - 100.0 / (1.0 + g.ewm(alpha=1 / 14, adjust=False).mean()
                                 / l.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan))
    ext = (pos.astype(float) + rsi.astype(float) / 100.0) / 2.0
    out = pd.DataFrame({"close": c, "pos": pos.astype(float), "rsi": rsi.astype(float), "ext": ext})
    return out


def get_weekly_outlook(as_of_date: date) -> list[dict]:
    """5-day mean-reversion outlook per index. Returns one dict per index."""
    results: list[dict] = []
    start = as_of_date - timedelta(days=260)
    for label, idx_name in _INDICES:
        h = query_dataframe("""
            SELECT trade_date, high_val, low_val, close_val
            FROM index_data WHERE index_name = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, [idx_name, start, as_of_date])
        ext_df = _extension_series(h)
        if ext_df is None or ext_df["ext"].dropna().empty:
            continue
        cur = ext_df.dropna(subset=["ext"]).iloc[-1]
        ext = float(cur["ext"]); rsi = float(cur["rsi"]); pos = float(cur["pos"])
        # percentile of current extension within the trailing ~120 sessions (point-in-time)
        tail = ext_df["ext"].dropna().tail(120)
        pctl = float((tail <= ext).mean()) if len(tail) else 0.5

        if ext <= _OVERSOLD:
            bias, dir_, exp = "UP", 1, _REVERT_UP
            rationale = ("Oversold — bottom of its 20-day range with low RSI. History favours a "
                         "mean-reversion BOUNCE over the next week.")
        elif ext >= _OVERBOUGHT:
            bias, dir_, exp = "DOWN", -1, _REVERT_DOWN
            rationale = ("Overbought — top of its 20-day range with high RSI. History favours a "
                         "PULLBACK / consolidation over the next week.")
        else:
            bias, dir_, exp = "NEUTRAL", 0, 0.0
            rationale = "Mid-range — no extension edge; weekly bias neutral."

        results.append({
            "label": label, "index_name": idx_name,
            "close": round(float(cur["close"]), 1),
            "extension": round(ext, 2), "extension_pctl": round(pctl * 100, 0),
            "rsi": round(rsi, 0), "pos_in_range": round(pos * 100, 0),
            "weekly_bias": bias, "direction": dir_,
            "expected_5d_pct": exp, "rationale": rationale,
        })
    return results
