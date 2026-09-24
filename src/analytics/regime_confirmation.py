"""
Regime Confirmation Board - is a downtrend actually turning, or not yet?

WHAT THIS IS
------------
A downtrend does not become an uptrend because it fell "enough". It becomes one
when a structural event confirms it. This module reports, per index, which rung
of that ladder the market is standing on right now, and attaches the MEASURED
historical outcome of that rung - never an assertion.

THE LADDER
----------
    NO SETUP          no established downtrend; nothing to confirm
    ARMED             downtrend established (>=5 sessions below the 20-DMA and
                      >=3% off the 20-day high), waiting. NOT a buy.
    CONFIRMED         close back above the 20-DMA from an ARMED state  (rule C1)
    CONFIRMED STRONG  also closed above the last confirmed swing high  (rule C3)
    FAILED            a confirmation fired, then 2 closes back below the 20-DMA

NO EDGE IS ESTABLISHED.  READ THIS BEFORE TRADING ANYTHING HERE.
-----------------------------------------------------------------
Pooled over 34 NSE indices, 2013-02..2026-09. Path-dependent trades: enter at the
NEXT session's close after the signal, exit on 2 closes back below the 20-DMA, a
2xATR trailing stop, or 40 sessions. 6bps round trip. Fractal(k=3) pivots usable
only k bars after they form (verified: 60/60 truncated-history classifications
match, 0 causality violations).

An earlier version of this file claimed a validated edge (C1 +0.35%, C3 +1.31%
vs a hold-matched null). THAT CLAIM IS WITHDRAWN. The null behind it resampled
each trade independently, ignoring that 34 correlated indices fire on the same
dates, which inflated the precision. With inference CLUSTERED BY DATE and the
benchmark set to each index's own TRAILING 500-session drift over the same
holding length (alpha, fully causal):

    era         C1 alpha   t        C3 alpha   t
    2013-2016     +0.14%  +0.32       +0.18%  +0.24
    2017-2019     -0.13%  -0.26       -0.23%  -0.31
    2020-2022     +1.59%  +0.97       +8.07%  +1.06     <- huge but NOT significant
    2023-2026     -1.87%  -4.69       -0.35%  -0.44     <- the only significant cell

Not one positive cell reaches significance. C1's Cochran Q = 15.72 (3 df,
p=0.001, I^2=81%): the eras genuinely differ, so no single pooled number is
meaningful -- and indeed the full-sample estimate flips with the weighting
(+1.42x the index per day in market if return-weighted, ~0 if eras are equally
weighted, -0.68% if inverse-variance weighted). An estimate that unstable is the
definition of no robust effect. C3 shows no heterogeneity (Q p=0.70, I^2=0%) and
no edge either; its former "z=+3.22" came from the variance of a few 2020-2022
trades (alpha +8.07% with a standard error of 7.60).

THE ONE ROBUST FINDING, agreed by three independent methods, is that the CURRENT
era is NEGATIVE:
    clustered alpha 2023-2026   -1.87%  (t -4.69)
    hold-matched null, 6y       rule -0.12% vs null +0.72%
    bps per day in market       rule  -2.2  vs index  +4.6

LIKELY MECHANISM (coherent, but NOT tradable)
---------------------------------------------
    era         ann.vol   VR(10)   AC(5)
    2013-2016     15.9%     1.01   -0.005
    2017-2019     12.2%     0.94   -0.034
    2020-2022     22.7%     1.12   +0.191   <- high vol, 5-day momentum
    2023-2026     12.6%     0.92   -0.073   <- low vol, mean-reverting

A reclaim rule needs positive multi-day autocorrelation to pay. 2020-2022 had it;
the current era does not. But conditioning on causal trailing VR or realised vol
FAILS to recover the edge: with 2020-2022 removed, every volatility and variance-
ratio quartile is negative (vol Q4 ex-2020-22 alpha -1.64%, t -2.21). So the
mechanism explains the history without yielding a rule.

OTHER MEASURED FACTS
--------------------
  * ACTING LATE: entering 1-2 sessions after the confirmation costs nothing
    (+0.77%/+0.78% vs +0.73% at lag 0). By session 3 it drops ~40% (+0.46%),
    and for C3 it collapses (+1.11% -> +0.16%). The window is ~2 sessions.
  * The MIN_DAYS_BELOW gate barely matters: min 1 through min 10 all return
    +0.75%..+1.01%. "Establishedness" of the downtrend is not what carries it.
  * COSTS: +0.73% at 6bps -> +0.54% at 25bps -> +0.29% at 50bps. Anything with
    option spreads or theta on top of this is not viable.
  * ARMED -> CONFIRMED takes a MEDIAN OF 22 SESSIONS on Nifty (6y, 22 episodes).
    Confirmation on day 3-4 of a fall is rare: it happened once in 22 episodes.
  * 4 of 22 Nifty downtrend episodes in 6y never confirmed at all.

WHAT THIS IS NOT
----------------
Not a directional forecast, not an edge in the current regime, not a strategy.
  * 58.8% of confirmations LOSE money (full sample). It paid via payoff ratio
    (~2.4), never via hit rate. 36.4% die within 5 sessions.
  * Nifty 50, last 6y: rule +30.0% on 288 days in market (19% of sessions,
    max DD -3.5%) vs buy & hold +104.5% (max DD -17.2%). Lower drawdown and
    lower exposure, far lower return. It is a risk-reduction tool at best.

Era audit + entry-lag/cost sensitivity: 2026-09-13. Calibration source: scratchpad regime7/regime8/calibrate runs, 2026-09-13.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["ConfirmationState", "get_confirmation_board", "DEFAULT_INDICES"]

# -- knobs (sensitivity-swept: 27/27 cells positive across MA 10/20/50 x
#    lookback 3/5/8 x depth 2/3/5%, so none of these sits on a cliff) ---------
MA_LEN = 20
MIN_DAYS_BELOW = 5
MIN_DRAWDOWN = -0.03
PIVOT_K = 3
FAIL_WINDOW = 40

DEFAULT_INDICES = ["Nifty 50", "Nifty Bank", "Nifty Financial Services",
                   "NIFTY Midcap 100", "Nifty IT", "Nifty Auto", "Nifty Pharma",
                   "Nifty FMCG", "Nifty Metal", "Nifty Realty", "Nifty Energy"]

# -- measured outcomes per rung (pooled, 34 indices, 13.6y) ------------------
STATE_STATS = {
    "CONFIRMED STRONG": dict(n=321, win=.470, avg=.0194, payoff=2.90, hold=12.6, die5=.361),
    "CONFIRMED": dict(n=1702, win=.412, avg=.0098, payoff=2.37, hold=12.6, die5=.364),
    "ARMED": dict(n=2650, win=.489, avg=.0012, payoff=1.25, hold=2.7, die5=.907),
    "FAILED": dict(n=712, win=.426, avg=.0126, payoff=2.37, hold=12.8, die5=.362),
    "NO SETUP": None,
}

# depth of the downtrend when the confirmation arrived (sessions below the MA)
DEPTH_STATS = {
    "5-7d": dict(n=336, win=.351, avg=.0041, payoff=2.33, die5=.399),
    "8-12d": dict(n=371, win=.439, avg=.0098, payoff=2.12, die5=.372),
    "13-19d": dict(n=492, win=.429, avg=.0170, payoff=3.02, die5=.337),
    "20d+": dict(n=503, win=.416, avg=.0064, payoff=1.99, die5=.360),
}

# size of the fall that preceded the confirmation
FALL_STATS = {
    "3-4.5%": dict(n=773, win=.388, avg=.0062, payoff=2.40),
    "4.5-7%": dict(n=564, win=.450, avg=.0132, payoff=2.58),
    ">7%": dict(n=365, win=.403, avg=.0119, payoff=2.18),
}

ACTION = {
    "CONFIRMED STRONG": "Highest rung. No edge established in any era; ~53% lose. Description only.",
    "CONFIRMED": "State reached. 59% lose. Since 2023 this rule is NEGATIVE vs simply holding.",
    "ARMED": "DO NOT ACT. Buying here without confirmation tested to zero edge (p=0.23).",
    "FAILED": "Whipsawed. Re-entry on the next confirmation historically beat a fresh one (full sample).",
    "NO SETUP": "No established downtrend. This rule has nothing to say.",
}


@dataclass
class ConfirmationState:
    index_name: str
    trade_date: date
    state: str = "NO SETUP"
    close: float = float("nan")
    ma: float = float("nan")
    swing_high: float = float("nan")
    days_below: int = 0
    drawdown: float = float("nan")
    dist_c1: float = float("nan")      # % move needed to reclaim the MA
    dist_c2: float = float("nan")      # % move needed to break the swing high
    days_since_signal: Optional[int] = None
    depth_bucket: str = ""
    fall_bucket: str = ""
    stats: Optional[dict] = None
    note: str = ""
    error: str = ""


def _depth_bucket(d: int) -> str:
    if d < MIN_DAYS_BELOW:
        return ""
    if d <= 7:
        return "5-7d"
    if d <= 12:
        return "8-12d"
    if d <= 19:
        return "13-19d"
    return "20d+"


def _fall_bucket(dd: float) -> str:
    if not np.isfinite(dd) or dd > MIN_DRAWDOWN:
        return ""
    if dd > -0.045:
        return "3-4.5%"
    if dd > -0.07:
        return "4.5-7%"
    return ">7%"


def _causal_swing_high(high: np.ndarray, k: int = PIVOT_K) -> np.ndarray:
    """Last fractal swing high CONFIRMED as of each bar.

    A pivot at bar j needs k bars after it, so it only becomes usable at j+k.
    Carrying it forward with that lag is what keeps this free of lookahead.
    """
    n = len(high)
    piv = np.full(n, np.nan)
    for i in range(k, n - k):
        if high[i] == high[i - k:i + k + 1].max():
            piv[i] = high[i]
    out = np.full(n, np.nan)
    cur = np.nan
    for t in range(n):
        j = t - k
        if j >= 0 and not np.isnan(piv[j]):
            cur = piv[j]
        out[t] = cur
    return out


def _classify(df: pd.DataFrame, index_name: str) -> ConfirmationState:
    c = df["close_val"].to_numpy(float)
    h = df["high_val"].to_numpy(float)
    n = len(c)
    st = ConfirmationState(index_name=index_name, trade_date=df["trade_date"].iloc[-1])
    if n < MA_LEN + PIVOT_K + 5:
        st.error = f"insufficient history ({n} rows)"
        return st

    ma = pd.Series(c).rolling(MA_LEN).mean().to_numpy()
    roll_max = pd.Series(c).rolling(MA_LEN).max().to_numpy()
    dd = c / roll_max - 1.0
    below = c < ma
    swing = _causal_swing_high(h)

    days_below = np.zeros(n, int)
    run = 0
    for i in range(n):
        run = run + 1 if below[i] else 0
        days_below[i] = run

    armed = (days_below >= MIN_DAYS_BELOW) & (dd <= MIN_DRAWDOWN)
    prev_armed = np.r_[False, armed[:-1]]
    c1 = prev_armed & (c > ma)
    c3 = c1 & (c > np.nan_to_num(swing, nan=np.inf))

    i = n - 1
    st.close, st.ma, st.swing_high = c[i], ma[i], swing[i]
    st.days_below, st.drawdown = int(days_below[i]), float(dd[i])
    st.dist_c1 = ma[i] / c[i] - 1.0
    st.dist_c2 = (swing[i] / c[i] - 1.0) if np.isfinite(swing[i]) else float("nan")

    lo = max(0, i - FAIL_WINDOW)
    recent = np.flatnonzero(c1[lo:i + 1])
    last_sig = (lo + recent[-1]) if len(recent) else None

    if c3[i]:
        st.state = "CONFIRMED STRONG"
        st.days_since_signal = 0
    elif c1[i]:
        st.state = "CONFIRMED"
        st.days_since_signal = 0
    elif last_sig is not None:
        st.days_since_signal = int(i - last_sig)
        seg = below[last_sig + 1:i + 1]
        failed = len(seg) >= 2 and bool(seg[-1] and seg[-2])
        if failed:
            st.state = "FAILED"
        elif c[i] > ma[i]:
            st.state = "CONFIRMED"
        else:
            st.state = "ARMED" if armed[i] else "NO SETUP"
    elif armed[i]:
        st.state = "ARMED"
    else:
        st.state = "NO SETUP"

    if st.days_below >= MIN_DAYS_BELOW:
        st.depth_bucket = _depth_bucket(st.days_below)
        st.fall_bucket = _fall_bucket(st.drawdown)
    st.stats = STATE_STATS.get(st.state)
    st.note = ACTION.get(st.state, "")
    return st


def get_confirmation_board(
    trade_date: date,
    indices: Optional[list] = None,
    lookback_sessions: int = 260,
) -> pd.DataFrame:
    """One row per index: which rung of the confirmation ladder it stands on.

    All features are causal as of `trade_date`; nothing after it is read.
    """
    names = indices or DEFAULT_INDICES
    start = trade_date - timedelta(days=int(lookback_sessions * 1.6))
    rows = []
    for name in names:
        df = query_dataframe(
            """
            SELECT trade_date, high_val, low_val, close_val
            FROM index_data
            WHERE index_name = ? AND trade_date BETWEEN ? AND ?
              AND close_val IS NOT NULL
            ORDER BY trade_date
            """,
            [name, start, trade_date],
        )
        if df.empty:
            rows.append(ConfirmationState(index_name=name, trade_date=trade_date,
                                          error="no data"))
            continue
        rows.append(_classify(df, name))

    out = pd.DataFrame([{
        "Index": s.index_name,
        "State": s.state,
        "Close": s.close,
        "20-DMA": s.ma,
        "Sessions below": s.days_below,
        "Off 20d high": s.drawdown,
        "Move to confirm": s.dist_c1,
        "Move to STRONG": s.dist_c2,
        "Depth bucket": s.depth_bucket,
        "Fall bucket": s.fall_bucket,
        "Hist win%": (s.stats or {}).get("win"),
        "Hist avg": (s.stats or {}).get("avg"),
        "Hist payoff": (s.stats or {}).get("payoff"),
        "Dies <=5d": (s.stats or {}).get("die5"),
        "Read": s.note,
        "Error": s.error,
    } for s in rows])
    order = {"CONFIRMED STRONG": 0, "CONFIRMED": 1, "ARMED": 2, "FAILED": 3, "NO SETUP": 4}
    return out.sort_values("State", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)


def _print_board(bd: pd.DataFrame) -> None:
    icon = {"CONFIRMED STRONG": "[++]", "CONFIRMED": "[+]", "ARMED": "[!]",
            "FAILED": "[x]", "NO SETUP": "[ ]"}
    print(f"{'':5s}{'Index':26s} {'State':17s} {'Close':>9s} {'20-DMA':>9s} "
          f"{'below':>6s} {'off hi':>8s} {'to confirm':>11s} {'to STRONG':>10s}")
    print("-" * 112)
    for _, r in bd.iterrows():
        if r["Error"]:
            print(f"     {r['Index']:26s} -- {r['Error']}")
            continue
        d2 = f"{r['Move to STRONG']:+.2%}" if pd.notna(r["Move to STRONG"]) else "n/a"
        print(f"{icon.get(r['State'], '   '):5s}{r['Index']:26s} {r['State']:17s} "
              f"{r['Close']:9,.0f} {r['20-DMA']:9,.0f} {r['Sessions below']:6d} "
              f"{r['Off 20d high']:+8.2%} {r['Move to confirm']:+11.2%} {d2:>10s}")
    print()
    for state in ("CONFIRMED STRONG", "CONFIRMED", "ARMED", "FAILED"):
        sub = bd[bd["State"] == state]
        if sub.empty:
            continue
        s = STATE_STATS[state]
        print(f"  {state}: {', '.join(sub['Index'])}")
        print(f"    {ACTION[state]}")
        if s:
            print(f"    measured: n={s['n']}  win {s['win']:.1%}  avg {s['avg']:+.2%}  "
                  f"payoff {s['payoff']:.2f}  mean hold {s['hold']:.1f}d  "
                  f"dies<=5d {s['die5']:.0%}")
    depths = [d for d in bd["Depth bucket"] if d]
    if depths:
        print("\n  depth of downtrend at confirmation (descriptive, not a rule):")
        for k, v in DEPTH_STATS.items():
            mark = " <-- here" if k in depths else ""
            print(f"    {k:7s} n={v['n']:4d}  win {v['win']:.1%}  avg {v['avg']:+.2%}  "
                  f"payoff {v['payoff']:.2f}  dies<=5d {v['die5']:.0%}{mark}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Regime confirmation board")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest session)")
    ap.add_argument("--all", action="store_true", help="every index with enough history")
    args = ap.parse_args()

    if args.date:
        td = date.fromisoformat(args.date)
    else:
        td = query_dataframe("SELECT max(trade_date) AS d FROM index_data")["d"].iloc[0]
    names = None
    if args.all:
        names = query_dataframe(
            "SELECT index_name FROM index_data GROUP BY 1 HAVING count(*)>3000 ORDER BY 1"
        )["index_name"].tolist()

    print(f"\nREGIME CONFIRMATION BOARD - as of {td}\n")
    _print_board(get_confirmation_board(td, names))
