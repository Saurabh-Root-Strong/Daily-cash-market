"""
Trend Survival - how much longer does the current trend have?

WHY THIS EXISTS WHEN THE CONFIRMATION RULE DID NOT
--------------------------------------------------
Across this project's testing, trend RETURNS proved unpredictable: joining a
running trend gives alpha ~0 in every age bucket, both directions, once serial
overlap is removed (see regime_confirmation for the full retraction).

DURATION is a different target, and it IS predictable. Hazard of a trend ending
falls from 0.190/day at age 1-2 to 0.059 at age 8-12 (correlation of hazard with
log-age: -0.80). This module models only that.

It says HOW LONG, never WHICH WAY. A downtrend likely to run another 9 sessions
is NOT a short signal - raw short P&L is negative at every trend age (-0.13% to
-0.59% over 10 sessions). Use it for option expiry choice, stop distance,
position sizing, and for knowing whether a reversal thesis is early.

WHAT DRIVES THE ESTIMATE (and the honest split)
-----------------------------------------------
A trend "ends" when price crosses back through the 20-DMA, so DISTANCE FROM THE
MA predicts the ending mechanically (AUC 0.727). That is arithmetic, not insight,
and it does most of the work. Features were therefore judged on what they add
INSIDE distance quintiles:

    feature                      raw AUC   controlled for distance
    distance from MA               0.727   (mechanical baseline)
    momentum / volatility          0.661   0.582
    counter-direction days (5)     0.609   0.559
    change in efficiency ratio     0.614   0.542
    efficiency ratio               0.626   0.526
    volatility expansion           0.503   0.505  <- adds nothing, excluded

The STRONG/FADING grade built from the survivors shows a +30pp headline gap in
5-session survival - but holding DISTANCE AND AGE FIXED the genuine incremental
contribution is +9.9pp (positive in 45 of 47 cells). Both numbers are true; the
+9.9pp is the one that is not mechanical. Quote the cell probability, not the gap.

VALIDATION
----------
  * OOS calibration (table built pre-2020, tested 2020-2026): mean absolute
    error 1.7pp across six probability bands. Quoted probabilities are reliable.
  * Frozen-threshold OOS: in-sample gap +30pp, out-of-sample +32pp.
  * Nifty 50 alone: +33pp full sample, +33pp post-2020 (not a small-index effect).
  * Threshold sensitivity: gap +32/+32/+33/+32pp across four cutoff choices.
  * Era stability of the gap: +28/+30/+36/+37pp (down), +27/+30/+28/+23pp (up).

CAVEAT: the grade's cutoffs were hand-set after inspecting these features, so the
grade itself is partly in-sample. Its defence is the frozen-threshold OOS test
and the four-era stability, not an untouched holdout.

Table: trend_survival_table.json (197 cells, full history, min n=30 per cell).
Rebuild with: python -m src.analytics.trend_survival --recalibrate
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["TrendSurvival", "get_trend_survival", "grade_from_features"]

MA_LEN = 20
ER_WINDOW = 20
# distance-from-MA quintile edges, from the pooled calibration sample
DIST_EDGES = [1e-06, 0.007436, 0.015469, 0.025075, 0.039664, 1.3544]
AGE_BINS = [0, 2, 4, 7, 12, 19, 29, 9999]
AGE_LABELS = ["1-2", "3-4", "5-7", "8-12", "13-19", "20-29", "30+"]

_TABLE: Optional[dict] = None


def _table() -> dict:
    global _TABLE
    if _TABLE is None:
        p = Path(__file__).with_name("trend_survival_table.json")
        _TABLE = json.loads(p.read_text()) if p.exists() else {}
    return _TABLE


@dataclass
class TrendSurvival:
    index_name: str
    trade_date: date
    direction: str = "FLAT"          # UP | DOWN
    age: int = 0
    grade: str = "INTACT"            # STRONG | INTACT | FADING
    close: float = float("nan")
    ma: float = float("nan")
    dist: float = float("nan")       # how far INTO the trend, as a fraction
    er: float = float("nan")
    der5: float = float("nan")
    mom: float = float("nan")
    counter: int = 0
    p_survive_5: Optional[float] = None
    p_survive_10: Optional[float] = None
    median_left: Optional[float] = None
    cell_n: int = 0
    age_pctile: Optional[float] = None    # vs this index's own past trends
    note: str = ""
    error: str = ""


def _er(c: np.ndarray, w: int = ER_WINDOW) -> np.ndarray:
    """Kaufman efficiency ratio: |net move| / total path. 1 = clean, 0 = chop."""
    net = np.abs(c - np.roll(c, w))
    path = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(w).sum().to_numpy()
    out = np.where(path > 0, net / np.maximum(path, 1e-12), np.nan)
    out[:w] = np.nan
    return out


def grade_from_features(er: float, der5: float, counter: float, mom: float) -> str:
    """STRONG / INTACT / FADING. Deliberately excludes distance-from-MA, which is
    the mechanical predictor and is conditioned on separately in the table."""
    s = 0
    s += 1 if er >= 0.45 else (0 if er >= 0.25 else -1)
    s += 1 if der5 > 0 else -1
    s += -1 if counter >= 3 else 0
    s += 1 if mom > 0.5 else (0 if mom > -0.5 else -1)
    return "STRONG" if s >= 2 else ("INTACT" if s >= 0 else "FADING")


def _bucket(v: float, edges: list) -> int:
    return int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 2))


def _age_bucket(age: int) -> int:
    for i in range(len(AGE_BINS) - 1):
        if AGE_BINS[i] < age <= AGE_BINS[i + 1]:
            return i
    return len(AGE_BINS) - 2


def _compute(df: pd.DataFrame, index_name: str) -> TrendSurvival:
    c = df["close_val"].to_numpy(float)
    n = len(c)
    st = TrendSurvival(index_name=index_name, trade_date=df["trade_date"].iloc[-1])
    if n < MA_LEN + 30:
        st.error = f"insufficient history ({n} rows)"
        return st

    ma = pd.Series(c).rolling(MA_LEN).mean().to_numpy()
    r1 = pd.Series(c).pct_change().to_numpy()
    vol20 = pd.Series(r1).rolling(MA_LEN).std().to_numpy()
    E = _er(c)
    up = c > ma
    side = 1 if up[-1] else -1

    age = 1
    for i in range(n - 2, -1, -1):
        if up[i] == up[-1]:
            age += 1
        else:
            break

    i = n - 1
    st.direction = "UP" if up[-1] else "DOWN"
    st.age = age
    st.close, st.ma = c[i], ma[i]
    st.dist = side * (c[i] - ma[i]) / ma[i]
    st.er = float(E[i])
    st.der5 = float(E[i] - E[i - 5]) if np.isfinite(E[i - 5]) else 0.0
    st.mom = float(side * (c[i] / c[i - 5] - 1) / vol20[i]) if vol20[i] > 0 else 0.0
    st.counter = int(np.sum(np.sign(r1[-5:]) != side))
    st.grade = grade_from_features(st.er, st.der5, st.counter, st.mom)

    key = f"{int(up[-1])}|{_age_bucket(age)}|{_bucket(max(st.dist, 0), DIST_EDGES)}|{st.grade}"
    cell = _table().get(key)
    if cell:
        st.p_survive_5 = cell["p5"]
        st.p_survive_10 = cell["p10"]
        st.median_left = cell["med"]
        st.cell_n = cell["n"]

    # where this age sits among this index's own completed trends, same direction
    lens, run, cur = [], 1, up[0]
    for k in range(1, n):
        if up[k] == cur:
            run += 1
        else:
            if cur == up[-1]:
                lens.append(run)
            run, cur = 1, up[k]
    if lens:
        st.age_pctile = float(np.mean(np.array(lens) < age))

    st.note = {
        "STRONG": "Deterioration markers clean - trend intact, expect it to run.",
        "INTACT": "Mixed markers. No strong view on how much longer.",
        "FADING": "Momentum and/or efficiency deteriorating - shorter life expected.",
    }[st.grade]
    return st


def get_trend_survival(
    trade_date: date,
    indices: Optional[list] = None,
    lookback_sessions: int = 0,
) -> pd.DataFrame:
    """One row per index: trend direction, age, strength grade, survival odds.

    Causal as of `trade_date`; no data after it is read. Uses FULL prior history
    by default so the 'older than X% of past trends' percentile is meaningful --
    a short window makes that number depend on the window, not on the trend.
    """
    from src.analytics.regime_confirmation import DEFAULT_INDICES
    names = indices or DEFAULT_INDICES
    rows = []
    for name in names:
        df = query_dataframe(
            """
            SELECT trade_date, close_val
            FROM index_data
            WHERE index_name = ? AND trade_date <= ?
              AND close_val IS NOT NULL
            ORDER BY trade_date
            """,
            [name, trade_date],
        )
        if df.empty:
            rows.append(TrendSurvival(index_name=name, trade_date=trade_date, error="no data"))
            continue
        rows.append(_compute(df, name))

    return pd.DataFrame([{
        "Index": s.index_name,
        "Trend": s.direction,
        "Age": s.age,
        "Grade": s.grade,
        "Into trend": s.dist,
        "Efficiency": s.er,
        "Momentum": s.mom,
        "Counter days": s.counter,
        "P(+5d)": s.p_survive_5,
        "P(+10d)": s.p_survive_10,
        "Median left": s.median_left,
        "Cell n": s.cell_n,
        "Longer than": s.age_pctile,
        "Read": s.note,
        "Error": s.error,
    } for s in rows])


def recalibrate(min_cell: int = 30, out: Optional[Path] = None) -> int:
    """Rebuild trend_survival_table.json from index_data.

    One observation per (index, session) where the trend state is known. The
    label is the REMAINING life of that state, so the last sessions of each
    series are necessarily unlabelled and are dropped - not a lookahead, but the
    reason a freshly-rebuilt table never covers the final few sessions.
    """
    names = query_dataframe(
        "SELECT index_name FROM index_data GROUP BY 1 HAVING count(*)>3000 ORDER BY 1"
    )["index_name"].tolist()
    rows = []
    for name in names:
        df = query_dataframe(
            "SELECT trade_date, close_val FROM index_data WHERE index_name = ? "
            "AND close_val IS NOT NULL ORDER BY trade_date", [name])
        c = df["close_val"].to_numpy(float)
        n = len(c)
        if n < 200:
            continue
        ma = pd.Series(c).rolling(MA_LEN).mean().to_numpy()
        r1 = pd.Series(c).pct_change().to_numpy()
        vol = pd.Series(r1).rolling(MA_LEN).std().to_numpy()
        E = _er(c)
        up = c > ma
        age = np.zeros(n, int)
        a = 0
        for i in range(n):
            a = a + 1 if (i > 0 and up[i] == up[i - 1]) else 1
            age[i] = a
        rem = np.zeros(n, int)
        run = 0
        for i in range(n - 1, -1, -1):
            run = (run + 1) if (i < n - 1 and up[i] == up[i + 1]) else 0
            rem[i] = run
        side = np.where(up, 1, -1)
        dist = side * (c - ma) / ma
        ctr = pd.Series((np.sign(r1) != side).astype(float)).rolling(5).sum().to_numpy()
        r5 = pd.Series(c).pct_change(5).to_numpy()
        for i in range(60, n - 11):          # -11: last bars have no usable label
            if not np.isfinite(E[i]) or not np.isfinite(vol[i]) or vol[i] <= 0:
                continue
            g = grade_from_features(E[i], E[i] - E[i - 5], ctr[i], side[i] * r5[i] / vol[i])
            rows.append((int(up[i]), _age_bucket(age[i]),
                         _bucket(max(dist[i], 0.0), DIST_EDGES), g, rem[i]))
    D = pd.DataFrame(rows, columns=["up", "ab", "dq", "grade", "rem"])
    tab = {}
    for k, g in D.groupby(["up", "ab", "dq", "grade"], observed=True):
        if len(g) < min_cell:
            continue
        tab[f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"] = dict(
            p5=round(float((g.rem >= 5).mean()), 3),
            p10=round(float((g.rem >= 10).mean()), 3),
            med=float(g.rem.median()), n=int(len(g)))
    path = out or Path(__file__).with_name("trend_survival_table.json")
    path.write_text(json.dumps(tab))
    return len(tab)


def _print(bd: pd.DataFrame) -> None:
    icon = {"STRONG": "[===]", "INTACT": "[== ]", "FADING": "[=  ]"}
    print(f"{'':6s}{'Index':26s} {'Trend':6s} {'Age':>4s} {'Grade':8s} "
          f"{'into':>7s} {'ER':>5s} {'P+5d':>6s} {'P+10d':>6s} {'med':>5s} {'older than':>11s}")
    print("-" * 108)
    for _, r in bd.iterrows():
        if r["Error"]:
            print(f"      {r['Index']:26s} -- {r['Error']}")
            continue
        p5 = f"{r['P(+5d)']*100:.0f}%" if pd.notna(r["P(+5d)"]) else "n/a"
        p10 = f"{r['P(+10d)']*100:.0f}%" if pd.notna(r["P(+10d)"]) else "n/a"
        md = f"{r['Median left']:.0f}d" if pd.notna(r["Median left"]) else "n/a"
        pc = f"{r['Longer than']*100:.0f}%" if pd.notna(r["Longer than"]) else "n/a"
        print(f"{icon.get(r['Grade'], '     '):6s}{r['Index']:26s} {r['Trend']:6s} {r['Age']:4d} "
              f"{r['Grade']:8s} {r['Into trend']*100:+6.2f}% {r['Efficiency']:5.2f} "
              f"{p5:>6s} {p10:>6s} {md:>5s} {pc:>11s}")
    print("\n  P(+5d)/P(+10d) = chance this trend is still alive in 5 / 10 sessions.")
    print("  Calibrated out of sample to 1.7pp mean absolute error.")
    print("  HOW LONG, not WHICH WAY: this is not a direction signal. Shorting a")
    print("  downtrend loses money at every age tested.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Trend survival board")
    ap.add_argument("--date", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--recalibrate", action="store_true",
                    help="rebuild trend_survival_table.json from index_data")
    args = ap.parse_args()

    if args.recalibrate:
        print(f"rebuilt trend_survival_table.json: {recalibrate()} cells")
        raise SystemExit(0)

    td = (date.fromisoformat(args.date) if args.date
          else query_dataframe("SELECT max(trade_date) AS d FROM index_data")["d"].iloc[0])
    names = None
    if args.all:
        names = query_dataframe(
            "SELECT index_name FROM index_data GROUP BY 1 HAVING count(*)>3000 ORDER BY 1"
        )["index_name"].tolist()
    print(f"\nTREND SURVIVAL - as of {td}\n")
    _print(get_trend_survival(td, names))
