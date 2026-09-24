"""Does WHEN / HOW a borrowing build-up is running change what a flag means?

    python scripts/build_slb_timeline_study.py      (after scripts/build_slb_study.py)

Writes data/slb_study/timeline.json.

PRE-REGISTERED (fixed before any result was seen; 3 questions x 2 horizons = 6 tests,
Bonferroni |t| > 2.64). Target = the study's sector-neutral forward return s5 / s21,
entry at the NEXT day's close. Flags = the board's exact definition.
    T1 flag age     fresh-flagged stocks: flag streak <= 5 sessions (NEW) minus
                    streak >= 11 sessions (OLD). Act on day one, or wait?
    T2 shorts' P&L  flagged stocks (fresh or heavy): close > 1.05 x average borrow
                    price (shorts LOSING, squeeze risk) minus close < 0.95 x it
                    (shorts WINNING).
    T3 covering     stocks heavy-flagged at any time in the last 20 sessions: pile
                    <= 70% of its episode peak (COVERING) minus >= 90% (HOLDING).
Quantities are ROLL-ADJUSTED and split-adjusted (src/analytics/slb_episodes.py);
average borrow price is the FIFO cost of the borrowed shares still open.
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analytics.slb_episodes import (START_FRACTION, effective_oi, episode_start,
                                        fifo_cost)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "slb_study")
H = (5, 21)
T_BAR = 2.64


def nw_t(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n))


def diff_test(df: pd.DataFrame, a: pd.Series, b: pd.Series, h: int) -> dict:
    """Daily mean of group a minus group b (both same-day), NW t, sign by year."""
    ma = df.loc[a].groupby("trade_date")[f"s{h}"].mean()
    mb = df.loc[b].groupby("trade_date")[f"s{h}"].mean()
    d = (ma - mb).dropna()
    yr = d.groupby(d.index.year).mean()
    t = nw_t(d.values, h)
    return {"diff_bps": round(float(d.mean()) * 1e4, 1), "t": round(t, 2),
            "n_days": int(len(d)), "n_a": int(a.sum()), "n_b": int(b.sum()),
            "a_bps": round(float(ma.mean()) * 1e4, 1), "b_bps": round(float(mb.mean()) * 1e4, 1),
            "years_same_sign": int((np.sign(yr) == np.sign(d.mean())).sum()),
            "n_years": int(len(yr)), "passes": bool(abs(t) > T_BAR)}


def main() -> None:
    panel = pd.read_parquet(os.path.join(OUT, "panel.parquet"))
    end = panel["trade_date"].max().date()
    eff = effective_oi(date(2017, 6, 1), end)
    print(f"effective outstanding: {len(eff):,} symbol-sessions", flush=True)

    # per symbol: FIFO cost, episode start / peak for every session
    feats = []
    for sym, g in eff.groupby("symbol"):
        g = g.sort_values("trade_date")
        e = g["eff"].values.astype(float)
        px = pd.Series(g["close_adj"].values, dtype=float).ffill().values
        cost = fifo_cost(e, px)
        start = np.zeros(len(e), dtype=int)
        peak = np.full(len(e), np.nan)
        for t in range(len(e)):
            if e[t] < 1000:
                continue
            s, _ = episode_start(e, t)
            start[t] = s
            peak[t] = np.nanmax(e[s:t + 1])
        feats.append(pd.DataFrame({
            "symbol": sym, "trade_date": g["trade_date"].values, "eff": e, "px": px,
            "cost": cost, "peak": peak, "age": np.arange(len(e)) - start + 1}))
    feats = pd.concat(feats, ignore_index=True)
    feats["trade_date"] = pd.to_datetime(feats["trade_date"])

    d = panel.loc[(panel["oi_qty"] > 0) & (panel["ca_recent"] == 0)].copy()
    g = d.groupby("trade_date")
    d["fresh"] = g["doi5"].rank(pct=True) > 0.9
    d["heavy"] = g["dtc"].rank(pct=True) > 0.9
    d = d.merge(feats, on=["trade_date", "symbol"], how="left")
    d = d.sort_values(["symbol", "trade_date"])

    # fresh-flag streak on the panel's own session grid
    sess = pd.Series(sorted(d["trade_date"].unique()))
    pos = pd.Series(np.arange(len(sess)), index=sess)
    d["pos"] = d["trade_date"].map(pos)
    streak = np.zeros(len(d), dtype=int)
    sym_v, pos_v, fr_v = d["symbol"].values, d["pos"].values, d["fresh"].values
    for i in range(len(d)):
        if fr_v[i]:
            cont = i > 0 and sym_v[i - 1] == sym_v[i] and pos_v[i - 1] == pos_v[i] - 1 and fr_v[i - 1]
            streak[i] = streak[i - 1] + 1 if cont else 1
    d["streak"] = streak
    # heavy at any time in the last 20 sessions (inclusive)
    d["heavy_recent"] = (d.groupby("symbol")["heavy"]
                          .transform(lambda s: s.astype(float).rolling(20, min_periods=1).max()) > 0)

    res = {}
    for h in H:
        x = d.loc[d[f"s{h}"].notna()]
        f = x["fresh"]
        res[f"T1_flag_age_{h}"] = diff_test(x, f & (x["streak"] <= 5), f & (x["streak"] >= 11), h)
        fl = x["fresh"] | x["heavy"]
        ratio = x["px"] / x["cost"]
        res[f"T2_shorts_pnl_{h}"] = diff_test(x, fl & (ratio > 1.05), fl & (ratio < 0.95), h)
        hr = x["heavy_recent"]
        off = x["eff"] / x["peak"]
        res[f"T3_covering_{h}"] = diff_test(x, hr & (off <= 0.70), hr & (off >= 0.90), h)
        for k, v in res.items():
            if k.endswith(f"_{h}"):
                print(f"{k:22s} diff {v['diff_bps']:+7.1f}bps t {v['t']:+.2f}  "
                      f"A {v['a_bps']:+.1f} B {v['b_bps']:+.1f}  yrs {v['years_same_sign']}/{v['n_years']}"
                      f"  nA {v['n_a']:,} nB {v['n_b']:,}", flush=True)

    # descriptive: how old is a build-up when the fresh flag first fires?
    first = d.loc[d["fresh"] & (d["streak"] == 1), "age"].dropna()
    ages = d.loc[d["fresh"] | d["heavy"], "age"].dropna()
    desc = {
        "first_fresh_flag_episode_age_median": float(first.median()),
        "first_fresh_flag_episode_age_p25_p75": [float(first.quantile(.25)), float(first.quantile(.75))],
        "flagged_episode_age_median": float(ages.median()),
        "fresh_streak_median": float(d.loc[d["fresh"], "streak"].median()),
        "start_fraction": START_FRACTION,
    }
    print(json.dumps(desc), flush=True)
    with open(os.path.join(OUT, "timeline.json"), "w") as fh:
        json.dump({"built_through": str(end), "tests": res, "describe": desc,
                   "bar": T_BAR}, fh, indent=1)


if __name__ == "__main__":
    main()
