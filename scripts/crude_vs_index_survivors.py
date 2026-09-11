"""Stage 3: try to KILL the grid's FDR survivors before anything is shown.

Natural gas 10d top-decile -> Energy / PSE / Metal / Oil&Gas next day, and
copper up-streak -> Nifty Metal next day, survived BH-FDR in stage 1. Each gets:
  * circular-shift permutation null (1,000 shifts)
  * control for the index's OWN 10-session momentum (a natgas rally and an
    energy-stock rally are the same event; the index may just be trending)
  * market-neutral: sector minus Nifty 50 (is it the sector, or the market?)
  * year-by-year excess (one year carrying it = one story, not a pattern)
  * episode-level test: first day of each run only, fully non-overlapping
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics.commodity_index import (START, align_to_sessions,  # noqa: E402
                                           commodity_features, episode_starts,
                                           index_outcomes, load_commodity,
                                           load_index_ohlc)
from crude_vs_index_study import ols_hac  # noqa: E402

IDX = ["Nifty 50", "Nifty Energy", "Nifty PSE", "Nifty Metal", "Nifty Oil & Gas"]
rng = np.random.default_rng(11)


def run(commodity: str, mask_fn, label: str, targets: list[str], outcome="oc1"):
    p = load_commodity(commodity)
    f = commodity_features(p)
    o, c = load_index_ohlc(IDX)
    a = align_to_sessions(f, c.index)
    oc = index_outcomes(o, c)
    valid = (a.r1.notna() & (a.index >= START)).values
    m = mask_fn(a).astype('boolean').fillna(False).to_numpy(bool) & valid
    print(f"\n######## {commodity}: {label}  -> {outcome}   ({m.sum()} days)")
    for ix in targets:
        y = oc[outcome][ix].values
        ok = valid & ~np.isnan(y)
        yy, xx = y[ok], m[ok]
        obs = yy[xx].mean() - yy[~xx].mean()
        T = len(yy)
        null = np.array([(lambda s: yy[s].mean() - yy[~s].mean())(np.roll(xx, k))
                         for k in rng.integers(60, T - 60, 1000)])
        pp = (np.abs(null) >= abs(obs)).mean()
        # own momentum control
        own = (c[ix].shift(1) / c[ix].shift(11) - 1).values
        ok2 = ok & ~np.isnan(own)
        X = np.column_stack([np.ones(ok2.sum()), m[ok2], own[ok2]])
        b, t = ols_hac(y[ok2], X, 5)
        # market neutral
        yn = y - oc[outcome]["Nifty 50"].values
        okn = ok & ~np.isnan(yn)
        bn, tn = ols_hac(yn[okn], np.column_stack([np.ones(okn.sum()), m[okn]]), 5)
        # episodes only
        st = episode_starts(pd.Series(m, index=c.index)).values & ok
        ep = y[st]
        base = y[ok]
        t_ep = (ep.mean() - base.mean()) / (base.std() / np.sqrt(len(ep)))
        yrs = pd.Series(y[ok & m] - np.nanmean(y[ok]), index=c.index[ok & m])
        by = (yrs.groupby(yrs.index.year).agg(["mean", "size"]))
        by["mean"] *= 100
        print(f"  {ix:16s} excess {obs*100:+.3f}%  perm p {pp:.3f} | "
              f"ctrl-own-mom t {t[1]:+.2f} (own t {t[2]:+.2f}) | "
              f"minus Nifty50 {bn[1]*100:+.3f}% t {tn[1]:+.2f} | "
              f"episodes {len(ep)} t {t_ep:+.2f}")
        print("     by year:", ", ".join(f"{y}:{r['mean']:+.2f}({int(r['size'])})"
                                         for y, r in by.iterrows()))


def main():
    ng = lambda a: a.pct10 >= 0.90
    run("NATURALGAS", ng, "10d move in top 10%",
        ["Nifty Energy", "Nifty PSE", "Nifty Metal", "Nifty Oil & Gas", "Nifty 50"])
    run("NATURALGAS", ng, "10d move in top 10%",
        ["Nifty Energy", "Nifty PSE", "Nifty Metal", "Nifty 50"], outcome="fwd5")
    run("COPPER", lambda a: a.up_streak >= 2, "up 2+ in a row",
        ["Nifty Metal", "Nifty 50"], outcome="oc1")
    run("COPPER", lambda a: a.up_streak >= 2, "up 2+ in a row",
        ["Nifty Metal"], outcome="gap")
    # placebo: does the SAME natgas rule 'work' with the signal shifted a month
    # later? a real lead effect should vanish; a regime coincidence persists.
    for sh in (-21, -5, 5, 21):
        run("NATURALGAS", lambda a, sh=sh: (a.pct10 >= 0.90).shift(sh),
            f"PLACEBO signal shifted {sh:+d} sessions", ["Nifty Energy", "Nifty Metal"])
    # cumulative BH over every grid run (all five commodities)
    import glob
    from crude_vs_index_study import bh
    G = pd.concat([pd.read_csv(f).assign(cmd=Path(f).parent.name)
                   for f in glob.glob(str(Path(__file__).resolve().parents[1]
                                          / "data/crude_study/*/forward_grid.csv"))])
    G["q_cum"] = bh(G.p_hac.values)
    tr = G.outcome != "gap"
    G.loc[tr, "q_cum_tr"] = bh(G.loc[tr, "p_hac"].values)
    print()
    print(f"CUMULATIVE: {len(G)} tests over {G.cmd.nunique()} commodities; "
          f"q<0.10: {(G.q_cum < .10).sum()}; tradable q<0.10: {(G.q_cum_tr < .10).sum()}")
    print(G.sort_values("p_hac").head(12)[["cmd", "condition", "index", "outcome",
          "n", "excess", "t", "q_cum", "q_cum_tr"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
