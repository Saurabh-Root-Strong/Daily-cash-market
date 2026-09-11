"""Stage 2 of the crude study: the relationship is REGIME-dependent.

Stage 1 (crude_vs_index_study.py) found the full-sample same-week correlation
of crude and Nifty ~0, but by year it swings from +0.26 (2020) to -0.68 (2026).
Questions here:
  1. How does the same-week link move over time (rolling, causal)?
  2. Is the regime PERSISTENT -- does the last 26 weeks' sign tell you the next
     26 weeks' sign? If not, a regime read is hindsight.
  3. Inside a causally-identified negative regime, does crude LEAD Nifty
     (forward, tradable), or is it only same-day/overnight?
  4. Plain-English table: weeks crude rose >3%, what Nifty did that same week,
     by year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crude_vs_index_study import (align, crude_features, load_crude,
                                  load_indices, ols_hac, outcomes, spear, START)

pd.set_option("display.width", 220)


def main() -> None:
    crude = load_crude()
    feat = crude_features(crude)
    o, c = load_indices()
    a = align(feat, c.index)
    out = outcomes(o, c)
    ok = a.r5.notna() & (c.index >= START)

    # non-overlapping 5-session blocks: crude r5 at the MCX close before the
    # session vs Nifty over the 5 sessions ending at the prior close (same5)
    idx = np.flatnonzero(ok.values)[::5]
    w = pd.DataFrame({
        "crude5": a.r5.values[idx],
        "n50": out["same5"]["Nifty 50"].values[idx],
        "bank": out["same5"]["Nifty Bank"].values[idx],
        "mid": out["same5"]["NIFTY Midcap 100"].values[idx],
        # what the NEXT block does, entered at the open after the crude read
        "n50_next": out["fwd5"]["Nifty 50"].values[idx],
        "gap": out["gap"]["Nifty 50"].values[idx],
    }, index=c.index[idx]).dropna()
    print(f"{len(w)} non-overlapping weeks {w.index.min().date()}..{w.index.max().date()}")

    # 1. rolling 26-week correlation, causal
    rc = w.crude5.rolling(26).corr(w.n50)
    print("\nRolling 26-week corr (crude week vs Nifty same week), sampled quarterly:")
    print(rc.resample("QE").last().round(2).to_string())
    print(f"share of weeks with trailing corr < -0.2: {(rc < -0.2).mean():.0%}, > +0.2: {(rc > 0.2).mean():.0%}")

    # 2. persistence: trailing 26w corr vs NEXT 26w corr, non-overlapping blocks
    fut = rc.shift(-26)
    blk = pd.DataFrame({"past": rc, "next": fut}).dropna().iloc[::26]
    print(f"\nPersistence, {len(blk)} disjoint half-years: corr(past, next) = "
          f"{blk.past.corr(blk.next):+.2f}; sign agrees {np.mean(np.sign(blk.past) == np.sign(blk.next)):.0%}")
    print(blk.round(2).to_string())

    # 3. inside a negative regime (causal: corr over the 26 weeks BEFORE this one)
    reg = rc.shift(1)
    for name, m in [("negative regime (trailing corr < -0.2)", reg < -0.2),
                    ("positive regime (trailing corr > +0.2)", reg > 0.2),
                    ("all weeks", reg.notna())]:
        sub = w[m.reindex(w.index).fillna(False)]
        if len(sub) < 15:
            continue
        up = sub.crude5 > 0.03
        X = np.column_stack([np.ones(len(sub)), sub.crude5.values])
        b_same, t_same = ols_hac(sub.n50.values, X, 1)
        b_nx, t_nx = ols_hac(sub.n50_next.values, X, 1)
        print(f"\n{name}: {len(sub)} weeks, crude up >3%: {up.sum()}")
        print(f"  SAME week : Nifty {sub.n50[up].mean() * 100:+.2f}% when crude >+3% vs "
              f"{sub.n50[~up].mean() * 100:+.2f}% otherwise; slope t {t_same[1]:+.2f}")
        print(f"  NEXT week : Nifty {sub.n50_next[up].mean() * 100:+.2f}% vs "
              f"{sub.n50_next[~up].mean() * 100:+.2f}%; slope t {t_nx[1]:+.2f}")

    # 4. plain table by year
    print("\nWeeks crude rose > +3%: Nifty 50 SAME week and NEXT week, by year")
    w["yr"] = w.index.year
    rows = []
    for y, g in w.groupby("yr"):
        u = g[g.crude5 > 0.03]
        rows.append(dict(year=y, weeks=len(g), crude_up_weeks=len(u),
                         nifty_same_avg=u.n50.mean() * 100,
                         nifty_same_down_pct=(u.n50 < 0).mean() * 100,
                         all_weeks_down_pct=(g.n50 < 0).mean() * 100,
                         nifty_next_avg=u.n50_next.mean() * 100,
                         corr_same=spear(g.crude5, g.n50)))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # 2026 in detail -- what crude and Nifty actually did
    m = crude[crude.index >= "2025-10-01"].close.resample("ME").last()
    n = c["Nifty 50"][c.index >= "2025-10-01"].resample("ME").last()
    print("\nMonth-end MCX crude (Rs/bbl) and Nifty 50:")
    print(pd.DataFrame({"crude": m, "nifty": n}).round(0).to_string())

    # daily overnight: crude's move AFTER Nifty's close shows up in the gap?
    # r1 at MCX close T covers 23:30(T-1)->23:30(T); Nifty gap(T+1) covers 15:30(T)->09:15(T+1)
    g = pd.DataFrame({"r1": a.r1, "gap": out["gap"]["Nifty 50"],
                      "yr": c.index.year}).dropna()
    print("\nDaily: crude day move vs NEXT Nifty gap (reaction, not tradable), Spearman by year")
    print(g.groupby("yr").apply(lambda d: round(spear(d.r1, d.gap), 3)).to_string())


if __name__ == "__main__":
    main()
