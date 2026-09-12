"""Stage 2 of the commodity x sector matrix: kill the forward survivors.

Stage 1 flagged 9 of 5,024 forward tests at FDR q<0.10. Most are the opening
GAP (a reaction you cannot trade off an MCX close). The tradable ones need the
trap stage 1 does not control for:

  TIME CLUSTERING. Events cluster in time -- MCX LEAD's up-streaks sit in
  2020-22 -- and stage 1 compares them with the mean of the WHOLE six years. In
  a period when everything rose, any event set from that period looks
  predictive. Here every survivor is re-tested with CALENDAR-YEAR fixed
  effects, i.e. against other days of the same year, and market-neutral
  (sector minus Nifty 50) at the same time.

  LIQUIDITY. LEAD's median session is Rs 98 Cr in 2022 and Rs 18 Cr by 2026.

Also fixes the direction test: "does the sector lead the commodity, or the
commodity lead the sector?" The MCX close dated T lands at 23:30, AFTER the
15:30 equity close of T, so the honest question is the NEXT full MCX session.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics.commodity_index import (align_to_sessions,  # noqa: E402
                                           commodity_features, episode_starts,
                                           index_outcomes, load_commodity,
                                           load_index_ohlc)
from crude_vs_index_study import ols_hac, spear  # noqa: E402

SECTORS = ["Nifty 50", "Nifty Metal", "Nifty Energy", "Nifty Commodities",
           "Nifty FMCG", "Nifty PSE", "Nifty Oil & Gas"]
YEARS = 6

SURVIVORS = [  # commodity, condition key, index, outcome
    ("LEAD", "up3", "Nifty PSE", "fwd20"),
    ("LEAD", "up3", "Nifty Energy", "fwd20"),
    ("LEAD", "up3", "Nifty Commodities", "cc1"),
    ("NATURALGAS", "pct10_hi", "Nifty Energy", "cc1"),
    ("NATURALGAS", "pct10_hi", "Nifty Energy", "oc1"),
    ("CRUDE OIL", "pct10_lo", "Nifty FMCG", "cc1"),
    ("CRUDE OIL", "pct10_lo", "Nifty FMCG", "oc1"),
    ("CRUDE OIL", "pct10_lo", "Nifty FMCG", "fwd5"),
    ("COPPER", "up3", "Nifty Metal", "oc1"),
    ("COPPER", "up3", "Nifty Metal", "cc1"),
]


def cond(a: pd.DataFrame, key: str) -> np.ndarray:
    m = {"up3": a.up_streak >= 3, "dn3": a.dn_streak >= 3,
         "pct10_hi": a.pct10 >= 0.90, "pct10_lo": a.pct10 <= 0.10,
         "z2": a.z1 > 2}[key]
    return m.fillna(False).to_numpy(bool)


def main() -> None:
    o, c = load_index_ohlc(SECTORS)
    start = c.index.max() - pd.Timedelta(days=int(YEARS * 365.25))
    outc = index_outcomes(o[SECTORS], c[SECTORS], (5, 20))
    yr = pd.get_dummies(c.index.year).values.astype(float)[:, 1:]   # year effects

    print(f"{'commodity':11s} {'cond':9s} {'index':18s} {'out':6s} "
          f"{'days':>5s} {'eps':>4s} {'excess':>8s} {'t':>6s} | "
          f"{'+year':>8s} {'t':>6s} | {'sector-only+year':>16s} {'t':>6s}")
    for cm, ck, ix, kn in SURVIVORS:
        p = load_commodity(cm)
        f = commodity_features(p)
        a = align_to_sessions(f, c.index)
        valid = a.r1.notna().values & np.asarray(c.index >= start)
        m = cond(a, ck) & valid
        y = outc[kn][ix].values
        ok = valid & ~np.isnan(y)
        eps = int(episode_starts(pd.Series(m, index=c.index), cooldown=5).sum())
        h = int(kn[3:]) if kn.startswith("fwd") else 1
        lag = max(h, 5)
        X0 = np.column_stack([np.ones(ok.sum()), m[ok]])
        b0, t0 = ols_hac(y[ok], X0, lag)
        X1 = np.column_stack([np.ones(ok.sum()), m[ok], yr[ok]])
        b1, t1 = ols_hac(y[ok], X1, lag)
        yn = y - outc[kn]["Nifty 50"].values
        okn = valid & ~np.isnan(yn)
        X2 = np.column_stack([np.ones(okn.sum()), m[okn], yr[okn]])
        b2, t2 = ols_hac(yn[okn], X2, lag)
        print(f"{cm:11s} {ck:9s} {ix:18s} {kn:6s} {int(m[ok].sum()):5d} {eps:4d} "
              f"{b0[1]*100:+8.3f} {t0[1]:+6.2f} | {b1[1]*100:+8.3f} {t1[1]:+6.2f} | "
              f"{b2[1]*100:+16.3f} {t2[1]:+6.2f}")

    # where do the events sit in time?
    print("\nevent years (episode starts):")
    for cm, ck in [("LEAD", "up3"), ("NATURALGAS", "pct10_hi"),
                   ("CRUDE OIL", "pct10_lo"), ("COPPER", "up3")]:
        p = load_commodity(cm)
        a = align_to_sessions(commodity_features(p), c.index)
        valid = a.r1.notna().values & np.asarray(c.index >= start)
        st = episode_starts(pd.Series(cond(a, ck) & valid, index=c.index), cooldown=5)
        yrs = pd.Series(c.index[st.values]).dt.year.value_counts().sort_index()
        print(f"  {cm:11s} {ck:9s} " + ", ".join(f"{y}:{n}" for y, n in yrs.items()))

    # ---- direction, done properly ----------------------------------------
    print("\n=== direction: who leads whom? (6y, daily) ===")
    print("MCX close T is 23:30, the equity close T is 15:30, so the commodity")
    print("session that is strictly AFTER the equity close is the NEXT one.")
    for cm in ("COPPER", "CRUDE OIL", "GOLD", "SILVER"):
        p = load_commodity(cm)
        r = p["lvl"] / p["lvl"].shift(1) - 1
        a = align_to_sessions(pd.DataFrame({"r1": r}), c.index)   # strictly before S
        cmd_before = a.r1                       # known before session S opens
        for ix in ("Nifty Metal", "Nifty Energy", "Nifty 50"):
            s = (c[ix] / c[ix].shift(1) - 1)
            g = pd.DataFrame({"before": cmd_before, "sector": s}).dropna()
            g = g[g.index >= start]
            # commodity's NEXT session return, strictly after the equity close
            nxt = g.before.shift(-1)
            lead_cmd = spear(g.before, g.sector)        # commodity -> sector today
            lead_sec = spear(g.sector, nxt.fillna(0))   # sector today -> commodity next
            print(f"  {cm:10s} vs {ix:14s}  commodity->sector same session {lead_cmd:+.3f} | "
                  f"sector->commodity next session {lead_sec:+.3f}  (n {len(g)})")


if __name__ == "__main__":
    main()
