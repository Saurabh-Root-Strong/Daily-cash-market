"""Commodity x Sector: which Indian sectors move with which commodities?

The user's question: "when copper/nickel/lead move, how does Nifty Metal do;
when gold/silver move, who moves; when crude moves, how do Oil & Gas, Energy
and Nifty 50 do -- so that if a commodity is rising we know which sector will
perform."

Three questions, kept strictly apart, because they have different answers:

  1. SAME WEEK   -- does the sector move WITH the commodity in the same week?
                    (a hedging/exposure fact, not a forecast)
  2. SECTOR-ONLY -- does it still hold after taking Nifty 50 out of the sector?
                    (or is it just "everything fell that week")
  3. NEXT        -- does the commodity's move tell you the sector's NEXT day /
                    week / month? (the only question worth money)

Timing: MCX closes 23:30, NSE 15:30, so a commodity state is attached only to
NSE sessions AFTER its close (src/analytics/commodity_index.py). Forward
outcomes run from the next OPEN, which is the first price a reader could pay.

Liquidity: a commodity-year whose median front-month session is below
MIN_MEDIAN_CR is dropped -- MCX NICKEL's median day is Rs 1 Cr from 2022 and
its "returns" are stale quotes, not prices.

Window: last 6 years by default (--years 6), with a split-half check.

Run:  python scripts/commodity_sector_matrix.py [--years 6] [--perm 400]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics.commodity_index import (align_to_sessions,  # noqa: E402
                                           commodity_features, episode_starts,
                                           index_outcomes, load_commodity,
                                           load_index_ohlc)
from crude_vs_index_study import bh, ols_hac, spear  # noqa: E402

COMMODITIES = ["CRUDE OIL", "NATURALGAS", "GOLD", "SILVER", "COPPER", "ZINC",
               "ALUMINIUM", "LEAD", "NICKEL"]

# Sectors with a real economic link to a commodity, plus controls (Bank, IT,
# Pharma) that should show nothing if the method is honest.
SECTORS = ["Nifty 50", "Nifty Metal", "Nifty Oil & Gas", "Nifty Energy",
           "Nifty Commodities", "Nifty Auto", "Nifty Consumer Durables",
           "Nifty FMCG", "Nifty PSE", "Nifty Infrastructure", "Nifty Realty",
           "Nifty Bank", "Nifty IT", "Nifty Pharma", "NIFTY Midcap 100",
           "NIFTY Smallcap 100"]

MIN_MEDIAN_CR = 50.0        # a commodity-year below this is not a market
HORIZONS = (5, 20)


def weekly_blocks(cmd: dict[str, pd.Series], idx: pd.DataFrame,
                  days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-overlapping 5-session returns on NSE days for every series."""
    c = pd.DataFrame({k: v.reindex(days, method="ffill") for k, v in cmd.items()},
                     index=days)
    cw = c / c.shift(5) - 1
    iw = idx.reindex(days)            # the SAME rows, or the blocks misalign
    iw = iw / iw.shift(5) - 1
    pos = np.arange(len(days))[::-1][::5][::-1]
    return cw.iloc[pos], iw.iloc[pos]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=6.0)
    ap.add_argument("--perm", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(17)
    out = Path(__file__).resolve().parents[1] / "data" / "crude_study" / "matrix"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    o, c = load_index_ohlc(SECTORS)
    start = c.index.max() - pd.Timedelta(days=int(args.years * 365.25))
    days = c.index[c.index >= start]
    print(f"window {days.min():%d %b %Y} .. {days.max():%d %b %Y}  "
          f"({len(days)} NSE sessions)")

    # ---- load commodities, drop illiquid commodity-years -------------------
    lvl, feats, dropped = {}, {}, {}
    for cm in COMMODITIES:
        # the gate is applied INSIDE the loader, before break detection, so a
        # window spanning a dropped year is a gap, not a measured move
        p = load_commodity(cm, min_median_turnover_cr=MIN_MEDIAN_CR)
        if p.empty:
            dropped[cm] = ["no liquid year"]
            continue
        tv = query_turnover(cm)
        bad = sorted(y for y, v in tv.items() if v < MIN_MEDIAN_CR)
        if bad:
            dropped[cm] = bad
        p = p[p.index >= start - pd.Timedelta(days=120)]
        if len(p) < 250:
            dropped.setdefault(cm, []).append("too few sessions left")
            continue
        lvl[cm] = p["lvl"]
        feats[cm] = commodity_features(p)
    for cm, yrs in dropped.items():
        print(f"  dropped {cm}: {yrs} (median session below Rs {MIN_MEDIAN_CR:.0f} Cr)")
    print(f"  testing {len(lvl)} commodities x {len(SECTORS)} indices")

    cw, iw = weekly_blocks(lvl, c[SECTORS], days)
    n50 = iw["Nifty 50"]

    # ---- 1 + 2. same week, raw and sector-only ----------------------------
    rows = []
    for cm in lvl:
        x = cw[cm]
        for ix in SECTORS:
            g = pd.DataFrame({"x": x, "y": iw[ix], "m": n50}).dropna()
            if len(g) < 60:
                continue
            ex = g.y - g.m                        # sector minus the market
            b, t = ols_hac(g.y.values, np.column_stack([np.ones(len(g)), g.x.values]), 1)
            be, te = ols_hac(ex.values, np.column_stack([np.ones(len(g)), g.x.values]), 1)
            half = len(g) // 2
            rows.append(dict(
                commodity=cm, index=ix, weeks=len(g),
                rho=spear(g.x, g.y), beta=b[1], t=t[1],
                rho_ex=spear(g.x, ex), beta_ex=be[1], t_ex=te[1],
                rho_h1=spear(g.x[:half], g.y[:half]),
                rho_h2=spear(g.x[half:], g.y[half:])))
    S = pd.DataFrame(rows)
    from math import erfc, sqrt
    S["p"] = [erfc(abs(v) / sqrt(2)) for v in S.t]
    S["q"] = bh(S.p.values)
    S["p_ex"] = [erfc(abs(v) / sqrt(2)) for v in S.t_ex]
    S["q_ex"] = bh(S.p_ex.values)
    S["stable"] = np.sign(S.rho_h1) == np.sign(S.rho_h2)
    S.to_csv(out / "same_week.csv", index=False)

    print("\n=== 1. SAME WEEK: rank correlation, commodity vs index ===")
    print(S.pivot(index="index", columns="commodity", values="rho").round(2).to_string())
    print(f"\n{int((S.q < 0.10).sum())} of {len(S)} pairs survive FDR q<0.10; "
          f"{int(S.stable.sum())} keep their sign in both halves")

    print("\n=== 2. SECTOR-ONLY: same, after subtracting Nifty 50 ===")
    print(S.pivot(index="index", columns="commodity", values="rho_ex").round(2).to_string())
    print(f"{int((S.q_ex < 0.10).sum())} pairs survive FDR on the sector-only version")
    top = S[S.q_ex < 0.10].reindex(S.rho_ex.abs().sort_values(ascending=False).index).dropna(subset=["rho_ex"])
    print("\nStrongest sector-only links (1% commodity move -> beta% sector-vs-market):")
    print(top[["commodity", "index", "weeks", "rho", "rho_ex", "beta_ex", "t_ex",
               "rho_h1", "rho_h2", "q_ex"]].head(20).round(3).to_string(index=False))

    # ---- 3. forward: does it PREDICT the next day / week / month? ---------
    print("\n=== 3. NEXT: does the commodity's move lead the sector? ===")
    fwd = []
    outc = index_outcomes(o[SECTORS], c[SECTORS], HORIZONS)
    keys = ["gap", "oc1", "cc1"] + [f"fwd{h}" for h in HORIZONS]
    for cm, f in feats.items():
        a = align_to_sessions(f, c.index)
        in_win = np.asarray(c.index >= start)
        valid = a.r1.notna().values & in_win
        conds = {
            "1-day jump > 2 sigma": (a.z1 > 2).fillna(False).to_numpy(bool),
            "1-day drop < -2 sigma": (a.z1 < -2).fillna(False).to_numpy(bool),
            "5d move top 10%": (a.pct5 >= 0.90).fillna(False).to_numpy(bool),
            "5d move bottom 10%": (a.pct5 <= 0.10).fillna(False).to_numpy(bool),
            "10d move top 10%": (a.pct10 >= 0.90).fillna(False).to_numpy(bool),
            "10d move bottom 10%": (a.pct10 <= 0.10).fillna(False).to_numpy(bool),
            "up 3+ sessions": (a.up_streak >= 3).fillna(False).to_numpy(bool),
            "down 3+ sessions": (a.dn_streak >= 3).fillna(False).to_numpy(bool),
        }
        for cn, m0 in conds.items():
            m = m0 & valid
            if m.sum() < 20:
                continue
            eps = int(episode_starts(pd.Series(m, index=c.index), cooldown=5).sum())
            for ix in SECTORS:
                for kn in keys:
                    y = outc[kn][ix].values
                    ok = valid & ~np.isnan(y)
                    if ok.sum() < 200 or (m & ok).sum() < 20:
                        continue
                    h = int(kn[3:]) if kn.startswith("fwd") else 1
                    yy, xx = y[ok], m[ok]
                    b, t = ols_hac(yy, np.column_stack([np.ones(len(yy)), xx]), max(h, 5))
                    # sector-only version: the same test on index minus Nifty 50
                    yn = y - outc[kn]["Nifty 50"].values
                    okn = valid & ~np.isnan(yn)
                    bn, tn = ols_hac(yn[okn], np.column_stack([np.ones(okn.sum()), m[okn]]),
                                     max(h, 5))
                    fwd.append(dict(commodity=cm, condition=cn, index=ix, outcome=kn,
                                    days=int(xx.sum()), episodes=eps,
                                    excess=b[1] * 100, t=t[1],
                                    excess_ex=bn[1] * 100, t_ex=tn[1]))
    F = pd.DataFrame(fwd)
    F["p"] = [erfc(abs(v) / sqrt(2)) for v in F.t]
    F["q"] = bh(F.p.values)
    tr = F.outcome != "gap"
    F.loc[tr, "q_tr"] = bh(F.loc[tr, "p"].values)
    F["p_ex"] = [erfc(abs(v) / sqrt(2)) for v in F.t_ex]
    F["q_ex"] = bh(F.p_ex.values)
    F.to_csv(out / "forward.csv", index=False)
    print(f"{len(F)} forward tests ({int(tr.sum())} tradable). "
          f"nominal |t|>1.96: {int((F.t.abs() > 1.96).sum())} vs "
          f"{0.05 * len(F):.0f} by chance. FDR q<0.10: {int((F.q < 0.10).sum())} "
          f"(tradable {int((F.q_tr < 0.10).sum())}, sector-only {int((F.q_ex < 0.10).sum())})")
    print("\nStrongest forward readings (any):")
    cols = ["commodity", "condition", "index", "outcome", "days", "episodes",
            "excess", "t", "excess_ex", "t_ex", "q", "q_tr"]
    print(F.reindex(F.t.abs().sort_values(ascending=False).index)[cols].head(15)
          .round(3).to_string(index=False))

    # ---- 4. how many independent bets is this really? ---------------------
    print("\n=== 4. how independent are these tests? ===")
    cc = cw.corr().abs()
    print("commodity weekly-return correlations (median off-diagonal "
          f"{np.median(cc.values[~np.eye(len(cc), dtype=bool)]):.2f}):")
    print(cc.round(2).to_string())
    ie = (iw[SECTORS].sub(n50, axis=0)).corr().abs()
    print(f"sector-minus-market correlations, median off-diagonal "
          f"{np.median(ie.values[~np.eye(len(ie), dtype=bool)]):.2f}")

    # ---- 5. does the sector lead the commodity instead? -------------------
    print("\n=== 5. direction: sector day -> commodity's NEXT close ===")
    for cm in ("COPPER", "CRUDE OIL", "GOLD"):
        if cm not in lvl:
            continue
        r = lvl[cm].pct_change()
        nxt = r.reindex(c.index, method="bfill")     # first MCX close after the session
        for ix in ("Nifty Metal", "Nifty Oil & Gas", "Nifty 50"):
            sr = (c[ix] / c[ix].shift(1) - 1)
            g = pd.DataFrame({"s": sr, "c": nxt}).dropna()
            g = g[g.index >= start]
            print(f"  {ix:16s} -> {cm:10s} next MCX close: rho {spear(g.s, g.c):+.3f} "
                  f"(n {len(g)})")


def query_turnover(commodity: str) -> dict[int, float]:
    from src.data.repository import query_dataframe
    df = query_dataframe(
        "SELECT year(trade_date) y, median(turnover_cr) m FROM commodity_daily "
        "WHERE commodity = ? GROUP BY 1", [commodity])
    return {int(r.y): float(r.m) for r in df.itertuples()}


if __name__ == "__main__":
    main()
