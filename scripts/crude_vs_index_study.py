"""Does rising crude oil hurt Nifty? Backtest, 2018-2026.

Question (user, 2026-09-11): when crude rises for several days in a row, or
spikes sharply, how do Nifty 50 / Bank / Midcap and the sector indices do next
day, next week, next month?

Crude = MCX CRUDEOIL front-month futures, roll-safe chained returns, built by
the Commodity_Forex_Market project's own `front_month_panel` (so the numbers
here are the same series that project tests). It is priced in RUPEES, which is
the import bill India actually pays.

THE TIMING TRAP. MCX crude closes at 23:30 IST, NSE at 15:30. "Crude rose
today, what did Nifty do today" mixes eight hours of crude information that
did not exist at Nifty's close. So every signal here is the crude state at the
last MCX close STRICTLY BEFORE the Nifty session, and outcomes are split:

    gap   = open_S / close_{S-1} - 1   reaction; you cannot trade it off the
                                       crude close, it happened overnight
    oc1   = close_S / open_S - 1       the tradable next day
    cc1   = close_S / close_{S-1} - 1  what "next day" means in plain speech
    fwdH  = close_{S+H-1} / open_S - 1 tradable next H sessions, H = 5, 10, 20

plus the CONTEMPORANEOUS read -- Nifty over the same k days crude rose -- which
is what the eye sees on a chart and is not a forecast.

Inference: excess over the unconditional mean of the same sample, Newey-West t
with lags covering both the outcome overlap and the event persistence, a
circular-shift permutation null (keeps both series' autocorrelation, breaks
only the alignment), BH-FDR over the whole grid, split halves, and a cut of
Mar-May 2020 (MCX crude went through zero and printed +37% days).

Run:  python scripts/crude_vs_index_study.py [--perm 500]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def spear(x, y) -> float:
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    ok = x.notna() & y.notna()
    return float(x[ok].rank().corr(y[ok].rank()))


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Newey-West (Bartlett) standard errors. X must include a constant."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    g = X * u[:, None]
    S = g.T @ g
    for L in range(1, min(lags, len(y) - 1) + 1):
        w = 1 - L / (lags + 1)
        G = g[L:].T @ g[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, b / np.sqrt(np.diag(V))

OUT = Path(__file__).resolve().parents[1] / "data" / "crude_study"

INDICES = ["Nifty 50", "Nifty Bank", "NIFTY Midcap 100", "NIFTY Smallcap 100",
           "Nifty Auto", "Nifty Oil & Gas", "Nifty Energy", "Nifty FMCG",
           "Nifty IT", "Nifty Metal", "Nifty Pharma", "Nifty Realty",
           "Nifty Infrastructure", "Nifty PSE"]
HORIZONS = (5, 10, 20)
START = "2018-07-02"          # first crude session after the June-2018 hole


# ----------------------------------------------------------------- data
# One construction, shared with the dashboard panel -- see
# src/analytics/commodity_index.py. The commodity series is the DCM copy of the
# CFM front-month panel (python -m src.cli sync-commodity).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics.commodity_index import (align_to_sessions,  # noqa: E402
                                           commodity_features, load_commodity,
                                           load_index_ohlc, index_outcomes)

COMMODITY = "CRUDE OIL"


def load_crude() -> pd.DataFrame:
    return load_commodity(COMMODITY)


def crude_features(p: pd.DataFrame) -> pd.DataFrame:
    return commodity_features(p)


def load_indices() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_index_ohlc(INDICES)


def outcomes(o: pd.DataFrame, c: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return index_outcomes(o, c, HORIZONS)


def align(feat: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    return align_to_sessions(feat, days)


# ----------------------------------------------------------------- stats
def conditions(a: pd.DataFrame) -> dict[str, pd.Series]:
    c = {}
    for n in (2, 3, 4, 5):
        c[f"crude up {n}+ days in a row"] = a.up_streak >= n
        c[f"crude down {n}+ days in a row"] = a.dn_streak >= n
    for k in (5, 10, 20):
        c[f"crude {k}d move in top 10%"] = a[f"pct{k}"] >= 0.90
        c[f"crude {k}d move in bottom 10%"] = a[f"pct{k}"] <= 0.10
    c["crude 1-day spike > +3%"] = a.r1 > 0.03
    c["crude 1-day drop < -3%"] = a.r1 < -0.03
    c["crude 1-day spike > 2 sigma"] = a.z1 > 2
    c["crude 1-day drop < -2 sigma"] = a.z1 < -2
    c["crude 20d up > +10%"] = a.r20 > 0.10
    c["crude 20d down < -10%"] = a.r20 < -0.10
    return {k: v.fillna(False) for k, v in c.items()}


def _hac(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x.astype(float)])
    b, t = ols_hac(y, X, lags)
    return float(b[1]), float(t[1])


def episodes(mask: pd.Series, gap: int) -> int:
    idx = np.flatnonzero(mask.values)
    if not len(idx):
        return 0
    return int(1 + (np.diff(idx) > gap).sum())


def perm_p(y: np.ndarray, x: np.ndarray, obs: float, n: int,
           rng: np.random.Generator) -> float:
    ok = ~np.isnan(y)
    y, x = y[ok], x[ok]
    if x.sum() < 5 or n == 0:
        return np.nan
    T = len(y)
    shifts = rng.integers(60, T - 60, size=n)
    null = np.empty(n)
    for i, s in enumerate(shifts):
        xs = np.roll(x, s)
        null[i] = y[xs].mean() - y[~xs].mean()
    return float((np.abs(null) >= abs(obs)).mean())


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    ps = p[ok]
    o = np.argsort(ps)
    m = len(ps)
    r = ps[o] * m / (np.arange(m) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    tmp = np.empty(m)
    tmp[o] = np.minimum(r, 1)
    q[ok] = tmp
    return q


# ----------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=300)
    ap.add_argument("--commodity", default="CRUDE OIL")
    args = ap.parse_args()
    global COMMODITY, OUT
    COMMODITY = args.commodity
    OUT = OUT / COMMODITY.replace(" ", "_").lower()
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)

    crude = load_crude()
    feat = crude_features(crude)
    o, c = load_indices()
    a = align(feat, c.index)
    outc = outcomes(o, c)
    conds = conditions(a)
    covid = (c.index >= "2020-03-01") & (c.index <= "2020-05-31")
    valid = a.r1.notna().values & (c.index >= START)

    print(f"crude sessions {len(crude)}  {crude.index.min().date()}..{crude.index.max().date()}")
    print(f"NSE sessions with a crude state {valid.sum()}")

    # ---------- 1. contemporaneous: does Nifty fall WHILE crude rises? ----------
    rows = []
    for k in (5, 10, 20):
        for ix in INDICES:
            y = outc[f"same{k}"][ix].values
            x = a[f"r{k}"].values
            ok = valid & ~np.isnan(y) & ~np.isnan(x)
            # non-overlapping samples only, so every point is independent
            sel = np.flatnonzero(ok)[::k]
            if len(sel) < 20:
                continue
            rho = spear(x[sel], y[sel])
            b, t = _hac(y[sel], x[sel], 1)
            rows.append(dict(window=k, index=ix, n=len(sel), spearman=rho,
                             beta=b, t=t))
    same = pd.DataFrame(rows)
    same.to_csv(OUT / "contemporaneous.csv", index=False)
    print("\n=== CONTEMPORANEOUS: index move over the SAME k days crude moved ===")
    print(same.pivot(index="index", columns="window", values="spearman").round(3).to_string())

    # by year, Nifty 50, 5d non-overlapping
    y5 = outc["same5"]["Nifty 50"]
    yr = []
    for Y in range(2018, 2027):
        m = valid & (c.index.year == Y) & y5.notna().values & a.r5.notna().values
        sel = np.flatnonzero(m)[::5]
        if len(sel) > 8:
            yr.append((Y, len(sel), round(spear(a.r5.values[sel], y5.values[sel]), 3)))
    print("\nNifty 50 vs crude, same-week Spearman by year:", yr)

    # ---------- 2. forward: after crude rises, what does the index do? ----------
    res = []
    keys = ["gap", "oc1", "cc1"] + [f"fwd{h}" for h in HORIZONS]
    for cname, mask in conds.items():
        mk = mask.values & valid
        for ix in INDICES:
            for kname in keys:
                y = outc[kname][ix].values
                ok = mk | valid
                ok &= ~np.isnan(y)
                yy, xx = y[ok], mk[ok]
                n_ev = int(xx.sum())
                if n_ev < 15:
                    continue
                h = int(kname[3:]) if kname.startswith("fwd") else 1
                ev, base = yy[xx].mean(), yy.mean()
                b, t = _hac(yy, xx, max(h, 5))
                pp = perm_p(yy, xx, b, args.perm if ix in ("Nifty 50", "Nifty Bank", "NIFTY Midcap 100") else 0, rng)
                # robustness: without Mar-May 2020, and by half
                okc = ok & ~covid
                b_nc, t_nc = _hac(y[okc], mk[okc], max(h, 5)) if mk[okc].sum() >= 15 else (np.nan, np.nan)
                first = ok & (c.index < "2022-07-01")
                second = ok & (c.index >= "2022-07-01")
                ex1 = y[first & mk].mean() - y[first].mean() if (first & mk).sum() >= 8 else np.nan
                ex2 = y[second & mk].mean() - y[second].mean() if (second & mk).sum() >= 8 else np.nan
                res.append(dict(
                    condition=cname, index=ix, outcome=kname, n=n_ev,
                    episodes=episodes(pd.Series(xx), h), ev_mean=ev * 100,
                    base_mean=base * 100, excess=b * 100,
                    hit=(yy[xx] > 0).mean() * 100, base_hit=(yy > 0).mean() * 100,
                    t=t, perm_p=pp, excess_ex2020=b_nc * 100, t_ex2020=t_nc,
                    ex_2018_22=ex1 * 100, ex_2022_26=ex2 * 100))
    R = pd.DataFrame(res)
    from math import erfc, sqrt
    R["p_hac"] = [erfc(abs(t) / sqrt(2)) for t in R.t]
    R["q_all"] = bh(R.p_hac.values)
    tradable = R.outcome != "gap"
    R.loc[tradable, "q_tradable"] = bh(R.loc[tradable, "p_hac"].values)
    R["same_sign_halves"] = np.sign(R.ex_2018_22) == np.sign(R.ex_2022_26)
    R.to_csv(OUT / "forward_grid.csv", index=False)

    print(f"\n=== FORWARD GRID: {len(R)} tests ({tradable.sum()} tradable) ===")
    print("nominal |t|>1.96:", int((R.t.abs() > 1.96).sum()),
          f" expected by chance ~{0.05 * len(R):.0f}")
    print("survive BH q<0.10 (all):", int((R.q_all < 0.10).sum()),
          "   tradable only:", int((R.q_tradable < 0.10).sum()))
    pd.set_option("display.width", 250)
    cols = ["condition", "index", "outcome", "n", "episodes", "ev_mean", "base_mean",
            "excess", "hit", "base_hit", "t", "perm_p", "t_ex2020", "ex_2018_22",
            "ex_2022_26", "q_all"]
    print("\nNifty 50 -- the user's question:")
    N = R[(R["index"] == "Nifty 50") & R.condition.str.contains("up|spike|top|> \\+")]
    print(N[cols].round(3).to_string(index=False))
    print("\nTop 25 by |t| anywhere:")
    print(R.reindex(R.t.abs().sort_values(ascending=False).index)[cols].head(25).round(3).to_string(index=False))

    # ---------- 3. continuous: crude k-day move vs forward index return ----------
    ic = []
    for k in (1, 5, 10, 20):
        for ix in INDICES:
            for kname in keys:
                y = outc[kname][ix].values
                x = a[f"r{k}"].values if k > 1 else a.r1.values
                ok = valid & ~np.isnan(y) & ~np.isnan(x)
                if ok.sum() < 200:
                    continue
                h = int(kname[3:]) if kname.startswith("fwd") else 1
                # own-momentum control: the index's own k-day move to the prior close
                own = (c[ix].shift(1) / c[ix].shift(1 + k) - 1).values
                ok2 = ok & ~np.isnan(own)
                X = np.column_stack([np.ones(ok2.sum()), x[ok2], own[ok2]])
                bb, tt = ols_hac(y[ok2], X, max(h, k))
                rho = spear(x[ok], y[ok])
                ic.append(dict(crude_k=k, index=ix, outcome=kname, n=int(ok.sum()),
                               spearman=rho, beta_ctrl=bb[1], t_ctrl=tt[1],
                               t_own=tt[2]))
    IC = pd.DataFrame(ic)
    IC.to_csv(OUT / "continuous.csv", index=False)
    print("\n=== CONTINUOUS (Spearman crude k-day move -> index outcome), Nifty 50 ===")
    print(IC[IC["index"] == "Nifty 50"].pivot(index="outcome", columns="crude_k",
          values="spearman").round(3).to_string())
    print("\nt (HAC, controlling own momentum), Nifty 50:")
    print(IC[IC["index"] == "Nifty 50"].pivot(index="outcome", columns="crude_k",
          values="t_ctrl").round(2).to_string())
    print("\n|t_ctrl|>2 anywhere:")
    print(IC[IC.t_ctrl.abs() > 2].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
