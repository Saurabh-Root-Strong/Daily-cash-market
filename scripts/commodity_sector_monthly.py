"""Monthly: can this month's commodity moves rank NEXT month's sectors?

The user wants a table for the current month and the next: "commodity X is
rising -> which sector performs". That is a CROSS-SECTIONAL forecast, and it is
tested here the only way that means anything:

    1. estimate each sector's sensitivity to each commodity using ONLY data
       before the month being scored (expanding window, re-estimated monthly);
    2. score every sector from the commodity moves known at that point:
           score_s = sum_c beta_cs(past only) * commodity move_c
    3. rank the sectors by score and compare with what they actually did over
       the NEXT month (and the month after), as excess over Nifty 50;
    4. measure rank IC per month, its t over months, and the spread between the
       top-3 and bottom-3 baskets;
    5. benchmark against the thing that is free: the sector's OWN 12-1 momentum.

Single-pair tests run alongside: commodity 1-month move -> sector next-month
excess, on NON-OVERLAPPING months, with a Reality Check over the whole grid.

Power is stated up front, because with ~98 months a null result could just mean
a small sample, and the reader deserves to know which it is.

Run: python scripts/commodity_sector_monthly.py [--years 8.2] [--shifts 400]
"""
from __future__ import annotations

import argparse
import sys
from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics.commodity_index import (MATRIX_MIN_MEDIAN_CR,  # noqa: E402
                                           SECTOR_MATRIX, align_to_sessions,
                                           commodity_features, load_commodity,
                                           load_index_ohlc)
from crude_vs_index_study import bh, ols_hac, spear  # noqa: E402

COMMODITIES = ["CRUDE OIL", "NATURALGAS", "GOLD", "SILVER", "COPPER", "ZINC",
               "ALUMINIUM", "LEAD", "NICKEL"]
MONTH = 21          # sessions
MIN_TRAIN = 36      # months of history before the first walk-forward score


def build(as_of_years: float):
    o, c = load_index_ohlc(SECTOR_MATRIX)
    start = c.index.max() - pd.Timedelta(days=int(as_of_years * 365.25))
    days = c.index[c.index >= start]
    sectors = [s for s in SECTOR_MATRIX if s in c.columns
               and c[s].reindex(days).notna().sum() > len(days) * 0.9]
    px = c[sectors].reindex(days)
    cmd = {}
    for cm in COMMODITIES:
        p = load_commodity(cm, min_median_turnover_cr=MATRIX_MIN_MEDIAN_CR)
        if len(p) < 400:
            continue
        a = align_to_sessions(commodity_features(p), days)      # known before S
        cmd[cm] = a["lvl"]                                      # chained level
    C = pd.DataFrame(cmd, index=days).ffill(limit=5)
    return px, C, sectors, o.reindex(days)[sectors]


def monthly_frames(px: pd.DataFrame, C: pd.DataFrame, opens: pd.DataFrame):
    """Sampled every MONTH sessions: commodity move behind, sector move ahead."""
    pos = np.arange(len(px))[::-1][::MONTH][::-1]
    pos = pos[(pos >= MONTH) & (pos + 2 * MONTH < len(px))]
    idx = px.index[pos]
    cm_now = (C.iloc[pos].values / C.iloc[pos - MONTH].values - 1)
    # forward excess: entered at the NEXT open, held one month / the month after
    nxt1 = (px.values[pos + MONTH] / opens.values[pos + 1] - 1)
    nxt2 = (px.values[pos + 2 * MONTH] / opens.values[pos + MONTH + 1] - 1)
    own = (px.iloc[pos].values / px.iloc[pos - MONTH].values - 1)
    X = pd.DataFrame(cm_now, index=idx, columns=C.columns)
    Y1 = pd.DataFrame(nxt1, index=idx, columns=px.columns)
    Y2 = pd.DataFrame(nxt2, index=idx, columns=px.columns)
    OWN = pd.DataFrame(own, index=idx, columns=px.columns)
    mkt = Y1["Nifty 50"] if "Nifty 50" in Y1 else Y1.mean(axis=1)
    mkt2 = Y2["Nifty 50"] if "Nifty 50" in Y2 else Y2.mean(axis=1)
    return X, Y1.sub(mkt, axis=0), Y2.sub(mkt2, axis=0), OWN


def walk_forward(X: pd.DataFrame, Y: pd.DataFrame, sectors: list[str]):
    """Score sectors each month from betas fitted on earlier months only."""
    scores = pd.DataFrame(index=X.index, columns=sectors, dtype=float)
    for i in range(MIN_TRAIN, len(X)):
        xs, ys = X.iloc[:i], Y.iloc[:i]
        for s in sectors:
            b = 0.0
            for cm in X.columns:
                g = pd.DataFrame({"x": xs[cm], "y": ys[s]}).dropna()
                if len(g) < 24 or g.x.var() == 0:
                    continue
                beta = g.x.cov(g.y) / g.x.var()
                v = X[cm].iloc[i]
                if pd.notna(v):
                    b += beta * v
            scores.loc[X.index[i], s] = b
    return scores.dropna(how="all")


def rank_ic(scores: pd.DataFrame, Y: pd.DataFrame) -> pd.Series:
    out = {}
    for d in scores.index:
        s, y = scores.loc[d].dropna(), Y.loc[d].dropna()
        common = s.index.intersection(y.index)
        if len(common) >= 6:
            out[d] = spear(s[common], y[common])
    return pd.Series(out).dropna()


def basket(scores: pd.DataFrame, Y: pd.DataFrame, k: int = 3) -> pd.Series:
    out = {}
    for d in scores.index:
        s, y = scores.loc[d].dropna(), Y.loc[d].dropna()
        common = s.index.intersection(y.index)
        if len(common) >= 2 * k:
            r = s[common].sort_values()
            out[d] = y[r.index[-k:]].mean() - y[r.index[:k]].mean()
    return pd.Series(out).dropna()


def tstat(x: pd.Series) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=8.2)
    ap.add_argument("--shifts", type=int, default=400)
    args = ap.parse_args()
    pd.set_option("display.width", 250)

    px, C, sectors, opens = build(args.years)
    X, Y1, Y2, OWN = monthly_frames(px, C, opens)
    print(f"{len(X)} non-overlapping months, {X.index.min():%b %Y} .. "
          f"{X.index.max():%b %Y}; {len(C.columns)} commodities, {len(sectors)} sectors")
    print(f"POWER: with {len(X)} months, a monthly rank IC needs to be about "
          f"{1.96 / np.sqrt(len(X)):.2f} to clear chance. Anything smaller than "
          f"that is undetectable here whether it exists or not.")

    # ── 1. walk-forward cross-sectional ranking ────────────────────────────
    for lab, Y in (("next month", Y1), ("month after next", Y2)):
        sc = walk_forward(X, Y, sectors)
        ic = rank_ic(sc, Y)
        sp = basket(sc, Y)
        own_sc = OWN.reindex(sc.index)[sectors]
        ic_own = rank_ic(own_sc, Y)
        print(f"\n=== walk-forward ranking, {lab} ({len(ic)} scored months) ===")
        print(f"  commodity-score rank IC  mean {ic.mean():+.3f}  t {tstat(ic):+.2f}  "
              f"share of months positive {(ic > 0).mean():.0%}")
        print(f"  top3-minus-bottom3       mean {sp.mean() * 100:+.2f}%/month  "
              f"t {tstat(sp):+.2f}  hit {(sp > 0).mean():.0%}")
        print(f"  the free benchmark (sector's own 1-month momentum): IC "
              f"{ic_own.mean():+.3f}  t {tstat(ic_own):+.2f}")

    # ── 2. single pairs, non-overlapping months ────────────────────────────
    rows = []
    for cm in X.columns:
        for s in sectors:
            for lab, Y in (("next month", Y1), ("month after", Y2)):
                g = pd.DataFrame({"x": X[cm], "y": Y[s]}).dropna()
                if len(g) < 40:
                    continue
                b, t = ols_hac(g.y.values,
                               np.column_stack([np.ones(len(g)), g.x.values]), 1)
                rows.append(dict(commodity=cm, sector=s, horizon=lab, months=len(g),
                                 ic=spear(g.x, g.y), beta=b[1], t=t[1]))
    P = pd.DataFrame(rows)
    P["p"] = [erfc(abs(v) / sqrt(2)) for v in P.t]
    P["q"] = bh(P.p.values)
    print(f"\n=== single pairs: {len(P)} tests, "
          f"{int((P.p < 0.05).sum())} nominal vs {0.05 * len(P):.0f} by chance, "
          f"{int((P.q < 0.10).sum())} survive FDR ===")
    print(P.reindex(P.t.abs().sort_values(ascending=False).index).head(10)
          .round(3).to_string(index=False))

    # ── 3. reality check over the pair grid ────────────────────────────────
    rng = np.random.default_rng(5)
    obs = float(P.t.abs().max())
    null = np.empty(args.shifts)
    Xv = {cm: X[cm].values for cm in X.columns}
    Yv = {(s, h): Y[s].values for h, Y in (("1", Y1), ("2", Y2)) for s in sectors}
    for i in range(args.shifts):
        best = 0.0
        for cm in X.columns:
            xs = np.roll(Xv[cm], rng.integers(3, len(X) - 3))
            for (s, h), yv in Yv.items():
                ok = ~np.isnan(xs) & ~np.isnan(yv)
                if ok.sum() < 40:
                    continue
                b, t = ols_hac(yv[ok], np.column_stack([np.ones(ok.sum()), xs[ok]]), 1)
                best = max(best, abs(t[1]))
        null[i] = best
    print(f"\nREALITY CHECK on the monthly pair grid: best real |t| {obs:.2f}; "
          f"null median {np.median(null):.2f}, 95th {np.percentile(null, 95):.2f}; "
          f"p = {(null >= obs).mean():.3f}")

    # ── 4. what IS knowable: same-month co-move, for scenarios ─────────────
    print("\n=== same-month co-move (NOT a forecast): 1% commodity move -> "
          "sector excess over the market, same month ===")
    same = []
    for cm in X.columns:
        for s in sectors:
            if s == "Nifty 50":
                continue
            # sector excess over the SAME month the commodity moved
            ex = (px[s].pct_change(MONTH) - px["Nifty 50"].pct_change(MONTH))
            g = pd.DataFrame({"x": X[cm], "y": ex.reindex(X.index)}).dropna()
            if len(g) < 40:
                continue
            b, t = ols_hac(g.y.values, np.column_stack([np.ones(len(g)), g.x.values]), 1)
            same.append(dict(commodity=cm, sector=s, months=len(g),
                             ic=spear(g.x, g.y), beta=b[1], t=t[1]))
    S = pd.DataFrame(same)
    S["p"] = [erfc(abs(v) / sqrt(2)) for v in S.t]
    S["q"] = bh(S.p.values)
    print(f"{int((S.q < 0.10).sum())} of {len(S)} survive FDR")
    print(S.reindex(S.t.abs().sort_values(ascending=False).index).head(12)
          .round(3).to_string(index=False))
    out = Path(__file__).resolve().parents[1] / "data" / "crude_study" / "matrix"
    S.to_csv(out / "monthly_same.csv", index=False)
    P.to_csv(out / "monthly_pairs.csv", index=False)


if __name__ == "__main__":
    main()
