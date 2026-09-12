"""Stage 3: one Reality Check over the WHOLE commodity x sector forward grid.

Stage 1 ran 5,024 forward tests and 9 cleared BH-FDR. FDR controls the false
DISCOVERY rate under assumptions that a grid this correlated strains: the nine
commodities share a global factor (gold-silver weekly correlation 0.80,
copper-zinc 0.65) and sixteen sectors share the market.

So ask the question a trader actually cares about: is the BEST reading in the
grid bigger than the best reading the SAME search finds in data where the
commodity cannot possibly predict anything? The commodity series are
circular-shifted (each commodity by its own random offset, so its streaks,
volatility clustering and decile structure survive intact -- only its alignment
to the Indian market is destroyed), the entire grid is rebuilt, and its maximum
statistic recorded. White's Reality Check, in its simplest form.

If the real maximum sits inside that null distribution, the survivors are what
searching 5,000 combinations of correlated series produces for free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics.commodity_index import (align_to_sessions,  # noqa: E402
                                           commodity_features, index_outcomes,
                                           load_commodity, load_index_ohlc)
from commodity_sector_matrix import COMMODITIES, MIN_MEDIAN_CR, SECTORS  # noqa: E402

YEARS = 6
N_SHIFT = 500
HORIZ = {"gap": 1, "oc1": 1, "cc1": 1, "fwd5": 5, "fwd20": 20}


def build():
    o, c = load_index_ohlc(SECTORS)
    start = c.index.max() - pd.Timedelta(days=int(YEARS * 365.25))
    win = np.asarray(c.index >= start)
    outc = index_outcomes(o[SECTORS], c[SECTORS], (5, 20))

    ys, names, hs = [], [], []
    for kn, h in HORIZ.items():
        for ix in SECTORS:
            raw = outc[kn][ix].values
            ys.append(raw)
            names.append((ix, kn, "raw"))
            hs.append(h)
            if ix != "Nifty 50":                    # sector minus the market
                ys.append(raw - outc[kn]["Nifty 50"].values)
                names.append((ix, kn, "sector-only"))
                hs.append(h)
    Y = np.column_stack(ys)

    masks, mnames = [], []
    for cm in COMMODITIES:
        p = load_commodity(cm, min_median_turnover_cr=MIN_MEDIAN_CR)
        if len(p) < 250:
            continue
        a = align_to_sessions(commodity_features(p), c.index)
        valid = a.r1.notna().values & win
        for cn, m in (("1d jump >2sd", a.z1 > 2), ("1d drop <-2sd", a.z1 < -2),
                      ("5d top 10%", a.pct5 >= 0.90), ("5d bottom 10%", a.pct5 <= 0.10),
                      ("10d top 10%", a.pct10 >= 0.90), ("10d bottom 10%", a.pct10 <= 0.10),
                      ("up 3+", a.up_streak >= 3), ("down 3+", a.dn_streak >= 3)):
            v = m.fillna(False).to_numpy(bool) & valid
            if v.sum() >= 20:
                masks.append(v)
                mnames.append((cm, cn))
    return np.column_stack(masks), Y, np.array(hs), mnames, names, c.index


def stats(M: np.ndarray, Y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Standardised mean difference for every (mask, outcome) pair.

    Overlapping outcome windows are handled by dividing the sample size by the
    horizon -- 20-session returns measured daily carry ~n/20 independent
    observations, and pretending otherwise is how a 20-day 'edge' gets a t of 4.
    """
    ok = ~np.isnan(Y)                                   # T x P
    Yf = np.where(ok, Y, 0.0)
    n1 = M.T.astype(float) @ ok.astype(float)           # K x P
    s1 = M.T.astype(float) @ Yf
    n0 = (~M).T.astype(float) @ ok.astype(float)
    s0 = (~M).T.astype(float) @ Yf
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m0 = s1 / n1, s0 / n0
        sd = np.sqrt(np.nansum(np.where(ok, (Y - np.nanmean(Y, 0)) ** 2, 0), 0)
                     / np.maximum(ok.sum(0) - 1, 1))     # P
        eff1, eff0 = n1 / H[None, :], n0 / H[None, :]
        se = sd[None, :] * np.sqrt(1 / np.maximum(eff1, 1) + 1 / np.maximum(eff0, 1))
        t = (m1 - m0) / se
    t[(n1 < 20) | (n0 < 50)] = np.nan
    return t


def main() -> None:
    M, Y, H, mnames, ynames, idx = build()
    print(f"grid: {M.shape[1]} commodity conditions x {Y.shape[1]} outcome series "
          f"= {M.shape[1] * Y.shape[1]:,} tests, {M.shape[0]} sessions")
    T = stats(M, Y, H)
    obs = np.nanmax(np.abs(T))
    flat = np.abs(T).ravel()
    order = np.argsort(np.where(np.isnan(flat), -1.0, flat))[::-1]   # NaN last
    print(f"\nbest |stat| in the real grid: {obs:.2f}")
    print("top 8 real readings:")
    for pos in order[:8]:
        i, j = divmod(pos, T.shape[1])
        print(f"  {mnames[i][0]:11s} {mnames[i][1]:15s} -> {ynames[j][0]:18s} "
              f"{ynames[j][1]:5s} {ynames[j][2]:11s}  |stat| {flat[pos]:.2f}")

    rng = np.random.default_rng(23)
    # shift each COMMODITY as a block, so every condition of one commodity keeps
    # its own timing relative to the others
    owner = np.array([mnames[i][0] for i in range(M.shape[1])])
    nulls = np.empty(N_SHIFT)
    beat = np.zeros_like(T)          # per-cell: how often the null matched it
    absT = np.abs(T)
    Tlen = M.shape[0]
    for s in range(N_SHIFT):
        Ms = np.empty_like(M)
        for cm in set(owner):
            k = rng.integers(60, Tlen - 60)
            sel = owner == cm
            Ms[:, sel] = np.roll(M[:, sel], k, axis=0)
        ns = np.abs(stats(Ms, Y, H))
        nulls[s] = np.nanmax(ns)
        beat += (ns >= absT)
    p = float((nulls >= obs).mean())
    print(f"\nnull (500 shifts): median best |stat| {np.median(nulls):.2f}, "
          f"95th pct {np.percentile(nulls, 95):.2f}, max {nulls.max():.2f}")
    print(f"REALITY CHECK p = {p:.3f}  -> "
          + ("the best finding is NOT distinguishable from searching noise"
             if p > 0.10 else "the best finding beats the search itself"))

    # how many real readings clear the null's 95th percentile?
    cut = np.percentile(nulls, 95)
    n_beat = int((np.abs(T) >= cut).sum())
    print(f"{n_beat} of {np.isfinite(T).sum():,} readings clear the null's 95th "
          f"percentile ({cut:.2f})")
    for pos in order[:n_beat][:10]:
        i, j = divmod(pos, T.shape[1])
        print(f"  beats null: {mnames[i][0]} {mnames[i][1]} -> {ynames[j][0]} "
              f"{ynames[j][1]} ({ynames[j][2]})")

    # per-pair verdicts for the candidates the earlier stages flagged
    print()
    print("CANDIDATES, judged two ways:")
    print(f"{'candidate':58s} {'|stat|':>7s} {'own p':>7s} {'vs search-aware cut':>21s}")
    want = [("NATURALGAS", "10d top 10%", "Nifty Energy", "oc1", "sector-only"),
            ("NATURALGAS", "10d top 10%", "Nifty Energy", "cc1", "sector-only"),
            ("LEAD", "up 3+", "Nifty PSE", "fwd20", "sector-only"),
            ("LEAD", "up 3+", "Nifty PSE", "fwd20", "raw"),
            ("CRUDE OIL", "10d bottom 10%", "Nifty FMCG", "oc1", "sector-only"),
            ("COPPER", "up 3+", "Nifty Metal", "oc1", "sector-only"),
            ("COPPER", "1d drop <-2sd", "Nifty Metal", "gap", "sector-only"),
            ("SILVER", "1d jump >2sd", "Nifty Metal", "gap", "sector-only")]
    for cm, cn, ix, kn, kind in want:
        try:
            i = mnames.index((cm, cn)); j = ynames.index((ix, kn, kind))
        except ValueError:
            continue
        own = beat[i, j] / N_SHIFT
        verdict = "BEATS the search" if absT[i, j] >= cut else "inside the search noise"
        print(f"{cm + ' ' + cn + ' -> ' + ix + ' ' + kn + ' (' + kind + ')':58s} "
              f"{absT[i, j]:7.2f} {own:7.3f} {verdict:>21s}")


if __name__ == "__main__":
    main()
