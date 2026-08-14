"""
FII / participant POSITIONING vs forward index returns — the 2012-2026 honest study.

Runs on the backfilled fao_participant + fii_derivatives_stats tables
(scripts/backfill_fii_history.py). Everything here is built to survive the three
traps that have killed every previous positioning claim in this repo:

  1. PUBLICATION LAG. Both reports publish ~18:00-19:30 IST on day D. You cannot act
     on them until the D+1 OPEN. So every tradable return is measured from the D+1
     open, never from the D close. Close-to-close IC is also printed, labelled
     NON-TRADABLE, purely to show how much of any "edge" is the lag.
  2. LOT-SIZE / ERA DRIFT. Nifty lot went 50 -> 75 -> 25 -> 50 -> 65 and the Oct-2024
     SEBI notional hike reset everything; participant mix shifted at each rule change.
     So NO raw contract counts are ever compared across time — only within-day ratios
     and causal rolling z-scores of those ratios.
  3. ONE-REGIME SAMPLES. Every stat is reported per era as well as pooled, and a
     feature that flips sign across eras is called dead regardless of pooled t.

Feature set (all as-of D, all causal):
  fii_fut_ratio     FII index-fut long share  = long/(long+short)          [the classic]
  fii_fut_d1/d5     1d / 5d change in that share                          [flow, not level]
  fii_fut_z         252d causal z-score of the share                      [extremes]
  client_fut_ratio  the retail counterparty's share                       [dumb-money mirror]
  pro_fut_ratio     prop-desk inventory
  dii_fut_ratio     domestic institutions
  fii_opt_dir       FII index-option directional posture:
                    (call_long + put_short) / (all four legs)
  fii_share_oi      FII share of total index-fut OI                       [participation]
  fii_net_fut_cr    FII net index-futures buy value, Rs Cr (fii_stats)    [flow in money]
  fii_net_opt_cr    FII net index-options buy value, Rs Cr

Outputs: rank-IC (+ non-overlapping block t), quintile spreads, extreme-z conditional
returns, per-era sign stability, and a Bonferroni threshold for the number of tests run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                          # noqa: BLE001
    pass

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from src.data.repository import query_dataframe             # noqa: E402

HOR = (1, 5, 10, 20)
ERAS = [("2018-19", "2018-01-01", "2019-12-31"),
        ("2020-21 covid+melt", "2020-01-01", "2021-12-31"),
        ("2022 bear", "2022-01-01", "2022-12-31"),
        ("2023-24", "2023-01-01", "2024-12-31"),
        ("2025-26", "2025-01-01", "2026-12-31")]


def load() -> pd.DataFrame:
    fao = query_dataframe("""
        SELECT trade_date, client_type,
               fut_idx_long, fut_idx_short,
               opt_idx_call_long, opt_idx_call_short,
               opt_idx_put_long,  opt_idx_put_short
        FROM fao_participant WHERE upper(data_type) = 'OI'""")
    fao["trade_date"] = pd.to_datetime(fao["trade_date"])

    piv = {}
    for ct, g in fao.groupby("client_type"):
        g = g.set_index("trade_date").sort_index()
        tot = g["fut_idx_long"] + g["fut_idx_short"]
        piv[f"{ct.lower()}_fut_ratio"] = g["fut_idx_long"] / tot.replace(0, np.nan)
        piv[f"{ct.lower()}_fut_oi"] = tot
        if ct == "FII":
            legs = (g["opt_idx_call_long"] + g["opt_idx_call_short"]
                    + g["opt_idx_put_long"] + g["opt_idx_put_short"])
            piv["fii_opt_dir"] = ((g["opt_idx_call_long"] + g["opt_idx_put_short"])
                                  / legs.replace(0, np.nan))
    F = pd.DataFrame(piv).sort_index()
    F["fii_share_oi"] = F["fii_fut_oi"] / F[[c for c in F if c.endswith("_fut_oi")]].sum(axis=1)

    # ---- the VOLUME file: FLOW, not stock. What FII actually traded that session,
    # as opposed to where its book is parked. Different signal, same 4 participants.
    vol = query_dataframe("""
        SELECT trade_date, client_type, fut_idx_long, fut_idx_short,
               opt_idx_call_long, opt_idx_put_long
        FROM fao_participant WHERE upper(data_type) = 'VOL'""")
    if not vol.empty:
        vol["trade_date"] = pd.to_datetime(vol["trade_date"])
        vpiv = {}
        for ct, g in vol.groupby("client_type"):
            g = g.set_index("trade_date").sort_index()
            tot = g["fut_idx_long"] + g["fut_idx_short"]
            vpiv[f"{ct.lower()}_vol_ratio"] = g["fut_idx_long"] / tot.replace(0, np.nan)
            vpiv[f"{ct.lower()}_vol_tot"] = tot
        V = pd.DataFrame(vpiv).sort_index()
        allvol = V[[c for c in V if c.endswith("_vol_tot")]].sum(axis=1)
        V["fii_vol_share"] = V["fii_vol_tot"] / allvol.replace(0, np.nan)
        # turnover intensity: is FII churning its book or sitting on it?
        F = F.join(V[["fii_vol_ratio", "client_vol_ratio", "pro_vol_ratio",
                      "fii_vol_share"]], how="left")
        F["fii_churn"] = (V["fii_vol_tot"] / F["fii_fut_oi"].replace(0, np.nan))

    st = query_dataframe("""
        SELECT trade_date, category, buy_value_cr, sell_value_cr
        FROM fii_derivatives_stats
        WHERE category IN ('INDEX FUTURES','INDEX OPTIONS')""")
    if not st.empty:
        st["trade_date"] = pd.to_datetime(st["trade_date"])
        st["net"] = st["buy_value_cr"] - st["sell_value_cr"]
        w = st.pivot_table("net", "trade_date", "category")
        F["fii_net_fut_cr"] = w.get("INDEX FUTURES")
        F["fii_net_opt_cr"] = w.get("INDEX OPTIONS")

    # derived: flow + causal z of the headline ratio
    F["fii_fut_d1"] = F["fii_fut_ratio"].diff()
    F["fii_fut_d5"] = F["fii_fut_ratio"].diff(5)
    m = F["fii_fut_ratio"].rolling(252, min_periods=120).mean()
    s = F["fii_fut_ratio"].rolling(252, min_periods=120).std()
    F["fii_fut_z"] = (F["fii_fut_ratio"] - m) / s

    n = query_dataframe("""
        SELECT trade_date, open_val, close_val FROM index_data
        WHERE index_name = 'Nifty 50' ORDER BY trade_date""")
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    n = n.set_index("trade_date").sort_index().astype(float)

    # ---- TRADABLE returns: entry at the D+1 OPEN (report published evening of D) ----
    op, cl = n["open_val"], n["close_val"]
    for h in HOR:
        # enter next open, exit close of the h-th session after that open
        F[f"trad{h}"] = (cl.shift(-h) / op.shift(-1) - 1).reindex(F.index) * 100
    # ---- NON-TRADABLE reference: close D -> close D+h (what a naive study measures) ----
    for h in HOR:
        F[f"naive{h}"] = (cl.shift(-h) / cl - 1).reindex(F.index) * 100
    # overnight gap D -> D+1 open: NOT capturable from a signal published after the close
    F["gap_nontrad"] = (op.shift(-1) / cl - 1).reindex(F.index) * 100
    F["nifty_200"] = cl.rolling(200).mean().reindex(F.index)
    F["above200"] = (cl.reindex(F.index) > F["nifty_200"])
    return F.dropna(subset=["fii_fut_ratio"])


FEATS = [# OI file — where the book is parked (stock)
         "fii_fut_ratio", "fii_fut_d1", "fii_fut_d5", "fii_fut_z",
         "client_fut_ratio", "pro_fut_ratio", "dii_fut_ratio",
         "fii_opt_dir", "fii_share_oi",
         # VOL file — what was actually traded that session (flow)
         "fii_vol_ratio", "client_vol_ratio", "pro_vol_ratio",
         "fii_vol_share", "fii_churn",
         # FII stats xls — the same flow in rupees
         "fii_net_fut_cr", "fii_net_opt_cr"]


def ic(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 60:
        return np.nan
    return float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank()))


def block_t(x: pd.Series, h: int) -> float:
    s = x.dropna().iloc[::max(h, 1)]
    if len(s) < 6 or s.std(ddof=1) == 0:
        return np.nan
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def main() -> None:
    F = load()
    print(f"panel {F.index.min().date()} -> {F.index.max().date()}  {len(F):,} days")
    have = [c for c in FEATS if c in F and F[c].notna().sum() > 200]
    print(f"features with data: {have}\n")

    print("=" * 104)
    print("1) RANK-IC — TRADABLE (enter D+1 open) vs NAIVE (enter D close, NOT executable)")
    print("=" * 104)
    print(f"  {'feature':18s} " + "".join(f"{'trad'+str(h):>9s}" for h in HOR)
          + "   |" + "".join(f"{'naive'+str(h):>9s}" for h in HOR))
    n_tests = 0
    for f in have:
        row = "".join(f"{ic(F[f], F[f'trad{h}']):+9.3f}" for h in HOR)
        row2 = "".join(f"{ic(F[f], F[f'naive{h}']):+9.3f}" for h in HOR)
        n_tests += len(HOR)
        print(f"  {f:18s} {row}   |{row2}")
    print(f"\n  gap D->D+1 open (NOT capturable, shown for reference): "
          f"IC vs fii_fut_ratio {ic(F['fii_fut_ratio'], F['gap_nontrad']):+.3f}")

    print("\n" + "=" * 104)
    print("2) QUINTILE SPREAD on the TRADABLE return (Q5-Q1), non-overlapping block t")
    print("=" * 104)
    for h in (1, 5, 10):
        print(f"\n  horizon {h}d:")
        print(f"    {'feature':18s} {'Q1':>8s} {'Q5':>8s} {'Q5-Q1':>8s} {'t(blk)':>8s} {'n':>7s}")
        for f in have:
            d = F[[f, f"trad{h}"]].dropna()
            if len(d) < 250:
                continue
            q = pd.qcut(d[f].rank(method="first"), 5, labels=False)
            lo, hi = d[q == 0][f"trad{h}"], d[q == 4][f"trad{h}"]
            spread = hi.mean() - lo.mean()
            print(f"    {f:18s} {lo.mean():+7.2f}% {hi.mean():+7.2f}% {spread:+7.2f}% "
                  f"{block_t(hi - hi.mean() + (hi.mean()-lo.mean()), h):+8.1f} {len(d):7d}")

    print("\n" + "=" * 104)
    print("3) EXTREMES — what happens after |z| > 2 on the FII index-fut long share")
    print("=" * 104)
    for lab, m in [("z < -2  (FII max short)", F["fii_fut_z"] < -2),
                   ("z < -1", F["fii_fut_z"] < -1),
                   ("z > +1", F["fii_fut_z"] > 1),
                   ("z > +2  (FII max long)", F["fii_fut_z"] > 2)]:
        sub = F[m]
        if len(sub) < 15:
            print(f"  {lab:26s} n {len(sub):4d}  (too few)"); continue
        print(f"  {lab:26s} n {len(sub):4d}  " +
              "  ".join(f"{h}d {sub[f'trad{h}'].mean():+5.2f}% (hit {(sub[f'trad{h}']>0).mean()*100:3.0f}%)"
                        for h in HOR))
    base = F
    print(f"  {'BASELINE all days':26s} n {len(base):4d}  " +
          "  ".join(f"{h}d {base[f'trad{h}'].mean():+5.2f}% (hit {(base[f'trad{h}']>0).mean()*100:3.0f}%)"
                    for h in HOR))

    print("\n" + "=" * 104)
    print("4) PER-ERA SIGN STABILITY — IC(feature, trad5) by era. A sign flip = dead signal.")
    print("=" * 104)
    print(f"  {'feature':18s} " + "".join(f"{lab[:12]:>14s}" for lab, _, _ in ERAS))
    for f in have:
        cells = []
        for _, a, b in ERAS:
            sub = F.loc[a:b]
            cells.append(f"{ic(sub[f], sub['trad5']):+14.3f}" if len(sub) > 120 else f"{'—':>14s}")
        print(f"  {f:18s} " + "".join(cells))

    print("\n" + "=" * 104)
    print("5) REGIME SPLIT — IC(feature, trad5) above vs below the 200-DMA")
    print("=" * 104)
    for f in have:
        a, b = F[F["above200"]], F[~F["above200"]]
        print(f"  {f:18s} above200 {ic(a[f], a['trad5']):+.3f} (n{len(a)})   "
              f"below200 {ic(b[f], b['trad5']):+.3f} (n{len(b)})")

    print("\n" + "=" * 104)
    print(f"MULTIPLE TESTING: {n_tests} IC tests in section 1 alone. Bonferroni |t| threshold "
          f"≈ {abs(round(2.8 + np.log(max(n_tests,1))/3, 1))}. Treat |IC| < 0.05 as noise.")
    print("Index-futures round trip ≈ 5-8 bps — any 1d edge below that is not tradable.")
    print("=" * 104)


if __name__ == "__main__":
    main()
