"""
Walk-forward validation: legacy TOP-DOWN sector delivery signal (1-day,
turnover-weighted delivery VALUE, mean/std z) vs the BOTTOM-UP ROBUST signal
shipped in src/analytics/sector_signal_v2.py (median/MAD, %-based, multi-day,
equal-weight/trimmed aggregation).

Two questions, across the full available history:
  (A) COHERENCE  — does the sector delivery signal agree with the fraction of its
                   OWN constituents that are accumulating?  (cross-sectional
                   Spearman of factor vs stock breadth, averaged over dates)
  (B) PREDICTION — does the factor at t predict forward sector excess return over
                   the next H trading days?  (cross-sectional Spearman = IC)

The RS/momentum factor is identical in both engines, so this isolates the
DELIVERY factor — the only thing the redesign changed.

Result (371-day DB, 2024-12 → 2026-07, 48 non-overlapping signal dates):
  (A) coherence rho:  legacy −0.055   robust +0.493
      contradictions:  legacy 26.2%    robust 10.1%
  (B) forward IC (10d / 20d):
      old_dv_z    +0.030 / +0.039     new_robz       +0.025 / +0.073
      old_dv5d    +0.063 / +0.029     new_ratio5     +0.044 / +0.036
                                      breadth_accum  +0.047 / +0.058
  => large coherence gain, no loss of (already weak) forward IC. Delivery is not
     the alpha (RS is); the win is a verdict that agrees with its own stocks.

Run:  python -m scripts.backtest_sector_signal_v2
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.repository import query_dataframe

MINTO   = 100.0        # >=1 Cr turnover to be a liquid constituent
H_LIST  = [10, 20]     # forward horizons (trading days)
STEP    = 5            # step between signal dates
WARM    = 110          # min history before first signal date


def _slope_vec(mat: pd.DataFrame) -> np.ndarray:
    n = mat.shape[0]
    x = np.arange(n, dtype=float)
    xd = x - x.mean()
    denom = (xd ** 2).sum()
    y = mat.values.astype(float)
    ym = np.nanmean(y, axis=0)
    return np.nansum((y - ym) * xd[:, None], axis=0) / denom


def main() -> None:
    P = query_dataframe("""
        SELECT b.symbol, s.sector, b.trade_date, b.deliv_per, b.turnover_lacs, b.close_price
        FROM daily_data b JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ','SM','ST') AND s.sector NOT IN ('ETF','Others')
        ORDER BY b.trade_date
    """)
    P["trade_date"] = pd.to_datetime(P["trade_date"]).dt.date
    sym_sector = P.drop_duplicates("symbol").set_index("symbol")["sector"]

    dp   = P.pivot_table("deliv_per",     "trade_date", "symbol").sort_index()
    to   = P.pivot_table("turnover_lacs", "trade_date", "symbol").sort_index()
    px   = P.pivot_table("close_price",   "trade_date", "symbol").sort_index()
    dval = (to * dp / 100.0 / 100.0)
    idx, sym = list(dp.index), dp.columns
    sec_of = sym_sector.reindex(sym).values
    dval_sec = dval.T.groupby(sec_of).sum().T
    print(f"panel rows={len(P):,} symbols={len(sym)} dates={len(idx)} ({idx[0]} -> {idx[-1]})")

    def sectors_frame(i):
        today, hist = dp.iloc[i], dp.iloc[i-100:i]
        liquid = (to.iloc[i] >= MINTO) & today.notna() & (hist.notna().sum() >= 40)
        if liquid.sum() < 50:
            return None
        med = hist.median(); mad = (hist - med).abs().median()
        robz = ((today - med) / (1.4826 * mad.replace(0, np.nan))).clip(-4, 4)
        ratio5 = dp.iloc[i-4:i+1].mean() / hist.mean().replace(0, np.nan)
        sl = pd.Series(_slope_vec(dp.iloc[i-14:i+1]), index=sym) / med.replace(0, np.nan)
        accum = ((ratio5 > 1.05) & (sl > 0)).astype(float)
        d = pd.DataFrame({"sector": sec_of, "robz": robz.values, "ratio5": ratio5.values,
                          "accum": accum.values, "liq": liquid.values}, index=sym)
        d = d[d["liq"]].dropna(subset=["robz", "ratio5"])
        g = d.groupby("sector")
        out = pd.DataFrame({
            "n": g.size(),
            "new_robz": g["robz"].apply(lambda s: (s.sort_values().iloc[int(len(s)*0.1):len(s)-int(len(s)*0.1)].mean() if len(s) > 5 else s.mean())),
            "new_ratio5": g["ratio5"].median(),
            "breadth_accum": g["accum"].mean(),
        })
        out = out[out["n"] >= 5]
        hval = dval_sec.iloc[i-100:i]
        out["old_dv_z"] = ((dval_sec.iloc[i] - hval.mean()) / hval.std(ddof=1).replace(0, np.nan)).reindex(out.index)
        out["old_dv5d"] = (dval_sec.iloc[i-4:i+1].mean() / hval.mean().replace(0, np.nan)).reindex(out.index)
        return out

    def fwd_excess(i, H):
        if i + H >= len(idx):
            return None
        r = px.iloc[i+H] / px.iloc[i] - 1
        d = pd.DataFrame({"sector": sec_of, "r": r.values}, index=sym).dropna()
        sec_r = d.groupby("sector")["r"].median()
        return sec_r - sec_r.median()

    sig_i = list(range(WARM, len(idx) - max(H_LIST) - 1, STEP))
    coh_old, coh_new = [], []
    ic = {f"{f}_{H}": [] for H in H_LIST
          for f in ["old_dv_z", "old_dv5d", "new_robz", "new_ratio5", "breadth_accum"]}
    contra_old = contra_new = contra_tot = 0

    for i in sig_i:
        S = sectors_frame(i)
        if S is None or len(S) < 8:
            continue
        b = S["breadth_accum"]
        if b.nunique() > 2:
            coh_old.append(spearmanr(S["old_dv_z"], b, nan_policy="omit").correlation)
            coh_new.append(spearmanr(S["new_robz"], b, nan_policy="omit").correlation)
        for fac, is_new in [("old_dv_z", False), ("new_robz", True)]:
            fr, br = S[fac].rank(pct=True), b.rank(pct=True)
            c = int(((fr < 0.34) & (br > 0.66)).sum() + ((fr > 0.66) & (br < 0.34)).sum())
            if is_new: contra_new += c
            else:      contra_old += c
        contra_tot += len(S)
        for H in H_LIST:
            fe = fwd_excess(i, H)
            if fe is None:
                continue
            fe = fe.reindex(S.index)
            for f in ["old_dv_z", "old_dv5d", "new_robz", "new_ratio5", "breadth_accum"]:
                v = S[f]; m = v.notna() & fe.notna()
                if m.sum() >= 6 and v[m].nunique() > 2:
                    ic[f"{f}_{H}"].append(spearmanr(v[m], fe[m]).correlation)

    mean = lambda a: float(np.nanmean(a)) if len(a) else float("nan")
    print(f"\nsignal dates used: {len(coh_new)}")
    print("\n(A) COHERENCE  corr(delivery factor, own-stock breadth):")
    print(f"    legacy 1-day ₹value z : {mean(coh_old):+.3f}")
    print(f"    robust bottom-up  z   : {mean(coh_new):+.3f}")
    print(f"\n    CONTRADICTIONS (factor rank opposite to breadth rank):")
    print(f"    legacy: {contra_old}/{contra_tot} = {contra_old/contra_tot*100:.1f}%")
    print(f"    robust: {contra_new}/{contra_tot} = {contra_new/contra_tot*100:.1f}%")
    print("\n(B) FORWARD IC (cross-sectional Spearman vs peer-excess return):")
    print(f"    {'factor':16s}  {'H=10d':>8s}  {'H=20d':>8s}")
    for f in ["old_dv_z", "old_dv5d", "new_robz", "new_ratio5", "breadth_accum"]:
        print(f"    {f:16s}  {mean(ic[f'{f}_10']):+8.3f}  {mean(ic[f'{f}_20']):+8.3f}")


if __name__ == "__main__":
    main()
