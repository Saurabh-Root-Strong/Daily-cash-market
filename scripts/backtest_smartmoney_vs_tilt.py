"""
HEAD-TO-HEAD: Smart Money (Daily delivery-accumulation signal) vs 1-2wk Forward Tilt
(cross-sectional momentum) — which engine's SECTOR picks make more / lose less?

FAITHFUL: walks the two REAL shipped engines forward over every (subsampled) trading day,
collects their per-sector calls, and joins realized forward sector returns. No proxy.

DATA REALITY (honest): Smart Money needs DELIVERY data, which exists only for the DCM
window 2024-12 -> 2026-07 (373 days, one broadly-BULL regime). So this comparison is
bull-only — it says which engine picks better WHEN THE MARKET IS RISING, not in a bear.

Sector forward return = turnover-weighted sector daily return, compounded fwd 5/10/20d,
measured RELATIVE to the cross-sectional median sector (true rotation) and to Nifty.

Each engine -> a per-sector score + a categorical BUY/AVOID:
  Forward Tilt : score=rank(momentum composite); BUY=OVERWEIGHT, AVOID=UNDERWEIGHT
  Smart Money  : score=accum_score; BUY=Secret/Confirmed/Early Accumulation,
                 AVOID=Distribution/Active Selling/Weakening
Metrics: fwd relative return of BUY & AVOID, long-short spread, hit-rate vs median,
rank-IC of each score, top-1 pick, and agreement analysis. Non-overlapping t-stats.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from datetime import date
from src.data.repository import query_dataframe
from src.analytics.sector_rotation import get_sector_rotation
from src.analytics.sector_forward_tilt import get_forward_tilt

STEP = 2                    # walk every 2nd trading day (speed)
MIN_TO = 1.0               # match the UI default slider
CACHE = r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/00fb1329-476b-4454-94e7-cefc7a5aa9d8/scratchpad/sm_vs_tilt_walk.parquet"

def sector_fwd_panel():
    """turnover-weighted sector daily return -> forward 5/10/20d, rel-to-median + rel-to-Nifty."""
    p = query_dataframe(
        """
        SELECT s.sector, b.trade_date,
               SUM(b.turnover_lacs*(b.close_price-b.prev_close)/NULLIF(b.prev_close,0)*100)
                 / NULLIF(SUM(CASE WHEN b.prev_close>0 THEN b.turnover_lacs END),0) AS wret
        FROM daily_data b JOIN v_sector_master s ON b.symbol=s.symbol
        WHERE b.series IN ('EQ','SM','ST') AND s.sector NOT IN ('ETF','Others')
        GROUP BY s.sector, b.trade_date ORDER BY b.trade_date
        """)
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    r = p.pivot_table("wret", "trade_date", "sector").sort_index()
    lr = np.log1p(r/100.0); cr = lr.cumsum()
    out = {}
    for h in (5,10,20):
        fwd = np.expm1(cr.shift(-h)-cr)*100
        out[h] = fwd
    nif = query_dataframe("SELECT trade_date, pct_chg FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    nif["trade_date"] = pd.to_datetime(nif["trade_date"]); nif=nif.set_index("trade_date")["pct_chg"].astype(float)
    ncr = np.log1p(nif/100.0).cumsum()
    nfwd = {h: np.expm1(ncr.shift(-h)-ncr)*100 for h in (5,10,20)}
    return out, nfwd

_BUY_SM  = ("Secret", "Confirmed", "Early")          # accumulation states (substring match)
_AVOID_SM = ("Distribution", "Active Selling", "Weakening")

def walk():
    dates = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data WHERE deliv_per IS NOT NULL ORDER BY trade_date")
    dts = pd.to_datetime(dates["trade_date"]).tolist()
    dts = dts[100::STEP]                              # need 100d history for SM; subsample
    rows = []
    t0 = time.time()
    for i, d in enumerate(dts):
        dd = d.date()
        try:
            sm = get_sector_rotation(dd, MIN_TO)
            ft, _ = get_forward_tilt(dd, MIN_TO)
        except Exception as e:                        # noqa: BLE001
            continue
        if sm is None or sm.empty or ft is None or ft.empty: continue
        sm = sm[["sector","signal","accum_score"]].copy()
        sm["sm_buy"] = sm["signal"].astype(str).str.contains("|".join(_BUY_SM))
        sm["sm_avoid"] = sm["signal"].astype(str).str.contains("|".join(_AVOID_SM))
        ft = ft[["sector","tilt","rank"]].copy()
        ft["ft_buy"] = ft["tilt"].eq("OVERWEIGHT")
        ft["ft_avoid"] = ft["tilt"].eq("UNDERWEIGHT")
        m = sm.merge(ft, on="sector", how="inner"); m["date"] = d
        rows.append(m)
        if i % 20 == 0: print(f"  {i}/{len(dts)}  {dd}  {time.time()-t0:.0f}s", flush=True)
    W = pd.concat(rows, ignore_index=True)
    W.to_parquet(CACHE)
    return W

if __name__ == "__main__":
    if os.path.exists(CACHE) and "--fresh" not in sys.argv:
        W = pd.read_parquet(CACHE); print(f"loaded cached walk: {len(W):,} rows")
    else:
        print("walking both engines forward (this takes a few minutes)...")
        W = walk()
    fwd, nfwd = sector_fwd_panel()
    # attach forward returns (relative to median sector + relative to Nifty)
    for h in (5,10,20):
        f = fwd[h]; med = f.median(axis=1)
        W[f"rel{h}"] = [ (f.loc[d,s]-med.loc[d]) if (d in f.index and s in f.columns) else np.nan
                         for d,s in zip(W["date"], W["sector"]) ]
        W[f"abs{h}"] = [ f.loc[d,s] if (d in f.index and s in f.columns) else np.nan
                         for d,s in zip(W["date"], W["sector"]) ]
    W["ftrank"] = W.groupby("date")["rank"].rank(pct=True)
    W["smrank"] = W.groupby("date")["accum_score"].rank(pct=True)

    print(f"\npanel: {W['date'].nunique()} dates {W['date'].min().date()}->{W['date'].max().date()} "
          f"| {len(W):,} sector-days | bull-only (delivery data limit)\n")

    def t_no(spread_by_date, h):
        s = spread_by_date.dropna()
        if len(s) < h+4: return np.nan
        sub = s.iloc[::h]; return sub.mean()/(sub.std()/np.sqrt(len(sub))) if len(sub)>=4 else np.nan

    print("="*100)
    print("1) BUY-bucket forward RELATIVE return (vs median sector) — higher = better picks")
    print("="*100)
    print(f"  {'engine / bucket':28s} {'fwd5':>8s} {'fwd10':>8s} {'fwd20':>8s} {'hit%(10d)':>9s} {'n':>7s}")
    for lab, mask in [("Smart Money  BUY", W["sm_buy"]), ("Smart Money  AVOID", W["sm_avoid"]),
                      ("Forward Tilt BUY", W["ft_buy"]), ("Forward Tilt AVOID", W["ft_avoid"])]:
        s = W[mask]
        hit = (s["rel10"]>0).mean()*100
        print(f"  {lab:28s} {s['rel5'].mean():+7.2f}% {s['rel10'].mean():+7.2f}% {s['rel20'].mean():+7.2f}% "
              f"{hit:8.0f}% {len(s):7d}")

    print("\n" + "="*100)
    print("2) LONG-SHORT spread (BUY - AVOID) per engine, fwd-10d, non-overlap t")
    print("="*100)
    for lab, bm, am in [("Smart Money", W["sm_buy"], W["sm_avoid"]),
                        ("Forward Tilt", W["ft_buy"], W["ft_avoid"])]:
        by = W[bm].groupby("date")["rel10"].mean(); ay = W[am].groupby("date")["rel10"].mean()
        sp = (by-ay);
        print(f"  {lab:14s} BUY {W[bm]['rel10'].mean():+.2f}%  AVOID {W[am]['rel10'].mean():+.2f}%  "
              f"spread {sp.mean():+.2f}%  t {t_no(sp,10):+.1f}")

    print("\n" + "="*100)
    print("3) RANK-IC — does a higher score => higher forward return? (Spearman, daily mean)")
    print("="*100)
    for lab, col in [("Smart Money accum_score", "smrank"), ("Forward Tilt momentum rank", "ftrank")]:
        ics = []
        for d,g in W.groupby("date"):
            gg = g.dropna(subset=["rel10", col])
            if len(gg) >= 8: ics.append(gg[col].corr(gg["rel10"], method="spearman"))
        ics = pd.Series(ics).dropna()
        tt = ics.mean()/(ics.std()/np.sqrt(len(ics))) if len(ics)>3 else np.nan
        print(f"  {lab:28s} mean IC {ics.mean():+.3f}  t {tt:+.1f}  (n days {len(ics)})")

    print("\n" + "="*100)
    print("4) TOP-1 pick each engine — fwd-10d relative (the single strongest call)")
    print("="*100)
    for lab, col in [("Smart Money (max accum_score)","accum_score"), ("Forward Tilt (max rank)","rank")]:
        top = W[W[col]==W.groupby("date")[col].transform("max")]
        print(f"  {lab:32s} fwd10 rel {top['rel10'].mean():+.2f}%  hit {(top['rel10']>0).mean()*100:.0f}%  n {len(top)}")

    print("\n" + "="*100)
    print("5) AGREEMENT — when BOTH say BUY vs only one, fwd-10d relative")
    print("="*100)
    both = W["sm_buy"] & W["ft_buy"]; only_sm = W["sm_buy"] & ~W["ft_buy"]; only_ft = W["ft_buy"] & ~W["sm_buy"]
    for lab, mask in [("BOTH buy", both), ("only Smart Money", only_sm), ("only Forward Tilt", only_ft)]:
        s = W[mask]; print(f"  {lab:20s} fwd10 rel {s['rel10'].mean():+.2f}%  hit {(s['rel10']>0).mean()*100:.0f}%  n {len(s)}")
