"""
VALIDATE the forward-tilt regime behaviour on DCM's OWN data, now that real bears
(2020 COVID, 2022) are loaded. Before, the downtrend-inversion + suppression was
validated only on Tradebot price data (211 F&O). This tests it on DCM sector data
2019-10 -> 2026, delivery included. Vectorized, causal.

Sector momentum (turnover-weighted sector return -> 10d rel-strength vs Nifty) ranked
cross-sectionally per day; Nifty regime = EMA20/50 stack; measure OW(top-q) - UW(bot-q)
forward-10d relative return per regime + per year. Also: does DELIVERY accumulation in
BEAR regimes predict recovery (the untested 'smart money buys the crash' thesis)?
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

def sector_panel():
    p = q("""SELECT s.sector, b.trade_date,
               SUM(b.turnover_lacs*(b.close_price-b.prev_close)/NULLIF(b.prev_close,0)*100)
                 /NULLIF(SUM(CASE WHEN b.prev_close>0 THEN b.turnover_lacs END),0) AS wret,
               SUM(b.turnover_lacs*b.deliv_per/100.0) AS dv
             FROM daily_data b JOIN v_sector_master s ON b.symbol=s.symbol
             WHERE b.series='EQ' AND s.sector NOT IN ('ETF','Others')
               AND b.turnover_lacs >= 500 AND b.prev_close >= 5
             GROUP BY s.sector, b.trade_date ORDER BY b.trade_date""")
    p["trade_date"] = pd.to_datetime(p["trade_date"]); return p

def nifty():
    n = q("SELECT trade_date, close_val, pct_chg FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"] = pd.to_datetime(n["trade_date"]); return n

def comp(s, nrow):
    lr = np.log1p(s/100.0); cr = lr.cumsum(); return np.expm1(cr - cr.shift(nrow))*100

if __name__ == "__main__":
    P = sector_panel(); N = nifty()
    ret = P.pivot_table("wret","trade_date","sector").sort_index().clip(-15, 15)  # kill CA spikes
    dv  = P.pivot_table("dv","trade_date","sector").sort_index()
    cr = np.log1p(ret/100).cumsum()
    mom = np.expm1(cr - cr.shift(10))*100
    fwd = np.expm1(cr.shift(-10) - cr)*100
    dvz = (dv - dv.rolling(100).mean())/dv.rolling(100).std()      # delivery accumulation z
    nc = N.set_index("trade_date")["close_val"].astype(float).reindex(ret.index)
    # RELATIVE to the MEDIAN SECTOR (cancels the broad-universe common drift; true rotation)
    rs = mom.sub(mom.median(axis=1), axis=0)
    relfwd = fwd.sub(fwd.median(axis=1), axis=0)
    nfwd = np.expm1(np.log1p(N.set_index("trade_date")["pct_chg"].astype(float).reindex(ret.index)/100).cumsum().shift(-10)
                    - np.log1p(N.set_index("trade_date")["pct_chg"].astype(float).reindex(ret.index)/100).cumsum())*100
    rank = rs.rank(axis=1, pct=True)

    # Nifty regime (EMA stack)
    e20 = nc.ewm(span=20,adjust=False).mean(); e50 = nc.ewm(span=50,adjust=False).mean()
    reg = pd.Series("CHOP", index=nc.index)
    reg[(nc>e20)&(e20>e50)] = "UP"; reg[(nc<e20)&(e20<e50)] = "DOWN"

    # long form
    L = pd.DataFrame({"rank":rank.stack(), "relfwd":relfwd.stack(), "dvz":dvz.stack()}).dropna()
    L = L.join(reg.rename("reg"), on="trade_date").join(nfwd.rename("nfwd"), on="trade_date")
    L["yr"] = L.index.get_level_values("trade_date").year

    print(f"DCM sector panel {ret.index.min().date()} -> {ret.index.max().date()} | {len(L):,} sector-days\n")
    print("="*90)
    print("1) OW - UW forward-10d RELATIVE, BY NIFTY REGIME (does the tilt invert in real bears?)")
    print("="*90)
    print(f"  {'regime':8s} {'OW':>8s} {'UW':>8s} {'OW-UW':>8s} {'days':>6s}")
    for r in ["UP","CHOP","DOWN"]:
        s = L[L.reg==r]; ow=s[s["rank"]>=0.75]["relfwd"].mean(); uw=s[s["rank"]<=0.25]["relfwd"].mean()
        print(f"  {r:8s} {ow:+7.2f}% {uw:+7.2f}% {ow-uw:+7.2f}% {s.index.get_level_values('trade_date').nunique():6d}")

    print("\n" + "="*90)
    print("2) OW-UW by YEAR (bear years 2020/2022 vs bull) — regime-independence check")
    print("="*90)
    for y in sorted(L.yr.unique()):
        s=L[L.yr==y]; ow=s[s["rank"]>=0.75]["relfwd"].mean(); uw=s[s["rank"]<=0.25]["relfwd"].mean()
        print(f"  {y}: OW-UW {ow-uw:+6.2f}%  (OW {ow:+.2f} UW {uw:+.2f})  n{len(s)}")

    print("\n" + "="*90)
    print("3) SMART-MONEY-IN-CRASH: in DOWN regime, does delivery ACCUMULATION (top dvz) predict")
    print("   recovery? fwd10 rel-to-Nifty of high-accum vs low-accum sectors, DOWN only.")
    print("="*90)
    dn = L[L.reg=="DOWN"]
    for lab,m in [("top delivery-accum (dvz>=+1)", dn.dvz>=1),("low delivery (dvz<=-1)", dn.dvz<=-1),
                  ("DOWN all", dn.dvz.notna())]:
        s=dn[m]; print(f"  {lab:32s} fwd10-rel {s['relfwd'].mean():+.2f}%  hit {(s['relfwd']>0).mean()*100:.0f}%  n{len(s)}")
