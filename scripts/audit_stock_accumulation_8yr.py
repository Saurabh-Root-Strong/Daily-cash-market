"""
STOCK-LEVEL accumulation signal backtest (8yr) — does the smart-money delivery signal, at the
STOCK level (the actual entry the user trades), predict forward outperformance?

The smart-money tab shows stocks by a delivery Z-score x price-direction matrix. This tests
that per-stock signal directly, now that 8yr of per-stock delivery exists. Open question from
the sector head-to-head: SM lost for SECTOR selection but may be sharper for STOCK picks.

Per stock (causal): dv = turnover_lacs*deliv_per/100 (delivery value); dv_z = (dv - 100d mean)/
100d std; price_5d = 5d return. States: Secret Accum (z>=1 & price down), Confirmed Accum
(z>=1 & price up), Distribution (z<=-.5 & up), Active Selling (z<=-.5 & down). Forward = 10d
return RELATIVE to that day's universe median (true stock selection). Metrics: rank-IC of dv_z,
per-state fwd-rel + hit + non-overlap t, year-split. Liquid universe (turnover>=5Cr).
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

def load():
    return q("""SELECT b.symbol, b.trade_date, b.close_price, b.prev_close,
                       b.turnover_lacs, b.deliv_per
                FROM daily_data b
                WHERE b.series='EQ' AND b.turnover_lacs>=500 AND b.prev_close>=5
                  AND b.deliv_per IS NOT NULL
                ORDER BY b.symbol, b.trade_date""")

if __name__=="__main__":
    print("loading 8yr per-stock delivery panel...")
    df=load(); df["trade_date"]=pd.to_datetime(df["trade_date"])
    df=df.sort_values(["symbol","trade_date"])
    g=df.groupby("symbol", sort=False)
    df["ret"]=g["close_price"].pct_change()*100
    df["dv"]=df["turnover_lacs"]*df["deliv_per"]/100.0
    df["dv_mean"]=g["dv"].transform(lambda s: s.rolling(100,min_periods=40).mean())
    df["dv_std"]=g["dv"].transform(lambda s: s.rolling(100,min_periods=40).std())
    df["dv_z"]=(df["dv"]-df["dv_mean"])/df["dv_std"].replace(0,np.nan)
    df["p5"]=g["close_price"].transform(lambda s: (s/s.shift(5)-1)*100)
    df["f10"]=g["close_price"].transform(lambda s: (s.shift(-10)/s-1)*100)
    # forward relative to universe median that day
    df["med10"]=df.groupby("trade_date")["f10"].transform("median")
    df["rel10"]=df["f10"]-df["med10"]
    d=df.dropna(subset=["dv_z","p5","rel10"]).copy()
    print(f"panel: {d['symbol'].nunique()} stocks x {d['trade_date'].nunique()} days | {len(d):,} stock-days "
          f"({d['trade_date'].min().date()}->{d['trade_date'].max().date()})\n")

    # state
    z,p=d["dv_z"],d["p5"]
    d["state"]=np.where((z>=1)&(p<0),"Secret Accum",
               np.where((z>=1)&(p>=0),"Confirmed Accum",
               np.where((z<=-0.5)&(p>=0),"Distribution",
               np.where((z<=-0.5)&(p<0),"Active Selling","Neutral"))))

    print("="*92); print("1) RANK-IC — does delivery-z rank stocks by forward 10d relative return? (Spearman)"); print("="*92)
    ics=[]
    for dt,gg in d.groupby("trade_date"):
        if len(gg)>=20: ics.append(gg["dv_z"].corr(gg["rel10"],method="spearman"))
    ics=pd.Series(ics).dropna()
    print(f"  mean IC {ics.mean():+.4f}  t {ics.mean()/(ics.std()/np.sqrt(len(ics))):+.1f}  (n days {len(ics)})")
    print(f"  (IC>0 significant => accumulation ranks stocks by future return; ~0 => no stock-selection skill)")

    print("\n"+"="*92); print("2) PER-STATE forward 10d RELATIVE return + hit + non-overlap t"); print("="*92)
    print(f"  {'state':18s} {'fwd10-rel':>10s} {'hit':>5s} {'t(non-ov)':>10s} {'n':>8s}")
    for s in ["Secret Accum","Confirmed Accum","Neutral","Distribution","Active Selling"]:
        gg=d[d["state"]==s]
        daily=gg.groupby("trade_date")["rel10"].mean()
        sub=daily.iloc[::10].dropna()
        t=sub.mean()/(sub.std()/np.sqrt(len(sub))) if len(sub)>=4 else np.nan
        print(f"  {s:18s} {gg['rel10'].mean():+9.2f}% {(gg['rel10']>0).mean()*100:4.0f}% {t:+9.1f} {len(gg):8d}")

    print("\n"+"="*92); print("3) TOP delivery-z DECILE vs BOTTOM — long/short stock spread, fwd10-rel"); print("="*92)
    d["zdec"]=d.groupby("trade_date")["dv_z"].transform(lambda x: pd.qcut(x,10,labels=False,duplicates="drop"))
    for lab,dec in [("top decile (z high)",9),("bottom decile (z low)",0)]:
        gg=d[d["zdec"]==dec]; print(f"  {lab:22s} fwd10-rel {gg['rel10'].mean():+.2f}%  hit {(gg['rel10']>0).mean()*100:.0f}%  n{len(gg)}")

    print("\n"+"="*92); print("4) YEAR-SPLIT — Secret+Confirmed Accum fwd10-rel (is stock-accum stable?)"); print("="*92)
    acc=d[d["state"].isin(["Secret Accum","Confirmed Accum"])].copy(); acc["yr"]=acc["trade_date"].dt.year
    for y in sorted(acc["yr"].unique()):
        gg=acc[acc["yr"]==y]
        if len(gg)<200: print(f"  {y}: thin"); continue
        print(f"  {y}: accum fwd10-rel {gg['rel10'].mean():+.2f}%  hit {(gg['rel10']>0).mean()*100:3.0f}%  n{len(gg)}")
