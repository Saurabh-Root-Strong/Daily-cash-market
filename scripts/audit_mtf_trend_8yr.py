"""
MULTI-TIMEFRAME TREND ALIGNMENT (Nifty 50, 8yr, 10x sharp — non-overlap t).

For a STOCK trader asking "is the market trend changing?" — read the trend at 4 bands and
test whether ALIGNMENT (how many bands agree) tells you when to enter vs stand aside.

Bands (each = close vs its EMA + EMA slope, causal):
  SWING  EMA10   (1-3 wk)     SHORT  EMA30 (1-2 mo)
  LONG   EMA60   (2.5-4 mo)   VLONG  EMA120 (4-6 mo)
band UP = close>EMA & EMA rising ; DN = close<EMA & EMA falling ; else FLAT.
n_up = count of UP bands (0-4) = alignment score.

Tests: (1) fwd return by n_up bucket at swing/short/long — does full alignment = safe entry?
(2) is all-DOWN a bottom (contrarian) or continuation? (3) TURNS — when a band flips up/down,
what follows (regime-change detection, expected to lag/contrarian). (4) year-split of all-up.
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

def load():
    n=q("SELECT trade_date,close_val FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"]=pd.to_datetime(n["trade_date"]); return n.set_index("trade_date")["close_val"].astype(float)
def fret(c,n): lr=np.log(c); return (np.exp(lr.shift(-n)-lr)-1)*100
def not_t(sig,fwd,h):
    idx=np.where(sig.values)[0]; v=fwd.values; picks=[]; last=-10**9
    for i in idx:
        if i-last>=h and np.isfinite(v[i]): picks.append(v[i]); last=i
    picks=np.array(picks)
    if len(picks)<4: return (np.nan,np.nan,len(picks),np.nan)
    return (picks.mean(), picks.mean()/(picks.std(ddof=1)/np.sqrt(len(picks))), len(picks),(picks>0).mean()*100)
def cell(sig,d,h):
    m,t,ni,hit=not_t(sig.reindex(d.index).fillna(False), d[f"f{h}"], h)
    if not np.isfinite(t): return "  thin  "
    vd="🟢" if abs(t)>=2 else ("·" if abs(t)>=1 else " ")
    return f"{m:+5.1f}%/t{t:+3.1f}/{hit:2.0f}{vd}"

BANDS={"SWING":10,"SHORT":30,"LONG":60,"VLONG":120}
HZ={"SWING":10,"SHORT":30,"LONG":65}

if __name__=="__main__":
    c=load(); d=pd.DataFrame(index=c.index); d["c"]=c
    for h in (10,30,65): d[f"f{h}"]=fret(c,h)
    updn={}
    for b,span in BANDS.items():
        e=c.ewm(span=span,adjust=False).mean(); slope=e-e.shift(max(5,span//4))
        updn[b]=np.where((c>e)&(slope>0),1,np.where((c<e)&(slope<0),-1,0))
    U=pd.DataFrame(updn,index=c.index)
    n_up=(U==1).sum(axis=1); n_dn=(U==-1).sum(axis=1)
    d["n_up"]=n_up; d["n_dn"]=n_dn
    base=pd.Series(False,index=c.index); base.iloc[130:]=True

    print(f"Nifty {c.index.min().date()}->{c.index.max().date()} | bands EMA{list(BANDS.values())}")
    print("cell = mean/t(non-overlap)/hit ; 🟢|t|>=2, ·1<=|t|<2\n")
    print("="*96); print("1) FORWARD by ALIGNMENT (n_up = # bands in uptrend, 0-4) — entry-timing map"); print("="*96)
    print(f"  {'alignment':22s} {'days':>5s} | {'SWING(10d)':>13s} | {'SHORT(30d)':>13s} | {'LONG(65d)':>13s}")
    for k in range(5):
        m=base&(n_up==k)
        lab=f"{k}/4 up" + (" (all UP)" if k==4 else " (all DOWN)" if k==0 else "")
        if m.sum()<30: print(f"  {lab:22s} n{int(m.sum()):4d} | thin"); continue
        print(f"  {lab:22s} n{int(m.sum()):4d} | {cell(m,d,10):>13s} | {cell(m,d,30):>13s} | {cell(m,d,65):>13s}")

    print("\n"+"="*96); print("2) ALL-DOWN = bottom (contrarian) or continuation? + ALL-UP persistence"); print("="*96)
    print(f"  all 4 DOWN (n_up=0)   | {cell(base&(n_up==0),d,10)} | {cell(base&(n_up==0),d,30)} | {cell(base&(n_up==0),d,65)}")
    print(f"  all 4 UP   (n_up=4)   | {cell(base&(n_up==4),d,10)} | {cell(base&(n_up==4),d,30)} | {cell(base&(n_up==4),d,65)}")

    print("\n"+"="*96); print("3) TREND TURNS — a band FLIPS this week (regime-change detection)"); print("="*96)
    for b in BANDS:
        s=pd.Series(U[b],index=c.index)
        turn_up=(s==1)&(s.shift(3)==-1)   # flipped from down to up within 3d
        turn_dn=(s==-1)&(s.shift(3)==1)
        print(f"  {b:6s} turn UP  | {cell(base&turn_up,d,10)} | {cell(base&turn_up,d,30)} | {cell(base&turn_up,d,65)}")
        print(f"  {b:6s} turn DN  | {cell(base&turn_dn,d,10)} | {cell(base&turn_dn,d,30)} | {cell(base&turn_dn,d,65)}")

    print("\n"+"="*96); print("4) YEAR-SPLIT — all-UP (n_up=4) fwd10, stability"); print("="*96)
    g=d[base&(n_up==4)].copy(); g["yr"]=g.index.year
    for y in sorted(g.yr.unique()):
        gg=g[g.yr==y]
        if len(gg)<10: print(f"  {y}: thin"); continue
        print(f"  {y}: fwd10 {gg.f10.mean():+.2f}%  hit {(gg.f10>0).mean()*100:3.0f}%  n{len(gg)}")

    print("\n"+"="*96); print("5) CURRENT READ (today) — per-band trend + alignment"); print("="*96)
    last=U.iloc[-1]
    for b in BANDS:
        v=last[b]; print(f"  {b:6s} (EMA{BANDS[b]:3d}): {'UP' if v==1 else 'DOWN' if v==-1 else 'FLAT'}")
    print(f"  alignment: {int(n_up.iloc[-1])}/4 up, {int(n_dn.iloc[-1])}/4 down")
