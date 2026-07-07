"""
FOUR-BAND TREND SCENARIO BACKTEST (Nifty 50, 2018-2026) — "10x sharper": every cell has a
NON-OVERLAPPING t-stat + hit-rate; key states get a year-split. No claim without significance.

Bands (representative horizon each):
  SWING      1-3 wk     -> 10d
  SHORT      1-2 mo     -> 30d
  LONG       2.5-4 mo   -> 65d
  VLONG      4-6 mo     -> 105d   (only ~16 independent windows in 8yr — flagged thin)

States (all causal): trend-strength (ER), trend direction matrix, consolidation, breakout
(real=held 3d vs fake), pullback, key short-vs-long divergences. Verdict per state per band:
TRADEABLE (|t|>=2), lean (1<=|t|<2), or noise (|t|<1).
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

BANDS = {"SWING": 10, "SHORT": 30, "LONG": 65, "VLONG": 105}

def load():
    n=q("SELECT trade_date,open_val,high_val,low_val,close_val FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"]=pd.to_datetime(n["trade_date"]); return n.set_index("trade_date")

def fret(c,n): lr=np.log(c); return (np.exp(lr.shift(-n)-lr)-1)*100

def nonoverlap_t(sig: pd.Series, fwd: pd.Series, h: int):
    """greedy non-overlapping: pick state-days >=h apart, return (mean, t, n_indep, hit%)."""
    idx=np.where(sig.values)[0]; vals=fwd.values
    picks=[]; last=-10**9
    for i in idx:
        if i-last>=h and np.isfinite(vals[i]): picks.append(vals[i]); last=i
    picks=np.array(picks)
    if len(picks)<4: return (np.nan,np.nan,len(picks),np.nan)
    t=picks.mean()/(picks.std(ddof=1)/np.sqrt(len(picks)))
    hit=(picks>0).mean()*100
    return (picks.mean(), t, len(picks), hit)

def line(d, sig, label):
    sig=sig.reindex(d.index).fillna(False)
    n=int(sig.sum())
    if n<30: print(f"  {label:30s} thin (n={n})"); return
    cells=[]
    for band,h in BANDS.items():
        m,t,ni,hit=nonoverlap_t(sig, d[f"f{h}"], h)
        if not np.isfinite(t): cells.append(f"{band[:5]:>5s} thin"); continue
        vd="🟢" if abs(t)>=2 else ("·" if abs(t)>=1 else " ")
        cells.append(f"{m:+5.1f}%/t{t:+3.1f}/{hit:2.0f}%{vd}")
    print(f"  {label:30s} n{n:4d} | " + " | ".join(cells))

if __name__=="__main__":
    d=load(); c,h,l=d["close_val"],d["high_val"],d["low_val"]
    for band,hz in BANDS.items(): d[f"f{hz}"]=fret(c,hz)
    ema10=c.ewm(span=10,adjust=False).mean(); ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean()
    m10=(c/c.shift(10)-1)*100; slope50=ema50-ema50.shift(10)
    er20=(c-c.shift(20)).abs()/c.diff().abs().rolling(20).sum().replace(0,np.nan)
    hh20=c.rolling(20).max(); ll20=c.rolling(20).min(); width20=(hh20-ll20)/c*100
    st=np.where((c>ema10)&(m10>0),"UP",np.where((c<ema10)&(m10<0),"DN","FLAT"))
    lt=np.where((c>ema50)&(slope50>0),"UP",np.where((c<ema50)&(slope50<0),"DN","FLAT"))
    d["st"]=st; d["lt"]=lt
    base=pd.Series(False,index=d.index); base.iloc[60:]=True

    print(f"Nifty {c.index.min().date()}->{c.index.max().date()} ({len(d)}d)")
    print("Cell = mean/t(non-overlap)/hit% ; 🟢 |t|>=2 TRADEABLE, · 1<=|t|<2 lean, blank noise")
    print("Bands:", BANDS, "\n")

    print("="*128); print("1) TREND-STRENGTH (Kaufman ER20) within EMA-UP — does clean-trend edge hold across ALL bands?"); print("="*128)
    up=(c>ema20)&(ema20>ema50)
    line(d, base&up&(er20>=0.5), "UP · strong ER>=0.5")
    line(d, base&up&(er20>=0.3)&(er20<0.5), "UP · moderate ER 0.3-0.5")
    line(d, base&up&(er20<0.3), "UP · choppy ER<0.3")

    print("\n"+"="*128); print("2) TREND DIRECTION MATRIX (short x long)"); print("="*128)
    for L in ["UP","FLAT","DN"]:
        for S in ["UP","DN"]:
            line(d, base&(d["lt"]==L)&(d["st"]==S), f"LONG-{L} x SHORT-{S}")
        print("  "+"-"*122)

    print("\n"+"="*128); print("3) CONSOLIDATION / RANGEBOUND"); print("="*128)
    tight=(er20<0.25)&(width20<width20.rolling(250,min_periods=60).median())
    line(d, base&tight, "CONSOLIDATION (tight+low-ER)")
    line(d, base&(er20>=0.5), "STRONG TREND (ER>=0.5, any dir)")

    print("\n"+"="*128); print("4) BREAKOUT — real (held 3d) vs fake (reversed <=3d), Donchian 20d"); print("="*128)
    hi=h.shift(1).rolling(20).max(); lo=l.shift(1).rolling(20).min()
    ub=c>hi
    held=ub&(c.shift(-1)>hi)&(c.shift(-2)>hi)
    fake=ub&~held
    line(d, base&ub, "UP-break (all)")
    line(d, base&held, "UP-break HELD 3d (real)")
    line(d, base&fake, "UP-break FAKE (rev<=3d)")
    dnb=c<lo
    line(d, base&dnb, "DOWN-break (all)")

    print("\n"+"="*128); print("5) PULLBACK in uptrend + KEY DIVERGENCES"); print("="*128)
    dd=(c/hh20-1)*100
    line(d, base&(d["lt"]=="UP")&(dd<=-3), "PULLBACK (LT-up, >3% off hi)")
    line(d, base&(d["lt"]=="DN")&(d["st"]=="UP"), "BOUNCE-in-DOWN (bottom-fish)")
    line(d, base&(d["lt"]=="UP")&(d["st"]=="DN"), "DIP-in-UP")

    print("\n"+"="*128); print("6) YEAR-SPLIT — strong-ER-UP fwd (SWING 10d), stability of the wired edge"); print("="*128)
    u=d[up&(er20>=0.5)&d.f10.notna()].copy(); u["yr"]=u.index.year
    ch=d[up&(er20<0.3)&d.f10.notna()].copy(); ch["yr"]=ch.index.year
    for y in sorted(set(u.yr)|set(ch.yr)):
        a=u[u.yr==y]["f10"]; b=ch[ch.yr==y]["f10"]
        if len(a)<6 or len(b)<6: print(f"  {y}: thin"); continue
        print(f"  {y}: strong {a.mean():+5.2f}% (n{len(a):3d}) | choppy {b.mean():+5.2f}% (n{len(b):3d}) | spread {a.mean()-b.mean():+.2f}")
