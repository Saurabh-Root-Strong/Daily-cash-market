"""
TWO-HORIZON TREND SCENARIO BACKTEST (Nifty 50, 2018-2026, ~2090 days). Validates every
trend state at BOTH bands the desk trades:
  SHORT band  = 1-3 weeks   -> fwd 5 / 10 / 15d
  LONG  band  = 1.5-3.5 mo  -> fwd 30 / 50 / 70d
States (all causal): short-trend x long-trend matrix, CONSOLIDATION (rangebound),
BREAKOUT (real vs fake, short+long), PULLBACK (dip in an uptrend). For each: mean fwd
return + hit-rate + non-overlapping t at both bands. Answers which state is tradeable at
which horizon, and whether the shipped med-term(40d)/divergence axis is the right cut.
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

def load():
    n = q("SELECT trade_date,open_val,high_val,low_val,close_val FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"] = pd.to_datetime(n["trade_date"]); return n.set_index("trade_date")

def fret(c,n): lr=np.log(c); return (np.exp(lr.shift(-n)-lr)-1)*100
def tno(s,h):
    s=s.dropna();
    if len(s)<h+4: return np.nan
    sub=s.iloc[::h]; return sub.mean()/(sub.std()/np.sqrt(len(sub))) if len(sub)>=4 else np.nan

SB=[5,10,15]; LB=[30,50,70]

def row(d, mask, label):
    g=d[mask]
    if len(g)<25: print(f"  {label:30s} thin (n={len(g)})"); return
    def cell(h):
        m=g[f"f{h}"].mean(); t=tno(d[mask][f"f{h}"] if False else g.set_index(d.index[mask])[f"f{h}"] if False else g[f"f{h}"], h)
        return m
    s=" ".join(f"{g[f'f{h}'].mean():+5.1f}" for h in SB)
    l=" ".join(f"{g[f'f{h}'].mean():+5.1f}" for h in LB)
    hs=(g["f10"]>0).mean()*100; hl=(g["f50"]>0).mean()*100
    print(f"  {label:30s} | SHORT {s} | LONG {l} | hit10 {hs:3.0f}% hit50 {hl:3.0f}% | n{len(g):4d}")

if __name__ == "__main__":
    d=load(); c,h,l=d["close_val"],d["high_val"],d["low_val"]
    for n in SB+LB: d[f"f{n}"]=fret(c,n)
    ema10=c.ewm(span=10,adjust=False).mean(); ema20=c.ewm(span=20,adjust=False).mean()
    ema50=c.ewm(span=50,adjust=False).mean()
    m10=(c/c.shift(10)-1)*100; m50=(c/c.shift(50)-1)*100
    er10=(c-c.shift(10)).abs()/c.diff().abs().rolling(10).sum().replace(0,np.nan)
    er20=(c-c.shift(20)).abs()/c.diff().abs().rolling(20).sum().replace(0,np.nan)
    slope50=ema50-ema50.shift(10)
    hh20=c.rolling(20).max(); ll20=c.rolling(20).min(); width20=(hh20-ll20)/c*100
    d["st"]=np.where((c>ema10)&(m10>0),"UP",np.where((c<ema10)&(m10<0),"DN","FLAT"))
    d["lt"]=np.where((c>ema50)&(slope50>0),"UP",np.where((c<ema50)&(slope50<0),"DN","FLAT"))
    base=d.index[60:]

    print(f"Nifty {c.index.min().date()}->{c.index.max().date()} | SHORT band {SB}d  LONG band {LB}d\n")
    print("="*110); print("1) SHORT-trend x LONG-trend MATRIX — fwd% at both bands (the core two-horizon map)")
    print("   read: does a short trend persist over 1-3wk? does the long trend dominate over 1.5-3.5mo?"); print("="*110)
    for lt in ["UP","FLAT","DN"]:
        for st in ["UP","FLAT","DN"]:
            row(d.loc[base], (d.loc[base,"lt"]==lt)&(d.loc[base,"st"]==st), f"LONG-{lt} x SHORT-{st}")
        print("  "+"-"*104)

    print("\n"+"="*110); print("2) CONSOLIDATION / RANGEBOUND — ER20<0.25 & tight 20d range; does it resolve directionally?")
    print("="*110)
    tight=(er20<0.25)&(width20<width20.rolling(250,min_periods=60).median())
    row(d.loc[base], tight.reindex(base).fillna(False), "CONSOLIDATION (tight+low-ER)")
    row(d.loc[base], (er20<0.25).reindex(base).fillna(False), "  low-ER only (ER20<0.25)")
    row(d.loc[base], (er20>=0.5).reindex(base).fillna(False), "  strong-trend (ER20>=0.5)")

    print("\n"+"="*110); print("3) BREAKOUT — Donchian close-break, real vs fake, at SHORT and LONG bands")
    print("="*110)
    for W in (20,50):
        hi=h.shift(1).rolling(W).max(); lo=l.shift(1).rolling(W).min()
        ub=(c>hi).reindex(base).fillna(False); db=(c<lo).reindex(base).fillna(False)
        row(d.loc[base], ub, f"UP-break {W}d (all)")
        row(d.loc[base], db, f"DN-break {W}d (all)")
        # fake filter: only breaks that HELD 3d (close stayed beyond level 3 days)
        held_u=((c>hi)&(c.shift(-1)>hi)&(c.shift(-2)>hi)).reindex(base).fillna(False)
        row(d.loc[base], held_u, f"  UP-break {W}d HELD 3d (real)")
        print("  "+"-"*104)

    print("\n"+"="*110); print("4) PULLBACK — in a LONG-UP trend, short dip >3% off 20d-high: buyable at which band?")
    print("="*110)
    dd=(c/hh20-1)*100
    pb=((d["lt"]=="UP")&(dd<=-3)&(dd.shift(1)>-3)).reindex(base).fillna(False)
    row(d.loc[base], pb, "PULLBACK in uptrend (>3% dip)")
    deep=((d["lt"]=="UP")&(dd<=-6)&(dd.shift(1)>-6)).reindex(base).fillna(False)
    row(d.loc[base], deep, "  deep pullback (>6% dip)")

    print("\n"+"="*110); print("5) KEY DIVERGENCES (short vs long disagree) — the tradeable edges")
    print("="*110)
    row(d.loc[base], (d.loc[base,"lt"]=="UP")&(d.loc[base,"st"]=="DN"), "DIP-in-UPtrend (LT-up,ST-dn)")
    row(d.loc[base], (d.loc[base,"lt"]=="DN")&(d.loc[base,"st"]=="UP"), "BOUNCE-in-DOWN (LT-dn,ST-up) BULLTRAP")
    row(d.loc[base], (d.loc[base,"lt"]=="UP")&(d.loc[base,"st"]=="UP"), "ALIGNED-UP")
    row(d.loc[base], (d.loc[base,"lt"]=="DN")&(d.loc[base,"st"]=="DN"), "ALIGNED-DOWN")
