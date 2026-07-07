"""
ACCURACY of the multi-timeframe trend read, per band per horizon (8yr Nifty, sharp).

Bands (from get_mtf_trend): swing EMA10 / short EMA30 / long EMA60 / vlong EMA120, state =
close-vs-EMA + slope. Question: when a band says UP/DOWN, does the market actually move that
way over that band's horizon — and BETTER than base rate?

Honest metrics (bull drift inflates any up-call):
  UP-acc   = P(fwd>0 | state UP)          base = P(fwd>0) unconditional
  DN-acc   = P(fwd<0 | state DOWN)        DN-base = 1-base
  UP-skill = UP-acc - base                (up-call value beyond drift)
  DN-skill = DN-acc - (1-base)            (<0 => down-calls WORSE than random — the contrarian tax)
Matched horizon: swing->10d short->30d long->65d vlong->105d. Also full band x horizon matrix,
alignment-state accuracy, and year-split.
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

BANDS={"swing":(10,10),"short":(30,30),"long":(60,65),"vlong":(120,105)}  # ema_span, matched_horizon
HZ=[10,30,65,105]

def load():
    n=q("SELECT trade_date,close_val FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"]=pd.to_datetime(n["trade_date"]); return n.set_index("trade_date")["close_val"].astype(float)
def fret(c,n): lr=np.log(c); return (np.exp(lr.shift(-n)-lr)-1)*100

if __name__=="__main__":
    c=load(); d=pd.DataFrame(index=c.index)
    for h in HZ: d[f"f{h}"]=fret(c,h)
    states={}
    for b,(span,_) in BANDS.items():
        e=c.ewm(span=span,adjust=False).mean(); sl=e-e.shift(max(5,span//4))
        states[b]=pd.Series(np.where((c>e)&(sl>0),"UP",np.where((c<e)&(sl<0),"DOWN","FLAT")),index=c.index)
    valid=d["f105"].notna()  # need all horizons; use widest for the matched table where relevant

    print(f"Nifty {c.index.min().date()}->{c.index.max().date()} ({len(c)}d)")
    print("="*104)
    print("1) MATCHED-HORIZON DIRECTIONAL ACCURACY per band (does the band's trend persist over its horizon?)")
    print("="*104)
    print(f"  {'band':6s} {'horizon':>7s} | {'base P(up)':>10s} | {'UP-acc':>7s} {'UP-skill':>8s} n | {'DN-acc':>7s} {'DN-skill':>8s} n")
    for b,(span,h) in BANDS.items():
        s=states[b]; f=d[f"f{h}"]; m=f.notna()
        base=(f[m]>0).mean()*100
        up=s[m]=="UP"; dn=s[m]=="DOWN"
        upacc=(f[m][up]>0).mean()*100 if up.sum() else np.nan
        dnacc=(f[m][dn]<0).mean()*100 if dn.sum() else np.nan
        print(f"  {b:6s} {h:6d}d | {base:9.0f}% | {upacc:6.0f}% {upacc-base:+7.0f}% {int(up.sum()):4d} | "
              f"{dnacc:6.0f}% {dnacc-(100-base):+7.0f}% {int(dn.sum()):4d}")
    print("  UP-skill>0 = up-call beats drift ; DN-skill<0 = down-call WORSE than random (contrarian tax)")

    print("\n"+"="*104)
    print("2) FULL band x forward-horizon accuracy MATRIX — UP-call P(fwd>0) [base in ()]")
    print("="*104)
    hdr="  "+f"{'band(UP)':10s}"+"".join(f"{h:>8d}d" for h in HZ)
    print(hdr)
    for b,(span,_) in BANDS.items():
        s=states[b]; line=f"  {b:10s}"
        for h in HZ:
            f=d[f"f{h}"]; m=f.notna()&(s=="UP")
            line+=f"{(f[m]>0).mean()*100:7.0f}%" if m.sum()>20 else "   thin"
        print(line)
    line="  "+f"{'BASE P(up)':10s}"+"".join(f"{(d[f'f{h}']>0).mean()*100:7.0f}%" for h in HZ); print(line)

    print("\n"+"="*104)
    print("3) DOWN-call accuracy MATRIX — P(fwd<0 | DOWN) [want > (100-base); low = market bounced]")
    print("="*104)
    print("  "+f"{'band(DN)':10s}"+"".join(f"{h:>8d}d" for h in HZ))
    for b,(span,_) in BANDS.items():
        s=states[b]; line=f"  {b:10s}"
        for h in HZ:
            f=d[f"f{h}"]; m=f.notna()&(s=="DOWN")
            line+=f"{(f[m]<0).mean()*100:7.0f}%" if m.sum()>20 else "   thin"
        print(line)
    print("  "+f"{'DN-base':10s}"+"".join(f"{(d[f'f{h}']<0).mean()*100:7.0f}%" for h in HZ))

    print("\n"+"="*104)
    print("4) ALIGNMENT-STATE accuracy — n_up bands, P(fwd10>0) & P(fwd65>0)")
    print("="*104)
    S=pd.DataFrame(states); n_up=(S=="UP").sum(axis=1)
    for k in range(5):
        m=(n_up==k)&d["f10"].notna()
        if m.sum()<30: print(f"  {k}/4 up: thin"); continue
        print(f"  {k}/4 up: P(fwd10>0) {(d['f10'][m]>0).mean()*100:3.0f}%  P(fwd65>0) {(d['f65'][(n_up==k)&d['f65'].notna()]>0).mean()*100:3.0f}%  n{int(m.sum())}")

    print("\n"+"="*104)
    print("5) YEAR-SPLIT — swing UP-acc (fwd10) & DOWN-acc, is per-band accuracy stable?")
    print("="*104)
    s=states["swing"]; d["yr"]=d.index.year
    for y in sorted(set(d.yr)):
        g=d[d.yr==y]; sg=s.reindex(g.index)
        up=g[(sg=="UP")&g.f10.notna()]; dn=g[(sg=="DOWN")&g.f10.notna()]
        if len(up)<15 or len(dn)<10: print(f"  {y}: thin"); continue
        print(f"  {y}: swing UP-acc {(up.f10>0).mean()*100:3.0f}% (n{len(up):3d}) | DOWN-acc {(dn.f10<0).mean()*100:3.0f}% (n{len(dn):3d})")
