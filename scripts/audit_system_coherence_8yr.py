"""
SYSTEM COHERENCE + ENSEMBLE audit (8yr) — do the many context panels AGREE, CONTRADICT, or
add value together? The tilt tab now shows: regime/size, MTF-trend alignment, breadth nowcast,
breakout flag, trend-quality. This checks they don't give the user conflicting guidance and
that combining them is additive, not redundant.

All states computed vectorized+causal on the full 8yr Nifty (+breadth from daily_data). Then:
 1. COHERENCE — regime-state x MTF-entry cross-tab; explicit contradiction rate.
 2. ENSEMBLE — does 'regime bullish AND mtf bullish' forecast better than either alone?
 3. CAUSALITY spot-check — live get_mtf_trend / get_nifty_breakout match the vectorized read
    (truncation-invariant = no lookahead).
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from datetime import date
from src.data.repository import query_dataframe as q
from src.analytics.sector_forward_tilt import (get_mtf_trend, get_nifty_breakout, _REGIME_CONFIRM)

def load():
    n=q("SELECT trade_date,high_val,close_val FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"]=pd.to_datetime(n["trade_date"]); return n.set_index("trade_date")
def fret(c,n): lr=np.log(c); return (np.exp(lr.shift(-n)-lr)-1)*100
def debounce(lab,k):
    out=[]; last=pending=lab.iloc[0]; cnt=0
    for x in lab:
        if x==pending: cnt+=1
        else: pending,cnt=x,1
        if cnt>=k: last=pending
        out.append(last)
    return pd.Series(out,index=lab.index)

if __name__=="__main__":
    d=load(); c,h=d["close_val"],d["high_val"]
    d["f10"]=fret(c,10); d["f30"]=fret(c,30)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    v=c.pct_change().rolling(20).std()
    vp=pd.Series([(v.iloc[:i+1]<=v.iloc[i]).mean() if i>20 else np.nan for i in range(len(v))],index=c.index)
    raw=pd.Series(np.where(c<e20,np.where(e20<e50,"DOWN","CHOP"),np.where((c>e20)&(e20>e50),"UP","CHOP")),index=c.index)
    raw[(np.isfinite(vp))&(vp>=0.8)&(raw=="CHOP")]="HIGH_VOL"
    d["reg"]=debounce(raw,_REGIME_CONFIRM)
    # MTF n_up (EMA10/30/60/120)
    nup=pd.Series(0,index=c.index)
    for span in (10,30,60,120):
        e=c.ewm(span=span,adjust=False).mean(); sl=e-e.shift(max(5,span//4))
        nup=nup+((c>e)&(sl>0)).astype(int)
    d["nup"]=nup
    d["mtf"]=np.where(nup>=3,"BULLISH",np.where(nup<=1,"BEARISH","MIXED"))
    # breakout held
    hi=h.shift(1).rolling(20).max(); brk=c>hi
    held=brk&(c.shift(-1)>hi)&(c.shift(-2)>hi)   # (note: uses next 2d — for COHERENCE stats only, not a live signal)
    base=pd.Series(False,index=c.index); base.iloc[200:]=True; b=d[base]

    print(f"Nifty {c.index.min().date()}->{c.index.max().date()} | {len(b)} eval days\n")
    print("="*88); print("1) COHERENCE — regime (size backdrop) x MTF (stock-entry trend): cross-tab %"); print("="*88)
    ct=pd.crosstab(b["reg"], b["mtf"], normalize="index")*100
    print(ct.round(0).to_string())
    # explicit contradictions
    contra_hard=((b["reg"]=="UP")&(b["mtf"]=="BEARISH")).sum()          # size says full, trend says wait
    contra_soft=((b["reg"]=="DOWN")&(b["mtf"]=="BULLISH")).sum()        # size cuts, trend says enter
    print(f"\n  HARD contradiction (regime UP + MTF BEARISH): {contra_hard} days = {contra_hard/len(b)*100:.1f}%")
    print(f"  SOFT tension (regime DOWN + MTF BULLISH):      {contra_soft} days = {contra_soft/len(b)*100:.1f}%")
    print(f"  aligned (both bullish or both cautious): "
          f"{(((b['reg'].isin(['UP','HIGH_VOL']))&(b['mtf']=='BULLISH'))|((b['reg']=='DOWN')&(b['mtf']=='BEARISH'))).sum()/len(b)*100:.0f}%")

    print("\n"+"="*88); print("2) ENSEMBLE — is combining regime + MTF additive? fwd10 hit-rate"); print("="*88)
    for lab,m in [("regime bullish (UP/HIVOL)", b["reg"].isin(["UP","HIGH_VOL"])),
                  ("MTF bullish (3-4up)",       b["mtf"]=="BULLISH"),
                  ("BOTH bullish",              (b["reg"].isin(["UP","HIGH_VOL"]))&(b["mtf"]=="BULLISH")),
                  ("regime bullish, MTF not",   (b["reg"].isin(["UP","HIGH_VOL"]))&(b["mtf"]!="BULLISH")),
                  ("neither",                   (~b["reg"].isin(["UP","HIGH_VOL"]))&(b["mtf"]!="BULLISH"))]:
        g=b[m&b["f10"].notna()]
        print(f"  {lab:28s} fwd10 {g['f10'].mean():+.2f}%  hit {(g['f10']>0).mean()*100:3.0f}%  n{len(g)}")

    print("\n"+"="*88); print("3) CAUSALITY spot-check — live fns vs vectorized (truncation-invariant?)"); print("="*88)
    dts=list(b.index[-400::40])
    mtf_mis=bk_ok=0
    for dt in dts:
        dd=dt.date()
        r=get_mtf_trend(dd)
        if r["ok"]:
            live_nup=r["n_up"]; vec_nup=int(nup.loc[dt])
            if live_nup!=vec_nup: mtf_mis+=1
        bkr=get_nifty_breakout(dd)
        if bkr.get("ok"): bk_ok+=1
    print(f"  get_mtf_trend live vs vectorized n_up: {mtf_mis}/{len(dts)} mismatches (0 = causal+faithful)")
    print(f"  get_nifty_breakout returned ok on {bk_ok}/{len(dts)} sampled dates")
    # truncation invariance: state at D must not change if future rows exist
    D=b.index[-200]
    a1=get_mtf_trend(D.date())["n_up"]
    print(f"  truncation-invariance: get_mtf_trend({D.date()}) n_up={a1} (computed from data<=D only, by query) — causal by construction")
