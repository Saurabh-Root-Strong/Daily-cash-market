"""
FULL 8-YEAR REGIME AUDIT (Nifty 50, 2018-01 -> 2026-07). Steps through every claim the
system makes about market state and checks it against realized behaviour. All causal.

A. Regime label distribution by year + known-episode sanity.
B. Does each shipped regime label DESCRIBE the market correctly? (fwd return by state)
C. Regime STABILITY — transitions, durations, whipsaw (fake regime flips).
D. TREND vs CONSOLIDATION — Kaufman efficiency ratio; is CHOP really choppy?
E. PULLBACK vs REVERSAL — in an uptrend, which dips recover vs turn into downtrends?
F. FAKE vs REAL BREAKOUT — Donchian close-breaks, 8yr (extends the 4yr level-break audit).
G. BREADTH nowcast (5-state) — does BULL/RECOVERING/WEAKENING/BEAR match forward reality?
"""
from __future__ import annotations
import sys; sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

def load():
    n = q("SELECT trade_date,open_val,high_val,low_val,close_val FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
    n["trade_date"] = pd.to_datetime(n["trade_date"])
    return n.set_index("trade_date")

def breadth():
    d = q("""SELECT b.trade_date, b.symbol, b.close_price FROM daily_data b
             WHERE b.series='EQ' AND b.turnover_lacs>=500 AND b.prev_close>=5""")
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    p = d.pivot_table("close_price","trade_date","symbol").sort_index()
    b50 = (p > p.rolling(50).mean()).sum(axis=1)/p.rolling(50).mean().notna().sum(axis=1).replace(0,np.nan)*100
    b200 = (p > p.rolling(200).mean()).sum(axis=1)/p.rolling(200).mean().notna().sum(axis=1).replace(0,np.nan)*100
    return b50, b200

def fret(c,n): lr=np.log(c); return (np.exp(lr.shift(-n)-lr)-1)*100

if __name__ == "__main__":
    d = load(); c,h,l = d["close_val"],d["high_val"],d["low_val"]
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    sma200=c.rolling(200).mean()
    r=c.pct_change(); vol20=r.rolling(20).std()
    volpct=pd.Series([(vol20.iloc[:i+1]<=vol20.iloc[i]).mean() if i>20 else np.nan for i in range(len(vol20))], index=c.index)
    ret5=fret(c,5).shift(5).mul(-1)  # trailing 5d approx not needed; use pct
    r5=(c/c.shift(5)-1)*100; r20=(c/c.shift(20)-1)*100
    for n in (5,10,20,40): d[f"f{n}"]=fret(c,n)

    # shipped regime replica (sector_forward_tilt._market_regime)
    reg=pd.Series("CHOP",index=c.index)
    reg[(r5<=-3)&(r20>0)]="REVERSAL"
    reg[(c<e20)&(e20<e50)&(reg=="CHOP")]="DOWN"
    reg[(volpct>=0.80)&(reg=="CHOP")]="HIGH_VOL"
    reg[(c>e20)&(e20>e50)&(reg=="CHOP")]="UP"
    # apply priority correctly: reversal>down>highvol>up>chop
    reg=pd.Series("CHOP",index=c.index)
    for i in range(len(c)):
        if i<60 or not np.isfinite(vol20.iloc[i]): reg.iloc[i]="UNKNOWN"; continue
        if r5.iloc[i]<=-3 and r20.iloc[i]>0: reg.iloc[i]="REVERSAL"
        elif c.iloc[i]<e20.iloc[i]<e50.iloc[i]: reg.iloc[i]="DOWN"
        elif np.isfinite(volpct.iloc[i]) and volpct.iloc[i]>=0.80: reg.iloc[i]="HIGH_VOL"
        elif c.iloc[i]>e20.iloc[i]>e50.iloc[i]: reg.iloc[i]="UP"
        else: reg.iloc[i]="CHOP"
    d["reg"]=reg; d["yr"]=c.index.year

    print("="*96); print("A) REGIME LABEL DISTRIBUTION BY YEAR (% of trading days)"); print("="*96)
    tab=d[d.reg!="UNKNOWN"].groupby("yr")["reg"].value_counts(normalize=True).unstack().fillna(0)*100
    print(tab.reindex(columns=["UP","CHOP","HIGH_VOL","DOWN","REVERSAL"]).round(0).to_string())

    print("\n"+"="*96); print("B) DOES EACH REGIME DESCRIBE THE MARKET? fwd Nifty return by state (should match name)"); print("="*96)
    print(f"  {'state':10s} {'days':>5s} {'f5':>7s} {'f10':>7s} {'f20':>7s} {'f40':>7s} {'P(f20>0)':>9s}")
    for s in ["UP","CHOP","HIGH_VOL","DOWN","REVERSAL"]:
        g=d[d.reg==s]
        print(f"  {s:10s} {len(g):5d} {g.f5.mean():+6.2f}% {g.f10.mean():+6.2f}% {g.f20.mean():+6.2f}% {g.f40.mean():+6.2f}% {(g.f20>0).mean()*100:8.0f}%")

    print("\n"+"="*96); print("C) REGIME STABILITY — transitions, median duration, whipsaw (flip back within 5d)"); print("="*96)
    rc=d[d.reg!="UNKNOWN"]["reg"]; trans=(rc!=rc.shift()).sum()
    runs=(rc!=rc.shift()).cumsum(); durs=rc.groupby(runs).size()
    # whipsaw: a run of length<=3 sandwiched by the SAME other regime
    seg=rc.groupby(runs).agg(lambda x:x.iloc[0]); seglen=rc.groupby(runs).size()
    whip=0
    for i in range(1,len(seg)-1):
        if seglen.iloc[i]<=3 and seg.iloc[i-1]==seg.iloc[i+1] and seg.iloc[i]!=seg.iloc[i-1]: whip+=1
    print(f"  transitions {trans} over {len(rc)} days | median run {durs.median():.0f}d | mean run {durs.mean():.1f}d | runs {len(durs)}")
    print(f"  whipsaw flips (<=3d island, same regime both sides) {whip} = {whip/len(durs)*100:.0f}% of runs")
    print(f"  regime durations by state (median days):")
    for s in ["UP","CHOP","HIGH_VOL","DOWN","REVERSAL"]:
        sd=seglen[seg==s];
        if len(sd): print(f"    {s:10s} median {sd.median():.0f}d  n_runs {len(sd)}  longest {sd.max()}d")

    print("\n"+"="*96); print("D) TREND vs CONSOLIDATION — Kaufman ER(10). ER<0.3 = choppy/consolidation."); print("="*96)
    er=(c-c.shift(10)).abs()/c.diff().abs().rolling(10).sum().replace(0,np.nan)
    d["er"]=er
    for s in ["UP","CHOP","HIGH_VOL","DOWN","REVERSAL"]:
        g=d[d.reg==s]; print(f"  {s:10s} mean ER {g.er.mean():.2f}  |  % days ER<0.3 (ranging) {(g.er<0.3).mean()*100:3.0f}%")
    print(f"  => CHOP should have LOW ER (ranging); trends HIGH ER. Check consistency above.")

    print("\n"+"="*96); print("E) PULLBACK vs REVERSAL — from an UP state, a >3% drop from 20d-high:"); print("="*96)
    hh20=c.rolling(20).max(); dd=(c/hh20-1)*100      # drawdown from 20d high
    inup=(d.reg=="UP")|(d.reg=="HIGH_VOL")
    dip=inup.shift(1).fillna(False)&(dd<=-3)&(dd.shift(1)>-3)   # first cross into -3% dip from an up state
    dipdays=d[dip.reindex(d.index).fillna(False)]
    rec=(dipdays.f20>0)
    print(f"  dips (>3% off 20d-high, from up state): {len(dipdays)} | recovered in 20d (f20>0): {rec.mean()*100:.0f}%  avg f20 {dipdays.f20.mean():+.2f}%")
    print(f"  => high recovery% = mostly PULLBACKS (buyable); low = many turn into REVERSALS")

    print("\n"+"="*96); print("F) FAKE vs REAL BREAKOUT — Donchian close-breaks, 8yr"); print("="*96)
    for W in (20,50):
        hi=h.shift(1).rolling(W).max(); lo=l.shift(1).rolling(W).min()
        up_bk=c>hi; dn_bk=c<lo
        # fake = reversed back inside within 5d
        def fake(sig,level,side):
            f=0;n=int(sig.sum())
            for i in np.where(sig.values)[0]:
                if i+5<len(c):
                    if side=="up" and (c.iloc[i+1:i+6]<level.iloc[i]).any(): f+=1
                    if side=="dn" and (c.iloc[i+1:i+6]>level.iloc[i]).any(): f+=1
            return n,f
        nu,fu=fake(up_bk,hi,"up"); nd,fd=fake(dn_bk,lo,"dn")
        gu=d[up_bk.reindex(d.index).fillna(False)]; gd=d[dn_bk.reindex(d.index).fillna(False)]
        print(f"  {W}d UP-break:  n{nu} fake(rev<5d) {fu/nu*100:3.0f}%  fwd20 {gu.f20.mean():+.2f}%  P(cont) {(gu.f20>0).mean()*100:.0f}%")
        print(f"  {W}d DN-break:  n{nd} fake(rev<5d) {fd/nd*100:3.0f}%  fwd20 {gd.f20.mean():+.2f}%  P(down) {(gd.f20<0).mean()*100:.0f}%")

    print("\n"+"="*96); print("G) BREADTH NOWCAST (5-state) vs forward reality (2018-2026)"); print("="*96)
    try:
        b50,b200=breadth(); b50=b50.reindex(c.index); px200=(c/sma200-1)*100
        st=pd.Series("NEUTRAL",index=c.index)
        st[(b50>=55)&(px200>0)]="BULL"; st[(b50>=55)&(px200<=0)]="RECOVERING"
        st[(b50<40)&(px200>0)]="WEAKENING"; st[(b50<40)&(px200<=0)]="BEAR"
        d["bst"]=st
        print(f"  {'state':11s} {'days':>5s} {'f10':>7s} {'f20':>7s} {'f40':>7s} {'P(f20>0)':>9s}")
        for s in ["BULL","RECOVERING","NEUTRAL","WEAKENING","BEAR"]:
            g=d[(d.bst==s)&d.f20.notna()]
            if len(g)<10: print(f"  {s:11s} thin"); continue
            print(f"  {s:11s} {len(g):5d} {g.f10.mean():+6.2f}% {g.f20.mean():+6.2f}% {g.f40.mean():+6.2f}% {(g.f20>0).mean()*100:8.0f}%")
    except Exception as e:
        print("  breadth step failed:", e)
