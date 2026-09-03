import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, duckdb
pd.set_option('display.width',250)
R=pd.read_pickle(r'C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad/replay45.pkl'); R['up']=R['actual_ret']>0
C_SIDE=0.40
def cls(ret,em,spot,mult=1.0):
    band=C_SIDE*em*mult/spot*100 if (em and spot and em>0) else 0.15
    return 'UP' if ret>band else ('DOWN' if ret<-band else 'SIDEWAYS')

print("=== Q. EXPIRY-DAY OVERRIDE (dte<=1) bypasses the composite ===")
e=R[R['dte']<=1].copy()
e['act']=[cls(r.actual_ret,r.em,r.spot) for r in e.itertuples()]
e['contradiction']=((e['direction']=='UP')&(e['composite']<0))|((e['direction']=='DOWN')&(e['composite']>0))
print(e[['trade_date','sym','dte','direction','confidence','composite','actual_ret','act','contradiction']].to_string(index=False))
d=e[e['direction'].isin(['UP','DOWN'])]
print(f"\nexpiry override: {len(d)} directional calls of {len(R[R['direction'].isin(['UP','DOWN'])])} total "
      f"({len(d)/len(R[R['direction'].isin(['UP','DOWN'])])*100:.0f}%); sign-hit={(((d['direction']=='UP')==d['up']).mean()*100):.1f}%; "
      f"score-sign contradictions={e['contradiction'].sum()}")
nd=R[(R['dte']>1)&(R['direction'].isin(['UP','DOWN']))]
print(f"non-expiry directional: n={len(nd)} sign-hit={(((nd['direction']=='UP')==nd['up']).mean()*100):.1f}% "
      f"mean signed={np.where(nd['direction']=='UP',nd['actual_ret'],-nd['actual_ret']).mean()*100:+.2f}bps")

print("\n=== R. SELF-REFERENTIAL SCORING: rescore with a CALIBRATED sigma ===")
# window coverage 81.7% -> sigma overestimated. find mult that yields 68% coverage
def cov(m): return (((R['next_close']>=R['spot']-R['em']*m)&(R['next_close']<=R['spot']+R['em']*m)).mean())
lo,hi=0.3,1.5
for _ in range(60):
    mid=(lo+hi)/2
    if cov(mid)<0.68: lo=mid
    else: hi=mid
m68=(lo+hi)/2
print(f"sigma multiplier for true 68% coverage = {m68:.3f}  (engine uses 1.000 -> band is {1/m68:.2f}x too wide)")
for label,mult in [('as-shipped (1.00)',1.0),(f'calibrated ({m68:.2f})',m68)]:
    RR=R.copy(); RR['act']=[cls(r.actual_ret,r.em,r.spot,mult) for r in RR.itertuples()]
    RR['ok']=RR['direction']==RR['act']
    s=RR[RR['direction']=='SIDEWAYS']; dd=RR[RR['direction'].isin(['UP','DOWN'])]
    print(f"{label:22s} succ_all={RR['ok'].mean()*100:5.1f}%  SIDEWAYS succ={s['ok'].mean()*100:5.1f}% (base rate of actual SIDEWAYS={(RR['act']=='SIDEWAYS').mean()*100:.1f}%)  dir succ={dd['ok'].mean()*100:5.1f}%")

print("\n=== S. TARGET (point forecast) skill vs naive random walk ===")
t=R.dropna(subset=['tgt'])
for sym,g in list(t.groupby('sym'))+[('POOLED',t)]:
    mt=(g['tgt']-g['next_close']).abs().mean(); m0=(g['spot']-g['next_close']).abs().mean()
    rt=np.sqrt(((g['tgt']-g['next_close'])**2).mean()); r0=np.sqrt(((g['spot']-g['next_close'])**2).mean())
    print(f"{sym:11s} n={len(g):3d} MAE_target={mt:8.1f} MAE_naive={m0:8.1f} skill={(1-mt/m0)*100:+6.2f}%  RMSE skill={(1-rt/r0)*100:+6.2f}%")

print("\n=== T. INSTITUTIONAL DATA STALENESS in window ===")
c=duckdb.connect('data/market_data.duckdb',read_only=True)
idx=set(pd.to_datetime(c.execute("SELECT DISTINCT trade_date FROM index_data WHERE index_name='Nifty 50' AND trade_date BETWEEN '2026-06-01' AND '2026-08-28'").df()['trade_date']).dt.date)
for tbl,cond in [('fao_participant',"AND data_type='OI'"),('fii_derivatives_stats',''),('fno_bhavcopy',"AND symbol='NIFTY'")]:
    have=set(pd.to_datetime(c.execute(f"SELECT DISTINCT trade_date FROM {tbl} WHERE trade_date BETWEEN '2026-06-01' AND '2026-08-28' {cond}").df()['trade_date']).dt.date)
    miss=sorted(d for d in idx if d not in have and d>=pd.Timestamp('2026-06-25').date())
    print(f"{tbl:24s} sessions_present={len(have & idx)}/{len(idx)}  missing in window: {miss}")

print("\n=== U. MULTIPLE-TESTING LEDGER ===")
tests=[]
for sym,g in R.groupby('sym'):
    d=g[g['direction'].isin(['UP','DOWN'])]
    if len(d)<3: continue
    k=int(((d['direction']=='UP')==d['up']).sum()); n=len(d)
    pmf=[math.comb(n,i)*0.5**n for i in range(n+1)]
    p=min(1.0,sum(v for v in pmf if v<=pmf[k]*(1+1e-9)))
    tests.append((sym,n,k,k/n*100,p))
T=pd.DataFrame(tests,columns=['sym','n','hits','hit_pct','p_raw'])
T=T.sort_values('p_raw'); T['rank']=range(1,len(T)+1)
T['bonferroni']=(T['p_raw']*len(T)).clip(upper=1); T['BH_q']=(T['p_raw']*len(T)/T['rank']).clip(upper=1)
print(T.round(3).to_string(index=False))
print("\nSurvivors at BH q<0.10:", list(T[T['BH_q']<0.10]['sym']) or "NONE")
