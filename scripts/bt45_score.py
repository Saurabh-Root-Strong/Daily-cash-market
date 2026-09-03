import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, duckdb
pd.set_option('display.width',250)

R = pd.read_pickle(r'C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad/replay45.pkl')
c = duckdb.connect('data/market_data.duckdb', read_only=True)
L = c.execute("SELECT trade_date, fno_symbol sym, direction_pred, composite_score, range_low, range_high, target_close, expected_move_pts, actual_return, direction_actual, was_correct FROM prediction_log").df()
L['trade_date']=pd.to_datetime(L['trade_date']).dt.date

M = R.merge(L, on=['trade_date','sym'], how='inner', suffixes=('','_log'))
print(f"=== A. CODE-DRIFT: replay(today's code) vs live log ===  matched {len(M)}/{len(R)}")
M['dcomp']=M['composite']-M['composite_score']
print(f"composite identical: {(M['dcomp'].abs()<1e-6).sum()}/{len(M)}   mean|Δ|={M['dcomp'].abs().mean():.3f}  max|Δ|={M['dcomp'].abs().max():.2f}")
print(f"direction identical: {(M['direction']==M['direction_pred']).sum()}/{len(M)}")
drift = M[M['dcomp'].abs()>1e-6]
if len(drift):
    print(drift.groupby('sym')['dcomp'].agg(['count','mean','min','max']).round(2).to_string())
    print("drift dates:", sorted(drift['trade_date'].unique())[:12])
print(f"actual_return identical: {(np.isclose(M['actual_ret'],M['actual_return'],atol=1e-3)).sum()}/{len(M)}")
print(f"multi-day forward gaps (>4 cal days): {(M['gap_days']>4).sum()}")

# band-consistent actual classification (same rule as engine scorer)
C_SIDE=0.40
def classify(ret, em, spot):
    band = C_SIDE*em/spot*100 if (em and spot and em>0) else 0.15
    return 'UP' if ret>band else ('DOWN' if ret<-band else 'SIDEWAYS')
R['act'] = [classify(r.actual_ret, r.em, r.spot) for r in R.itertuples()]
R['correct'] = R['direction']==R['act']
R['up'] = R['actual_ret']>0

def binom_p(k,n,p=0.5):
    if not n: return float('nan')
    k=int(k);n=int(n)
    pmf=[math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(n+1)]
    return min(1.0,sum(v for v in pmf if v<=pmf[k]*(1+1e-9)))

print("\n=== B. REPLAY SCORECARD (homogeneous current code, 45 sessions) ===")
out=[]
for sym,g in list(R.groupby('sym'))+[('POOLED',R)]:
    d=g[g['direction'].isin(['UP','DOWN'])]
    s=g[g['direction']=='SIDEWAYS']
    sign=((d['direction']=='UP')==d['up']).mean()*100 if len(d) else np.nan
    nxt=g['next_close']
    cov=((nxt>=g['rlo'])&(nxt<=g['rhi'])).mean()*100
    # realized signed return of taking the call, in bps
    ret=np.where(d['direction']=='UP',d['actual_ret'],-d['actual_ret'])*100
    out.append(dict(sym=sym,n=len(g),n_dir=len(d),
        succ3=g['correct'].mean()*100, fail3=(~g['correct']).mean()*100,
        dir_succ3=d['correct'].mean()*100 if len(d) else np.nan,
        dir_sign=sign, p=binom_p(round(sign/100*len(d)),len(d)) if len(d) else np.nan,
        sw_n=len(s), sw_succ=s['correct'].mean()*100 if len(s) else np.nan,
        IC=g['composite'].rank().corr(g['actual_ret'].rank()),
        cov1s=cov, bps_per_call=ret.mean() if len(d) else np.nan,
        bps_tot=ret.sum() if len(d) else np.nan))
print(pd.DataFrame(out).round(2).to_string(index=False))

print("\n=== C. FAILURE / FALL RATE decomposition (pooled) ===")
print(pd.crosstab(R['direction'],R['act'],margins=True).to_string())
print("\nsign-only confusion (pred vs actual sign, directional calls):")
d=R[R['direction'].isin(['UP','DOWN'])]
print(pd.crosstab(d['direction'],np.where(d['up'],'act_UP','act_DOWN')).to_string())
print("\nWORST failures — directional call, move went the other way hard:")
bad=d[((d['direction']=='UP')&(d['actual_ret']<-0.3))|((d['direction']=='DOWN')&(d['actual_ret']>0.3))]
print(bad[['trade_date','sym','direction','confidence','composite','actual_ret','dte']].sort_values('actual_ret').to_string(index=False))

print("\n=== D. RANGE-BAND breaks (fall outside 1sigma) ===")
R['out_lo']=R['next_close']<R['rlo']; R['out_hi']=R['next_close']>R['rhi']
print(R.groupby('sym')[['out_lo','out_hi']].agg(['sum']).to_string())
R['excess']=np.where(R['out_hi'],(R['next_close']-R['rhi'])/R['spot']*100,np.where(R['out_lo'],(R['rlo']-R['next_close'])/R['spot']*100,0))
print("mean overshoot when broken (% of spot):", round(R.loc[R['excess']>0,'excess'].mean(),3))

print("\n=== E. DTE / expiry-day edge cases ===")
R['expiry_day']=R['dte']<=1
for k,g in R.groupby('expiry_day'):
    d=g[g['direction'].isin(['UP','DOWN'])]
    print(f"dte<=1={k}: n={len(g)} succ3={g['correct'].mean()*100:.1f}% n_dir={len(d)} "
          f"sign={(((d['direction']=='UP')==d['up']).mean()*100 if len(d) else float('nan')):.1f}% cov={(((g['next_close']>=g['rlo'])&(g['next_close']<=g['rhi'])).mean()*100):.1f}%")

print("\n=== F. composite distribution vs +-8 threshold ===")
print(R.groupby('sym')['composite'].describe().round(2).to_string())
print("share |composite|>=8:", round((R['composite'].abs()>=8).mean()*100,1),"%")
print("share |composite|>=12:", round((R['composite'].abs()>=12).mean()*100,1),"%")

print("\n=== G. per-signal IC (replay, pooled 45d) ===")
recs=[]
for r in R.itertuples():
    for name,cat,score,dirn in r.sig:
        recs.append((r.trade_date,r.sym,name,cat,score,r.actual_ret))
S=pd.DataFrame(recs,columns=['trade_date','sym','name','cat','score','ret'])
agg=S.groupby(['cat','name']).apply(lambda g: pd.Series({
    'n':len(g),'fire_pct':(g['score']!=0).mean()*100,'mean_score':g['score'].mean(),
    'IC':g['score'].rank().corr(g['ret'].rank()) if g['score'].nunique()>1 else np.nan,
    'hit':((g['score']>0)==(g['ret']>0))[g['score']!=0].mean()*100 if (g['score']!=0).any() else np.nan}), include_groups=False)
print(agg.sort_values('IC').round(3).to_string())

print("\n=== H. family net IC ===")
fam=S[S['score']!=0].groupby(['trade_date','sym','cat'])['score'].sum().reset_index()
fam=fam.merge(R[['trade_date','sym','actual_ret']],on=['trade_date','sym'])
print(fam.groupby('cat').apply(lambda g: pd.Series({'n':len(g),'IC':g['score'].rank().corr(g['actual_ret'].rank()),'net_mean':g['score'].mean()}), include_groups=False).round(3).to_string())
