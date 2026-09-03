import sys, os, re, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, duckdb
pd.set_option('display.width',250)
rng=np.random.default_rng(7)
R = pd.read_pickle(r'C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad/replay45.pkl')
R['up']=R['actual_ret']>0

print("=== I. REGIME OF THE TEST WINDOW ===")
c=duckdb.connect('data/market_data.duckdb',read_only=True)
for s,n in {"NIFTY":"Nifty 50","BANKNIFTY":"Nifty Bank","FINNIFTY":"Nifty Financial Services","MIDCPNIFTY":"Nifty Midcap Select"}.items():
    g=R[R['sym']==s]
    tot=(g.sort_values('trade_date')['actual_ret']/100+1).prod()-1
    print(f"{s:11s} window ret={tot*100:+.2f}%  daily sd={g['actual_ret'].std():.2f}%  up-days={g['up'].mean()*100:.1f}%  |ret| mean={g['actual_ret'].abs().mean():.2f}%")
v=c.execute("SELECT avg(close_val) FROM index_data WHERE index_name='India VIX' AND trade_date BETWEEN '2026-06-25' AND '2026-08-27'").fetchone()[0]
v2=c.execute("SELECT avg(close_val) FROM index_data WHERE index_name='India VIX' AND trade_date BETWEEN '2024-12-01' AND '2026-06-24'").fetchone()[0]
print(f"India VIX avg: window={v:.2f}  prior(Dec24-Jun26)={v2:.2f}")

print("\n=== J. COMPOSITE MONOTONICITY (quintiles of composite -> next-day return) ===")
for sym,g in list(R.groupby('sym'))+[('POOLED',R)]:
    g=g.copy(); g['q']=pd.qcut(g['composite'],5,labels=False,duplicates='drop')
    t=g.groupby('q').agg(n=('actual_ret','size'),mean_ret=('actual_ret','mean'),up=('up','mean'))
    print(f"-- {sym}: " + "  ".join(f"Q{int(q)+1}:{r.mean_ret:+.2f}%(n{int(r.n)},up{r.up*100:.0f}%)" for q,r in t.iterrows()))

print("\n=== K. THRESHOLD SWEEP (|composite|>=T -> take direction, hold 1 day) ===")
for T in [3,5,6,8,10,12]:
    d=R[R['composite'].abs()>=T].copy()
    if d.empty: continue
    sgn=np.sign(d['composite'])
    ret=(sgn*d['actual_ret']).values*100
    hit=((sgn>0)==d['up']).mean()*100
    se=ret.std(ddof=1)/math.sqrt(len(ret)) if len(ret)>1 else float('nan')
    boot=np.array([rng.choice(ret,len(ret),replace=True).mean() for _ in range(4000)])
    print(f"T={T:>2}: n={len(d):3d} hit={hit:5.1f}% mean={ret.mean():+7.2f}bps t={ret.mean()/se if se else float('nan'):+5.2f} "
          f"boot95=[{np.percentile(boot,2.5):+.1f},{np.percentile(boot,97.5):+.1f}] total={ret.sum():+.0f}bps")

print("\n=== L. PER-INDEX shipped-threshold (>=8) economics, cost-aware ===")
COST=6.0  # bps round-trip, index futures (brokerage+STT+impact) - conservative floor
for sym,g in list(R.groupby('sym'))+[('POOLED',R)]:
    d=g[g['direction'].isin(['UP','DOWN'])]
    if d.empty: print(f"{sym:11s} no directional calls"); continue
    ret=np.where(d['direction']=='UP',d['actual_ret'],-d['actual_ret'])*100
    net=ret-COST
    print(f"{sym:11s} n={len(d):3d} gross={ret.mean():+7.2f}bps net(after {COST}bps)={net.mean():+7.2f}bps  "
          f"cum_net={net.sum():+8.1f}bps  win={((ret>0).mean()*100):.0f}%")

print("\n=== M. SIGNAL IC, names normalised ===")
recs=[]
for r in R.itertuples():
    for name,cat,score,dirn in r.sig:
        nm=re.sub(r'\d+[\d,\.]*','#',name)
        recs.append((r.trade_date,r.sym,nm,cat,score,r.actual_ret))
S=pd.DataFrame(recs,columns=['trade_date','sym','name','cat','score','ret'])
S=S[S['score']!=0]
agg=S.groupby(['cat','name']).apply(lambda g: pd.Series({
 'n':len(g),'mean_score':g['score'].mean(),
 'IC':(g['score'].rank().corr(g['ret'].rank()) if g['score'].nunique()>1 else np.nan),
 'dir_hit':((np.sign(g['score'])>0)==(g['ret']>0)).mean()*100,
 'mean_signed_ret_bps':(np.sign(g['score'])*g['ret']).mean()*100}),include_groups=False)
agg=agg[agg['n']>=8].sort_values('mean_signed_ret_bps')
print(agg.round(2).to_string())

print("\n=== N. FAMILY signed-return contribution (net score sign vs next-day move) ===")
fam=S.groupby(['trade_date','sym','cat'])['score'].sum().reset_index()
fam=fam.merge(R[['trade_date','sym','actual_ret']],on=['trade_date','sym'])
fam=fam[fam['score'].abs()>1e-9]
f=fam.groupby('cat').apply(lambda g: pd.Series({'n':len(g),
  'IC':g['score'].rank().corr(g['actual_ret'].rank()),
  'hit':((g['score']>0)==(g['actual_ret']>0)).mean()*100,
  'bps':(np.sign(g['score'])*g['actual_ret']).mean()*100,
  't':(np.sign(g['score'])*g['actual_ret']*100).mean()/((np.sign(g['score'])*g['actual_ret']*100).std(ddof=1)/math.sqrt(len(g)))}),include_groups=False)
print(f.round(2).to_string())

print("\n=== O. LONGER CONTEXT: all live-logged (2026-06-19+) vs backfilled replay era ===")
L=c.execute("SELECT trade_date,fno_symbol sym,direction_pred,composite_score,actual_return,was_correct,range_low,range_high,spot_close FROM prediction_log WHERE outcome_filled").df()
L['trade_date']=pd.to_datetime(L['trade_date'])
for label,mask in [('backfilled 2024-12..2026-06-18',L['trade_date']<'2026-06-19'),('live 2026-06-19+',L['trade_date']>='2026-06-19')]:
    g=L[mask]; d=g[g['direction_pred'].isin(['UP','DOWN'])]
    nxt=g['spot_close']*(1+g['actual_return']/100)
    print(f"{label:32s} n={len(g):5d} succ3={g['was_correct'].mean()*100:5.1f}% n_dir={len(d):4d} "
          f"sign={(((d['direction_pred']=='UP')==(d['actual_return']>0)).mean()*100):5.1f}% "
          f"IC={g['composite_score'].rank().corr(g['actual_return'].rank()):+.3f} "
          f"cov1s={(((nxt>=g['range_low'])&(nxt<=g['range_high'])).mean()*100):5.1f}%")

print("\n=== P. RANGE band calibration test (is 1sigma really 68%?) ===")
for sym,g in list(R.groupby('sym'))+[('POOLED',R)]:
    ins=((g['next_close']>=g['rlo'])&(g['next_close']<=g['rhi']))
    n=len(g); k=ins.sum(); p=k/n
    se=math.sqrt(p*(1-p)/n)
    z=(p-0.68)/se if se>0 else float('nan')
    print(f"{sym:11s} cov={p*100:5.1f}% (k={k}/{n}) 95%CI=[{max(0,p-1.96*se)*100:.1f},{min(1,p+1.96*se)*100:.1f}]  z vs 68%={z:+.2f}")
