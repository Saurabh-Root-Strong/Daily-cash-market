import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
SP = (r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
      r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad")
rng = np.random.default_rng(23)
D = pd.read_pickle(SP + '/fp_panel.pkl')
order = ['deep ITM','ITM','ATM','OTM','deep OTM']

def nw_t(s, lag):
    x = s.dropna().values.astype(float); n=len(x)
    if n < 25: return float('nan')
    e = x-x.mean(); v=(e@e)/n
    for L in range(1,lag+1): v += 2*(1-L/(lag+1))*((e[L:]@e[:-L])/n)
    return x.mean()/math.sqrt(v/n) if v>0 else float('nan')

cands={}
A = D[D['agree']!='COLLIDE']
for h in (1,3,5,10):
    cands[f'LS {h}d'] = (A[f'f{h}']*A['pred']).groupby(A['trade_date']).mean()
    for m in order:
        s=A[A['dom_money']==m]
        if len(s)>200: cands[f'LS {h}d {m}']=(s[f'f{h}']*s['pred']).groupby(s['trade_date']).mean()
    for N in (5,10,25,50):
        r=D.sort_values('dom_add_cr',ascending=False).groupby('trade_date').head(N)
        a=r[r['agree']!='COLLIDE']
        cands[f'LS {h}d top{N}']=(a[f'f{h}']*a['pred']).groupby(a['trade_date']).mean()
    for q in range(5):
        s=A[pd.qcut(A['dom_add_cr'],5,labels=False,duplicates='drop')==q]
        cands[f'LS {h}d addQ{q+1}']=(s[f'f{h}']*s['pred']).groupby(s['trade_date']).mean()
    for fa in ['LONG BUILDUP','SHORT COVERING','SHORT BUILDUP','LONG UNWINDING']:
        s=A[A['fut_action']==fa]
        cands[f'{fa} {h}d']=(s[f'f{h}']*s['pred']).groupby(s['trade_date']).mean()
for lab in ['AGREE BULL','AGREE BEAR','COLLIDE']:
    s=D[D['agree']==lab]
    cands[f'{lab} 3d']=s['f3'].groupby(s['trade_date']).mean()

tt={k:nw_t(v,3) for k,v in cands.items()}
top=pd.Series(tt).sort_values(key=abs,ascending=False).head(12)
print("=== M. FULL SEARCH — every cell examined in this study ===")
print(f"candidates tested: {len(cands)}")
print(top.round(2).to_string())
idxu=sorted(set().union(*[set(v.index) for v in cands.values()]))
M=pd.DataFrame({k:v.reindex(idxu) for k,v in cands.items()}); Mc=M-M.mean()
T=len(Mc); Mv=Mc.values
obs=np.nanmax(np.abs([tt[k] for k in M.columns]))
B,block=3000,5; nulls=np.empty(B)
for b in range(B):
    idx=[]
    while len(idx)<T:
        st=rng.integers(0,T); ln=rng.geometric(1/block)
        idx.extend(((st+np.arange(ln))%T).tolist())
    S=Mv[np.array(idx[:T])]
    with np.errstate(invalid='ignore'):
        mu=np.nanmean(S,axis=0); sd=np.nanstd(S,axis=0,ddof=1); n_=(~np.isnan(S)).sum(axis=0)
        nulls[b]=np.nanmax(np.abs(mu/(sd/np.sqrt(n_))))
print(f"\nbest |t| observed = {obs:.2f}")
print(f"date-block null max|t|: median {np.median(nulls):.2f}, 95th {np.percentile(nulls,95):.2f}, 99th {np.percentile(nulls,99):.2f}")
print(f"White Reality Check p = {(nulls>=obs).mean():.4f}")
