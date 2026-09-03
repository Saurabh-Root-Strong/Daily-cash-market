"""Stress-test the ONE candidate that survived the search: composite -> overnight gap.

Kill-tests, in order of how likely each is to destroy the result:
  G1 date-clustered / Newey-West inference (4 correlated indices, serial corr)
  G2 sub-period stability
  G3 is it just today's move? (momentum control)
  G4 is it just close-location-in-range? (the known CLR gap edge)
  G5 incremental value over both controls (multivariate)
  G6 tradeable net of cost, and after the open-slippage haircut
  G7 reality check redone with DATE-block resampling
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np

pd.set_option('display.width', 250)
SP = (r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
      r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad")
rng = np.random.default_rng(3)

D = pd.read_pickle(SP + "/scored_long.pkl").dropna(subset=['gap']).copy()
D = D[np.sign(D['composite']) != 0]
D['sgn'] = np.sign(D['composite'])
D['r'] = D['sgn'] * D['gap'] * 100          # bps captured overnight
D['clr'] = (D['close_val'] - D['low_val']) / (D['high_val'] - D['low_val']).replace(0, np.nan)
D['date'] = pd.to_datetime(D['trade_date'])


def nw_t(x, lags=5):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x); mu = x.mean(); e = x - mu
    g0 = (e @ e) / n
    v = g0
    for L in range(1, lags + 1):
        gl = (e[L:] @ e[:-L]) / n
        v += 2 * (1 - L / (lags + 1)) * gl
    return mu, mu / math.sqrt(v / n), n


print("=== G1. INFERENCE: naive vs date-clustered vs Newey-West ===")
by_day = D.groupby('date')['r'].mean()
mu = D['r'].mean()
t_naive = mu / (D['r'].std(ddof=1) / math.sqrt(len(D)))
t_clust = by_day.mean() / (by_day.std(ddof=1) / math.sqrt(len(by_day)))
m_nw, t_nw, n_nw = nw_t(by_day.values, lags=5)
print(f"pooled n={len(D)} over {len(by_day)} sessions")
print(f"  mean = {mu:+.2f} bps")
print(f"  naive t              = {t_naive:+.2f}")
print(f"  date-clustered t     = {t_clust:+.2f}   <- the honest one")
print(f"  + Newey-West(5) t    = {t_nw:+.2f}")
print(f"  hit-rate = {(D['r']>0).mean()*100:.2f}%   per-session hit = {(by_day>0).mean()*100:.2f}%")
print()
for sym, g in D.groupby('sym'):
    m, t, n = nw_t(g.sort_values('date')['r'].values, lags=5)
    print(f"  {sym:11s} n={n:4d} mean={m:+6.2f}bps NW-t={t:+5.2f} hit={(g['r']>0).mean()*100:5.2f}%")

print("\n=== G2. SUB-PERIOD STABILITY ===")
D['yr'] = D['date'].dt.year
D['half'] = np.where(D['date'] < '2025-09-01', 'H1', 'H2')
for col in ['yr', 'half']:
    o = []
    for k, g in D.groupby(col):
        bd = g.groupby('date')['r'].mean()
        o.append(dict(period=k, n=len(g), sessions=len(bd), bps=g['r'].mean(),
                      t_clust=bd.mean() / (bd.std(ddof=1) / math.sqrt(len(bd))),
                      hit=(g['r'] > 0).mean() * 100))
    print(pd.DataFrame(o).round(2).to_string(index=False))

print("\n=== G3/G4. IS IT JUST MOMENTUM OR CLOSE-LOCATION? ===")
for lab, sig in [("composite sign", D['sgn']),
                 ("today's return sign", np.sign(D['pct_chg'])),
                 ("CLR>0.5 -> long", np.where(D['clr'] > 0.5, 1, -1))]:
    r = np.asarray(sig, float) * D['gap'].values * 100
    bd = pd.Series(r, index=D['date'].values).groupby(level=0).mean()
    print(f"  {lab:22s} mean={np.nanmean(r):+6.2f}bps  clustered t="
          f"{bd.mean()/(bd.std(ddof=1)/math.sqrt(len(bd))):+5.2f}  "
          f"hit={(r>0).mean()*100:5.2f}%")
print(f"\n  corr(composite, today's return)   = {D['composite'].corr(D['pct_chg']):+.3f}")
print(f"  corr(composite, CLR)              = {D['composite'].corr(D['clr']):+.3f}")
print(f"  corr(sign agreement comp vs ret)  = "
      f"{(np.sign(D['composite'])==np.sign(D['pct_chg'])).mean()*100:.1f}% of days agree")

print("\n=== G5. INCREMENTAL VALUE — OLS gap ~ composite + today's ret + CLR ===")
X = pd.DataFrame({'const': 1.0,
                  'comp_z': (D['composite'] - D['composite'].mean()) / D['composite'].std(),
                  'ret': D['pct_chg'].values,
                  'clr': D['clr'].values}).dropna()
y = D.loc[X.index, 'gap'].values * 100
Xv = X.values
beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
resid = y - Xv @ beta
# cluster-robust (by date) covariance
dates = D.loc[X.index, 'date'].values
XtX_inv = np.linalg.inv(Xv.T @ Xv)
meat = np.zeros((Xv.shape[1], Xv.shape[1]))
for d in np.unique(dates):
    m = dates == d
    u = Xv[m].T @ resid[m]
    meat += np.outer(u, u)
V = XtX_inv @ meat @ XtX_inv
se = np.sqrt(np.diag(V))
for nm, b, s in zip(X.columns, beta, se):
    print(f"  {nm:8s} beta={b:+8.3f} bps  cluster-SE={s:6.3f}  t={b/s:+6.2f}")
print("  (comp_z t is the question: does the composite survive the two controls?)")

print("\n=== G6. TRADEABILITY of the gap capture ===")
print("  buy/sell index FUTURES at the close, exit at the next open")
gross = D['r'].mean()
for cost, lab in [(2.0, "futures, tight"), (4.0, "futures, realistic"),
                  (6.0, "futures + slippage at the open")]:
    print(f"    gross {gross:+.2f}bps - {cost:.1f}bps ({lab}) = {gross-cost:+.2f} bps/night")
print(f"  per-index gross: " + "  ".join(
    f"{s}={g['r'].mean():+.1f}" for s, g in D.groupby('sym')))
print("  NOTE: an overnight index-futures position carries full margin + gap risk;")
print(f"        realised sd of the signed gap = {D['r'].std():.1f} bps "
      f"(worst night {D['r'].min():.0f} bps)")
q = D['r'].quantile([0.01, 0.05, 0.5, 0.95, 0.99])
print(f"  signed-gap quantiles bps: " + "  ".join(f"p{int(k*100)}={v:+.0f}" for k, v in q.items()))

print("\n=== G7. REALITY CHECK redone with DATE-block resampling ===")
piv = D.pivot_table(index='date', columns='sym', values='r')
cand = {'gap POOLED': by_day}
for s in piv.columns:
    cand[f'gap {s}'] = piv[s].dropna()
# rebuild the full candidate list at session level for an honest max-t
Dall = pd.read_pickle(SP + "/scored_long.pkl")
Dall['date'] = pd.to_datetime(Dall['trade_date'])
series = {}
for col in ['gap', 'intra', 'f1', 'f2', 'f3', 'f5', 'f10']:
    sub = Dall.dropna(subset=[col])
    sub = sub[np.sign(sub['composite']) != 0]
    r = np.sign(sub['composite']) * sub[col] * 100
    series[f'horizon {col}'] = pd.Series(r.values, index=sub['date'].values).groupby(level=0).mean()
for s, g in Dall.dropna(subset=['gap']).groupby('sym'):
    g = g[np.sign(g['composite']) != 0]
    series[f'gap {s}'] = pd.Series((np.sign(g['composite']) * g['gap'] * 100).values,
                                   index=g['date'].values)
idxu = sorted(set().union(*[set(v.index) for v in series.values()]))
M = pd.DataFrame({k: v.reindex(idxu) for k, v in series.items()})
M = M.dropna(how='all')
obs = {k: (M[k].mean() / (M[k].std(ddof=1) / math.sqrt(M[k].notna().sum())))
       for k in M.columns}
obs_t = max(obs.values())
best = max(obs, key=obs.get)
Mc = M - M.mean()
T = len(Mc)
B, block = 3000, 10
nulls = np.empty(B)
Mv = Mc.values
for b in range(B):
    idx = []
    while len(idx) < T:
        st = rng.integers(0, T)
        ln = rng.geometric(1 / block)
        idx.extend(((st + np.arange(ln)) % T).tolist())
    S = Mv[np.array(idx[:T])]
    with np.errstate(invalid='ignore'):
        mu_ = np.nanmean(S, axis=0)
        sd_ = np.nanstd(S, axis=0, ddof=1)
        n_ = (~np.isnan(S)).sum(axis=0)
        nulls[b] = np.nanmax(mu_ / (sd_ / np.sqrt(n_)))
p = (nulls >= obs_t).mean()
print(f"  candidates (session-level): {len(M.columns)}")
print("  " + "  ".join(f"{k}:{v:+.2f}" for k, v in sorted(obs.items(), key=lambda x: -x[1])))
print(f"\n  best = {best}  t={obs_t:.2f}")
print(f"  date-block null max-t: median {np.median(nulls):.2f}, 95th {np.percentile(nulls,95):.2f}")
print(f"  Reality Check p = {p:.4f}")
