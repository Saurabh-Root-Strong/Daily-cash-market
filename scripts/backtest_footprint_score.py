"""Score the Operator Footprint agreement/collision matrix. Run part 1 first."""
from __future__ import annotations
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

pd.set_option('display.width', 260)
SP = (r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
      r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad")
rng = np.random.default_rng(11)

opt = pd.read_pickle(SP + '/fp_opt_events.pkl')
fut = pd.read_pickle(SP + '/fp_fut.pkl')
cash = pd.read_pickle(SP + '/fp_cash.pkl')
FWD = {h: pd.read_pickle(SP + f'/fp_fwd{h}.pkl') for h in (1, 3, 5, 10)}

# ── per symbol-day option summary ────────────────────────────────────────────
opt['w_lean'] = opt['add_cr'] * opt['opt_lean']
g = opt.groupby(['trade_date', 'symbol'])
osum = g.agg(add_cr=('add_cr', 'sum'), w=('w_lean', 'sum'),
             n_events=('add_cr', 'size')).reset_index()
osum['opt_net'] = osum['w'] / osum['add_cr']
# dominant strike = the single largest money-add that day (where size showed up)
dom = opt.loc[g['add_cr'].idxmax(), ['trade_date', 'symbol', 'moneyness',
                                     'opt_action', 'opt_lean', 'add_cr',
                                     'option_type']]
dom = dom.rename(columns={'moneyness': 'dom_money', 'opt_action': 'dom_action',
                          'opt_lean': 'dom_lean', 'add_cr': 'dom_add_cr',
                          'option_type': 'dom_type'})
osum = osum.merge(dom, on=['trade_date', 'symbol'], how='left')

D = fut[['trade_date', 'symbol', 'fut_action', 'fut_lean', 'fut_oi_pct']].merge(
    osum, on=['trade_date', 'symbol'], how='inner')
D = D.merge(cash[['trade_date', 'symbol', 'ret_pct']], on=['trade_date', 'symbol'],
            how='left')
D['opt_sign'] = np.sign(D['opt_net'])
D = D[D['opt_sign'] != 0].copy()
for h in FWD:
    s = FWD[h].stack().rename(f'f{h}').reset_index()
    s.columns = ['trade_date', 'symbol', f'f{h}']
    D = D.merge(s, on=['trade_date', 'symbol'], how='left')
D = D.dropna(subset=['f3'])
D['agree'] = np.where(D['fut_lean'] == D['opt_sign'],
                      np.where(D['fut_lean'] > 0, 'AGREE BULL', 'AGREE BEAR'), 'COLLIDE')
D['pred'] = np.where(D['agree'] == 'COLLIDE', 0, D['fut_lean'])
print(f"panel: {len(D):,} symbol-days with BOTH a futures label and an option event")
print(f"       {D['trade_date'].nunique()} sessions, {D['symbol'].nunique()} symbols, "
      f"{D['trade_date'].min():%Y-%m-%d} .. {D['trade_date'].max():%Y-%m-%d}")
D.to_pickle(SP + '/fp_panel.pkl')


def nw_t(per_date: pd.Series, lag: int):
    x = per_date.dropna().values.astype(float)
    n = len(x)
    if n < 25: return float('nan')
    e = x - x.mean(); v = (e @ e) / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / math.sqrt(v / n) if v > 0 else float('nan')


def cell(df, col, lag, signed=None):
    """mean forward excess (pp), date-clustered Newey-West t, hit-rate."""
    if df.empty: return dict(n=0, pp=np.nan, t=np.nan, hit=np.nan)
    v = df[col] if signed is None else df[col] * signed
    pd_ = v.groupby(df['trade_date']).mean()
    return dict(n=len(df), pp=float(v.mean()), t=nw_t(pd_, lag),
                hit=float((v > 0).mean() * 100))


print("\n" + "=" * 100)
print("A. THE HEADLINE — futures x options AGREEMENT, forward EXCESS over the F&O universe")
print("=" * 100)
rows = []
for h, lag in [(1, 1), (3, 3), (5, 5), (10, 10)]:
    for lab in ['AGREE BULL', 'AGREE BEAR', 'COLLIDE']:
        s = D[D['agree'] == lab]
        r = cell(s, f'f{h}', lag)
        rows.append(dict(horizon=f'{h}d', cell=lab, **r))
    # the tradeable version: long AGREE BULL, short AGREE BEAR
    s = D[D['agree'] != 'COLLIDE']
    r = cell(s, f'f{h}', lag, signed=s['pred'])
    rows.append(dict(horizon=f'{h}d', cell='LONG/SHORT (signed)', **r))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nRANDOM CONTROL (same cell sizes, random symbol-days) — must sit at ~0:")
rc = []
for h, lag in [(3, 3)]:
    for lab in ['AGREE BULL', 'AGREE BEAR', 'COLLIDE']:
        k = (D['agree'] == lab).sum()
        samp = D.sample(k, random_state=5)
        rc.append(dict(horizon=f'{h}d', cell=f'RANDOM n={k}', **cell(samp, f'f{h}', lag)))
print(pd.DataFrame(rc).round(3).to_string(index=False))

print("\n" + "=" * 100)
print("B. FULL 4x2 MATRIX — futures action x options net lean (3-day forward excess)")
print("=" * 100)
rows = []
for fa in ['LONG BUILDUP', 'SHORT COVERING', 'SHORT BUILDUP', 'LONG UNWINDING']:
    for ol, oln in [(1, 'options BULLISH'), (-1, 'options BEARISH')]:
        s = D[(D['fut_action'] == fa) & (D['opt_sign'] == ol)]
        rows.append(dict(futures=fa, options=oln, **cell(s, 'f3', 3)))
M = pd.DataFrame(rows)
print(M.round(3).to_string(index=False))

print("\n" + "=" * 100)
print("C. MONEYNESS — where the size showed up (dominant strike), 3-day forward excess")
print("=" * 100)
order = ['deep ITM', 'ITM', 'ATM', 'OTM', 'deep OTM']
rows = []
for m in order:
    for lab in ['AGREE BULL', 'AGREE BEAR', 'COLLIDE']:
        s = D[(D['dom_money'] == m) & (D['agree'] == lab)]
        rows.append(dict(moneyness=m, cell=lab, **cell(s, 'f3', 3)))
    s = D[(D['dom_money'] == m) & (D['agree'] != 'COLLIDE')]
    rows.append(dict(moneyness=m, cell='LONG/SHORT (signed)',
                     **cell(s, 'f3', 3, signed=s['pred'])))
C = pd.DataFrame(rows)
print(C.round(3).to_string(index=False))
print("\nevent share by moneyness (where size actually shows up):")
print((D['dom_money'].value_counts(normalize=True).reindex(order) * 100).round(1).to_string())

print("\n" + "=" * 100)
print("D. THE CONTROL THAT MATTERS — is this just today's price move?")
print("=" * 100)
print(f"  agreement sign vs today's return sign: "
      f"{(np.sign(D['ret_pct']) == D['pred']).mean()*100:.1f}% of AGREE days match")
print(f"  corr(pred, today's return) = {D['pred'].corr(D['ret_pct']):+.3f}")
sub = D[D['agree'] != 'COLLIDE'].copy().reset_index(drop=True)
print(f"\n  raw signed 3d excess          : {cell(sub,'f3',3,signed=sub['pred'])}")
mom = np.sign(sub['ret_pct'])
print(f"  today's-return sign alone (3d): {cell(sub,'f3',3,signed=mom)}")
# OLS f3 ~ pred + today's return, date-clustered SE
X = pd.DataFrame({'const': 1.0, 'pred': sub['pred'].values,
                  'ret': sub['ret_pct'].values}).dropna()
y = sub.loc[X.index, 'f3'].values
Xv = X.values
beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
res = y - Xv @ beta
dts = sub.loc[X.index, 'trade_date'].values
XtXi = np.linalg.inv(Xv.T @ Xv); meat = np.zeros((3, 3))
for d in np.unique(dts):
    m = dts == d
    u = Xv[m].T @ res[m]
    meat += np.outer(u, u)
se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
print("\n  OLS  f3 ~ pred + today's return   (date-clustered SE)")
for nm, b, s in zip(X.columns, beta, se):
    print(f"    {nm:6s} beta={b:+7.4f}pp  SE={s:6.4f}  t={b/s:+6.2f}")

print("\n" + "=" * 100)
print("E. EDGE CASES / ROBUSTNESS")
print("=" * 100)
D['yr'] = D['trade_date'].dt.year
print("-- by era (signed 3d, AGREE only):")
for k, s in sub.groupby(sub['trade_date'].dt.year):
    print(f"   {k}: {cell(s,'f3',3,signed=s['pred'])}")
print("-- by conviction (|options net lean| decile):")
sub2 = sub.copy(); sub2['q'] = pd.qcut(sub2['opt_net'].abs(), 4, labels=False, duplicates='drop')
for k, s in sub2.groupby('q'):
    print(f"   |opt_net| Q{int(k)+1}: {cell(s,'f3',3,signed=s['pred'])}")
print("-- by futures OI move size (|fut_oi_pct| quartile):")
sub2['fq'] = pd.qcut(sub2['fut_oi_pct'].abs(), 4, labels=False, duplicates='drop')
for k, s in sub2.groupby('fq'):
    print(f"   |fut OI %| Q{int(k)+1}: {cell(s,'f3',3,signed=s['pred'])}")
print("-- CE-driven vs PE-driven dominant strike:")
for k, s in sub.groupby('dom_type'):
    print(f"   dom {k}: {cell(s,'f3',3,signed=s['pred'])}")
print("-- number of option events that day (breadth of the footprint):")
sub2['nq'] = pd.cut(sub2['n_events'], [0, 1, 2, 4, 999], labels=['1', '2', '3-4', '5+'])
for k, s in sub2.groupby('nq', observed=True):
    print(f"   {k} events: {cell(s,'f3',3,signed=s['pred'])}")

print("\n" + "=" * 100)
print("F. MULTIPLE TESTING — every cell tested, FDR + max-t reality check")
print("=" * 100)
cands = {}
for h, lag in [(1, 1), (3, 3), (5, 5), (10, 10)]:
    s = D[D['agree'] != 'COLLIDE']
    cands[f'LS {h}d'] = (s[f'f{h}'] * s['pred']).groupby(s['trade_date']).mean()
    for m in order:
        ss = s[s['dom_money'] == m]
        if len(ss) > 200:
            cands[f'LS {h}d {m}'] = (ss[f'f{h}'] * ss['pred']).groupby(ss['trade_date']).mean()
for lab in ['AGREE BULL', 'AGREE BEAR', 'COLLIDE']:
    s = D[D['agree'] == lab]
    cands[f'{lab} 3d'] = s['f3'].groupby(s['trade_date']).mean()
tt = {k: nw_t(v, 3) for k, v in cands.items()}
res = pd.DataFrame({'cand': list(tt), 't': list(tt.values())}).sort_values('t', key=abs,
                                                                          ascending=False)
print(res.round(2).to_string(index=False))
idxu = sorted(set().union(*[set(v.index) for v in cands.values()]))
Mx = pd.DataFrame({k: v.reindex(idxu) for k, v in cands.items()})
Mc = Mx - Mx.mean()
T = len(Mc); Mv = Mc.values
obs = np.nanmax(np.abs([tt[k] for k in Mx.columns]))
B, block = 2000, 5
nulls = np.empty(B)
for b in range(B):
    idx = []
    while len(idx) < T:
        st = rng.integers(0, T); ln = rng.geometric(1 / block)
        idx.extend(((st + np.arange(ln)) % T).tolist())
    S = Mv[np.array(idx[:T])]
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(S, axis=0); sd = np.nanstd(S, axis=0, ddof=1)
        n_ = (~np.isnan(S)).sum(axis=0)
        nulls[b] = np.nanmax(np.abs(mu / (sd / np.sqrt(n_))))
print(f"\n  candidates={len(Mx.columns)}  best |t|={obs:.2f}")
print(f"  date-block null max|t|: median {np.median(nulls):.2f}, 95th {np.percentile(nulls,95):.2f}")
print(f"  Reality Check p = {(nulls >= obs).mean():.4f}")

print("\n" + "=" * 100)
print("G. COST REALITY — single-stock cash, 3-day hold")
print("=" * 100)
best = cell(sub, 'f3', 3, signed=sub['pred'])
print(f"  best gross signed excess at 3d = {best['pp']:+.3f}pp = {best['pp']*100:+.1f} bps")
for c, lab in [(12, 'BTST/intraday-style, liquid large cap'),
               (22, 'delivery incl STT'), (40, 'realistic all-in retail single stock')]:
    print(f"    net of {c} bps round trip ({lab}): {best['pp']*100-c:+.1f} bps")
