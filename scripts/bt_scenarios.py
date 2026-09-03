"""Scenario battery over the full-era replay, with an honest multiple-testing ledger.

Every slice/transform we LOOK AT is registered as a candidate, and the final
White-style Reality Check tests the best observed t-stat against a stationary-
bootstrap null of the whole search. This is the guard against finding an edge
by searching.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np, duckdb

pd.set_option('display.width', 260)
SP = (r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
      r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad")
rng = np.random.default_rng(11)

R = pd.read_pickle(SP + "/replay_long.pkl")
err = R[R['sym'] == 'ERROR']
print(f"replay rows={len(R)}  engine exceptions={len(err)}")
if len(err):
    print(err[['trade_date', 'direction']].head(20).to_string(index=False))
R = R[R['sym'] != 'ERROR'].copy()
R['trade_date'] = pd.to_datetime(R['trade_date']).dt.date

c = duckdb.connect('data/market_data.duckdb', read_only=True)
NAMES = {"NIFTY": "Nifty 50", "BANKNIFTY": "Nifty Bank",
         "FINNIFTY": "Nifty Financial Services", "MIDCPNIFTY": "Nifty Midcap Select"}
H = {}
for s, n in NAMES.items():
    d = c.execute("SELECT trade_date, open_val, high_val, low_val, close_val, pct_chg "
                  "FROM index_data WHERE index_name=? AND trade_date>='2024-06-01' "
                  "ORDER BY trade_date", [n]).df()
    d['trade_date'] = pd.to_datetime(d['trade_date']).dt.date
    d = d.reset_index(drop=True)
    d['ma200'] = d['close_val'].rolling(200, min_periods=100).mean()
    d['rv20'] = d['close_val'].pct_change().rolling(20).std() * 100
    for h in (1, 2, 3, 5, 10):
        d[f'f{h}'] = (d['close_val'].shift(-h) / d['close_val'] - 1) * 100
    d['gap'] = (d['open_val'].shift(-1) / d['close_val'] - 1) * 100
    d['intra'] = (d['close_val'].shift(-1) / d['open_val'].shift(-1) - 1) * 100
    d['sym'] = s
    H[s] = d
vix = c.execute("SELECT trade_date, close_val vix FROM index_data "
                "WHERE index_name='India VIX' AND trade_date>='2024-06-01'").df()
vix['trade_date'] = pd.to_datetime(vix['trade_date']).dt.date

F = pd.concat(H.values())
D = R.merge(F, on=['trade_date', 'sym'], how='left').merge(vix, on='trade_date', how='left')
D['above200'] = D['close_val'] > D['ma200']
D['dow'] = pd.to_datetime(D['trade_date']).dt.dayofweek
D = D.dropna(subset=['f1']).copy()
print(f"scored index-days={len(D)}  span {D.trade_date.min()} .. {D.trade_date.max()}")
D.to_pickle(SP + "/scored_long.pkl")


def stats_of(r):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 5:
        return dict(n=n, mean=np.nan, t=np.nan, hit=np.nan)
    se = r.std(ddof=1) / math.sqrt(n)
    return dict(n=n, mean=r.mean(), t=r.mean() / se if se > 0 else np.nan,
                hit=(r > 0).mean() * 100)


CAND = []


def add(label, sub, sig, col='f1'):
    r = (np.asarray(sig, float) * sub[col].values) * 100
    r = r[~np.isnan(r)]
    if len(r) >= 20:
        CAND.append((label, r))


print("\n=== S1. FULL-ERA HEADLINE (engine's own verdict) ===")
D['act_up'] = D['f1'] > 0
rows = []
for sym, g in list(D.groupby('sym')) + [('POOLED', D)]:
    d = g[g['direction'].isin(['UP', 'DOWN'])]
    sgn = np.where(d['direction'] == 'UP', 1, -1)
    st = stats_of(sgn * d['f1'] * 100)
    nc = g['close_val'] * (1 + g['f1'] / 100)
    rows.append(dict(sym=sym, n=len(g), n_dir=len(d), dir_share=len(d) / len(g) * 100,
                     sign_hit=((sgn > 0) == d['act_up']).mean() * 100 if len(d) else np.nan,
                     bps=st['mean'], t=st['t'],
                     IC=g['composite'].rank().corr(g['f1'].rank()),
                     cov1s=((nc >= g['rlo']) & (nc <= g['rhi'])).mean() * 100))
    add(f"S1 dir-calls {sym}", d, sgn)
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n=== S2. HORIZON SCAN (sign of composite, all days) ===")
out = []
for col, lab in [('gap', 'overnight gap'), ('intra', 'next-day open->close'),
                 ('f1', 'close->close t+1'), ('f2', 't+2'), ('f3', 't+3'),
                 ('f5', 't+5'), ('f10', 't+10')]:
    sub = D.dropna(subset=[col])
    sub = sub[np.sign(sub['composite']) != 0]
    sg = np.sign(sub['composite']).values
    st = stats_of(sg * sub[col] * 100)
    out.append(dict(horizon=lab, n=st['n'], bps=st['mean'], t=st['t'], hit=st['hit'],
                    IC=sub['composite'].rank().corr(sub[col].rank())))
    add(f"S2 horizon {lab}", sub, sg, col)
print(pd.DataFrame(out).round(3).to_string(index=False))

print("\n=== S3. OVERNIGHT GAP x INDEX (only horizon with prior evidence) ===")
o = []
for sym, g in D.groupby('sym'):
    g = g.dropna(subset=['gap'])
    g = g[np.sign(g['composite']) != 0]
    sg = np.sign(g['composite']).values
    st = stats_of(sg * g['gap'] * 100)
    o.append(dict(sym=sym, **st, IC=g['composite'].rank().corr(g['gap'].rank())))
    add(f"S3 gap {sym}", g, sg, 'gap')
print(pd.DataFrame(o).round(3).to_string(index=False))

print("\n=== S4. TRANSFORMS of the composite (t+1) ===")
D['z'] = D.groupby('sym')['composite'].transform(lambda s: (s - s.mean()) / s.std())
D['dcomp'] = D.sort_values('trade_date').groupby('sym')['composite'].diff()
o = []
specs = [("raw sign", D, lambda x: np.sign(x['composite'])),
         ("FADE (contrarian)", D, lambda x: -np.sign(x['composite'])),
         ("z>1 only", D[D['z'].abs() > 1], lambda x: np.sign(x['z'])),
         ("composite CHANGE", D.dropna(subset=['dcomp']), lambda x: np.sign(x['dcomp']))]
for lab, sub, fn in specs:
    sg = fn(sub).values
    m = sg != 0
    sub2, sg2 = sub[m], sg[m]
    st = stats_of(sg2 * sub2['f1'] * 100)
    o.append(dict(transform=lab, **st))
    add(f"S4 {lab}", sub2, sg2)
w = (D['composite'] / D['composite'].abs().max()).values
o.append(dict(transform="conviction-weighted", **stats_of(w * D['f1'] * 100)))
print(pd.DataFrame(o).round(3).to_string(index=False))

print("\n=== S5. FAMILY-AGREEMENT filter ===")
VOTE = {'Price Action', 'Institutional', 'Statistical Regime', 'Options OI', 'Futures OI'}


def agree(sig):
    d = {}
    for name, cat, score in sig:
        d[cat] = d.get(cat, 0) + score
    return (sum(1 for k, v in d.items() if k in VOTE and v > 0)
            - sum(1 for k, v in d.items() if k in VOTE and v < 0))


D['n_agree'] = [agree(s) for s in D['sig']]
o = []
for k in [1, 2, 3, 4]:
    sub = D[D['n_agree'].abs() >= k]
    sg = np.sign(sub['n_agree']).values
    st = stats_of(sg * sub['f1'] * 100)
    o.append(dict(min_families=k, **st))
    add(f"S5 agree>={k}", sub, sg)
print(pd.DataFrame(o).round(3).to_string(index=False))

print("\n=== S6. REGIME CONDITIONING (t+1, composite sign) ===")
D['vix_b'] = pd.qcut(D['vix'], 3, labels=['lowVIX', 'midVIX', 'highVIX'])
D['rv_b'] = pd.qcut(D['rv20'], 3, labels=['lowRV', 'midRV', 'highRV'])
D['dte_b'] = pd.cut(D['dte'], [-1, 1, 3, 7, 60], labels=['dte<=1', 'dte2-3', 'dte4-7', 'dte8+'])
o = []
for col in ['vix_b', 'rv_b', 'dte_b', 'above200', 'dow']:
    for k, g in D.groupby(col, observed=True):
        g = g[np.sign(g['composite']) != 0]
        sg = np.sign(g['composite']).values
        st = stats_of(sg * g['f1'] * 100)
        o.append(dict(slice=f"{col}={k}", **st))
        add(f"S6 {col}={k}", g, sg)
print(pd.DataFrame(o).round(2).to_string(index=False))

print("\n=== S7. STABILITY: halves / years ===")
D['half'] = np.where(pd.to_datetime(D['trade_date']) < pd.Timestamp('2025-09-01'), 'H1', 'H2')
D['yr'] = pd.to_datetime(D['trade_date']).dt.year
for col in ['half', 'yr']:
    o = []
    for k, g in D.groupby(col):
        d = g[g['direction'].isin(['UP', 'DOWN'])]
        sgn = np.where(d['direction'] == 'UP', 1, -1)
        st = stats_of(sgn * d['f1'] * 100)
        o.append(dict(period=k, n_dir=st['n'], sign_hit=st['hit'], bps=st['mean'],
                      t=st['t'], IC_all=g['composite'].rank().corr(g['f1'].rank())))
    print(pd.DataFrame(o).round(3).to_string(index=False))

print("\n=== S8. THIN-CHAIN / structural subsets ===")
if 'liquid' in D.columns:
    for k, g in D.groupby('liquid'):
        d = g[g['direction'].isin(['UP', 'DOWN'])]
        sgn = np.where(d['direction'] == 'UP', 1, -1)
        nc = g['close_val'] * (1 + g['f1'] / 100)
        hit = ((sgn > 0) == (d['f1'] > 0)).mean() * 100 if len(d) else float('nan')
        print(f"opt_chain_liquid={k}: n={len(g)} n_dir={len(d)} sign={hit:.1f}% "
              f"cov1s={((nc >= g['rlo']) & (nc <= g['rhi'])).mean() * 100:.1f}% "
              f"idx={sorted(g['sym'].unique())}")

print("\n=== S9. RANGE CALIBRATION, full era ===")
D['nc'] = D['close_val'] * (1 + D['f1'] / 100)
for sym, g in list(D.groupby('sym')) + [('POOLED', D)]:
    ins = (g['nc'] >= g['rlo']) & (g['nc'] <= g['rhi'])
    n, p = len(g), ins.mean()
    se = math.sqrt(p * (1 - p) / n)
    print(f"{sym:11s} cov={p*100:5.1f}% n={n:5d} "
          f"95%CI=[{(p-1.96*se)*100:.1f},{(p+1.96*se)*100:.1f}] z_vs68={(p-0.68)/se:+.2f}")
print("\nper index x year:")
piv = D.assign(ins=(D['nc'] >= D['rlo']) & (D['nc'] <= D['rhi'])).pivot_table(
    index='sym', columns='yr', values='ins', aggfunc='mean') * 100
print(piv.round(1).to_string())

print("\n=== S10. POWER — what could this test even detect? ===")
for n in [20, 39, 100, 200, 400, 1000, 1970]:
    se = 0.5 / math.sqrt(n)
    print(f"n={n:5d}: 80%-power min detectable hit-rate = {50 + (1.96+0.84)*se*100:.1f}% "
          f"(SE={se*100:.2f}pp)")

print("\n=== S11. REALITY CHECK — max-t over the whole search ===")
print(f"candidates examined: {len(CAND)}")
tab = [dict(candidate=l, **stats_of(r)) for l, r in CAND]
T = pd.DataFrame(tab).sort_values('t', ascending=False)
print(T.round(2).to_string(index=False))
L = min(len(r) for _, r in CAND)
Mx = np.vstack([r[:L] for _, r in CAND])
Mc = Mx - Mx.mean(axis=1, keepdims=True)
obs_t = np.nanmax([stats_of(r)['t'] for _, r in CAND])
B, block = 2000, 5
nulls = np.empty(B)
for b in range(B):
    idx = []
    while len(idx) < L:
        st_ = rng.integers(0, L)
        ln = rng.geometric(1 / block)
        idx.extend(((st_ + np.arange(ln)) % L).tolist())
    S = Mc[:, np.array(idx[:L])]
    se = S.std(axis=1, ddof=1) / math.sqrt(L)
    nulls[b] = np.nanmax(np.where(se > 0, S.mean(axis=1) / se, np.nan))
p_rc = (nulls >= obs_t).mean()
print(f"\nbest observed t across all {len(CAND)} candidates = {obs_t:.2f}")
print(f"bootstrap null max-t: median {np.median(nulls):.2f}, 95th pct {np.percentile(nulls,95):.2f}")
print(f"White Reality Check p = {p_rc:.3f}  -> "
      f"{'NOTHING survives the search cost' if p_rc > 0.10 else 'SURVIVES, investigate'}")

print("\n=== S12. EFFECTIVE SAMPLE SIZE — the 4 indices are not 4 independent tests ===")
W = D.pivot_table(index='trade_date', columns='sym', values='f1')
C = W.corr()
print(C.round(3).to_string())
k = C.shape[0]
rho = (C.values.sum() - k) / (k * (k - 1))
n_eff_per_day = k / (1 + (k - 1) * rho)
print(f"\nmean pairwise corr of daily index returns = {rho:.3f}")
print(f"effective independent series per day = {n_eff_per_day:.2f} (not {k})")
sessions = W.shape[0]
print(f"=> {len(D)} index-days is really ~{sessions*n_eff_per_day:.0f} independent observations")
print(f"   pooled t-stats above are inflated by ~sqrt({k}/{n_eff_per_day:.2f}) "
      f"= {(k/n_eff_per_day)**0.5:.2f}x")

print("\n=== S13. DATE-CLUSTERED SE on the pooled directional test ===")
d = D[D['direction'].isin(['UP', 'DOWN'])].copy()
d['r'] = np.where(d['direction'] == 'UP', 1, -1) * d['f1'] * 100
by_day = d.groupby('trade_date')['r'].mean()
mu = d['r'].mean()
se_naive = d['r'].std(ddof=1) / math.sqrt(len(d))
se_clust = by_day.std(ddof=1) / math.sqrt(len(by_day))
print(f"n_calls={len(d)} over {len(by_day)} distinct sessions")
print(f"mean={mu:+.2f}bps  naive t={mu/se_naive:+.2f}  date-clustered t={by_day.mean()/se_clust:+.2f}")

print("\n=== S14. HOW LONG TO VALIDATE? sample needed to prove a real edge ===")
call_rate = (D['direction'].isin(['UP', 'DOWN'])).mean()
for target in [52, 53, 55, 57, 60]:
    p = target / 100
    n_need = ((1.96 * 0.5 + 0.84 * math.sqrt(p * (1 - p))) / (p - 0.5)) ** 2
    sessions_need = n_need / (n_eff_per_day * call_rate)
    print(f"  to prove a {target}% hit-rate at 80% power: n={n_need:.0f} calls "
          f"-> {sessions_need:.0f} sessions ({sessions_need/250:.1f} years) "
          f"at the current {call_rate*100:.1f}% call-rate")

print("\n=== S15. OPTION-BUYER TRANSLATION of the measured edge ===")
d2 = D[D['direction'].isin(['UP', 'DOWN'])]
gross = (np.where(d2['direction'] == 'UP', 1, -1) * d2['f1']).mean() * 100
print(f"measured gross index move in the called direction: {gross:+.2f} bps of spot")
for prem_pct, delta in [(0.45, 0.50), (0.70, 0.50), (1.10, 0.50)]:
    pnl = gross * delta                      # bps of spot captured by the option
    cost = prem_pct * 100 * 0.05             # ~5% round-trip friction on the premium
    theta = prem_pct * 100 * 0.20            # ~1 day of theta on a weekly ATM
    print(f"  ATM premium {prem_pct:.2f}% of spot, delta {delta}: "
          f"capture {pnl:+.1f}bps  friction -{cost:.1f}bps  1-day theta -{theta:.1f}bps "
          f"=> net {pnl-cost-theta:+.1f} bps of spot")
print("  (an index-futures trade avoids theta but still pays ~2-6 bps round trip)")
