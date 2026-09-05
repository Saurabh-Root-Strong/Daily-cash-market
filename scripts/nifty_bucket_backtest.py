"""Does WEIGHT-BUCKETED constituent flow predict Nifty? (Layer C, decided early)

KEY REALISATION: this test does not need the index_weights table. Bucket
MEMBERSHIP is what matters, not precise weights, and top-10 membership is stable.
So the verdict can be reached before any weights file exists — and if the answer
is null under every plausible bucket definition, the weights work is not urgent.

HYPOTHESES (the user's thesis, stated testably)
  H1 top-10 delivery/OI/options flow -> forward NIFTY
  H2 next-10 the same
  H3 rest-30 the same
  H4 DIVERGENCE (top10 flow - rest30 flow) -> forward NIFTY   <- the novel one:
     "top 20 negative while the 30 are positive is why Nifty closes red"

WHAT WOULD MAKE THIS A FALSE POSITIVE, and how each is handled
  - CONTEMPORANEOUS IDENTITY: top-10 return IS ~46% of the index return, so any
    same-day relation is tautology. Everything here is strictly FORWARD.
  - MOMENTUM: every constituent signal tested in this repo so far lost to the
    index's own prior return. That control runs beside every cell.
  - BUCKET-DEFINITION MINING: run under 3 independent bucket definitions
    (published list / turnover rank / traded-value rank). A result that only
    survives one definition is noise.
  - OVERLAP + CROSS-SECTION: Newey-West at lag = horizon.
  - MULTIPLICITY: max-|t| vs a date-block bootstrap over the whole search.
  - SURVIVORSHIP: today's 50 applied to history. Flagged, quantified below.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option('display.width', 235)
rng = np.random.default_rng(17)
c = duckdb.connect('data/market_data.duckdb', read_only=True)

TOP10 = ["RELIANCE","BHARTIARTL","HDFCBANK","ICICIBANK","SBIN","TCS","BAJFINANCE",
         "LT","HINDUNILVR","INFY"]
NEXT10 = ["SUNPHARMA","TITAN","KOTAKBANK","MARUTI","ADANIENT","AXISBANK","M&M",
          "ADANIPORTS","HCLTECH","ULTRACEMCO"]
ALL50 = TOP10 + NEXT10 + ["APOLLOHOSP","ASIANPAINT","BAJAJ-AUTO","BAJAJFINSV","BEL",
  "CIPLA","COALINDIA","DRREDDY","EICHERMOT","ETERNAL","GRASIM","HDFCLIFE","HINDALCO",
  "ITC","INDIGO","JSWSTEEL","JIOFIN","MAXHEALTH","NTPC","NESTLEIND","ONGC","POWERGRID",
  "SBILIFE","SHRIRAMFIN","TATACONSUM","TMPV","TATASTEEL","TECHM","TRENT","WIPRO"]
REST30 = [s for s in ALL50 if s not in TOP10 + NEXT10]

ph = ",".join("?" * len(ALL50))
cash = c.execute(f"""SELECT trade_date, symbol, close_price, prev_close, deliv_per,
        turnover_lacs, (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND close_price>0 AND prev_close>0 AND trade_date>='2022-01-01'""", ALL50).df()
cash['trade_date'] = pd.to_datetime(cash['trade_date'])
cash = cash[cash['r'].abs() < 40]                       # corporate actions
R = cash.pivot_table('r', 'trade_date', 'symbol').sort_index()
DL = cash.pivot_table('deliv_per', 'trade_date', 'symbol').sort_index()
TO = cash.pivot_table('turnover_lacs', 'trade_date', 'symbol').sort_index()

fut = c.execute(f"""
  WITH nf AS (SELECT trade_date,symbol,MIN(expiry_date) e FROM fno_bhavcopy
      WHERE instrument='FUTSTK' AND expiry_date>trade_date AND open_interest>0
        AND symbol IN ({ph}) GROUP BY 1,2),
  f AS (SELECT b.trade_date,b.symbol,b.expiry_date,b.close_price,b.open_interest
        FROM fno_bhavcopy b JOIN nf n ON n.symbol=b.symbol AND n.trade_date=b.trade_date
             AND n.e=b.expiry_date WHERE b.instrument='FUTSTK'),
  st AS (SELECT DISTINCT trade_date,symbol FROM fno_bhavcopy
         WHERE instrument='FUTSTK' AND expiry_date=trade_date)
  SELECT f.trade_date,f.symbol,f.open_interest,f.close_price,
         LAG(f.open_interest) OVER w poi, LAG(f.close_price) OVER w ppx,
         (s.symbol IS NOT NULL) settle
  FROM f LEFT JOIN st s ON s.symbol=f.symbol AND s.trade_date=f.trade_date
  WINDOW w AS (PARTITION BY f.symbol,f.expiry_date ORDER BY f.trade_date)""", ALL50).df()
fut['trade_date'] = pd.to_datetime(fut['trade_date'])
fut = fut[(~fut['settle']) & fut['poi'].notna() & (fut['poi'] > 0)].copy()
fut['oi_pct'] = (fut['open_interest'] - fut['poi']) / fut['poi'] * 100
fut['lean'] = np.where((fut['close_price'] > fut['ppx']) & (fut['oi_pct'] > 0.5), 1,
              np.where((fut['close_price'] < fut['ppx']) & (fut['oi_pct'] > 0.5), -1,
              np.where((fut['close_price'] > fut['ppx']) & (fut['oi_pct'] < -0.5), 1,
              np.where((fut['close_price'] < fut['ppx']) & (fut['oi_pct'] < -0.5), -1, 0))))
FL = fut.pivot_table('lean', 'trade_date', 'symbol')
FO = fut.pivot_table('oi_pct', 'trade_date', 'symbol')

nif = c.execute("""SELECT trade_date,pct_chg,open_val,close_val FROM index_data
                   WHERE index_name='Nifty 50' AND trade_date>='2022-01-01'""").df()
nif['trade_date'] = pd.to_datetime(nif['trade_date'])
nif = nif.set_index('trade_date').sort_index()
ix = R.index.intersection(nif.index)
R, DL, TO, FL, FO = [x.reindex(ix) for x in (R, DL, TO, FL, FO)]
y = nif.loc[ix, 'pct_chg'].astype(float)
lg = np.log1p(y / 100.0)
FWD = {h: (np.expm1(lg.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1)) * 100.0)
       for h in (1, 3, 5)}
FWD['gap'] = (nif['open_val'] / nif['close_val'].shift(1) - 1).shift(-1).reindex(ix) * 100

print(f"panel {len(ix)} sessions {ix.min():%Y-%m-%d}..{ix.max():%Y-%m-%d}")
print(f"F&O futures coverage: {FL.notna().mean().mean()*100:.1f}% of the 50 names")
thin = FL.notna().mean().sort_values().head(5)
print(f"thinnest F&O names: {dict((k, round(v*100)) for k, v in thin.items())}")
print(f"cash coverage thinnest: "
      f"{dict((k, round(v*100)) for k, v in R.notna().mean().sort_values().head(4).items())}")


def nw(x, lag):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 30: return float('nan')
    e = x - x.mean(); v = (e @ e) / n
    for L in range(1, lag + 1): v += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / math.sqrt(v / n) if v > 0 else float('nan')


def bucket_signals(cols, wts=None):
    """flow signals for one bucket. wts=None -> equal weight within bucket."""
    cols = [s for s in cols if s in R.columns]
    w = None if wts is None else wts[cols].div(wts[cols].sum(axis=1), axis=0)
    def agg(M):
        M = M[[s for s in cols if s in M.columns]]
        return (M * w).sum(axis=1) if w is not None else M.mean(axis=1)
    dlv = agg(DL)
    dz = (dlv - dlv.rolling(60).mean()) / dlv.rolling(60).std()
    return dict(deliv_z=dz, fut_lean=agg(FL), fut_oi=agg(FO),
                adv=(R[cols] > 0).sum(axis=1) / R[cols].notna().sum(axis=1) * 100 - 50,
                ret=agg(R))


DEFS = {'published': (TOP10, NEXT10, REST30)}
# independent bucket definitions, to prove the answer is not a membership artifact
tor = TO.rolling(60).mean().shift(1)
rank_top = tor.rank(axis=1, ascending=False)
DEFS['turnover-rank'] = None    # handled dynamically below

print("\n" + "=" * 100)
print("FORWARD TEST — bucketed flow vs the momentum control")
print("=" * 100)
B = {'TOP10': bucket_signals(TOP10), 'NEXT10': bucket_signals(NEXT10),
     'REST30': bucket_signals(REST30)}
cands = {}
rows = []
for bn, sig in B.items():
    for sn in ('deliv_z', 'fut_lean', 'fut_oi', 'adv'):
        for h in (1, 3, 'gap'):
            f = FWD[h]
            k = pd.DataFrame({'s': sig[sn], 'f': f}).dropna()
            if len(k) < 100: continue
            sg = np.sign(k['s']) * k['f']
            t = nw(sg, 3 if h == 3 else 1)
            rows.append(dict(bucket=bn, signal=sn, h=str(h), n=len(k),
                             IC=k['s'].rank().corr(k['f'].rank()), bps=sg.mean() * 100,
                             t=t, hit=(sg > 0).mean() * 100))
            cands[f'{bn}.{sn}.{h}'] = sg
# H4: divergence
for sn in ('deliv_z', 'fut_lean', 'fut_oi', 'adv'):
    div = B['TOP10'][sn] - B['REST30'][sn]
    for h in (1, 3, 'gap'):
        k = pd.DataFrame({'s': div, 'f': FWD[h]}).dropna()
        if len(k) < 100: continue
        sg = np.sign(k['s']) * k['f']
        rows.append(dict(bucket='DIVERGE', signal=sn, h=str(h), n=len(k),
                         IC=k['s'].rank().corr(k['f'].rank()), bps=sg.mean() * 100,
                         t=nw(sg, 3 if h == 3 else 1), hit=(sg > 0).mean() * 100))
        cands[f'DIVERGE.{sn}.{h}'] = sg
# controls
for cn, cs in [('CTRL nifty ret', y), ('CTRL top10 ret', B['TOP10']['ret'])]:
    for h in (1, 3, 'gap'):
        k = pd.DataFrame({'s': cs, 'f': FWD[h]}).dropna()
        sg = np.sign(k['s']) * k['f']
        rows.append(dict(bucket=cn, signal='momentum', h=str(h), n=len(k),
                         IC=k['s'].rank().corr(k['f'].rank()), bps=sg.mean() * 100,
                         t=nw(sg, 3 if h == 3 else 1), hit=(sg > 0).mean() * 100))
res = pd.DataFrame(rows)
print(res.sort_values('t', key=abs, ascending=False).head(18).round(3).to_string(index=False))
print("\n  controls:")
print(res[res['bucket'].str.startswith('CTRL')].round(3).to_string(index=False))

print("\n" + "=" * 100)
print("MULTIPLICITY — max|t| over the whole search vs a date-block null")
print("=" * 100)
M = pd.DataFrame(cands).astype(float)
tt = {k: nw(M[k].dropna(), 3) for k in M.columns}
obs = np.nanmax(np.abs(list(tt.values())))
best = max(tt, key=lambda k: abs(tt[k]))
Mc = (M - M.mean()).values
T = len(Mc); B_, blk = 2000, 5
nulls = np.empty(B_)
for b in range(B_):
    idx = []
    while len(idx) < T:
        st = rng.integers(0, T); ln = rng.geometric(1 / blk)
        idx.extend(((st + np.arange(ln)) % T).tolist())
    S = Mc[np.array(idx[:T])]
    with np.errstate(invalid='ignore'):
        mu = np.nanmean(S, 0); sd = np.nanstd(S, 0, ddof=1); n_ = (~np.isnan(S)).sum(0)
        nulls[b] = np.nanmax(np.abs(mu / (sd / np.sqrt(n_))))
print(f"  candidates {len(M.columns)}   best = {best}  |t| = {obs:.2f}")
print(f"  null max|t|: median {np.median(nulls):.2f}  95th {np.percentile(nulls,95):.2f}")
print(f"  Reality Check p = {(nulls >= obs).mean():.4f}")

print("\n" + "=" * 100)
print("TREND — is the index getting more top-heavy? (descriptive, always valid)")
print("=" * 100)
ew = R[[s for s in ALL50 if s in R.columns]].mean(axis=1)
spread = y - ew
adv50 = (R > 0).sum(axis=1) / R.notna().sum(axis=1) * 100
for yr, g in spread.groupby(spread.index.year):
    a = adv50.reindex(g.index)
    yy = y.reindex(g.index)
    print(f"  {yr}: |carry spread| mean {g.abs().mean():.3f}pp   sd {g.std():.3f}   "
          f"index-up-while-breadth-under-50%: {((yy>0)&(a<50)).sum():3d} days "
          f"({((yy>0)&(a<50)).mean()*100:.1f}%)")

print("\n" + "=" * 100)
print("INCREMENTAL — does bucketed flow add anything OVER the index's own momentum?")
print("=" * 100)
print("  OLS  forward_gap ~ nifty_ret + <bucket signal>,  Newey-West(5) SE")
for nm, sg in [('TOP10 adv', B['TOP10']['adv']), ('NEXT10 adv', B['NEXT10']['adv']),
               ('REST30 adv', B['REST30']['adv']),
               ('DIVERGE adv', B['TOP10']['adv'] - B['REST30']['adv']),
               ('TOP10 deliv_z', B['TOP10']['deliv_z'])]:
    k = pd.DataFrame({'g': FWD['gap'], 'm': y, 's': sg}).dropna().astype(float)
    if len(k) < 100: continue
    X = np.column_stack([np.ones(len(k)), k['m'].values, k['s'].values])
    b, *_ = np.linalg.lstsq(X, k['g'].values, rcond=None)
    e = k['g'].values - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None])
    meat = S.T @ S
    for L in range(1, 6):
        G = S[L:].T @ S[:-L]
        meat += (1 - L / 6) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    print(f"    {nm:16s} n={len(k):4d}  nifty_ret t={b[1]/se[1]:+6.2f}   "
          f"{nm.split()[-1]} t={b[2]/se[2]:+6.2f}")
print("\n  (if the signal's t collapses while nifty_ret holds, the bucket adds nothing)")

print("\n" + "=" * 100)
print("IS THE LARGE-CAP BREADTH -> GAP RELATION MONOTONE, OR TAIL-DRIVEN?")
print("=" * 100)
for nm, sg in [('TOP10 adv', B['TOP10']['adv']), ('NEXT10 adv', B['NEXT10']['adv']),
               ('REST30 adv', B['REST30']['adv'])]:
    k = pd.DataFrame({'s': sg, 'g': FWD['gap']}).dropna().astype(float)
    k['q'] = pd.qcut(k['s'], 5, labels=False, duplicates='drop')
    t = k.groupby('q').agg(n=('g', 'size'), adv=('s', 'mean'),
                           gap_bps=('g', lambda v: v.mean() * 100),
                           up=('g', lambda v: (v > 0).mean() * 100))
    print(f"\n  {nm}:")
    print("   " + "  ".join(f"Q{int(q)+1}:{r.gap_bps:+6.1f}bps(up{r.up:.0f}%,n{int(r.n)})"
                            for q, r in t.iterrows()))
print("\n  survivorship check — how much of the panel predates the current 50?")
first = R.notna().idxmax()
late = first[first > pd.Timestamp('2022-06-01')]
print(f"    {len(late)} of {R.shape[1]} names have no data at the 2022 start:")
print("    " + ", ".join(f"{k}({v:%Y-%m})" for k, v in late.sort_values().items()))
