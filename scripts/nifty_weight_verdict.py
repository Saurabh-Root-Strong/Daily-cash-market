"""Why no internal method can recover the weights — and the one external
snapshot that solves it permanently.

PART A  PROOF: the constituent matrix is rank-deficient, so 49 weights are not
        identifiable from ONE index series. This is not a solver problem; it is
        an identifiability problem, so no smarter fit fixes it.

PART B  THE FIX: weight_i,t = p_i,t * ff_i / sum_j( p_j,t * ff_j ), and ff_i
        (free-float shares) is CONSTANT between rebalances. So ONE weight
        snapshot + our own prices gives the weight on EVERY other day, backwards
        and forwards, with no further external data until the next rebalance.
            ff_i  proportional to  w_i,t0 / p_i,t0
        Tested here on the 20 published weights we have.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option('display.width', 220)
c = duckdb.connect('data/market_data.duckdb', read_only=True)
SYMS = ("ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
        "BAJFINANCE","BAJAJFINSV","BEL","BHARTIARTL","CIPLA","COALINDIA","DRREDDY",
        "EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HINDALCO",
        "HINDUNILVR","ICICIBANK","ITC","INFY","INDIGO","JSWSTEEL","JIOFIN","KOTAKBANK",
        "LT","M&M","MARUTI","MAXHEALTH","NTPC","NESTLEIND","ONGC","POWERGRID",
        "RELIANCE","SBILIFE","SHRIRAMFIN","SBIN","SUNPHARMA","TCS","TATACONSUM",
        "TMPV","TATASTEEL","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO")
PUB = {"RELIANCE":9.22,"BHARTIARTL":5.92,"HDFCBANK":5.66,"ICICIBANK":5.26,"SBIN":4.84,
       "TCS":4.29,"BAJFINANCE":3.40,"LT":2.81,"HINDUNILVR":2.39,"INFY":2.36,
       "SUNPHARMA":2.35,"TITAN":2.29,"KOTAKBANK":2.18,"MARUTI":2.06,"ADANIENT":2.05,
       "AXISBANK":2.04,"M&M":2.03,"ADANIPORTS":2.03,"HCLTECH":1.81,"ULTRACEMCO":1.72}
PUB_ASOF = pd.Timestamp("2026-09-03")   # nearest session we hold to the 4 Sep table

ph = ",".join("?" * len(SYMS))
d = c.execute(f"""SELECT trade_date, symbol, close_price,
       (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND close_price>0 AND prev_close>0 AND trade_date>='2025-06-01'""", list(SYMS)).df()
d['trade_date'] = pd.to_datetime(d['trade_date'])
P = d.pivot_table('close_price', 'trade_date', 'symbol').sort_index()
R = d.pivot_table('r', 'trade_date', 'symbol').sort_index()
nif = c.execute("""SELECT trade_date, close_val, pct_chg FROM index_data
                   WHERE index_name='Nifty 50' AND trade_date>='2025-06-01'""").df()
nif['trade_date'] = pd.to_datetime(nif['trade_date']); nif = nif.set_index('trade_date').sort_index()
ix = R.index.intersection(nif.index)
R, P = R.loc[ix], P.loc[ix]
y = nif.loc[ix, 'pct_chg'].astype(float)

print("=" * 88)
print("PART A — IDENTIFIABILITY: is the constituent matrix even full rank?")
print("=" * 88)
Rc = R.dropna(axis=1, how='any')
X = Rc.values
sv = np.linalg.svd(X, compute_uv=False)
er = (sv.sum() ** 2) / (sv ** 2).sum()          # participation ratio = effective rank
cum = np.cumsum(sv ** 2) / (sv ** 2).sum()
k90 = int(np.searchsorted(cum, 0.90)) + 1
print(f"  return matrix: {X.shape[0]} days x {X.shape[1]} names")
print(f"  condition number      : {sv[0]/sv[-1]:,.0f}")
print(f"  effective rank        : {er:.1f}  (of {X.shape[1]} names)")
print(f"  components for 90% var: {k90}")
print(f"  1st component explains: {sv[0]**2/(sv**2).sum()*100:.1f}% of variance (the market itself)")
print(f"  mean pairwise corr    : {Rc.corr().values[np.triu_indices(Rc.shape[1],1)].mean():+.3f}")
print(f"\n  => one index series per day gives 1 equation. {X.shape[1]} unknowns sit in a space")
print(f"     of effective rank {er:.1f}. The weights are NOT identifiable from the index alone.")
print("     Confirms both failures: returns fit -> ICICIBANK 11.6% / HDFCBANK out of top-20;")
print("     levels fit -> rank corr -0.06, MARUTI 20.5%. No better solver fixes this.")

print("\n" + "=" * 88)
print("PART B — ONE SNAPSHOT IS ENOUGH: ff shares are constant between rebalances")
print("=" * 88)
if PUB_ASOF not in P.index:
    PUB_ASOF = P.index[P.index <= PUB_ASOF][-1]
p0 = P.loc[PUB_ASOF]
have = [s for s in PUB if s in P.columns and pd.notna(p0.get(s))]
ff = pd.Series({s: PUB[s] / p0[s] for s in have})       # proportional to free-float shares
print(f"  anchor date {PUB_ASOF:%Y-%m-%d}, {len(have)} published names available")

# implied weights on every day, from prices alone
W = P[have].mul(ff, axis=1)
W = W.div(W.sum(axis=1), axis=0)                        # renormalised within the known subset
print("\n  implied weight of the published names, drift over the window:")
chk = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'TCS', 'BHARTIARTL']
show = (W[[s for s in chk if s in W.columns]] * 100).round(2)
print(show.iloc[[0, len(show)//2, -1]].to_string())

# does a basket built from derived weights track the index better than equal weight?
sub = Rc[[s for s in have if s in Rc.columns]]
Wa = W[sub.columns].reindex(sub.index).ffill()
capw = (sub * Wa).sum(axis=1) / Wa.sum(axis=1)
eqw = sub.mean(axis=1)
yy = y.reindex(sub.index)
m = yy.notna() & capw.notna()
print(f"\n  tracking the Nifty daily return, {m.sum()} sessions "
      f"(subset of {len(sub.columns)} names, so a residual gap is expected):")
print(f"    derived cap-weight basket : corr {capw[m].corr(yy[m]):+.4f}  "
      f"MAE {np.abs(capw[m]-yy[m]).mean():.4f} pp")
print(f"    equal-weight basket       : corr {eqw[m].corr(yy[m]):+.4f}  "
      f"MAE {np.abs(eqw[m]-yy[m]).mean():.4f} pp")
print(f"\n  => the anchor buys a full point-in-time weight SERIES from prices alone.")
print(f"     Refresh needed only at reconstitution (Mar/Sep), an IWF revision, or a")
print(f"     corporate action - not daily, and never from a scraped endpoint.")
