"""Recover Nifty weights from PRICE LEVELS, not returns.

WHY THIS MIGHT WORK WHERE THE RETURNS FIT FAILED
    Nifty is free-float cap weighted:   Nifty_t = k * sum_i ( price_i,t * ffshares_i )
    ffshares_i is CONSTANT between corporate actions and rebalances. So a single
    vector s_i = k*ffshares_i must reproduce the index LEVEL on every day in the
    window simultaneously. That is a far tighter constraint than the returns
    regression, which only has to match daily changes and can therefore shuffle
    weight between collinear names (that fit gave ICICIBANK 11.6% vs a published
    5.26% and dropped HDFCBANK out of the top 20 entirely).

    weight_i,t = s_i * price_i,t / sum_j ( s_j * price_j,t )   <- time-varying,
    which is correct: weights drift with relative price even between rebalances.

VALIDATION (the whole point — do not trust the fit, test it)
    V1 does it reproduce the index LEVEL out of sample?
    V2 does it rank the constituents like the published table?
    V3 is HDFCBANK back near rank 3, and are the bucket sums near 46/67/33?
    V4 is the solution STABLE across refits, or does it wander (= unidentified)?
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
# published table from the user's screenshots (Arihant, 4 Sep 2026) - ranks 1-20 only,
# used ONLY to score the recovery, never as an input
PUB = {"RELIANCE":9.22,"BHARTIARTL":5.92,"HDFCBANK":5.66,"ICICIBANK":5.26,"SBIN":4.84,
       "TCS":4.29,"BAJFINANCE":3.40,"LT":2.81,"HINDUNILVR":2.39,"INFY":2.36,
       "SUNPHARMA":2.35,"TITAN":2.29,"KOTAKBANK":2.18,"MARUTI":2.06,"ADANIENT":2.05,
       "AXISBANK":2.04,"M&M":2.03,"ADANIPORTS":2.03,"HCLTECH":1.81,"ULTRACEMCO":1.72}

ph = ",".join("?" * len(SYMS))
d = c.execute(f"""SELECT trade_date, symbol, close_price FROM daily_data
    WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST') AND close_price>0
      AND trade_date>='2025-06-01'""", list(SYMS)).df()
d['trade_date'] = pd.to_datetime(d['trade_date'])
P = d.pivot_table('close_price', 'trade_date', 'symbol').sort_index()
nif = c.execute("""SELECT trade_date, close_val FROM index_data
                   WHERE index_name='Nifty 50' AND trade_date>='2025-06-01'""").df()
nif['trade_date'] = pd.to_datetime(nif['trade_date'])
nif = nif.set_index('trade_date').sort_index()['close_val'].astype(float)
ix = P.index.intersection(nif.index)
P = P.loc[ix].dropna(axis=1, how='any')          # need a complete price matrix
y = nif.loc[ix]
print(f"price panel {P.shape[0]} sessions x {P.shape[1]} symbols "
      f"({P.index.min():%Y-%m-%d}..{P.index.max():%Y-%m-%d})")
missing = sorted(set(SYMS) - set(P.columns))
print(f"dropped for incomplete history: {missing}")


def nnls_pg(X, yv, iters=20000, ):
    """Non-negative least squares by projected gradient (no scipy here)."""
    n = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ yv
    L = np.linalg.norm(XtX, 2)
    w = np.full(n, yv.mean() / (X.mean() * n))
    for _ in range(iters):
        w = np.maximum(w - (XtX @ w - Xty) / L, 0.0)
    return w


print("\n=== V1/V4. ROLLING FIT ON LEVELS — 120d train, next 20d frozen ===")
WIN = 120
rows, sols = [], {}
for end in range(WIN, len(ix), 20):
    Xtr = P.iloc[end - WIN:end].values
    ytr = y.iloc[end - WIN:end].values
    s = nnls_pg(Xtr, ytr)
    te = slice(end, min(end + 20, len(ix)))
    Xte, yte = P.iloc[te].values, y.iloc[te].values
    if len(yte) < 5: continue
    pred = Xte @ s
    err = np.abs(pred - yte) / yte * 100
    rows.append(dict(as_of=str(ix[end].date()), oos_mape_pct=err.mean(),
                     oos_max_err_pct=err.max()))
    sols[ix[end]] = pd.Series(s, index=P.columns)
res = pd.DataFrame(rows)
print(f"  {len(res)} fits.  OOS |error| on the index LEVEL:")
print(f"    median MAPE {res['oos_mape_pct'].median():.4f}%   worst {res['oos_max_err_pct'].max():.3f}%")
print(res.tail(4).round(4).to_string(index=False))

last_dt = max(sols)
s = sols[last_dt]
w = (s * P.loc[last_dt]); w = (w / w.sum() * 100).sort_values(ascending=False)

print(f"\n=== V2/V3. RECOVERED WEIGHTS as of {last_dt:%Y-%m-%d} vs published ===")
cmp = pd.DataFrame({'recovered': w.round(2)})
cmp['published'] = pd.Series(PUB)
cmp['diff'] = (cmp['recovered'] - cmp['published']).round(2)
print(cmp.head(22).to_string())
k = cmp.dropna()
print(f"\n  on the 20 published names: corr {k['recovered'].corr(k['published']):+.4f}   "
      f"rank corr {k['recovered'].rank().corr(k['published'].rank()):+.4f}   "
      f"mean |diff| {k['diff'].abs().mean():.2f}pp")
print(f"  HDFCBANK recovered rank: {list(w.index).index('HDFCBANK')+1}  (published rank 3)")
print(f"  top-10 sum  {w.head(10).sum():.2f}%  (published 46.15%)")
print(f"  top-20 sum  {w.head(20).sum():.2f}%  (published 66.71%)")
print(f"  rest sum    {w.iloc[20:].sum():.2f}%  (published 33.29%)")

print("\n=== V4. STABILITY — does the top-10 membership hold across refits? ===")
keys = sorted(sols)[-6:]
sets = []
for kdt in keys:
    ww = (sols[kdt] * P.loc[kdt]); ww = ww / ww.sum() * 100
    t10 = set(ww.sort_values(ascending=False).head(10).index)
    sets.append(t10)
    print(f"  {kdt:%Y-%m-%d} top10: {', '.join(sorted(t10))}")
inter = set.intersection(*sets) if sets else set()
print(f"  names in EVERY refit's top-10: {len(inter)}/10 -> {sorted(inter)}")
