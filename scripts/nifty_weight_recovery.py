"""Can we recover Nifty 50 index weights from our own data, point-in-time?

We have no free-float / shares-outstanding table, so weights cannot be computed
the way NSE computes them. But the index return IS the weighted sum of its
constituents' returns, so the weights are RECOVERABLE by constrained regression:

    r_nifty(t) = sum_i w_i * r_i(t),   w_i >= 0,  sum w_i = 1

Fitted on a rolling window this gives weights implied by the index itself — our
own data, point-in-time, no external snapshot, and SELF-VALIDATING: if the fit
reproduces the index return, the weights are right.

Alternative considered and rejected: hardcode the weights from a web table. That
is a single snapshot applied to history (the same survivorship trap already
documented for v_sector_master), and the two published tables in the source
screenshots do not even reconcile with each other.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option('display.width', 230)
c = duckdb.connect('data/market_data.duckdb', read_only=True)

# the shipped constituent list (index_prediction._NIFTY50_SYMBOLS, reviewed Jun 2026)
SYMS = ("ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK","BAJAJ-AUTO",
        "BAJFINANCE","BAJAJFINSV","BEL","BHARTIARTL","CIPLA","COALINDIA","DRREDDY",
        "EICHERMOT","ETERNAL","GRASIM","HCLTECH","HDFCBANK","HDFCLIFE","HINDALCO",
        "HINDUNILVR","ICICIBANK","ITC","INFY","INDIGO","JSWSTEEL","JIOFIN","KOTAKBANK",
        "LT","M&M","MARUTI","MAXHEALTH","NTPC","NESTLEIND","ONGC","POWERGRID",
        "RELIANCE","SBILIFE","SHRIRAMFIN","SBIN","SUNPHARMA","TCS","TATACONSUM",
        "TMPV","TATASTEEL","TECHM","TITAN","TRENT","ULTRACEMCO","WIPRO")

ph = ",".join("?" * len(SYMS))
px = c.execute(f"""
    SELECT trade_date, symbol,
           (close_price - prev_close)/NULLIF(prev_close,0)*100 AS r
    FROM daily_data
    WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND trade_date >= '2024-01-01' AND prev_close > 0
""", list(SYMS)).df()
px['trade_date'] = pd.to_datetime(px['trade_date'])
R = px.pivot_table('r', 'trade_date', 'symbol').sort_index()
nif = c.execute("""SELECT trade_date, pct_chg, close_val FROM index_data
                   WHERE index_name='Nifty 50' AND trade_date >= '2024-01-01'""").df()
nif['trade_date'] = pd.to_datetime(nif['trade_date'])
nif = nif.set_index('trade_date').sort_index()

print(f"constituent return panel: {R.shape[0]} sessions x {R.shape[1]} symbols")
print(f"coverage: {R.notna().mean().mean()*100:.1f}% of symbol-days present")
thin = R.notna().mean().sort_values().head(6)
print(f"least-covered symbols:\n{(thin*100).round(1).to_string()}")

print("\n=== SANITY: does our index_data agree with the screenshot's 4 Sep 2026 Nifty? ===")
s = nif[nif.index == '2026-09-04']
print(f"  our index_data 2026-09-04: {s['close_val'].iloc[0] if len(s) else 'ABSENT'}"
      f"  (screenshot claimed 23,897.70, +24.25 / +0.10%)")
print(f"  our last 3 sessions:\n{nif.tail(3)[['close_val','pct_chg']].round(2).to_string()}")

# ── constrained weight recovery: min ||Xw - y||, w>=0, sum w = 1 ──────────────
def solve_w(X, y, iters=4000, lr=0.35):
    """Projected-gradient least squares on the simplex (no scipy in this env)."""
    n = X.shape[1]
    w = np.full(n, 1.0 / n)
    XtX = X.T @ X
    Xty = X.T @ y
    L = np.linalg.norm(XtX, 2) + 1e-9
    for _ in range(iters):
        g = (XtX @ w - Xty) / L
        w = w - lr * g
        # project onto the probability simplex
        u = np.sort(w)[::-1]
        css = np.cumsum(u) - 1
        rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
        w = np.maximum(w - css[rho] / (rho + 1), 0)
    return w


print("\n=== WEIGHT RECOVERY — rolling constrained fit, out-of-sample check ===")
common = R.index.intersection(nif.index)
Rm = R.loc[common]
ym = nif.loc[common, 'pct_chg'].astype(float)
WIN = 250
rows = []
recovered = {}
for end in range(WIN, len(common), 20):
    tr = slice(end - WIN, end)
    Xtr = Rm.iloc[tr].fillna(0.0).values
    ytr = ym.iloc[tr].values
    ok = ~np.isnan(ytr)
    if ok.sum() < 100: continue
    w = solve_w(Xtr[ok], ytr[ok])
    # OUT-OF-SAMPLE: next 20 sessions, weights frozen
    te = slice(end, min(end + 20, len(common)))
    Xte = Rm.iloc[te].fillna(0.0).values
    yte = ym.iloc[te].values
    m = ~np.isnan(yte)
    if m.sum() < 5: continue
    pred = Xte[m] @ w
    err = pred - yte[m]
    ss = 1 - (err ** 2).sum() / ((yte[m] - yte[m].mean()) ** 2).sum()
    rows.append(dict(as_of=str(common[end].date()), oos_R2=ss,
                     mae_pp=np.abs(err).mean(), corr=np.corrcoef(pred, yte[m])[0, 1]))
    recovered[common[end]] = pd.Series(w, index=Rm.columns)
res = pd.DataFrame(rows)
print(f"  {len(res)} rolling fits, 250d train -> next 20d frozen")
print(f"  OOS R2      : median {res['oos_R2'].median():.4f}   min {res['oos_R2'].min():.4f}")
print(f"  OOS corr    : median {res['corr'].median():.4f}")
print(f"  OOS MAE     : median {res['mae_pp'].median():.4f} pp of index return")
print(res.tail(5).round(4).to_string(index=False))

print("\n=== RECOVERED WEIGHTS (latest fit) vs the screenshot's published table ===")
last = recovered[max(recovered)]
top = (last.sort_values(ascending=False) * 100).round(2)
print(top.head(20).to_string())
print(f"\n  recovered top-10 sum : {top.head(10).sum():.2f}%   (screenshot said 46.15%)")
print(f"  recovered top-20 sum : {top.head(20).sum():.2f}%   (screenshot said 66.71%)")
print(f"  recovered rest-30 sum: {top.tail(30).sum():.2f}%   (screenshot said 33.29%)")
print(f"  weights sum to       : {top.sum():.2f}%")
