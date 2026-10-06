"""The options read and the expiry calendar.

    python scripts/ilc_options_cycle_audit.py

1 The call/put OI gap by sessions-to-expiry, pooled and per index. This is the
  profile behind index_largecap._opt_cycle_norm.
2 Whether the Nifty Top-10 put-vs-call lead is only that calendar: the outcome
  by cycle phase, the lead with phase dummies, and within each phase.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd
from src.analytics.index_largecap import INDEX_BUCKETS, _min_valid

c = duckdb.connect("data/market_data.duckdb", read_only=True)
syms = sorted({s for m in INDEX_BUCKETS.values() for b in m["buckets"].values() for s in b})
ph = ",".join("?" * len(syms))
o = c.execute(f"""SELECT trade_date, symbol, option_type ot, expiry_date, SUM(open_interest) oi
    FROM fno_bhavcopy WHERE instrument='OPTSTK' AND option_type IN ('CE','PE') AND open_interest>0
      AND symbol IN ({ph}) AND trade_date>='2022-01-01' AND expiry_date>trade_date
    GROUP BY 1,2,3,4""", syms).df()
o["trade_date"] = pd.to_datetime(o.trade_date); o["expiry_date"] = pd.to_datetime(o.expiry_date)
days = sorted(o.trade_date.unique()); prev = {d: p for d, p in zip(days[1:], days[:-1])}
settle = sorted(pd.to_datetime(c.execute("SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK' AND expiry_date=trade_date").df().trade_date))
pos = {d: i for i, d in enumerate(days)}
nxt = {}                                   # sessions to the NEXT settlement (0 = settlement day)
for d in days:
    fut = [s for s in settle if s >= d and s in pos]
    nxt[d] = (pos[fut[0]] - pos[d]) if fut else np.nan
byday = dict(tuple(o.groupby("trade_date")))
rows = []
for d in days[1:]:
    if d in settle: continue
    p = prev[d]; a = byday[d]; b = byday[p]; b = b[b.expiry_date > d]
    A = a.groupby(["symbol", "ot"]).oi.sum().unstack(); B = b.groupby(["symbol", "ot"]).oi.sum().unstack()
    j = ((A - B) / B * 100).clip(-50, 50).dropna()
    for sym, meta in INDEX_BUCKETS.items():
        for bn, mem in meta["buckets"].items():
            x = j.reindex([m for m in mem if m in j.index]).dropna()
            if len(x) >= _min_valid(len(mem)):
                rows.append((d, sym, bn, nxt[d], x.PE.mean() - x.CE.mean()))
D = pd.DataFrame(rows, columns=["d", "idx", "b", "dte", "diff"]).dropna()
D["dte"] = D.dte.astype(int)
prof = D.groupby("dte")["diff"].agg(["mean", "median", "size"])
print("mean (PE% - CE%) by sessions-to-expiry, all buckets pooled:")
print("  " + "  ".join(f"{k}:{v:+.2f}" for k, v in prof["mean"].items() if k <= 24))
print("per index (mean diff at dte 1,2,3,5,10,15,19,20,21):")
for sym in INDEX_BUCKETS:
    s = D[D.idx == sym].groupby("dte")["diff"].mean()
    print(f"  {sym:11s} " + "  ".join(f"{k}:{s.get(k, np.nan):+.2f}" for k in (1, 2, 3, 5, 10, 15, 19, 20, 21)))

# ── the lead: Nifty Top 10 ──────────────────────────────────────────────────
ix = c.execute("SELECT trade_date, high_val, low_val, close_val, pct_chg FROM index_data WHERE index_name='Nifty 50' AND trade_date>='2021-12-01'").df()
ix["trade_date"] = pd.to_datetime(ix.trade_date); ix = ix.set_index("trade_date").sort_index().astype(float)
lg = np.log1p(ix.pct_chg / 100)
d5 = np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100
clr = (ix.close_val - ix.low_val) / (ix.high_val - ix.low_val).replace(0, np.nan) - .5
T = D[(D.idx == "NIFTY") & (D.b == "Top 10")].set_index("d")
T["d5"] = d5.reindex(T.index); T["clr"] = clr.reindex(T.index); T["mom"] = ix.pct_chg.reindex(T.index)
T = T.dropna()
mu = D.groupby("dte")["diff"].mean()                 # pooled seasonal profile
T["adj"] = T["diff"] - T.dte.map(mu)
T["x"] = np.sign(T["diff"].where(T["diff"].abs() >= .25, 0)); T["xa"] = np.sign(T.adj.where(T.adj.abs() >= .25, 0))

def nw(y, X, lag=5):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b; S = X * e[:, None]; M = S.T @ S
    for L in range(1, lag + 1):
        G = S[L:].T @ S[:-L]; M += (1 - L / (lag + 1)) * (G + G.T)
    V = np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(V @ M @ V)); return b / se
print(f"\nNifty Top 10, n={len(T)}")
print("next-5d return by sessions-to-expiry bucket (is the OUTCOME itself calendar-driven?):")
T["ph"] = pd.cut(T.dte, [0, 2, 5, 10, 15, 30], labels=["1-2", "3-5", "6-10", "11-15", "16+"])
print("  " + "  ".join(f"dte {k}: {v*100:+.0f}bps (n{n}, put-side {p*100:.0f}%)" for k, v, n, p in
                       zip(*[T.groupby("ph", observed=True).d5.mean().index, T.groupby("ph", observed=True).d5.mean(),
                             T.groupby("ph", observed=True).size(), T.groupby("ph", observed=True).x.apply(lambda s: (s > 0).mean())])))
for nm in ("x", "xa"):
    up, dn = T[T[nm] > 0].d5, T[T[nm] < 0].d5
    t0 = nw(T.d5.values, T[["mom", "clr", nm]].values)[3]
    ph = pd.get_dummies(T.ph, drop_first=True).astype(float).values
    t1 = nw(T.d5.values, np.column_stack([T[["mom", "clr"]].values, ph, T[nm].values]))[-1]
    lab = "RAW label      " if nm == "x" else "CYCLE-ADJUSTED "
    print(f"  {lab} put-side {len(up)}d {up.mean()*100:+.1f}bps ({(up>0).mean()*100:.0f}% up) | call-side {len(dn)}d "
          f"{dn.mean()*100:+.1f}bps ({(dn>0).mean()*100:.0f}% up) | t beside CLR+mom {t0:+.2f} | + cycle-phase dummies {t1:+.2f}")
print("within each phase, raw put-side minus call-side next-5d (bps):")
for k, g in T.groupby("ph", observed=True):
    u, d_ = g[g.x > 0].d5, g[g.x < 0].d5
    print(f"   dte {k}: {(u.mean()-d_.mean())*100:+.0f}bps  (put-side n{len(u)}, call-side n{len(d_)})")
