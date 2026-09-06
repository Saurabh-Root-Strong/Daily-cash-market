"""ANALOGUE ENGINE: "when the flow looked like today, what did NIFTY do next?"

A fourth distinct question. Earlier files tested continuous signals (v2), discrete
state cells (patterns) and a weighted composite. This one does nearest-neighbour
matching on the FULL 9-dimensional flow state and reports the forward distribution
of the matched days.

PRIOR, from this codebase, and it is not encouraging:
  * the intraday analogue engine was null on 3,570 states / 51 sessions, and 13-D
    matching DESTROYED a 1-D signal that had IC +0.506 on its own;
  * the DCM analogue panel was null in all three of its modes (|HAC t| < 1.5,
    permutation p = 0.43).
So the burden here is on the engine to beat controls, not on me to disprove it.

STATE VECTOR (9 dims: 3 buckets x 3 legs)
    deliv    per-symbol delivery z vs each name's own 100d normal, bucket mean
    fut      signed futures pressure: total forward OI change x price direction
    opt      PE OI change - CE OI change (the relative CE/PE flow read)
Each dim is standardised by its OWN TRAILING mean/sd (250d, shifted) so the
distance metric is causal and does not drift with the level of any leg.

EVERY TRAP THIS HAS TO SURVIVE, and how each is handled
 1 LOOKAHEAD          neighbours are drawn strictly from days < t, and the
                      standardisation uses trailing stats only.
 2 WINDOW OVERLAP     a neighbour at t-2 shares 3 of its 5 forward days with t.
                      A PURGE GAP of `horizon` sessions before t is enforced, the
                      standard fix from combinatorial-purged CV.
 3 SELF-MATCH         t can never be its own neighbour (implied by 1, asserted).
 4 EFFECTIVE n        matched days cluster in time. Inference is on EPISODES.
 5 THE CURSE          in 9 dims, "nearest" may be nowhere near. Distance
                      distributions are reported, plus a 1-D and 3-D variant.
 6 RANDOM-k CONTROL   the decisive test. If k nearest neighbours give the same
                      forward distribution as k RANDOM past days, the engine is
                      theatre. Run at every k.
 7 REGIME DRIFT       a 2022 analogue may not apply in 2026. Recency-weighted and
                      recent-only variants run beside the full-history one.
 8 DEGENERATE MATCH   if the k-th distance is ~0 for many days the state space is
                      too coarse; if it is huge, nothing is really an analogue.
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option("display.width", 250)
rng = np.random.default_rng(61)

TOP10 = ["RELIANCE", "BHARTIARTL", "HDFCBANK", "ICICIBANK", "SBIN", "TCS",
         "BAJFINANCE", "LT", "HINDUNILVR", "INFY"]
NEXT10 = ["SUNPHARMA", "TITAN", "KOTAKBANK", "MARUTI", "ADANIENT", "AXISBANK",
          "M&M", "ADANIPORTS", "HCLTECH", "ULTRACEMCO"]
REST30 = ["APOLLOHOSP", "ASIANPAINT", "BAJAJ-AUTO", "BAJAJFINSV", "BEL", "CIPLA",
          "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HDFCLIFE",
          "HINDALCO", "ITC", "INDIGO", "JSWSTEEL", "JIOFIN", "MAXHEALTH", "NTPC",
          "NESTLEIND", "ONGC", "POWERGRID", "SBILIFE", "SHRIRAMFIN", "TATACONSUM",
          "TMPV", "TATASTEEL", "TECHM", "TRENT", "WIPRO"]
BUCK = {"TOP10": TOP10, "NEXT10": NEXT10, "REST30": REST30}
ALL50 = TOP10 + NEXT10 + REST30
ph = ",".join("?" * len(ALL50))

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2022-01-01")
a = ap.parse_args()
WARM = (pd.Timestamp(a.start) - pd.Timedelta(days=400)).date().isoformat()
c = duckdb.connect("data/market_data.duckdb", read_only=True)

cash = c.execute(f"""SELECT trade_date, symbol, deliv_per,
        (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND close_price>0 AND prev_close>0 AND trade_date>=?""", [*ALL50, WARM]).df()
cash["trade_date"] = pd.to_datetime(cash["trade_date"])
cash = cash[cash["r"].abs() < 40]
R = cash.pivot_table("r", "trade_date", "symbol").sort_index()
DL = cash.pivot_table("deliv_per", "trade_date", "symbol").sort_index()

fno = c.execute(f"""SELECT trade_date, symbol, expiry_date, instrument, option_type,
        SUM(open_interest) oi FROM fno_bhavcopy
    WHERE instrument IN ('FUTSTK','OPTSTK') AND open_interest>0
      AND symbol IN ({ph}) AND expiry_date > trade_date AND trade_date>=?
    GROUP BY 1,2,3,4,5""", [*ALL50, WARM]).df()
fno["trade_date"] = pd.to_datetime(fno["trade_date"])
SETTLE = set(pd.to_datetime(c.execute(
    "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK'"
    " AND expiry_date=trade_date").df()["trade_date"]))
nif = c.execute("""SELECT trade_date,open_val,high_val,low_val,close_val,pct_chg
    FROM index_data WHERE index_name='Nifty 50' AND trade_date>=?""", [WARM]).df()
nif["trade_date"] = pd.to_datetime(nif["trade_date"])
nif = nif.set_index("trade_date").sort_index().astype(float)


def fwd_oi(df, extra=None):
    days = sorted(df["trade_date"].unique())
    g = dict(tuple(df.groupby("trade_date")))
    keys = ["symbol", "expiry_date"] + (extra or [])
    grp = ["symbol"] + (extra or [])
    out = []
    for d, p in zip(days[1:], days[:-1]):
        if d in SETTLE:
            continue
        m = g[d].merge(g[p], on=keys, suffixes=("", "_p"))
        if m.empty:
            continue
        s = m.groupby(grp)[["oi", "oi_p"]].sum()
        s = s[s["oi_p"] > 0]
        s["pct"] = ((s["oi"] - s["oi_p"]) / s["oi_p"] * 100).clip(-50, 50)
        s["trade_date"] = d
        out.append(s.reset_index())
    return pd.concat(out, ignore_index=True)


F = fwd_oi(fno[fno["instrument"] == "FUTSTK"])
O = fwd_oi(fno[fno["instrument"] == "OPTSTK"], extra=["option_type"])
FOI = F.pivot_table("pct", "trade_date", "symbol")
CE = O[O["option_type"] == "CE"].pivot_table("pct", "trade_date", "symbol")
PE = O[O["option_type"] == "PE"].pivot_table("pct", "trade_date", "symbol")


def persym_z(M, cols, base=100):
    S = M[[s for s in cols if s in M.columns]]
    mu = S.rolling(base, min_periods=max(30, base // 2)).mean().shift(1)
    sd = S.rolling(base, min_periods=max(30, base // 2)).std().shift(1)
    return ((S - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan).mean(axis=1)


raw = {}
for bn, cols in BUCK.items():
    cols = [s for s in cols if s in R.columns]
    sub = R[cols]
    f = FOI[[s for s in cols if s in FOI.columns]].reindex(sub.index)
    ce = CE[[s for s in cols if s in CE.columns]].reindex(sub.index)
    pe = PE[[s for s in cols if s in PE.columns]].reindex(sub.index)
    raw[f"{bn}_deliv"] = persym_z(DL, cols)
    raw[f"{bn}_fut"] = (np.sign(sub) * f).mean(axis=1)
    raw[f"{bn}_opt"] = (pe - ce).mean(axis=1)
X = pd.DataFrame(raw).sort_index()
# Causal standardisation: trailing 250d, shifted, so day t never sees itself.
#
# min_periods IS LOAD-BEARING. The futures/options legs are NaN on every expiry
# session (forward OI jumps mechanically there, so those days are excluded), which
# punches ~5% holes into those series. pandas defaults min_periods to the window,
# so a 250-row window containing ~12 NaNs NEVER qualifies -- every z came back NaN
# and the whole state matrix was empty, while the gap-free delivery legs looked
# fine. A silent, total failure that only showed up as "no data".
_W, _MP = 250, 150
Z = ((X - X.rolling(_W, min_periods=_MP).mean().shift(1)) /
     X.rolling(_W, min_periods=_MP).std().shift(1)).replace([np.inf, -np.inf], np.nan)

IX = Z.index.intersection(nif.index)
y = nif.loc[IX, "pct_chg"]
lg = np.log1p(y / 100.0)
FW = {1: y.shift(-1),
      5: np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100.0}
CLR = ((nif["close_val"] - nif["low_val"]) /
       (nif["high_val"] - nif["low_val"]).replace(0, np.nan)).reindex(IX)

Z = Z.reindex(IX).dropna()
Z = Z[Z.index >= pd.Timestamp(a.start)]
if Z.empty:
    diag = pd.DataFrame({
        "raw_rows": X.notna().sum(),
        "raw_first": X.apply(lambda s_: s_.first_valid_index()),
        "z_rows": ((X - X.rolling(250).mean().shift(1)) /
                   X.rolling(250).std().shift(1)).notna().sum()})
    print("STATE MATRIX EMPTY — per-dimension diagnosis:")
    print(diag.to_string())
    raise SystemExit(1)
DIMS = list(Z.columns)
print(f"state matrix {Z.shape[0]} sessions x {Z.shape[1]} dims  "
      f"{Z.index.min():%Y-%m-%d}..{Z.index.max():%Y-%m-%d}")
print(f"dims: {DIMS}")
CORR = Z.corr()
off = CORR.values[~np.eye(len(DIMS), dtype=bool)]
print(f"\ninter-dimension correlation: mean |rho| {np.abs(off).mean():.3f}  "
      f"max {np.abs(off).max():.3f}")
ev = np.linalg.eigvalsh(CORR.values)[::-1]
pr = ev.sum() ** 2 / (ev ** 2).sum()
print(f"participation ratio {pr:.2f} of {len(DIMS)} dims  "
      f"(the state space is EFFECTIVELY {pr:.1f}-dimensional, not {len(DIMS)})")


def episodes(idx, maxgap=5):
    if len(idx) == 0:
        return []
    idx = sorted(idx)
    epi, cur = [], [idx[0]]
    for p, q in zip(idx, idx[1:]):
        if q - p <= maxgap:
            cur.append(q)
        else:
            epi.append(cur)
            cur = [q]
    epi.append(cur)
    return epi


V = Z.values
N = len(Z)
MINHIST = 250


def run(k, h, dims=None, mode="knn", recent=None):
    """Walk-forward analogue prediction. Returns (pred, actual, dist_k) aligned."""
    cols = [DIMS.index(d) for d in (dims or DIMS)]
    W = V[:, cols]
    act = FW[h].reindex(Z.index).values
    preds, acts, dk, rows = [], [], [], []
    for i in range(MINHIST, N):
        if not np.isfinite(act[i]):
            continue
        # PURGE: a neighbour within `h` sessions of t shares forward days with t
        hi = i - h
        lo = max(0, hi - recent) if recent else 0
        if hi - lo < 60:
            continue
        cand = np.arange(lo, hi)
        ok = np.isfinite(act[cand])
        cand = cand[ok]
        if len(cand) < 60:
            continue
        if mode == "random":
            sel = rng.choice(cand, size=min(k, len(cand)), replace=False)
            d_k = np.nan
        else:
            dist = np.abs(W[cand] - W[i]).sum(axis=1)      # L1: robust in 9 dims
            order = np.argsort(dist)[:k]
            sel = cand[order]
            d_k = dist[order[-1]]
        assert i not in sel, "self-match leaked into the neighbour set"
        preds.append(np.nanmean(act[sel]))
        acts.append(act[i])
        dk.append(d_k)
        rows.append(i)
    return (np.array(preds), np.array(acts), np.array(dk), np.array(rows))


def nw(x, lag):
    """Newey-West t. The analogue fires EVERY day, so the sample is one unbroken
    block and the episode clustering used elsewhere in this repo degenerates to a
    single episode (it printed nan). Overlap is handled by the HAC lag instead."""
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 40:
        return float("nan")
    e = x - x.mean()
    v = (e @ e) / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / math.sqrt(v / n) if v > 0 else float("nan")


def score(p, a_, rows, tag, lag=1):
    if len(p) < 50:
        print(f"  {tag:44s} too few rows ({len(p)})")
        return None
    ic = pd.Series(p).rank().corr(pd.Series(a_).rank())
    sg = np.sign(p) * a_
    print(f"  {tag:44s} n={len(p):4d}  IC {ic:+.4f}  {sg.mean():+.3f}%  "
          f"hit {(sg>0).mean()*100:.1f}%  NW-t {nw(sg, lag):+5.2f}")
    return ic


print("\n" + "=" * 108)
print("THE DECISIVE TEST — k nearest neighbours vs k RANDOM past days")
print("=" * 108)
for h in (1, 5):
    base_up = (FW[h].reindex(Z.index).dropna() > 0).mean() * 100
    print(f"\n horizon {h}d   (base rate up {base_up:.1f}%)")
    for k in (10, 25, 50, 100):
        p, a_, dk, rows = run(k, h)
        score(p, a_, rows, f"k={k:3d} NEAREST (all 9 dims)", h)
        p2, a2, _, rows2 = run(k, h, mode="random")
        score(p2, a2, rows2, f"k={k:3d} RANDOM  (control)", h)

print("\n" + "=" * 108)
print("DIMENSIONALITY — does matching on FEWER dims work better? (the 13-D lesson)")
print("=" * 108)
SETS = {
    "TOP10 only (3 dims)": ["TOP10_deliv", "TOP10_fut", "TOP10_opt"],
    "delivery only (3 dims)": ["TOP10_deliv", "NEXT10_deliv", "REST30_deliv"],
    "futures only (3 dims)": ["TOP10_fut", "NEXT10_fut", "REST30_fut"],
    "options only (3 dims)": ["TOP10_opt", "NEXT10_opt", "REST30_opt"],
    "TOP10 futures only (1 dim)": ["TOP10_fut"],
    "TOP10 delivery only (1 dim)": ["TOP10_deliv"],
}
for h in (1, 5):
    print(f"\n horizon {h}d, k=25")
    _p, _a, _, _r = run(25, h); score(_p, _a, _r, "ALL 9 dims", h)
    for nm, ds in SETS.items():
        p, a_, dk, rows = run(25, h, dims=ds)
        score(p, a_, rows, nm)

print("\n" + "=" * 108)
print("REGIME DRIFT — are recent analogues better than the whole history?")
print("=" * 108)
for h in (1, 5):
    print(f"\n horizon {h}d, k=25")
    for rec, nm in ((None, "all history"), (500, "last 500 sessions"),
                    (250, "last 250 sessions")):
        p, a_, dk, rows = run(25, h, recent=rec)
        score(p, a_, rows, nm, h)

print("\n" + "=" * 108)
print("MATCH QUALITY — is 'nearest' actually near, in 9 dimensions?")
print("=" * 108)
p, a_, dk, rows = run(25, 5)
allpair = []
for _ in range(4000):
    i, j = rng.integers(MINHIST, N, 2)
    allpair.append(np.abs(V[i] - V[j]).sum())
allpair = np.array(allpair)
print(f"  distance to the 25th nearest neighbour: median {np.median(dk):.2f}  "
      f"p10 {np.percentile(dk,10):.2f}  p90 {np.percentile(dk,90):.2f}")
print(f"  distance between two RANDOM days:       median {np.median(allpair):.2f}  "
      f"p10 {np.percentile(allpair,10):.2f}")
print(f"  ratio of medians {np.median(dk)/np.median(allpair):.3f}  "
      f"(1.00 would mean the nearest 25 are no closer than chance)")
print(f"  share of random pairs closer than the median k-th neighbour: "
      f"{(allpair < np.median(dk)).mean()*100:.1f}%")

print("\n" + "=" * 108)
print("DOES SIMILARITY BUY ANYTHING? outcome dispersion, near vs far neighbours")
print("=" * 108)
h = 5
act = FW[h].reindex(Z.index).values
near_sd, far_sd = [], []
for i in range(MINHIST, N, 3):
    if not np.isfinite(act[i]):
        continue
    cand = np.arange(0, i - h)
    cand = cand[np.isfinite(act[cand])]
    if len(cand) < 200:
        continue
    dist = np.abs(V[cand] - V[i]).sum(axis=1)
    o = np.argsort(dist)
    near = act[cand[o[:25]]]
    far = act[cand[o[-25:]]]
    near_sd.append(abs(near.mean() - act[i]))
    far_sd.append(abs(far.mean() - act[i]))
print(f"  |mean(25 NEAREST) - actual|  {np.mean(near_sd):.3f}%")
print(f"  |mean(25 FARTHEST) - actual| {np.mean(far_sd):.3f}%")
print("  (if the nearest are not closer to the truth, similarity carries nothing)")

print("\n" + "=" * 108)
print("CONTROL — the analogue prediction vs the index's own close-strength")
print("=" * 108)
for h in (1, 5):
    p, a_, dk, rows = run(25, h)
    idx = Z.index[rows]
    k = pd.DataFrame({"p": p, "a": a_, "clr": CLR.reindex(idx).values - 0.5,
                      "mom": y.reindex(idx).values}).dropna()
    X_ = np.column_stack([np.ones(len(k)), k["mom"], k["clr"], k["p"]])
    b, *_ = np.linalg.lstsq(X_, k["a"].values, rcond=None)
    e = k["a"].values - X_ @ b
    XtXi = np.linalg.inv(X_.T @ X_)
    S = X_ * e[:, None]
    meat = S.T @ S
    for L in range(1, h + 1):
        G = S[L:].T @ S[:-L]
        meat += (1 - L / (h + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    print(f"  {h}d  n={len(k)}  mom t={b[1]/se[1]:+5.2f}  clr t={b[2]/se[2]:+5.2f}  "
          f"ANALOGUE t={b[3]/se[3]:+5.2f}")
