"""Stage 2 - stress the only three things v2 left standing.

v2 found:
  * PANEL B (futures + options OI) is dead. RC p=0.65, best |t| 2.74 BELOW the
    null median of 2.96. Nothing to stress.
  * PANEL A best = TOP10.adv -> overnight gap, RC p=0.045, but in an OLS beside
    close-strength the signal drops to t=+1.51 while CLR holds t=+3.38. Known.
  * NEW: at the WEEK horizon CLR dies (t=+0.01) and breadth survives (t=+2.08).
    v1 never ran h=5, so this could not have been seen before.

THREE STRESSES
  S1 OVERLAP. h=5 daily observations share 4 of 5 days. Newey-West assumes that
     away asymptotically; with ~1,300 rows the EFFECTIVE n is ~260. Re-run on
     strictly NON-OVERLAPPING weekly blocks, which needs no HAC assumption.
  S2 THE GATE. "delivery up AND OI up AND price up" fired 31 times in 496 days
     and printed t=nan because n<40. Its INVERSE also paid (+50bps at h=5), which
     is the signature of a time-clustered sample, not a direction. Block bootstrap.
  S3 OOS. Fit nothing, but split 2022-2024 / 2025-2026 and check the sign holds.
     Then subtract cost.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option("display.width", 250)
rng = np.random.default_rng(29)
c = duckdb.connect("data/market_data.duckdb", read_only=True)

TOP10 = ["RELIANCE", "BHARTIARTL", "HDFCBANK", "ICICIBANK", "SBIN", "TCS",
         "BAJFINANCE", "LT", "HINDUNILVR", "INFY"]
NEXT10 = ["SUNPHARMA", "TITAN", "KOTAKBANK", "MARUTI", "ADANIENT", "AXISBANK",
          "M&M", "ADANIPORTS", "HCLTECH", "ULTRACEMCO"]
REST30 = ["APOLLOHOSP", "ASIANPAINT", "BAJAJ-AUTO", "BAJAJFINSV", "BEL", "CIPLA",
          "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HDFCLIFE",
          "HINDALCO", "ITC", "INDIGO", "JSWSTEEL", "JIOFIN", "MAXHEALTH", "NTPC",
          "NESTLEIND", "ONGC", "POWERGRID", "SBILIFE", "SHRIRAMFIN", "TATACONSUM",
          "TMPV", "TATASTEEL", "TECHM", "TRENT", "WIPRO"]
ALL50 = TOP10 + NEXT10 + REST30
ph = ",".join("?" * len(ALL50))

cash = c.execute(f"""SELECT trade_date, symbol, deliv_per,
        (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND close_price>0 AND prev_close>0 AND trade_date>='2021-06-01'""", ALL50).df()
cash["trade_date"] = pd.to_datetime(cash["trade_date"])
cash = cash[cash["r"].abs() < 40]
R = cash.pivot_table("r", "trade_date", "symbol").sort_index()
DL = cash.pivot_table("deliv_per", "trade_date", "symbol").sort_index()

nif = c.execute("""SELECT trade_date,open_val,high_val,low_val,close_val,pct_chg
    FROM index_data WHERE index_name='Nifty 50' AND trade_date>='2021-06-01'""").df()
nif["trade_date"] = pd.to_datetime(nif["trade_date"])
nif = nif.set_index("trade_date").sort_index().astype(float)
IX = R.index.intersection(nif.index)
y = nif.loc[IX, "pct_chg"]
lg = np.log1p(y / 100.0)
F5 = np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100.0
F1 = y.shift(-1)
GAP = (nif["open_val"] / nif["close_val"].shift(1) - 1).shift(-1).reindex(IX) * 100
CLR = ((nif["close_val"] - nif["low_val"]) /
       (nif["high_val"] - nif["low_val"]).replace(0, np.nan)).reindex(IX)


def adv(cols):
    s = R[[x for x in cols if x in R.columns]]
    return (s > 0).sum(axis=1) / s.notna().sum(axis=1) * 100 - 50


def persym_z(M, cols, base=100):
    S = M[[s for s in cols if s in M.columns]]
    mu, sd = S.rolling(base).mean().shift(1), S.rolling(base).std().shift(1)
    return ((S - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan).mean(axis=1)


A10, A20, A30 = adv(TOP10), adv(NEXT10), adv(REST30)
DIV = A10 - A30
DZ10, DZ30 = persym_z(DL, TOP10), persym_z(DL, REST30)

print("=" * 100)
print("S1  NON-OVERLAPPING WEEKS - does the h=5 breadth result survive without HAC?")
print("=" * 100)
print("  Daily h=5 rows share 4 of 5 days. Effective n is ~1/5 of the nominal n.")
for nm, s in [("TOP10 adv", A10), ("NEXT10 adv", A20), ("REST30 adv", A30),
              ("DIVERGE adv", DIV), ("TOP10 deliv_z", DZ10),
              ("DIVERGE deliv_z", DZ10 - DZ30)]:
    k = pd.DataFrame({"s": s, "f": F5}).dropna().astype(float)
    for off in (0,):
        kk = k.iloc[off::5]                      # strictly disjoint 5-day windows
        sg = np.sign(kk["s"]) * kk["f"]
        t = sg.mean() / (sg.std(ddof=1) / math.sqrt(len(sg)))
        ic = kk["s"].rank().corr(kk["f"].rank())
        print(f"  {nm:16s} disjoint weeks n={len(kk):4d}  "
              f"{sg.mean()*100:+7.1f}bps  t={t:+5.2f}  IC={ic:+.3f}  "
              f"hit={(sg>0).mean()*100:.1f}%")
# all 5 phase offsets, to show the answer is not an alignment artifact
print("\n  every phase offset for DIVERGE adv (an offset-dependent result is noise):")
k = pd.DataFrame({"s": DIV, "f": F5}).dropna().astype(float)
for off in range(5):
    kk = k.iloc[off::5]
    sg = np.sign(kk["s"]) * kk["f"]
    print(f"    offset {off}: n={len(kk)}  {sg.mean()*100:+7.1f}bps  "
          f"t={sg.mean()/(sg.std(ddof=1)/math.sqrt(len(sg))):+5.2f}")

print("\n" + "=" * 100)
print("S2  THE CONJUNCTION GATE - 31 events, and its INVERSE also paid")
print("=" * 100)
fut = c.execute(f"""SELECT trade_date, symbol, expiry_date, open_interest
    FROM fno_bhavcopy WHERE instrument='FUTSTK' AND open_interest>0
      AND symbol IN ({ph}) AND expiry_date > trade_date""", ALL50).df()
fut["trade_date"] = pd.to_datetime(fut["trade_date"])
SETTLE = set(pd.to_datetime(c.execute(
    "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK'"
    " AND expiry_date=trade_date").df()["trade_date"]))
days = sorted(fut["trade_date"].unique())
g = {k_: v for k_, v in fut.groupby("trade_date")}
rowsf = []
for d, p in zip(days[1:], days[:-1]):
    if d in SETTLE:
        continue
    m = g[d].merge(g[p], on=["symbol", "expiry_date"], suffixes=("", "_p"))
    if m.empty:
        continue
    s = m.groupby("symbol")[["open_interest", "open_interest_p"]].sum()
    s = s[s["open_interest_p"] > 0]
    s["pct"] = ((s["open_interest"] - s["open_interest_p"]) /
                s["open_interest_p"] * 100).clip(-50, 50)
    s["trade_date"] = d
    rowsf.append(s.reset_index())
FO = pd.concat(rowsf).pivot_table("pct", "trade_date", "symbol")
FOI = FO[[s for s in TOP10 if s in FO.columns]].mean(axis=1)
RET10 = R[[s for s in TOP10 if s in R.columns]].mean(axis=1)

k = pd.DataFrame({"d": DZ10, "f": FOI, "r": RET10, "y5": F5, "y1": F1,
                  "gap": GAP}).dropna().astype(float)
on = (k.d > 0.5) & (k.f > 0) & (k.r > 0)
off = (k.d < -0.5) & (k.f < 0) & (k.r < 0)
print(f"  panel n={len(k)}   BULL {on.sum()} events   BEAR {off.sum()} events")
for hn, col in [("gap", "gap"), ("1d", "y1"), ("5d", "y5")]:
    a, b_, base = k[col][on], k[col][off], k[col]
    # block bootstrap the DIFFERENCE, respecting event clustering
    obs = a.mean() - base.mean()
    idx = np.arange(len(k))
    nulls = np.empty(4000)
    for i in range(4000):
        sh = rng.integers(0, len(k))
        perm = np.roll(idx, sh)                  # circular shift preserves clusters
        nulls[i] = k[col].values[perm][on.values].mean() - base.mean()
    p = (np.abs(nulls) >= abs(obs)).mean()
    print(f"  {hn:4s} BULL {a.mean()*100:+7.1f}bps (up {(a>0).mean()*100:.0f}%)   "
          f"BEAR {b_.mean()*100:+7.1f}bps (up {(b_>0).mean()*100:.0f}%)   "
          f"base {base.mean()*100:+6.1f}   excess {obs*100:+6.1f}bps  "
          f"circular-shift p={p:.3f}")
print("\n  event clustering - how many distinct months do the 31 BULL days fall in?")
bd = k.index[on]
print(f"    {len(bd)} days across {len(set(zip(bd.year, bd.month)))} months; "
      f"longest run {max((sum(1 for _ in grp) for _, grp in __import__('itertools').groupby(np.diff(np.where(on.values)[0]) == 1)), default=0)} consecutive")

print("\n" + "=" * 100)
print("S3  OOS SPLIT + COST - the sign has to hold out of sample and clear the floor")
print("=" * 100)
CUT = pd.Timestamp("2025-01-01")
for nm, s, f, hz in [("TOP10 adv", A10, GAP, "gap"), ("TOP10 adv", A10, F5, "5d"),
                     ("DIVERGE adv", DIV, F5, "5d"),
                     ("REST30 adv", A30, GAP, "gap"),
                     ("DIVERGE deliv_z", DZ10 - DZ30, F5, "5d")]:
    k = pd.DataFrame({"s": s, "f": f}).dropna().astype(float)
    sg = np.sign(k["s"]) * k["f"]
    a, b_ = sg[sg.index < CUT], sg[sg.index >= CUT]
    ta = a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))
    tb = b_.mean() / (b_.std(ddof=1) / math.sqrt(len(b_)))
    print(f"  {nm:16s} h={hz:3s}  IN  n={len(a):4d} {a.mean()*100:+7.1f}bps t={ta:+5.2f}"
          f"   OUT n={len(b_):4d} {b_.mean()*100:+7.1f}bps t={tb:+5.2f}")
print("\n  cost floor: NIFTY futures round trip ~2-4bps; capturing an OVERNIGHT GAP")
print("  means trading the close and the open, which this repo measured at 2-6bps.")

print("\n" + "=" * 100)
print("S4  IS THE TOP-10 READ SEPARABLE FROM PLAIN BREADTH? (the whole premise)")
print("=" * 100)
print("  If top-10 breadth and rest-30 breadth carry the SAME information, then")
print("  bucketing by weight buys nothing and the panel is a decomposition only.")
b = pd.DataFrame({"a10": A10, "a20": A20, "a30": A30}).dropna()
print(f"  corr(top10 adv, rest30 adv)  = {b['a10'].corr(b['a30']):+.3f}")
print(f"  corr(top10 adv, next10 adv)  = {b['a10'].corr(b['a20']):+.3f}")
for f, hn in [(GAP, "gap"), (F5, "5d")]:
    k = pd.DataFrame({"f": f, "a10": A10, "a30": A30}).dropna().astype(float)
    X = np.column_stack([np.ones(len(k)), k["a10"], k["a30"]])
    bb, *_ = np.linalg.lstsq(X, k["f"].values, rcond=None)
    e = k["f"].values - X @ bb
    XtXi = np.linalg.inv(X.T @ X)
    S = X * e[:, None]
    meat = S.T @ S
    L = 1 if hn == "gap" else 5
    for l_ in range(1, L + 1):
        G = S[l_:].T @ S[:-l_]
        meat += (1 - l_ / (L + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    print(f"  h={hn:3s} horse race  top10 t={bb[1]/se[1]:+5.2f}   "
          f"rest30 t={bb[2]/se[2]:+5.2f}   (both in the same regression)")
