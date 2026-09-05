"""Stage 3 - close it out. Three questions left, then the descriptive deliverable.

STAGE 2 VERDICT SO FAR
  The h=5 breadth result was an artifact of overlapping windows. On strictly
  disjoint weeks it is t=+0.12, and the five phase offsets run -8.5 to +28.6 bps
  (t -0.71 to +2.40) - the answer depends on which weekday you start counting.
  The OOS half kills every candidate except the overnight gap, which decays.

REMAINING
  Q1 The conjunction gate's 5-day excess had circular-shift p=0.021, but its 31
     events contain an 18-day consecutive run. Leave-one-episode-out.
  Q2 Top-10 breadth beat rest-30 breadth in a horse race (t +2.91 vs -0.09). Does
     it beat CLOSE-STRENGTH, or is it the same thing wearing a different hat?
  Q3 What size of edge could this panel even DETECT? If the answer is "nothing
     under 8bps", a null result is uninformative and must be reported as such.
  D  The descriptive table the user actually asked for: given a bucket state
     today, what has NIFTY done the next day and the next week?

Evaluation window is 2022-01-01+ throughout. Stage 2 leaked 2021-H2 in because
the cash pull starts 2021-06 to warm the 100-day rolling baselines.
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option("display.width", 250)
rng = np.random.default_rng(31)
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
START = pd.Timestamp("2022-01-01")

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


A10, A30 = adv(TOP10), adv(REST30)
DZ10 = persym_z(DL, TOP10)

print("=" * 100)
print("Q1  THE GATE, MINUS ITS BIGGEST EPISODE")
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
rf = []
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
    rf.append(s.reset_index())
FO = pd.concat(rf).pivot_table("pct", "trade_date", "symbol")
FOI = FO[[s for s in TOP10 if s in FO.columns]].mean(axis=1)
RET10 = R[[s for s in TOP10 if s in R.columns]].mean(axis=1)

k = pd.DataFrame({"d": DZ10, "f": FOI, "r": RET10, "y5": F5}).dropna().astype(float)
on = ((k.d > 0.5) & (k.f > 0) & (k.r > 0)).values
pos = np.where(on)[0]
# split the events into episodes: a gap of >5 sessions starts a new one
epi, cur = [], [int(pos[0])]
for a, b_ in zip(pos, pos[1:]):
    if b_ - a <= 5:                     # same episode: forward windows overlap
        cur.append(int(b_))
    else:
        epi.append(cur)
        cur = [int(b_)]
epi.append(cur)
assert sum(len(e) for e in epi) == len(pos), "episode split lost events"
print(f"  {len(pos)} BULL events fall into {len(epi)} distinct episodes: "
      f"sizes {sorted((len(e) for e in epi), reverse=True)}")
allm = k["y5"].values[pos].mean() * 100
base = k["y5"].mean() * 100
print(f"  all events  {allm:+7.1f}bps   base {base:+.1f}bps   "
      f"excess {allm-base:+.1f}bps")
# one episode = one independent draw. Average the EPISODE means, then t-test them.
em = np.array([k["y5"].values[e].mean() * 100 for e in epi])
te = (em.mean() - base) / (em.std(ddof=1) / math.sqrt(len(em)))
print(f"  episode means: {len(em)} independent draws, mean {em.mean():+.1f}bps, "
      f"sd {em.std(ddof=1):.1f}  =>  t vs base = {te:+.2f}")
for i, e in enumerate(sorted(epi, key=len, reverse=True)[:4]):
    rest = [p for p in pos if p not in e]
    m = k["y5"].values[rest].mean() * 100
    print(f"  drop episode {i+1} ({len(e)}d, {k.index[e[0]]:%Y-%m-%d}): "
          f"remaining {len(rest)}d -> {m:+7.1f}bps  excess {m-base:+.1f}bps")
print("  => an 'edge' that needs one episode to survive is that episode, not an edge.")

print("\n" + "=" * 100)
print("Q2  TOP-10 BREADTH vs CLOSE-STRENGTH at the overnight gap")
print("=" * 100)


def ols(k, cols, lag):
    X = np.column_stack([np.ones(len(k))] + [k[c].values for c in cols])
    b, *_ = np.linalg.lstsq(X, k["f"].values, rcond=None)
    e = k["f"].values - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = X * e[:, None]
    meat = S.T @ S
    for L in range(1, lag + 1):
        G = S[L:].T @ S[:-L]
        meat += (1 - L / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return b, se


kk = pd.DataFrame({"f": GAP, "a10": A10, "a30": A30, "clr": CLR - 0.5,
                   "m": y}).dropna().astype(float)
kk = kk[kk.index >= START]
for cols in (["a10"], ["a10", "a30"], ["a10", "clr"], ["a10", "a30", "clr", "m"]):
    b, se = ols(kk, cols, 1)
    print("  " + "  ".join(f"{c}={b[i+1]/se[i+1]:+5.2f}" for i, c in enumerate(cols))
          + f"    (n={len(kk)}, model = {' + '.join(cols)})")
print(f"\n  corr(top10 adv, CLR) = {kk['a10'].corr(kk['clr']):+.3f}")
print("  => breadth and close-strength are two readings of the SAME session shape.")

print("\n" + "=" * 100)
print("Q3  POWER - what edge could 1,150 sessions even detect?")
print("=" * 100)
for hn, f in [("gap", GAP), ("1d", F1), ("5d", F5)]:
    v = f[f.index >= START].dropna()
    sd = v.std()
    n = len(v)
    mde = 2.8 * sd / math.sqrt(n) * 100          # 80% power, two-sided 5%
    neff = n / (5 if hn == "5d" else 1)
    mde_e = 2.8 * sd / math.sqrt(neff) * 100
    print(f"  {hn:4s} n={n:4d}  sd={sd:.2f}%  minimum detectable edge "
          f"{mde:5.1f}bps  (on non-overlapping n={int(neff)}: {mde_e:5.1f}bps)")
print("  Cost floor for a NIFTY futures round trip is 2-4bps, and gap capture")
print("  2-6bps. At 5 days the panel cannot resolve anything under ~30bps, so a")
print("  null there is WEAK evidence, not proof of absence. The gap column is the")
print("  only horizon where this data can see an economically relevant effect.")

print("\n" + "=" * 100)
print("D   DESCRIPTIVE - what NIFTY actually did after each bucket state (2022+)")
print("=" * 100)
st = pd.DataFrame({"a10": A10, "a30": A30, "dz10": DZ10, "gap": GAP,
                   "d1": F1, "d5": F5}).dropna()
st = st[st.index >= START]
st["state"] = np.select(
    [(st.a10 > 0) & (st.a30 > 0), (st.a10 > 0) & (st.a30 <= 0),
     (st.a10 <= 0) & (st.a30 > 0)],
    ["broad up", "TOP-10 CARRIED (narrow up)", "top-10 lagged (broad up)"],
    default="broad down")
t = st.groupby("state").agg(
    days=("gap", "size"),
    gap_bps=("gap", lambda v: round(v.mean() * 100, 1)),
    gap_up=("gap", lambda v: round((v > 0).mean() * 100, 1)),
    d1_bps=("d1", lambda v: round(v.mean() * 100, 1)),
    d1_up=("d1", lambda v: round((v > 0).mean() * 100, 1)),
    d5_bps=("d5", lambda v: round(v.mean() * 100, 1)),
    d5_up=("d5", lambda v: round((v > 0).mean() * 100, 1)))
print(t.to_string())
print(f"\n  unconditional base: gap {st.gap.mean()*100:+.1f}bps "
      f"(up {(st.gap>0).mean()*100:.1f}%)   1d {st.d1.mean()*100:+.1f}bps "
      f"(up {(st.d1>0).mean()*100:.1f}%)   5d {st.d5.mean()*100:+.1f}bps "
      f"(up {(st.d5>0).mean()*100:.1f}%)")
print("\n  same split, but conditioning on TOP-10 DELIVERY as well:")
st["dz"] = np.where(st.dz10 > 0.5, "deliv HIGH", np.where(st.dz10 < -0.5, "deliv LOW", "deliv mid"))
t2 = st.groupby(["state", "dz"]).agg(
    days=("gap", "size"), gap_bps=("gap", lambda v: round(v.mean() * 100, 1)),
    d1_bps=("d1", lambda v: round(v.mean() * 100, 1)),
    d5_bps=("d5", lambda v: round(v.mean() * 100, 1)),
    d5_up=("d5", lambda v: round((v > 0).mean() * 100, 1)))
print(t2[t2.days >= 25].to_string())
