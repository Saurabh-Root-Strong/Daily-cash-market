"""Stage 4 - decompose the one thing still standing.

The 3-leg BULL gate (top-10 delivery z>0.5 AND total forward futures OI up AND
top-10 price up) pays +76bps over the next 5 sessions against a -2bps base, and
it is NOT one episode: 21 independent episodes, t=+2.02, and dropping any single
episode leaves +57 to +94bps. My "it's one market episode" hypothesis was wrong.

So the question becomes: WHICH LEG. A 3-way AND on 492 days is a nested search -
there are 7 non-empty subsets of the legs, and picking the best of 7 after seeing
them is exactly how a 6.3%-frequency artifact gets born. Every subset is priced
here, at the same horizon, with episode-clustered inference.

The tell to watch for: if the 1-leg and 2-leg versions pay NOTHING and only the
full 3-way AND pays, the gate is not "three confirming signals" - it is a
6%-of-days subsample that happens to sit in good weeks.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd
from itertools import combinations

pd.set_option("display.width", 250)
rng = np.random.default_rng(37)
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
CLR = ((nif["close_val"] - nif["low_val"]) /
       (nif["high_val"] - nif["low_val"]).replace(0, np.nan)).reindex(IX)

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


def persym_z(M, cols, base=100):
    S = M[[s for s in cols if s in M.columns]]
    mu, sd = S.rolling(base).mean().shift(1), S.rolling(base).std().shift(1)
    return ((S - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan).mean(axis=1)


K = pd.DataFrame({
    "d": persym_z(DL, TOP10),
    "f": FO[[s for s in TOP10 if s in FO.columns]].mean(axis=1),
    "r": R[[s for s in TOP10 if s in R.columns]].mean(axis=1),
    "clr": CLR, "y": F5}).dropna().astype(float)
BASE = K["y"].mean() * 100
print(f"panel n={len(K)}  {K.index.min():%Y-%m-%d}..{K.index.max():%Y-%m-%d}  "
      f"base 5d {BASE:+.1f}bps")

LEG = {"deliv": (K.d > 0.5).values, "futOI": (K.f > 0).values, "price": (K.r > 0).values}


def episodes(mask, maxgap=5):
    pos = np.where(mask)[0]
    if len(pos) == 0:
        return []
    epi, cur = [], [int(pos[0])]
    for a, b_ in zip(pos, pos[1:]):
        if b_ - a <= maxgap:
            cur.append(int(b_))
        else:
            epi.append(cur)
            cur = [int(b_)]
    epi.append(cur)
    return epi


def score(mask, label):
    n = int(mask.sum())
    if n < 8:
        return None
    epi = episodes(mask)
    em = np.array([K["y"].values[e].mean() * 100 for e in epi])
    t = ((em.mean() - BASE) / (em.std(ddof=1) / math.sqrt(len(em)))
         if len(em) > 2 and em.std(ddof=1) > 0 else float("nan"))
    m = K["y"].values[mask].mean() * 100
    print(f"  {label:34s} fires {n:3d}d ({n/len(K)*100:4.1f}%)  {len(epi):2d} episodes  "
          f"{m:+7.1f}bps  excess {m-BASE:+7.1f}  up {(K['y'].values[mask]>0).mean()*100:4.0f}%"
          f"  episode-t {t:+5.2f}")
    return m - BASE


print("\n" + "=" * 104)
print("ALL 7 LEG SUBSETS - if only the 3-way AND pays, it is a subsample, not confirmation")
print("=" * 104)
for k_ in (1, 2, 3):
    for combo in combinations(LEG, k_):
        m = np.logical_and.reduce([LEG[x] for x in combo])
        score(m, " AND ".join(combo))

print("\n" + "=" * 104)
print("THE BEAR SIDE - the mirror gate paid +50bps too. A direction signal cannot")
print("do that. Pricing every subset's inverse:")
print("=" * 104)
NEG = {"deliv": (K.d < -0.5).values, "futOI": (K.f < 0).values, "price": (K.r < 0).values}
for k_ in (1, 2, 3):
    for combo in combinations(NEG, k_):
        m = np.logical_and.reduce([NEG[x] for x in combo])
        score(m, "NOT " + " AND NOT ".join(combo))

print("\n" + "=" * 104)
print("RANDOM-GATE CONTROL - how big an excess does a 6.3%-of-days random pick give?")
print("=" * 104)
full = np.logical_and.reduce([LEG[x] for x in LEG])
nfire = int(full.sum())
obs = K["y"].values[full].mean() * 100 - BASE
# resample CONTIGUOUS blocks matched to the real gate's episode-size profile
sizes = [len(e) for e in episodes(full)]
draws = np.empty(5000)
for i in range(5000):
    sel = np.zeros(len(K), bool)
    for sz in sizes:
        st = rng.integers(0, len(K) - sz)
        sel[st:st + sz] = True
    draws[i] = K["y"].values[sel].mean() * 100 - BASE
print(f"  real gate: {nfire} days in {len(sizes)} episodes, excess {obs:+.1f}bps")
print(f"  {len(sizes)} random blocks of the same sizes: mean {draws.mean():+.1f}  "
      f"sd {draws.std():.1f}  95th {np.percentile(draws,95):+.1f}  "
      f"99th {np.percentile(draws,99):+.1f}")
print(f"  p(random >= real) = {(draws >= obs).mean():.4f}")

print("\n" + "=" * 104)
print("AND THE ONE CONTROL THAT HAS KILLED EVERY SIGNAL IN THIS REPO")
print("=" * 104)
hi = (K["clr"] > K["clr"].quantile(0.8)).values
print(f"  gate days that are ALSO a top-quintile close: "
      f"{int((full & hi).sum())} of {nfire} ({(full & hi).sum()/max(nfire,1)*100:.0f}%) "
      f"vs {hi.mean()*100:.0f}% unconditionally")
score(hi, "CLR top quintile alone")
score(full & ~hi, "gate WITHOUT a strong close")
print("\n  If the gate keeps its excess on days that did NOT close strong, it is")
print("  carrying information close-strength does not. That is the only way this")
print("  survives, because CLR beat every bucket signal at the gap horizon.")
