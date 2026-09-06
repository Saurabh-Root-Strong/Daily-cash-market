"""The COMPOSITE the user actually described, built and then tested honestly.

WHAT IS NEW HERE (i.e. what the earlier files did NOT test)
  1 The OPTIONS read as a RELATIVE CE-vs-PE flow. nifty_bucket_patterns.py scored
    options against the STOCK's own move ("CE OI up while price fell = call
    writing"). The user's construction is different: "call side OI increasing,
    put side OI DECREASING" - a divergence between the two sides, independent of
    price. Untested until now, and it is a genuinely different signal.
  2 A WEIGHTED COMPOSITE across all three buckets. Everything before this tested
    buckets separately or required them to agree exactly. "Summarise these things
    entirely" is a scoring problem, not an agreement problem.

CONSTRUCTION, per bucket, each leg in [-1, +1]
    delivery  +1 if the bucket's per-symbol delivery z > +0.3, -1 if < -0.3
    futures   OI-price matrix on ROLL-IMMUNE total forward OI:
              LONG BUILDUP / SHORT COVERING = +1, SHORT BUILDUP / LONG UNWINDING = -1
    options   sign(PE OI change - CE OI change): put side building relative to the
              call side = +1 (put writing), call side building = -1 (call writing)
    bucket score = mean of the three legs
    NET = 0.46*TOP10 + 0.24*NEXT10 + 0.30*REST30, roughly the published weight
          share of each bucket. Equal-weight and top-10-only variants run beside
          it so the answer cannot be an artifact of the weighting.

HOW IT IS JUDGED. Not on whether the number looks sensible - on whether it beats
    (a) the unconditional base rate, and
    (b) the index's own close-strength, which has beaten every constituent signal
        tested in this repo so far.
    Plus a walk-forward split, because a score assembled after seeing the data is
    not evidence until it survives out of sample.
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option("display.width", 250)
rng = np.random.default_rng(51)

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
WEIGHT = {"TOP10": 0.46, "NEXT10": 0.24, "REST30": 0.30}
ALL50 = TOP10 + NEXT10 + REST30
ph = ",".join("?" * len(ALL50))

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2022-01-01")
a = ap.parse_args()
WARM = (pd.Timestamp(a.start) - pd.Timedelta(days=220)).date().isoformat()
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

IX = R.index.intersection(nif.index)
y = nif.loc[IX, "pct_chg"]
lg = np.log1p(y / 100.0)
F5 = np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100.0
F1 = y.shift(-1)
CLR = ((nif["close_val"] - nif["low_val"]) /
       (nif["high_val"] - nif["low_val"]).replace(0, np.nan)).reindex(IX)


def persym_z(M, cols, base=100):
    S = M[[s for s in cols if s in M.columns]]
    mu, sd = S.rolling(base).mean().shift(1), S.rolling(base).std().shift(1)
    return ((S - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan).mean(axis=1)


LEGS = {}
for bn, cols in BUCK.items():
    cols = [s for s in cols if s in R.columns]
    sub = R[cols]
    f = FOI[[s for s in cols if s in FOI.columns]].reindex(sub.index)
    ce = CE[[s for s in cols if s in CE.columns]].reindex(sub.index)
    pe = PE[[s for s in cols if s in PE.columns]].reindex(sub.index)
    dz = persym_z(DL, cols)
    dleg = pd.Series(np.where(dz > 0.3, 1.0, np.where(dz < -0.3, -1.0, 0.0)),
                     index=dz.index).where(dz.notna())
    # futures: OI-price matrix per symbol, averaged. Long buildup and short
    # covering are both price-up; short buildup and long unwinding both price-down.
    fm = (np.sign(sub) * np.where(f.abs() > 0.5, 1.0, 0.0))
    fleg = pd.DataFrame(fm, index=sub.index, columns=sub.columns).where(
        f.notna()).mean(axis=1)
    fleg = fleg / fleg.abs().rolling(250).max().clip(lower=0.05)   # scale to ~[-1,1]
    # options: the user's read - call side building vs put side building
    oleg = np.sign(pe - ce).mean(axis=1)
    LEGS[bn] = pd.DataFrame({"deliv": dleg, "fut": fleg.clip(-1, 1), "opt": oleg})
    LEGS[bn]["score"] = LEGS[bn][["deliv", "fut", "opt"]].mean(axis=1)

NET = sum(LEGS[b]["score"] * WEIGHT[b] for b in BUCK)
EQ = sum(LEGS[b]["score"] for b in BUCK) / 3
T10 = LEGS["TOP10"]["score"]

K = pd.DataFrame({"net": NET, "eq": EQ, "t10": T10, "clr": CLR - 0.5,
                  "mom": y, "d1": F1, "d5": F5}).dropna()
K = K[K.index >= pd.Timestamp(a.start)]
print(f"panel {len(K)} sessions {K.index.min():%Y-%m-%d}..{K.index.max():%Y-%m-%d}")
print(f"BASE  next day {K.d1.mean():+.3f}% (up {(K.d1>0).mean()*100:.1f}%)   "
      f"next 5 {K.d5.mean():+.3f}% (up {(K.d5>0).mean()*100:.1f}%)")
print(f"score spread: net sd {K.net.std():.3f}  range {K.net.min():+.2f}..{K.net.max():+.2f}")


def nw(x, lag):
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < 40:
        return float("nan")
    e = x - x.mean()
    v = (e @ e) / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / math.sqrt(v / n) if v > 0 else float("nan")


print("\n" + "=" * 100)
print("DOES THE COMPOSITE FORECAST? (signed return, Newey-West at lag = horizon)")
print("=" * 100)
for nm in ("net", "eq", "t10", "clr", "mom"):
    for h, lag in (("d1", 1), ("d5", 5)):
        sg = np.sign(K[nm]) * K[h]
        ic = K[nm].rank().corr(K[h].rank())
        tag = "  <- CONTROL" if nm in ("clr", "mom") else ""
        print(f"  {nm:4s} {h}  IC {ic:+.4f}  {sg.mean():+.3f}%  t={nw(sg,lag):+5.2f}  "
              f"hit {(sg>0).mean()*100:.1f}%{tag}")

print("\n" + "=" * 100)
print("QUINTILES - is the score monotone in the outcome?")
print("=" * 100)
for nm in ("net", "t10"):
    for h in ("d1", "d5"):
        k = K.copy()
        k["q"] = pd.qcut(k[nm], 5, labels=False, duplicates="drop")
        g = k.groupby("q")[h].agg(n="size", pct="mean",
                                  up=lambda v: (v > 0).mean() * 100)
        print(f"  {nm} {h}: " + "  ".join(
            f"Q{int(q)+1}:{r.pct:+.3f}%({r.up:.0f}%,n{int(r.n)})"
            for q, r in g.iterrows()))

print("\n" + "=" * 100)
print("INCREMENTAL - anything left after close-strength and momentum?")
print("=" * 100)
for h, lag in (("d1", 1), ("d5", 5)):
    k = K.dropna()
    X = np.column_stack([np.ones(len(k)), k["mom"], k["clr"], k["net"]])
    b, *_ = np.linalg.lstsq(X, k[h].values, rcond=None)
    e = k[h].values - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = X * e[:, None]
    meat = S.T @ S
    for L in range(1, lag + 1):
        G = S[L:].T @ S[:-L]
        meat += (1 - L / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    print(f"  {h}  n={len(k)}  mom t={b[1]/se[1]:+5.2f}  clr t={b[2]/se[2]:+5.2f}  "
          f"COMPOSITE t={b[3]/se[3]:+5.2f}")

print("\n" + "=" * 100)
print("WALK-FORWARD - the sign has to hold out of sample")
print("=" * 100)
CUT = pd.Timestamp("2024-09-01")
for nm in ("net", "t10"):
    for h in ("d1", "d5"):
        sg = np.sign(K[nm]) * K[h]
        i_, o_ = sg[sg.index < CUT], sg[sg.index >= CUT]
        print(f"  {nm} {h}  IN  n={len(i_):4d} {i_.mean():+.3f}% hit {(i_>0).mean()*100:.1f}%"
              f"    OUT n={len(o_):4d} {o_.mean():+.3f}% hit {(o_>0).mean()*100:.1f}%")

print("\n" + "=" * 100)
print("WHAT THE SCORE IS WORTH AS A DISPLAYED CALL (strong readings only)")
print("=" * 100)
for nm in ("net", "t10"):
    for cut in (0.20, 0.35, 0.50):
        for h in ("d1", "d5"):
            up = K[K[nm] >= cut][h]
            dn = K[K[nm] <= -cut][h]
            if len(up) < 25 or len(dn) < 25:
                continue
            print(f"  {nm} |score|>={cut:.2f} {h}:  BULL n={len(up):4d} {up.mean():+.3f}% "
                  f"(up {(up>0).mean()*100:.1f}%)   BEAR n={len(dn):4d} {dn.mean():+.3f}% "
                  f"(up {(dn>0).mean()*100:.1f}%)   spread {up.mean()-dn.mean():+.3f}pp")

print("\n" + "=" * 100)
print("LEG-BY-LEG - which leg, if any, carries the composite?")
print("=" * 100)
for bn in BUCK:
    for leg in ("deliv", "fut", "opt"):
        v = LEGS[bn][leg].reindex(K.index)
        for h, lag in (("d1", 1), ("d5", 5)):
            sg = np.sign(v) * K[h]
            t = nw(sg, lag)
            if abs(t) >= 1.6:
                print(f"  {bn:7s} {leg:6s} {h}  {sg.mean():+.3f}%  t={t:+5.2f}  "
                      f"hit {(sg>0).mean()*100:.1f}%  n={sg.notna().sum()}")
print("  (only |t| >= 1.6 shown; nothing printed means no leg reaches even that)")
