"""Does WEIGHT-BUCKETED constituent flow forecast NIFTY? v2 - the full thesis.

WHAT v1 GOT WRONG, and why this file exists
  1 v1's futures leg used the NEAR-MONTH contract with only the settlement day
    dropped. The ILC self-audit later measured near-month OI at DTE 0-2:
    mean -47.95%, 99.94% of rows below -5%. Positions MIGRATE to the next
    contract; they do not close. So ~14% of v1's futures rows were a roll
    artifact. Here every futures/options figure is TOTAL OI across every live
    expiry, both sessions gated to the SAME forward expiry set.
  2 v1 computed FWD[5] and never tested it. The user asked for the week.
  3 v1 never touched OPTIONS at all.
  4 v1 tested each signal ALONE. The user's thesis is a CONJUNCTION - "heavy
    delivery AND rising OI AND call buying, together". Tested here as a
    composite and as a hard AND-gate.

TWO PANELS, NOT ONE. daily_data starts 2018 but fno_bhavcopy starts 2024-07-24.
Mixing them would silently shrink the cash test to the F&O window. So:
    PANEL A  cash flow  2022-01 .. 2026-09   (~1,150 sessions)
    PANEL B  F&O flow   2024-07 .. 2026-09   (~520 sessions)
Panel B is HALF a market cycle. Its power is stated, not assumed.

FALSE-POSITIVE CONTROLS (each one killed a previous result in this repo)
  CONTEMPORANEOUS IDENTITY  top-10 return IS ~46% of the index. All forward.
  MOMENTUM + CLR            every constituent signal tested here so far lost to
                            the index's own close-strength. Both run beside every
                            cell and in an incremental OLS.
  MULTIPLICITY              max|t| vs a stationary date-block bootstrap.
  OVERLAP                   Newey-West at lag = horizon.
  SURVIVORSHIP              today's 50 on 2022 data. Quantified at the end.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option('display.width', 250)
rng = np.random.default_rng(17)
c = duckdb.connect('data/market_data.duckdb', read_only=True)

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
BUCK = {"TOP10": TOP10, "NEXT10": NEXT10, "REST30": REST30}
ph = ",".join("?" * len(ALL50))
DELIV_BASE = 100

# -- cash --------------------------------------------------------------------
cash = c.execute(f"""SELECT trade_date, symbol, deliv_per, turnover_lacs,
        deliv_qty*avg_price/100000.0 AS deliv_val_lacs,
        (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND close_price>0 AND prev_close>0 AND trade_date>='2021-06-01'""", ALL50).df()
cash["trade_date"] = pd.to_datetime(cash["trade_date"])
cash = cash[cash["r"].abs() < 40]                      # corporate actions
R = cash.pivot_table("r", "trade_date", "symbol").sort_index()
DL = cash.pivot_table("deliv_per", "trade_date", "symbol").sort_index()
DV = cash.pivot_table("deliv_val_lacs", "trade_date", "symbol").sort_index()

# -- futures: TOTAL forward OI, same expiry set both sessions -----------------
fut = c.execute(f"""SELECT trade_date, symbol, expiry_date, open_interest
    FROM fno_bhavcopy WHERE instrument='FUTSTK' AND open_interest>0
      AND symbol IN ({ph}) AND expiry_date > trade_date""", ALL50).df()
fut["trade_date"] = pd.to_datetime(fut["trade_date"])

# -- options: TOTAL forward OI by side ---------------------------------------
opt = c.execute(f"""SELECT trade_date, symbol, expiry_date, option_type,
        SUM(open_interest) oi
    FROM fno_bhavcopy WHERE instrument='OPTSTK' AND open_interest>0
      AND symbol IN ({ph}) AND expiry_date > trade_date
    GROUP BY 1,2,3,4""", ALL50).df()
opt["trade_date"] = pd.to_datetime(opt["trade_date"])

# settlement sessions: forward OI jumps +14.25% vs +0.47% elsewhere
SETTLE = set(pd.to_datetime(c.execute(
    "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK'"
    " AND expiry_date=trade_date").df()["trade_date"]))


def fwd_oi_change(df, oicol, extra=None):
    """% change in TOTAL forward OI, both sessions restricted to the SAME set of
    live expiries. A roll (front 100->20, next 20->100) must net to zero."""
    days = sorted(df["trade_date"].unique())
    prev = {d: p for p, d in zip(days, days[1:])}
    g = {k: v for k, v in df.groupby("trade_date")}
    keys = ["symbol", "expiry_date"] + (extra or [])
    grp = ["symbol"] + (extra or [])
    out = []
    for d in days[1:]:
        if d in SETTLE:                    # front settles, everyone rolls IN
            continue
        a, b = g[d], g.get(prev[d])
        if b is None:
            continue
        m = a.merge(b, on=keys, suffixes=("", "_p"))
        if m.empty:
            continue
        s = m.groupby(grp)[[oicol, oicol + "_p"]].sum()
        s = s[s[oicol + "_p"] > 0]
        s["pct"] = (s[oicol] - s[oicol + "_p"]) / s[oicol + "_p"] * 100
        s["trade_date"] = d
        out.append(s.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


FUT = fwd_oi_change(fut, "open_interest")
FUT["pct"] = FUT["pct"].clip(-50, 50)      # one contract printed +895%
FO = FUT.pivot_table("pct", "trade_date", "symbol")

OPT = fwd_oi_change(opt, "oi", extra=["option_type"])
OPT["pct"] = OPT["pct"].clip(-50, 50)
CE = OPT[OPT["option_type"] == "CE"].pivot_table("pct", "trade_date", "symbol")
PE = OPT[OPT["option_type"] == "PE"].pivot_table("pct", "trade_date", "symbol")
oa = opt.groupby(["trade_date", "symbol", "option_type"])["oi"].sum().unstack()
PCR = (oa["PE"] / oa["CE"].replace(0, np.nan)).unstack()

# -- index -------------------------------------------------------------------
nif = c.execute("""SELECT trade_date,open_val,high_val,low_val,close_val,pct_chg
    FROM index_data WHERE index_name='Nifty 50' AND trade_date>='2021-06-01'""").df()
nif["trade_date"] = pd.to_datetime(nif["trade_date"])
nif = nif.set_index("trade_date").sort_index().astype(float)
IX = R.index.intersection(nif.index)
y = nif.loc[IX, "pct_chg"]
lg = np.log1p(y / 100.0)
FWD = {h: (np.expm1(lg.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1)) * 100.0)
       for h in (1, 2, 5, 10)}
FWD["gap"] = (nif["open_val"] / nif["close_val"].shift(1) - 1).shift(-1).reindex(IX) * 100
HZ = ["gap", 1, 2, 5, 10]
NWLAG = {"gap": 1, 1: 1, 2: 2, 5: 5, 10: 10}
CLR = ((nif["close_val"] - nif["low_val"]) /
       (nif["high_val"] - nif["low_val"]).replace(0, np.nan)).reindex(IX)
MOM = y

print(f"cash panel   {len(IX)} sessions  {IX.min():%Y-%m-%d}..{IX.max():%Y-%m-%d}")
print(f"futures rows {len(FUT):,}  days {FUT['trade_date'].nunique()}  "
      f"{FUT['trade_date'].min():%Y-%m-%d}..{FUT['trade_date'].max():%Y-%m-%d}")
print(f"options rows {len(OPT):,}  days {OPT['trade_date'].nunique()}  "
      f"names/day median {int(CE.notna().sum(axis=1).median())} of 50")
print(f"settlement sessions dropped: {len(SETTLE & set(IX))}")


def persym_z(M, cols, base=DELIV_BASE):
    """z each symbol against ITS OWN trailing mean, then average. A bucket-level
    z is partly a COVERAGE signal (Rest-30 corr(count, mean delivery) = +0.674)."""
    S = M[[s for s in cols if s in M.columns]]
    mu, sd = S.rolling(base).mean().shift(1), S.rolling(base).std().shift(1)
    return ((S - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan).mean(axis=1)


def signals(cols):
    cols = [s for s in cols if s in R.columns]

    def mn(M):
        k = [s for s in cols if s in M.columns]
        return M[k].mean(axis=1) if k else pd.Series(dtype=float)

    ce, pe, fo = mn(CE), mn(PE), mn(FO)
    dz = persym_z(DL, cols)
    sub = R[cols]
    # OI-price matrix on ROLL-IMMUNE total OI, priced off the CASH return, which
    # does not break across a roll the way the futures series does
    lean = (np.sign(sub) * np.sign(FO[[s for s in cols if s in FO.columns]]
                                   .reindex(sub.index))).mean(axis=1)
    pcr = mn(PCR)
    s = dict(
        deliv_z=dz,
        delivval_z=persym_z(DV, cols),
        adv=(sub > 0).sum(axis=1) / sub.notna().sum(axis=1) * 100 - 50,
        ret=mn(R),
        fut_oi=fo,
        fut_lean=lean,
        call_oi=ce,
        put_oi=pe,
        pw_cw=pe - ce,                      # put writing minus call writing
        pcr_chg=pcr - pcr.shift(1),
    )
    zz = lambda v: (v - v.rolling(250).mean()) / v.rolling(250).std()
    # the user's literal conjunction, as a continuous score
    s["COMPOSITE"] = (zz(dz).fillna(0) + zz(fo).fillna(0) + zz(pe - ce).fillna(0)) / 3
    return s


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


B = {k: signals(v) for k, v in BUCK.items()}
B["DIVERGE"] = {sn: B["TOP10"][sn] - B["REST30"][sn] for sn in B["TOP10"]}

CASH_SIG = ["deliv_z", "delivval_z", "adv"]
FNO_SIG = ["fut_oi", "fut_lean", "call_oi", "put_oi", "pw_cw", "pcr_chg", "COMPOSITE"]

rows, candA, candB = [], {}, {}
for bn, sig in B.items():
    for sn in CASH_SIG + FNO_SIG:
        for h in HZ:
            k = pd.DataFrame({"s": sig[sn], "f": FWD[h]}).dropna().astype(float)
            if len(k) < 120:
                continue
            sg = np.sign(k["s"]) * k["f"]
            rows.append(dict(panel="A-cash" if sn in CASH_SIG else "B-fno",
                             bucket=bn, signal=sn, h=str(h), n=len(k),
                             IC=round(k["s"].rank().corr(k["f"].rank()), 4),
                             bps=round(sg.mean() * 100, 2),
                             t=round(nw(sg, NWLAG[h]), 2),
                             hit=round((sg > 0).mean() * 100, 1)))
            (candA if sn in CASH_SIG else candB)[f"{bn}.{sn}.{h}"] = sg
for cn, cs in [("CTRL momentum", MOM), ("CTRL clr", CLR - 0.5)]:
    for h in HZ:
        k = pd.DataFrame({"s": cs, "f": FWD[h]}).dropna().astype(float)
        sg = np.sign(k["s"]) * k["f"]
        rows.append(dict(panel="ctrl", bucket=cn, signal="-", h=str(h), n=len(k),
                         IC=round(k["s"].rank().corr(k["f"].rank()), 4),
                         bps=round(sg.mean() * 100, 2), t=round(nw(sg, NWLAG[h]), 2),
                         hit=round((sg > 0).mean() * 100, 1)))
res = pd.DataFrame(rows)

print("\n" + "=" * 104)
print("CONTROLS FIRST - the bar every bucket signal has to clear")
print("=" * 104)
print(res[res.panel == "ctrl"].to_string(index=False))
for pn, lbl in [("A-cash", "PANEL A - cash flow, 2022+"),
                ("B-fno", "PANEL B - F&O flow, 2024-07+")]:
    print("\n" + "=" * 104)
    print(f"{lbl}   (top 15 by |t|)")
    print("=" * 104)
    print(res[res.panel == pn].sort_values("t", key=abs, ascending=False)
          .head(15).to_string(index=False))

print("\n" + "=" * 104)
print("MULTIPLICITY - max|t| over each panel's whole search vs a date-block null")
print("=" * 104)
for nm, cd in [("A-cash", candA), ("B-fno", candB)]:
    M = pd.DataFrame(cd).astype(float)
    if M.empty:
        continue
    tt = {k: nw(M[k].dropna(), 5) for k in M.columns}
    obs = np.nanmax(np.abs(list(tt.values())))
    best = max(tt, key=lambda k: abs(tt[k]) if not np.isnan(tt[k]) else -1)
    Mc = (M - M.mean()).values
    T = len(Mc)
    nulls = np.empty(2000)
    for b in range(2000):
        idx = []
        while len(idx) < T:
            st = rng.integers(0, T)
            ln = rng.geometric(1 / 5)
            idx.extend(((st + np.arange(ln)) % T).tolist())
        S = Mc[np.array(idx[:T])]
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(S, 0)
            sd = np.nanstd(S, 0, ddof=1)
            n_ = (~np.isnan(S)).sum(0)
            nulls[b] = np.nanmax(np.abs(mu / (sd / np.sqrt(np.maximum(n_, 1)))))
    print(f"  {nm}: {len(M.columns)} candidates, T={T}   best={best} |t|={obs:.2f}")
    print(f"         null max|t| median {np.median(nulls):.2f}  95th "
          f"{np.percentile(nulls,95):.2f}   Reality Check p = {(nulls>=obs).mean():.4f}")

print("\n" + "=" * 104)
print("INCREMENTAL - does bucket flow add anything OVER momentum + close-strength?")
print("=" * 104)


def ols_nw(k, cols, lag):
    X = np.column_stack([np.ones(len(k))] + [k[c].values for c in cols])
    b, *_ = np.linalg.lstsq(X, k["f"].values, rcond=None)
    e = k["f"].values - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = X * e[:, None]
    meat = S.T @ S
    for L in range(1, lag + 1):
        G = S[L:].T @ S[:-L]
        meat += (1 - L / (lag + 1)) * (G + G.T)
    return b, np.sqrt(np.diag(XtXi @ meat @ XtXi))


for bn in ("TOP10", "REST30", "DIVERGE"):
    for sn in ("adv", "deliv_z", "fut_oi", "pw_cw", "COMPOSITE"):
        for h in ("gap", 5):
            k = pd.DataFrame({"f": FWD[h], "m": MOM, "c": CLR - 0.5,
                              "s": B[bn][sn]}).dropna().astype(float)
            if len(k) < 120:
                continue
            b, se = ols_nw(k, ["m", "c", "s"], NWLAG[h])
            print(f"  {bn:8s} {sn:10s} h={str(h):3s} n={len(k):4d}  "
                  f"mom t={b[1]/se[1]:+5.2f}  clr t={b[2]/se[2]:+5.2f}  "
                  f"SIGNAL t={b[3]/se[3]:+5.2f}")

print("\n" + "=" * 104)
print("THE USER'S HARD GATE - top-10 delivery UP *and* futures OI UP *and* price UP")
print("=" * 104)
d, f, r = B["TOP10"]["deliv_z"], B["TOP10"]["fut_oi"], B["TOP10"]["ret"]
for h in HZ:
    k = pd.DataFrame({"d": d, "f": f, "r": r, "y": FWD[h]}).dropna().astype(float)
    if len(k) < 60:
        continue
    on = (k.d > 0.5) & (k.f > 0) & (k.r > 0)
    off = (k.d < -0.5) & (k.f < 0) & (k.r < 0)
    print(f"  h={str(h):3s} n={len(k):4d}  BULL fires {on.sum():3d}d -> "
          f"{k.y[on].mean()*100:+7.1f}bps (up {(k.y[on]>0).mean()*100:.0f}%, "
          f"t={nw(k.y[on], NWLAG[h]):+.2f})   BEAR {off.sum():3d}d -> "
          f"{k.y[off].mean()*100:+7.1f}bps   base {k.y.mean()*100:+.1f}bps")

print("\n" + "=" * 104)
print("QUINTILE LADDERS - monotone, or one tail?")
print("=" * 104)
for bn, sn, h in [("TOP10", "adv", "gap"), ("REST30", "adv", "gap"),
                  ("TOP10", "deliv_z", 5), ("TOP10", "COMPOSITE", 5),
                  ("TOP10", "pw_cw", 5), ("DIVERGE", "adv", 5)]:
    k = pd.DataFrame({"s": B[bn][sn], "f": FWD[h]}).dropna().astype(float)
    if len(k) < 120:
        continue
    k["q"] = pd.qcut(k["s"], 5, labels=False, duplicates="drop")
    g = k.groupby("q")["f"].agg(n="size", bps=lambda v: v.mean() * 100,
                                up=lambda v: (v > 0).mean() * 100)
    print(f"  {bn}.{sn} h={h}: " + "  ".join(
        f"Q{int(q)+1}:{r.bps:+6.1f}({r.up:.0f}%,n{int(r.n)})" for q, r in g.iterrows()))

print("\n" + "=" * 104)
print("STABILITY - does anything hold year by year?")
print("=" * 104)
for bn, sn, h in [("TOP10", "adv", "gap"), ("REST30", "adv", "gap"),
                  ("TOP10", "COMPOSITE", 5)]:
    k = pd.DataFrame({"s": B[bn][sn], "f": FWD[h]}).dropna().astype(float)
    sg = np.sign(k["s"]) * k["f"]
    out = [f"{yr}:{g.mean()*100:+6.1f}bps(n{len(g)})" for yr, g in sg.groupby(sg.index.year)]
    print(f"  {bn}.{sn} h={h}  " + "  ".join(out))

print("\n" + "=" * 104)
print("SURVIVORSHIP - how much of the 2022 panel predates the current 50?")
print("=" * 104)
first = R.notna().idxmax()
late = first[first > pd.Timestamp("2022-06-01")]
print(f"  {len(late)} of {R.shape[1]} names absent at the 2022 start: " +
      ", ".join(f"{k}({v:%Y-%m})" for k, v in late.sort_values().items()))
