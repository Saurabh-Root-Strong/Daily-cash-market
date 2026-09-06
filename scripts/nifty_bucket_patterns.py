"""The PATTERN CATALOGUE: given each bucket's configuration today, what did NIFTY do?

This is a different question from nifty_bucket_backtest_v2.py. That file tested
CONTINUOUS signals (is more delivery better?). This one tests DISCRETE STATES -
"top-10 delivery falling AND call writing AND short buildup" - which is how the
read is actually made, and answers it as a lookup table with base rates.

PER BUCKET, PER SESSION, five categorical reads:
  deliv_dir    delivery % vs each name's OWN 100-day normal, averaged   UP/FLAT/DOWN
  dval_dir     delivery TURNOVER VALUE the same way                     UP/FLAT/DOWN
  fut_state    OI-price matrix, majority across the bucket              LONG BUILDUP /
               SHORT BUILDUP / SHORT COVERING / LONG UNWINDING / MIXED
  opt_read     CE/PE OI direction against the stock's own move          CALL WRITING /
               CALL BUYING / PUT WRITING / PUT BUYING / MIXED
  oi_dir       total forward FUTURES OI                                 UP/FLAT/DOWN

THREE TRAPS THIS FILE HAS TO AVOID, all of them already paid for in this repo:
  ROLLOVER   near-month OI falls ~48% at 0-2 DTE because positions MIGRATE. Every
             OI figure here is TOTAL across all live expiries, both sessions gated
             to the SAME forward set, settlement sessions dropped.
  COVERAGE   a bucket mean over "whatever names reported" moves with composition,
             not flow (Rest-30 corr(count, mean delivery) was +0.674). Everything
             is per-symbol first, then averaged.
  LOOKUP-TABLE MINING  this produces ~150 cells. Some WILL look excellent. Every
             cell gets an episode-clustered t against the unconditional base, then
             Benjamini-Hochberg FDR across the whole catalogue. A cell that does
             not survive FDR is reported as a base rate and nothing more.
"""
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
rng = np.random.default_rng(43)

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
ap.add_argument("--zcut", type=float, default=0.3, help="|z| below this is FLAT")
a = ap.parse_args()
WARM = (pd.Timestamp(a.start) - pd.Timedelta(days=220)).date().isoformat()

c = duckdb.connect("data/market_data.duckdb", read_only=True)

cash = c.execute(f"""SELECT trade_date, symbol, deliv_per,
        deliv_qty*avg_price/100000.0 AS dval,
        (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND close_price>0 AND prev_close>0 AND trade_date>=?""", [*ALL50, WARM]).df()
cash["trade_date"] = pd.to_datetime(cash["trade_date"])
cash = cash[cash["r"].abs() < 40]
R = cash.pivot_table("r", "trade_date", "symbol").sort_index()
DL = cash.pivot_table("deliv_per", "trade_date", "symbol").sort_index()
DV = cash.pivot_table("dval", "trade_date", "symbol").sort_index()

fno = c.execute(f"""SELECT trade_date, symbol, expiry_date, instrument, option_type,
        SUM(open_interest) oi
    FROM fno_bhavcopy
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
    """% change in TOTAL forward OI, both days over the SAME live-expiry set."""
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
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


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
GAP = (nif["open_val"] / nif["close_val"].shift(1)).reindex(IX).shift(-1) * 100 - 100


def persym_z(M, cols, base=100):
    S = M[[s for s in cols if s in M.columns]]
    mu, sd = S.rolling(base).mean().shift(1), S.rolling(base).std().shift(1)
    return ((S - mu) / sd.where(sd > 1e-9)).replace([np.inf, -np.inf], np.nan).mean(axis=1)


def dirn(v, cut):
    return pd.Series(np.where(v > cut, "UP", np.where(v < -cut, "DOWN", "FLAT")),
                     index=v.index)


def bucket_frame(cols):
    """Per-symbol classification first, then a MAJORITY vote across the bucket."""
    cols = [s for s in cols if s in R.columns]
    sub = R[cols]
    f = FOI[[s for s in cols if s in FOI.columns]].reindex(sub.index)
    ce = CE[[s for s in cols if s in CE.columns]].reindex(sub.index)
    pe = PE[[s for s in cols if s in PE.columns]].reindex(sub.index)
    up, oiup = sub > 0, f > 0.5
    oidn = f < -0.5
    # OI-price matrix, per symbol
    st = pd.DataFrame(np.where(up & oiup, "LONG BUILDUP",
                      np.where(~up & oiup, "SHORT BUILDUP",
                      np.where(up & oidn, "SHORT COVERING",
                      np.where(~up & oidn, "LONG UNWINDING", "")))),
                      index=sub.index, columns=sub.columns).where(f.notna())
    # options read, per symbol: OI direction against the stock's own move
    orv = pd.DataFrame(np.where((ce > 0.5) & ~up, "CALL WRITING",
                       np.where((ce > 0.5) & up, "CALL BUYING",
                       np.where((pe > 0.5) & up, "PUT WRITING",
                       np.where((pe > 0.5) & ~up, "PUT BUYING", "")))),
                       index=sub.index, columns=sub.columns).where(ce.notna())

    def vote(D, cats, minn=4):
        """Plurality across the bucket. Counted per category rather than with a
        per-row value_counts, which blew up on sessions where NO name has an F&O
        read (idxmax raises 'Encountered all NA values') and was far slower."""
        cnt = pd.DataFrame({cat: (D == cat).sum(axis=1) for cat in cats},
                           index=D.index)
        tot = cnt.sum(axis=1)
        mx = cnt.max(axis=1)
        top = cnt.idxmax(axis=1)
        # a plurality, not a coin flip; and never a verdict off 3 names
        ok = (tot >= minn) & (mx > tot * 0.4)
        return pd.Series(np.where(ok, top, "MIXED"), index=D.index)

    FSTATES = ("LONG BUILDUP", "SHORT BUILDUP", "SHORT COVERING", "LONG UNWINDING")
    OSTATES = ("CALL WRITING", "CALL BUYING", "PUT WRITING", "PUT BUYING")

    return pd.DataFrame({
        "deliv_dir": dirn(persym_z(DL, cols), a.zcut),
        "dval_dir": dirn(persym_z(DV, cols), a.zcut),
        "oi_dir": dirn(f.mean(axis=1), 0.5),
        "fut_state": vote(st, FSTATES),
        "opt_read": vote(orv, OSTATES),
        "adv": (sub > 0).sum(axis=1) / sub.notna().sum(axis=1) * 100,
    })


BF = {k: bucket_frame(v) for k, v in BUCK.items()}
K = pd.DataFrame({"gap": GAP, "d1": F1, "d5": F5})
for b, fr in BF.items():
    for col in fr.columns:
        K[f"{b}_{col}"] = fr[col]
K = K[K.index >= pd.Timestamp(a.start)].dropna(subset=["d1", "d5"])
print(f"panel {len(K)} sessions  {K.index.min():%Y-%m-%d}..{K.index.max():%Y-%m-%d}")
fnoc = K["TOP10_fut_state"].ne("MIXED").mean() * 100
print(f"sessions with a usable futures read: {fnoc:.0f}%   "
      f"options read: {K['TOP10_opt_read'].ne('MIXED').mean()*100:.0f}%")
BASE = {h: K[h].mean() for h in ("gap", "d1", "d5")}
print(f"UNCONDITIONAL BASE   gap {BASE['gap']:+.3f}%   next day {BASE['d1']:+.3f}%   "
      f"next 5 sessions {BASE['d5']:+.3f}%")
print(f"                     up  {(K['gap']>0).mean()*100:.1f}%   "
      f"{(K['d1']>0).mean()*100:.1f}%   {(K['d5']>0).mean()*100:.1f}%")


def episodes(mask, maxgap=5):
    pos = np.where(mask)[0]
    if len(pos) == 0:
        return []
    epi, cur = [], [int(pos[0])]
    for x, z in zip(pos, pos[1:]):
        if z - x <= maxgap:
            cur.append(int(z))
        else:
            epi.append(cur)
            cur = [int(z)]
    epi.append(cur)
    return epi


CELLS = []


def cell(mask, label, group, hs=("d1", "d5")):
    n = int(mask.sum())
    if n < 15:
        return
    rec = dict(group=group, pattern=label, days=n)
    epi = episodes(mask)
    for h in hs:
        v = K[h].values[mask]
        # An EPISODE is one independent draw: consecutive firings share most of
        # their forward window, so day-weighted and episode-weighted means differ.
        # Both are reported, and the t-stat belongs to the episode one -- quoting a
        # day-weighted excess beside an episode t produced cells whose excess and
        # t disagreed in SIGN, which is unreadable.
        em = np.array([K[h].values[e].mean() for e in epi])
        t = ((em.mean() - BASE[h]) / (em.std(ddof=1) / math.sqrt(len(em)))
             if len(em) > 2 and em.std(ddof=1) > 1e-12 else np.nan)
        rec[f"{h}_pct"] = round(float(v.mean()), 3)
        rec[f"{h}_vs_base"] = round(float(v.mean() - BASE[h]), 3)
        rec[f"{h}_epi_exc"] = round(float(em.mean() - BASE[h]), 3)
        rec[f"{h}_up"] = round(float((v > 0).mean() * 100), 1)
        rec[f"{h}_t"] = round(float(t), 2) if t == t else np.nan
    rec["episodes"] = len(epi)
    CELLS.append(rec)


# ── A: the futures OI-price matrix, per bucket ───────────────────────────────
for b in BUCK:
    for s in ("LONG BUILDUP", "SHORT BUILDUP", "SHORT COVERING", "LONG UNWINDING"):
        cell((K[f"{b}_fut_state"] == s).values, s, f"A futures | {b}")
# ── B: the options read, per bucket ──────────────────────────────────────────
for b in BUCK:
    for s in ("CALL WRITING", "CALL BUYING", "PUT WRITING", "PUT BUYING"):
        cell((K[f"{b}_opt_read"] == s).values, s, f"B options | {b}")
# ── C: delivery direction x futures OI direction, per bucket ────────────────
for b in BUCK:
    for dd in ("UP", "DOWN"):
        for od in ("UP", "DOWN"):
            cell(((K[f"{b}_deliv_dir"] == dd) & (K[f"{b}_oi_dir"] == od)).values,
                 f"deliv {dd} + futOI {od}", f"C deliv x OI | {b}")
# ── D: delivery VALUE direction, per bucket ─────────────────────────────────
for b in BUCK:
    for dd in ("UP", "DOWN"):
        cell((K[f"{b}_dval_dir"] == dd).values, f"deliv VALUE {dd}",
             f"D deliv value | {b}")
# ── E: the user's worked example, and its mirror ────────────────────────────
for b in BUCK:
    cell(((K[f"{b}_deliv_dir"] == "DOWN") & (K[f"{b}_opt_read"] == "CALL WRITING") &
          (K[f"{b}_fut_state"] == "SHORT BUILDUP")).values,
         "deliv DOWN + CALL WRITING + SHORT BUILDUP", f"E worked example | {b}")
    cell(((K[f"{b}_deliv_dir"] == "UP") & (K[f"{b}_opt_read"] == "PUT WRITING") &
          (K[f"{b}_fut_state"] == "LONG BUILDUP")).values,
         "deliv UP + PUT WRITING + LONG BUILDUP", f"E worked example | {b}")
# ── F: ALL THREE BUCKETS aligned ────────────────────────────────────────────
for s in ("LONG BUILDUP", "SHORT BUILDUP", "SHORT COVERING", "LONG UNWINDING"):
    cell(np.logical_and.reduce([(K[f"{b}_fut_state"] == s).values for b in BUCK]),
         f"all 3 buckets {s}", "F all buckets aligned")
for d in ("UP", "DOWN"):
    cell(np.logical_and.reduce([(K[f"{b}_oi_dir"] == d).values for b in BUCK]),
         f"all 3 buckets futOI {d}", "F all buckets aligned")
    cell(np.logical_and.reduce([(K[f"{b}_deliv_dir"] == d).values for b in BUCK]),
         f"all 3 buckets delivery {d}", "F all buckets aligned")
    cell(np.logical_and.reduce(
        [(K[f"{b}_oi_dir"] == d).values for b in BUCK] +
        [(K[f"{b}_deliv_dir"] == d).values for b in BUCK]),
        f"all 3 buckets delivery AND futOI {d}", "F all buckets aligned")
# ── G: heavyweights vs the rest disagreeing ─────────────────────────────────
for s in ("LONG BUILDUP", "SHORT BUILDUP"):
    o = "SHORT BUILDUP" if s == "LONG BUILDUP" else "LONG BUILDUP"
    cell(((K["TOP10_fut_state"] == s) & (K["REST30_fut_state"] == o)).values,
         f"TOP10 {s} vs REST30 {o}", "G divergence")

RES = pd.DataFrame(CELLS)
if RES.empty:
    print("\nno cell reached the 15-session floor")
    raise SystemExit(0)

# ── multiplicity across the whole catalogue ─────────────────────────────────
tv = pd.concat([RES["d1_t"], RES["d5_t"]]).dropna()
pv = 2 * (1 - pd.Series(tv).abs().apply(
    lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))))
order = np.argsort(pv.values)
m = len(pv)
bh = np.full(m, False)
crit = 0.10 * (np.arange(1, m + 1)) / m
srt = pv.values[order]
below = np.where(srt <= crit)[0]
if len(below):
    bh[order[:below[-1] + 1]] = True
print(f"\ncatalogue: {len(RES)} cells x 2 horizons = {m} tests")
print(f"Benjamini-Hochberg at FDR 10%: {int(bh.sum())} survive   "
      f"(smallest p = {srt[0]:.4f}, needed <= {crit[0]:.5f})")

for grp, g in RES.groupby("group"):
    print("\n" + "=" * 118)
    print(grp)
    print("=" * 118)
    print(g.drop(columns=["group"]).to_string(index=False))
