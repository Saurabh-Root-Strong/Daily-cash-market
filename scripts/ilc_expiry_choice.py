"""WHICH EXPIRY should the bucket OI read use? Measured, not asserted.

THE PANEL TODAY sums open interest across EVERY live expiry
(`expiry_date > trade_date`), matching each contract to ITSELF on the prior
session. It is not near-month and it is not next-month.

THE PROPOSAL is near-month, rolling to the next series when the front is 1-2 days
from expiry. That is the standard desk convention and the objection behind it is
sound: the front month is where the liquidity and the intent are, and far months
are thin.

WHAT KILLED PLAIN NEAR-MONTH the first time (scripts/ilc_self_audit.py): reading
the front contract's OI without a roll gave DTE 0-2 a mean change of -47.95% with
99.94% of rows below -5%, because positions MIGRATE rather than close. But that
measurement conflated TWO different things, and this file separates them:

    (a) the CONTRACT SWITCH   comparing front-contract OI on day t against a
                              DIFFERENT front contract on day t-1. Pure artifact.
    (b) genuine ROLL BLEED    the same contract really does lose OI into expiry
                              as holders move to the next series.

(a) is fixable by matching each contract to itself and rolling early - which is
exactly the proposal. (b) is not fixable, only avoidable, and the question is at
what DTE it starts to bite. If it bites at DTE 5 then a roll at DTE 2 is too late.

TESTED HERE
  Q1 How much of the OI actually sits in the near month? If it is ~95%, near vs
     total is a distinction without a difference and the argument is settled by
     arithmetic.
  Q2 Same-contract OI change by DTE, for the FRONT contract only. This isolates
     genuine roll bleed from the switch artifact and says where the roll must be.
  Q3 Rolled near-month vs total forward, as the panel would actually compute them:
     correlation, disagreement rate, and how often the bucket's read FLIPS.
  Q4 Does either version behave better on the one thing that matters - do the
     panel's reads become more stable, and does the null get any less null?
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

from src.analytics.index_largecap import INDEX_BUCKETS
from src.data.repository import query_dataframe

pd.set_option("display.width", 240)
B = INDEX_BUCKETS["NIFTY"]["buckets"]
ALL50 = tuple(s for v in B.values() for s in v)
ph = ",".join("?" * len(ALL50))
START = "2022-01-01"

print("=" * 104)
print("Q0  WHAT EXPIRIES EXIST for stock F&O, and when do they settle?")
print("=" * 104)
ex = query_dataframe(f"""
    SELECT expiry_date, MIN(trade_date) first_seen, MAX(trade_date) last_seen,
           COUNT(DISTINCT trade_date) sessions
    FROM fno_bhavcopy WHERE instrument='FUTSTK' AND symbol IN ({ph})
      AND trade_date >= ? GROUP BY 1 ORDER BY 1 DESC LIMIT 8
""", [*ALL50, START])
ex["expiry_date"] = pd.to_datetime(ex["expiry_date"])
ex["weekday"] = ex["expiry_date"].dt.day_name()
print(ex.to_string(index=False))
print("\n  stock F&O is MONTHLY only (no weeklies), so 'near month' means the")
print("  contract expiring this calendar month until it settles.")

print("\n" + "=" * 104)
print("Q1  HOW MUCH OI IS IN THE NEAR MONTH? (if ~95%, the choice barely matters)")
print("=" * 104)
share = query_dataframe(f"""
    WITH f AS (
      SELECT trade_date, symbol, instrument, expiry_date,
             SUM(open_interest) oi
      FROM fno_bhavcopy
      WHERE instrument IN ('FUTSTK','OPTSTK') AND open_interest > 0
        AND symbol IN ({ph}) AND expiry_date > trade_date AND trade_date >= ?
      GROUP BY 1,2,3,4),
    r AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY trade_date, symbol, instrument
                                   ORDER BY expiry_date) rn,
             DATE_DIFF('day', trade_date, expiry_date) dte
      FROM f)
    SELECT instrument,
           CASE WHEN rn = 1 THEN 'near' WHEN rn = 2 THEN 'next' ELSE 'far' END leg,
           SUM(oi) oi
    FROM r GROUP BY 1,2
""", [*ALL50, START])
tot = share.groupby("instrument")["oi"].transform("sum")
share["share_%"] = (share["oi"] / tot * 100).round(2)
print(share.pivot(index="instrument", columns="leg", values="share_%").to_string())

print("\n" + "=" * 104)
print("Q2  SAME-CONTRACT OI CHANGE BY DTE — where does genuine roll bleed start?")
print("=" * 104)
print("  Each contract is matched to ITSELF on the prior session, so the")
print("  contract-switch artifact is removed. What remains is real bleed.")
dte = query_dataframe(f"""
    WITH f AS (
      SELECT trade_date, symbol, instrument, expiry_date, SUM(open_interest) oi
      FROM fno_bhavcopy
      WHERE instrument IN ('FUTSTK','OPTSTK') AND open_interest > 0
        AND symbol IN ({ph}) AND trade_date >= ?
      GROUP BY 1,2,3,4),
    l AS (
      SELECT *, LAG(oi) OVER (PARTITION BY symbol, instrument, expiry_date
                              ORDER BY trade_date) poi,
             ROW_NUMBER() OVER (PARTITION BY trade_date, symbol, instrument
                                ORDER BY expiry_date) rn,
             DATE_DIFF('day', trade_date, expiry_date) dte
      FROM f WHERE expiry_date >= trade_date)
    SELECT instrument, rn, dte, oi, poi FROM l WHERE poi > 0
""", [*ALL50, START])
dte["pct"] = (dte["oi"] - dte["poi"]) / dte["poi"] * 100
for inst in ("FUTSTK", "OPTSTK"):
    d = dte[(dte["instrument"] == inst) & (dte["rn"] == 1)]
    d = d[d["dte"] >= 0]
    bins = [(0, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12), (13, 20), (21, 40)]
    print(f"\n  {inst} FRONT contract, same-contract change:")
    for lo, hi in bins:
        g = d[(d["dte"] >= lo) & (d["dte"] <= hi)]["pct"]
        if len(g) < 30:
            continue
        print(f"    DTE {lo:2d}-{hi:2d}  n={len(g):6,}  mean {g.mean():+7.2f}%  "
              f"median {g.median():+6.2f}%  share below -5%: "
              f"{(g < -5).mean()*100:5.1f}%")

print("\n" + "=" * 104)
print("Q3  ROLLED NEAR-MONTH vs TOTAL FORWARD — do they disagree in practice?")
print("=" * 104)


def series(roll_dte: int | None):
    """Bucket-level daily OI % change.

    roll_dte=None -> TOTAL forward OI across every live expiry (what ships today).
    roll_dte=N    -> NEAR month, switching to the next series once the front has
                     N or fewer days left. Every contract is still matched to
                     ITSELF across sessions, so the switch never fabricates a jump.
    """
    if roll_dte is None:
        sql = f"""
        WITH f AS (
          SELECT trade_date, symbol, expiry_date, SUM(open_interest) oi
          FROM fno_bhavcopy WHERE instrument='FUTSTK' AND open_interest>0
            AND symbol IN ({ph}) AND expiry_date > trade_date AND trade_date >= ?
          GROUP BY 1,2,3),
        l AS (SELECT *, LAG(oi) OVER (PARTITION BY symbol, expiry_date
                                      ORDER BY trade_date) poi FROM f)
        SELECT trade_date, symbol, SUM(oi) oi, SUM(poi) poi
        FROM l WHERE poi > 0 GROUP BY 1,2"""
        params = [*ALL50, START]
    else:
        sql = f"""
        WITH f AS (
          SELECT trade_date, symbol, expiry_date, SUM(open_interest) oi
          FROM fno_bhavcopy WHERE instrument='FUTSTK' AND open_interest>0
            AND symbol IN ({ph}) AND expiry_date > trade_date AND trade_date >= ?
          GROUP BY 1,2,3),
        e AS (SELECT *, DATE_DIFF('day', trade_date, expiry_date) dte,
                     ROW_NUMBER() OVER (PARTITION BY trade_date, symbol
                       ORDER BY expiry_date) rn FROM f),
        pick AS (
          SELECT trade_date, symbol,
                 MIN(CASE WHEN dte > {roll_dte} THEN expiry_date END) chosen
          FROM e GROUP BY 1,2),
        s AS (SELECT f.* FROM f JOIN pick p ON p.symbol=f.symbol
                AND p.trade_date=f.trade_date AND p.chosen=f.expiry_date),
        l AS (SELECT s.trade_date, s.symbol, s.expiry_date, s.oi,
                     (SELECT oi FROM f x WHERE x.symbol=s.symbol
                        AND x.expiry_date=s.expiry_date
                        AND x.trade_date < s.trade_date
                      ORDER BY x.trade_date DESC LIMIT 1) poi
              FROM s)
        SELECT trade_date, symbol, oi, poi FROM l WHERE poi > 0"""
        params = [*ALL50, START]
    df = query_dataframe(sql, params)
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["pct"] = ((df["oi"] - df["poi"]) / df["poi"] * 100).clip(-50, 50)
    return df.pivot_table("pct", "trade_date", "symbol")


TOT = series(None)
variants = {f"near, roll at DTE<={d}": series(d) for d in (1, 2, 5)}
for bn, mem in B.items():
    cols = [s for s in mem if s in TOT.columns]
    base = TOT[cols].mean(axis=1)
    print(f"\n  {bn}")
    for nm, V in variants.items():
        c2 = [s for s in cols if s in V.columns]
        v = V[c2].mean(axis=1)
        j = pd.DataFrame({"tot": base, "near": v}).dropna()
        flip = (np.sign(j["tot"]) != np.sign(j["near"])).mean() * 100
        print(f"    {nm:24s} n={len(j):5d}  corr {j['tot'].corr(j['near']):+.3f}  "
              f"mean tot {j['tot'].mean():+.3f}% vs near {j['near'].mean():+.3f}%  "
              f"SIGN FLIPS on {flip:.1f}% of sessions")

print("\n" + "=" * 104)
print("Q4  WHICH IS CLEANER? stability of the read, and the expiry-week distortion")
print("=" * 104)
settle = set(pd.to_datetime(query_dataframe(
    "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK'"
    " AND expiry_date=trade_date AND trade_date >= ?", [START])["trade_date"]))
cols = [s for s in B["Top 10"] if s in TOT.columns]
allv = {"TOTAL forward (shipped)": TOT[cols].mean(axis=1)}
for nm, V in variants.items():
    allv[nm] = V[[s for s in cols if s in V.columns]].mean(axis=1)
for nm, v in allv.items():
    v = v.dropna()
    idx = v.index
    is_exp_week = pd.Series(
        [any(abs((d - s).days) <= 3 for s in settle) for d in idx], index=idx)
    a, b = v[is_exp_week], v[~is_exp_week]
    print(f"  {nm:26s} sd {v.std():5.2f}  |mean| expiry-week {a.mean():+6.3f}% "
          f"vs other {b.mean():+6.3f}%  gap {abs(a.mean()-b.mean()):.3f}pp")
print("\n  A version whose expiry-week mean matches its ordinary-week mean is the")
print("  one not carrying a mechanical roll distortion into the bucket read.")
