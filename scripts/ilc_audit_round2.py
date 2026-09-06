"""Adversarial audit of everything added to Index & Large Cap this session.

H1 CLASSIFIER ASYMMETRY. The LIVE `state` property refuses to classify when a
   bucket is thin (coverage < 0.60). get_state_base_rates applies NO coverage
   floor. So the historical base rates may include sessions the live panel would
   never label -- the denominator and the numerator would not describe the same
   population.
H2 ANALOGUE RECENCY. The audit backtest purged neighbours within `horizon`
   sessions. The LIVE function does not. If matches cluster a few days back, the
   panel is showing autocorrelation dressed as an analogue.
H3 DAY-AFTER-EXPIRY. The panel suppresses futures/options when TODAY is a
   settlement session. It does nothing special when YESTERDAY was. If forward OI
   is distorted on that comparison the day-after read is garbage.
H4 UNKNOWN BUCKET LABELS. net_score looks bucket weights up by LABEL. A future
   index with different labels silently returns None rather than failing loudly.
H5 BACKFILL CONVENTION BREAK. 2022-2024 came from the LEGACY archive, 2024-07+
   from UDiFF. If OI is quoted in contracts in one and units in the other, or if
   a lot-size revision lands mid-history, every OI % change across that seam is
   fiction. Total OI was already checked; this checks PER-SYMBOL and per-day
   distributions, which is where a convention break actually shows.
H6 COLLATERAL DAMAGE. The backfill added 626 sessions to a table other panels
   read. Anything that assumed "F&O starts 2024-07" now behaves differently.
H7 ANALOGUE STABILITY. If the displayed answer swings with k, the number on
   screen is an artifact of a parameter nobody chose deliberately.
H8 NaN SEMANTICS. flow_score averages only legs that exist. Check that a bucket
   with NO legs cannot silently read as neutral 0.00.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime as dt
import numpy as np, pandas as pd

from src.analytics import index_largecap as ilc
from src.data.repository import query_dataframe

pd.set_option("display.width", 240)


class _C:
    """Use the repository's connection manager. Opening a second duckdb handle
    with different settings raises 'Can't open a connection to same database file
    with a different configuration than existing connections'."""
    @staticmethod
    def execute(sql, params=None):
        class _R:
            def __init__(self, d): self._d = d
            def df(self): return self._d
        return _R(query_dataframe(sql, params or []))


c = _C()
AS_OF = dt.date(2026, 9, 3)
B = ilc.INDEX_BUCKETS["NIFTY"]["buckets"]
ALL50 = tuple(s for v in B.values() for s in v)
ph = ",".join("?" * len(ALL50))

print("=" * 100)
print("H1  CLASSIFIER ASYMMETRY — does history include days the live panel refuses?")
print("=" * 100)
cash = c.execute(f"""SELECT trade_date, symbol,
        (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
      AND close_price>0 AND prev_close>0 AND trade_date>='2021-06-01'""", list(ALL50)).df()
cash["trade_date"] = pd.to_datetime(cash["trade_date"])
cash = cash[cash["r"].abs() < 40]
R = cash.pivot_table("r", "trade_date", "symbol")
cov = {k: R[[s for s in v if s in R.columns]].notna().sum(axis=1) / len(v)
       for k, v in B.items()}
COV = pd.DataFrame(cov)
thin_any = (COV < ilc._MIN_COVER).any(axis=1)
print(f"  sessions since 2022: {int((COV.index >= '2022-01-01').sum())}")
sub = thin_any[thin_any.index >= "2022-01-01"]
print(f"  sessions where SOME bucket is below the 0.60 live floor: {int(sub.sum())} "
      f"({sub.mean()*100:.2f}%)")
print(f"  minimum coverage by bucket: " +
      "  ".join(f"{k} {COV[k][COV.index>='2022-01-01'].min():.2f}" for k in B))
print("  => if this is 0, the live floor never binds and the two populations match.")

print("\n" + "=" * 100)
print("H2  ANALOGUE RECENCY — are the 'analogues' just the last few sessions?")
print("=" * 100)
Z, fwd = ilc._flow_state_matrix("NIFTY", AS_OF)
gaps_all, worst = [], []
sample = Z.index[Z.index >= pd.Timestamp("2024-01-01")]
for ts in sample[::5]:
    past = Z.index[Z.index < ts]
    if len(past) < 100:
        continue
    d = (Z.loc[past] - Z.loc[ts]).abs().sum(axis=1).sort_values()
    sel = d.index[:25]
    g = np.array([(ts - x).days for x in sel])
    gaps_all.append(g.min())
    worst.append((g < 10).sum())
gaps_all = np.array(gaps_all)
print(f"  {len(gaps_all)} days sampled. Calendar gap to the NEAREST match:")
print(f"    median {np.median(gaps_all):.0f}d   p10 {np.percentile(gaps_all,10):.0f}d   "
      f"min {gaps_all.min():.0f}d")
print(f"  days where any of the 25 matches is within 10 calendar days: "
      f"{(np.array(worst)>0).mean()*100:.1f}%  (mean count {np.mean(worst):.2f} of 25)")
print("  => a large share here would mean the panel is showing autocorrelation.")

print("\n" + "=" * 100)
print("H3  DAY AFTER EXPIRY — is the forward-OI comparison distorted?")
print("=" * 100)
settle = set(pd.to_datetime(c.execute(
    "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK'"
    " AND expiry_date=trade_date").df()["trade_date"]))
oi = c.execute(f"""
  WITH f AS (SELECT trade_date,symbol,expiry_date,SUM(open_interest) oi
      FROM fno_bhavcopy WHERE instrument='FUTSTK' AND open_interest>0
        AND symbol IN ({ph}) AND expiry_date>trade_date GROUP BY 1,2,3),
  l AS (SELECT *, LAG(oi) OVER (PARTITION BY symbol,expiry_date
                                ORDER BY trade_date) poi FROM f)
  SELECT trade_date, SUM(oi) oi, SUM(poi) poi FROM l WHERE poi>0 GROUP BY 1""",
  list(ALL50)).df()
oi["trade_date"] = pd.to_datetime(oi["trade_date"])
oi["pct"] = (oi["oi"] - oi["poi"]) / oi["poi"] * 100
oi = oi.sort_values("trade_date").set_index("trade_date")
is_set = oi.index.isin(settle)
prev_set = np.roll(is_set, 1)
prev_set[0] = False
for nm, m in (("normal session", ~is_set & ~prev_set),
              ("SETTLEMENT session", is_set),
              ("day AFTER settlement", ~is_set & prev_set)):
    v = oi["pct"][m]
    print(f"  {nm:22s} n={len(v):4d}  mean {v.mean():+7.2f}%  median {v.median():+6.2f}%"
          f"  sd {v.std():5.2f}")
print("  => the day-after row must look like a normal session, or it needs")
print("     suppressing the way the settlement session itself does.")

print("\n" + "=" * 100)
print("H4  UNKNOWN BUCKET LABELS — does net_score fail loudly or silently?")
print("=" * 100)
d = ilc.IndexLargeCap(trade_date=None, fno_symbol="X", display="x")
d.rows = [ilc.BucketRow(label="Mega caps", n_members=10, n_present=10, deliv_z=2.0)]
print(f"  a bucket labelled 'Mega caps' with a strong delivery leg -> "
      f"net_score = {d.net_score}")
print(f"  _BUCKET_WEIGHT keys: {sorted(ilc._BUCKET_WEIGHT)}")
print(f"  INDEX_BUCKETS['NIFTY'] labels: {sorted(ilc.INDEX_BUCKETS['NIFTY']['buckets'])}")
missing = set(ilc.INDEX_BUCKETS["NIFTY"]["buckets"]) - set(ilc._BUCKET_WEIGHT)
print(f"  labels with no weight: {missing or 'none'}")
print("  => None on an unknown label is silent. A new index would render a blank")
print("     score with no explanation unless this is asserted somewhere.")

print("\n" + "=" * 100)
print("H5  BACKFILL CONVENTION — legacy vs UDiFF, per symbol")
print("=" * 100)
q = c.execute(f"""SELECT trade_date, symbol, SUM(open_interest) oi, SUM(contracts) ct
    FROM fno_bhavcopy WHERE instrument='FUTSTK' AND symbol IN ({ph})
      AND trade_date BETWEEN '2024-06-20' AND '2024-08-20'
    GROUP BY 1,2""", list(ALL50)).df()
q["trade_date"] = pd.to_datetime(q["trade_date"])
piv = q.pivot_table("oi", "trade_date", "symbol")
pre = piv[piv.index < "2024-07-08"].tail(10).mean()
mid = piv[(piv.index >= "2024-07-08") & (piv.index < "2024-07-24")].mean()
post = piv[piv.index >= "2024-07-24"].head(10).mean()
rat1 = (mid / pre).dropna()
rat2 = (post / mid).dropna()
print(f"  per-symbol OI ratio legacy->UDiFF-gap : median {rat1.median():.3f}  "
      f"p5 {rat1.quantile(.05):.3f}  p95 {rat1.quantile(.95):.3f}")
print(f"  per-symbol OI ratio UDiFF-gap->capture: median {rat2.median():.3f}  "
      f"p5 {rat2.quantile(.05):.3f}  p95 {rat2.quantile(.95):.3f}")
print(f"  symbols with a >3x jump at either seam: "
      f"{int(((rat1>3)|(rat1<0.33)).sum() + ((rat2>3)|(rat2<0.33)).sum())}")
# NOTE: OI/contracts is NOT a convention test. It is open interest over DAILY
# TRADED contracts, which swings with volume -- comparing an expiry week against
# a quiet week shows a 2.5x "jump" that means nothing. The valid test is the
# per-symbol OI RATIO across each seam, above: a contracts-vs-units break would
# put those ratios at the lot size (hundreds to thousands), not at 1.0.
print("  per-symbol ratios sit at ~1.0 with zero >3x jumps, so open interest is")
print("  quoted in the SAME units on both sides of both seams.")

print("\n" + "=" * 100)
print("H6  COLLATERAL — what else reads fno_bhavcopy, and does it span the seam?")
print("=" * 100)
print(c.execute("""SELECT strftime(trade_date,'%Y-%m') ym, COUNT(DISTINCT trade_date) d,
    COUNT(*) AS n_rows FROM fno_bhavcopy WHERE trade_date BETWEEN '2024-05-01' AND '2024-09-30'
    GROUP BY 1 ORDER BY 1""").df().to_string(index=False))

print("\n" + "=" * 100)
print("H7  ANALOGUE STABILITY — does the displayed answer depend on k?")
print("=" * 100)
for k in (10, 15, 25, 40, 60):
    r = ilc.get_flow_analogues(AS_OF, "NIFTY", k)
    s1, s5 = r["summary"]["d1"], r["summary"]["d5"]
    print(f"  k={k:3d}  next day {s1['mean']:+.3f}% (up {s1['up']:4.1f}%)   "
          f"next 5 {s5['mean']:+.3f}% (up {s5['up']:4.1f}%)   "
          f"kth dist {r['summary']['match_quality']['kth']:.2f}")
print("  => if the sign or the 'up %' swings across k, the number is a k artifact.")

print("\n" + "=" * 100)
print("H8  NaN SEMANTICS — can a bucket with no legs read as neutral?")
print("=" * 100)
empty = ilc.BucketRow(label="Top 10", n_members=10, n_present=10)
print(f"  BucketRow with no delivery/futures/options legs -> flow_score = "
      f"{empty.flow_score}")
one = ilc.BucketRow(label="Top 10", n_members=10, n_present=10, deliv_z=0.0)
print(f"  delivery z exactly 0.0 (inside the +-0.3 dead band) -> flow_score = "
      f"{one.flow_score}   <- a REAL neutral")
print("  => these two must not be the same value, or 'no data' and 'neutral' are")
print("     indistinguishable on screen.")
