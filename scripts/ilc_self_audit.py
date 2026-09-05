"""Adversarial audit of the Index & Large Cap panel I just shipped.

A  SURVIVORSHIP: the trend uses TODAY's 50 on 2022 data. Nifty reconstitutes twice
   a year, so ~9 of the 2022 members are missing and replaced by names that were
   promoted BECAUSE they outperformed. That biases 2022 breadth UP, which would
   understate "carried" days in 2022 and manufacture the rising trend. Tested
   against a membership-independent basket.
B  DELIVERY Z: the bucket delivery mean is taken over whatever names are present
   that day. ETERNAL/TMPV have 30%/18% coverage, so the mean jumps when the
   composition changes rather than when delivery changes.
C  FUTURES OI: an unweighted mean of per-name OI % changes. One thin contract at
   +300% can carry the whole bucket read.
D  MOVERS: head(3)/tail(3) on a bucket with <6 names present would list the same
   stock as both a top gainer and a top loser.
E  CARRY SPREAD: is index_data's Nifty return really comparable to the plain mean
   of MY 50 symbols, or is the identity leaking membership error?
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb, numpy as np, pandas as pd
from src.analytics.index_largecap import INDEX_BUCKETS, _DELIV_BASE

pd.set_option('display.width', 225)
c = duckdb.connect('data/market_data.duckdb', read_only=True)
B = INDEX_BUCKETS["NIFTY"]["buckets"]
ALL50 = tuple(s for v in B.values() for s in v)
ph = ",".join("?" * len(ALL50))

cash = c.execute(f"""SELECT trade_date,symbol,deliv_per,turnover_lacs,
    (close_price-prev_close)/NULLIF(prev_close,0)*100 r
  FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
    AND close_price>0 AND prev_close>0 AND trade_date>='2021-06-01'""", list(ALL50)).df()
cash['trade_date'] = pd.to_datetime(cash['trade_date'])
cash = cash[cash['r'].abs() < 40]
R = cash.pivot_table('r', 'trade_date', 'symbol').sort_index()
DL = cash.pivot_table('deliv_per', 'trade_date', 'symbol').sort_index()
nif = c.execute("""SELECT trade_date,pct_chg FROM index_data
    WHERE index_name='Nifty 50' AND trade_date>='2021-06-01'""").df()
nif['trade_date'] = pd.to_datetime(nif['trade_date'])
y = nif.set_index('trade_date')['pct_chg'].astype(float).reindex(R.index)

print("=" * 92)
print("A. SURVIVORSHIP — is the concentration trend an artifact of today's members?")
print("=" * 92)


def trend(Rm, yv, label, min_names=None):
    cov = Rm.notna().sum(axis=1)
    # for a wide, ragged universe the bar must be an ABSOLUTE name count, not a
    # share of the column space: 80%% of 1,735 tickers is never present on one day,
    # which silently emptied this test on the first run.
    k = cov >= (min_names if min_names else max(20, int(Rm.shape[1] * 0.8)))
    Rk, yk = Rm[k], yv[k]
    adv = (Rk > 0).sum(axis=1) / Rk.notna().sum(axis=1) * 100
    out = []
    for yr, g in yk.groupby(yk.index.year):
        if len(g) < 150: continue
        a = adv.reindex(g.index)
        out.append((int(yr), len(g), round(float(((g > 0) & (a < 50)).mean() * 100), 1)))
    print(f"  {label}")
    print("    " + "  ".join(f"{yr}:{p}%(n{n})" for yr, n, p in out))
    return out


base = trend(R, y, f"TODAY'S 50 (as shipped, {R.shape[1]} names)")

# membership-independent basket: every stock liquid enough, point-in-time, no index list
uni = c.execute("""
  WITH liq AS (SELECT trade_date, symbol,
      LAG(turnover_lacs) OVER (PARTITION BY symbol ORDER BY trade_date) tl,
      (close_price-prev_close)/NULLIF(prev_close,0)*100 r
    FROM daily_data WHERE series IN ('EQ','SM','ST') AND close_price>0
      AND prev_close>0 AND trade_date>='2021-06-01')
  SELECT trade_date, symbol, r FROM liq WHERE tl >= 10000 AND ABS(r) < 40""").df()
uni['trade_date'] = pd.to_datetime(uni['trade_date'])
U = uni.pivot_table('r', 'trade_date', 'symbol').sort_index()
print(f"\n  membership-independent universe: {U.shape[1]} symbols "
      f"(prior-session turnover >= 100 Cr), median {int(U.notna().sum(axis=1).median())} names/day")
ub = trend(U, y.reindex(U.index), "BROAD LIQUID UNIVERSE (no index list at all)", min_names=120)

# fixed-history subset: only names present for the whole span
full = R.columns[R.notna().mean() > 0.97]
fb = trend(R[full], y, f"ONLY FULL-HISTORY NAMES ({len(full)} of 50)")
print("\n  => if the rise survives a basket that never uses the index list, it is real.")

print("\n" + "=" * 92)
print("B. DELIVERY Z — does the bucket mean move with COMPOSITION rather than delivery?")
print("=" * 92)
for lbl, mem in B.items():
    m = [s for s in mem if s in DL.columns]
    sub = DL[m]
    cnt = sub.notna().sum(axis=1)
    dm = sub.mean(axis=1)
    k = cnt > 0
    corr = cnt[k].corr(dm[k])
    # how much does the mean jump on days the member count changes?
    dchg = cnt.diff().fillna(0) != 0
    jump = dm.diff().abs()
    print(f"  {lbl:8s} names/day min {int(cnt.min())} max {int(cnt.max())}  "
          f"corr(count, mean delivery) {corr:+.3f}  "
          f"mean |Δdelivery| on count-change days {jump[dchg].mean():.2f} "
          f"vs {jump[~dchg].mean():.2f} on stable days")

print("\n" + "=" * 92)
print("C. FUTURES OI — is the bucket mean driven by thin-contract outliers?")
print("=" * 92)
fut = c.execute(f"""
  WITH nf AS (SELECT trade_date,symbol,MIN(expiry_date) e FROM fno_bhavcopy
      WHERE instrument='FUTSTK' AND expiry_date>trade_date AND open_interest>0
        AND symbol IN ({ph}) GROUP BY 1,2),
  f AS (SELECT b.trade_date,b.symbol,b.open_interest,b.close_price,b.expiry_date
        FROM fno_bhavcopy b JOIN nf n ON n.symbol=b.symbol AND n.trade_date=b.trade_date
          AND n.e=b.expiry_date WHERE b.instrument='FUTSTK')
  SELECT trade_date,symbol,open_interest,
         LAG(open_interest) OVER (PARTITION BY symbol,expiry_date ORDER BY trade_date) poi
  FROM f""", list(ALL50)).df()
fut = fut[fut['poi'].notna() & (fut['poi'] > 0)].copy()
fut['oi_pct'] = (fut['open_interest'] - fut['poi']) / fut['poi'] * 100
q = fut['oi_pct'].quantile([.001, .01, .5, .99, .999])
print("  per-name daily OI % change distribution:")
print("   " + "  ".join(f"p{k*100:g}={v:+.1f}" for k, v in q.items()))
print(f"   |oi_pct| > 50%: {(fut['oi_pct'].abs()>50).mean()*100:.2f}% of rows   "
      f"> 100%: {(fut['oi_pct'].abs()>100).mean()*100:.3f}%   max {fut['oi_pct'].max():,.0f}%")
fut['trade_date'] = pd.to_datetime(fut['trade_date'])
FO = fut.pivot_table('oi_pct', 'trade_date', 'symbol')
for lbl, mem in B.items():
    m = [s for s in mem if s in FO.columns]
    raw = FO[m].mean(axis=1)
    med = FO[m].median(axis=1)
    wins = FO[m].clip(-50, 50).mean(axis=1)
    d = (raw - wins).abs()
    print(f"  {lbl:8s} mean vs winsorised(+-50%): mean|diff| {d.mean():.2f}pp  "
          f"max {d.max():.1f}pp  days where they differ >2pp: "
          f"{(d>2).mean()*100:.1f}%   corr(mean,median) {raw.corr(med):+.3f}")

print("\n" + "=" * 92)
print("D. MOVERS OVERLAP — can a stock appear as both top gainer and top loser?")
print("=" * 92)
worst = 0
for lbl, mem in B.items():
    m = [s for s in mem if s in R.columns]
    cnt = R[m].notna().sum(axis=1)
    bad = (cnt < 6) & (cnt > 0)
    worst = max(worst, int(bad.sum()))
    print(f"  {lbl:8s} sessions with <6 names present: {int(bad.sum())} "
          f"(min names on any day: {int(cnt.min())})")
print(f"  => head(3)/tail(3) overlap is possible on {worst} sessions; a guard is cheap.")

print("\n" + "=" * 92)
print("E. CARRY SPREAD — is the equal-weight leg comparable to index_data's return?")
print("=" * 92)
cov = R.notna().sum(axis=1)
ew = R.mean(axis=1)
sp = (y - ew).dropna()
print(f"  sessions {len(sp)}   mean spread {sp.mean():+.4f}pp   sd {sp.std():.4f}")
print(f"  corr(spread, names present) {sp.corr(cov.reindex(sp.index)):+.3f}  "
      f"(a strong link would mean the spread is measuring COVERAGE, not concentration)")
for n, g in sp.groupby(cov.reindex(sp.index)):
    if len(g) > 30:
        print(f"    {int(n)} names present: {len(g):4d} sessions, mean spread {g.mean():+.4f}pp")
