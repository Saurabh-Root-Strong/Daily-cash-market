"""Does picking a forward horizon actually forecast THAT horizon?

Answers, per horizon h in {10,20,30,40,50,60}:
  H1 do the lists even CHANGE with h? (top-4 / bottom-4 overlap)
  H2 OVERWEIGHT bucket: forward EXCESS vs the equal-weight sector basket over h
  H3 OVERWEIGHT bucket: forward ABSOLUTE return over h (what a user reads as
     "will perform well") and how often it is simply positive
  H4 UNDERWEIGHT bucket: same, i.e. is "AVOID/trim" a worst-performer forecast
  H5 horizon-matching: is the h-tilt better AT h than the 1-2wk tilt is at h?
Inference: Newey-West at lag=h on the per-date cross-sectional mean (windows overlap).
Ranking math mirrors sector_forward_tilt._tilt_history (rank bands, no gates).
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.data.repository import query_dataframe
from src.analytics.sector_forward_tilt import (_W_RS2, _W_RS1, _W_DV5, _DV_FLOW,
                                               _DV_BASE, _OW_RANK, _UW_RANK)

pd.set_option('display.width', 250)
HZ = [10, 20, 30, 40, 50, 60]
LBL = {10: "1-2 wk", 20: "3-4 wk", 30: "5-6 wk", 40: "7-8 wk", 50: "9-10 wk", 60: "11-12 wk"}

panel = query_dataframe("""
    WITH base AS (
        SELECT s.sector, b.trade_date, b.turnover_lacs, b.deliv_per,
               (b.close_price - b.prev_close)/NULLIF(b.prev_close,0)*100 AS raw_r,
               LEAST(GREATEST((b.close_price-b.prev_close)/NULLIF(b.prev_close,0)*100,-25),25) AS r,
               LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol ORDER BY b.trade_date) AS w_lag
        FROM daily_data b INNER JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ','SM','ST') AND s.sector NOT IN ('ETF','Others')
    )
    SELECT sector, trade_date,
           SUM(turnover_lacs*deliv_per/100.0)/100.0 AS daily_dv_cr,
           SUM(w_lag*r)/NULLIF(SUM(CASE WHEN r IS NOT NULL THEN w_lag END),0) AS wtd_ret_pct
    FROM base WHERE w_lag >= 100 AND ABS(raw_r) < 40
    GROUP BY sector, trade_date ORDER BY sector, trade_date
""")
nf = query_dataframe("SELECT trade_date, close_val, pct_chg FROM index_data "
                     "WHERE index_name='Nifty 50' ORDER BY trade_date")

ret = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
dvc = panel.pivot_table("daily_dv_cr", "trade_date", "sector").sort_index()
ret.index = pd.to_datetime(ret.index); dvc.index = pd.to_datetime(dvc.index)
nf['trade_date'] = pd.to_datetime(nf['trade_date'])
nser = nf.set_index('trade_date')['pct_chg'].astype(float).reindex(ret.index)
print(f"panel: {ret.shape[0]} sessions x {ret.shape[1]} sectors  "
      f"{ret.index.min():%Y-%m-%d} .. {ret.index.max():%Y-%m-%d}")

lg = np.log1p(ret / 100.0)
nlg = np.log1p(nser.fillna(0.0) / 100.0)
_tr = lambda n: np.expm1(lg.rolling(n).sum()) * 100.0
_rk = lambda d: d.rank(axis=1, pct=True)
dv5 = dvc.rolling(_DV_FLOW).mean() / dvc.shift(1).rolling(_DV_BASE).mean()

# forward h-session return per sector, and the equal-weight sector basket
FWD, EXC = {}, {}
for h in HZ:
    f = (np.expm1(lg.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1)) * 100.0)
    FWD[h] = f
    EXC[h] = f.sub(f.mean(axis=1), axis=0)

RANK = {}
for h in HZ:
    L, S = h, max(2, h // 2)
    rsL = _tr(L).sub(np.expm1(nlg.rolling(L).sum()) * 100.0, axis=0)
    rsS = _tr(S).sub(np.expm1(nlg.rolling(S).sum()) * 100.0, axis=0)
    RANK[h] = _rk(_W_RS2 * _rk(rsL) + _W_RS1 * _rk(rsS) + _W_DV5 * _rk(dv5))


def nw_t(s: pd.Series, lag: int):
    x = s.dropna().values.astype(float)
    n = len(x)
    if n < 30: return float('nan'), n
    e = x - x.mean(); v = (e @ e) / n
    for L in range(1, lag + 1):
        v += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    return x.mean() / math.sqrt(v / n), n


print("\n=== H1. DO THE LISTS CHANGE WITH THE HORIZON? (mean overlap of top-4 / bottom-4) ===")
top = {h: RANK[h].apply(lambda r: set(r.nlargest(4).index) if r.notna().sum() >= 8 else set(), axis=1) for h in HZ}
bot = {h: RANK[h].apply(lambda r: set(r.nsmallest(4).index) if r.notna().sum() >= 8 else set(), axis=1) for h in HZ}
ov = pd.DataFrame(index=[LBL[h] for h in HZ], columns=[LBL[h] for h in HZ], dtype=float)
for a in HZ:
    for b in HZ:
        m = [len(x & y) / 4 * 100 for x, y in zip(top[a], top[b]) if x and y]
        ov.loc[LBL[a], LBL[b]] = np.mean(m)
print("TOP-4 (BUY list) overlap %:"); print(ov.round(0).to_string())
print(f"\n1-2wk vs 3-4wk buy-list overlap: {ov.loc['1-2 wk','3-4 wk']:.0f}%  "
      f"bottom-4: {np.mean([len(x&y)/4*100 for x,y in zip(bot[10],bot[20]) if x and y]):.0f}%")

print("\n=== H2/H3/H4. WHAT EACH BUCKET ACTUALLY DID OVER ITS OWN HORIZON ===")
rows = []
for h in HZ:
    r = RANK[h]
    for lab, msk in [("OVERWEIGHT", r >= _OW_RANK), ("UNDERWEIGHT", r <= _UW_RANK)]:
        ex = EXC[h].where(msk); ab = FWD[h].where(msk)
        per_date = ex.mean(axis=1).dropna()
        t, nd = nw_t(per_date, h)
        exv = ex.values[~np.isnan(ex.values)]
        abv = ab.values[~np.isnan(ab.values)]
        rows.append(dict(horizon=LBL[h], bucket=lab, n=len(exv),
                         excess_pp=exv.mean(), NW_t=t, beat_avg_pct=(exv > 0).mean() * 100,
                         abs_ret_pp=abv.mean(), abs_up_pct=(abv > 0).mean() * 100))
E = pd.DataFrame(rows)
print(E.round(2).to_string(index=False))

print("\n=== H3b. BASELINE — what did the AVERAGE sector do over the same h? ===")
for h in HZ:
    a = FWD[h].values[~np.isnan(FWD[h].values)]
    print(f"  {LBL[h]:9s} avg sector {a.mean():+6.2f}pp over {h} sessions, positive {(a>0).mean()*100:5.1f}% of the time")

print("\n=== H5. IS THE HORIZON-MATCHED LOOKBACK ACTUALLY BETTER AT THAT HORIZON? ===")
print("  (OW excess over h, ranked by the h-tilt vs ranked by the shipped 1-2wk tilt)")
for h in HZ:
    out = []
    for src in (h, 10):
        msk = RANK[src] >= _OW_RANK
        ex = EXC[h].where(msk)
        t, _ = nw_t(ex.mean(axis=1).dropna(), h)
        v = ex.values[~np.isnan(ex.values)]
        out.append((v.mean(), t))
    print(f"  {LBL[h]:9s} matched({h}d lookback) {out[0][0]:+.3f}pp t{out[0][1]:+.2f}   "
          f"| fixed(10d lookback) {out[1][0]:+.3f}pp t{out[1][1]:+.2f}")

print("\n=== H6. ERA STABILITY of the OW excess (does the sign hold?) ===")
eras = [("2018-21", "2018-01-01", "2021-12-31"), ("2022-24", "2022-01-01", "2024-12-31"),
        ("2025-26", "2025-01-01", "2026-12-31")]
rows = []
for h in HZ:
    msk = RANK[h] >= _OW_RANK
    ex = EXC[h].where(msk)
    d = dict(horizon=LBL[h])
    for nm, a, b in eras:
        w = ex.loc[a:b].values
        d[nm] = np.nanmean(w)
    rows.append(d)
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== H7. HOW OFTEN IS AN OVERWEIGHT SECTOR SIMPLY DOWN IN ABSOLUTE TERMS? ===")
for h in [10, 20]:
    msk = RANK[h] >= _OW_RANK
    ab = FWD[h].where(msk).values
    ab = ab[~np.isnan(ab)]
    ex = EXC[h].where(msk).values; ex = ex[~np.isnan(ex)]
    both = FWD[h].where(msk).values.ravel(); bo = EXC[h].where(msk).values.ravel()
    m = ~np.isnan(both) & ~np.isnan(bo)
    beat_but_down = ((bo[m] > 0) & (both[m] < 0)).mean() * 100
    print(f"  {LBL[h]:9s} OVERWEIGHT: down in absolute terms {(ab<0).mean()*100:5.1f}% of the time; "
          f"beat the average sector WHILE FALLING {beat_but_down:5.1f}% of the time")

print("\n=== H8. FAIR BASELINE for 'beat the average sector' (skew makes <50% normal) ===")
for h in [10, 20, 40, 60]:
    r = RANK[h]; ex = EXC[h]
    allv = ex.values[~np.isnan(ex.values)]
    neu = ex.where((r > _UW_RANK) & (r < _OW_RANK)).values
    neu = neu[~np.isnan(neu)]
    ow = ex.where(r >= _OW_RANK).values; ow = ow[~np.isnan(ow)]
    uw = ex.where(r <= _UW_RANK).values; uw = uw[~np.isnan(uw)]
    print(f"  {LBL[h]:9s} beat-avg%: ALL {(allv>0).mean()*100:5.2f} | NEUTRAL {(neu>0).mean()*100:5.2f} "
          f"| OW {(ow>0).mean()*100:5.2f} | UW {(uw>0).mean()*100:5.2f}   "
          f"(OW-NEU {(ow>0).mean()*100-(neu>0).mean()*100:+.2f}pp, UW-NEU {(uw>0).mean()*100-(neu>0).mean()*100:+.2f}pp)")
    print(f"            median excess pp: NEUTRAL {np.median(neu):+.3f} | OW {np.median(ow):+.3f} | UW {np.median(uw):+.3f}")
