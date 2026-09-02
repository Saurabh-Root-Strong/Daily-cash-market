"""Should the delivery factor scale with the horizon, like the RS legs do?

SHIPPED: rs_long = h, rs_short = h/2  (scaled)  but dv5d = 5d / 100d  (FIXED).
So at 11-12wk the momentum legs look back 60d/30d while the flow factor still
reads the last 5 sessions -- 1/12th of the forecast window.

Tested per horizon h in {10,20,30,40,50,60}:
  D1 standalone: does the delivery factor predict ANYTHING at horizon h,
     at the shipped 5/100 or at any scaled window?
  D2 composite: OVERWEIGHT-bucket forward excess under
       (a) shipped        0.60 rsL + 0.25 rsS + 0.15 dv(5/100)
       (b) scaled flow    same, dv(h/2 / 100)
       (c) scaled both    same, dv(h / 10h)
       (d) dv DROPPED     0.706 rsL + 0.294 rsS   (weights renormalised)
  D3 how much does dv5d even move the list? (rank churn it causes)
Inference: Newey-West at lag=h on the per-date cross-sectional mean.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.data.repository import query_dataframe
from src.analytics.sector_forward_tilt import (_W_RS2, _W_RS1, _W_DV5,
                                               _DV_FLOW, _DV_BASE, _OW_RANK, _UW_RANK)

pd.set_option('display.width', 250)
HZ = [10, 20, 30, 40, 50, 60]
LBL = {10: "1-2 wk", 20: "3-4 wk", 30: "5-6 wk", 40: "7-8 wk", 50: "9-10 wk", 60: "11-12 wk"}

panel = query_dataframe("""
    WITH base AS (
        SELECT s.sector, b.trade_date, b.turnover_lacs, b.deliv_per,
               (b.close_price-b.prev_close)/NULLIF(b.prev_close,0)*100 AS raw_r,
               LEAST(GREATEST((b.close_price-b.prev_close)/NULLIF(b.prev_close,0)*100,-25),25) AS r,
               LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol ORDER BY b.trade_date) AS w_lag
        FROM daily_data b INNER JOIN v_sector_master s ON b.symbol=s.symbol
        WHERE b.series IN ('EQ','SM','ST') AND s.sector NOT IN ('ETF','Others')
    )
    SELECT sector, trade_date,
           SUM(turnover_lacs*deliv_per/100.0)/100.0 AS daily_dv_cr,
           SUM(w_lag*r)/NULLIF(SUM(CASE WHEN r IS NOT NULL THEN w_lag END),0) AS wtd_ret_pct
    FROM base WHERE w_lag>=100 AND ABS(raw_r)<40
    GROUP BY sector, trade_date ORDER BY sector, trade_date
""")
nf = query_dataframe("SELECT trade_date, pct_chg FROM index_data "
                     "WHERE index_name='Nifty 50' ORDER BY trade_date")
ret = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
dvc = panel.pivot_table("daily_dv_cr", "trade_date", "sector").sort_index()
ret.index = pd.to_datetime(ret.index); dvc.index = pd.to_datetime(dvc.index)
nf['trade_date'] = pd.to_datetime(nf['trade_date'])
nser = nf.set_index('trade_date')['pct_chg'].astype(float).reindex(ret.index)
lg = np.log1p(ret/100.0); nlg = np.log1p(nser.fillna(0.0)/100.0)
_tr = lambda n: np.expm1(lg.rolling(n).sum())*100.0
_rk = lambda d: d.rank(axis=1, pct=True)
print(f"panel {ret.shape[0]} sessions x {ret.shape[1]} sectors "
      f"{ret.index.min():%Y-%m-%d}..{ret.index.max():%Y-%m-%d}")

EXC = {}
for h in HZ:
    f = np.expm1(lg.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1))*100.0
    EXC[h] = f.sub(f.mean(axis=1), axis=0)


def nw_t(s, lag):
    x = s.dropna().values.astype(float); n = len(x)
    if n < 30: return float('nan')
    e = x-x.mean(); v = (e@e)/n
    for L in range(1, lag+1): v += 2*(1-L/(lag+1))*((e[L:]@e[:-L])/n)
    return x.mean()/math.sqrt(v/n) if v > 0 else float('nan')


def dv_factor(flow, base):
    return dvc.rolling(flow).mean() / dvc.shift(1).rolling(base).mean()


def ow_stats(rank, h):
    m = rank >= _OW_RANK
    ex = EXC[h].where(m)
    v = ex.values[~np.isnan(ex.values)]
    return v.mean(), nw_t(ex.mean(axis=1).dropna(), h), len(v)


print("\n=== D1. STANDALONE — does the delivery factor predict anything, at any window? ===")
print("    (cross-sectional Spearman IC of the dv factor vs forward excess at h)")
rows = []
for h in HZ:
    r = dict(horizon=LBL[h])
    for lab, (fl, bs) in {'dv 5/100 (shipped)': (5, 100),
                          'dv h/2 /100': (max(2, h//2), 100),
                          'dv h /100': (h, 100),
                          'dv h /10h': (h, min(10*h, 400))}.items():
        d = dv_factor(fl, bs)
        ic = _rk(d).corrwith(_rk(EXC[h]), axis=1).dropna()
        r[lab] = f"{ic.mean():+.4f} (t{nw_t(ic, h):+.2f})"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== D2. COMPOSITE — OVERWEIGHT bucket forward excess under 4 builds ===")
rows = []
for h in HZ:
    L, S = h, max(2, h//2)
    rsL = _tr(L).sub(np.expm1(nlg.rolling(L).sum())*100.0, axis=0)
    rsS = _tr(S).sub(np.expm1(nlg.rolling(S).sum())*100.0, axis=0)
    rL, rS = _rk(rsL), _rk(rsS)
    builds = {
        'a shipped dv 5/100': _rk(_W_RS2*rL + _W_RS1*rS + _W_DV5*_rk(dv_factor(5, 100))),
        'b dv flow=h/2':      _rk(_W_RS2*rL + _W_RS1*rS + _W_DV5*_rk(dv_factor(max(2, h//2), 100))),
        'c dv flow=h base=10h': _rk(_W_RS2*rL + _W_RS1*rS + _W_DV5*_rk(dv_factor(h, min(10*h, 400)))),
        'd dv DROPPED':       _rk((_W_RS2/(_W_RS2+_W_RS1))*rL + (_W_RS1/(_W_RS2+_W_RS1))*rS),
    }
    r = dict(horizon=LBL[h])
    for k, rank in builds.items():
        pp, t, n = ow_stats(rank, h)
        r[k] = f"{pp:+.3f}pp t{t:+.2f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== D3. HOW MUCH DOES dv5d ACTUALLY MOVE THE BUY LIST? ===")
for h in HZ:
    L, S = h, max(2, h//2)
    rsL = _tr(L).sub(np.expm1(nlg.rolling(L).sum())*100.0, axis=0)
    rsS = _tr(S).sub(np.expm1(nlg.rolling(S).sum())*100.0, axis=0)
    rL, rS = _rk(rsL), _rk(rsS)
    with_dv = _rk(_W_RS2*rL + _W_RS1*rS + _W_DV5*_rk(dv_factor(5, 100)))
    no_dv = _rk((_W_RS2/(_W_RS2+_W_RS1))*rL + (_W_RS1/(_W_RS2+_W_RS1))*rS)
    a = with_dv.apply(lambda r: set(r.nlargest(6).index) if r.notna().sum() >= 8 else set(), axis=1)
    b = no_dv.apply(lambda r: set(r.nlargest(6).index) if r.notna().sum() >= 8 else set(), axis=1)
    ov = np.mean([len(x & y)/6*100 for x, y in zip(a, b) if x and y])
    diff = np.mean([6-len(x & y) for x, y in zip(a, b) if x and y])
    print(f"  {LBL[h]:9s} buy-list overlap with/without dv5d = {ov:5.1f}%  "
          f"({diff:.2f} of 6 names changed on an average day)")

print("\n=== D4. IS THE 5-DAY FLOW EVEN STABLE ACROSS THE HOLD? ===")
d5 = dv_factor(5, 100)
for h in HZ:
    a = d5.shift(-h)
    keep = (_rk(d5) >= 0.75) & (_rk(a) >= 0.75)
    top = (_rk(d5) >= 0.75)
    print(f"  {LBL[h]:9s} a sector in the top-quartile of dv5d today is still there "
          f"{keep.sum().sum()/top.sum().sum()*100:5.1f}% of the time {h} sessions later "
          f"(random = 25%)")
