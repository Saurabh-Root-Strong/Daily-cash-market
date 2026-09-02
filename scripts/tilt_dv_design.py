"""Design the horizon-scaled delivery factor under the constraint:
   1-2 wk (h=10) must stay BIT-IDENTICAL to the shipped 5/100.

Candidate parameterisations (all reduce to 5/100 at h=10 unless noted):
   S1  flow=h/2, base=100        scale flow only            (5/100 at h=10 OK)
   S2  flow=h/2, base=10h        scale both                 (5/100 at h=10 OK)  <- user's design
   S3  flow=h/2, base=min(10h,B) same, capped by data       (5/100 at h=10 OK)
   S4  flow=h,   base=10h        (10/100 at h=10 -> BREAKS the 1-2wk build)
   S0  shipped 5/100

Also measures the FLOW-IN-BASE OVERLAP problem: the baseline includes the flow
window, so a surge inflates its own denominator. Shipped overlap is 5/100 = 5%.
Scaling flow but NOT base drives it to 30/100 = 30% at 11-12wk.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from datetime import date
from src.data.repository import query_dataframe
from src.analytics.sector_forward_tilt import (_W_RS2, _W_RS1, _W_DV5, _OW_RANK,
                                               _load_sector_panel)

pd.set_option('display.width', 250)
HZ = [10, 20, 30, 40, 50, 60]
LBL = {10: "1-2 wk", 20: "3-4 wk", 30: "5-6 wk", 40: "7-8 wk", 50: "9-10 wk", 60: "11-12 wk"}

print("=== HARD CONSTRAINT: how much history does the LIVE loader actually give? ===")
p = _load_sector_panel(date(2026, 9, 1), 100.0)
p['trade_date'] = pd.to_datetime(p['trade_date'])
n_sess = p['trade_date'].nunique()
print(f"  _load_sector_panel returns {n_sess} trading sessions "
      f"({p['trade_date'].min():%Y-%m-%d} .. {p['trade_date'].max():%Y-%m-%d})")
print(f"  a base of 10*h needs: " + "  ".join(f"{LBL[h]}={10*h}" for h in HZ))
print(f"  -> feasible today only for h where 10h <= {n_sess}: "
      f"{[LBL[h] for h in HZ if 10*h <= n_sess] or 'NONE beyond 1-2wk'}")
print("  => S2 as-written needs _load_sector_panel widened; S3 caps instead.")

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
EXC = {h: (lambda f: f.sub(f.mean(axis=1), axis=0))(
    np.expm1(lg.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1))*100.0) for h in HZ}


def nw_t(s, lag):
    x = s.dropna().values.astype(float); n = len(x)
    if n < 30: return float('nan')
    e = x-x.mean(); v = (e@e)/n
    for L in range(1, lag+1): v += 2*(1-L/(lag+1))*((e[L:]@e[:-L])/n)
    return x.mean()/math.sqrt(v/n) if v > 0 else float('nan')


def dvf(flow, base):
    return dvc.rolling(flow).mean() / dvc.shift(1).rolling(base).mean()


def ow(rank, h):
    ex = EXC[h].where(rank >= _OW_RANK)
    v = ex.values[~np.isnan(ex.values)]
    return v.mean(), nw_t(ex.mean(axis=1).dropna(), h)


DESIGNS = {
    'S0 shipped 5/100':       lambda h: (5, 100),
    'S1 h/2 / 100':           lambda h: (max(2, h//2), 100),
    'S2 h/2 / 10h':           lambda h: (max(2, h//2), 10*h),
    'S3 h/2 / min(10h,150)':  lambda h: (max(2, h//2), min(10*h, 150)),
    'S4 h / 10h  (breaks 1-2wk)': lambda h: (h, 10*h),
}

print("\n=== A. FLOW-IN-BASE OVERLAP (numerator sits inside the denominator) ===")
rows = []
for h in HZ:
    r = dict(horizon=LBL[h])
    for k, fn in DESIGNS.items():
        fl, bs = fn(h)
        r[k] = f"{fl}/{bs} = {fl/bs*100:.0f}%"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== B. DOES h=10 STAY IDENTICAL? (requirement) ===")
for k, fn in DESIGNS.items():
    fl, bs = fn(10)
    print(f"  {k:30s} at h=10 -> flow={fl} base={bs}   "
          f"{'IDENTICAL' if (fl, bs) == (5, 100) else '*** CHANGES THE SHIPPED 1-2wk BUILD ***'}")

print("\n=== C. OVERWEIGHT-bucket forward excess under each design ===")
rows = []
for h in HZ:
    L, S = h, max(2, h//2)
    rL = _rk(_tr(L).sub(np.expm1(nlg.rolling(L).sum())*100.0, axis=0))
    rS = _rk(_tr(S).sub(np.expm1(nlg.rolling(S).sum())*100.0, axis=0))
    r = dict(horizon=LBL[h])
    for k, fn in DESIGNS.items():
        fl, bs = fn(h)
        rank = _rk(_W_RS2*rL + _W_RS1*rS + _W_DV5*_rk(dvf(fl, bs)))
        pp, t = ow(rank, h)
        r[k] = f"{pp:+.3f} t{t:+.2f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== D. STANDALONE IC of the dv factor under each design ===")
rows = []
for h in HZ:
    r = dict(horizon=LBL[h])
    for k, fn in DESIGNS.items():
        fl, bs = fn(h)
        ic = _rk(dvf(fl, bs)).corrwith(_rk(EXC[h]), axis=1).dropna()
        r[k] = f"{ic.mean():+.4f} t{nw_t(ic, h):+.2f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== E. HOW MUCH HISTORY EACH DESIGN NEEDS vs what the loader gives ===")
for k, fn in DESIGNS.items():
    need = max(fn(h)[1] for h in HZ)
    print(f"  {k:30s} max base = {need:4d} sessions  "
          f"{'OK' if need <= n_sess else f'NEEDS LOADER WIDENED (have {n_sess})'}")

print("\n=== F. SANITY: does S3's cap bite, and where? ===")
for h in HZ:
    fl, bs = DESIGNS['S3 h/2 / min(10h,150)'](h)
    print(f"  {LBL[h]:9s} flow={fl:2d} base={bs:3d} "
          f"{'(capped from ' + str(10*h) + ')' if 10*h > 150 else ''}  overlap {fl/bs*100:.0f}%")
