"""
Multi-timeframe (Daily x Weekly) price-action edge validation — WALK-FORWARD, causal.

Tests every scenario the user described, honestly, cross-sectionally on the full DCM
daily panel (371 trading days, ~1268 liquid stocks). All features use info <= t; forward
returns are t+1..t+H. Edge is measured RELATIVE to the cross-sectional median that day
(equal-weight universe) so we isolate stock SELECTION, not market beta.

Scenarios:
  A. Conditional momentum quality: among daily-UP stocks, do weekly-UP beat weekly-DOWN?
     (the core thesis: 'daily pop is false when weekly disagrees')
  B. Full 4-quadrant D x W alignment forward returns + hit-rates.
  C. Breakout: close breaks 20d Donchian by >1% out of a prior consolidation -> follow-through?
  D. Range position: near weekly support (turning up) vs near resistance.
  E. Continuous IC of each raw signal vs forward relative return (the real significance anchor).
"""
from __future__ import annotations
import sys
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

MIN_TO = 100.0     # turnover_lacs liquidity floor
D_MOM = 10         # daily momentum window (trading days)
W_MOM = 50         # 'weekly' trend window ~10 weeks
DON  = 20          # Donchian breakout window
HS   = [5, 10]     # forward horizons

print("loading panel...")
P = q("""
    WITH liq AS (
      SELECT DISTINCT symbol FROM daily_data
      WHERE series IN ('EQ','SM','ST') AND turnover_lacs>=? )
    SELECT b.symbol, b.trade_date, b.open_price o, b.high_price h,
           b.low_price l, b.close_price c
    FROM daily_data b INNER JOIN liq ON liq.symbol=b.symbol
    WHERE b.series IN ('EQ','SM','ST') AND b.close_price>0
    ORDER BY b.trade_date, b.symbol
""", [MIN_TO])
P["trade_date"] = pd.to_datetime(P["trade_date"])
C = P.pivot_table("c","trade_date","symbol")
H = P.pivot_table("h","trade_date","symbol")
L = P.pivot_table("l","trade_date","symbol")
dates = C.index
print(f"panel: {C.shape[0]} days x {C.shape[1]} symbols")

# require enough history to form weekly trend
logc = np.log(C)
# daily momentum (net) and daily efficiency ratio over D_MOM
d_net = C / C.shift(D_MOM) - 1.0
d_path = C.diff().abs().rolling(D_MOM).sum()
d_er = (C - C.shift(D_MOM)).abs() / d_path.replace(0, np.nan)
d_dir = np.sign(d_net)
# weekly trend: net over W_MOM + slope sign via EMA20w proxy (ema of weekly ~ ema span 50 daily)
w_net = C / C.shift(W_MOM) - 1.0
ema_s = C.ewm(span=20, adjust=False).mean()
ema_l = C.ewm(span=50, adjust=False).mean()
w_dir = np.sign(ema_s - ema_l)                     # weekly structure up/down
w_er = (C - C.shift(W_MOM)).abs() / C.diff().abs().rolling(W_MOM).sum().replace(0,np.nan)
# Donchian range position over W_MOM (0=at support,1=at resistance) + breakout over DON
roll_hi = H.rolling(W_MOM).max(); roll_lo = L.rolling(W_MOM).min()
range_pos = (C - roll_lo) / (roll_hi - roll_lo).replace(0, np.nan)
don_hi = H.shift(1).rolling(DON).max(); don_lo = L.shift(1).rolling(DON).min()
atr = (H - L).rolling(DON).mean() / C
atr_pctile = atr.rank(axis=0, pct=True)            # per-symbol time-series percentile of tightness
consol = atr.rolling(DON).mean().rank(axis=1, pct=True)  # cross-sectional (lower=tighter) proxy
brk_up = (C > don_hi * 1.01)
brk_dn = (C < don_lo * 0.99)

# forward RELATIVE returns (vs cross-sectional median that day)
fwd = {}
for Hh in HS:
    f = C.shift(-Hh) / C - 1.0
    fwd[Hh] = f.sub(f.median(axis=1), axis=0)       # relative to universe median

# walk-forward sample every 5 trading days, require warmup
STEP = 5; WARM = W_MOM + 5
rows = []
for i in range(WARM, len(dates) - max(HS) - 1, STEP):
    d = dates[i]
    rec = pd.DataFrame({
        "d_net": d_net.iloc[i], "d_er": d_er.iloc[i], "d_dir": d_dir.iloc[i],
        "w_net": w_net.iloc[i], "w_dir": w_dir.iloc[i], "w_er": w_er.iloc[i],
        "range_pos": range_pos.iloc[i], "consol": consol.iloc[i],
        "brk_up": brk_up.iloc[i], "brk_dn": brk_dn.iloc[i],
    })
    for Hh in HS: rec[f"f{Hh}"] = fwd[Hh].iloc[i]
    rec["date"] = d
    rows.append(rec.dropna(subset=["d_net","w_net","range_pos","f5","f10"]))
S = pd.concat(rows, ignore_index=True)
print(f"pooled stock-week obs: {len(S):,}\n")

def stat(mask, label):
    x = S[mask]
    if len(x) < 200: return f"  {label:42s} n={len(x):6d}  (thin)"
    out = f"  {label:42s} n={len(x):6d}"
    for Hh in HS:
        r = x[f"f{Hh}"]; out += f"   f{Hh}: {r.mean()*100:+.2f}% (win {(r>0).mean()*100:4.1f}%)"
    return out

print("="*110); print("A) CORE THESIS — among DAILY-UP stocks, does WEEKLY agreement matter?"); print("="*110)
up = S["d_dir"] > 0
print(stat(up & (S["w_dir"]>0), "daily-up + weekly-up (aligned)"))
print(stat(up & (S["w_dir"]<=0), "daily-up + weekly-DOWN (user's 'false pop')"))
print(stat(up, "  all daily-up (baseline)"))

print("\n"+"="*110); print("B) FULL 4-QUADRANT  Daily x Weekly direction"); print("="*110)
for dd,dl in [(1,"D-up"),(-1,"D-dn")]:
    for wd,wl in [(1,"W-up"),(-1,"W-dn")]:
        print(stat((S["d_dir"]==dd)&(S["w_dir"]==wd), f"{dl} + {wl}"))

print("\n"+"="*110); print("C) BREAKOUT follow-through (close breaks 20d Donchian by >1%)"); print("="*110)
print(stat(S["brk_up"], "breakout UP (any)"))
print(stat(S["brk_up"] & (S["w_dir"]>0), "breakout UP + weekly-up (aligned)"))
print(stat(S["brk_up"] & (S["consol"]<=0.33), "breakout UP + prior tight consolidation"))
print(stat(S["brk_dn"], "breakdown DN (any)"))
print(stat(S["brk_dn"] & (S["w_dir"]<0), "breakdown DN + weekly-dn (aligned)"))

print("\n"+"="*110); print("D) RANGE POSITION — support bounce vs resistance rejection"); print("="*110)
print(stat((S["range_pos"]<=0.20) & (S["w_dir"]>0), "near weekly SUPPORT + weekly-up (dip-buy)"))
print(stat((S["range_pos"]<=0.20) & (S["d_dir"]>0), "near SUPPORT + daily turning up"))
print(stat((S["range_pos"]>=0.80) & (S["d_dir"]<0), "near RESISTANCE + daily rolling over"))
print(stat((S["range_pos"]>=0.80), "near RESISTANCE (any)"))

print("\n"+"="*110); print("E) CONTINUOUS IC (Spearman, pooled per-date mean) — the significance anchor"); print("="*110)
def daily_ic(sig, Hh):                     # spearman = pearson on ranks (scipy-free)
    ics=[]
    for dt,g in S.groupby("date"):
        gg=g[[sig,f"f{Hh}"]].dropna()
        if len(gg)>=20:
            ics.append(gg[sig].rank().corr(gg[f"f{Hh}"].rank()))
    ics=np.array(ics);
    return ics.mean(), ics.std(), len(ics)
for sig in ["d_net","w_net","d_er","w_er","range_pos"]:
    for Hh in HS:
        m,s,n=daily_ic(sig,Hh)
        t = m/(s/np.sqrt(n)) if s>0 else 0
        print(f"  {sig:10s} vs f{Hh:<3d}  IC {m:+.3f}  ICIR {m/s if s>0 else 0:+.2f}  t {t:+.1f}  (n_days {n})")

print("\n"+"="*110); print("F) ROBUSTNESS — breakout-from-consolidation across time halves + net of cost"); print("="*110)
bo = S["brk_up"] & (S["consol"]<=0.33)
mid = S["date"].median()
for lab,mask in [("H1 (first half)", bo & (S["date"]<=mid)), ("H2 (second half)", bo & (S["date"]>mid))]:
    x=S[mask]
    if len(x)>=100:
        for Hh in HS:
            r=x[f"f{Hh}"]; print(f"  {lab:18s} f{Hh}: n={len(x):5d}  {r.mean()*100:+.2f}%  win {(r>0).mean()*100:.1f}%")
# per-date long-basket (breakout names) relative return -> t-stat over dates
bk = S[bo].groupby("date")["f10"].mean().dropna()
print(f"\n  per-date breakout-basket f10 relative: mean {bk.mean()*100:+.2f}%  t={bk.mean()/(bk.std()/np.sqrt(len(bk))):+.1f}  (n_dates {len(bk)})")
# vs random same-size baskets each date (MC null)
rng=np.random.default_rng(7); nulls=[]
for _ in range(500):
    tot=0;cnt=0
    for dt,g in S.groupby("date"):
        k=int((g["brk_up"]&(g["consol"]<=0.33)).sum())
        if k>0: tot+=g["f10"].sample(k,random_state=int(rng.integers(1e9))).mean(); cnt+=1
    nulls.append(tot/cnt)
nulls=np.array(nulls); real=bk.mean()
print(f"  MC null random-basket f10: mean {nulls.mean()*100:+.2f}%  p(real>null)={ (nulls<real).mean():.3f}")
