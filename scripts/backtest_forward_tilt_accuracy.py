"""
ACCURACY / RELIABILITY backtest of the SHIPPED forward-tilt ranking.

Faithful: reconstructs the engine's exact composite (0.60 rank(rs_2w) + 0.25 rank(rs_1w)
+ 0.15 rank(dv5d)) via the validated build_factors (causal), plus the engine's regime
label per date, then measures REALIZED forward returns. No lookahead.

Questions answered in plain terms:
  1. When a sector is OVERWEIGHT (top quartile), how often does it beat the median sector
     over the next 5/10/20d?  (per-call accuracy)  vs random 50%.
  2. Are forward returns MONOTONIC across rank quintiles?  (is the ranking meaningful)
  3. Rebalance win-rate: over independent (non-overlapping) periods, how often does the
     OW basket beat the UW basket, and beat the equal-weight benchmark?
  4. Top pick (#1) accuracy: beats median / beats Nifty, and average edge.
  5. REGIME-conditional accuracy — does accuracy collapse in the states the gate flags?
  6. Stability: first half vs second half.
  7. Per-sector reliability: which sectors the OW call is trustworthy for.
"""
from __future__ import annotations
import sys
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 60)
from scripts.factor_ic_diagnostic import load_panel, load_nifty, build_factors, HORIZONS

df = build_factors(load_panel(), load_nifty())
g = df.groupby("trade_date")
df["composite"] = (0.60*g["rs_2w"].transform(lambda x: x.rank(pct=True))
                 + 0.25*g["rs_1w"].transform(lambda x: x.rank(pct=True))
                 + 0.15*g["dv5d"].transform(lambda x: x.rank(pct=True)))
df["crank"] = df.groupby("trade_date")["composite"].rank(pct=True)

# cross-sectional median forward return per date (the benchmark each call is judged vs)
for h in HORIZONS:
    df[f"med_{h}"] = df.groupby("trade_date")[f"fwd_{h}"].transform("median")

# ── regime label per date (vectorized replica of engine _market_regime) ──────
nf = load_nifty().sort_values("trade_date").reset_index(drop=True)
close = nf["close_val"].astype(float) if "close_val" in nf else None
# load_nifty in factor_ic returns only trade_date+nret; reload close for EMAs
raw = None
from src.data.repository import query_dataframe
raw = query_dataframe("SELECT trade_date, close_val, pct_chg FROM index_data WHERE index_name='Nifty 50' ORDER BY trade_date")
raw["trade_date"] = pd.to_datetime(raw["trade_date"]); raw["nret"] = raw["pct_chg"].astype(float)
c = raw["close_val"].astype(float); r = raw["nret"]/100.0
ema20 = c.ewm(span=20, adjust=False).mean(); ema50 = c.ewm(span=50, adjust=False).mean()
vol20 = r.rolling(20).std()
volpct = pd.Series([ (vol20.iloc[:i+1] <= vol20.iloc[i]).mean() if i>20 else np.nan for i in range(len(vol20)) ])
lr = np.log1p(raw["nret"]/100.0); cr = lr.cumsum()
ret5 = (np.expm1(cr - cr.shift(5))*100); ret20 = (np.expm1(cr - cr.shift(20))*100)
def label(i):
    if i < 60 or np.isnan(vol20.iloc[i]): return "UNKNOWN"
    px=c.iloc[i]
    if ret5.iloc[i] <= -3 and ret20.iloc[i] > 0: return "REVERSAL"
    if px < ema20.iloc[i] < ema50.iloc[i]: return "TRENDING_DOWN"
    if np.isfinite(volpct.iloc[i]) and volpct.iloc[i] >= 0.80: return "HIGH_VOL"
    if px > ema20.iloc[i] > ema50.iloc[i]: return "TRENDING_UP"
    return "CHOPPY"
raw["regime"] = [label(i) for i in range(len(raw))]
df = df.merge(raw[["trade_date","regime"]], on="trade_date", how="left")

dfx = df.dropna(subset=["crank"]).copy()
print(f"panel: {dfx['sector'].nunique()} sectors x {dfx['trade_date'].nunique()} days "
      f"| {dfx['trade_date'].min().date()} -> {dfx['trade_date'].max().date()}")

# ============================================================================
print("\n"+"="*92)
print("1) PER-CALL ACCURACY — P(sector beats cross-sectional MEDIAN over next h)")
print("   OW = top quartile (crank>=0.75, the engine's OVERWEIGHT). Random ~ 50%.")
print("="*92)
print(f"  {'horizon':8s} {'OW acc':>8s} {'UW acc':>8s} {'all acc':>8s} {'OW avg vs med':>14s} {'n_OW':>7s}")
for h in HORIZONS:
    s = dfx.dropna(subset=[f"fwd_{h}"])
    beat = (s[f"fwd_{h}"] > s[f"med_{h}"]).astype(float)
    ow = s["crank"]>=0.75; uw = s["crank"]<=0.25
    edge = (s[f"fwd_{h}"] - s[f"med_{h}"])
    print(f"  h={h:<6d} {beat[ow].mean()*100:7.1f}% {beat[uw].mean()*100:7.1f}% {beat.mean()*100:7.1f}% "
          f"{edge[ow].mean():+13.2f}% {int(ow.sum()):7d}")

print("\n"+"="*92)
print("2) MONOTONICITY — mean forward return by composite QUINTILE (Q1 worst rank .. Q5 best)")
print("   Reliable ranking => returns rise Q1->Q5. Shown vs-Nifty (true rotation).")
print("="*92)
for h in [5,10,20]:
    s = dfx.dropna(subset=[f"fwd_{h}",f"n_fwd_{h}"]).copy()
    s["rel"] = s[f"fwd_{h}"] - s[f"n_fwd_{h}"]
    s["Q"] = pd.qcut(s["crank"], 5, labels=[1,2,3,4,5])
    m = s.groupby("Q", observed=True)["rel"].mean()
    mono = "MONOTONIC ✓" if list(m.values)==sorted(m.values) else "not strictly monotonic"
    print(f"  h={h:2d}  Q1 {m.get(1,float('nan')):+.2f}  Q2 {m.get(2,float('nan')):+.2f}  "
          f"Q3 {m.get(3,float('nan')):+.2f}  Q4 {m.get(4,float('nan')):+.2f}  Q5 {m.get(5,float('nan')):+.2f}  "
          f"| Q5-Q1 {m.get(5,0)-m.get(1,0):+.2f}%  [{mono}]")

print("\n"+"="*92)
print("3) REBALANCE WIN-RATE — independent (non-overlapping) H-day periods")
print("   How often does OW basket beat UW basket / beat equal-weight benchmark?")
print("="*92)
dates = np.array(sorted(dfx["trade_date"].unique()))
for h in [5,10]:
    piv_r = dfx.pivot_table("crank","trade_date","sector")
    piv_f = dfx.pivot_table(f"fwd_{h}","trade_date","sector")
    di = list(range(0, len(dates)-h-1, h))
    ow_gt_uw, ow_gt_bm, spreads = [], [], []
    for i in di:
        d = pd.Timestamp(dates[i])
        if d not in piv_r.index: continue
        rk = piv_r.loc[d].dropna(); fw = piv_f.loc[d]
        rk = rk[fw.reindex(rk.index).notna()]
        if len(rk) < 12: continue
        ow = fw.reindex(rk[rk>=0.75].index).mean()
        uw = fw.reindex(rk[rk<=0.25].index).mean()
        bm = fw.reindex(rk.index).mean()
        ow_gt_uw.append(ow>uw); ow_gt_bm.append(ow>bm); spreads.append(ow-uw)
    n=len(ow_gt_uw)
    print(f"  h={h:2d}: OW>UW {np.mean(ow_gt_uw)*100:4.0f}% | OW>benchmark {np.mean(ow_gt_bm)*100:4.0f}% "
          f"| avg OW-UW spread {np.mean(spreads):+.2f}% | periods {n}")

print("\n"+"="*92)
print("4) TOP-PICK (#1 ranked) ACCURACY")
print("="*92)
for h in [5,10]:
    top = dfx[dfx["crank"]==dfx.groupby("trade_date")["crank"].transform("max")].dropna(subset=[f"fwd_{h}",f"n_fwd_{h}"])
    bm = (top[f"fwd_{h}"] > top[f"med_{h}"]).mean()
    bn = (top[f"fwd_{h}"] > top[f"n_fwd_{h}"]).mean()
    print(f"  h={h:2d}: #1 beats median {bm*100:4.0f}% | beats Nifty {bn*100:4.0f}% | "
          f"avg vs-Nifty {np.mean(top[f'fwd_{h}']-top[f'n_fwd_{h}']):+.2f}% | n {len(top)}")

print("\n"+"="*92)
print("5) REGIME-CONDITIONAL accuracy (h=10, OW top-quartile beats median) — validates the gate")
print("="*92)
s = dfx.dropna(subset=["fwd_10"]); s=s[s["crank"]>=0.75]
beat = (s["fwd_10"]>s["med_10"]).astype(float)
for reg,gg in s.assign(beat=beat).groupby("regime"):
    print(f"  {reg:16s} OW acc {gg['beat'].mean()*100:5.1f}%  avg vs-med "
          f"{(gg['fwd_10']-gg['med_10']).mean():+.2f}%  n {len(gg)}")

print("\n"+"="*92)
print("6) STABILITY — quintile spread & OW accuracy, first half vs second half (h=10, vs-Nifty)")
print("="*92)
s = dfx.dropna(subset=["fwd_10","n_fwd_10"]).copy(); s["rel"]=s["fwd_10"]-s["n_fwd_10"]
mid = s["trade_date"].quantile(0.5)
for lab,seg in [("first half", s[s["trade_date"]<=mid]), ("second half", s[s["trade_date"]>mid])]:
    q = pd.qcut(seg["crank"],5,labels=[1,2,3,4,5])
    m = seg.groupby(q, observed=True)["rel"].mean()
    ow = seg[seg["crank"]>=0.75]; acc=(ow["fwd_10"]>ow["med_10"]).mean()
    print(f"  {lab:12s} Q5-Q1 {m.get(5,0)-m.get(1,0):+.2f}%  OW acc {acc*100:4.1f}%  "
          f"[{seg['trade_date'].min().date()} -> {seg['trade_date'].max().date()}]  n {len(seg)}")

print("\n"+"="*92)
print("7) PER-SECTOR RELIABILITY — when this sector was OW, did it beat median next 10d?")
print("="*92)
s = dfx[dfx["crank"]>=0.75].dropna(subset=["fwd_10"]).copy()
s["beat"]=(s["fwd_10"]>s["med_10"]).astype(float); s["edge"]=s["fwd_10"]-s["med_10"]
tab = s.groupby("sector").agg(n=("beat","size"), acc=("beat","mean"), avg_edge=("edge","mean"))
tab = tab[tab["n"]>=8].sort_values("acc", ascending=False)
tab["acc"]=(tab["acc"]*100).round(0); tab["avg_edge"]=tab["avg_edge"].round(2)
print(tab.to_string())
