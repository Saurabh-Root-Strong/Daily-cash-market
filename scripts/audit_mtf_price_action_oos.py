"""
ADVERSARIAL AUDIT of the shipped MTF price-action edge, on 4 YEARS of previous data
(211 F&O-universe daily bars, 2022-05..2026-06) — well before the DCM DuckDB window
(2024-12..2026-07) that the feature was built on. Attacks every weak point:

  1. NON-OVERLAPPING t-stats (step = horizon) — kills the autocorrelation that inflated
     the original t=3.2 (10d fwd stepped 5d = 50% overlap).
  2. ERA SPLIT: out-of-window (pre 2024-12-26) vs in-window — is it curve-fit to the bull?
  3. REGIME split (Nifty bull/bear/chop) — does the breakout survive non-bull tape?
  4. PARAMETER sensitivity (margin / consol pctl / Donchian / horizon) — is 1%/20d cherry-picked?
  5. NET-OF-COST — does a realistic round-trip eat the edge?
  6. ALIGNMENT gate re-test (non-overlapping, per era).

All features causal (<= t). Forward returns RELATIVE to the day's cross-sectional median
within the 211-name universe (isolates selection, removes market beta).
"""
from __future__ import annotations
import sys, glob, os
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd

DDIR = r"d:/Python Projects/Tradebot/data/historical/daily"
IS_START = pd.Timestamp("2024-12-26")   # DCM DuckDB window start = in-sample boundary
D_MOM, W_MOM, DON = 10, 20, 20
BRK_M, CONSOL_P, NEAR = 0.01, 0.33, 0.20

# ── load panel ────────────────────────────────────────────────────────────────
def load(sym_glob, tag):
    frames={}
    for f in glob.glob(os.path.join(DDIR, sym_glob)):
        if "INDEX" in f: continue
        s=os.path.basename(f).replace("NSE_","").replace("_EQ_daily.parquet","")
        d=pd.read_parquet(f); d["ts"]=pd.to_datetime(d["ts"]).dt.normalize()
        frames[s]=d.set_index("ts")
    return frames
F=load("*_EQ_daily.parquet","eq")
C=pd.DataFrame({s:d["close"] for s,d in F.items()}).sort_index()
H=pd.DataFrame({s:d["high"]  for s,d in F.items()}).sort_index()
L=pd.DataFrame({s:d["low"]   for s,d in F.items()}).sort_index()
# liquidity proxy: require price & volume present; drop rows all-nan
V=pd.DataFrame({s:d["volume"] for s,d in F.items()}).reindex(C.index)
dv=(C*V)  # rupee volume proxy
print(f"universe {C.shape[1]} stocks | {C.shape[0]} bars | {C.index.min().date()}..{C.index.max().date()}")

nif=pd.read_parquet(os.path.join(DDIR,"NSE_NIFTY50_INDEX_daily.parquet"))
nif["ts"]=pd.to_datetime(nif["ts"]).dt.normalize(); nif=nif.set_index("ts")["close"].reindex(C.index).ffill()

# ── features (vectorized, causal) ──────────────────────────────────────────────
d_net = C/C.shift(D_MOM)-1
d_dir = np.sign(d_net)
ema_s = C.ewm(span=20,adjust=False).mean(); ema_l=C.ewm(span=50,adjust=False).mean()
w_dir = np.sign(ema_s-ema_l)
don_hi= H.shift(1).rolling(DON).max(); don_lo=L.shift(1).rolling(DON).min()
atr20 = ((H-L)/C).rolling(DON).mean()
roll_hi=H.rolling(W_MOM).max(); roll_lo=L.rolling(W_MOM).min()
range_pos=(C-roll_lo)/(roll_hi-roll_lo)
# liquidity mask: rupee-volume in top 60% that day (avoid illiquid tails)
liq = dv.rank(axis=1,pct=True) >= 0.20

def brk_up(margin): return C > don_hi*(1+margin)
def brk_dn(margin): return C < don_lo*(1-margin)
def tight(pctl):    return atr20.rank(axis=1,pct=True) <= pctl

def fwd_rel(H_):
    f=C.shift(-H_)/C-1
    return f.sub(f.median(axis=1),axis=0)

# ── regime label (Nifty): trend via 20/50 SMA + 20d realized vol pctile ─────────
nret=nif.pct_change()
sma20=nif.rolling(20).mean(); sma50=nif.rolling(50).mean()
r20=nif/nif.shift(20)-1
def regime(dt):
    if dt not in nif.index: return "NA"
    p=nif.loc[dt]; s20=sma20.loc[dt]; s50=sma50.loc[dt]; rr=r20.loc[dt]
    if any(pd.isna(x) for x in [s20,s50,rr]): return "NA"
    if p>s20>s50 and rr>0: return "BULL"
    if p<s20<s50 and rr<0: return "BEAR"
    return "CHOP"

def basket_stats(mask_df, H_, era=None, cost=0.0, regime_filter=None):
    """Non-overlapping (step=H_) per-date breakout-basket relative return series."""
    fr=fwd_rel(H_)
    dates=C.index
    vals=[]; wins=[]; ns=[]
    i=W_MOM+5
    while i < len(dates)-H_-1:
        dt=dates[i]
        if era=="IS" and dt<IS_START: i+=H_; continue
        if era=="OOS" and dt>=IS_START: i+=H_; continue
        if regime_filter and regime(dt)!=regime_filter: i+=H_; continue
        m=mask_df.iloc[i] & liq.iloc[i]
        r=fr.iloc[i][m.values].dropna()
        if len(r)>=2:
            vals.append(r.mean()-cost); wins.append((r>0).mean()); ns.append(len(r))
        i+=H_
    if len(vals)<8: return None
    v=np.array(vals)
    t=v.mean()/(v.std(ddof=1)/np.sqrt(len(v)))
    return dict(mean=v.mean()*100, t=t, win=np.mean(wins)*100, ndates=len(v), avgn=np.mean(ns))

def show(label, s):
    if s is None: print(f"  {label:46s}  (thin)"); return
    print(f"  {label:46s} n_dates={s['ndates']:3d} avg_names={s['avgn']:4.1f}  "
          f"fwd-rel {s['mean']:+.2f}%  t={s['t']:+.2f}  win {s['win']:.1f}%")

BO = brk_up(BRK_M) & tight(CONSOL_P)     # the shipped breakout-from-consolidation
print("\n"+"="*104); print("1) NON-OVERLAPPING breakout-from-consolidation — full 4yr, then ERA split"); print("="*104)
for H_ in (5,10,20):
    print(f" horizon {H_}d:")
    show("full 4yr (2022-2026)", basket_stats(BO,H_))
    show("OOS  (pre-2024-12, NEVER in DCM window)", basket_stats(BO,H_,era="OOS"))
    show("IS   (2024-12+, the build window)", basket_stats(BO,H_,era="IS"))

print("\n"+"="*104); print("2) REGIME split (Nifty) — does it survive non-bull tape? [H=10, non-overlap]"); print("="*104)
for rg in ("BULL","CHOP","BEAR"):
    show(f"{rg}", basket_stats(BO,10,regime_filter=rg))

print("\n"+"="*104); print("3) PARAMETER sensitivity [H=10, full 4yr, non-overlap]"); print("="*104)
print(" breakout margin:")
for mg in (0.005,0.01,0.02): show(f"  margin {mg*100:.1f}%", basket_stats(brk_up(mg)&tight(CONSOL_P),10))
print(" consolidation pctl:")
for cp in (0.25,0.33,0.40): show(f"  consol<= {cp:.2f}", basket_stats(brk_up(BRK_M)&tight(cp),10))
print(" breakout WITHOUT tight base (control):")
show("  any breakout, no consol filter", basket_stats(brk_up(BRK_M),10))
show("  tight base but NO breakout (coiling)", basket_stats(tight(CONSOL_P)&~brk_up(BRK_M)&~brk_dn(BRK_M),10))

print("\n"+"="*104); print("4) NET-OF-COST [H=10, full 4yr] round-trip stock cost swept"); print("="*104)
for cst in (0.0,0.002,0.004,0.006): show(f"  cost {cst*100:.1f}%/rt", basket_stats(BO,10,cost=cst))

print("\n"+"="*104); print("5) BREAKDOWN follow-through (short thesis) — should FAIL (they bounce)"); print("="*104)
show("breakdown-from-tight-base fwd (H=10)", basket_stats(brk_dn(BRK_M)&tight(CONSOL_P),10))

print("\n"+"="*104); print("6) ALIGNMENT gate (daily-up: weekly-up vs weekly-DOWN) [H=10, non-overlap]"); print("="*104)
up_wu=(d_dir>0)&(w_dir>0); up_wd=(d_dir>0)&(w_dir<0)
for era in (None,"OOS","IS"):
    tag={None:"full",  "OOS":"OOS ","IS":"IS  "}[era]
    su=basket_stats(up_wu,10,era=era); sd=basket_stats(up_wd,10,era=era)
    if su and sd:
        print(f"  {tag}: daily-up+weekly-UP win {su['win']:.1f}% ({su['mean']:+.2f}%)   "
              f"vs daily-up+weekly-DOWN win {sd['win']:.1f}% ({sd['mean']:+.2f}%)   "
              f"spread {su['win']-sd['win']:+.1f}pp")
