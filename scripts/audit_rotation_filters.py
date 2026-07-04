"""
Sector-rotation FILTER AUDIT — which price-action filter axes are redundant, which
carry reliable forward edge, walk-forward causal on the full DCM panel.

Replicates the EXACT UI classifications (src/analytics/price_action.py):
  pa_class (Clean/Volatile/Choppy/Quiet), gappy flag, high-vol flag,
  breakout state (Breakout/Extended/Bounce/Coiling), mtf_align (4 quadrants).

Questions:
  1. OVERLAP — Cramér's V / conditional cross-tabs between the five axes.
  2. RELIABILITY — forward 10d relative return + per-date basket t per bucket.
  3. INCREMENTAL — does an axis add anything conditional on the validated ones?
"""
from __future__ import annotations
import sys
sys.path.insert(0, r"d:/Python Projects/Daily_Cash_Market")
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, pandas as pd
from src.data.repository import query_dataframe as q

MIN_TO = 100.0
LOOK   = 60      # pa_class window
D_MOM, DON = 10, 20
ER_TREND, ER_RANGE = 0.25, 0.12
HIVOL, GAPPY_F, GAP_PCT = 4.5, 0.35, 1.0
CONSOL_PCTL = 0.33
HF = 10          # forward horizon (days)
STEP = 5

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
C = P.pivot_table("c","trade_date","symbol"); O = P.pivot_table("o","trade_date","symbol")
H = P.pivot_table("h","trade_date","symbol"); L = P.pivot_table("l","trade_date","symbol")
dates = C.index
print(f"panel: {C.shape[0]} days x {C.shape[1]} symbols")

rng_ = (H - L)
body_pct = ((C - O).abs() / rng_.replace(0, np.nan))
avg_body = body_pct.rolling(LOOK, min_periods=40).mean()
gap = ((O - C.shift(1)).abs() / C.shift(1) * 100)
gap_freq = (gap > GAP_PCT).rolling(LOOK, min_periods=40).mean()
atr_pct = (rng_ / C * 100).rolling(LOOK, min_periods=40).mean()
er60 = (C - C.shift(LOOK)).abs() / C.diff().abs().rolling(LOOK).sum().replace(0, np.nan)

# MTF legs (identical formulas to _mtf_fields)
d_dir = np.sign(C / C.shift(D_MOM) - 1.0)
ema_s = C.ewm(span=20, adjust=False).mean(); ema_l = C.ewm(span=50, adjust=False).mean()
w_dir = np.sign(ema_s - ema_l)
don_hi = H.shift(1).rolling(DON).max(); don_lo = L.shift(1).rolling(DON).min()
brk_up = C > don_hi * 1.01; brk_dn = C < don_lo * 0.99
atr20 = (rng_ / C * 100).rolling(DON, min_periods=15).mean()

f = C.shift(-HF) / C - 1.0
fwd = f.sub(f.median(axis=1), axis=0)
# forward realized vol (risk check for the flag filters)
fvol = C.pct_change().shift(-HF).rolling(HF).std().shift(HF - 1) * np.sqrt(252) * 100
fvol = (C.pct_change().rolling(HF).std().shift(-HF) * np.sqrt(252) * 100)  # vol over t+1..t+HF
fgap = (gap > GAP_PCT).rolling(HF).mean().shift(-HF)                        # fwd gap freq

WARM = 75
rows = []
for i in range(WARM, len(dates) - HF - 1, STEP):
    d = dates[i]
    rec = pd.DataFrame({
        "er": er60.iloc[i], "body": avg_body.iloc[i], "atr": atr_pct.iloc[i],
        "gapf": gap_freq.iloc[i], "atr20": atr20.iloc[i],
        "d_dir": d_dir.iloc[i], "w_dir": w_dir.iloc[i],
        "brk_up": brk_up.iloc[i], "brk_dn": brk_dn.iloc[i],
        "f10": fwd.iloc[i], "fvol": fvol.iloc[i], "fgap": fgap.iloc[i],
    }).dropna(subset=["er","body","atr","atr20","f10"])
    if len(rec) < 100: continue
    med = rec["body"].median()
    trending = rec["er"] >= ER_TREND; ranging = rec["er"] <= ER_RANGE
    decisive = rec["body"] >= med; hv = rec["atr"] >= HIVOL
    cls = np.where(trending, np.where(hv | ~decisive, "Volatile", "Clean"),
                   np.where(decisive, "Choppy", "Quiet"))
    rec["pa_class"] = cls
    rec["gappy"] = rec["gapf"] >= GAPPY_F
    rec["hivol"] = hv
    tight = rec["atr20"] <= rec["atr20"].quantile(CONSOL_PCTL)
    rec["brk"] = np.where(rec["brk_up"], np.where(tight, "Breakout", "Extended"),
                  np.where(rec["brk_dn"], "Bounce", np.where(tight, "Coiling", "None")))
    rec["align"] = np.select(
        [(rec.d_dir>0)&(rec.w_dir>0), (rec.d_dir>0)&(rec.w_dir<0),
         (rec.d_dir<0)&(rec.w_dir>0), (rec.d_dir<0)&(rec.w_dir<0)],
        ["ConfirmUp","FalsePop","Pullback","DownAlign"], "Neutral")
    rec["date"] = d
    rows.append(rec)
S = pd.concat(rows, ignore_index=True)
print(f"pooled obs: {len(S):,}  dates: {S['date'].nunique()}\n")

def cramers_v(a, b):
    ct = pd.crosstab(a, b)
    chi2 = (((ct - np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum())**2
             / (np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum())).values.sum())
    n = ct.values.sum(); r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def basket_t(mask, col="f10"):
    """per-date basket mean -> t over dates (the honest anchor)"""
    g = S[mask].groupby("date")[col].mean().dropna()
    if len(g) < 8: return len(S[mask]), np.nan, np.nan, np.nan
    t = g.mean() / (g.std() / np.sqrt(len(g)))
    return int(mask.sum()), g.mean()*100, t, (S[mask][col] > 0).mean()*100

print("="*100); print("1) OVERLAP between filter axes (Cramér's V; 0=independent, 1=identical)"); print("="*100)
for a, b in [("pa_class","brk"),("pa_class","align"),("brk","align"),
             ("pa_class","hivol"),("pa_class","gappy"),("gappy","hivol")]:
    print(f"  {a:9s} x {b:9s}  V = {cramers_v(S[a], S[b]):.3f}")

print("\n  pa_class x brk conditional rows (P(brk-state | pa_class), %):")
print((pd.crosstab(S["pa_class"], S["brk"], normalize="index")*100).round(1).to_string())
print("\n  pa_class x align conditional rows (%):")
print((pd.crosstab(S["pa_class"], S["align"], normalize="index")*100).round(1).to_string())
print("\n  pa_class x hivol conditional rows (%):")
print((pd.crosstab(S["pa_class"], S["hivol"], normalize="index")*100).round(1).to_string())

# Quiet Range vs Coiling set overlap
qm = S["pa_class"]=="Quiet"; cm = S["brk"]=="Coiling"
print(f"\n  Quiet-Range vs Coiling: Jaccard {((qm&cm).sum()/ (qm|cm).sum()):.2f}   "
      f"P(Coil|Quiet) {cm[qm].mean()*100:.0f}%   P(Quiet|Coil) {qm[cm].mean()*100:.0f}%")

print("\n"+"="*100); print(f"2) RELIABILITY — fwd {HF}d RELATIVE return per bucket (per-date basket t)"); print("="*100)
def show(mask, label):
    n, m, t, w = basket_t(mask)
    print(f"  {label:34s} n={n:6d}  f10 {m:+.2f}%  t {t:+.1f}  win {w:4.1f}%")
print("  -- PA character classes --")
for c in ["Clean","Volatile","Choppy","Quiet"]: show(S["pa_class"]==c, f"pa_class = {c}")
print("  -- MTF setup states --")
for c in ["Breakout","Extended","Bounce","Coiling"]: show(S["brk"]==c, f"brk = {c}")
print("  -- Alignment quadrants --")
for c in ["ConfirmUp","FalsePop","Pullback","DownAlign"]: show(S["align"]==c, f"align = {c}")
print("  -- Risk flags (return) --")
show(S["gappy"], "gappy = True"); show(S["hivol"], "hivol = True")

print("\n"+"="*100); print("3) RISK FLAGS — do they predict forward RISK (their actual claim)?"); print("="*100)
for flag in ["gappy","hivol"]:
    a = S[S[flag]]; b = S[~S[flag]]
    print(f"  {flag:6s} True : fwd vol {a['fvol'].mean():5.1f}%  fwd gap-freq {a['fgap'].mean()*100:4.1f}%   (n {len(a):,})")
    print(f"  {flag:6s} False: fwd vol {b['fvol'].mean():5.1f}%  fwd gap-freq {b['fgap'].mean()*100:4.1f}%   (n {len(b):,})")

print("\n"+"="*100); print("4) INCREMENTAL VALUE — does pa_class add anything on top of the validated axes?"); print("="*100)
cu = S["align"]=="ConfirmUp"
print("  within align=ConfirmUp:")
for c in ["Clean","Volatile","Choppy","Quiet"]: show(cu & (S["pa_class"]==c), f"  + pa_class {c}")
bo = S["brk"]=="Breakout"
print("  within brk=Breakout (the validated edge):")
for c in ["Clean","Volatile","Choppy","Quiet"]: show(bo & (S["pa_class"]==c), f"  + pa_class {c}")
print("  within brk=Breakout, alignment gate:")
show(bo & (S["w_dir"]>0), "  + weekly-up")
show(bo & (S["w_dir"]<=0), "  + weekly-down/flat")
print("  ER continuous IC (per-date Spearman) as reference:")
ics = [g["er"].rank().corr(g["f10"].rank()) for _, g in S.groupby("date") if len(g) >= 20]
ics = np.array([x for x in ics if np.isfinite(x)])
print(f"    er60 vs f10: IC {ics.mean():+.3f}  t {ics.mean()/(ics.std()/np.sqrt(len(ics))):+.1f}")

print("\n"+"="*100); print("5) HALF-SPLIT stability of any bucket that looks alive"); print("="*100)
mid = S["date"].median()
for lab, m in [("Breakout", bo), ("Clean", S["pa_class"]=="Clean"),
               ("ConfirmUp", cu), ("FalsePop", S["align"]=="FalsePop")]:
    for hl, hm in [("H1", S["date"]<=mid), ("H2", S["date"]>mid)]:
        n, mn, t, w = basket_t(m & hm)
        print(f"  {lab:10s} {hl}: n={n:6d}  f10 {mn:+.2f}%  t {t:+.1f}  win {w:4.1f}%")
