"""Structural audit of the composite: are the family caps attainable, is the
vote symmetric, and how stable are the regime labels day to day?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np

pd.set_option('display.width', 240)
SP = (r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
      r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad")
from src.analytics.index_prediction import _FAMILY_CAPS, _FAMILY_CAP_DEFAULT

D = pd.read_pickle(SP + "/scored_long.pkl")

print("=== T1. ARE THE FAMILY CAPS ATTAINABLE? (observed net score vs cap) ===")
rec = []
for r in D.itertuples():
    fam = {}
    for name, cat, score in r.sig:
        fam[cat] = fam.get(cat, 0.0) + score
    for k, v in fam.items():
        rec.append((r.trade_date, r.sym, k, v))
FN = pd.DataFrame(rec, columns=['trade_date', 'sym', 'cat', 'net'])
t = FN.groupby('cat')['net'].agg(['count', 'min', 'max', 'mean',
                                  lambda s: (s > 0).mean() * 100,
                                  lambda s: (s < 0).mean() * 100])
t.columns = ['n', 'min_net', 'max_net', 'mean_net', 'pct_bull', 'pct_bear']
t['cap'] = [_FAMILY_CAPS.get(i, _FAMILY_CAP_DEFAULT) for i in t.index]
t['cap_hit_up'] = [(FN[(FN.cat == i) & (FN.net >= t.loc[i, 'cap'])].shape[0]
                    / max(t.loc[i, 'n'], 1) * 100) if t.loc[i, 'cap'] > 0 else np.nan
                   for i in t.index]
t['headroom_up'] = t['cap'] - t['max_net']
t['headroom_dn'] = t['min_net'] + t['cap']
print(t.round(2).to_string())
print("\nheadroom_up > 0  => the cap can NEVER be reached on the bullish side")
print("headroom_dn > 0  => the cap can NEVER be reached on the bearish side")

print("\n=== T2. VOTE SYMMETRY per family ===")
for cat, g in FN.groupby('cat'):
    cap = _FAMILY_CAPS.get(cat, _FAMILY_CAP_DEFAULT)
    if cap <= 0:
        continue
    print(f"{cat:20s} cap={cap:4.1f}  net range [{g['net'].min():+.1f},{g['net'].max():+.1f}]  "
          f"bull-days {(g['net']>0).mean()*100:5.1f}%  bear-days {(g['net']<0).mean()*100:5.1f}%  "
          f"mean {g['net'].mean():+.2f}")

print("\n=== T3. COMPOSITE reachable range vs the verdict thresholds (8 / 12) ===")
capsum = sum(v for v in _FAMILY_CAPS.values() if v > 0)
print(f"theoretical |composite| max from caps = {capsum:.1f}")
print(f"observed: min={D['composite'].min():.2f} max={D['composite'].max():.2f} "
      f"sd={D['composite'].std():.2f}")
for th in [8, 12]:
    print(f"  |composite| >= {th}: {(D['composite'].abs()>=th).mean()*100:5.2f}% of days "
          f"(UP {((D['composite']>=th).mean()*100):.2f}% / DOWN {((D['composite']<=-th).mean()*100):.2f}%)")

print("\n=== T4. HMM label stability (refit every day on a 90D window) ===")
def hmm_lab(sig):
    for name, cat, score in sig:
        if name.startswith("HMM Regime:"):
            return name.split(":")[1].strip().split(" ")[0]
    return None
D2 = D.sort_values('trade_date').copy()
D2['hmm'] = [hmm_lab(s) for s in D2['sig']]
for sym, g in D2.groupby('sym'):
    lab = g['hmm'].dropna()
    flips = (lab != lab.shift(1)).iloc[1:].sum()
    print(f"{sym:11s} n={len(lab):4d}  label flips={flips:4d} ({flips/max(len(lab)-1,1)*100:.1f}% of days)  "
          f"mix={lab.value_counts().to_dict()}")

print("\n=== T5. does the 'Bear' HMM label actually mark down-days? ===")
for lab, g in D2.dropna(subset=['hmm']).groupby('hmm'):
    print(f"HMM={lab:9s} n={len(g):4d}  same-day ret={g['pct_chg'].mean():+.3f}%  "
          f"next-day ret={g['f1'].mean():+.3f}%  next-day up={((g['f1']>0).mean()*100):.1f}%")

print("\n=== T6. verdict-vs-composite consistency ===")
bad = D[((D['direction'] == 'UP') & (D['composite'] < 0)) |
        ((D['direction'] == 'DOWN') & (D['composite'] > 0))]
print(f"cards printing a direction opposite their own composite: {len(bad)}/{len(D)} "
      f"({len(bad)/len(D)*100:.1f}%)")
print(bad.groupby(['sym', 'direction']).size().to_string())
print(f"of those, dte<=1 (expiry override): {(bad['dte']<=1).sum()}/{len(bad)}")
b = bad[bad['direction'].isin(['UP', 'DOWN'])]
sgn = np.where(b['direction'] == 'UP', 1, -1)
print(f"their sign-hit: {((sgn>0)==(b['f1']>0)).mean()*100:.1f}%  "
      f"vs composite-sign hit on the same days: {((np.sign(b['composite'])>0)==(b['f1']>0)).mean()*100:.1f}%")

print("\n=== T7. how often does the engine emit NO usable data? ===")
print(D['data_ok'].value_counts().to_string())
if 'liquid' in D.columns:
    print(D.groupby('sym')['liquid'].mean().mul(100).round(1).to_string())

print("\n=== T8. CONVICTION AXIS: is composite/20 ever near 1.0? ===")
from src.analytics.index_prediction import _CONVICTION_DIVISOR, _C_SIDE
D['conv'] = (D['composite'] / _CONVICTION_DIVISOR).clip(-1, 1)
print(f"divisor={_CONVICTION_DIVISOR}  C_SIDE={_C_SIDE}")
print(f"|conviction| max observed = {D['conv'].abs().max():.3f}  (1.0 = full-scale target)")
span = ((D['conv'].abs() - _C_SIDE) / (1 - _C_SIDE)).clip(lower=0)
print(f"target span max = {span.max():.3f}  -> max |target| is "
      f"{(_C_SIDE + span.max()*(1-_C_SIDE))*100:.1f}% of the 1sigma band")
tm = (D['tgt'] - D['spot']).abs() / D['em']
print(f"observed |target_move| / expected_move: max={tm.max():.3f} "
      f"mean(nonzero)={tm[tm>0].mean():.3f}  share with a nonzero target={(tm>0).mean()*100:.1f}%")

print("\n=== T9. UNIT MISMATCH: is composite/20 correlated with realised move/sigma at all? ===")
D['move_sig'] = (D['f1'] / 100 * D['spot']) / D['em']
print(f"corr(conviction, realised move/sigma) = {D['conv'].corr(D['move_sig']):+.4f}")
print(f"corr(|conviction|, |realised move|/sigma) = {D['conv'].abs().corr(D['move_sig'].abs()):+.4f}")
print("(the second is the claim that a bigger composite implies a bigger move)")
