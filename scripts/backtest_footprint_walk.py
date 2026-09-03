import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
pd.set_option('display.width', 260)
SP = (r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
      r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad")
D = pd.read_pickle(SP + '/fp_panel.pkl')

def nw_t(s, lag):
    x = s.dropna().values.astype(float); n = len(x)
    if n < 25: return float('nan')
    e = x - x.mean(); v = (e @ e) / n
    for L in range(1, lag+1): v += 2*(1-L/(lag+1))*((e[L:] @ e[:-L])/n)
    return x.mean()/math.sqrt(v/n) if v > 0 else float('nan')

print("=== H. RE-CENTRED ON THE PANEL'S OWN BASELINE ===")
print("  the scored panel is not a random slice of the F&O universe, so its own")
print("  mean is the fair zero, not the universe mean.")
for h in (1,3,5,10):
    base = D[f'f{h}'].mean()
    print(f"\n  horizon {h}d — panel baseline {base:+.4f}pp")
    for lab in ['AGREE BULL','AGREE BEAR','COLLIDE']:
        s = D[D['agree']==lab]
        raw = s[f'f{h}'].mean(); adj = raw - base
        # sign the adjustment so + always = 'the call was right'
        sgn = 1 if lab=='AGREE BULL' else (-1 if lab=='AGREE BEAR' else np.nan)
        d = (s[f'f{h}'] - base)
        t = nw_t(d.groupby(s['trade_date']).mean(), h)
        corr = f"{adj*sgn*100:+6.1f}" if sgn==sgn else "     -"
        print(f"    {lab:11s} n={len(s):6d} raw={raw:+.4f} adj={adj:+.4f}pp "
              f"t={t:+5.2f}  in-called-direction={corr} bps")

print("\n=== I. SIGN-TEST: did the call at least get the DIRECTION right more often? ===")
for h in (1,3,5,10):
    base = D[f'f{h}'].median()
    a = D[D['agree']!='COLLIDE']
    right = ((a[f'f{h}'] - base) * a['pred'] > 0).mean()*100
    n = len(a); se = math.sqrt(0.25/n)*100
    print(f"  {h:2d}d: called direction right {right:.2f}% of {n:,} calls  "
          f"(coin flip 50.00%, 1 SE = {se:.2f}pp)  z={(right-50)/se:+.2f}")

print("\n=== J. WALKTHROUGH — 12 Aug 2026, exactly what you asked ===")
for day in ['2026-08-12','2026-08-03']:
    d = D[D['trade_date']==day].copy()
    if d.empty:
        print(f"\n  {day}: no session / no events"); continue
    d = d.sort_values('dom_add_cr', ascending=False)
    print(f"\n  --- {day}: {len(d)} stocks carried BOTH a futures label and an option event ---")
    print(f"      agreement mix: " + ", ".join(f"{k}={v}" for k,v in d['agree'].value_counts().items()))
    show = d.head(14)[['symbol','fut_action','dom_action','dom_money','dom_add_cr',
                       'opt_net','agree','ret_pct','f1','f3','f5']]
    show = show.rename(columns={'dom_add_cr':'add_Cr','ret_pct':'today%',
                                'f1':'fwd1d','f3':'fwd3d','f5':'fwd5d'})
    print(show.round(2).to_string(index=False))
    for lab in ['AGREE BULL','AGREE BEAR','COLLIDE']:
        s = d[d['agree']==lab]
        if s.empty: continue
        sgn = 1 if lab=='AGREE BULL' else (-1 if lab=='AGREE BEAR' else 0)
        ok = ((s['f3']*sgn) > 0).mean()*100 if sgn else np.nan
        print(f"      {lab:11s} n={len(s):3d}  mean fwd3d excess {s['f3'].mean():+.2f}pp"
              + (f"  went the called way {ok:.0f}% of the time" if sgn else ""))

print("\n=== K. IS THE WHOLE THING JUST A SIZE/LIQUIDITY SORT? ===")
D['add_q'] = pd.qcut(D['dom_add_cr'], 5, labels=False, duplicates='drop')
for k, s in D.groupby('add_q'):
    a = s[s['agree']!='COLLIDE']
    v = a['f3']*a['pred']
    print(f"  dom_add_cr Q{int(k)+1} (median {s['dom_add_cr'].median():5.1f} Cr): "
          f"n={len(a):5d} signed3d={v.mean():+.4f}pp t={nw_t(v.groupby(a['trade_date']).mean(),3):+5.2f}")

print("\n=== L. DOES A BIGGER FOOTPRINT EVER HELP? top-N by money added each day ===")
for N in (5, 10, 25):
    r = D.sort_values('dom_add_cr', ascending=False).groupby('trade_date').head(N)
    a = r[r['agree']!='COLLIDE']
    v = a['f3']*a['pred']
    print(f"  top-{N:2d}/day: n={len(a):5d} signed3d={v.mean():+.4f}pp "
          f"t={nw_t(v.groupby(a['trade_date']).mean(),3):+5.2f} hit={(v>0).mean()*100:.2f}%")
