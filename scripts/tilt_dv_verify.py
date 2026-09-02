import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import pandas as pd
from src.analytics.sector_forward_tilt import (get_forward_tilt, _dv_windows,
                                               _panel_lookback_cal, _load_sector_panel)
pd.set_option('display.width', 220)
D = date(2026, 9, 1)

# pre-change 1-2wk board, captured BEFORE the edit (scripts/tilt_explain.py run)
BEFORE = {
 'Diversified':(1.000,'OVERWEIGHT'), 'Textiles':(0.958,'NEUTRAL'),
 'Infrastructure':(0.917,'OVERWEIGHT'), 'Metals & Mining':(0.875,'OVERWEIGHT'),
 'Consumer Services':(0.833,'NEUTRAL'), 'Media & Entertainment':(0.792,'OVERWEIGHT'),
 'Telecom':(0.750,'OVERWEIGHT'), 'Automobile':(0.708,'NEUTRAL'), 'IT':(0.667,'NEUTRAL'),
 'Pharma & Healthcare':(0.625,'NEUTRAL'), 'Banking':(0.583,'NEUTRAL'),
 'Logistics':(0.542,'NEUTRAL'), 'Oil & Gas':(0.500,'NEUTRAL'),
 'Financial Services':(0.458,'NEUTRAL'), 'Capital Goods':(0.417,'NEUTRAL'),
 'Renewables':(0.375,'NEUTRAL'), 'Cement & Building Materials':(0.333,'NEUTRAL'),
 'Power & Utilities':(0.292,'NEUTRAL'), 'Realty':(0.250,'UNDERWEIGHT'),
 'Defence':(0.208,'WATCH'), 'Gems & Jewellery':(0.167,'UNDERWEIGHT'),
 'Consumer Durables':(0.125,'WATCH'), 'Chemicals':(0.062,'UNDERWEIGHT'),
 'FMCG':(0.062,'UNDERWEIGHT')}

print("=== REGRESSION: is the 1-2wk board byte-identical to before the change? ===")
t0=time.time(); df,_ = get_forward_tilt(D, horizon_days=10); el=time.time()-t0
bad=[]
for s,(rk,tl) in BEFORE.items():
    row = df[df['sector']==s]
    if row.empty: bad.append(f"{s}: MISSING"); continue
    r=row.iloc[0]
    if abs(float(r['rank'])-rk)>1e-3 or r['tilt']!=tl:
        bad.append(f"{s}: rank {r['rank']:.3f} vs {rk:.3f}, tilt {r['tilt']} vs {tl}")
print(f"  sectors compared: {len(BEFORE)}   mismatches: {len(bad)}   ({el:.1f}s)")
for b in bad: print("   ", b)
print("  RESULT:", "IDENTICAL — 1-2wk build untouched" if not bad else "*** DRIFTED ***")

print("\n=== windows + panel depth actually delivered per horizon ===")
rows=[]
for lbl,h in [("1-2 wk",10),("3-4 wk",20),("5-6 wk",30),("7-8 wk",40),("9-10 wk",50),("11-12 wk",60)]:
    fl,bs=_dv_windows(h); cal=_panel_lookback_cal(h)
    t0=time.time(); p=_load_sector_panel(D,100.0,cal); tl=time.time()-t0
    n=pd.to_datetime(p['trade_date']).nunique()
    rows.append(dict(horizon=lbl, flow=fl, base=bs, overlap=f"{fl/bs*100:.0f}%",
                     cal_days=cal, sessions=n, enough="YES" if n>=bs else "NO",
                     load_s=round(tl,2)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== boards render at every horizon, and the buy list moves ===")
for lbl,h in [("1-2 wk",10),("3-4 wk",20),("7-8 wk",40),("11-12 wk",60)]:
    t0=time.time(); d,reg = get_forward_tilt(D, horizon_days=h); el=time.time()-t0
    ow=list(d[d['tilt']=='OVERWEIGHT']['sector'])
    print(f"  {lbl:9s} ({el:4.1f}s) dv={reg.get('dv_flow_days')}/{reg.get('dv_base_days')}"
          f"  OW: {', '.join(ow) if ow else '(none)'}")
