import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import numpy as np, pandas as pd
from src.analytics.sector_forward_tilt import (get_forward_tilt, _OW_RANK, _UW_RANK,
    _W_RS2, _W_RS1, _W_DV5, _WATCH_BREADTH, _WATCH_RS_MAX, _MIN_LIQ_NAMES)
pd.set_option('display.width', 260)

df, reg = get_forward_tilt(date(2026,9,1), horizon_days=10)
df = df.sort_values('rank', ascending=False)
show = df[['sector','rs_2w','rs_1w','dv5d','score','rank','accum_breadth','n_liq',
           'thin','persistence','revert','tilt','days_in_tilt']]
print("=== LIVE BOARD 01 Sep 2026, horizon 1-2wk (h=10) ===")
print(show.round(3).to_string(index=False))
print(f"\nthresholds: OVERWEIGHT rank>={_OW_RANK}  UNDERWEIGHT rank<={_UW_RANK}")
print(f"WATCH if accum_breadth>={_WATCH_BREADTH} AND rank<={_WATCH_RS_MAX}")
print(f"weights: rs_long {_W_RS2} / rs_short {_W_RS1} / dv5d {_W_DV5}  (ranks, not raw)")
print(f"thin if n_liq < {_MIN_LIQ_NAMES}")
print(f"\nregime: verdict={reg.get('verdict')} size_hint={reg.get('size_hint')} "
      f"dispersion={reg.get('dispersion'):.2f} conf_mult={reg.get('confidence_mult')}")

print("\n=== WHAT THE RAW RANK WOULD HAVE SAID, BEFORE THE GATES ===")
d = df.copy()
d['raw_tilt'] = np.where(d['rank']>=_OW_RANK,'OVERWEIGHT',
                 np.where(d['rank']<=_UW_RANK,'UNDERWEIGHT','NEUTRAL'))
ch = d[d['raw_tilt']!=d['tilt']][['sector','rank','accum_breadth','n_liq','thin',
                                  'persistence','revert','raw_tilt','tilt']]
print(ch.round(3).to_string(index=False) if len(ch) else "  (no gate fired today)")

print("\n=== HOW OFTEN EACH GATE FIRES (sampled sessions 2024-2026) ===")
import random
dts = pd.to_datetime(pd.read_pickle(
    r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
    r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad/fp_cash.pkl")['trade_date']).dt.date.unique()
dts = sorted(dts)[::13][-30:]
tot = dict(rows=0, watch=0, thin_demote=0, revert_demote=0, ow=0, uw=0, neu=0)
for dd in dts:
    try:
        f,_ = get_forward_tilt(dd, horizon_days=10)
    except Exception:
        continue
    if f.empty: continue
    raw = np.where(f['rank']>=_OW_RANK,'OVERWEIGHT',
           np.where(f['rank']<=_UW_RANK,'UNDERWEIGHT','NEUTRAL'))
    tot['rows'] += len(f)
    tot['watch'] += int((f['tilt']=='WATCH').sum())
    tot['thin_demote'] += int(((raw=='OVERWEIGHT') & f['thin'].values).sum())
    tot['revert_demote'] += int(((raw=='OVERWEIGHT') & f['revert'].values &
                                 ~f['thin'].values).sum())
    tot['ow'] += int((f['tilt']=='OVERWEIGHT').sum())
    tot['uw'] += int((f['tilt']=='UNDERWEIGHT').sum())
    tot['neu'] += int((f['tilt']=='NEUTRAL').sum())
n = tot['rows']
print(f"  {len(dts)} sessions, {n} sector-rows")
for k in ('ow','uw','neu','watch','thin_demote','revert_demote'):
    print(f"    {k:14s} {tot[k]:5d}  ({tot[k]/n*100:5.1f}% of rows)")
