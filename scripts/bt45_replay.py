"""45-session REPLAY backtest: recompute today's engine as-of each date, score vs realized."""
import sys, os, time, math, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import pandas as pd, numpy as np
from src.analytics.index_prediction import get_index_predictions, _build_market_context
from src.data.repository import query_dataframe

NAMES = {"NIFTY":"Nifty 50","BANKNIFTY":"Nifty Bank","FINNIFTY":"Nifty Financial Services","MIDCPNIFTY":"Nifty Midcap Select"}

dts = query_dataframe("SELECT DISTINCT trade_date FROM index_data WHERE index_name='Nifty 50' AND trade_date<='2026-08-28' ORDER BY trade_date DESC LIMIT 60")
dates = sorted(pd.to_datetime(dts['trade_date']).dt.date.tolist())
window = dates[-46:-1]     # 45 sessions with a next-day outcome available
print("replay window", window[0], "..", window[-1], len(window))

fwd = {}
for s,n in NAMES.items():
    d = query_dataframe("SELECT trade_date, close_val, pct_chg FROM index_data WHERE index_name=? AND trade_date>=? ORDER BY trade_date",[n, dates[-60]])
    d['trade_date']=pd.to_datetime(d['trade_date']).dt.date
    fwd[s]=d.set_index('trade_date')

rows=[]
t0=time.time()
for i,td in enumerate(window):
    ctx = _build_market_context(td)
    preds = get_index_predictions(td, persist=False, market_ctx=ctx)
    for p in preds:
        h = fwd[p.fno_symbol]
        nxt = h[h.index > td]
        if nxt.empty: continue
        nd = nxt.index[0]
        rows.append(dict(trade_date=td, next_date=nd, gap_days=(nd-td).days, sym=p.fno_symbol,
            direction=p.direction, confidence=p.confidence, composite=p.composite_score,
            spot=p.spot_close, dte=p.days_to_expiry,
            em=p.expected_move_pts, rlo=p.range_low, rhi=p.range_high, tgt=p.target_close,
            side_band=p.sideways_band_pts,
            bo_up=p.breakout_up_start, bo_dn=p.breakout_dn_end,
            wk_lo=p.wk_range_low, wk_hi=p.wk_range_high, wk_exp=p.wk_expiry,
            actual_ret=float(nxt['pct_chg'].iloc[0]), next_close=float(nxt['close_val'].iloc[0]),
            n_sig=len(p.signals),
            sig=[(s.name,s.category,s.score,s.direction) for s in p.signals]))
    if i%10==0: print(f"  {i+1}/{len(window)} {td} {time.time()-t0:.0f}s", flush=True)
df=pd.DataFrame(rows)
df.to_pickle(r'C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad/replay45.pkl')
print("done", len(df), f"{time.time()-t0:.0f}s")
