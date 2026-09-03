"""Full-F&O-era REPLAY of the Index Prediction engine (homogeneous current code)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.analytics.index_prediction import get_index_predictions, _build_market_context
from src.data.repository import query_dataframe

OUT = r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad/replay_long.pkl"
d = query_dataframe("SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE symbol='NIFTY' ORDER BY trade_date")
dates = sorted(pd.to_datetime(d['trade_date']).dt.date.tolist())
dates = [x for x in dates if x >= __import__('datetime').date(2024,9,2)]
print("replay", dates[0], "->", dates[-1], len(dates), flush=True)

rows=[]; t0=time.time()
for i,td in enumerate(dates):
    try:
        ctx=_build_market_context(td)
        for p in get_index_predictions(td, persist=False, market_ctx=ctx):
            rows.append(dict(trade_date=td, sym=p.fno_symbol, direction=p.direction,
                confidence=p.confidence, composite=p.composite_score, spot=p.spot_close,
                dte=p.days_to_expiry, em=p.expected_move_pts, rlo=p.range_low, rhi=p.range_high,
                tgt=p.target_close, liquid=p.opt_chain_liquid, data_ok=p.data_available,
                pcr=p.pcr, fut_oi=p.fut_oi, near_expiry=p.near_expiry, n_sig=len(p.signals),
                sig=[(s.name,s.category,s.score) for s in p.signals]))
    except Exception as exc:
        rows.append(dict(trade_date=td, sym='ERROR', direction=str(exc)[:120]))
    if i%25==0:
        print(f"  {i+1}/{len(dates)} {td} {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_pickle(OUT)
pd.DataFrame(rows).to_pickle(OUT)
print("DONE", len(rows), f"{time.time()-t0:.0f}s", flush=True)
