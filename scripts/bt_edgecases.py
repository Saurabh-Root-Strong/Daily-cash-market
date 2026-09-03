"""Operational failure-mode probes for the Index Prediction engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import pandas as pd
from src.analytics.index_prediction import (
    get_index_prediction_for, _build_market_context, _get_two_fno_dates)
from src.data.repository import query_dataframe

pd.set_option('display.width', 240)


def sess(table, where=""):
    d = query_dataframe(f"SELECT DISTINCT trade_date FROM {table} "
                        f"WHERE trade_date>='2024-09-02' {where}")
    return set(pd.to_datetime(d['trade_date']).dt.date)


idx = sess("index_data", "AND index_name='Nifty 50'")
fno = sess("fno_bhavcopy", "AND symbol='NIFTY'")
vix = sess("index_data", "AND index_name='India VIX'")
print("=== E0. SESSION COVERAGE ===")
print("index_data but NO fno_bhavcopy:", sorted(idx - fno))
print("fno_bhavcopy but NO index_data:", sorted(fno - idx))
print("index_data but NO India VIX   :", sorted(idx - vix))

print("\n=== E1. STALE-F&O HOLE: does the engine flag that it used an older chain? ===")
last = max(idx)
probe = date(last.year, last.month, last.day)
# a calendar day that is NOT a session (the day after the last session)
from datetime import timedelta
nonsess = last + timedelta(days=1)
while nonsess in idx:
    nonsess += timedelta(days=1)
td, pd_ = _get_two_fno_dates("NIFTY", nonsess)
print(f"asked for F&O as of {nonsess} (not a session) -> engine loaded {td} / prev {pd_}")
ctx = _build_market_context(nonsess)
p = get_index_prediction_for(nonsess, "NIFTY", persist=False, market_ctx=ctx)
real = get_index_prediction_for(last, "NIFTY", persist=False,
                                market_ctx=_build_market_context(last))
print(f"  stamped as_of={p.as_of_date}  data_available={p.data_available}  note='{p.note}'")
print(f"  spot={p.spot_close} composite={p.composite_score} dir={p.direction} "
      f"conf={p.confidence} dte={p.days_to_expiry}")
print(f"  real {last}: spot={real.spot_close} composite={real.composite_score} "
      f"dir={real.direction}")
same = (p.spot_close == real.spot_close
        and abs(p.composite_score - real.composite_score) < 1e-9)
print(f"  => silently reused the previous session, unflagged? {same}")

print("\n=== E2. FUTURE date with no data at all ===")
fut = last + timedelta(days=45)
p2 = get_index_prediction_for(fut, "NIFTY", persist=False,
                              market_ctx=_build_market_context(fut))
print(f"  as_of={p2.as_of_date} data_available={p2.data_available} spot={p2.spot_close} "
      f"dir={p2.direction} composite={p2.composite_score} note='{p2.note}'")

print("\n=== E3. EXPIRY ROLLS in the last 3 months (near option expiry changed) ===")
d = query_dataframe("""
    SELECT trade_date, symbol, MIN(expiry_date) near_exp
    FROM fno_bhavcopy
    WHERE instrument='OPTIDX' AND expiry_date>trade_date AND trade_date>='2026-06-01'
    GROUP BY 1,2 ORDER BY 2,1""")
d['trade_date'] = pd.to_datetime(d['trade_date']).dt.date
d['near_exp'] = pd.to_datetime(d['near_exp']).dt.date
d['rolled'] = d.groupby('symbol')['near_exp'].transform(lambda s: s != s.shift(1))
print(d.groupby('symbol')['rolled'].sum().to_string())

print("\n=== E4. does yesterday's chain contain today's near expiry? (OI-diff validity) ===")
bad = 0
tot = 0
for symbol in ["NIFTY", "BANKNIFTY"]:
    dd = d[d.symbol == symbol].sort_values('trade_date')
    for prev_row, row in zip(dd.itertuples(), list(dd.itertuples())[1:]):
        tot += 1
        q = query_dataframe("""SELECT count(*) n FROM fno_bhavcopy
            WHERE symbol=? AND trade_date=? AND instrument='OPTIDX' AND expiry_date=?""",
                            [symbol, prev_row.trade_date, row.near_exp])
        if int(q['n'].iloc[0]) == 0:
            bad += 1
            print(f"  {symbol} {row.trade_date}: prior session has NO rows for the "
                  f"near expiry {row.near_exp} -> OI diff suppressed")
print(f"  checked {tot} session pairs, {bad} with no prior-session chain")
