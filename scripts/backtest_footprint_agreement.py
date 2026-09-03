"""Operator Footprint — does FUTURES x OPTIONS AGREEMENT predict the stock?

The shipped backtest (backtest_operator_footprint.py) tested the INGREDIENTS
separately and found no edge. This tests the thing the tab actually shows and
the user actually reads: what happens when the futures label and the options
label AGREE, versus when they COLLIDE.

    futures LONG BUILDUP + options CALL BUYING / PUT WRITING  -> AGREE BULL
    futures SHORT BUILDUP + options CALL WRITING / PUT BUYING -> AGREE BEAR
    opposite                                                  -> COLLIDE

Split every result by moneyness (deep ITM / ITM / ATM / OTM / deep OTM), which
is the second half of the question: does size showing up ITM behave differently
from size showing up OTM?

METHOD / GUARDS
  - Labels replicate operator_footprint._classify exactly: OI direction x the
    CONTRACT'S OWN price change, same-expiry matched vs the prior session.
  - Near expiry only (min expiry > trade_date), futures settlement days dropped.
  - Target = forward CASH return EXCESS over the point-in-time F&O universe mean,
    so a market-wide move cannot masquerade as a stock signal.
  - Corporate actions: |daily raw return| >= 40% dropped (NSE price bands make a
    genuine cash move that large impossible).
  - Inference DATE-CLUSTERED with Newey-West at lag = horizon; forward windows
    overlap and same-date stocks are correlated, so a per-row t is meaningless.
  - A RANDOM control basket of the same size runs beside every cell. If it does
    not sit at ~0 the benchmark is broken, not the signal.
  - THE CONTROL THAT MATTERS: the buy/write label is ~85-90% the day's price
    direction (module docstring, measured). So every cell is re-run conditioned
    on the day's own return, and the incremental value is regressed out.
"""
from __future__ import annotations
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, duckdb

pd.set_option('display.width', 260)
SP = (r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/"
      r"31f747ed-99c3-493a-86b8-24cdb5c1d337/scratchpad")
MIN_ADD_CR = 15.0        # operator_footprint._MIN_ADD_CR (hard floor, no lookahead)
MIN_NOTIONAL_CR = 1.0    # operator_footprint._MIN_NOTIONAL_CR
con = duckdb.connect('data/market_data.duckdb', read_only=True)

# ── 1. cash panel: spot, forward returns, point-in-time F&O universe ──────────
print("building cash panel ...", flush=True)
cash = con.execute("""
    WITH fno_syms AS (
        SELECT DISTINCT trade_date, symbol FROM fno_bhavcopy WHERE instrument='FUTSTK'
    )
    SELECT d.trade_date, d.symbol, d.close_price AS spot, d.prev_close,
           (d.close_price - d.prev_close)/NULLIF(d.prev_close,0)*100 AS ret_pct
    FROM daily_data d
    INNER JOIN fno_syms f ON f.symbol=d.symbol AND f.trade_date=d.trade_date
    WHERE d.series IN ('EQ','SM','ST') AND d.close_price>0 AND d.prev_close>0
""").df()
cash['trade_date'] = pd.to_datetime(cash['trade_date'])
cash = cash[cash['ret_pct'].abs() < 40]                      # corporate-action cut
px = cash.pivot_table('spot', 'trade_date', 'symbol').sort_index()
lg = np.log1p(px.pct_change())
FWD = {}
for h in (1, 3, 5, 10):
    f = (np.expm1(lg.iloc[::-1].rolling(h).sum().iloc[::-1].shift(-1)) * 100.0)
    FWD[h] = f.sub(f.mean(axis=1), axis=0)                   # excess vs universe mean
print(f"  cash panel {px.shape[0]} sessions x {px.shape[1]} symbols")

# ── 2. futures label per symbol-day (near month, same-expiry matched) ─────────
print("building futures labels ...", flush=True)
fut = con.execute("""
    WITH nearf AS (
        SELECT trade_date, symbol, MIN(expiry_date) AS exp
        FROM fno_bhavcopy WHERE instrument='FUTSTK' AND expiry_date > trade_date
          AND open_interest > 0 GROUP BY 1,2
    ),
    f AS (
        SELECT b.trade_date, b.symbol, b.expiry_date, b.close_price, b.open_interest,
               b.chg_in_oi
        FROM fno_bhavcopy b INNER JOIN nearf n
          ON n.symbol=b.symbol AND n.trade_date=b.trade_date AND n.exp=b.expiry_date
        WHERE b.instrument='FUTSTK'
    ),
    settle AS (   -- any FUTSTK contract expiring today = settlement session
        SELECT DISTINCT trade_date, symbol FROM fno_bhavcopy
        WHERE instrument='FUTSTK' AND expiry_date = trade_date
    )
    SELECT f.trade_date, f.symbol, f.expiry_date, f.open_interest, f.chg_in_oi,
           f.close_price,
           LAG(f.close_price)     OVER w AS prev_px,
           LAG(f.open_interest)   OVER w AS prev_oi,
           LAG(f.trade_date)      OVER w AS prev_dt,
           (s.symbol IS NOT NULL) AS is_settle
    FROM f LEFT JOIN settle s ON s.symbol=f.symbol AND s.trade_date=f.trade_date
    WINDOW w AS (PARTITION BY f.symbol, f.expiry_date ORDER BY f.trade_date)
""").df()
fut['trade_date'] = pd.to_datetime(fut['trade_date'])
fut = fut[(~fut['is_settle']) & fut['prev_px'].notna() & fut['prev_oi'].notna()].copy()
fut['f_oi_chg'] = fut['open_interest'] - fut['prev_oi']
fut['f_px_chg'] = fut['close_price'] - fut['prev_px']
fut = fut[fut['f_oi_chg'].abs() > 0]
_o = np.where(fut['f_oi_chg'] > 0, 'up', 'down')
_p = np.where(fut['f_px_chg'] >= 0, 'up', 'down')
fut['fut_action'] = np.select(
    [(_o == 'up') & (_p == 'up'), (_o == 'up') & (_p == 'down'),
     (_o == 'down') & (_p == 'up'), (_o == 'down') & (_p == 'down')],
    ['LONG BUILDUP', 'SHORT BUILDUP', 'SHORT COVERING', 'LONG UNWINDING'], default='')
fut['fut_lean'] = fut['fut_action'].map(
    {'LONG BUILDUP': 1, 'SHORT COVERING': 1, 'SHORT BUILDUP': -1, 'LONG UNWINDING': -1})
fut['fut_oi_pct'] = fut['f_oi_chg'] / fut['prev_oi'].clip(lower=1) * 100
print(f"  futures symbol-days {len(fut):,}")

# ── 3. option strike events: label + moneyness + money added ──────────────────
print("building option events (16M rows, this is the slow one) ...", flush=True)
opt = con.execute(f"""
    WITH nearo AS (
        SELECT trade_date, symbol, MIN(expiry_date) AS exp
        FROM fno_bhavcopy WHERE instrument='OPTSTK' AND expiry_date > trade_date
          AND open_interest > 0 GROUP BY 1,2
    ),
    o AS (
        SELECT b.trade_date, b.symbol, b.expiry_date, b.option_type, b.strike_price,
               b.close_price, b.open_interest, b.chg_in_oi,
               LAG(b.close_price) OVER w AS prev_px
        FROM fno_bhavcopy b INNER JOIN nearo n
          ON n.symbol=b.symbol AND n.trade_date=b.trade_date AND n.exp=b.expiry_date
        WHERE b.instrument='OPTSTK' AND b.open_interest > 0
        WINDOW w AS (PARTITION BY b.symbol, b.expiry_date, b.option_type,
                                  b.strike_price ORDER BY b.trade_date)
    )
    SELECT o.trade_date, o.symbol, o.option_type, o.strike_price,
           o.open_interest, o.chg_in_oi, o.close_price, o.prev_px, c.spot,
           GREATEST(o.chg_in_oi,0) * c.spot / 1e7 AS add_cr,
           o.open_interest * c.spot / 1e7        AS notional_cr,
           CASE
             WHEN c.spot <= 0 THEN 'n/a'
             WHEN (CASE WHEN o.option_type='PE' THEN -(o.strike_price/c.spot-1.0)
                        ELSE (o.strike_price/c.spot-1.0) END) <= -0.10 THEN 'deep ITM'
             WHEN (CASE WHEN o.option_type='PE' THEN -(o.strike_price/c.spot-1.0)
                        ELSE (o.strike_price/c.spot-1.0) END) <= -0.02 THEN 'ITM'
             WHEN (CASE WHEN o.option_type='PE' THEN -(o.strike_price/c.spot-1.0)
                        ELSE (o.strike_price/c.spot-1.0) END) <   0.02 THEN 'ATM'
             WHEN (CASE WHEN o.option_type='PE' THEN -(o.strike_price/c.spot-1.0)
                        ELSE (o.strike_price/c.spot-1.0) END) <   0.10 THEN 'OTM'
             ELSE 'deep OTM' END AS moneyness
    FROM o INNER JOIN (
        SELECT trade_date, symbol, close_price AS spot FROM daily_data
        WHERE series IN ('EQ','SM','ST') AND close_price > 0
    ) c ON c.symbol=o.symbol AND c.trade_date=o.trade_date
    WHERE o.prev_px IS NOT NULL
      AND GREATEST(o.chg_in_oi,0) * c.spot / 1e7 >= {MIN_ADD_CR}
      AND o.open_interest * c.spot / 1e7 >= {MIN_NOTIONAL_CR}
""").df()
opt['trade_date'] = pd.to_datetime(opt['trade_date'])
opt['prem_chg'] = opt['close_price'] - opt['prev_px']
_oi = np.where(opt['chg_in_oi'] > 0, 'up', 'down')
_pr = np.where(opt['prem_chg'] >= 0, 'up', 'down')
key = opt['option_type'] + '|' + _oi + '|' + _pr
LEAN = {'CE|up|up': 1, 'CE|up|down': -1, 'CE|down|up': 1, 'CE|down|down': -1,
        'PE|up|up': -1, 'PE|up|down': 1, 'PE|down|up': -1, 'PE|down|down': 1}
NAME = {'CE|up|up': 'CALL BUYING', 'CE|up|down': 'CALL WRITING',
        'CE|down|up': 'CALL WRITERS COVERING', 'CE|down|down': 'CALL LONGS EXITING',
        'PE|up|up': 'PUT BUYING', 'PE|up|down': 'PUT WRITING',
        'PE|down|up': 'PUT WRITERS COVERING', 'PE|down|down': 'PUT LONGS EXITING'}
opt['opt_lean'] = key.map(LEAN)
opt['opt_action'] = key.map(NAME)
opt = opt[opt['opt_lean'].notna() & (opt['moneyness'] != 'n/a')].copy()
print(f"  option strike-events {len(opt):,} over {opt['trade_date'].nunique()} sessions, "
      f"{opt['symbol'].nunique()} symbols")
opt.to_pickle(SP + '/fp_opt_events.pkl')
fut.to_pickle(SP + '/fp_fut.pkl')
for h in FWD:
    FWD[h].to_pickle(SP + f'/fp_fwd{h}.pkl')
cash.to_pickle(SP + '/fp_cash.pkl')
print("saved intermediates.")
