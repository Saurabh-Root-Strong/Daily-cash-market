"""Audit of the corporate-action adjusted return (src/analytics/adjusted_returns.py).

    python scripts/adjusted_returns_audit.py

Read-only. Re-run after a corp_actions refresh or a change to the rule.

  A  EQUIVALENCE   ret_sql() equals the raw (close - prev_close) / prev_close on
                   every row with no split, bonus or demerger -- the adjustment
                   may not move an ordinary day
  B  COVERAGE      how many -35%-or-worse prints remain in the sector universe
                   after adjustment, and what they are (real crashes, or actions
                   missing from NSE's table)
  C  RENAMES       corporate actions NSE files under a symbol the price row did
                   not carry yet, and whether the rename map reaches them
  D  SPEED         a sector-style aggregation, raw against adjusted

Last run 06 Oct 2026: A 571 of 4,229,636 rows differ, all on an action day;
B 239 raw -> 3 adjusted (POLICYBZR real, HERITGFOOD and GRAUWEIL not in the
table); C 22 action days reached through 31 rename pairs; D about one extra
second on a whole-market scan (no measurable change in page load time).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.analytics import adjusted_returns as ar
from src.data.repository import query_dataframe as q

pd.set_option("display.width", 220)
RAW = "(b.close_price - b.prev_close) / b.prev_close * 100"

print("=== A. equivalence, every EQ/SM/ST row")
r = q(f"""
    SELECT COUNT(*) AS n,
           SUM(CASE WHEN ABS({ar.ret_sql('b')} - {RAW}) > 1e-9 THEN 1 ELSE 0 END) AS differ,
           SUM(CASE WHEN {ar.factor_sql('b')} <> 1 THEN 1 ELSE 0 END) AS split_bonus,
           SUM(CASE WHEN {ar.demerger_sql('b')} THEN 1 ELSE 0 END) AS demerger,
           SUM(CASE WHEN ABS({ar.ret_sql('b')} - {RAW}) > 1e-9 AND {ar.factor_sql('b')} = 1
                     AND NOT {ar.demerger_sql('b')} THEN 1 ELSE 0 END) AS unexplained
    FROM daily_data b WHERE b.series IN ('EQ','SM','ST') AND b.prev_close > 0""").iloc[0]
print(f"  rows {int(r.n):,}  differ {int(r.differ)}  (split/bonus {int(r.split_bonus)}, "
      f"demerger {int(r.demerger)})  unexplained {int(r.unexplained)}")
assert int(r.unexplained) == 0, "an ordinary row moved"

print("\n=== B. coverage in the sector universe (ETFs and 'Others' excluded), 2022+")
u = q(f"""
    SELECT b.trade_date, b.symbol, {RAW} AS raw, {ar.ret_sql('b')} AS adj,
           b.open_price / b.prev_close * 100 - 100 AS open_gap, b.turnover_lacs / 100 AS cr
    FROM daily_data b JOIN v_sector_master s ON s.symbol = b.symbol
    WHERE b.series = 'EQ' AND b.prev_close > 0 AND b.trade_date >= '2022-01-01'
      AND s.sector NOT IN ('ETF', 'Others') AND {RAW} < -35""")
left = u[u["adj"] < -35]
print(f"  raw prints below -35%: {len(u)}   after adjustment: {len(left)}")
print(left.sort_values("cr", ascending=False).to_string(index=False, float_format="%.1f"))

print("\n=== C. renames")
m = ar.ca_symbol_map()
print(f"  rename pairs in use: {len(m)}")
hit = q(f"""
    SELECT COUNT(*) AS n FROM daily_data b
    WHERE b.series = 'EQ' AND b.prev_close > 0 AND {ar.symbol_sql('b')} <> b.symbol
      AND ({ar.factor_sql('b')} <> 1 OR {ar.demerger_sql('b')})""")
print(f"  action days reached only through the map: {int(hit.n.iloc[0])}")

print("\n=== D. speed, daily cross-sectional median since 2024")
for name, expr in (("raw", RAW), ("adjusted", ar.ret_sql("b"))):
    t0 = time.time()
    q(f"""SELECT b.trade_date, MEDIAN({expr}) FROM daily_data b
          WHERE b.series = 'EQ' AND b.prev_close > 0 AND b.trade_date >= '2024-01-01'
          GROUP BY 1""")
    print(f"  {name:9s} {time.time() - t0:.2f}s")
