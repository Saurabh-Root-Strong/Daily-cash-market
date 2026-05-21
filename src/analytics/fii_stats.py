"""
FII Derivatives Statistics analytics — buy/sell value by contract category.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_fii_stats_latest", "get_fii_stats_history"]

_CATEGORY_ORDER = {
    "Index Futures":      1,
    "Index Call Options": 2,
    "Index Put Options":  3,
    "Index Options":      4,
    "Stock Futures":      5,
    "Stock Call Options": 6,
    "Stock Put Options":  7,
    "Stock Options":      8,
    "Total":              9,
}


def get_fii_stats_latest(as_of_date: date) -> pd.DataFrame:
    """Latest FII derivatives stats on or before as_of_date."""
    df = query_dataframe("""
        WITH latest AS (
            SELECT MAX(trade_date) AS max_date
            FROM fii_derivatives_stats
            WHERE trade_date <= ?
        )
        SELECT
            f.trade_date, f.category,
            f.buy_contracts, f.sell_contracts,
            f.buy_contracts - f.sell_contracts AS net_contracts,
            f.buy_value_cr,  f.sell_value_cr,
            f.buy_value_cr  - f.sell_value_cr  AS net_value_cr,
            f.oi_contracts, f.oi_value_cr
        FROM fii_derivatives_stats f
        INNER JOIN latest l ON f.trade_date = l.max_date
        ORDER BY
            CASE f.category
                WHEN 'Index Futures'      THEN 1
                WHEN 'Index Call Options' THEN 2
                WHEN 'Index Put Options'  THEN 3
                WHEN 'Index Options'      THEN 4
                WHEN 'Stock Futures'      THEN 5
                WHEN 'Stock Call Options' THEN 6
                WHEN 'Stock Put Options'  THEN 7
                WHEN 'Stock Options'      THEN 8
                WHEN 'Total'              THEN 9
                ELSE 10
            END
    """, [as_of_date])
    return df


def get_fii_stats_history(as_of_date: date, lookback_days: int = 90) -> pd.DataFrame:
    """FII derivatives stats over the last lookback_days calendar days."""
    start = as_of_date - timedelta(days=lookback_days)
    return query_dataframe("""
        SELECT
            trade_date, category,
            buy_contracts, sell_contracts,
            buy_contracts - sell_contracts AS net_contracts,
            buy_value_cr, sell_value_cr,
            buy_value_cr - sell_value_cr   AS net_value_cr,
            oi_contracts, oi_value_cr
        FROM fii_derivatives_stats
        WHERE trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date DESC, category
    """, [start, as_of_date])
