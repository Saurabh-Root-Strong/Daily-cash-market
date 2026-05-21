"""
F&O Participant Analytics — Big Players Tracker.

Key insight:
  OI (Open Interest) shows POSITIONS OUTSTANDING — what FII/DII/Client/Pro
  are actually holding.  Cumulative net OI in Index Futures is the most direct
  proxy for institutional market direction conviction.

  Volume shows intraday ACTIVITY — useful for detecting sudden position building.

Signal logic:
  Index Futures NET = fut_idx_long - fut_idx_short
  • Positive (net long) → institutions bullish on market direction
  • Negative (net short) → institutions hedged / bearish on market

  Index Options:
  • opt_idx_call_net = call_long - call_short (positive = net long calls = bullish)
  • opt_idx_put_net  = put_long  - put_short  (positive = net long puts = bearish hedge)
  • opt_idx_net (delta proxy) = call_net - put_net (positive = overall bullish options stance)

  PCR = Total Put OI / Total Call OI (across all participants)
  • PCR > 1.3 = excessive puts → contrarian bullish
  • PCR < 0.7 = excessive calls → contrarian bearish / complacency
  • PCR 0.8–1.2 = neutral
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_fao_latest",
    "get_fao_daily",
    "get_fao_cumulative",
    "get_fao_available_dates",
]


def get_fao_available_dates() -> list[date]:
    """Return all trade_dates present in fao_participant, most recent first."""
    df = query_dataframe(
        "SELECT DISTINCT trade_date FROM fao_participant "
        "ORDER BY trade_date DESC"
    )
    if df.empty:
        return []
    return list(df["trade_date"])


def get_fao_latest(
    as_of_date: date,
    data_type: str = "OI",
) -> pd.DataFrame:
    """
    Latest available date's full breakdown for all 4 participants.

    Returns Futures + Index Options + Stock Options + Totals.
    """
    sql = """
        WITH latest AS (
            SELECT MAX(trade_date) AS max_date
            FROM fao_participant
            WHERE data_type = ?
              AND trade_date <= ?
        )
        SELECT
            f.trade_date,
            f.client_type,
            -- Index Futures
            f.fut_idx_long,
            f.fut_idx_short,
            f.fut_idx_long  - f.fut_idx_short  AS fut_idx_net,
            CAST(f.fut_idx_long AS DOUBLE) * 100.0
                / NULLIF(f.fut_idx_long + f.fut_idx_short, 0)  AS fut_idx_ls_pct,
            -- Stock Futures
            f.fut_stk_long,
            f.fut_stk_short,
            f.fut_stk_long  - f.fut_stk_short  AS fut_stk_net,
            -- Index Options — Call
            f.opt_idx_call_long,
            f.opt_idx_call_short,
            f.opt_idx_call_long  - f.opt_idx_call_short  AS opt_idx_call_net,
            -- Index Options — Put
            f.opt_idx_put_long,
            f.opt_idx_put_short,
            f.opt_idx_put_long   - f.opt_idx_put_short   AS opt_idx_put_net,
            -- Options delta proxy: net call exposure minus net put exposure
            (f.opt_idx_call_long - f.opt_idx_call_short)
            - (f.opt_idx_put_long - f.opt_idx_put_short) AS opt_idx_net,
            -- Stock Options
            f.opt_stk_call_long,
            f.opt_stk_call_short,
            f.opt_stk_call_long  - f.opt_stk_call_short  AS opt_stk_call_net,
            f.opt_stk_put_long,
            f.opt_stk_put_short,
            f.opt_stk_put_long   - f.opt_stk_put_short   AS opt_stk_put_net,
            -- Total
            f.total_long,
            f.total_short,
            f.total_long - f.total_short AS total_net
        FROM fao_participant f
        INNER JOIN latest l ON f.trade_date = l.max_date
        WHERE f.data_type = ?
        ORDER BY
            CASE f.client_type
                WHEN 'FII'    THEN 1
                WHEN 'DII'    THEN 2
                WHEN 'Client' THEN 3
                WHEN 'Pro'    THEN 4
                ELSE 5
            END
    """
    return query_dataframe(sql, [data_type, as_of_date, data_type])


def get_fao_daily(
    as_of_date: date,
    lookback_days: int = 90,
    data_type: str = "OI",
) -> pd.DataFrame:
    """
    Daily raw + net positions (Futures + Options) for all participants.

    One row per (trade_date × client_type), sorted most recent first.
    """
    start = as_of_date - timedelta(days=lookback_days)
    sql = """
        SELECT
            trade_date,
            client_type,
            -- Index Futures
            fut_idx_long,
            fut_idx_short,
            fut_idx_long  - fut_idx_short  AS fut_idx_net,
            CAST(fut_idx_long AS DOUBLE) * 100.0
                / NULLIF(fut_idx_long + fut_idx_short, 0)  AS fut_idx_ls_pct,
            -- Index Options
            opt_idx_call_long,
            opt_idx_call_short,
            opt_idx_call_long  - opt_idx_call_short  AS opt_idx_call_net,
            opt_idx_put_long,
            opt_idx_put_short,
            opt_idx_put_long   - opt_idx_put_short   AS opt_idx_put_net,
            (opt_idx_call_long - opt_idx_call_short)
            - (opt_idx_put_long - opt_idx_put_short) AS opt_idx_net,
            -- Stock Futures + Options
            fut_stk_long,
            fut_stk_short,
            fut_stk_long  - fut_stk_short  AS fut_stk_net,
            opt_stk_call_long - opt_stk_call_short AS opt_stk_call_net,
            opt_stk_put_long  - opt_stk_put_short  AS opt_stk_put_net,
            -- Total
            total_long,
            total_short,
            total_long - total_short AS total_net
        FROM fao_participant
        WHERE data_type = ?
          AND trade_date >  ?
          AND trade_date <= ?
        ORDER BY trade_date DESC,
            CASE client_type
                WHEN 'FII'    THEN 1
                WHEN 'DII'    THEN 2
                WHEN 'Client' THEN 3
                WHEN 'Pro'    THEN 4
                ELSE 5
            END
    """
    return query_dataframe(sql, [data_type, start, as_of_date])


def get_fao_cumulative(
    as_of_date: date,
    start_date: Optional[date] = None,
    data_type: str = "OI",
) -> pd.DataFrame:
    """
    Cumulative net positions (running SUM from start_date) per participant.

    Includes Futures + Index Options (Call net, Put net, combined delta).
    """
    if start_date is None:
        start_date = as_of_date - timedelta(days=365)

    sql = """
        SELECT
            trade_date,
            client_type,
            -- Daily deltas
            fut_idx_long  - fut_idx_short   AS daily_fut_idx_net,
            opt_idx_call_long - opt_idx_call_short AS daily_opt_idx_call_net,
            opt_idx_put_long  - opt_idx_put_short  AS daily_opt_idx_put_net,
            (opt_idx_call_long - opt_idx_call_short)
            - (opt_idx_put_long - opt_idx_put_short) AS daily_opt_idx_net,
            total_long - total_short        AS daily_total_net,
            -- Running cumulative sums
            SUM(fut_idx_long  - fut_idx_short)   OVER w AS cum_fut_idx_net,
            SUM(opt_idx_call_long - opt_idx_call_short) OVER w AS cum_opt_idx_call_net,
            SUM(opt_idx_put_long  - opt_idx_put_short)  OVER w AS cum_opt_idx_put_net,
            SUM(
                (opt_idx_call_long - opt_idx_call_short)
                - (opt_idx_put_long - opt_idx_put_short)
            ) OVER w AS cum_opt_idx_net,
            SUM(total_long - total_short)        OVER w AS cum_total_net,
            -- Raw for L/S ratio
            fut_idx_long,
            fut_idx_short,
            CAST(fut_idx_long AS DOUBLE) * 100.0
                / NULLIF(fut_idx_long + fut_idx_short, 0)  AS fut_idx_ls_pct,
            opt_idx_call_long,
            opt_idx_call_short,
            opt_idx_put_long,
            opt_idx_put_short
        FROM fao_participant
        WHERE data_type = ?
          AND trade_date >= ?
          AND trade_date <= ?
        WINDOW w AS (
            PARTITION BY client_type
            ORDER BY trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
        ORDER BY trade_date,
            CASE client_type
                WHEN 'FII'    THEN 1
                WHEN 'DII'    THEN 2
                WHEN 'Client' THEN 3
                WHEN 'Pro'    THEN 4
                ELSE 5
            END
    """
    return query_dataframe(sql, [data_type, start_date, as_of_date])
