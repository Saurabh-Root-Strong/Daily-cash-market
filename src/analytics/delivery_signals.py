from datetime import date
from typing import Optional
import pandas as pd

from src.data.repository import query_dataframe
from src.analytics.base import (
    get_min_turnover_filter, get_delivery_window, get_volume_window, get_thresholds
)
from src.logging_setup import get_logger

log = get_logger(__name__)


def get_stock_metrics(trade_date: date, min_turnover_lacs: Optional[float] = None) -> pd.DataFrame:
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()
    deliv_window = get_delivery_window()
    vol_window = get_volume_window()

    sql = f"""
        WITH base AS (
            SELECT
                b.trade_date,
                b.symbol,
                b.series,
                COALESCE(s.company_name, b.symbol) AS company_name,
                COALESCE(s.sector, 'Others') AS sector,
                COALESCE(s.industry, 'Others') AS industry,
                b.close_price,
                b.prev_close,
                CASE
                    WHEN b.prev_close > 0
                    THEN (b.close_price - b.prev_close) / b.prev_close * 100
                    ELSE NULL
                END AS price_change_pct,
                b.ttl_trd_qnty,
                b.turnover_lacs,
                b.deliv_qty,
                b.deliv_per,
                AVG(b.deliv_per) OVER (
                    PARTITION BY b.symbol, b.series
                    ORDER BY b.trade_date
                    ROWS BETWEEN {deliv_window} PRECEDING AND 1 PRECEDING
                ) AS deliv_per_10d_avg,
                AVG(b.ttl_trd_qnty) OVER (
                    PARTITION BY b.symbol, b.series
                    ORDER BY b.trade_date
                    ROWS BETWEEN {vol_window} PRECEDING AND 1 PRECEDING
                ) AS vol_20d_avg
            FROM daily_data b
            LEFT JOIN sector_master s ON b.symbol = s.symbol
            WHERE b.turnover_lacs >= ?
        )
        SELECT
            b.trade_date,
            b.symbol,
            b.series,
            b.company_name,
            b.sector,
            b.industry,
            b.close_price,
            b.prev_close,
            b.price_change_pct,
            b.ttl_trd_qnty,
            b.turnover_lacs,
            b.deliv_qty,
            b.deliv_per,
            b.deliv_per_10d_avg,
            (b.deliv_per / NULLIF(b.deliv_per_10d_avg, 0)) AS deliv_ratio,
            b.vol_20d_avg,
            (b.ttl_trd_qnty / NULLIF(b.vol_20d_avg, 0)) AS vol_ratio,
            (b.deliv_per / 100.0 * b.turnover_lacs) AS deliv_value_lacs
        FROM base b
        WHERE b.trade_date = ?
        ORDER BY (b.deliv_per / NULLIF(b.deliv_per_10d_avg, 0)) DESC NULLS LAST
    """

    return query_dataframe(sql, [min_turnover_lacs, trade_date])


def get_top_accumulation(trade_date: date, limit: int = 20) -> pd.DataFrame:
    df = get_stock_metrics(trade_date)
    if df.empty:
        return df
    _, acc_threshold = 0, 0
    acc_threshold, _ = get_thresholds()
    filtered = df[df["deliv_ratio"] >= acc_threshold].copy()
    return filtered.nlargest(limit, "deliv_ratio")


def get_top_distribution(trade_date: date, limit: int = 20) -> pd.DataFrame:
    df = get_stock_metrics(trade_date)
    if df.empty:
        return df
    _, dist_threshold = get_thresholds()
    filtered = df[
        (df["price_change_pct"] > 1.0) &
        (df["deliv_ratio"] < dist_threshold)
    ].copy()
    return filtered.nlargest(limit, "price_change_pct")


def get_stock_history(symbol: str, days: int = 60) -> pd.DataFrame:
    sql = """
        SELECT
            b.trade_date,
            b.close_price,
            b.prev_close,
            CASE
                WHEN b.prev_close > 0
                THEN (b.close_price - b.prev_close) / b.prev_close * 100
                ELSE NULL
            END AS price_change_pct,
            b.ttl_trd_qnty,
            b.turnover_lacs,
            b.deliv_qty,
            b.deliv_per
        FROM daily_data b
        WHERE b.symbol = ?
        ORDER BY b.trade_date DESC
        LIMIT ?
    """
    df = query_dataframe(sql, [symbol, days])
    return df.sort_values("trade_date").reset_index(drop=True)
