"""
Data layer public API.

Import from here — never from sub-modules directly — so internal
reorganisation stays invisible to callers.
"""
from src.data.repository import (
    MarketDataRepository,
    get_repository,
    _set_repository,
    upsert_daily_data,
    update_delivery_data,
    upsert_sector_master,
    log_run,
    get_latest_trade_date,
    get_available_dates,
    get_total_row_count,
    get_dates_present,
    query_dataframe,
)
from src.data.schema import initialize_schema
from src.data.connection import ConnectionManager

__all__ = [
    "MarketDataRepository",
    "get_repository",
    "_set_repository",
    "ConnectionManager",
    "initialize_schema",
    "upsert_daily_data",
    "update_delivery_data",
    "upsert_sector_master",
    "log_run",
    "get_latest_trade_date",
    "get_available_dates",
    "get_total_row_count",
    "get_dates_present",
    "query_dataframe",
]
