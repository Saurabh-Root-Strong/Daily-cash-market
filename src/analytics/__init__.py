"""
Analytics layer public API.

The dashboard imports from here.  Never import from sub-modules directly.
"""
from src.analytics.base import (
    get_analytics_config,
    get_min_turnover_filter,
    get_delivery_window,
    get_volume_window,
    get_thresholds,
    get_weighting_method,
    get_latest_trade_date,
    get_available_dates,
)
from src.analytics.delivery_signals import (
    get_stock_metrics,
    get_stock_history,
)
from src.analytics.sector_aggregator import (
    aggregate_by_sector,
    get_sector_drilldown,
    get_sector_history,
    get_sector_master_performance,
    get_subsector_master_performance,
    get_subsector_stocks_performance,
)

__all__ = [
    "get_analytics_config",
    "get_min_turnover_filter",
    "get_delivery_window",
    "get_volume_window",
    "get_thresholds",
    "get_weighting_method",
    "get_latest_trade_date",
    "get_available_dates",
    "get_stock_metrics",
    "get_stock_history",
    "aggregate_by_sector",
    "get_sector_drilldown",
    "get_sector_history",
    "get_sector_master_performance",
    "get_subsector_master_performance",
    "get_subsector_stocks_performance",
]
