"""Cached analytics wrappers — import from .queries."""
from src.dashboard.cache.queries import (
    cached_aggregate_by_sector,
    cached_sector_drilldown,
    cached_sector_history,
    cached_sector_master_performance,
    cached_subsector_master_performance,
    cached_subsector_stocks_performance,
    cached_stock_metrics,
    cached_stock_history,
    cached_all_stocks,
    cached_stock_suggestions,
    cached_search_stocks,
    cached_sector_rotation,
    cached_sector_rotation_history,
    cached_sector_stocks_rotation,
)

__all__ = [
    "cached_aggregate_by_sector",
    "cached_sector_drilldown",
    "cached_sector_history",
    "cached_sector_master_performance",
    "cached_subsector_master_performance",
    "cached_subsector_stocks_performance",
    "cached_stock_metrics",
    "cached_stock_history",
    "cached_all_stocks",
    "cached_stock_suggestions",
    "cached_search_stocks",
    "cached_sector_rotation",
    "cached_sector_rotation_history",
    "cached_sector_stocks_rotation",
]
