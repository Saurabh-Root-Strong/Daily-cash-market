"""Cached analytics wrappers — import from .queries."""
from src.dashboard.cache.queries import (
    cached_sector_master_performance,
    cached_subsector_master_performance,
    cached_subsector_stocks_performance,
    cached_all_stocks,
    cached_search_stocks,
    cached_sector_rotation,
    cached_sector_rotation_history,
    cached_sector_stocks_rotation,
)

__all__ = [
    "cached_sector_master_performance",
    "cached_subsector_master_performance",
    "cached_subsector_stocks_performance",
    "cached_all_stocks",
    "cached_search_stocks",
    "cached_sector_rotation",
    "cached_sector_rotation_history",
    "cached_sector_stocks_rotation",
]
