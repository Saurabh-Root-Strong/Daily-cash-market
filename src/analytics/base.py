"""
Analytics configuration helpers and data-access re-exports.

Analytics modules import from here, not from src.data or src.core directly,
so the analytics public API surface stays stable.
"""
from __future__ import annotations

from typing import Tuple

from src.core.config import get_config
from src.data.repository import get_available_dates, get_latest_trade_date

__all__ = [
    "get_min_turnover_filter",
    "get_delivery_window",
    "get_volume_window",
    "get_thresholds",
    "get_weighting_method",
    "get_analytics_config",
    # Re-exported for analytics consumers
    "get_latest_trade_date",
    "get_available_dates",
]


def get_analytics_config():
    """Return the typed AnalyticsConfig — prefer this over individual getters."""
    return get_config().analytics


def get_min_turnover_filter() -> float:
    return get_config().analytics.min_turnover_lacs


def get_delivery_window() -> int:
    return get_config().analytics.delivery_avg_window


def get_volume_window() -> int:
    return get_config().analytics.volume_avg_window


def get_thresholds() -> Tuple[float, float]:
    ana = get_config().analytics
    return ana.accumulation_threshold, ana.distribution_threshold


def get_weighting_method() -> str:
    return get_config().analytics.weighting_method
