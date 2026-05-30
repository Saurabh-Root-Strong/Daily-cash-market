"""
Analytics configuration helpers and data-access re-exports.

Analytics modules import from here, not from src.data or src.core directly,
so the analytics public API surface stays stable.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from src.core.config import get_config
from src.data.repository import get_available_dates, get_latest_trade_date

__all__ = [
    "get_min_turnover_filter",
    "get_delivery_window",
    "get_volume_window",
    "get_thresholds",
    "get_weighting_method",
    "get_analytics_config",
    "minmax01",
    "rank01",
    # Re-exported for analytics consumers
    "get_latest_trade_date",
    "get_available_dates",
]


def minmax01(s: pd.Series) -> pd.Series:
    """Min-max scale a Series to [0,1]. Outlier-sensitive; prefer rank01 for scoring."""
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


def rank01(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank in [0,1]; missing → 0.5 (neutral).

    Distribution-free, so one outlier can't crush the rest toward 0 (the min-max
    failure mode), and a score built on it means the same thing every day. A
    348-day factor study (scripts/sector_score_compare.py) showed a rank blend
    with relative-strength beats the old min-max score at every 5/10/20-day horizon.
    """
    return s.rank(pct=True).fillna(0.5)


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
