"""
Analytics layer public API.

All dashboard code imports from here — never from sub-modules directly.
Lazy imports inside each export keep app startup fast.
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
    get_all_stocks,
    get_stock_close_prices,
    search_stocks_performance,
    search_stock_suggestions,
)
from src.analytics.sector_rotation import (
    get_sector_rotation,
    get_sector_rotation_history,
    get_sector_stocks_rotation,
    get_sector_rotation_timeframe,
    get_rotation_clock_backtest,
    get_sector_rotation_custom_range,
    get_sector_rs_custom_range,
    get_sector_stocks_custom_range,
)
from src.analytics.fao_participants import (
    get_fao_latest,
    get_fao_daily,
    get_fao_cumulative,
    get_fao_available_dates,
)
from src.analytics.fii_stats import (
    get_fii_stats_latest,
    get_fii_stats_history,
)
from src.analytics.index_momentum import (
    get_index_snapshot,
    get_index_history,
    get_index_heatmap,
)
from src.analytics.market_intelligence import get_market_intelligence
from src.analytics.signal_backtest import run_signal_backtest
from src.analytics.fno_activity import (
    get_fno_summary_stats,
    get_expiry_calendar,
    get_index_expiry_oi,
    get_index_futures_rollover,
    get_stock_oi_leaders,
    get_index_symbols_active,
    get_fno_dates_available,
    get_expiry_oi_history,
)
from src.analytics.fno_stocks import (
    get_fno_stock_oi_signals,
    get_sector_oi_summary,
)
from src.analytics.fno_signals import get_fno_composite_signals
from src.analytics.fno_expiry import (
    get_stock_monthly_expiries,
    get_stock_expiry_matrix,
    get_index_full_structure,
    get_options_chain,
    get_index_options_chain,
)

__all__ = [
    # base
    "get_analytics_config",
    "get_min_turnover_filter",
    "get_delivery_window",
    "get_volume_window",
    "get_thresholds",
    "get_weighting_method",
    "get_latest_trade_date",
    "get_available_dates",
    # delivery signals
    "get_stock_metrics",
    "get_stock_history",
    # sector aggregator
    "aggregate_by_sector",
    "get_sector_drilldown",
    "get_sector_history",
    "get_sector_master_performance",
    "get_subsector_master_performance",
    "get_subsector_stocks_performance",
    "get_all_stocks",
    "get_stock_close_prices",
    "search_stocks_performance",
    "search_stock_suggestions",
    # sector rotation
    "get_sector_rotation",
    "get_sector_rotation_history",
    "get_sector_stocks_rotation",
    "get_sector_rotation_timeframe",
    "get_rotation_clock_backtest",
    "get_sector_rotation_custom_range",
    "get_sector_rs_custom_range",
    "get_sector_stocks_custom_range",
    # F&O participants
    "get_fao_latest",
    "get_fao_daily",
    "get_fao_cumulative",
    "get_fao_available_dates",
    # FII stats
    "get_fii_stats_latest",
    "get_fii_stats_history",
    # index momentum
    "get_index_snapshot",
    "get_index_history",
    "get_index_heatmap",
    # market intelligence
    "get_market_intelligence",
    "run_signal_backtest",
    # F&O activity
    "get_fno_summary_stats",
    "get_expiry_calendar",
    "get_index_expiry_oi",
    "get_index_futures_rollover",
    "get_stock_oi_leaders",
    "get_index_symbols_active",
    "get_fno_dates_available",
    "get_expiry_oi_history",
    # F&O stocks
    "get_fno_stock_oi_signals",
    "get_sector_oi_summary",
    # F&O signals
    "get_fno_composite_signals",
    # F&O expiry
    "get_stock_monthly_expiries",
    "get_stock_expiry_matrix",
    "get_index_full_structure",
    "get_options_chain",
    "get_index_options_chain",
]
