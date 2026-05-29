"""
Streamlit-cached wrappers for every analytics query the dashboard makes.

All views import from here — never from src.analytics directly.
TTL = 300 s (5 min): data only changes when a new daily fetch completes.
Using lazy imports inside each function keeps app startup fast.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

_TTL = 300  # seconds


@st.cache_data(ttl=_TTL)
def cached_sector_master_performance(
    trade_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_aggregator import get_sector_master_performance
    return get_sector_master_performance(trade_date, min_turnover_lacs=min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_subsector_master_performance(
    trade_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_aggregator import get_subsector_master_performance
    return get_subsector_master_performance(trade_date, min_turnover_lacs=min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_subsector_stocks_performance(
    trade_date: date, sector: str, industry: str, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_aggregator import get_subsector_stocks_performance
    return get_subsector_stocks_performance(trade_date, sector, industry, min_turnover_lacs)


@st.cache_data(ttl=1800)  # 30 min — stock list rarely changes
def cached_all_stocks() -> pd.DataFrame:
    from src.analytics.sector_aggregator import get_all_stocks
    return get_all_stocks()


@st.cache_data(ttl=1800)
def cached_stock_close_prices(symbols: tuple, trade_date: date) -> dict:
    from src.analytics.sector_aggregator import get_stock_close_prices
    return get_stock_close_prices(symbols, trade_date)


@st.cache_data(ttl=_TTL)
def cached_search_stocks(
    trade_date: date, query: str, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_aggregator import search_stocks_performance
    return search_stocks_performance(trade_date, query, min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_sector_rotation(trade_date: date, min_turnover_lacs: float) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_sector_rotation
    return get_sector_rotation(trade_date, min_turnover_lacs=min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_sector_rotation_history(
    sector: str, trade_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_sector_rotation_history
    return get_sector_rotation_history(sector, trade_date, min_turnover_lacs=min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_sector_stocks_rotation(
    sector: str, trade_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_sector_stocks_rotation
    return get_sector_stocks_rotation(sector, trade_date, min_turnover_lacs=min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_fao_latest(trade_date: date, data_type: str) -> pd.DataFrame:
    from src.analytics.fao_participants import get_fao_latest
    return get_fao_latest(trade_date, data_type=data_type)


@st.cache_data(ttl=_TTL)
def cached_fao_daily(trade_date: date, lookback_days: int, data_type: str) -> pd.DataFrame:
    from src.analytics.fao_participants import get_fao_daily
    return get_fao_daily(trade_date, lookback_days=lookback_days, data_type=data_type)


@st.cache_data(ttl=_TTL)
def cached_fao_cumulative(
    trade_date: date, start_date: date, data_type: str
) -> pd.DataFrame:
    from src.analytics.fao_participants import get_fao_cumulative
    return get_fao_cumulative(trade_date, start_date=start_date, data_type=data_type)


@st.cache_data(ttl=1800)
def cached_fao_available_dates() -> list:
    from src.analytics.fao_participants import get_fao_available_dates
    return get_fao_available_dates()


@st.cache_data(ttl=_TTL)
def cached_index_snapshot(trade_date: date) -> pd.DataFrame:
    from src.analytics.index_momentum import get_index_snapshot
    return get_index_snapshot(trade_date)


@st.cache_data(ttl=_TTL)
def cached_index_history(index_name: str, trade_date: date, lookback_days: int = 120) -> pd.DataFrame:
    from src.analytics.index_momentum import get_index_history
    return get_index_history(index_name, trade_date, lookback_days)


@st.cache_data(ttl=_TTL)
def cached_index_heatmap(trade_date: date) -> pd.DataFrame:
    from src.analytics.index_momentum import get_index_heatmap
    return get_index_heatmap(trade_date)


@st.cache_data(ttl=_TTL)
def cached_market_intelligence(trade_date: date):
    from src.analytics.market_intelligence import get_market_intelligence
    return get_market_intelligence(trade_date)


@st.cache_data(ttl=_TTL)
def cached_fii_stats_latest(trade_date: date) -> pd.DataFrame:
    from src.analytics.fii_stats import get_fii_stats_latest
    return get_fii_stats_latest(trade_date)


@st.cache_data(ttl=_TTL)
def cached_fii_stats_history(trade_date: date, lookback_days: int = 90) -> pd.DataFrame:
    from src.analytics.fii_stats import get_fii_stats_history
    return get_fii_stats_history(trade_date, lookback_days=lookback_days)


@st.cache_data(ttl=_TTL)
def cached_fno_summary(trade_date: date) -> dict:
    from src.analytics.fno_activity import get_fno_summary_stats
    return get_fno_summary_stats(trade_date)


@st.cache_data(ttl=_TTL)
def cached_fno_expiry_calendar(trade_date: date) -> pd.DataFrame:
    from src.analytics.fno_activity import get_expiry_calendar
    return get_expiry_calendar(trade_date)


@st.cache_data(ttl=_TTL)
def cached_fno_index_expiry_oi(trade_date: date, symbol: str) -> pd.DataFrame:
    from src.analytics.fno_activity import get_index_expiry_oi
    return get_index_expiry_oi(trade_date, symbol)


@st.cache_data(ttl=_TTL)
def cached_index_futures_rollover(trade_date: date, symbol: str) -> pd.DataFrame:
    from src.analytics.fno_activity import get_index_futures_rollover
    return get_index_futures_rollover(trade_date, symbol)


@st.cache_data(ttl=_TTL)
def cached_fno_stock_leaders(trade_date: date, top_n: int = 25) -> pd.DataFrame:
    from src.analytics.fno_activity import get_stock_oi_leaders
    return get_stock_oi_leaders(trade_date, top_n)


@st.cache_data(ttl=_TTL)
def cached_fno_index_symbols(trade_date: date) -> list:
    from src.analytics.fno_activity import get_index_symbols_active
    return get_index_symbols_active(trade_date)


@st.cache_data(ttl=_TTL)
def cached_fno_dates_available() -> list:
    from src.analytics.fno_activity import get_fno_dates_available
    return get_fno_dates_available()


@st.cache_data(ttl=_TTL)
def cached_fno_expiry_oi_history(symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
    from src.analytics.fno_activity import get_expiry_oi_history
    return get_expiry_oi_history(symbol, from_date, to_date)


@st.cache_data(ttl=_TTL)
def cached_available_dates(limit: int = 500) -> list:
    from src.analytics.base import get_available_dates
    return get_available_dates(limit=limit)


@st.cache_data(ttl=_TTL)
def cached_fno_stock_signals(trade_date: date, min_fut_oi: int = 50_000) -> pd.DataFrame:
    from src.analytics.fno_stocks import get_fno_stock_oi_signals
    return get_fno_stock_oi_signals(trade_date, min_fut_oi=min_fut_oi)


@st.cache_data(ttl=_TTL)
def cached_sector_oi_summary(trade_date: date, min_fut_oi: int = 50_000) -> pd.DataFrame:
    from src.analytics.fno_stocks import get_sector_oi_summary
    return get_sector_oi_summary(trade_date, min_fut_oi=min_fut_oi)


@st.cache_data(ttl=_TTL)
def cached_fno_composite_signals(trade_date: date, min_fut_oi: int = 50_000) -> pd.DataFrame:
    from src.analytics.fno_signals import get_fno_composite_signals
    return get_fno_composite_signals(trade_date, min_fut_oi=min_fut_oi)


@st.cache_data(ttl=_TTL)
def cached_stock_monthly_expiries(trade_date: date) -> list:
    from src.analytics.fno_expiry import get_stock_monthly_expiries
    return get_stock_monthly_expiries(trade_date)


@st.cache_data(ttl=_TTL)
def cached_stock_expiry_matrix(trade_date: date, min_fut_oi: int = 50_000) -> pd.DataFrame:
    from src.analytics.fno_expiry import get_stock_expiry_matrix
    return get_stock_expiry_matrix(trade_date, min_fut_oi=min_fut_oi)


@st.cache_data(ttl=_TTL)
def cached_index_full_structure(trade_date: date, symbol: str) -> pd.DataFrame:
    from src.analytics.fno_expiry import get_index_full_structure
    return get_index_full_structure(trade_date, symbol)


@st.cache_data(ttl=_TTL)
def cached_options_chain(
    trade_date: date, symbol: str, expiry_date: date, instrument: str = "OPTSTK"
) -> pd.DataFrame:
    from src.analytics.fno_expiry import get_options_chain
    return get_options_chain(trade_date, symbol, expiry_date, instrument)


@st.cache_data(ttl=_TTL)
def cached_index_options_chain(
    trade_date: date, symbol: str, expiry_date: date, n_strikes: int = 15
) -> pd.DataFrame:
    from src.analytics.fno_expiry import get_index_options_chain
    return get_index_options_chain(trade_date, symbol, expiry_date, n_strikes)


@st.cache_data(ttl=_TTL)
def cached_sector_rotation_timeframe(
    trade_date: date, window_trading_days: int, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_sector_rotation_timeframe
    return get_sector_rotation_timeframe(trade_date, window_trading_days, min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_rotation_clock_backtest(
    trade_date: date, window_trading_days: int, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_rotation_clock_backtest
    return get_rotation_clock_backtest(trade_date, window_trading_days, min_turnover_lacs)


@st.cache_data(ttl=3600)   # 1 hour — backtest is expensive; re-runs daily at most
def cached_signal_backtest(
    end_date: date,
    backtest_days: int = 60,
    threshold_pct: float = 0.25,
):
    from src.analytics.signal_backtest import run_signal_backtest
    return run_signal_backtest(end_date, backtest_days=backtest_days, threshold_pct=threshold_pct)


@st.cache_data(ttl=_TTL)
def cached_sector_rotation_custom_range(
    from_date: date, to_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_sector_rotation_custom_range
    return get_sector_rotation_custom_range(from_date, to_date, min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_sector_rs_custom_range(
    from_date: date, to_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_sector_rs_custom_range
    return get_sector_rs_custom_range(from_date, to_date, min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_sector_stocks_custom_range(
    sector: str, from_date: date, to_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    from src.analytics.sector_rotation import get_sector_stocks_custom_range
    return get_sector_stocks_custom_range(sector, from_date, to_date, min_turnover_lacs)


# ── FPI Capital Flow ─────────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def cached_fpi_available_dates() -> list:
    from src.analytics.fpi_flows import get_fpi_available_dates
    return get_fpi_available_dates()


@st.cache_data(ttl=1800)
def cached_fpi_date_range() -> tuple:
    from src.analytics.fpi_flows import get_fpi_date_range
    return get_fpi_date_range()


@st.cache_data(ttl=_TTL)
def cached_fpi_summary(as_of_date: date, lookback_days: int = 180) -> pd.DataFrame:
    from src.analytics.fpi_flows import get_fpi_summary
    return get_fpi_summary(as_of_date, lookback_days)


@st.cache_data(ttl=_TTL)
def cached_fpi_category_breakdown(as_of_date: date, lookback_days: int = 15) -> pd.DataFrame:
    from src.analytics.fpi_flows import get_fpi_category_breakdown
    return get_fpi_category_breakdown(as_of_date, lookback_days)


@st.cache_data(ttl=_TTL)
def cached_fpi_risk_appetite(as_of_date: date, lookback_days: int = 90) -> pd.DataFrame:
    from src.analytics.fpi_flows import get_fpi_risk_appetite
    return get_fpi_risk_appetite(as_of_date, lookback_days)


@st.cache_data(ttl=_TTL)
def cached_fpi_15d_outlook(as_of_date: date) -> dict:
    from src.analytics.fpi_flows import get_fpi_15d_outlook
    return get_fpi_15d_outlook(as_of_date)


# ── Index Prediction ──────────────────────────────────────────────────────────

@st.cache_data(ttl=_TTL)
def cached_index_predictions(trade_date: date) -> list:
    from src.analytics.index_prediction import get_index_predictions
    return get_index_predictions(trade_date)


# ── Sector Signal Backtest ────────────────────────────────────────────────────

@st.cache_data(ttl=_TTL)
def cached_sector_signal_log(
    as_of_date: date, min_turnover_lacs: float, lookback_dates: int = 30
) -> pd.DataFrame:
    from src.analytics.sector_signal_backtest import get_sector_signal_log
    return get_sector_signal_log(as_of_date, min_turnover_lacs, lookback_dates)


@st.cache_data(ttl=_TTL)
def cached_sector_accuracy_summary(
    as_of_date: date, min_turnover_lacs: float, lookback_dates: int = 60
) -> dict:
    from src.analytics.sector_signal_backtest import get_sector_accuracy_summary
    return get_sector_accuracy_summary(as_of_date, min_turnover_lacs, lookback_dates)
