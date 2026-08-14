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
def cached_market_regime(trade_date: date) -> dict:
    """
    Market regime computed from Nifty50 EMAs + VIX + FII flow + HMM state.
    Returns a dict with: regime, score, signals, invest_label, invest_caption,
    avoid_label, avoid_caption — all consumed by _render_smart_money().
    """
    from src.analytics.sector_rotation import get_market_regime
    return get_market_regime(trade_date)


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
def cached_price_action(
    trade_date: date, lookback_days: int = 60, min_turnover_lacs: float = 1.0
) -> pd.DataFrame:
    """Per-stock price-action character (trend efficiency + candle anatomy)."""
    from src.analytics.price_action import get_price_action
    return get_price_action(trade_date, lookback_days=lookback_days,
                            min_turnover_lacs=min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_forward_tilt(trade_date: date, min_turnover_lacs: float = 1.0,
                        horizon_days: int = 10):
    """
    Forward sector tilt + regime meta. Returns (DataFrame, dict).

    `horizon_days` scales the RS lookbacks with the forward window; 10 is the
    validated 1-2wk build and is bit-identical to the previous behaviour.
    """
    from src.analytics.sector_forward_tilt import get_forward_tilt
    return get_forward_tilt(trade_date, min_turnover_lacs=min_turnover_lacs,
                            horizon_days=horizon_days)


@st.cache_data(ttl=_TTL)
def cached_tilt_replay(as_of: date, horizon_days: int = 10,
                       min_turnover_lacs: float = 1.0, today: date | None = None) -> dict:
    """Replay the tilt as of a past date and score what followed. See module docstring."""
    from src.analytics.sector_forward_tilt import get_tilt_replay
    return get_tilt_replay(as_of, horizon_days=horizon_days,
                           min_turnover_lacs=min_turnover_lacs, today=today)


@st.cache_data(ttl=_TTL)
def cached_operator_footprint(trade_date: date, min_notional_cr: float = 5.0) -> dict:
    """Unusual single-stock F&O positioning. DESCRIPTIVE - no measured forward edge."""
    from src.analytics.operator_footprint import get_operator_footprint
    return get_operator_footprint(trade_date, min_notional_cr=min_notional_cr)


@st.cache_data(ttl=_TTL)
def cached_clock_replay(as_of: date, window: int = 5,
                        min_turnover_lacs: float = 1.0,
                        today: date | None = None) -> dict:
    """Replay the Rotation Clock as of a past date and score what followed."""
    from src.analytics.sector_rotation import get_clock_replay
    return get_clock_replay(as_of, window_trading_days=window,
                            min_turnover_lacs=min_turnover_lacs, today=today)


@st.cache_data(ttl=_TTL)
def cached_clock_stock_detail(sector: str, trade_date: date, min_turnover_lacs: float = 1.0,
                              lookback_days: int = 10):
    """
    Stocks inside a Rotation-Clock sector, ranked by delivery-vs-own-normal.

    `lookback_days` MUST be the Clock's selected Analysis Period (5/10/22/65).
    It is part of the cache key, so each window caches separately.
    """
    from src.analytics.sector_rotation import get_clock_stock_detail
    return get_clock_stock_detail(sector, trade_date, min_turnover_lacs,
                                  lookback_days=lookback_days)


@st.cache_data(ttl=_TTL)
def cached_sector_month_map(trade_date: date):
    """Sector x month seasonality grid + meta. DESCRIPTIVE — see module docstring."""
    from src.analytics.sector_seasonality import get_sector_month_map
    return get_sector_month_map(trade_date)


@st.cache_data(ttl=_TTL)
def cached_month_suggestion(trade_date: date, top_k: int = 4) -> dict:
    """Causal (walk-forward) sector suggestion for the month about to start."""
    from src.analytics.sector_seasonality import get_next_month_suggestion
    return get_next_month_suggestion(trade_date, top_k=top_k)


@st.cache_data(ttl=_TTL)
def cached_seasonality_record(trade_date: date, top_k: int = 3, lens: str = "dcm",
                              mode: str = "month", since: str | None = None) -> dict:
    """
    Walk-forward track record of the seasonality rule.

    `mode="persistence"` is the CONTROL (ranks sectors ignoring the calendar
    month). It must be displayed next to `mode="month"`, because on DCM buckets
    the control reproduces ~96% of the seasonal rule's return — the difference
    between the two is the only genuinely seasonal part.
    """
    from src.analytics.sector_seasonality import get_seasonality_track_record
    return get_seasonality_track_record(trade_date, top_k=top_k, lens=lens,
                                        mode=mode, since=since)


@st.cache_data(ttl=_TTL)
def cached_sector_defensive(trade_date: date, min_turnover_lacs: float = 1.0) -> pd.DataFrame:
    """Per-sector defensive metrics (beta, down/up-capture, RS). DESCRIPTIVE context only."""
    from src.analytics.sector_defensive import get_sector_defensive_metrics
    return get_sector_defensive_metrics(trade_date, min_turnover_lacs=min_turnover_lacs)


@st.cache_data(ttl=_TTL)
def cached_market_breadth(trade_date: date) -> dict:
    """Large-cap breadth + 200-DMA regime NOWCAST (situational context, not a forecast)."""
    from src.analytics.sector_forward_tilt import get_market_breadth
    return get_market_breadth(trade_date)


@st.cache_data(ttl=_TTL)
def cached_nifty_breakout(trade_date: date) -> dict:
    """Nifty 20d-high breakout state (HELD vs FAILED) — sharpest 8yr signal, context flag."""
    from src.analytics.sector_forward_tilt import get_nifty_breakout
    return get_nifty_breakout(trade_date)


@st.cache_data(ttl=_TTL)
def cached_mtf_trend(trade_date: date) -> dict:
    """Multi-timeframe Nifty trend (swing/short/long/vlong) + entry posture — 'is trend changing?'"""
    from src.analytics.sector_forward_tilt import get_mtf_trend
    return get_mtf_trend(trade_date)


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
def cached_fno_positioning_by_symbol(trade_date: date) -> pd.DataFrame:
    """Per-symbol F&O futures/options positioning — overlay for Sector Rotation."""
    from src.analytics.fno_stocks import get_fno_positioning_by_symbol
    return get_fno_positioning_by_symbol(trade_date)


@st.cache_data(ttl=_TTL)
def cached_sector_fno_aggregate(trade_date: date) -> pd.DataFrame:
    """Sector-level F&O bias badge — overlay for Sector Rotation cards."""
    from src.analytics.fno_stocks import get_sector_fno_aggregate
    return get_sector_fno_aggregate(trade_date)


@st.cache_data(ttl=_TTL)
def cached_fno_expiry_breakdown(trade_date: date) -> pd.DataFrame:
    """Per-symbol near/next/far futures OI change % and options PCR for all 3 expiries."""
    from src.analytics.fno_stocks import get_fno_expiry_breakdown_by_symbol
    return get_fno_expiry_breakdown_by_symbol(trade_date)


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
def cached_market_context(trade_date: date):
    """
    Shared market context for a trade date — the heaviest single piece of the
    prediction pipeline (VIX, FAO/FII flows, constituents, breadth). Cached once
    per date and reused across all indices AND across the Index Prediction and
    Prediction Memory pages, so switching index / page never rebuilds it.
    """
    from src.analytics.index_prediction import _build_market_context
    return _build_market_context(trade_date)


@st.cache_data(ttl=_TTL)
def cached_index_predictions(trade_date: date) -> list:
    # persist=False: read-only display path — daily logging is done by cmd_daily,
    # so the dashboard never needs to take a prediction_log write lock on render.
    from src.analytics.index_prediction import get_index_predictions
    return get_index_predictions(
        trade_date, persist=False, market_ctx=cached_market_context(trade_date),
    )


@st.cache_data(ttl=_TTL)
def cached_index_prediction_one(trade_date: date, fno_symbol: str):
    """Single-index prediction — for views that display one index at a time."""
    from src.analytics.index_prediction import get_index_prediction_for
    return get_index_prediction_for(
        trade_date, fno_symbol, persist=False, market_ctx=cached_market_context(trade_date),
    )


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


# ── Sector Rotation Memory Engine ────────────────────────────────────────────

@st.cache_data(ttl=_TTL)
def cached_sector_memory_context(
    as_of_date:     date,
    sector:         str,
    signal:         str,
    regime_label:   str,
    dv_ratio:       float,
    z_pct:          float,
    rs_1w:          "float | None",
    ema20_above:    bool,
    ema_cross_bull: "bool | None",
    vix:            "float | None",
    fii_5d_cr:      "float | None",
    hmm_state:      "str | None",
    pcr:            "float | None",
):
    """
    Historical outcome statistics for (sector, signal, market-regime conditions).
    Returns a SectorMemoryContext dataclass from sector_memory.py.
    All arguments are hashable so Streamlit can cache the result for 5 minutes.
    """
    from src.analytics.sector_memory import get_sector_memory_context
    return get_sector_memory_context(
        as_of_date     = as_of_date,
        sector         = sector,
        signal         = signal,
        regime_label   = regime_label,
        dv_ratio       = dv_ratio,
        z_pct          = z_pct,
        rs_1w          = rs_1w,
        ema20_above    = ema20_above,
        ema_cross_bull = ema_cross_bull,
        vix            = vix,
        fii_5d_cr      = fii_5d_cr,
        hmm_state      = hmm_state,
        pcr            = pcr,
    )


@st.cache_data(ttl=_TTL)
def cached_rotation_clock_accuracy(selected_date: date, window: int, min_turnover: float):
    """Multi-period (walk-forward) rotation-clock hit-rate + per-phase forward edge."""
    from src.analytics.sector_rotation import get_rotation_clock_accuracy
    try:
        return get_rotation_clock_accuracy(selected_date, window, 40, min_turnover)
    except Exception:
        return {}


@st.cache_data(ttl=_TTL)
def cached_market_scenario(selected_date: date):
    """7-scenario market classification + recommended ranking factor + playbook."""
    from src.analytics.sector_scenario import classify_scenario
    return classify_scenario(selected_date)


@st.cache_data(ttl=_TTL)
def cached_sector_fno_buildup(selected_date: date):
    """Per-sector stock-futures OI buildup (long/short/covering) — F&O confirmation."""
    from src.analytics.sector_scenario import get_sector_fno_buildup
    try:
        return get_sector_fno_buildup(selected_date)
    except Exception:
        import pandas as pd
        return pd.DataFrame()


@st.cache_data(ttl=_TTL)
def cached_sector_overlay(selected_date: date, min_turnover: float):
    """
    get_sector_rotation() sharpened by the memory overlay.

    Adds adj_score / conviction / memory_edge / expected_rs_2w / mem_episodes /
    mem_basis and re-sorts by adj_score. The ~39 per-sector memory lookups run
    once per 5-minute cache window. Falls back to the plain rotation frame on any
    error so the page never hard-fails on the memory layer.
    """
    from src.analytics.sector_rotation import get_sector_rotation, get_market_regime
    from src.analytics.sector_memory import apply_memory_overlay
    rot = get_sector_rotation(selected_date, min_turnover_lacs=min_turnover)
    if rot is None or rot.empty:
        return rot
    try:
        regime = get_market_regime(selected_date)
        return apply_memory_overlay(rot, selected_date, regime)
    except Exception:
        return rot


@st.cache_data(ttl=_TTL)
def cached_next_month_context(as_of: date) -> dict:
    """Conditions monitor for the next 1-4 weeks — base rates + FII positioning
    state. NOT a forecast; see src/analytics/market_context.py.

    Deliberately NOT named cached_market_context: that name is already taken
    above and feeds Index Prediction. Appending a second definition would have
    shadowed it and silently handed Index Prediction the wrong payload.
    """
    from src.analytics.market_context import get_market_context
    return get_market_context(as_of)


@st.cache_data(ttl=_TTL)
def cached_analogues(as_of: date, mode: str = "price") -> dict:
    """Past setups resembling `as_of` and what followed. Descriptive — the
    analogue vote tested null/inverted; see src/analytics/market_context.py.

    `mode`: "price" (price structure, 2013+) or "price+fii" (adds FII
    positioning, 2018+). Neither beats the base rate; see the module note."""
    from src.analytics.market_context import get_analogues
    return get_analogues(as_of, mode=mode)


@st.cache_data(ttl=_TTL)
def cached_fii_only_view(as_of: date) -> dict:
    """FII-only market read. Carries its own measured track record — see
    src/analytics/fii_only.py before treating the stance as a forecast."""
    from src.analytics.fii_only import get_fii_only_view
    return get_fii_only_view(as_of)


@st.cache_data(ttl=_TTL)
def cached_fii_actions(as_of: date) -> dict:
    """What FIIs did per leg — buildup/unwinding/covering, read not inferred."""
    from src.analytics.fii_only import get_fii_actions
    return get_fii_actions(as_of)


@st.cache_data(ttl=_TTL)
def cached_fii_footprint(as_of: date, symbol: str = "NIFTY") -> dict:
    """FII size in the index book + market-wide option walls + rollover."""
    from src.analytics.fii_only import get_fii_footprint
    return get_fii_footprint(as_of, symbol)
