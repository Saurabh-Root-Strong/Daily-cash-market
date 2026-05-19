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
