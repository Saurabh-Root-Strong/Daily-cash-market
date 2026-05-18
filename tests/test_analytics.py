"""
Analytics layer tests — run against an isolated temp DuckDB via the temp_db fixture.

All tests are independent; each gets a fresh database.
"""
from __future__ import annotations

from datetime import date

from tests.conftest import insert_ohlcv_history

_SYMBOLS = [
    ("RELIANCE", "Energy",  50.0),
    ("TCS",      "IT",      40.0),
]


def test_stock_metrics_delivery_ratio(temp_db):
    trade_date = insert_ohlcv_history(date(2024, 7, 22), _SYMBOLS)

    from src.analytics.delivery_signals import get_stock_metrics
    df = get_stock_metrics(trade_date, min_turnover_lacs=0)

    assert not df.empty, "get_stock_metrics returned empty DataFrame"
    assert "deliv_ratio" in df.columns

    valid = df[df["deliv_ratio"].notna()]
    if not valid.empty:
        assert (valid["deliv_ratio"] > 0).all(), "deliv_ratio must be positive"


def test_sector_aggregation_weighted_averages(temp_db):
    trade_date = insert_ohlcv_history(date(2024, 7, 22), _SYMBOLS)

    from src.analytics.sector_aggregator import aggregate_by_sector
    df = aggregate_by_sector(trade_date, min_turnover_lacs=0)

    assert not df.empty, "aggregate_by_sector returned empty DataFrame"
    assert "wtd_deliv_per" in df.columns
    assert "wtd_price_change_pct" in df.columns
    assert {"Energy", "IT"}.intersection(set(df["sector"]))


def test_sector_drilldown_top_stocks(temp_db):
    trade_date = insert_ohlcv_history(date(2024, 7, 22), _SYMBOLS)

    from src.analytics.sector_aggregator import get_sector_drilldown
    result = get_sector_drilldown(trade_date, "Energy", top_n=5)

    assert result, "get_sector_drilldown returned empty/None"
    assert "top_by_delivery_pct" in result
    assert "contribution_table" in result
    assert "sector_summary" in result
    assert not result["top_by_delivery_pct"].empty
    assert "symbol" in result["top_by_delivery_pct"].columns
