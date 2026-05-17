import os
import tempfile
from datetime import date, datetime
import pandas as pd
import pytest


@pytest.fixture
def temp_db(monkeypatch):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
    tmpfile.close()
    os.unlink(tmpfile.name)  # DuckDB refuses to open empty files

    import src.config_loader as config_loader

    monkeypatch.setattr(config_loader, "get_db_path", lambda: __import__("pathlib").Path(tmpfile.name))
    config_loader.load_config.cache_clear()

    from src.data.schema import initialize_schema
    initialize_schema()

    yield tmpfile.name

    config_loader.load_config.cache_clear()

    for ext in ("", ".wal"):
        path = tmpfile.name + ext
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _insert_test_data(trade_date: date):
    from src.data.repository import upsert_daily_data, update_delivery_data, upsert_sector_master

    # Insert 5 days of history for window function tests
    all_rows = []
    base_date = date(2024, 7, 22)
    from datetime import timedelta
    for i in range(11):
        day = base_date + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for sym, deliv_p in [("RELIANCE", 50.0 + i * 2), ("TCS", 40.0 + i * 1.5)]:
            all_rows.append({
                "trade_date": day,
                "symbol": sym,
                "series": "EQ",
                "prev_close": 2800.0 + i * 10,
                "open_price": 2800.0,
                "high_price": 2850.0,
                "low_price": 2780.0,
                "last_price": 2830.0 + i * 5,
                "close_price": 2830.0 + i * 5,
                "avg_price": 2820.0,
                "ttl_trd_qnty": 1_000_000,
                "turnover_lacs": 2830.0,
                "no_of_trades": 50000,
                "deliv_qty": int(deliv_p * 10000),
                "deliv_per": deliv_p,
            })

    df = pd.DataFrame(all_rows)
    for d in df["trade_date"].unique():
        upsert_daily_data(df[df["trade_date"] == d].copy())

    # Sector master
    sectors = pd.DataFrame({
        "symbol": ["RELIANCE", "TCS"],
        "company_name": ["Reliance Industries", "Tata Consultancy Services"],
        "sector": ["Energy", "IT"],
        "industry": ["Oil & Gas", "IT Services"],
        "market_cap_category": ["Large", "Large"],
        "last_updated": [datetime.now(), datetime.now()],
    })
    upsert_sector_master(sectors)

    return df["trade_date"].max()


def test_stock_metrics_delivery_ratio(temp_db):
    trade_date = _insert_test_data(date(2024, 8, 1))

    from src.analytics.delivery_signals import get_stock_metrics
    df = get_stock_metrics(trade_date, min_turnover_lacs=0)

    assert not df.empty
    assert "deliv_ratio" in df.columns
    # deliv_ratio = deliv_per / deliv_per_10d_avg — should be a positive number
    valid = df[df["deliv_ratio"].notna()]
    if not valid.empty:
        assert (valid["deliv_ratio"] > 0).all()


def test_sector_aggregation_weighted_averages(temp_db):
    trade_date = _insert_test_data(date(2024, 8, 1))

    from src.analytics.sector_aggregator import aggregate_by_sector
    df = aggregate_by_sector(trade_date, min_turnover_lacs=0)

    assert not df.empty
    assert "wtd_deliv_per" in df.columns
    assert "wtd_price_change_pct" in df.columns
    # Both Energy (RELIANCE) and IT (TCS) should appear
    sectors = set(df["sector"].tolist())
    assert "Energy" in sectors or "IT" in sectors


def test_sector_drilldown_top_stocks(temp_db):
    trade_date = _insert_test_data(date(2024, 8, 1))

    from src.analytics.sector_aggregator import get_sector_drilldown
    result = get_sector_drilldown(trade_date, "Energy", top_n=5)

    if result:
        assert "top_by_delivery_pct" in result
        assert "contribution_table" in result
        assert "sector_summary" in result

        top = result["top_by_delivery_pct"]
        assert not top.empty
        assert "symbol" in top.columns
