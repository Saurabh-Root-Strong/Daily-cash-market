"""
Shared pytest fixtures.

temp_db — spins up an isolated in-memory-style DuckDB for each test function.
Uses _set_repository() so every layer that calls get_repository() receives
the test database automatically — no monkeypatching of config paths needed.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.repository import MarketDataRepository, _set_repository
from src.data.schema import initialize_schema


@pytest.fixture
def temp_db():
    """
    Isolated DuckDB for one test.

    Yields the path string so tests can inspect the file if needed.
    Cleans up the .duckdb and .wal files on exit.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False)
    tmp.close()
    os.unlink(tmp.name)  # DuckDB refuses to open a zero-byte file

    repo = MarketDataRepository(tmp.name)
    _set_repository(repo)
    initialize_schema()

    yield tmp.name

    _set_repository(None)  # restore global singleton to None

    for ext in ("", ".wal"):
        p = Path(tmp.name + ext)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def insert_ohlcv_history(
    start: date,
    symbols: list[tuple[str, str, float]],  # (symbol, sector, base_deliv_pct)
    days: int = 11,
) -> date:
    """
    Insert `days` days of synthetic OHLCV + sector_master rows.

    Returns the latest trade_date inserted (skip weekends).
    Useful in multiple test modules — import from conftest via fixture or directly.
    """
    from src.data.repository import upsert_daily_data, update_delivery_data, upsert_sector_master

    all_rows: list[dict] = []
    d = start
    inserted_dates: list[date] = []

    for i in range(days * 2):  # overshoot to get enough weekdays
        if len(inserted_dates) >= days:
            break
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        for sym, _sector, base_deliv in symbols:
            all_rows.append({
                "trade_date":  d,
                "symbol":      sym,
                "series":      "EQ",
                "prev_close":  2800.0 + i * 10,
                "open_price":  2800.0,
                "high_price":  2850.0,
                "low_price":   2780.0,
                "last_price":  2830.0 + i * 5,
                "close_price": 2830.0 + i * 5,
                "avg_price":   2820.0,
                "ttl_trd_qnty": 1_000_000,
                "turnover_lacs": 2830.0,
                "no_of_trades":  50_000,
                "deliv_qty":   int(base_deliv * 10_000),
                "deliv_per":   base_deliv + i * 0.5,
            })
        inserted_dates.append(d)
        d += timedelta(days=1)

    df = pd.DataFrame(all_rows)
    for td in df["trade_date"].unique():
        upsert_daily_data(df[df["trade_date"] == td].copy())

    # Sector master
    sectors = pd.DataFrame({
        "symbol":             [s[0] for s in symbols],
        "company_name":       [f"{s[0]} Ltd" for s in symbols],
        "sector":             [s[1] for s in symbols],
        "industry":           [s[1] for s in symbols],
        "market_cap_category": "Large",
        "last_updated":       datetime.now(),
    })
    upsert_sector_master(sectors)

    return max(inserted_dates)
