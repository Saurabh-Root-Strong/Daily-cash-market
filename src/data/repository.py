"""
MarketDataRepository — every read/write against DuckDB lives here.

No SQL string may appear anywhere else in the codebase.
The module exposes a process-level singleton via get_repository(); tests may
replace it with _set_repository(repo) for isolation.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.connection import ConnectionManager

__all__ = [
    "MarketDataRepository",
    "get_repository",
    "_set_repository",
    # Module-level convenience wrappers (keep analytics layer import-compatible)
    "upsert_daily_data",
    "update_delivery_data",
    "upsert_sector_master",
    "log_run",
    "get_latest_trade_date",
    "get_available_dates",
    "get_total_row_count",
    "get_dates_present",
    "query_dataframe",
]


class MarketDataRepository:
    """All DuckDB operations in one place — swap db_path to switch databases."""

    def __init__(self, db_path: str | Path) -> None:
        self._cm = ConnectionManager(db_path)

    @property
    def db_path(self) -> Path:
        return self._cm.db_path

    def raw_connection(self):
        """For Streamlit @st.cache_resource — caller closes."""
        return self._cm.raw_connection()

    # ── Writes ────────────────────────────────────────────────────────────────

    def upsert_daily_data(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        trade_date = df["trade_date"].iloc[0]
        with self._cm.connect() as conn:
            conn.execute("DELETE FROM daily_data WHERE trade_date = ?", [trade_date])
            conn.register("_upsert_df", df)
            conn.execute("INSERT INTO daily_data SELECT * FROM _upsert_df")
        return len(df)

    def update_delivery_data(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        with self._cm.connect() as conn:
            conn.register("_deliv_df", df)
            conn.execute("""
                UPDATE daily_data d
                SET
                    deliv_qty = src.deliv_qty,
                    deliv_per = src.deliv_per
                FROM _deliv_df src
                WHERE d.trade_date = src.trade_date
                  AND d.symbol     = src.symbol
                  AND d.series     = src.series
            """)

    def upsert_sector_master(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        # Ensure all expected columns are present (back-fill optional ones)
        for col, default in [("category", ""), ("market_cap_category", "")]:
            if col not in df.columns:
                df = df.copy()
                df[col] = default
        with self._cm.connect() as conn:
            conn.register("_sector_df", df)
            conn.execute("""
                INSERT INTO sector_master
                    (symbol, company_name, sector, industry, category, market_cap_category, last_updated)
                SELECT
                    symbol, company_name, sector, industry, category, market_cap_category, last_updated
                FROM _sector_df
                ON CONFLICT (symbol) DO UPDATE SET
                    company_name        = excluded.company_name,
                    sector              = excluded.sector,
                    industry            = excluded.industry,
                    category            = excluded.category,
                    market_cap_category = excluded.market_cap_category,
                    last_updated        = excluded.last_updated
            """)

    def log_run(
        self,
        run_type: str,
        trade_date: Optional[datetime.date],
        status: str,
        rows_inserted: int = 0,
        error_message: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> None:
        with self._cm.connect() as conn:
            conn.execute("""
                INSERT INTO run_log
                    (run_type, trade_date, status, rows_inserted, error_message, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [run_type, trade_date, status, rows_inserted, error_message, duration_seconds])

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_latest_trade_date(self) -> Optional[datetime.date]:
        with self._cm.connect() as conn:
            row = conn.execute("SELECT MAX(trade_date) FROM daily_data").fetchone()
        return row[0] if row else None

    def get_available_dates(self, limit: int = 90) -> list[datetime.date]:
        with self._cm.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM daily_data ORDER BY trade_date DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [r[0] for r in rows]

    def get_total_row_count(self) -> int:
        with self._cm.connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()
        return row[0] if row else 0

    def get_dates_present(self, dates: list[datetime.date]) -> set[datetime.date]:
        if not dates:
            return set()
        with self._cm.connect() as conn:
            conn.execute("CREATE TEMP TABLE IF NOT EXISTS _check_dates (d DATE)")
            conn.execute("DELETE FROM _check_dates")
            for d in dates:
                conn.execute("INSERT INTO _check_dates VALUES (?)", [d])
            rows = conn.execute("""
                SELECT DISTINCT trade_date FROM daily_data
                WHERE trade_date IN (SELECT d FROM _check_dates)
            """).fetchall()
        return {r[0] for r in rows}

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self._cm.connect() as conn:
            if params:
                return conn.execute(sql, params).df()
            return conn.execute(sql).df()


# ── Module-level singleton ────────────────────────────────────────────────────

_repository: MarketDataRepository | None = None


def get_repository() -> MarketDataRepository:
    """Return the process-wide repository.  Replace with _set_repository() in tests."""
    global _repository
    if _repository is None:
        from src.core.config import get_config
        _repository = MarketDataRepository(get_config().database.resolved_path)
    return _repository


def _set_repository(repo: MarketDataRepository | None) -> None:
    """Override the singleton — for test isolation only."""
    global _repository
    _repository = repo


# ── Backward-compatible module-level wrappers ─────────────────────────────────
# Analytics and ingestion layers call these directly.

def upsert_daily_data(df: pd.DataFrame) -> int:
    return get_repository().upsert_daily_data(df)


def update_delivery_data(df: pd.DataFrame) -> None:
    return get_repository().update_delivery_data(df)


def upsert_sector_master(df: pd.DataFrame) -> None:
    return get_repository().upsert_sector_master(df)


def log_run(
    run_type: str,
    trade_date: Optional[datetime.date],
    status: str,
    rows_inserted: int = 0,
    error_message: Optional[str] = None,
    duration_seconds: float = 0.0,
) -> None:
    return get_repository().log_run(
        run_type, trade_date, status, rows_inserted, error_message, duration_seconds
    )


def get_latest_trade_date() -> Optional[datetime.date]:
    return get_repository().get_latest_trade_date()


def get_available_dates(limit: int = 90) -> list[datetime.date]:
    return get_repository().get_available_dates(limit)


def get_total_row_count() -> int:
    return get_repository().get_total_row_count()


def get_dates_present(dates: list[datetime.date]) -> set[datetime.date]:
    return get_repository().get_dates_present(dates)


def query_dataframe(sql: str, params: list | None = None) -> pd.DataFrame:
    return get_repository().query(sql, params)
