from datetime import date, datetime
from typing import Optional, Set, List
import pandas as pd

from src.data.connection import get_connection


def upsert_daily_data(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    trade_date = df["trade_date"].iloc[0]
    with get_connection() as conn:
        conn.execute("DELETE FROM daily_data WHERE trade_date = ?", [trade_date])
        conn.register("_upsert_df", df)
        conn.execute("INSERT INTO daily_data SELECT * FROM _upsert_df")
        return len(df)


def update_delivery_data(df: pd.DataFrame) -> None:
    if df.empty:
        return
    with get_connection() as conn:
        conn.register("_deliv_df", df)
        conn.execute("""
            UPDATE daily_data d
            SET
                deliv_qty = src.deliv_qty,
                deliv_per = src.deliv_per
            FROM _deliv_df src
            WHERE d.trade_date = src.trade_date
              AND d.symbol = src.symbol
              AND d.series = src.series
        """)


def upsert_sector_master(df: pd.DataFrame) -> None:
    if df.empty:
        return
    with get_connection() as conn:
        conn.register("_sector_df", df)
        conn.execute("""
            INSERT INTO sector_master
            SELECT * FROM _sector_df
            ON CONFLICT (symbol) DO UPDATE SET
                company_name = excluded.company_name,
                sector = excluded.sector,
                industry = excluded.industry,
                market_cap_category = excluded.market_cap_category,
                last_updated = excluded.last_updated
        """)


def log_run(
    run_type: str,
    trade_date: Optional[date],
    status: str,
    rows_inserted: int = 0,
    error_message: Optional[str] = None,
    duration_seconds: float = 0.0,
) -> None:
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO run_log (run_type, trade_date, status, rows_inserted, error_message, duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [run_type, trade_date, status, rows_inserted, error_message, duration_seconds])


def get_latest_trade_date() -> Optional[date]:
    with get_connection() as conn:
        result = conn.execute("SELECT MAX(trade_date) FROM daily_data").fetchone()
        return result[0] if result else None


def get_available_dates(limit: int = 90) -> List[date]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT trade_date FROM daily_data ORDER BY trade_date DESC LIMIT ?",
            [limit]
        ).fetchall()
        return [r[0] for r in rows]


def get_total_row_count() -> int:
    with get_connection() as conn:
        result = conn.execute("SELECT COUNT(*) FROM daily_data").fetchone()
        return result[0] if result else 0


def get_dates_present(dates_list: List[date]) -> Set[date]:
    if not dates_list:
        return set()
    with get_connection() as conn:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _check_dates (d DATE)")
        conn.execute("DELETE FROM _check_dates")
        for d in dates_list:
            conn.execute("INSERT INTO _check_dates VALUES (?)", [d])
        rows = conn.execute("""
            SELECT DISTINCT trade_date FROM daily_data
            WHERE trade_date IN (SELECT d FROM _check_dates)
        """).fetchall()
        return {r[0] for r in rows}


def query_dataframe(sql: str, params: Optional[list] = None) -> pd.DataFrame:
    with get_connection() as conn:
        if params:
            return conn.execute(sql, params).df()
        return conn.execute(sql).df()
