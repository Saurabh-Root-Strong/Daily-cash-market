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
    "upsert_daily_data",
    "update_delivery_data",
    "upsert_sector_master",
    "upsert_fao_data",
    "upsert_fii_stats",
    "upsert_fno_bhavcopy",
    "upsert_index_data",
    "upsert_fpi_flows",
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

    def upsert_fao_data(self, df: pd.DataFrame) -> int:
        """Insert or replace F&O participant rows for the dates present in df."""
        if df.empty:
            return 0
        dates = df["trade_date"].unique().tolist()
        ph = ", ".join("?" * len(dates))
        with self._cm.connect() as conn:
            conn.execute(f"DELETE FROM fao_participant WHERE trade_date IN ({ph})", dates)
            conn.register("_fao_df", df)
            conn.execute("""
                INSERT INTO fao_participant
                    (trade_date, client_type, data_type,
                     fut_idx_long, fut_idx_short, fut_stk_long, fut_stk_short,
                     opt_idx_call_long, opt_idx_call_short,
                     opt_idx_put_long,  opt_idx_put_short,
                     opt_stk_call_long, opt_stk_call_short,
                     opt_stk_put_long,  opt_stk_put_short,
                     total_long, total_short)
                SELECT
                    trade_date, client_type, data_type,
                    fut_idx_long, fut_idx_short, fut_stk_long, fut_stk_short,
                    opt_idx_call_long, opt_idx_call_short,
                    opt_idx_put_long,  opt_idx_put_short,
                    opt_stk_call_long, opt_stk_call_short,
                    opt_stk_put_long,  opt_stk_put_short,
                    total_long, total_short
                FROM _fao_df
            """)
        return len(df)

    def upsert_fii_stats(self, df: pd.DataFrame) -> int:
        """Insert or replace FII Derivatives Statistics rows for dates present in df."""
        if df.empty:
            return 0
        dates = df["trade_date"].unique().tolist()
        ph = ", ".join("?" * len(dates))
        with self._cm.connect() as conn:
            conn.execute(f"DELETE FROM fii_derivatives_stats WHERE trade_date IN ({ph})", dates)
            conn.register("_fii_stats_df", df)
            conn.execute("""
                INSERT INTO fii_derivatives_stats
                    (trade_date, category, buy_contracts, sell_contracts,
                     buy_value_cr, sell_value_cr, oi_contracts, oi_value_cr)
                SELECT
                    trade_date, category, buy_contracts, sell_contracts,
                    buy_value_cr, sell_value_cr, oi_contracts, oi_value_cr
                FROM _fii_stats_df
            """)
        return len(df)

    def upsert_fno_bhavcopy(self, df: pd.DataFrame) -> int:
        """Insert or replace FNO Bhavcopy rows for dates present in df."""
        if df.empty:
            return 0
        # NSE source file can have negative OI (roll adjustments); clip to 0
        if "open_interest" in df.columns:
            df = df.copy()
            df["open_interest"] = df["open_interest"].clip(lower=0)
        dates = df["trade_date"].unique().tolist()
        ph = ", ".join("?" * len(dates))
        with self._cm.connect() as conn:
            conn.execute(f"DELETE FROM fno_bhavcopy WHERE trade_date IN ({ph})", dates)
            conn.register("_fno_df", df)
            conn.execute("""
                INSERT INTO fno_bhavcopy
                    (trade_date, instrument, symbol, expiry_date,
                     strike_price, option_type,
                     open_price, high_price, low_price, close_price, settle_price,
                     contracts, value_lacs, open_interest, chg_in_oi)
                SELECT
                    trade_date, instrument, symbol, expiry_date,
                    strike_price, option_type,
                    open_price, high_price, low_price, close_price, settle_price,
                    contracts, value_lacs, open_interest, chg_in_oi
                FROM _fno_df
            """)
        return len(df)

    def upsert_fpi_flows(self, df: pd.DataFrame) -> int:
        """Upsert FPI NSDL flow rows. Overwrites any existing rows for the same (trade_date, category)."""
        if df.empty:
            return 0
        dates = df["trade_date"].unique().tolist()
        ph = ", ".join("?" * len(dates))
        with self._cm.connect() as conn:
            conn.execute(f"DELETE FROM fpi_nsdl_flows WHERE trade_date IN ({ph})", dates)
            conn.register("_fpi_df", df)
            conn.execute("""
                INSERT INTO fpi_nsdl_flows
                    (trade_date, category, gross_purchase_cr, gross_sales_cr, net_investment_cr)
                SELECT
                    trade_date, category, gross_purchase_cr, gross_sales_cr, net_investment_cr
                FROM _fpi_df
            """)
        return len(df)

    def upsert_index_data(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        with self._cm.connect() as conn:
            conn.register("_idx_df", df)
            conn.execute("""
                INSERT INTO index_data
                    (trade_date, index_name, open_val, high_val, low_val, close_val,
                     prev_close, points_chg, pct_chg, volume, turnover_cr,
                     pe_ratio, pb_ratio, div_yield)
                SELECT
                    trade_date, index_name, open_val, high_val, low_val, close_val,
                    prev_close, points_chg, pct_chg, volume, turnover_cr,
                    pe_ratio, pb_ratio, div_yield
                FROM _idx_df
                ON CONFLICT (trade_date, index_name) DO UPDATE SET
                    open_val    = excluded.open_val,
                    high_val    = excluded.high_val,
                    low_val     = excluded.low_val,
                    close_val   = excluded.close_val,
                    prev_close  = excluded.prev_close,
                    points_chg  = excluded.points_chg,
                    pct_chg     = excluded.pct_chg,
                    volume      = excluded.volume,
                    turnover_cr = excluded.turnover_cr,
                    pe_ratio    = excluded.pe_ratio,
                    pb_ratio    = excluded.pb_ratio,
                    div_yield   = excluded.div_yield
            """)
        return len(df)

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
        # GROUP BY instead of DISTINCT: DuckDB DISTINCT+ORDER BY+LIMIT mis-fires
        with self._cm.connect() as conn:
            rows = conn.execute(
                "SELECT trade_date FROM daily_data GROUP BY trade_date ORDER BY trade_date DESC LIMIT ?",
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
        ph = ", ".join("?" * len(dates))
        with self._cm.connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT trade_date FROM daily_data WHERE trade_date IN ({ph})",
                dates,
            ).fetchall()
        return {r[0] for r in rows}

    def upsert_prediction(self, row: dict) -> None:
        """Insert or replace one prediction_log row (DELETE + INSERT — DuckDB safe)."""
        with self._cm.connect() as conn:
            conn.execute(
                "DELETE FROM prediction_log WHERE trade_date = ? AND fno_symbol = ?",
                [row["trade_date"], row["fno_symbol"]],
            )
            conn.execute("""
                INSERT INTO prediction_log (
                    trade_date, fno_symbol,
                    direction_pred, confidence_pred, composite_score, signal_count,
                    feat_pcr, feat_max_pain_dist, feat_carry,
                    feat_fii_net, feat_fii_5d_cumul, feat_fii_delta,
                    feat_vix, feat_vix_5d_chg, feat_breadth,
                    feat_hurst, feat_entropy, feat_oi_score,
                    hmm_state, memory_label,
                    actual_return, direction_actual, was_correct,
                    outcome_filled, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,now())
            """, [
                row["trade_date"], row["fno_symbol"],
                row.get("direction_pred"), row.get("confidence_pred"),
                row.get("composite_score"), row.get("signal_count"),
                row.get("feat_pcr"),          row.get("feat_max_pain_dist"),
                row.get("feat_carry"),
                row.get("feat_fii_net"),      row.get("feat_fii_5d_cumul"),
                row.get("feat_fii_delta"),
                row.get("feat_vix"),          row.get("feat_vix_5d_chg"),
                row.get("feat_breadth"),
                row.get("feat_hurst"),        row.get("feat_entropy"),
                row.get("feat_oi_score"),
                row.get("hmm_state"),         row.get("memory_label"),
                row.get("actual_return"),     row.get("direction_actual"),
                row.get("was_correct"),       row.get("outcome_filled", False),
            ])

    def get_unfilled_predictions(self) -> "pd.DataFrame":
        return self.query(
            "SELECT * FROM prediction_log WHERE outcome_filled = FALSE ORDER BY trade_date"
        )

    def fill_prediction_outcome(
        self,
        trade_date: datetime.date,
        fno_symbol: str,
        actual_return: float,
        direction_actual: str,
        was_correct: bool,
    ) -> None:
        with self._cm.connect() as conn:
            conn.execute("""
                UPDATE prediction_log
                SET actual_return = ?, direction_actual = ?,
                    was_correct = ?, outcome_filled = TRUE
                WHERE trade_date = ? AND fno_symbol = ?
            """, [actual_return, direction_actual, was_correct, trade_date, fno_symbol])

    def get_filled_predictions(self, fno_symbol: str, limit: int = 500) -> "pd.DataFrame":
        return self.query("""
            SELECT * FROM prediction_log
            WHERE fno_symbol = ? AND outcome_filled = TRUE
            ORDER BY trade_date DESC
            LIMIT ?
        """, [fno_symbol, limit])

    def get_distinct_dates(self, table: str) -> set[datetime.date]:
        """All distinct trade_dates present in any table. Used for gap detection."""
        with self._cm.connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT trade_date FROM {table}"
            ).fetchall()
        return {r[0].date() if hasattr(r[0], "date") else r[0] for r in rows}

    def execute_ddl(self, *statements: str) -> None:
        """Execute DDL statements (CREATE/ALTER TABLE, CREATE INDEX) in one connection."""
        with self._cm.connect() as conn:
            for sql in statements:
                conn.execute(sql)

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


def upsert_fao_data(df: pd.DataFrame) -> int:
    return get_repository().upsert_fao_data(df)


def upsert_fii_stats(df: pd.DataFrame) -> int:
    return get_repository().upsert_fii_stats(df)


def upsert_fno_bhavcopy(df: pd.DataFrame) -> int:
    return get_repository().upsert_fno_bhavcopy(df)


def upsert_fpi_flows(df: pd.DataFrame) -> int:
    return get_repository().upsert_fpi_flows(df)


def upsert_index_data(df: pd.DataFrame) -> int:
    return get_repository().upsert_index_data(df)


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
