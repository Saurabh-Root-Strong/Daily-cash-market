"""
Database DDL — idempotent, safe to run repeatedly.

Call initialize_schema() once at startup or after init-db.
All table/index definitions live here; no DDL anywhere else.
"""
from __future__ import annotations

from src.data.repository import get_repository
from src.core.logging import get_logger

__all__ = ["initialize_schema"]

log = get_logger(__name__)


def initialize_schema() -> None:
    """Create all tables, sequences, and indexes if they do not exist."""
    repo = get_repository()
    with repo._cm.connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_data (
                trade_date        DATE       NOT NULL,
                symbol            VARCHAR    NOT NULL,
                series            VARCHAR    NOT NULL,
                prev_close        DOUBLE,
                open_price        DOUBLE,
                high_price        DOUBLE,
                low_price         DOUBLE,
                last_price        DOUBLE,
                close_price       DOUBLE,
                avg_price         DOUBLE,
                ttl_trd_qnty      BIGINT,
                turnover_lacs     DOUBLE,
                no_of_trades      BIGINT,
                deliv_qty         BIGINT,
                deliv_per         DOUBLE,
                PRIMARY KEY (trade_date, symbol, series)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sector_master (
                symbol              VARCHAR PRIMARY KEY,
                company_name        VARCHAR,
                sector              VARCHAR,
                industry            VARCHAR,
                category            VARCHAR,
                market_cap_category VARCHAR,
                last_updated        TIMESTAMP
            )
        """)

        # Non-destructive migration: add category column to existing databases
        try:
            conn.execute(
                "ALTER TABLE sector_master ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT ''"
            )
        except Exception as exc:
            log.debug("sector_master.category migration skipped (already applied): %s", exc)

        conn.execute("CREATE SEQUENCE IF NOT EXISTS run_log_seq START 1")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_log (
                run_id           BIGINT    PRIMARY KEY DEFAULT nextval('run_log_seq'),
                run_timestamp    TIMESTAMP DEFAULT now(),
                run_type         VARCHAR,
                trade_date       DATE,
                status           VARCHAR,
                rows_inserted    INTEGER,
                error_message    VARCHAR,
                duration_seconds DOUBLE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_data (
                trade_date   DATE    NOT NULL,
                index_name   VARCHAR NOT NULL,
                open_val     DOUBLE,
                high_val     DOUBLE,
                low_val      DOUBLE,
                close_val    DOUBLE,
                prev_close   DOUBLE,
                points_chg   DOUBLE,
                pct_chg      DOUBLE,
                volume       BIGINT,
                turnover_cr  DOUBLE,
                pe_ratio     DOUBLE,
                pb_ratio     DOUBLE,
                div_yield    DOUBLE,
                PRIMARY KEY (trade_date, index_name)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS fao_participant (
                trade_date         DATE    NOT NULL,
                client_type        VARCHAR NOT NULL,
                data_type          VARCHAR NOT NULL,
                fut_idx_long       BIGINT  DEFAULT 0,
                fut_idx_short      BIGINT  DEFAULT 0,
                fut_stk_long       BIGINT  DEFAULT 0,
                fut_stk_short      BIGINT  DEFAULT 0,
                opt_idx_call_long  BIGINT  DEFAULT 0,
                opt_idx_call_short BIGINT  DEFAULT 0,
                opt_idx_put_long   BIGINT  DEFAULT 0,
                opt_idx_put_short  BIGINT  DEFAULT 0,
                opt_stk_call_long  BIGINT  DEFAULT 0,
                opt_stk_call_short BIGINT  DEFAULT 0,
                opt_stk_put_long   BIGINT  DEFAULT 0,
                opt_stk_put_short  BIGINT  DEFAULT 0,
                total_long         BIGINT  DEFAULT 0,
                total_short        BIGINT  DEFAULT 0,
                PRIMARY KEY (trade_date, client_type, data_type)
            )
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_date        ON daily_data(trade_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_symbol      ON daily_data(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_date_symbol ON daily_data(trade_date, symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sm_sector      ON sector_master(sector)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_id_date        ON index_data(trade_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_id_name        ON index_data(index_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fao_date       ON fao_participant(trade_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fao_type       ON fao_participant(client_type)")

        # FII Derivatives Statistics — buy/sell value by contract type (Index/Stock F&O)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fii_derivatives_stats (
                trade_date      DATE    NOT NULL,
                category        VARCHAR NOT NULL,
                buy_contracts   BIGINT  DEFAULT 0,
                sell_contracts  BIGINT  DEFAULT 0,
                buy_value_cr    DOUBLE  DEFAULT 0,
                sell_value_cr   DOUBLE  DEFAULT 0,
                oi_contracts    BIGINT  DEFAULT 0,
                oi_value_cr     DOUBLE  DEFAULT 0,
                PRIMARY KEY (trade_date, category)
            )
        """)
        # Non-destructive migration: add oi_value_cr if table pre-existed
        try:
            conn.execute(
                "ALTER TABLE fii_derivatives_stats ADD COLUMN IF NOT EXISTS oi_value_cr DOUBLE DEFAULT 0"
            )
        except Exception as exc:
            log.debug("fii_derivatives_stats.oi_value_cr migration skipped (already applied): %s", exc)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fiis_date ON fii_derivatives_stats(trade_date)")

        # F&O Bhavcopy — per-instrument snapshot (futures + options, index + stock)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fno_bhavcopy (
                trade_date    DATE    NOT NULL,
                instrument    VARCHAR NOT NULL,
                symbol        VARCHAR NOT NULL,
                expiry_date   DATE    NOT NULL,
                strike_price  DOUBLE  DEFAULT 0,
                option_type   VARCHAR DEFAULT 'XX',
                open_price    DOUBLE,
                high_price    DOUBLE,
                low_price     DOUBLE,
                close_price   DOUBLE,
                settle_price  DOUBLE,
                contracts     BIGINT  DEFAULT 0,
                value_lacs    DOUBLE  DEFAULT 0,
                open_interest BIGINT  DEFAULT 0,
                chg_in_oi     BIGINT  DEFAULT 0,
                PRIMARY KEY (trade_date, instrument, symbol, expiry_date, strike_price, option_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fnob_date       ON fno_bhavcopy(trade_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fnob_symbol     ON fno_bhavcopy(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fnob_expiry     ON fno_bhavcopy(expiry_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fnob_instrument ON fno_bhavcopy(instrument)")
