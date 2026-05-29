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

    # ── Core tables ───────────────────────────────────────────────────────────
    repo.execute_ddl(
        """
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
        """,
        """
        CREATE TABLE IF NOT EXISTS sector_master (
            symbol              VARCHAR PRIMARY KEY,
            company_name        VARCHAR,
            sector              VARCHAR,
            industry            VARCHAR,
            category            VARCHAR,
            market_cap_category VARCHAR,
            last_updated        TIMESTAMP
        )
        """,
        "CREATE SEQUENCE IF NOT EXISTS run_log_seq START 1",
        """
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
        """,
        """
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
        """,
        """
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
        """,
        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_dd_date        ON daily_data(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_dd_symbol      ON daily_data(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_dd_date_symbol ON daily_data(trade_date, symbol)",
        "CREATE INDEX IF NOT EXISTS idx_sm_sector      ON sector_master(sector)",
        "CREATE INDEX IF NOT EXISTS idx_id_date        ON index_data(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_id_name        ON index_data(index_name)",
        "CREATE INDEX IF NOT EXISTS idx_fao_date       ON fao_participant(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_fao_type       ON fao_participant(client_type)",
        # FII Derivatives Statistics
        """
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
        """,
        "CREATE INDEX IF NOT EXISTS idx_fiis_date ON fii_derivatives_stats(trade_date)",
        # F&O Bhavcopy
        """
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
        """,
        "CREATE INDEX IF NOT EXISTS idx_fnob_date       ON fno_bhavcopy(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_fnob_symbol     ON fno_bhavcopy(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_fnob_expiry     ON fno_bhavcopy(expiry_date)",
        "CREATE INDEX IF NOT EXISTS idx_fnob_instrument ON fno_bhavcopy(instrument)",
        # FPI NSDL Capital Flows
        """
        CREATE TABLE IF NOT EXISTS fpi_nsdl_flows (
            trade_date        DATE    NOT NULL,
            category          VARCHAR NOT NULL,
            gross_purchase_cr DOUBLE  DEFAULT 0,
            gross_sales_cr    DOUBLE  DEFAULT 0,
            net_investment_cr DOUBLE  DEFAULT 0,
            PRIMARY KEY (trade_date, category)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_fpi_date ON fpi_nsdl_flows(trade_date)",
        # ── Prediction Memory Log ─────────────────────────────────────────────
        # Stores every prediction made + actual outcome filled next trading day.
        # Feature vector enables similarity search: find past days that looked
        # like today and check what actually happened.
        """
        CREATE TABLE IF NOT EXISTS prediction_log (
            trade_date       DATE    NOT NULL,
            fno_symbol       VARCHAR NOT NULL,
            -- Prediction (made at EOD on trade_date)
            direction_pred   VARCHAR,
            confidence_pred  VARCHAR,
            composite_score  DOUBLE,
            signal_count     INTEGER,
            -- Feature vector (8-dim, normalised [0,1]) for similarity search
            -- 12-dimensional fingerprint (all raw, normalised at query time)
            feat_pcr             DOUBLE,
            feat_max_pain_dist   DOUBLE,
            feat_carry           DOUBLE,
            feat_fii_net         DOUBLE,
            feat_fii_5d_cumul    DOUBLE,
            feat_fii_delta       DOUBLE,
            feat_vix             DOUBLE,
            feat_vix_5d_chg      DOUBLE,
            feat_breadth         DOUBLE,
            feat_hurst           DOUBLE,
            feat_entropy         DOUBLE,
            feat_oi_score        DOUBLE,
            -- Context labels (for regime-conditional accuracy)
            hmm_state        VARCHAR,
            memory_label     VARCHAR,
            -- Actual outcome (filled T+1 after next day's data arrives)
            actual_return    DOUBLE,
            direction_actual VARCHAR,
            was_correct      BOOLEAN,
            outcome_filled   BOOLEAN DEFAULT FALSE,
            created_at       TIMESTAMP DEFAULT now(),
            PRIMARY KEY (trade_date, fno_symbol)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_plog_date    ON prediction_log(trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_plog_symbol  ON prediction_log(fno_symbol)",
        "CREATE INDEX IF NOT EXISTS idx_plog_outcome ON prediction_log(outcome_filled)",
    )

    # ── Non-destructive migrations (ALTER TABLE — may already exist) ───────────
    try:
        repo.execute_ddl(
            "ALTER TABLE sector_master ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT ''"
        )
    except Exception as exc:
        log.debug("sector_master.category migration skipped (already applied): %s", exc)

    try:
        repo.execute_ddl(
            "ALTER TABLE fii_derivatives_stats ADD COLUMN IF NOT EXISTS oi_value_cr DOUBLE DEFAULT 0"
        )
    except Exception as exc:
        log.debug("fii_derivatives_stats.oi_value_cr migration skipped (already applied): %s", exc)

    # prediction_log — add 4 new feature columns (may not exist in older DBs)
    for col in [
        "feat_max_pain_dist DOUBLE DEFAULT 0",
        "feat_fii_5d_cumul  DOUBLE DEFAULT 0",
        "feat_fii_delta     DOUBLE DEFAULT 0",
        "feat_vix_5d_chg    DOUBLE DEFAULT 0",
    ]:
        try:
            repo.execute_ddl(f"ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS {col}")
        except Exception as exc:
            log.debug("prediction_log.%s migration skipped: %s", col.split()[0], exc)
