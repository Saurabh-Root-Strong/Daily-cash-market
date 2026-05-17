from src.data.connection import get_connection


def initialize_schema() -> None:
    with get_connection() as conn:
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
                symbol             VARCHAR PRIMARY KEY,
                company_name       VARCHAR,
                sector             VARCHAR,
                industry           VARCHAR,
                market_cap_category VARCHAR,
                last_updated       TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS run_log_seq START 1
        """)

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

        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_date ON daily_data(trade_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_symbol ON daily_data(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dd_date_symbol ON daily_data(trade_date, symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sm_sector ON sector_master(sector)")
