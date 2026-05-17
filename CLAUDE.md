# NSE Daily Cash Market Dashboard — Claude Code Context

## What This Project Does

A local dashboard for daily NSE (National Stock Exchange of India) equity cash market analysis.
- Auto-fetches NSE bhavcopy + delivery data daily at 6:30 PM IST via Windows Task Scheduler
- Stores 60+ days of historical data in DuckDB
- Computes sector-level metrics: turnover-weighted average price change % and delivery % per sector
- Streamlit dashboard with sector overview, drill-down, stock detail, and accumulation/distribution signals

## 4-Layer Architecture — NEVER bypass layers

```
dashboard/  -->  analytics/  -->  data/  -->  ingestion/
(Streamlit)      (signals)        (DuckDB)     (NSE HTTP)
```

Each layer imports ONLY from the layer directly below it.

| Layer | Location | Allowed to import |
|-------|----------|-------------------|
| Dashboard | `src/dashboard/` | `src/analytics/` only |
| Analytics | `src/analytics/` | `src/data/` only |
| Data | `src/data/` | nothing internal |
| Ingestion | `src/ingestion/` | `src/data/` only |

**Things that break if you bypass layers:**
- Dashboard importing from `src/data/` directly → SQL leaks into UI, impossible to test
- Analytics importing from `src/ingestion/` → circular deps, UI becomes network-coupled
- Ingestion importing from `src/analytics/` → circular deps
- SQL outside `src/data/` → schema changes cascade unpredictably

## Key Design Decisions

**DuckDB** (not SQLite): window functions (AVG OVER PARTITION BY) are first-class. Required for rolling delivery averages.

**UDiFF bhavcopy format** (post-July 2024): URL pattern `BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip`. Columns use abbreviated names (TckrSymb, TtlTrfVal, etc.). TtlTrfVal is in rupees — divide by 100,000 for lakhs.

**Separate MTO delivery file**: UDiFF does NOT include delivery data. Fetch `MTO_DDMMYYYY.DAT` separately and UPDATE the already-inserted bhavcopy rows.

**Equity series filter**: Keep only `SctySrs IN ('EQ','SM','ST')` and `Sgmt == 'CM'`. BE (ETFs/BEES) and BZ (bonds) are excluded.

**Turnover-weighted defaults**: Sector metrics default to turnover-weighted averages (not simple). Simple averages are computed alongside and shown in dashboard.

**Cookie priming**: NSE blocks direct requests. NSEClient visits `nseindia.com/` and `/all-reports` to prime cookies before any data fetch.

## Database Schema

```sql
daily_data (
    trade_date DATE, symbol VARCHAR, series VARCHAR,
    prev_close DOUBLE, open_price DOUBLE, high_price DOUBLE,
    low_price DOUBLE, last_price DOUBLE, close_price DOUBLE,
    avg_price DOUBLE, ttl_trd_qnty BIGINT, turnover_lacs DOUBLE,
    no_of_trades BIGINT, deliv_qty BIGINT, deliv_per DOUBLE,
    PRIMARY KEY (trade_date, symbol, series)
)

sector_master (
    symbol VARCHAR PRIMARY KEY, company_name VARCHAR,
    sector VARCHAR, industry VARCHAR, market_cap_category VARCHAR,
    last_updated TIMESTAMP
)

run_log (
    run_id BIGINT PRIMARY KEY, run_timestamp TIMESTAMP,
    run_type VARCHAR, trade_date DATE, status VARCHAR,
    rows_inserted INTEGER, error_message VARCHAR, duration_seconds DOUBLE
)
```

## Common Tasks

### Add a new signal
1. Add computation in `src/analytics/delivery_signals.py` (pure pandas/SQL, no I/O)
2. Call it from `src/dashboard/pages/signals.py`
3. Add a test in `tests/test_analytics.py`

### Handle NSE URL changes
Edit ONLY `src/ingestion/bhavcopy_fetcher.py` or `src/ingestion/delivery_fetcher.py`. URL strings must never appear elsewhere.

### Add a new chart
Add a pure Plotly function to `src/dashboard/components/charts.py`, call it from the relevant page module.

### Add a new sector index
Add an entry to `INDEX_DEFINITIONS` in `src/ingestion/sector_fetcher.py`, then run `python -m src.cli seed-sectors`.

### Add a column to daily_data
1. Update `src/data/schema.py` `initialize_schema()`
2. Update `src/ingestion/bhavcopy_fetcher.py` `transform_to_schema()`
3. Update `src/data/repository.py` upsert logic
4. Drop and recreate the DB: `python -m src.cli init-db` then `backfill 60`

## Running Tests

```bash
pytest tests/ -v
```

All 7 tests must pass. Common failures:
- `BinderException: Values list "b" does not have a column named "deliv_ratio"` → use the full expression in ORDER BY, not the alias
- `IO Error: not a valid DuckDB database file` → temp_db fixture must os.unlink() empty tempfile before DuckDB opens it
- `FileNotFoundError: Config file not found` → PROJECT_ROOT = Path(__file__).resolve().parent.parent (from src/)

## CLI Commands

```bash
python -m src.cli init-db           # Create/reset schema
python -m src.cli backfill [days]   # Backfill N trading days (default 60)
python -m src.cli daily             # Fetch today (or most recent available)
python -m src.cli seed-sectors      # Fetch sector constituents from NSE
python -m src.cli reload-overrides  # Apply sector_overrides.csv without re-fetching
```
