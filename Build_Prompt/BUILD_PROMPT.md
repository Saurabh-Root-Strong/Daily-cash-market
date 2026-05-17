# Master Prompt for Claude Code

If you'd rather have Claude Code generate this project from scratch in your VS Code
(instead of unzipping the prebuilt one), paste the prompt below into a fresh
Claude Code session in an empty folder.

---

## THE PROMPT — copy everything between the horizontal lines below into Claude Code

---

I want you to build a complete, production-quality NSE (National Stock Exchange of India) equity cash market dashboard. This will be a long-term project, so structure matters more than speed. Build it in the current empty folder.

## Project Overview

A local dashboard for daily NSE equity-cash analysis with these features:
1. Auto-fetches NSE bhavcopy + delivery data daily at 6:30 PM IST via Windows Task Scheduler
2. Stores 60+ days of historical data in DuckDB
3. Computes sector-level metrics: turnover-weighted average price change % and delivery % per sector
4. Streamlit dashboard with: sector overview (dual-axis chart), drill-down (click sector → see top-delivery stocks), individual stock detail, and accumulation/distribution signals
5. Runs on Windows 11, designed to be extended through Claude Code

## Architecture — STRICTLY 4 LAYERS

```
dashboard/   ──>  analytics/   ──>  data/   ──>  ingestion/
(Streamlit)       (signals)         (DuckDB)     (NSE HTTP)
```

Each layer only imports from the layer below. NEVER let a higher layer reach past its neighbor.

- `src/ingestion/` — NSE HTTP client, fetchers, parsers, orchestrator
- `src/data/` — connection, schema, repository pattern (ALL SQL lives here)
- `src/analytics/` — pure signal computations, no I/O except via repository
- `src/dashboard/` — Streamlit UI, calls analytics functions

## Critical Technical Facts

1. **NSE format changed in July 2024.** Use UDiFF Common Bhavcopy Final:
   - URL: `https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip`
   - UDiFF columns: `TradDt, Sgmt, TckrSymb, SctySrs, OpnPric, HghPric, LwPric, ClsPric, LastPric, PrvsClsgPric, TtlTradgVol, TtlTrfVal, TtlNbOfTxsExctd`
   - TtlTrfVal is in rupees — divide by 100,000 to get lakhs
   - Filter `Sgmt == 'CM'` (cash market) and `SctySrs IN ('EQ','BE','BZ','SM','ST')` for equities only
   - The `F_0000` suffix = Final consolidated file (available ~4 PM IST). Earlier files (`I1`, `I2`) are interim — don't use them.

2. **Delivery data is in a separate file.** UDiFF excludes delivery, so we fetch the MTO file:
   - URL: `https://nsearchives.nseindia.com/archives/equities/mto/MTO_DDMMYYYY.DAT`
   - Format: text file, data rows start with "20", comma-separated: `20,SrNo,Symbol,Series,TradedQty,DeliveryQty,DeliveryPct`

3. **NSE blocks direct requests.** The HTTP client must:
   - Use a real-browser User-Agent
   - Prime cookies by visiting `https://www.nseindia.com/` and `/all-reports` first
   - Retry 3 times with delay, resetting the session on each retry
   - Treat 404 as "expected" (weekend/holiday)

4. **Sectoral indices** for sector classification — fetch constituents from:
   - `https://nsearchives.nseindia.com/content/indices/ind_<filename>list.csv`
   - Cover: NIFTY PHARMA, IT, BANK, AUTO, FMCG, METAL, REALTY, ENERGY, MEDIA, PSU BANK, PVT BANK, FIN SERVICE, CONSUMER DURABLES, HEALTHCARE, OIL AND GAS

5. **DuckDB, not SQLite** — better for window functions. Single file at `data/market_data.duckdb`.

## Files to Create

### Root level
- `CLAUDE.md` — architecture context for future Claude Code sessions (very important)
- `README.md` — comprehensive user docs
- `QUICKSTART.md` — command cheatsheet
- `requirements.txt`: requests>=2.31, pandas>=2.0, duckdb>=0.10, streamlit>=1.30, plotly>=5.18, pyyaml>=6.0, python-dateutil>=2.8.2, pytest>=7.4
- `.gitignore` — exclude venv, __pycache__, .pyc, logs/*.log, data/*.duckdb*, .pytest_cache, .idea, .DS_Store
- `setup.bat` — Windows installer: checks Python, creates venv, installs deps, runs `python -m src.cli init-db`, then `backfill 60`, then `seed-sectors`, then registers Task Scheduler
- `dashboard.bat` — activates venv, runs `streamlit run src/dashboard/streamlit_app.py`
- `fetch_now.bat` — activates venv, runs `python -m src.cli daily`
- `run_tests.bat` — activates venv, runs `pytest tests/ -v`

### .vscode/
- `settings.json` — sets Python interpreter to `./venv/Scripts/python.exe`, enables pytest, type checking basic
- `launch.json` — debug configs for: Streamlit Dashboard, Daily Fetch, Backfill (60 days), Seed Sectors, Run Tests

### config/
- `settings.yaml` — all tunable params: database.path, ingestion (URLs, user_agent, retries, timeout, polite_delay), backfill.trading_days=60, analytics (delivery_avg_window=10, volume_avg_window=20, accumulation_threshold=1.2, distribution_threshold=0.8, min_turnover_lacs=100, weighting_method='turnover'), dashboard.port=8501, scheduler.run_time='18:30', sectors.sources (the list of sectoral indices), logging.level='INFO', logging.log_dir='logs'
- `sector_overrides.csv` — user-editable, columns: symbol, sector, industry, notes — start with header + comments only

### scripts/
- `run_daily.bat` — called by Task Scheduler, runs `python -m src.cli daily`

### src/
- `__init__.py` (empty)
- `config_loader.py` — `load_config()` cached with lru_cache, `get_db_path()`, `get_log_dir()`. PROJECT_ROOT computed from this file's location: `Path(__file__).resolve().parent.parent` (this file is in src/, so .parent.parent gives the project root)
- `logging_setup.py` — `get_logger(name)` that sets up root logger once with file + stream handlers writing to `logs/nse_dashboard_YYYY-MM-DD.log`
- `cli.py` — argparse-style dispatcher for: `init-db`, `backfill [days]`, `daily`, `seed-sectors`, `reload-overrides`. Make it runnable as `python -m src.cli <cmd>`

### src/data/
- `__init__.py` (empty)
- `connection.py` — `get_connection()` context manager that yields a DuckDB connection, auto-closes; also `get_raw_connection()` non-context version for Streamlit caching
- `schema.py` — `initialize_schema()` creates 3 tables (idempotent):
  - `daily_data` (trade_date DATE, symbol VARCHAR, series VARCHAR, prev_close DOUBLE, open_price, high_price, low_price, last_price, close_price, avg_price DOUBLE, ttl_trd_qnty BIGINT, turnover_lacs DOUBLE, no_of_trades BIGINT, deliv_qty BIGINT, deliv_per DOUBLE) — PK (trade_date, symbol, series)
  - `sector_master` (symbol PK, company_name, sector, industry, market_cap_category, last_updated TIMESTAMP)
  - `run_log` (run_id PK from sequence, run_timestamp, run_type, trade_date, status, rows_inserted, error_message, duration_seconds)
  - Indexes on trade_date, symbol, (trade_date, symbol), sector
- `repository.py` — ALL DB operations: `upsert_daily_data(df)` (delete+insert by date for idempotency), `update_delivery_data(df)` (UPDATE matching rows), `upsert_sector_master(df)`, `log_run(...)`, `get_latest_trade_date()`, `get_available_dates(limit)`, `get_total_row_count()`, `get_dates_present(dates_list)` (returns set of dates already in DB), `query_dataframe(sql, params)` for analytics layer

### src/ingestion/
- `__init__.py` (empty)
- `nse_client.py` — `NSEClient` class: lazy session creation, cookie priming, `get(url, expect_404_ok=False)` and `get_text(url, expect_404_ok=False)` with retries, session reset on failure
- `bhavcopy_fetcher.py` — `build_url(date)`, `fetch_bhavcopy(date, client)` returns DataFrame or None, `transform_to_schema(raw_df, trade_date)` does column mapping + filters Sgmt=CM and equity series + converts turnover to lakhs + computes avg_price = turnover_rupees/volume + sets deliv_qty/deliv_per to None (will be filled by delivery fetcher)
- `delivery_fetcher.py` — `build_url(date)`, `fetch_delivery(date, client)` parses MTO text file, returns DataFrame with columns: trade_date, symbol, series, deliv_qty, deliv_per
- `sector_fetcher.py` — `INDEX_DEFINITIONS` dict mapping each index to (filename, display_sector), `fetch_one_index(name, filename, sector, client)`, `fetch_all_sectors(client)` returns deduplicated DataFrame
- `orchestrator.py` — coordinates everything:
  - `fetch_one_date(date, client)` returns (status, rows) — fetches bhavcopy first, transforms, upserts, then fetches delivery, calls update_delivery_data, logs run
  - `run_daily_job()` — entry point for Task Scheduler. Tries today, walks back up to 7 days if unavailable
  - `run_backfill(days=None, skip_existing=True)` — backfills N trading days (weekdays only), skips dates already in DB
  - `seed_sectors(reload_only_overrides=False)` — fetches from NSE, dedupes, tags missing stocks as "Others", applies overrides from CSV
  - `_apply_overrides(overrides_df)` — UPDATE sector_master from CSV rows

### src/analytics/
- `__init__.py` (empty)
- `base.py` — getter helpers reading from config: get_min_turnover_filter, get_delivery_window, get_volume_window, get_thresholds (returns tuple), get_weighting_method; plus re-exports `get_latest_trade_date`, `get_available_dates`
- `delivery_signals.py`:
  - `get_stock_metrics(trade_date, min_turnover_lacs=None)` — main query returning per-stock DataFrame with: symbol, series, company_name, sector, industry, close_price, prev_close, price_change_pct, ttl_trd_qnty, turnover_lacs, deliv_qty, deliv_per, deliv_per_10d_avg, deliv_ratio, vol_20d_avg, vol_ratio, deliv_value_lacs. Uses a CTE with window functions: `AVG(deliv_per) OVER (PARTITION BY symbol, series ORDER BY trade_date ROWS BETWEEN N PRECEDING AND 1 PRECEDING)`. IMPORTANT: `ORDER BY` in the outer SELECT must use the *expression* `(b.deliv_per / NULLIF(b.deliv_per_10d_avg, 0))`, NOT the alias `deliv_ratio`, because DuckDB doesn't recognize the outer SELECT's alias when ordering on a CTE column.
  - `get_top_accumulation(trade_date, limit)` — nlargest by deliv_ratio
  - `get_top_distribution(trade_date, limit)` — filter price_change_pct > 1 AND deliv_ratio < 0.8, then nlargest by price_change_pct
  - `get_stock_history(symbol, days=60)` — last N days of a stock's price + delivery
- `sector_aggregator.py`:
  - `aggregate_by_sector(trade_date, weighting=None, min_turnover_lacs=None)` — groups stocks by sector, computes BOTH simple and turnover-weighted averages for price_change_pct AND deliv_per, also returns top_delivery_symbol per sector (the "why" answer), accumulation_count, distribution_count, stock_count, total_turnover_lacs, total_deliv_value_lacs
  - `get_sector_drilldown(trade_date, sector_name, top_n=10)` returns dict with keys: top_by_delivery_pct, top_by_delivery_value, top_by_turnover, contribution_table (with turnover_share_pct and deliv_value_share_pct columns), sector_summary
  - `get_sector_history(sector_name, days=60)` — daily sector aggregates for trend charts

### src/dashboard/
- `__init__.py` (empty)
- `streamlit_app.py` — main entry, page config wide layout, sidebar with date selector + min_turnover filter + page radio (Sector Overview / Stock Detail / Signals), imports the page modules and dispatches. Must do `sys.path.insert(0, PROJECT_ROOT)` so `from src.xxx` imports work when launched via `streamlit run`.
- `pages/__init__.py` (empty)
- `pages/sector_overview.py` — `render(selected_date, min_turnover)`: shows the dual-axis chart (bars = weighted delivery %, line = weighted price change %), reading guide expander, full sector detail dataframe, sector selectbox for drill-down, 5 KPI cards for selected sector, then 4 tabs (Top by Delivery %, Top by Delivery Value, Top by Turnover, Contribution Treemap), then 60-day sector trend chart
- `pages/stock_detail.py` — `render(selected_date, min_turnover)`: stock picker, 5 KPI cards (close+Δ%, deliv%, 10d avg, deliv ratio, sector), 60-day price+delivery dual-axis chart, raw data expander
- `pages/signals.py` — `render(selected_date, min_turnover)`: two tabs for top accumulation and top distribution
- `components/__init__.py` (empty)
- `components/charts.py` — pure Plotly functions (no Streamlit imports):
  - `sector_dual_axis_chart(sector_df)` — bars colored by delivery % on Blues scale, line with markers colored by price change on red-gray-green scale, two y-axes, horizontal legend
  - `sector_trend_chart(history_df, sector_name)` — 60-day trend bars+line
  - `stock_price_chart(history_df, symbol)` — close price line + delivery % bars (dual-axis)
  - `contribution_treemap(contribution_df, sector_name)` — px.treemap sized by deliv_value_lacs, colored by price_change_pct
- `components/tables.py` — STOCK_TABLE_COLUMNS and SECTOR_TABLE_COLUMNS dicts mapping column names to `st.column_config.*` formatters (NumberColumn with %.2f%%, ProgressColumn for share columns, etc.)

### tests/
- `__init__.py` (empty)
- `test_ingestion.py` — 4 tests using synthetic DataFrames, no NSE calls: filter to equity series, filter to CM segment, turnover rupees→lakhs conversion, avg_price computation
- `test_analytics.py` — uses a `temp_db` pytest fixture that:
  1. Creates `tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False)`
  2. Closes it AND `os.unlink(tmpfile.name)` — DuckDB refuses to open empty files
  3. Monkeypatches `src.config_loader.get_db_path` to return the temp path
  4. Calls `config_loader.load_config.cache_clear()` before AND after the test
  5. Cleans up the .duckdb and .duckdb.wal files in teardown
  Tests: stock metrics computes correct delivery ratio (window function math), sector aggregation produces correct weighted averages, drill-down returns correct top stocks

## Key Implementation Rules

1. **All SQL lives in `src/data/repository.py` or analytics modules calling `repository.query_dataframe()`** — never in dashboard or higher.
2. **Streamlit imports forbidden in analytics/, ingestion/, data/** — those must stay UI-agnostic.
3. **All NSE URL strings live only in `src/ingestion/`** — never elsewhere.
4. **Use turnover as default weighting** for sector metrics, but compute simple averages alongside and show both in the dashboard.
5. **The dual-axis sector chart is the flagship.** It's the user's primary view. Make it the default page.
6. **When clicking a sector**, drill-down must show: WHY the sector shows that delivery number — i.e., the top-delivery stocks within it, plus their contribution share.

## Validation Steps

After generating all files, run:
1. `python -m src.cli init-db` — should print "Database schema initialized" and exit 0
2. `pytest tests/ -v` — all 7 tests must pass. The most common failure modes:
   - `BinderException: Values list "b" does not have a column named "deliv_ratio"` → fix the ORDER BY in delivery_signals.py to use the expression, not the alias
   - `IO Error: ... exists, but it is not a valid DuckDB database file` → the temp_db fixture must `os.unlink()` the empty tempfile before DuckDB opens it
   - `FileNotFoundError: Config file not found at ...` → `PROJECT_ROOT = Path(__file__).resolve().parent.parent` in config_loader.py (NOT `.parent.parent.parent`)
3. Confirm all module imports work: `python -c "from src.dashboard.pages import sector_overview, stock_detail, signals; print('OK')"`

## CLAUDE.md Content

Make CLAUDE.md explain: what the project does, the 4-layer architecture rule, key design decisions (DuckDB, UDiFF format, separate MTO delivery file, turnover-weighted defaults, EQ/BE filter), common tasks (how to add a new signal, how to handle URL changes, how to add a chart), the schema, the "things that will break if you bypass the layers" warnings, and how to run tests.

Build everything now. After generating the files, run validation steps and report which pass/fail.

---

## End of master prompt

---

## After Claude Code Builds It

Whether you used the prebuilt zip or had Claude Code generate the project, the next steps are the same:

1. Open VS Code terminal
2. Run `setup.bat`
3. Wait ~4 minutes for backfill
4. Run `dashboard.bat`
