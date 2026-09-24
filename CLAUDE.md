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

## Bulk / Block Deals

`bulk_block_deals` (2010-present) + `short_selling`, fetched year-at-a-time from
`api/historicalOR/bulk-block-short-deals?...&csv=true`. The JSON form of that
endpoint is hard-capped at 70 rows with no working page parameter, and
`/content/equities/bulk.csv` on nsearchives serves the latest session only, so
`csv=true` over a date range is the only route to history.

Three traps that will corrupt any read of this data:
1. A trade put through the BLOCK window that also crosses the 0.5% bulk threshold
   is filed in BOTH feeds, identically (1,724 twins since 2018, Rs 5.19 lakh Cr).
   Summing the feeds double-counts.
2. Block-window quantity settles OUTSIDE the regular market, so it is not inside
   `daily_data.ttl_trd_qnty` — "% of day volume" can exceed 100% and is only
   meaningful for regular-market lines.
3. A quarter of bulk symbol-days net to zero: one client buying and selling the
   same quantity the same day (mostly HFT market makers). Gross quantity scores
   that as heavy buying. Always net.

`deal_event_outcomes` (built by `scripts/build_deal_outcomes.py`) scores every
event against matched non-event stock-days — same session, liquidity quintile,
volume-spike decile, same-day-move decile — because a deal day is by construction
a high-volume, big-move day and those revert on their own. Measured 2018-2026:
any bulk deal -1.5% / 1 month (t = -3.5), bulk net BUY -1.9% (t = -3.7), same-day
round trip -2.3% (t = -4.1), block deals ~0 at both horizons. Reality check over
44 candidates: p = 0.0025.

`deal_client_events` (`scripts/build_client_events.py`) answers "can I follow a
specific buyer?" — no. Ranking entities on their own already-resolved earlier buys
and then measuring their LATER buys, the best-record quartile still returns -1.0% /
1wk and -1.4% / 1mo. Only the WORST quartile separates (-2.5% / 1mo), so the
ranking has weak avoidance value and zero following value.

## SLB (Securities Lending & Borrowing)

`slb_bhavcopy` + `slb_open_positions` (2017-present, `scripts/backfill_slb.py`, raw
cache `data/slb_raw/`), `slb_foreclosure` (current notices). Daily refresh:
`orchestrator._fetch_slb_window`. Files are listed by `/api/merged-daily-reports?key=SLBS`,
not by the capital-market list. Traps:
1. `SLBM_BC_*.DAT` has NO header; the 17 columns were mapped by reconciliation.
2. Fees are rupees per share for the WHOLE remaining tenure. Annualise by days left.
   ROLLOVER series (OD/ON/O1/O2...) price an extension. Never annualise them by days
   to the near expiry.
3. Return date moved Thursday -> TUESDAY in 2026 (the NCL FAQ still says Thursday).
   Read expiry from the file.
4. `daily_data.prev_close` is NOT adjusted for ANY corporate action. That covers
   dividends, and also splits and bonuses: BAJFINANCE 2025-06-16 shows prev_close
   9,331 against a 938 close. So a guard like "prev_close != yesterday's close"
   never fires. Use the `corp_actions` table instead
   (`src/ingestion/corp_actions_fetcher.py`, refreshed nightly): `share_factor` =
   new shares per old share, `kind`, `dividend`. Its endpoint serves ex-dates only
   up to about the next session.
5. The market grew ~5x in 2025-26, so use same-day ranks, never fixed levels.
6. ROLLS double-count. In the ~7 sessions before a return date, borrowers take
   next month's loan while the old one is still open. Raw outstanding therefore
   swells, then drops on the date. `src/analytics/slb_episodes.roll_adjust` removes
   the overlap (INFY Aug-2026: ~10M rolled, ~8.6M returned). Anything about WHEN
   borrowing started, or how it was added, must use the roll-adjusted series.

Measured (`data/slb_study/summary.json`): the top decile of fresh borrowing trails
the middle deciles by ~60 bps over 21 sessions (sector-neutral, 8/9 years). It
survives controls for momentum, volatility, liquidity, dividends and futures basis.
Deciles 1-8 are flat. Borrow fee plus STT eat the gap, so it is an AVOID flag, not
a short.

Build-up timing (`scripts/build_slb_timeline_study.py`, `timeline.json`): 6
pre-registered tests, and none passes |t| > 2.64.
- New vs old flag: t 1.5.
- Shorts losing vs winning: t −1.1. There is NO squeeze; losing-shorts names did
  worse.
- Covering vs holding: +0.47%, t 2.1. The closest of the three, still not proven.

So the "Building since" / phase / avg-borrow-price columns are description, not
forecast.

### daily_data defect: untrustworthy prev_close

439 rows in 2018-2026 carry a `prev_close` the same row's own open contradicts.
2026-04-20 is the worst session (8 symbols): NIRLON prints prev_close 1.40 against
an open of 540.00 and a close of 539.79, i.e. a reported +38,456% day. Anything that
COMPOUNDS `close/prev_close` turns that into a fake 385x in the stock's return
series. Both deal builders now derive a `ratio_clean` (fall back to close/open when
prev_close is contradicted, else clamp to 0.5-2.0) before compounding.

Live panels are largely shielded by accident, not design: these symbols are absent
from `sector_master` and all sit under the default Rs 1 Cr turnover filter. Any NEW
backtest that compounds returns must apply the same guard.

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
