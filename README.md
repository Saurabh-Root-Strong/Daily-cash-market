# NSE Daily Cash Market Dashboard

A local dashboard for daily NSE equity cash market analysis — sector-level delivery and price metrics, individual stock drill-down, and accumulation/distribution signals.

## Features

- **Sector Overview**: Dual-axis chart showing turnover-weighted delivery % and price change % per sector
- **Drill-Down**: Click any sector to see which stocks are driving its delivery numbers
- **Stock Detail**: 60-day price + delivery trend for any individual stock
- **Signals**: Top accumulation (high delivery relative to average) and distribution stocks

## Requirements

- Windows 11 (for Task Scheduler auto-fetch)
- Python 3.10+
- Internet access to NSE archives

## Quick Start

```
1. Run setup.bat         -- installs dependencies, seeds 60 days of data
2. Run dashboard.bat     -- opens the dashboard at http://localhost:8501
```

## Manual Commands

| Action | Command |
|--------|---------|
| Launch dashboard | `dashboard.bat` |
| Fetch today's data | `fetch_now.bat` |
| Run tests | `run_tests.bat` |
| Backfill N days | `python -m src.cli backfill N` |
| Re-seed sectors | `python -m src.cli seed-sectors` |
| Apply sector overrides | `python -m src.cli reload-overrides` |

## Customizing Sectors

Edit `config/sector_overrides.csv` to assign any stock to a custom sector, then run:
```
python -m src.cli reload-overrides
```

## Architecture

```
dashboard/  -->  analytics/  -->  data/  -->  ingestion/
(Streamlit)      (signals)        (DuckDB)     (NSE HTTP)
```

See `CLAUDE.md` for full architecture documentation (designed for Claude Code sessions).

## Data Sources

- **Bhavcopy**: NSE UDiFF Common Bhavcopy Final (post-July 2024 format)
- **Delivery**: NSE MTO (Market-wide Position Limits) DAT file
- **Sectors**: NSE sectoral index constituent lists
- **Storage**: DuckDB at `data/market_data.duckdb`

## Auto-Fetch Setup

`setup.bat` registers a Windows Task Scheduler job that runs daily at 6:30 PM IST.
NSE typically publishes final data by ~4 PM IST; 6:30 PM gives a comfortable buffer.
