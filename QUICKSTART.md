# Quick Start Cheatsheet

## First-time setup
```
setup.bat
```

## Daily use
```
dashboard.bat           launch Streamlit at localhost:8501
fetch_now.bat           fetch today's NSE data manually
```

## CLI commands
```
python -m src.cli init-db               reset/init database schema
python -m src.cli backfill 60           backfill last 60 trading days
python -m src.cli backfill 5            backfill last 5 trading days
python -m src.cli daily                 fetch today (or most recent)
python -m src.cli seed-sectors          fetch sector data from NSE
python -m src.cli reload-overrides      apply config/sector_overrides.csv
```

## Testing
```
run_tests.bat
pytest tests/ -v
```

## Config
```
config/settings.yaml          all tunable parameters
config/sector_overrides.csv   manually assign stocks to sectors
```

## Data
```
data/market_data.duckdb       the database
logs/nse_dashboard_*.log      daily log files
```
