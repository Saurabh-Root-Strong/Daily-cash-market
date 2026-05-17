import time
from datetime import date, timedelta
from typing import Optional, Tuple
import pandas as pd

from src.logging_setup import get_logger

log = get_logger(__name__)

_MAX_LOOKBACK_DAYS = 7


def _trading_days_back(n: int) -> list:
    result = []
    d = date.today()
    while len(result) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            result.append(d)
    return result


def fetch_one_date(trade_date: date, client) -> Tuple[str, int]:
    from src.ingestion.bhavcopy_fetcher import fetch_bhavcopy
    from src.ingestion.delivery_fetcher import fetch_delivery
    from src.data.repository import upsert_daily_data, update_delivery_data, log_run

    t0 = time.time()
    try:
        bhavcopy_df = fetch_bhavcopy(trade_date, client)
        if bhavcopy_df is None or bhavcopy_df.empty:
            log_run("daily", trade_date, "skipped", 0, "No bhavcopy data", time.time() - t0)
            return "skipped", 0

        rows = upsert_daily_data(bhavcopy_df)
        log.info("Inserted %d bhavcopy rows for %s", rows, trade_date)

        delivery_df = fetch_delivery(trade_date, client)
        if delivery_df is not None and not delivery_df.empty:
            update_delivery_data(delivery_df)
            log.info("Updated delivery data for %s", trade_date)

        duration = time.time() - t0
        log_run("daily", trade_date, "success", rows, None, duration)
        return "success", rows

    except Exception as exc:
        duration = time.time() - t0
        log.error("Failed to process %s: %s", trade_date, exc)
        log_run("daily", trade_date, "error", 0, str(exc), duration)
        return "error", 0


def run_daily_job() -> None:
    from src.ingestion.nse_client import NSEClient
    from src.data.schema import initialize_schema

    initialize_schema()
    client = NSEClient()

    d = date.today()
    for _ in range(_MAX_LOOKBACK_DAYS):
        if d.weekday() >= 5:
            d -= timedelta(days=1)
            continue
        status, rows = fetch_one_date(d, client)
        if status == "success":
            log.info("Daily job complete: %s (%d rows)", d, rows)
            return
        d -= timedelta(days=1)

    log.warning("Daily job: no data found in last %d days", _MAX_LOOKBACK_DAYS)


def run_backfill(days: int = 60, skip_existing: bool = True) -> None:
    from src.ingestion.nse_client import NSEClient
    from src.data.schema import initialize_schema
    from src.data.repository import get_dates_present

    initialize_schema()
    client = NSEClient()

    target_dates = _trading_days_back(days)

    if skip_existing:
        existing = get_dates_present(target_dates)
        to_fetch = [d for d in target_dates if d not in existing]
        log.info("Backfill: %d dates needed, %d already present", len(to_fetch), len(existing))
    else:
        to_fetch = target_dates

    for i, d in enumerate(sorted(to_fetch)):
        log.info("Backfill [%d/%d]: %s", i + 1, len(to_fetch), d)
        fetch_one_date(d, client)
        time.sleep(1.0)

    log.info("Backfill complete: processed %d dates", len(to_fetch))


def seed_sectors(reload_only_overrides: bool = False) -> None:
    from src.ingestion.nse_client import NSEClient
    from src.data.schema import initialize_schema
    from src.data.repository import upsert_sector_master, query_dataframe
    from src.ingestion.sector_fetcher import fetch_all_sectors
    import os
    from pathlib import Path

    initialize_schema()

    if not reload_only_overrides:
        client = NSEClient()
        sector_df = fetch_all_sectors(client)

        if not sector_df.empty:
            # Tag any symbol in daily_data not in sector_df as "Others"
            traded = query_dataframe("SELECT DISTINCT symbol FROM daily_data")
            if not traded.empty:
                missing = set(traded["symbol"]) - set(sector_df["symbol"])
                if missing:
                    from datetime import datetime
                    others = pd.DataFrame({
                        "symbol": list(missing),
                        "company_name": "",
                        "sector": "Others",
                        "industry": "Others",
                        "market_cap_category": "",
                        "last_updated": datetime.now(),
                    })
                    sector_df = pd.concat([sector_df, others], ignore_index=True)

            upsert_sector_master(sector_df)
            log.info("Seeded %d sector records", len(sector_df))

    _apply_overrides()


def _apply_overrides() -> None:
    from src.config_loader import PROJECT_ROOT
    from src.data.repository import upsert_sector_master
    from datetime import datetime

    override_path = PROJECT_ROOT / "config" / "sector_overrides.csv"
    if not override_path.exists():
        return

    try:
        df = pd.read_csv(override_path, comment="#")
        df.columns = [c.strip() for c in df.columns]
        if df.empty or "symbol" not in df.columns:
            return

        df["company_name"] = df.get("company_name", "")
        df["market_cap_category"] = df.get("market_cap_category", "")
        df["last_updated"] = datetime.now()

        upsert_sector_master(df[["symbol", "company_name", "sector", "industry",
                                  "market_cap_category", "last_updated"]])
        log.info("Applied %d sector overrides", len(df))
    except Exception as exc:
        log.error("Failed to apply overrides: %s", exc)
