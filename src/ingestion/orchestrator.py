"""
Ingestion orchestrator — coordinates fetchers, repository, and logging.

Design principles:
  - Accepts client and repo via parameters (dependency injection) so tests can
    substitute fakes without touching global state.
  - run_daily_job() and run_backfill() are the only entry points called by CLI.
  - fetch_one_date() is the atomic unit: bhavcopy → upsert → delivery → update.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.core.logging import get_logger
from src.data.repository import MarketDataRepository, get_repository
from src.ingestion.bhavcopy_fetcher import BhavCopyFetcher
from src.ingestion.delivery_fetcher import DeliveryFetcher
from src.ingestion.http_client import NSEHttpClient

__all__ = ["fetch_one_date", "run_daily_job", "run_backfill", "seed_sectors"]

log = get_logger(__name__)

_MAX_LOOKBACK_DAYS = 7


def _weekdays_back(n: int) -> list[date]:
    """Return the n most-recent weekdays (Mon–Fri) before today."""
    result: list[date] = []
    d = date.today()
    while len(result) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            result.append(d)
    return result


def fetch_one_date(
    trade_date: date,
    client: NSEHttpClient,
    repo: Optional[MarketDataRepository] = None,
) -> tuple[str, int]:
    """
    Fetch bhavcopy + delivery for one trading date.

    Returns
    -------
    (status, rows_inserted)
      status in {"success", "skipped", "error"}
    """
    repo = repo or get_repository()
    t0 = time.perf_counter()

    try:
        bhavcopy_df = BhavCopyFetcher(client).fetch(trade_date)
        if bhavcopy_df.empty:
            repo.log_run("daily", trade_date, "skipped", 0, "No bhavcopy data",
                         time.perf_counter() - t0)
            return "skipped", 0

        rows = repo.upsert_daily_data(bhavcopy_df)
        log.info("Inserted %d bhavcopy rows for %s", rows, trade_date)

        delivery_df = DeliveryFetcher(client).fetch(trade_date)
        if not delivery_df.empty:
            repo.update_delivery_data(delivery_df)
            log.info("Updated delivery data for %s", trade_date)

        duration = time.perf_counter() - t0
        repo.log_run("daily", trade_date, "success", rows, None, duration)
        return "success", rows

    except Exception as exc:
        duration = time.perf_counter() - t0
        log.error("Failed to process %s: %s", trade_date, exc, exc_info=True)
        repo.log_run("daily", trade_date, "error", 0, str(exc), duration)
        return "error", 0


def run_daily_job(
    client: Optional[NSEHttpClient] = None,
    repo: Optional[MarketDataRepository] = None,
) -> None:
    """Try today's data; walk back up to _MAX_LOOKBACK_DAYS for the latest available."""
    from src.data.schema import initialize_schema

    initialize_schema()
    client = client or NSEHttpClient()
    repo = repo or get_repository()

    d = date.today()
    for _ in range(_MAX_LOOKBACK_DAYS):
        if d.weekday() >= 5:
            d -= timedelta(days=1)
            continue
        status, rows = fetch_one_date(d, client, repo)
        if status == "success":
            log.info("Daily job complete: %s (%d rows)", d, rows)
            return
        d -= timedelta(days=1)

    log.warning("Daily job: no data found in last %d days", _MAX_LOOKBACK_DAYS)


def run_backfill(
    days: int = 60,
    skip_existing: bool = True,
    client: Optional[NSEHttpClient] = None,
    repo: Optional[MarketDataRepository] = None,
) -> None:
    """Backfill the last `days` trading days."""
    from src.data.schema import initialize_schema

    initialize_schema()
    client = client or NSEHttpClient()
    repo = repo or get_repository()

    target_dates = _weekdays_back(days)

    if skip_existing:
        existing = repo.get_dates_present(target_dates)
        to_fetch = [d for d in target_dates if d not in existing]
        log.info("Backfill: %d needed, %d already present", len(to_fetch), len(existing))
    else:
        to_fetch = target_dates

    for i, d in enumerate(sorted(to_fetch)):
        log.info("Backfill [%d/%d]: %s", i + 1, len(to_fetch), d)
        fetch_one_date(d, client, repo)
        time.sleep(1.0)

    log.info("Backfill complete: processed %d dates", len(to_fetch))


def seed_sectors(
    reload_only_overrides: bool = False,
    client: Optional[NSEHttpClient] = None,
    repo: Optional[MarketDataRepository] = None,
) -> None:
    """Fetch NSE index constituents and populate sector_master."""
    from src.data.schema import initialize_schema
    from src.ingestion.sector_fetcher import SectorFetcher
    from src.core.config import get_config, PROJECT_ROOT

    initialize_schema()
    repo = repo or get_repository()

    if not reload_only_overrides:
        client = client or NSEHttpClient()
        sector_df = SectorFetcher(client).fetch()

        if not sector_df.empty:
            # Tag symbols in daily_data but not in any index as "Others"
            traded = repo.query("SELECT DISTINCT symbol FROM daily_data")
            if not traded.empty:
                missing = set(traded["symbol"]) - set(sector_df["symbol"])
                if missing:
                    from datetime import datetime
                    others = pd.DataFrame({
                        "symbol":             list(missing),
                        "company_name":       "",
                        "sector":             "Others",
                        "industry":           "Others",
                        "market_cap_category": "",
                        "last_updated":       datetime.now(),
                    })
                    sector_df = pd.concat([sector_df, others], ignore_index=True)

            repo.upsert_sector_master(sector_df)
            log.info("Seeded %d sector records", len(sector_df))

    _apply_overrides(repo)


def _apply_overrides(repo: MarketDataRepository) -> None:
    from src.core.config import PROJECT_ROOT
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
        repo.upsert_sector_master(
            df[["symbol", "company_name", "sector", "industry",
                "market_cap_category", "last_updated"]]
        )
        log.info("Applied %d sector overrides", len(df))
    except Exception as exc:
        log.error("Failed to apply overrides: %s", exc)
