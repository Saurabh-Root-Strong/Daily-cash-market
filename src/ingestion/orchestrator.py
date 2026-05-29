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
from src.ingestion.fao_fetcher import FAOParticipantFetcher
from src.ingestion.fii_stats_fetcher import FIIStatsFetcher
from src.ingestion.fno_bhavcopy_fetcher import FNOBhavCopyFetcher
from src.ingestion.fpi_nsdl_fetcher import fetch_fpi_monthly
from src.ingestion.http_client import NSEHttpClient
from src.ingestion.index_fetcher import IndexFetcher

__all__ = [
    "fetch_one_date", "run_daily_job", "run_backfill",
    "run_index_backfill", "run_fao_backfill", "run_fii_stats_backfill",
    "run_fno_backfill", "seed_sectors",
]

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

        # Fetch index data (niftyindices.com — independent of NSE bhavcopy)
        try:
            idx_df = IndexFetcher().fetch(trade_date)
            if not idx_df.empty:
                repo.upsert_index_data(idx_df)
                log.info("Inserted %d index rows for %s", len(idx_df), trade_date)
        except Exception as idx_exc:
            log.warning("Index fetch failed for %s: %s", trade_date, idx_exc)

        # Fetch F&O participant-wise OI + Volume (non-fatal — separate data source)
        try:
            fao_df = FAOParticipantFetcher(client).fetch(trade_date)
            if not fao_df.empty:
                repo.upsert_fao_data(fao_df)
                log.info("Inserted %d F&O participant rows for %s", len(fao_df), trade_date)
        except Exception as fao_exc:
            log.warning("F&O participant fetch failed for %s: %s", trade_date, fao_exc)

        # Fetch FII Derivatives Statistics (non-fatal — buy/sell value by contract type)
        try:
            fii_stats_df = FIIStatsFetcher(client).fetch(trade_date)
            if not fii_stats_df.empty:
                repo.upsert_fii_stats(fii_stats_df)
                log.info("Inserted %d FII stats rows for %s", len(fii_stats_df), trade_date)
        except Exception as fii_exc:
            log.warning("FII stats fetch failed for %s: %s", trade_date, fii_exc)

        # Fetch FNO Bhavcopy — direct NSE archive URLs work for the last ~7 days.
        # The no-date API is skipped for historical dates (fetcher handles this).
        try:
            from datetime import date as _date
            days_ago = (_date.today() - trade_date).days
            if 0 <= days_ago <= 6:
                fno_df = FNOBhavCopyFetcher(client).fetch(trade_date)
                if not fno_df.empty:
                    repo.upsert_fno_bhavcopy(fno_df)
                    log.info("Inserted %d FNO bhavcopy rows for %s", len(fno_df), trade_date)
        except Exception as fno_exc:
            log.warning("FNO bhavcopy fetch failed for %s: %s", trade_date, fno_exc)

        duration = time.perf_counter() - t0
        repo.log_run("daily", trade_date, "success", rows, None, duration)
        return "success", rows

    except Exception as exc:
        duration = time.perf_counter() - t0
        log.error("Failed to process %s: %s", trade_date, exc, exc_info=True)
        repo.log_run("daily", trade_date, "error", 0, str(exc), duration)
        return "error", 0


def _fill_supplementary_gaps(
    client: NSEHttpClient,
    repo: MarketDataRepository,
    lookback_days: int = 10,
) -> None:
    """
    Scan last `lookback_days` trading days and fill any gaps in supplementary
    tables (FAO, FII stats, Index data).  Bhavcopy presence is the gate —
    we only attempt a fill when bhavcopy already exists for that date.

    FNO bhavcopy is included for dates within the last 6 days — NSE keeps DAT
    files for ~7 days via direct archive URLs.  Older gaps require manual import.
    FPI flows are excluded: NSDL 1-2 day lag is expected — no alert needed.
    """
    target_dates = _weekdays_back(lookback_days)
    bhavcopy_dates = repo.get_dates_present(target_dates)

    fno_window = date.today() - timedelta(days=6)

    fao_dates = repo.get_distinct_dates("fao_participant")
    fii_dates = repo.get_distinct_dates("fii_derivatives_stats")
    idx_dates = repo.get_distinct_dates("index_data")
    fno_dates = repo.get_distinct_dates("fno_bhavcopy")

    gaps_filled = 0
    for d in sorted(target_dates):
        if d not in bhavcopy_dates:
            continue  # Not a trading day — skip

        needs_fao = d not in fao_dates
        needs_fii = d not in fii_dates
        needs_idx = d not in idx_dates
        needs_fno = d not in fno_dates and d >= fno_window

        if not (needs_fao or needs_fii or needs_idx or needs_fno):
            continue

        log.info("Supplementary gap for %s — FAO=%s  FII=%s  Index=%s  FNO=%s",
                 d, needs_fao, needs_fii, needs_idx, needs_fno)

        if needs_fao:
            try:
                df = FAOParticipantFetcher(client).fetch(d)
                if not df.empty:
                    repo.upsert_fao_data(df)
                    log.info("Gap-filled FAO for %s (%d rows)", d, len(df))
                    gaps_filled += 1
                else:
                    log.info("FAO gap-fill %s: NSE returned empty", d)
            except Exception as e:
                log.warning("FAO gap-fill failed %s: %s", d, e)
            time.sleep(0.4)

        if needs_fii:
            try:
                df = FIIStatsFetcher(client).fetch(d)
                if not df.empty:
                    repo.upsert_fii_stats(df)
                    log.info("Gap-filled FII stats for %s (%d rows)", d, len(df))
                    gaps_filled += 1
                else:
                    log.info("FII-stats gap-fill %s: NSE returned empty", d)
            except Exception as e:
                log.warning("FII-stats gap-fill failed %s: %s", d, e)
            time.sleep(0.4)

        if needs_idx:
            try:
                df = IndexFetcher().fetch(d)
                if not df.empty:
                    repo.upsert_index_data(df)
                    log.info("Gap-filled Index data for %s (%d rows)", d, len(df))
                    gaps_filled += 1
                else:
                    log.info("Index gap-fill %s: returned empty", d)
            except Exception as e:
                log.warning("Index gap-fill failed %s: %s", d, e)
            time.sleep(0.4)

        if needs_fno:
            try:
                df = FNOBhavCopyFetcher(client).fetch(d)
                if not df.empty:
                    repo.upsert_fno_bhavcopy(df)
                    log.info("Gap-filled FNO bhavcopy for %s (%d rows)", d, len(df))
                    gaps_filled += 1
                else:
                    log.info("FNO gap-fill %s: NSE returned empty", d)
            except Exception as e:
                log.warning("FNO gap-fill failed %s: %s", d, e)
            time.sleep(0.4)

    if gaps_filled:
        log.info("Supplementary gap-fill complete: %d tables updated", gaps_filled)


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
        # Skip NSE request entirely if bhavcopy is already present (retry-safe)
        if d in repo.get_dates_present([d]):
            log.info("Daily job: bhavcopy already present for %s — skipping main fetch", d)
            break
        status, rows = fetch_one_date(d, client, repo)
        if status == "success":
            log.info("Daily job complete: %s (%d rows)", d, rows)
            break
        d -= timedelta(days=1)
    else:
        log.warning("Daily job: no data found in last %d days", _MAX_LOOKBACK_DAYS)

    # NOTE: Memory engine outcome update (update_outcomes) is intentionally NOT
    # called here. Orchestrator is ingestion layer and must not import analytics.
    # update_outcomes() is called from cmd_daily in src/cli.py after this returns.

    # FPI Capital Flow — NSDL publishes with 1-2 day lag, so fetch monthly to
    # catch all days in the current month including any lagged updates.
    try:
        result = fetch_fpi_monthly()
        if result["error"]:
            log.warning("FPI monthly fetch: %s", result["error"])
        else:
            log.info("FPI monthly fetch: %d rows for %d dates",
                     result["rows_inserted"], len(result["dates"]))
    except Exception as fpi_exc:
        log.warning("FPI fetch failed: %s", fpi_exc)

    # Always run gap-fill: catches FAO/FII/Index lags regardless of bhavcopy status
    try:
        _fill_supplementary_gaps(client, repo, lookback_days=10)
    except Exception as e:
        log.warning("Supplementary gap-fill failed: %s", e)

    # Log health summary so the daily run log always shows data completeness
    try:
        from src.data.health import run_health_check
        h = run_health_check(lookback_days=10)
        if h.has_errors:
            for s in h.error_sources:
                log.warning("DATA GAP: %s missing %s", s.label,
                            [str(x) for x in s.critical_missing])
        else:
            log.info("Data health OK — all tables current through %s",
                     max((s.latest_date for s in h.sources.values() if s.latest_date), default="?"))
    except Exception as e:
        log.warning("Health check failed: %s", e)


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


def run_index_backfill(
    days: int = 120,
    repo: Optional[MarketDataRepository] = None,
) -> None:
    """Backfill NiftyIndices daily snapshots for the last `days` weekdays."""
    from src.data.schema import initialize_schema

    initialize_schema()
    repo = repo or get_repository()
    fetcher = IndexFetcher()
    target_dates = _weekdays_back(days)

    existing = repo.get_distinct_dates("index_data")
    to_fetch = [d for d in sorted(target_dates) if d not in existing]
    log.info("Index backfill: %d dates needed, %d already present",
             len(to_fetch), len(existing))

    for i, d in enumerate(to_fetch):
        log.info("Index backfill [%d/%d]: %s", i + 1, len(to_fetch), d)
        df = fetcher.fetch(d)
        if not df.empty:
            repo.upsert_index_data(df)
        time.sleep(0.5)

    log.info("Index backfill complete: %d dates processed", len(to_fetch))


def run_fao_backfill(
    days: int = 365,
    client: Optional[NSEHttpClient] = None,
    repo: Optional[MarketDataRepository] = None,
) -> None:
    """Backfill F&O participant-wise OI + Volume for the last `days` weekdays."""
    from src.data.schema import initialize_schema

    initialize_schema()
    client = client or NSEHttpClient()
    repo   = repo   or get_repository()

    target_dates = _weekdays_back(days)

    existing = repo.get_distinct_dates("fao_participant")
    to_fetch = [d for d in sorted(target_dates) if d not in existing]
    log.info("F&O backfill: %d dates needed, %d already present",
             len(to_fetch), len(existing))

    for i, d in enumerate(to_fetch):
        log.info("F&O backfill [%d/%d]: %s", i + 1, len(to_fetch), d)
        try:
            fao_df = FAOParticipantFetcher(client).fetch(d)
            if not fao_df.empty:
                repo.upsert_fao_data(fao_df)
                log.info("  [OK] %d rows inserted", len(fao_df))
            else:
                log.info("  [--] No data (holiday/weekend)")
        except Exception as exc:
            log.warning("  [!!] Failed: %s", exc)
        time.sleep(1.5)

    log.info("F&O backfill complete: %d dates processed", len(to_fetch))


def run_fii_stats_backfill(
    days: int = 365,
    client: Optional[NSEHttpClient] = None,
    repo: Optional[MarketDataRepository] = None,
) -> None:
    """Backfill FII Derivatives Statistics for the last `days` weekdays."""
    from src.data.schema import initialize_schema

    initialize_schema()
    client = client or NSEHttpClient()
    repo   = repo   or get_repository()

    target_dates = _weekdays_back(days)

    existing = repo.get_distinct_dates("fii_derivatives_stats")
    to_fetch = [d for d in sorted(target_dates) if d not in existing]
    log.info("FII Stats backfill: %d dates needed, %d already present",
             len(to_fetch), len(existing))

    for i, d in enumerate(to_fetch):
        log.info("FII Stats backfill [%d/%d]: %s", i + 1, len(to_fetch), d)
        try:
            df = FIIStatsFetcher(client).fetch(d)
            if not df.empty:
                repo.upsert_fii_stats(df)
                log.info("  [OK] %d rows inserted", len(df))
            else:
                log.info("  [--] No data (holiday/weekend)")
        except Exception as exc:
            log.warning("  [!!] Failed: %s", exc)
        time.sleep(1.5)

    log.info("FII Stats backfill complete: %d dates processed", len(to_fetch))


def run_fno_backfill(
    days: int = 7,
    client: Optional[NSEHttpClient] = None,
    repo: Optional[MarketDataRepository] = None,
) -> None:
    """
    Fetch F&O Bhavcopy for the last `days` trading days.

    NSE keeps the DAT files on direct archive URLs for ~7 days.
    For today: uses the no-date API (always available).
    For past dates (up to 6 days ago): uses direct archive URLs.
    Older dates are not available from NSE and require manual fo*.zip import.
    """
    from src.data.schema import initialize_schema

    initialize_schema()
    client = client or NSEHttpClient()
    repo   = repo   or get_repository()

    target_dates = _weekdays_back(days)

    existing = repo.get_distinct_dates("fno_bhavcopy")
    to_fetch = [d for d in sorted(target_dates) if d not in existing]
    log.info("FNO backfill: %d dates needed, %d already present",
             len(to_fetch), len(existing))

    fetcher = FNOBhavCopyFetcher(client)
    for i, d in enumerate(to_fetch):
        log.info("FNO backfill [%d/%d]: %s", i + 1, len(to_fetch), d)
        try:
            df = fetcher.fetch(d)
            if not df.empty:
                actual_date = df["trade_date"].iloc[0]
                repo.upsert_fno_bhavcopy(df)
                log.info("  [OK] %d rows stored for %s", len(df), actual_date)
            else:
                log.info("  [--] No data (NSE archive expired or holiday)")
        except Exception as exc:
            log.warning("  [!!] Failed: %s", exc)
        time.sleep(1.0)

    log.info("FNO backfill complete: %d dates processed", len(to_fetch))


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
        overrides = pd.read_csv(override_path, comment="#")
        overrides.columns = [c.strip() for c in overrides.columns]
        if overrides.empty or "symbol" not in overrides.columns:
            return

        # Fetch existing rows so we don't wipe company_name / market_cap_category
        existing = repo.query(
            "SELECT symbol, company_name, market_cap_category FROM sector_master"
        ).set_index("symbol")

        rows = []
        now = datetime.now()
        for _, ov in overrides.iterrows():
            sym = ov["symbol"]
            ex = existing.loc[sym] if sym in existing.index else None
            rows.append({
                "symbol":              sym,
                "company_name":        (ex["company_name"]        if ex is not None else "") or "",
                "sector":              ov.get("sector",   "") or "",
                "industry":            ov.get("industry", "") or "",
                "category":            ov.get("category", "") or "",
                "market_cap_category": (ex["market_cap_category"] if ex is not None else "") or "",
                "last_updated":        now,
            })

        repo.upsert_sector_master(pd.DataFrame(rows))
        log.info("Applied %d sector overrides", len(rows))
    except Exception as exc:
        log.error("Failed to apply overrides: %s", exc)
