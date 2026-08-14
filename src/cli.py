import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_init_db(_args) -> int:
    from src.data.schema import initialize_schema
    initialize_schema()
    print("Database schema initialized.")
    return 0


def cmd_backfill(args) -> int:
    from src.ingestion.orchestrator import run_backfill
    days = int(args.days) if args.days else 60
    run_backfill(days=days)
    return 0


def run_daily_pipeline(write_marker: bool = True) -> int:
    """Canonical end-of-day routine — the SINGLE code path for the daily update.

    Fetch EOD data → log index predictions → fetch FII/DII cash → fill prediction
    outcomes → record sector memory → refresh the last_updated marker. Called by BOTH
    the 7:30 PM scheduler (cmd_daily / run_daily.bat) AND the dashboard 'Refresh Data'
    button, so a MANUAL refresh is a complete substitute for a missed scheduled run —
    no half-updates and no marker drift. (Refresh previously called only run_daily_job(),
    which advanced the data but skipped prediction logging and left the marker stale.)
    """
    from datetime import date
    from src.ingestion.orchestrator import run_daily_job
    run_daily_job()

    # Resolve the latest TRADING day we actually have index data for. Both the
    # index-prediction log and the sector-memory snapshot are stamped with this
    # date — NOT date.today(). On a weekend / holiday / pre-EOD run, date.today()
    # would stamp rows with a non-trading calendar date (e.g. a Sunday), creating
    # phantom rows and leaving the real last-session record unlogged.
    try:
        from src.data.repository import query_dataframe
        _latest = query_dataframe("SELECT max(trade_date) AS d FROM index_data")
        trade_dt = _latest["d"].iloc[0] if _latest is not None and not _latest.empty else None
        if hasattr(trade_dt, "date"):
            trade_dt = trade_dt.date()
    except Exception:
        trade_dt = None
    if trade_dt is None:
        trade_dt = date.today()

    # Compute & LOG index predictions deterministically (persist=True).
    # This makes prediction_log complete every trading day regardless of whether
    # anyone opens the dashboard — the dashboard now renders read-only
    # (persist=False) to avoid taking a write lock on every page load.
    try:
        from src.analytics.index_prediction import get_index_predictions
        preds = get_index_predictions(trade_dt, persist=True)
        logged = sum(1 for p in preds if getattr(p, "data_available", False))
        if logged:
            print(f"Index predictions: logged {logged} prediction(s) for {trade_dt}")
    except Exception as exc:
        print(f"Index prediction logging failed (non-fatal): {exc}")

    # FII/DII daily cash flows — fetch NSE provisional figures.
    try:
        from src.ingestion.fii_dii_cash_fetcher import run_fii_dii_cash_daily
        n = run_fii_dii_cash_daily()
        if n:
            print(f"FII/DII cash: updated {n} day(s) from NSE")
    except Exception as exc:
        print(f"FII/DII cash fetch failed (non-fatal): {exc}")

    # Fill prediction outcomes AFTER ingestion completes.
    # CLI sits above all layers and may call both ingestion and analytics.
    # update_outcomes fills any matured pending row, so date.today() (a maturity
    # cutoff, not a stamp) is fine here.
    try:
        from src.analytics.memory_engine import update_outcomes
        filled = update_outcomes(date.today())
        if filled:
            print(f"Memory engine: filled outcomes for {filled} predictions")
    except Exception as exc:
        print(f"Memory engine outcome update failed (non-fatal): {exc}")

    # Sector rotation memory — record the latest session's sector signals + fill
    # matured forward outcomes so the engine self-improves with each new trading
    # day. The snapshot is stamped with the resolved trading date (not today);
    # fill_forward_outcomes uses today() only as a maturity cutoff.
    try:
        from datetime import date as _date
        from src.analytics.sector_memory import (
            record_daily_snapshot, fill_forward_outcomes,
        )
        recorded = record_daily_snapshot(trade_dt)
        if recorded:
            print(f"Sector memory: recorded {recorded} sector rows for {trade_dt}")
        sec_filled = fill_forward_outcomes(_date.today())
        if sec_filled:
            print(f"Sector memory: filled forward outcomes for {sec_filled} records")
    except Exception as exc:
        print(f"Sector memory update failed (non-fatal): {exc}")

    # Refresh the last_updated marker from the DB's latest trade date — single source of
    # truth, written by EVERY caller (scheduler and dashboard) so it can never go stale.
    if write_marker and trade_dt is not None:
        try:
            from pathlib import Path
            marker = Path(__file__).resolve().parents[1] / "logs" / "last_updated.txt"
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(str(trade_dt)[:10])
        except Exception as exc:
            print(f"last_updated marker write failed (non-fatal): {exc}")
    return 0


def cmd_daily(_args) -> int:
    """Scheduled daily entrypoint (run_daily.bat) — delegates to the canonical pipeline
    that the dashboard Refresh button also uses, so both paths stay identical."""
    return run_daily_pipeline()


def cmd_seed_sectors(_args) -> int:
    from src.ingestion.orchestrator import seed_sectors
    seed_sectors()
    return 0


def cmd_reload_overrides(_args) -> int:
    from src.ingestion.orchestrator import seed_sectors
    seed_sectors(reload_only_overrides=True)
    return 0


def cmd_backfill_indices(args) -> int:
    from src.ingestion.orchestrator import run_index_backfill
    days = int(args.days) if args.days else 120
    print(f"Backfilling index data for last {days} trading days...")
    run_index_backfill(days=days)
    print("Index backfill complete.")
    return 0


def cmd_backfill_fao(args) -> int:
    from src.ingestion.orchestrator import run_fao_backfill
    days = int(args.days) if args.days else 365
    print(f"Backfilling F&O participant data for last {days} trading days (~{days//5} weeks)...")
    print("This fetches from NSE — may take several minutes. Press Ctrl+C to stop.")
    run_fao_backfill(days=days)
    print("F&O backfill complete.")
    return 0


def cmd_backfill_fii_stats(args) -> int:
    from src.ingestion.orchestrator import run_fii_stats_backfill
    days = int(args.days) if args.days else 365
    print(f"Backfilling FII Derivatives Statistics for last {days} trading days...")
    print("Source: NSE Archives — F&O FII Derivatives Statistics. Press Ctrl+C to stop.")
    run_fii_stats_backfill(days=days)
    print("FII stats backfill complete.")
    return 0


def cmd_backfill_fno(args) -> int:
    from src.ingestion.orchestrator import run_fno_backfill
    days = int(args.days) if args.days else 50
    print(f"Backfilling F&O Bhavcopy (FNO) data for last {days} trading days...")
    print("Source: NSE Archives — UDiFF F&O Bhavcopy (served from 2024-01-01, no expiry). "
          "Press Ctrl+C to stop.")
    run_fno_backfill(days=days)
    print("FNO backfill complete.")
    return 0


def cmd_fill_gaps(args) -> int:
    """Fill FAO + FII stats + Index + FNO gaps for last N trading days (targeted)."""
    from src.ingestion.orchestrator import _fill_supplementary_gaps
    from src.ingestion.http_client import NSEHttpClient
    from src.data.repository import get_repository
    days = int(args.days) if args.days else 7
    print(f"Scanning last {days} trading days for FAO / FII-stats / Index / FNO gaps…")
    client = NSEHttpClient()
    repo   = get_repository()
    _fill_supplementary_gaps(client, repo, lookback_days=days)
    print("Gap-fill complete. Restart the dashboard (or click Refresh Data) to see updates.")
    return 0


def cmd_import_fpi(_args) -> int:
    """Import historical Excel files from data/fpi_imports/ folder."""
    from src.ingestion.fpi_nsdl_fetcher import import_fpi_folder
    result = import_fpi_folder()
    print(f"FPI import: {result['files_processed']} file(s) processed, "
          f"{result['rows_inserted']} rows upserted.")
    if result["errors"]:
        for err in result["errors"]:
            print(f"  WARNING: {err}")
    return 0


def cmd_fetch_fpi_latest(_args) -> int:
    """Fetch today's FPI data from NSDL Latest.aspx (auto, no login needed)."""
    from src.ingestion.fpi_nsdl_fetcher import fetch_fpi_latest
    result = fetch_fpi_latest()
    if result["error"]:
        print(f"ERROR: {result['error']}")
        return 1
    print(f"FPI latest: {result['rows_inserted']} rows for {result['dates']}")
    return 0


def cmd_data_health(_args) -> int:
    """Show completeness status for all data tables."""
    from datetime import date
    from src.data.health import run_health_check
    h = run_health_check(lookback_days=10)
    print(f"\nData Health Report — {h.as_of.strftime('%d %b %Y')}")
    print(f"Checking last {len(h.trading_days)} trading days "
          f"({h.trading_days[0]} to {h.trading_days[-1]})\n")
    print(f"  {'Source':<22} {'Latest':>10}  {'Status'}")
    print(f"  {'-'*22} {'-'*10}  {'-'*30}")
    for s in h.sources.values():
        latest_str = str(s.latest_date) if s.latest_date else "NEVER"
        if s.level == "ok":
            status = "OK — fully current"
        elif s.level == "warn":
            status = f"OK — {len(s.missing_dates)}d NSDL lag (expected)"
        else:
            missing = ", ".join(str(d) for d in s.critical_missing)
            status = f"MISSING: {missing}"
        icon = {"ok": "OK", "warn": "~~", "error": "!!"  }[s.level]
        print(f"  {icon} {s.label:<21} {latest_str:>10}  {status}")
    print()
    auto_fixable = [s for s in h.error_sources if s.table != "fno_bhavcopy"]
    fno_errors   = [s for s in h.error_sources if s.table == "fno_bhavcopy"]

    if auto_fixable:
        print("ACTION: Run `python -m src.cli fill-gaps 10` to auto-fix the gaps above.")
    if fno_errors:
        missing = sorted(d for s in fno_errors for d in s.critical_missing)
        span = (date.today() - min(missing)).days if missing else 0
        missing_str = ", ".join(str(d) for d in missing)
        print(f"FNO NOTE: {missing_str}")
        if missing and min(missing) >= date(2024, 1, 1):
            # UDiFF zips are served for the whole format era — auto-recoverable.
            reach = max(10, int(span * 5 / 7) + 5)   # calendar span -> weekdays
            print(f"          Auto-recover: python -m src.cli backfill-fno {reach}")
        else:
            print("          Pre-dates the UDiFF format (2024-01-01) — download fo*.zip from")
            print("          nseindia.com/all-reports-derivatives into data/fii_imports/,")
            print("          then run: python -m src.cli import-fno-bhav")
    if not h.has_errors:
        print("All data current.")
    return 1 if h.has_errors else 0


def cmd_import_fno_bhav(_args) -> int:
    """Import old-format NSE FNO bhavcopy zip files (fo*.zip) from data/fii_imports/."""
    from src.ingestion.fno_bhavcopy_fetcher import import_fno_folder
    result = import_fno_folder()
    print(f"FNO import: {result['files_processed']} file(s) processed, "
          f"{result['rows_inserted']} rows upserted.")
    if result["errors"]:
        for err in result["errors"]:
            print(f"  WARNING: {err}")
    return 0


def cmd_import_fii_stats(_args) -> int:
    """Import FII Derivatives Statistics XLS files from data/fii_imports/ folder."""
    from src.ingestion.fii_stats_fetcher import import_fii_stats_folder
    result = import_fii_stats_folder()
    print(f"FII stats import: {result['files_processed']} file(s) processed, "
          f"{result['rows_inserted']} rows upserted.")
    if result["errors"]:
        for err in result["errors"]:
            print(f"  WARNING: {err}")
    return 0


def cmd_backfill_predictions(args) -> int:
    """Seed the prediction memory engine with historical predictions + outcomes."""
    from datetime import date, timedelta
    from src.analytics.memory_engine import backfill_predictions

    days = int(args.days) if args.days else 60
    to_date   = date.today()
    from_date = to_date - timedelta(days=days * 2)   # extra buffer for weekends

    print(f"Backfilling prediction memory for last ~{days} trading days...")
    print("This recomputes the full 24-signal engine for each historical date.")
    print("May take several minutes. Press Ctrl+C to stop.\n")

    result = backfill_predictions(from_date, to_date)

    print(f"  Trading days processed : {result['processed']}")
    print(f"  Predictions stored     : {result['stored']}")
    print(f"  Outcomes filled        : {result['outcomes_filled']}")
    if result.get("errors"):
        print(f"  Errors                 : {result['errors']}")
    print("\nMemory engine seeded. Restart the dashboard to see accuracy stats.")
    return 0


def cmd_fetch_fpi_monthly(_args) -> int:
    """Fetch all days of the current month from NSDL Monthly.aspx (auto, no login needed)."""
    from src.ingestion.fpi_nsdl_fetcher import fetch_fpi_monthly
    result = fetch_fpi_monthly()
    if result["error"]:
        print(f"ERROR: {result['error']}")
        return 1
    print(f"FPI monthly: {result['rows_inserted']} rows for {len(result['dates'])} dates")
    if result["dates"]:
        print(f"  Range: {result['dates'][0]} to {result['dates'][-1]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="src.cli", description="NSE Dashboard CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init-db", help="Initialize database schema")

    bp = sub.add_parser("backfill", help="Backfill N trading days of stock data")
    bp.add_argument("days", nargs="?", default=None, help="Number of trading days (default 60)")

    ibp = sub.add_parser("backfill-indices", help="Backfill N trading days of index data")
    ibp.add_argument("days", nargs="?", default=None, help="Number of trading days (default 120)")

    fbp = sub.add_parser("backfill-fao", help="Backfill F&O participant OI/Volume data")
    fbp.add_argument("days", nargs="?", default=None, help="Number of trading days (default 365)")

    fsp = sub.add_parser("backfill-fii-stats", help="Backfill FII Derivatives Statistics (buy/sell value)")
    fsp.add_argument("days", nargs="?", default=None, help="Number of trading days (default 365)")

    fnop = sub.add_parser("backfill-fno", help="Backfill F&O Bhavcopy (futures+options; UDiFF, up to ~1yr)")
    fnop.add_argument("days", nargs="?", default=None, help="Number of trading days (default 50; up to ~250 = 1yr)")

    gp = sub.add_parser("fill-gaps", help="Fill FAO+FII gaps for recent N days (targeted, fast)")
    gp.add_argument("days", nargs="?", default=None, help="Number of trading days to scan (default 7)")

    sub.add_parser("daily", help="Fetch today's data")
    sub.add_parser("seed-sectors", help="Fetch and seed sector master")
    sub.add_parser("reload-overrides", help="Apply sector_overrides.csv")
    sub.add_parser("import-fpi",
                   help="Import NSDL FPI Excel files from data/fpi_imports/ into DB")
    sub.add_parser("import-fii-stats",
                   help="Import NSE FII Derivatives Statistics XLS files from data/fii_imports/ into DB")
    sub.add_parser("import-fno-bhav",
                   help="Import old-format NSE FNO bhavcopy zip files (fo*.zip) from data/fii_imports/ into DB")
    sub.add_parser("data-health",
                   help="Show completeness status for all data tables")
    sub.add_parser("fetch-fpi-latest",
                   help="Auto-fetch today's FPI data from NSDL (no login needed)")
    bp2 = sub.add_parser("backfill-predictions",
                         help="Seed prediction memory engine with historical predictions + outcomes")
    bp2.add_argument("days", nargs="?", default=None,
                     help="Number of trading days to backfill (default 60)")

    sub.add_parser("fetch-fpi-monthly",
                   help="Auto-fetch current month's FPI data from NSDL (no login needed)")

    args = parser.parse_args()

    handlers = {
        "init-db":             cmd_init_db,
        "backfill":            cmd_backfill,
        "backfill-indices":    cmd_backfill_indices,
        "backfill-fao":        cmd_backfill_fao,
        "backfill-fii-stats":  cmd_backfill_fii_stats,
        "backfill-fno":        cmd_backfill_fno,
        "fill-gaps":           cmd_fill_gaps,
        "daily":               cmd_daily,
        "seed-sectors":        cmd_seed_sectors,
        "reload-overrides":    cmd_reload_overrides,
        "import-fpi":          cmd_import_fpi,
        "import-fii-stats":    cmd_import_fii_stats,
        "import-fno-bhav":     cmd_import_fno_bhav,
        "data-health":         cmd_data_health,
        "fetch-fpi-latest":    cmd_fetch_fpi_latest,
        "fetch-fpi-monthly":      cmd_fetch_fpi_monthly,
        "backfill-predictions":   cmd_backfill_predictions,
    }

    if args.command not in handlers:
        parser.print_help()
        return 1

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
