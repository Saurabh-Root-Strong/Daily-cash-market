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


def cmd_daily(_args) -> int:
    from src.ingestion.orchestrator import run_daily_job
    run_daily_job()
    return 0


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
    print("Source: NSE Archives — F&O-Bhavcopy File (DAT). Press Ctrl+C to stop.")
    run_fno_backfill(days=days)
    print("FNO backfill complete.")
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

    fnop = sub.add_parser("backfill-fno", help="Backfill F&O Bhavcopy DAT data (futures+options)")
    fnop.add_argument("days", nargs="?", default=None, help="Number of trading days (default 50)")

    sub.add_parser("daily", help="Fetch today's data")
    sub.add_parser("seed-sectors", help="Fetch and seed sector master")
    sub.add_parser("reload-overrides", help="Apply sector_overrides.csv")

    args = parser.parse_args()

    handlers = {
        "init-db":             cmd_init_db,
        "backfill":            cmd_backfill,
        "backfill-indices":    cmd_backfill_indices,
        "backfill-fao":        cmd_backfill_fao,
        "backfill-fii-stats":  cmd_backfill_fii_stats,
        "backfill-fno":        cmd_backfill_fno,
        "daily":               cmd_daily,
        "seed-sectors":        cmd_seed_sectors,
        "reload-overrides":    cmd_reload_overrides,
    }

    if args.command not in handlers:
        parser.print_help()
        return 1

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
