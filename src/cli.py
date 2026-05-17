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


def main() -> int:
    parser = argparse.ArgumentParser(prog="src.cli", description="NSE Dashboard CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init-db", help="Initialize database schema")

    bp = sub.add_parser("backfill", help="Backfill N trading days")
    bp.add_argument("days", nargs="?", default=None, help="Number of trading days (default 60)")

    sub.add_parser("daily", help="Fetch today's data")
    sub.add_parser("seed-sectors", help="Fetch and seed sector master")
    sub.add_parser("reload-overrides", help="Apply sector_overrides.csv")

    args = parser.parse_args()

    handlers = {
        "init-db": cmd_init_db,
        "backfill": cmd_backfill,
        "daily": cmd_daily,
        "seed-sectors": cmd_seed_sectors,
        "reload-overrides": cmd_reload_overrides,
    }

    if args.command not in handlers:
        parser.print_help()
        return 1

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
