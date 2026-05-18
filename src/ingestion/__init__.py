"""
Ingestion layer public API.

Entry points for the CLI and orchestrator.  Never import from sub-modules
directly — import from here so internal paths can change freely.
"""
from src.ingestion.http_client import NSEHttpClient
from src.ingestion.base import BaseFetcher
from src.ingestion.orchestrator import (
    fetch_one_date,
    run_daily_job,
    run_backfill,
    seed_sectors,
)

__all__ = [
    "NSEHttpClient",
    "BaseFetcher",
    "fetch_one_date",
    "run_daily_job",
    "run_backfill",
    "seed_sectors",
]
