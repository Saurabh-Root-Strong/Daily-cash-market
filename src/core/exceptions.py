"""
Project-wide exception hierarchy.

Raise the most-specific subclass so callers can catch at the right level.
"""
from __future__ import annotations

import datetime

__all__ = [
    "NSEDashboardError",
    "FetchError",
    "ParseError",
    "SchemaError",
    "DataIngestionError",
    "MarketClosedError",
    "ConfigurationError",
]


class NSEDashboardError(Exception):
    """Root exception — catch this to handle any project error."""


class FetchError(NSEDashboardError):
    """HTTP fetch from NSE failed after all retries."""

    def __init__(self, url: str, reason: str, status_code: int | None = None) -> None:
        self.url = url
        self.reason = reason
        self.status_code = status_code
        code_str = f"[{status_code}] " if status_code is not None else ""
        super().__init__(f"Fetch failed {code_str}{url}: {reason}")


class ParseError(NSEDashboardError):
    """Could not parse response into the expected schema."""


class SchemaError(NSEDashboardError):
    """Database DDL or schema migration failed."""


class DataIngestionError(NSEDashboardError):
    """A full ingestion pipeline run failed."""


class MarketClosedError(NSEDashboardError):
    """Market was closed on the requested date — no data expected."""

    def __init__(self, trade_date: datetime.date, reason: str) -> None:
        self.trade_date = trade_date
        self.reason = reason
        super().__init__(f"Market closed on {trade_date}: {reason}")


class ConfigurationError(NSEDashboardError):
    """Invalid or missing application configuration."""
