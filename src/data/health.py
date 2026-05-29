"""
Data health check — scans all tables for gaps against bhavcopy trading days.

Grace periods:
  fpi_nsdl_flows : 2 days  (NSDL publishes 1-2 days late — expected, no alert)
  everything else: 0 days  (direct NSE archive URLs, same-day available)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List

from src.data.repository import query_dataframe

__all__ = ["DataHealth", "SourceStatus", "run_health_check"]

# ── Configuration ─────────────────────────────────────────────────────────────

_SOURCES = {
    "daily_data":            ("Bhavcopy",       0),
    "fao_participant":       ("FAO Participant", 0),
    "fii_derivatives_stats": ("FII Statistics",  0),
    "fno_bhavcopy":          ("FNO Bhavcopy",    0),
    "index_data":            ("Index Data",      0),
    "fpi_nsdl_flows":        ("FPI Flows",       2),   # NSDL lag — 2-day grace
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SourceStatus:
    table:         str
    label:         str
    latest_date:   date | None
    missing_dates: List[date]   # trading days with bhavcopy but missing here
    grace_days:    int

    @property
    def critical_missing(self) -> List[date]:
        """Missing dates outside the grace period — these need fixing."""
        today = date.today()
        return [d for d in self.missing_dates
                if (today - (d.date() if hasattr(d, "date") else d)).days > self.grace_days]

    @property
    def level(self) -> str:
        """'ok' | 'warn' | 'error'"""
        if not self.critical_missing:
            if self.missing_dates:
                return "warn"   # within grace period
            return "ok"
        return "error"

    @property
    def status_icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️", "error": "❌"}[self.level]


@dataclass
class DataHealth:
    as_of:          date
    trading_days:   List[date]
    sources:        dict[str, SourceStatus] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return all(s.level in ("ok", "warn") for s in self.sources.values())

    @property
    def has_errors(self) -> bool:
        return any(s.level == "error" for s in self.sources.values())

    @property
    def error_sources(self) -> List[SourceStatus]:
        return [s for s in self.sources.values() if s.level == "error"]

    @property
    def warn_sources(self) -> List[SourceStatus]:
        return [s for s in self.sources.values() if s.level == "warn"]


# ── Core function ─────────────────────────────────────────────────────────────

def run_health_check(lookback_days: int = 10) -> DataHealth:
    """
    Check all data tables for the last `lookback_days` trading days.
    Uses bhavcopy (daily_data) as the ground truth for which days are trading days.
    """
    today = date.today()

    # Ground truth: all trading days present in daily_data
    cutoff = today - timedelta(days=lookback_days * 2)   # extra buffer for weekends
    all_trading = query_dataframe(
        f"SELECT DISTINCT trade_date FROM daily_data "
        f"WHERE trade_date >= '{cutoff}' ORDER BY trade_date DESC "
        f"LIMIT {lookback_days}"
    )["trade_date"].tolist()

    trading_set = set(all_trading)

    health = DataHealth(as_of=today, trading_days=sorted(all_trading))

    for table, (label, grace) in _SOURCES.items():
        try:
            if not all_trading:
                health.sources[table] = SourceStatus(
                    table=table, label=label, latest_date=None,
                    missing_dates=[], grace_days=grace
                )
                continue

            cutoff_str = str(min(all_trading))
            raw = query_dataframe(
                f"SELECT DISTINCT trade_date FROM {table} "
                f"WHERE trade_date >= '{cutoff_str}'"
            )["trade_date"].tolist()
            present = {(d.date() if hasattr(d, "date") else d) for d in raw}
            trading_norm = {(d.date() if hasattr(d, "date") else d) for d in trading_set}
            latest = max(present) if present else None
            missing = sorted(d for d in trading_norm if d not in present)

            health.sources[table] = SourceStatus(
                table=table, label=label, latest_date=latest,
                missing_dates=missing, grace_days=grace
            )
        except Exception as exc:
            # Table might not exist yet
            health.sources[table] = SourceStatus(
                table=table, label=label, latest_date=None,
                missing_dates=list(all_trading), grace_days=grace
            )

    return health
