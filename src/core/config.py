"""
Typed application configuration.

Single source of truth for every tunable value.  Call get_config() anywhere —
it reads settings.yaml once and returns a frozen dataclass tree.  No more raw
dict subscripting like cfg["analytics"]["min_turnover_lacs"].
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigurationError

__all__ = [
    "AppConfig",
    "DatabaseConfig",
    "IngestionConfig",
    "AnalyticsConfig",
    "DashboardConfig",
    "LoggingConfig",
    "SectorSource",
    "get_config",
    "PROJECT_ROOT",
]

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatabaseConfig:
    path: str

    @property
    def resolved_path(self) -> Path:
        return PROJECT_ROOT / self.path


@dataclass(frozen=True)
class IngestionConfig:
    user_agent: str
    retries: int
    timeout: int
    polite_delay: float
    bhavcopy_url: str   # template with {date} placeholder (YYYYMMDD)
    delivery_url: str   # template with {date} placeholder (DDMMYYYY)


@dataclass(frozen=True)
class AnalyticsConfig:
    delivery_avg_window: int
    volume_avg_window: int
    accumulation_threshold: float   # deliv_ratio ≥ this → accumulating
    distribution_threshold: float   # deliv_ratio < this → distributing
    weighting_method: str           # "turnover" | "equal"
    min_turnover_lacs: float


@dataclass(frozen=True)
class DashboardConfig:
    port: int


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_dir: str

    @property
    def resolved_log_dir(self) -> Path:
        return PROJECT_ROOT / self.log_dir


@dataclass(frozen=True)
class SectorSource:
    name: str
    filename: str
    sector: str


@dataclass(frozen=True)
class BackfillConfig:
    trading_days: int


# ── Root config ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppConfig:
    database: DatabaseConfig
    ingestion: IngestionConfig
    analytics: AnalyticsConfig
    dashboard: DashboardConfig
    logging: LoggingConfig
    backfill: BackfillConfig
    sector_sources: list[SectorSource] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        try:
            ing = raw["ingestion"]
            ana = raw["analytics"]
            return cls(
                database=DatabaseConfig(path=raw["database"]["path"]),
                ingestion=IngestionConfig(
                    user_agent=ing["user_agent"],
                    retries=ing["retries"],
                    timeout=ing["timeout"],
                    polite_delay=ing["polite_delay"],
                    bhavcopy_url=ing["bhavcopy_url"],
                    delivery_url=ing["delivery_url"],
                ),
                analytics=AnalyticsConfig(
                    delivery_avg_window=ana["delivery_avg_window"],
                    volume_avg_window=ana["volume_avg_window"],
                    accumulation_threshold=ana["accumulation_threshold"],
                    distribution_threshold=ana["distribution_threshold"],
                    weighting_method=ana["weighting_method"],
                    min_turnover_lacs=ana["min_turnover_lacs"],
                ),
                dashboard=DashboardConfig(
                    port=raw.get("dashboard", {}).get("port", 8501)
                ),
                logging=LoggingConfig(
                    level=raw.get("logging", {}).get("level", "INFO"),
                    log_dir=raw.get("logging", {}).get("log_dir", "logs"),
                ),
                backfill=BackfillConfig(
                    trading_days=raw.get("backfill", {}).get("trading_days", 100)
                ),
                sector_sources=[
                    SectorSource(
                        name=s["name"],
                        filename=s["filename"],
                        sector=s["sector"],
                    )
                    for s in raw.get("sectors", {}).get("sources", [])
                ],
            )
        except KeyError as exc:
            raise ConfigurationError(f"Missing required config key: {exc}") from exc


@functools.lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache settings.yaml.  Call get_config.cache_clear() in tests."""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.from_dict(raw)
