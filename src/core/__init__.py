"""
Core primitives — no imports from any other src/ layer.
Dependency order: core ← data ← ingestion / analytics ← dashboard
"""
from src.core.config import AppConfig, get_config, PROJECT_ROOT
from src.core.exceptions import (
    NSEDashboardError,
    FetchError,
    ParseError,
    SchemaError,
    DataIngestionError,
    MarketClosedError,
    ConfigurationError,
)
from src.core.logging import get_logger
from src.core.types import TradeDate, Symbol, Sector, Industry, TurnoverLacs, DeliveryPct

__all__ = [
    "AppConfig", "get_config", "PROJECT_ROOT",
    "NSEDashboardError", "FetchError", "ParseError", "SchemaError",
    "DataIngestionError", "MarketClosedError", "ConfigurationError",
    "get_logger",
    "TradeDate", "Symbol", "Sector", "Industry", "TurnoverLacs", "DeliveryPct",
]
