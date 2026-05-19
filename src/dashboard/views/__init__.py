"""Dashboard views — one render() per page."""
from src.dashboard.views import sector_overview, sector_performance, signals, stock_detail

__all__ = ["sector_overview", "sector_performance", "signals", "stock_detail"]
