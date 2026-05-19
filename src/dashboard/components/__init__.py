"""Dashboard UI components — charts, tables, KPI strips, filter builder."""
from src.dashboard.components.charts import (
    contribution_treemap,
    outlook_bar_chart,
    period_comparison_chart,
    sector_overview_chart,
    sector_trend_chart,
    stock_price_chart,
    sub_sector_chart,
)
from src.dashboard.components.filters import (
    apply_filters,
    render_filter_builder,
    render_filter_summary,
)
from src.dashboard.components.kpi import (
    market_kpi_strip,
    performance_kpi_strip,
    sector_kpi_strip,
    stock_kpi_strip,
)
from src.dashboard.components.tables import (
    SECTOR_TABLE_COLUMNS,
    STOCK_TABLE_COLUMNS,
    to_display_df,
)

__all__ = [
    # charts
    "contribution_treemap",
    "outlook_bar_chart",
    "period_comparison_chart",
    "sector_overview_chart",
    "sector_trend_chart",
    "stock_price_chart",
    "sub_sector_chart",
    # filters
    "apply_filters",
    "render_filter_builder",
    "render_filter_summary",
    # kpi
    "market_kpi_strip",
    "performance_kpi_strip",
    "sector_kpi_strip",
    "stock_kpi_strip",
    # tables
    "SECTOR_TABLE_COLUMNS",
    "STOCK_TABLE_COLUMNS",
    "to_display_df",
]
