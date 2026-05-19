"""Stock Detail page — single stock price history and metrics."""
from __future__ import annotations

from datetime import date

import streamlit as st

from src.dashboard.cache.queries import cached_stock_history, cached_stock_metrics
from src.dashboard.components.charts import stock_price_chart
from src.dashboard.components.kpi import stock_kpi_strip
from src.dashboard.components.tables import STOCK_TABLE_COLUMNS, to_display_df


def render(selected_date: date, min_turnover: float) -> None:
    st.header("Stock Detail")

    metrics_df = cached_stock_metrics(selected_date, min_turnover)

    if metrics_df.empty:
        st.warning("No stock data for selected date. Run a backfill first.")
        return

    symbols         = sorted(metrics_df["symbol"].unique().tolist())
    selected_symbol = st.selectbox("Select stock", symbols)

    if not selected_symbol:
        return

    row = metrics_df[metrics_df["symbol"] == selected_symbol]
    if row.empty:
        st.info("No data for this stock on selected date.")
        return

    stock_kpi_strip(row.iloc[0])

    history_df = cached_stock_history(selected_symbol, days=60)
    if not history_df.empty:
        st.plotly_chart(stock_price_chart(history_df, selected_symbol),
                        use_container_width=True)

    with st.expander("Raw data"):
        display = to_display_df(row, STOCK_TABLE_COLUMNS)
        show_cols = {k: v for k, v in STOCK_TABLE_COLUMNS.items() if k in display.columns}
        st.dataframe(display, column_config=show_cols, use_container_width=True, hide_index=True)
