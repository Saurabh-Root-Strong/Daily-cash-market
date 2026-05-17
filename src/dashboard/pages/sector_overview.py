from datetime import date
import streamlit as st

from src.analytics.sector_aggregator import aggregate_by_sector, get_sector_drilldown, get_sector_history
from src.dashboard.components.charts import (
    sector_dual_axis_chart, sector_trend_chart, contribution_treemap
)
from src.dashboard.components.tables import SECTOR_TABLE_COLUMNS, STOCK_TABLE_COLUMNS


def render(selected_date: date, min_turnover: float) -> None:
    st.header("Sector Overview")

    sector_df = aggregate_by_sector(selected_date, min_turnover_lacs=min_turnover)

    if sector_df.empty:
        st.warning("No sector data available for the selected date. Run a backfill first.")
        return

    fig = sector_dual_axis_chart(sector_df)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this chart"):
        st.markdown("""
        - **Bars (left axis)**: Turnover-weighted average delivery % per sector — high bars = more long-term conviction buying.
        - **Line (right axis)**: Turnover-weighted average price change % — green/red = sector moved up/down.
        - **High delivery + positive price**: accumulation (smart money buying on strength).
        - **Low delivery + positive price**: weak rally, price pushed up without conviction.
        """)

    st.subheader("Sector Detail")
    display_cols = {k: v for k, v in SECTOR_TABLE_COLUMNS.items() if k in sector_df.columns}
    st.dataframe(sector_df, column_config=display_cols, use_container_width=True)

    st.subheader("Drill Down by Sector")
    sectors = sector_df["sector"].tolist()
    selected_sector = st.selectbox("Select sector to drill down", sectors)

    if selected_sector:
        drilldown = get_sector_drilldown(selected_date, selected_sector)
        if not drilldown:
            st.info("No drill-down data for this sector.")
            return

        summary = drilldown["sector_summary"]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Stocks", summary["stock_count"])
        col2.metric("Total Turnover (L)", f"{summary['total_turnover_lacs']:,.0f}")
        col3.metric("Deliv Value (L)", f"{summary['total_deliv_value_lacs']:,.0f}")
        col4.metric("Avg Price Chg", f"{summary['avg_price_change_pct']:.2f}%")
        col5.metric("Avg Deliv %", f"{summary['avg_deliv_per']:.1f}%")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Top by Delivery %", "Top by Delivery Value", "Top by Turnover", "Contribution Treemap"
        ])

        stock_cols = {k: v for k, v in STOCK_TABLE_COLUMNS.items() if k in drilldown["top_by_delivery_pct"].columns}

        with tab1:
            st.dataframe(drilldown["top_by_delivery_pct"], column_config=stock_cols, use_container_width=True)

        with tab2:
            st.dataframe(drilldown["top_by_delivery_value"], column_config=stock_cols, use_container_width=True)

        with tab3:
            st.dataframe(drilldown["top_by_turnover"], column_config=stock_cols, use_container_width=True)

        with tab4:
            contrib_df = drilldown["contribution_table"]
            if not contrib_df.empty and "deliv_value_lacs" in contrib_df.columns:
                treemap = contribution_treemap(contrib_df, selected_sector)
                st.plotly_chart(treemap, use_container_width=True)

        st.subheader(f"{selected_sector} — 60-Day Trend")
        history_df = get_sector_history(selected_sector, days=60)
        if not history_df.empty:
            trend_fig = sector_trend_chart(history_df, selected_sector)
            st.plotly_chart(trend_fig, use_container_width=True)
