"""
Sector Overview page.

Renders aggregate sector metrics, a click-to-drill bar chart, and
a drilldown panel with sub-sector breakdown and top-stock tables.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from src.dashboard.cache.queries import (
    cached_aggregate_by_sector,
    cached_sector_drilldown,
    cached_sector_history,
)
from src.dashboard.components.charts import (
    contribution_treemap,
    sector_overview_chart,
    sector_trend_chart,
    sub_sector_chart,
)
from src.dashboard.components.kpi import market_kpi_strip, sector_kpi_strip
from src.dashboard.components.tables import SECTOR_TABLE_COLUMNS, STOCK_TABLE_COLUMNS, to_display_df


def render(selected_date: date, min_turnover: float) -> None:

    col_title, col_help = st.columns([6, 1])
    with col_title:
        st.subheader("Sector Overview")
    with col_help:
        st.markdown(
            "<span title='Click any bar on the chart to drill into that sector&#39;s "
            "sub-sectors and top stocks.'>&#x2753; How to use</span>",
            unsafe_allow_html=True,
        )

    sector_df = cached_aggregate_by_sector(selected_date, min_turnover)

    if sector_df.empty:
        st.warning("No data for selected date. Run backfill first.")
        return

    market_kpi_strip(sector_df)
    st.markdown("---")

    with st.expander("ℹ️  How to read this chart — click to expand", expanded=False):
        st.markdown("""
        | Element | Meaning |
        |---------|---------|
        | **Blue bars** | Turnover-weighted avg delivery % per sector — taller bar = more conviction buying |
        | **Green dot** | Sector price went **up** today |
        | **Red dot** | Sector price went **down** today |
        | **High bar + green dot** | Classic accumulation — institutions buying on strength |
        | **High bar + red dot** | Buying on dip — delivery is high even as price falls |
        | **Low bar + green dot** | Weak rally — price pushed up without real conviction |
        | **Click a bar** | Drills into that sector's sub-sectors and top stocks |
        """)

    # ── Main chart (click-to-drill) ───────────────────────────────────────────
    event = st.plotly_chart(
        sector_overview_chart(sector_df),
        use_container_width=True,
        on_select="rerun",
        key="sector_chart",
    )

    # ── Resolve selected sector ───────────────────────────────────────────────
    clicked_sector = None
    if event and event.get("selection") and event["selection"].get("points"):
        pt = event["selection"]["points"][0]
        clicked_sector = pt.get("x") or pt.get("label")

    sectors_list = sector_df["sector"].tolist()

    if clicked_sector and clicked_sector in sectors_list:
        selected_sector = clicked_sector
    else:
        selected_sector = st.selectbox(
            "Or select a sector to drill down",
            ["— select —"] + sectors_list,
            index=0,
            help="Click a bar above, or pick a sector from this list",
        )
        if selected_sector == "— select —":
            selected_sector = None

    # ── Drilldown panel ───────────────────────────────────────────────────────
    if selected_sector:
        st.markdown(f"### {selected_sector}")

        row = sector_df[sector_df["sector"] == selected_sector].iloc[0]
        sector_kpi_strip(row)

        drilldown = cached_sector_drilldown(selected_date, selected_sector)

        if not drilldown:
            st.info("No drill-down data available.")
        else:
            sub_fig = sub_sector_chart(drilldown, selected_sector)
            if sub_fig.data:
                st.plotly_chart(sub_fig, use_container_width=True)

            tab1, tab2, tab3, tab4 = st.tabs([
                "📦 Top Delivery %",
                "💰 Top Delivery Value",
                "📊 Top Turnover",
                "🗺️ Treemap",
            ])

            top_deliv_pct_df   = to_display_df(drilldown["top_by_delivery_pct"],   STOCK_TABLE_COLUMNS)
            top_deliv_value_df = to_display_df(drilldown["top_by_delivery_value"],  STOCK_TABLE_COLUMNS)
            top_turnover_df    = to_display_df(drilldown["top_by_turnover"],        STOCK_TABLE_COLUMNS)
            stock_cols = {k: v for k, v in STOCK_TABLE_COLUMNS.items()
                          if k in top_deliv_pct_df.columns}

            with tab1:
                st.caption("Stocks with highest delivery % today — strongest conviction buying")
                st.dataframe(top_deliv_pct_df,   column_config=stock_cols,
                             use_container_width=True, hide_index=True)
            with tab2:
                st.caption("Stocks where the most ₹ value was actually delivered (taken home)")
                st.dataframe(top_deliv_value_df, column_config=stock_cols,
                             use_container_width=True, hide_index=True)
            with tab3:
                st.caption("Most actively traded stocks by turnover in this sector")
                st.dataframe(top_turnover_df,    column_config=stock_cols,
                             use_container_width=True, hide_index=True)
            with tab4:
                contrib = drilldown["contribution_table"]
                if not contrib.empty and "deliv_value_lacs" in contrib.columns:
                    st.caption("Box size = delivery value, color = price change (green = up, red = down)")
                    st.plotly_chart(contribution_treemap(contrib, selected_sector),
                                    use_container_width=True)

        with st.expander(f"📈 {selected_sector} — 60-Day Trend", expanded=False):
            hist = cached_sector_history(selected_sector, days=60)
            if not hist.empty:
                st.plotly_chart(sector_trend_chart(hist, selected_sector),
                                use_container_width=True)

    # ── Full sector table ─────────────────────────────────────────────────────
    with st.expander("📋 Full Sector Summary Table", expanded=False):
        sector_display_df = to_display_df(sector_df, SECTOR_TABLE_COLUMNS)
        show_cols = {k: v for k, v in SECTOR_TABLE_COLUMNS.items()
                     if k in sector_display_df.columns}
        st.dataframe(sector_display_df, column_config=show_cols,
                     use_container_width=True, hide_index=True)
