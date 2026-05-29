from datetime import date
import streamlit as st
import plotly.graph_objects as go

import pandas as pd

from src.analytics.sector_aggregator import aggregate_by_sector, get_sector_drilldown, get_sector_history
from src.dashboard.components.charts import sector_trend_chart, contribution_treemap
from src.dashboard.components.tables import SECTOR_TABLE_COLUMNS, STOCK_TABLE_COLUMNS, to_display_df

# ── helpers ───────────────────────────────────────────────────────────────────

def _sector_bar_chart(sector_df: pd.DataFrame) -> go.Figure:
    df = sector_df.sort_values("wtd_deliv_per", ascending=False).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["sector"],
        y=df["wtd_deliv_per"],
        name="Delivery %",
        marker=dict(
            color=df["wtd_deliv_per"],
            colorscale="Blues",
            showscale=False,
            line=dict(width=0),
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Delivery %: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Color price-change markers red/green/grey
    colors = df["wtd_price_change_pct"].apply(
        lambda v: "#2ca02c" if (v is not None and v > 0)
        else ("#d62728" if (v is not None and v < 0) else "#888")
    ).tolist()

    fig.add_trace(go.Scatter(
        x=df["sector"],
        y=df["wtd_price_change_pct"],
        name="Price Chg %",
        mode="markers+lines",
        marker=dict(color=colors, size=10, symbol="circle",
                    line=dict(width=1.5, color="white")),
        line=dict(color="rgba(255,165,0,0.4)", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Price Chg: %{y:.2f}%<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        dragmode=False,
        clickmode="event",
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=12),
            showgrid=False,
        ),
        yaxis=dict(title="Delivery %", side="left", showgrid=True,
                   gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title="Price Change %", side="right", overlaying="y",
                    zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
                    showgrid=False),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=13)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(b=140, t=20, l=60, r=60),
        height=460,
        hovermode="x unified",
        bargap=0.35,
    )
    return fig


def _sub_sector_chart(drilldown: dict, sector_name: str) -> go.Figure:
    stocks = drilldown.get("top_by_delivery_pct", pd.DataFrame())
    if stocks.empty:
        return go.Figure()

    sub = (
        stocks.groupby("industry", as_index=False)
        .agg(
            avg_deliv_per=("deliv_per", "mean"),
            avg_price_chg=("price_change_pct", "mean"),
            stock_count=("symbol", "count"),
        )
        .sort_values("avg_deliv_per", ascending=False)
    )

    colors = sub["avg_price_chg"].apply(
        lambda v: "#2ca02c" if v > 0 else ("#d62728" if v < 0 else "#888")
    ).tolist()

    fig = go.Figure(go.Bar(
        x=sub["industry"],
        y=sub["avg_deliv_per"],
        marker_color=colors,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Avg Delivery %: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
        text=sub["stock_count"].apply(lambda n: f"{n} stocks"),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{sector_name} — Sub-Sector Delivery %",
        xaxis=dict(tickangle=-25, tickfont=dict(size=11)),
        yaxis=dict(title="Avg Delivery %"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(b=120, t=50),
        bargap=0.4,
    )
    return fig


# ── main render ───────────────────────────────────────────────────────────────

def render(selected_date: date, min_turnover: float) -> None:

    # ── Title row with "?" help ───────────────────────────────────────────────
    col_title, col_help = st.columns([6, 1])
    with col_title:
        st.subheader("Sector Overview")
    with col_help:
        st.markdown(
            "<span title='Click any bar on the chart to drill into that sector&#39;s sub-sectors and top stocks.'>"
            "&#x2753; How to use</span>",
            unsafe_allow_html=True,
        )

    sector_df = aggregate_by_sector(selected_date, min_turnover_lacs=min_turnover)

    if sector_df.empty:
        st.warning("No data for selected date. Run backfill first.")
        return

    # ── KPI strip ─────────────────────────────────────────────────────────────
    total_stocks  = int(sector_df["stock_count"].sum())
    total_to      = sector_df["total_turnover_lacs"].sum()
    avg_deliv     = (sector_df["wtd_deliv_per"] * sector_df["stock_count"]).sum() / total_stocks
    avg_price_chg = (sector_df["wtd_price_change_pct"] * sector_df["stock_count"]).sum() / total_stocks
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sectors", len(sector_df),
              help="Number of broad market sectors with data today")
    k2.metric("Stocks", total_stocks,
              help="Total stocks in today's universe (min turnover filter applied)")
    k3.metric("Total Traded Value", f"₹{total_to/100:.0f} Cr",
              help="Combined traded value (turnover) across all sectors today")
    k4.metric("Avg Delivery %", f"{avg_deliv:.1f}%",
              help="Turnover-weighted average delivery % across all stocks — higher = more conviction buying")
    k5.metric("Avg Price Chg", f"{avg_price_chg:+.2f}%",
              help="Turnover-weighted average price change across all stocks today")

    st.markdown("---")

    # ── Legend / reading guide ────────────────────────────────────────────────
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
    fig = _sector_bar_chart(sector_df)

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="sector_chart",
    )

    # ── Determine selected sector (click or selectbox fallback) ───────────────
    clicked_sector = None
    if event and event.get("selection") and event["selection"].get("points"):
        pt = event["selection"]["points"][0]
        # bar charts use "label", scatter uses "x"
        clicked_sector = pt.get("x") or pt.get("label")

    sectors_list = sector_df["sector"].tolist()

    if clicked_sector and clicked_sector in sectors_list:
        # auto-scroll hint
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

    # ── Drill-down panel ──────────────────────────────────────────────────────
    if selected_sector:
        st.markdown(f"### {selected_sector}")

        row = sector_df[sector_df["sector"] == selected_sector].iloc[0]
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Stocks", int(row["stock_count"]),
                  help="Number of stocks in this sector today")
        d2.metric("Wtd Delivery %", f"{row['wtd_deliv_per']:.1f}%",
                  help="Turnover-weighted average delivery % for this sector")
        d3.metric("Wtd Price Chg", f"{row['wtd_price_change_pct']:+.2f}%",
                  help="Turnover-weighted average price change for this sector")
        d4.metric("Traded Value", f"₹{row['total_turnover_lacs']/100:.0f} Cr",
                  help="Total traded value (turnover) in this sector today")
        d5.metric("Top Delivery Stock", str(row["top_delivery_symbol"]) if row["top_delivery_symbol"] else "—",
                  help="Stock with the highest delivery % in this sector today")

        drilldown = get_sector_drilldown(selected_date, selected_sector)

        if not drilldown:
            st.info("No drill-down data available.")
        else:
            # Sub-sector bar chart
            sub_fig = _sub_sector_chart(drilldown, selected_sector)
            if sub_fig.data:
                st.plotly_chart(sub_fig, use_container_width=True)

            # Tabs for stock tables
            tab1, tab2, tab3, tab4 = st.tabs([
                "📦 Top Delivery %",
                "💰 Top Delivery Value",
                "📊 Top Turnover",
                "🗺️ Treemap",
            ])

            top_deliv_pct_df   = to_display_df(drilldown["top_by_delivery_pct"],   STOCK_TABLE_COLUMNS)
            top_deliv_value_df = to_display_df(drilldown["top_by_delivery_value"], STOCK_TABLE_COLUMNS)
            top_turnover_df    = to_display_df(drilldown["top_by_turnover"],       STOCK_TABLE_COLUMNS)

            stock_cols = {k: v for k, v in STOCK_TABLE_COLUMNS.items()
                         if k in top_deliv_pct_df.columns}

            with tab1:
                st.caption("Stocks with highest delivery % today — strongest conviction buying")
                st.dataframe(top_deliv_pct_df,
                             column_config=stock_cols, use_container_width=True, hide_index=True)

            with tab2:
                st.caption("Stocks where the most ₹ value was actually delivered (taken home)")
                st.dataframe(top_deliv_value_df,
                             column_config=stock_cols, use_container_width=True, hide_index=True)

            with tab3:
                st.caption("Most actively traded stocks by turnover in this sector")
                st.dataframe(top_turnover_df,
                             column_config=stock_cols, use_container_width=True, hide_index=True)

            with tab4:
                contrib = drilldown["contribution_table"]
                if not contrib.empty and "deliv_value_lacs" in contrib.columns:
                    st.caption("Box size = delivery value, color = price change (green = up, red = down)")
                    st.plotly_chart(
                        contribution_treemap(contrib, selected_sector),
                        use_container_width=True,
                    )

        # 60-day trend
        with st.expander(f"📈 {selected_sector} — 60-Day Trend", expanded=False):
            hist = get_sector_history(selected_sector, days=60)
            if not hist.empty:
                st.plotly_chart(sector_trend_chart(hist, selected_sector),
                                use_container_width=True)

    # ── Full sector table (collapsed by default) ──────────────────────────────
    with st.expander("📋 Full Sector Summary Table", expanded=False):
        sector_display_df = to_display_df(sector_df, SECTOR_TABLE_COLUMNS)
        show_cols = {k: v for k, v in SECTOR_TABLE_COLUMNS.items() if k in sector_display_df.columns}
        st.dataframe(sector_display_df, column_config=show_cols,
                     use_container_width=True, hide_index=True)
