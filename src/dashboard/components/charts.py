import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def sector_dual_axis_chart(sector_df: pd.DataFrame) -> go.Figure:
    if sector_df.empty:
        return go.Figure()

    df = sector_df.sort_values("wtd_deliv_per", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["sector"],
        y=df["wtd_deliv_per"],
        name="Delivery %",
        marker=dict(
            color=df["wtd_deliv_per"],
            colorscale="Blues",
            showscale=False,
        ),
        yaxis="y1",
    ))

    price_colors = df["wtd_price_change_pct"].apply(
        lambda v: "#d62728" if (v is not None and v < 0)
        else ("#2ca02c" if (v is not None and v > 0) else "#7f7f7f")
    )

    fig.add_trace(go.Scatter(
        x=df["sector"],
        y=df["wtd_price_change_pct"],
        name="Price Chg %",
        mode="lines+markers",
        marker=dict(color=price_colors, size=8),
        line=dict(color="orange", width=2),
        yaxis="y2",
    ))

    fig.update_layout(
        title="Sector Overview — Delivery % vs Price Change %",
        xaxis=dict(title="Sector", tickangle=-30),
        yaxis=dict(title="Delivery %", side="left"),
        yaxis2=dict(title="Price Change %", side="right", overlaying="y", zeroline=True),
        legend=dict(orientation="h", y=1.1, x=0),
        height=480,
        margin=dict(b=120),
    )
    return fig


def sector_trend_chart(history_df: pd.DataFrame, sector_name: str) -> go.Figure:
    if history_df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=history_df["trade_date"],
        y=history_df["avg_deliv_per"],
        name="Avg Delivery %",
        marker_color="steelblue",
        yaxis="y1",
    ))

    fig.add_trace(go.Scatter(
        x=history_df["trade_date"],
        y=history_df["avg_price_change_pct"],
        name="Avg Price Chg %",
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(color="orange"),
        yaxis="y2",
    ))

    fig.update_layout(
        title=f"{sector_name} — 60-Day Trend",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Avg Delivery %", side="left"),
        yaxis2=dict(title="Avg Price Chg %", side="right", overlaying="y"),
        legend=dict(orientation="h"),
        height=350,
    )
    return fig


def stock_price_chart(history_df: pd.DataFrame, symbol: str) -> go.Figure:
    if history_df.empty:
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_df["trade_date"],
        y=history_df["close_price"],
        name="Close Price",
        mode="lines+markers",
        line=dict(color="#1f77b4"),
        yaxis="y1",
    ))

    fig.add_trace(go.Bar(
        x=history_df["trade_date"],
        y=history_df["deliv_per"],
        name="Delivery %",
        marker_color="rgba(255,127,14,0.5)",
        yaxis="y2",
    ))

    fig.update_layout(
        title=f"{symbol} — Price & Delivery %",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Close Price (₹)", side="left"),
        yaxis2=dict(title="Delivery %", side="right", overlaying="y"),
        legend=dict(orientation="h"),
        height=400,
    )
    return fig


def contribution_treemap(contribution_df: pd.DataFrame, sector_name: str) -> go.Figure:
    if contribution_df.empty:
        return go.Figure()

    fig = px.treemap(
        contribution_df,
        path=["symbol"],
        values="deliv_value_lacs",
        color="price_change_pct",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=f"{sector_name} — Delivery Value Contribution",
        hover_data=["company_name", "deliv_per", "turnover_lacs"],
    )
    fig.update_layout(height=400)
    return fig
