"""
All Plotly chart factory functions for the dashboard.

Every function is pure: takes DataFrames / scalars, returns go.Figure.
No Streamlit calls here — views call st.plotly_chart() themselves.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dashboard.constants import (
    ACCENT_COLOR,
    GRID_COLOR,
    NEGATIVE_COLOR,
    NEUTRAL_COLOR,
    PAPER_BG,
    PERIOD_COLORS,
    PERIOD_LABELS,
    PLOT_BG,
    POSITIVE_COLOR,
    SIGNAL_COLORS,
    PRICE_KEYS,
    DELIV_KEYS,
)


# ── Color helpers ─────────────────────────────────────────────────────────────
def _sign_color(val) -> str:
    if val is None or (hasattr(val, "__float__") and pd.isna(float(val))):
        return NEUTRAL_COLOR
    return POSITIVE_COLOR if float(val) > 0 else (NEGATIVE_COLOR if float(val) < 0 else NEUTRAL_COLOR)


def _sign_colors(series: pd.Series) -> list[str]:
    return [_sign_color(v) for v in series]


# ── Sector Overview ───────────────────────────────────────────────────────────
def sector_overview_chart(sector_df: pd.DataFrame) -> go.Figure:
    """Main bar+scatter chart for the Sector Overview page (click-to-drill)."""
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
        hovertemplate="<b>%{x}</b><br>Delivery %: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["sector"],
        y=df["wtd_price_change_pct"],
        name="Price Chg %",
        mode="markers+lines",
        marker=dict(
            color=_sign_colors(df["wtd_price_change_pct"]),
            size=10, symbol="circle",
            line=dict(width=1.5, color="white"),
        ),
        line=dict(color=ACCENT_COLOR, width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Price Chg: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        dragmode=False, clickmode="event",
        xaxis=dict(tickangle=-35, tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title="Delivery %",    side="left",  showgrid=True, gridcolor=GRID_COLOR),
        yaxis2=dict(title="Price Change %", side="right", overlaying="y",
                    zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", showgrid=False),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=13)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(b=140, t=20, l=60, r=60), height=460,
        hovermode="x unified", bargap=0.35,
    )
    return fig


def sub_sector_chart(drilldown: dict, sector_name: str) -> go.Figure:
    """Sub-sector bar chart using turnover-weighted delivery % (the correct conviction metric).

    Simple avg(deliv%) is misleading — a ₹5000 stock at 60% delivery delivers
    far more real money than a ₹50 stock at 70%. Weighting by turnover (price×qty)
    gives the true picture of where institutional money is actually being committed.
    """
    sub = drilldown.get("subsector_summary", pd.DataFrame())
    if sub.empty:
        return go.Figure()

    deliv_val_cr = sub["total_deliv_value_lacs"] / 100
    hover = (
        "<b>%{x}</b><br>"
        "Wtd Delivery %: <b>%{y:.1f}%</b><br>"
        "Delivery Value: ₹" + deliv_val_cr.round(1).astype(str) + " Cr<br>"
        "Stocks: " + sub["stock_count"].astype(str) +
        " &nbsp;|&nbsp; Simple Avg: " + sub["simple_deliv_per"].round(1).astype(str) + "%"
        "<extra></extra>"
    )

    fig = go.Figure(go.Bar(
        x=sub["industry"],
        y=sub["wtd_deliv_per"],
        marker_color=_sign_colors(sub["avg_price_chg"]),
        hovertemplate=hover,
        text=sub.apply(
            lambda r: f"₹{r['total_deliv_value_lacs']/100:.0f} Cr  |  {int(r['stock_count'])} stocks",
            axis=1,
        ),
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=f"{sector_name} — Sub-Sector Wtd Delivery % (by ₹ Traded Value)",
        xaxis=dict(tickangle=-25, tickfont=dict(size=11)),
        yaxis=dict(title="Turnover-Weighted Delivery %"),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        height=380, margin=dict(b=120, t=50), bargap=0.4,
    )
    return fig


# ── Sector Trend ──────────────────────────────────────────────────────────────
def sector_trend_chart(history_df: pd.DataFrame, sector_name: str) -> go.Figure:
    """60-day bar (delivery) + scatter (price) time-series for a single sector."""
    if history_df.empty:
        return go.Figure()

    df        = history_df.copy()
    bar_clrs  = _sign_colors(df["avg_price_change_pct"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["trade_date"], y=df["avg_deliv_per"],
        name="Wtd Delivery %",
        marker=dict(color=bar_clrs, opacity=0.85), yaxis="y1",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Delivery %: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["trade_date"], y=df["avg_price_change_pct"],
        name="Wtd Price Chg %",
        mode="lines+markers",
        marker=dict(size=6, color=bar_clrs, line=dict(width=1, color="white")),
        line=dict(color="rgba(255,165,0,0.7)", width=1.5, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Price Chg: %{y:+.2f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickformat="%d %b", tickangle=-30, showgrid=False, type="date"),
        yaxis=dict(title="Wtd Delivery %", side="left",
                   showgrid=True, gridcolor=GRID_COLOR, rangemode="tozero"),
        yaxis2=dict(title="Wtd Price Chg %", side="right", overlaying="y",
                    zeroline=True, zerolinecolor="rgba(255,255,255,0.4)",
                    zerolinewidth=1.5, showgrid=False),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=12)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(t=10, b=60, l=60, r=60), height=360,
        hovermode="x unified", bargap=0.25,
    )
    return fig


# ── Stock Detail ──────────────────────────────────────────────────────────────
def stock_price_chart(history_df: pd.DataFrame, symbol: str) -> go.Figure:
    """60-day close price (line) + delivery % (bar) dual-axis chart for one stock."""
    if history_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df["trade_date"], y=history_df["close_price"],
        name="Close Price",
        mode="lines+markers",
        line=dict(color="#1f77b4"), yaxis="y1",
    ))
    fig.add_trace(go.Bar(
        x=history_df["trade_date"], y=history_df["deliv_per"],
        name="Delivery %",
        marker_color="rgba(255,127,14,0.5)", yaxis="y2",
    ))
    fig.update_layout(
        title=f"{symbol} — Price & Delivery %",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Close Price (₹)", side="left"),
        yaxis2=dict(title="Delivery %", side="right", overlaying="y"),
        legend=dict(orientation="h"), height=400,
    )
    return fig


# ── Sector Drilldown ──────────────────────────────────────────────────────────
def contribution_treemap(contribution_df: pd.DataFrame, sector_name: str) -> go.Figure:
    """Treemap: box size = delivery value, color = price change (green/red)."""
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


# ── Sector Performance ────────────────────────────────────────────────────────
def outlook_bar_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of composite sector outlook scores."""
    top    = scored_df.head(10)
    colors = top["Signal"].map(SIGNAL_COLORS).fillna(NEUTRAL_COLOR).tolist()

    fig = go.Figure(go.Bar(
        x=top["Score"], y=top["sector"], orientation="h",
        marker_color=colors, text=top["Signal"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Composite Score (0–100)", range=[0, 115]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(l=160, r=200, t=20, b=40), height=360,
    )
    return fig


def period_comparison_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """Grouped horizontal bar chart comparing 1W / 2W / 1M / 3M for all sectors."""
    col_map = {
        "Delivery %":     tuple(DELIV_KEYS),
        "Price Change %": tuple(PRICE_KEYS),
    }
    cols   = [c for c in col_map[metric] if c in df.columns]
    labels = PERIOD_LABELS[: len(cols)]
    sdf    = df.sort_values(cols[-1], ascending=True).fillna(0)

    fig = go.Figure()
    for col, name, color in zip(cols, labels, PERIOD_COLORS):
        fig.add_trace(go.Bar(
            y=sdf["sector"], x=sdf[col], name=name, orientation="h",
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(title=metric, zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.3)", ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=12)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(l=160, r=40, t=40, b=50),
        height=max(440, len(df) * 28 + 120),
        bargap=0.18, bargroupgap=0.06,
    )
    return fig


# ── Legacy alias (kept for any code that imported the old name) ───────────────
sector_dual_axis_chart = sector_overview_chart
