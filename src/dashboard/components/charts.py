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
    """Horizontal bar chart of ALL sector outlook scores.

    All sectors shown (not top-10). Sorted score high→low.
    A dotted divider separates positive signals from Neutral/Distributing.
    Rich hover shows DV Ratio, Z-Score, Breadth, and price context.
    """
    if scored_df.empty:
        return go.Figure()

    # Sort score high→low; for horizontal bar Plotly renders bottom=last row,
    # so use ascending=True to get highest score at the top visually.
    df = scored_df.sort_values("Score", ascending=True).copy()
    colors = df["Signal"].map(SIGNAL_COLORS).fillna(NEUTRAL_COLOR).tolist()

    def _hover(row) -> str:
        dv  = row.get("dv_ratio",          float("nan"))
        z   = row.get("z_score",           float("nan"))
        br  = row.get("breadth",           float("nan"))
        pm  = row.get("2W_price_chg_pct",  row.get("1W_price_chg_pct", float("nan")))
        dv_s = f"{dv:.2f}×" if not pd.isna(dv) else "—"
        z_s  = f"{z:+.1f}σ" if not pd.isna(z)  else "—"
        br_s = f"{br*100:.0f}%" if not pd.isna(br) else "—"
        pm_s = f"{pm:+.1f}%"   if not pd.isna(pm) else "—"
        return (
            f"<b>{row['sector']}</b>  {row['Signal']}<br>"
            f"Score: <b>{row['Score']:.0f}/100</b><br>"
            f"─────────────────<br>"
            f"DV Ratio: <b>{dv_s}</b>  — delivery vs own 100D avg<br>"
            f"Z-Score:  <b>{z_s}</b>  — statistical abnormality<br>"
            f"Breadth:  <b>{br_s}</b>  — stocks above own norm<br>"
            f"2W Price: <b>{pm_s}</b>"
        )

    hover_texts = [_hover(row) for _, row in df.iterrows()]

    fig = go.Figure(go.Bar(
        x=df["Score"],
        y=df["sector"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        opacity=0.88,
        text=df["Signal"],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover_texts,
    ))

    # Dotted divider between positive and neutral/negative signals
    avoid_signals = {"🔴 Distributing", "⚪ Neutral"}
    positive_rows = [i for i, sig in enumerate(df["Signal"].values) if sig not in avoid_signals]
    if positive_rows:
        divider_y = positive_rows[-1] + 0.5  # between last positive and first neutral
        fig.add_hline(
            y=divider_y,
            line_dash="dot", line_width=1.5,
            line_color="rgba(255,255,255,0.28)",
        )
        fig.add_annotation(
            x=119, y=divider_y,
            text="<b>↑ Buy signals  |  Neutral/Avoid ↓</b>",
            showarrow=False, font=dict(size=9, color="rgba(255,255,255,0.40)"),
            xanchor="right", yanchor="bottom",
        )

    n = len(df)
    fig.update_layout(
        xaxis=dict(
            title="Composite Score (0–100)  ·  Relative rank within today's sector universe",
            range=[0, 130],
            tickfont=dict(size=10),
        ),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(l=180, r=130, t=15, b=50),
        height=max(380, n * 26 + 60),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=12, bordercolor="rgba(255,255,255,0.2)"),
    )
    return fig


def signal_bar_chart(
    df: pd.DataFrame,
    col: str,
    x_title: str,
    fmt: str,
    ref_val: float | None,
    thresholds: list[tuple[float, str]],
) -> go.Figure:
    """Single-metric horizontal bar — sorted by signal strength, colored by threshold.

    thresholds: list of (min_value, color) sorted descending — first match wins.
    """
    sdf = df.dropna(subset=[col]).sort_values(col, ascending=True).copy()

    sorted_t = sorted(thresholds, key=lambda t: t[0], reverse=True)

    def _color(v: float) -> str:
        for threshold, color in sorted_t:
            if v >= threshold:
                return color
        return sorted_t[-1][1]

    colors = [_color(float(v)) for v in sdf[col]]
    texts  = [fmt.format(v) for v in sdf[col]]

    fig = go.Figure(go.Bar(
        y=sdf["sector"], x=sdf[col], orientation="h",
        marker_color=colors, opacity=0.9,
        text=texts, textposition="outside", textfont=dict(size=10),
        hovertemplate=f"<b>%{{y}}</b><br>{x_title}: %{{text}}<extra></extra>",
    ))

    if ref_val is not None:
        fig.add_vline(
            x=ref_val,
            line_dash="dash", line_width=1.5,
            line_color="rgba(255,255,255,0.35)",
            annotation_text=f"avg ({fmt.format(ref_val)})",
            annotation_position="top right",
            annotation_font=dict(size=10, color="rgba(255,255,255,0.5)"),
        )

    fig.update_layout(
        xaxis=dict(title=x_title, zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.25)"),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        margin=dict(l=185, r=110, t=20, b=50),
        height=max(460, len(sdf) * 22 + 110),
    )
    return fig


def period_comparison_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """Grouped horizontal bar chart comparing 1W / 2W / 1M / 3M for all sectors.
    Sorted by 1W (most recent period) so the freshest signal drives ordering.
    """
    col_map = {
        "Delivered Value (Cr)": tuple(DELIV_KEYS),
        "Price Change %":       tuple(PRICE_KEYS),
    }
    cols   = [c for c in col_map[metric] if c in df.columns]
    labels = PERIOD_LABELS[: len(cols)]
    # Sort by 1W (cols[0]) — most recent period, not 3M which always favours large sectors
    sdf    = df.sort_values(cols[0], ascending=True).fillna(0)
    is_cr  = metric == "Delivered Value (Cr)"

    fig = go.Figure()
    for col, name, color in zip(cols, labels, PERIOD_COLORS):
        hover = (
            f"<b>%{{y}}</b><br>{name}: ₹%{{x:.0f}} Cr<extra></extra>"
            if is_cr else
            f"<b>%{{y}}</b><br>{name}: %{{x:.2f}}%<extra></extra>"
        )
        fig.add_trace(go.Bar(
            y=sdf["sector"], x=sdf[col], name=name, orientation="h",
            marker_color=color, opacity=0.85,
            hovertemplate=hover,
        ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(title=metric, zeroline=True,
                   zerolinecolor="rgba(255,255,255,0.3)",
                   ticksuffix=" Cr" if is_cr else "%"),
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
