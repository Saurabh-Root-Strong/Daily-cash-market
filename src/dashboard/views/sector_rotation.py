"""
Sector Rotation — Smart Money / Institutional Activity Tracker.

Answers: WHERE are institutions putting money (short + long term)?
         WHERE are they quietly exiting before retail notices?

Signal logic: Rising delivery Z-Score = abnormally high institutional activity.
Delivery Z-Score >= 1σ above 100D norm = smart money entering.
Delivery Z-Score <= -0.5σ = institutions reducing exposure.
Combined with 1W cumulative price direction → four quadrant classification.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.cache.queries import (
    cached_rotation_clock_backtest,
    cached_sector_rotation,
    cached_sector_rotation_history,
    cached_sector_rotation_custom_range,
    cached_sector_rotation_timeframe,
    cached_sector_rs_custom_range,
    cached_sector_stocks_custom_range,
    cached_sector_stocks_rotation,
    cached_fno_positioning_by_symbol,
    cached_sector_fno_aggregate,
)
from src.dashboard.constants import NEGATIVE_COLOR, POSITIVE_COLOR, PLOT_BG, PAPER_BG, GRID_COLOR
from src.dashboard.components.charts import hex_to_rgba as _hex_to_rgba  # deduped helper


_SIGNAL_META = {
    "🔥 Secret Accumulation":   {"color": "#00c853", "rank": 0, "invest": True},
    "✅ Confirmed Accumulation": {"color": "#69f0ae", "rank": 1, "invest": True},
    "👀 Early Accumulation":     {"color": "#b9f6ca", "rank": 2, "invest": True},
    "📊 Volume Spike":           {"color": "#ffd600", "rank": 3, "invest": False},
    "⚖️ Neutral":               {"color": "#888888", "rank": 4, "invest": False},
    "📉 Weakening":             {"color": "#ffab40", "rank": 5, "invest": False},
    "⚠️ Distribution Trap":     {"color": "#ff6d00", "rank": 6, "invest": False},
    "❌ Active Selling":        {"color": "#d50000", "rank": 7, "invest": False},
}

_PHASE_META = {
    "Leading":   {"color": "#00c853", "label": "💰 Leading",   "desc": "Delivery rising + price rising — institutions & price aligned"},
    "Improving": {"color": "#40c4ff", "label": "🔍 Improving", "desc": "Delivery rising + price falling — contrarian accumulation zone"},
    "Neutral":   {"color": "#888888", "label": "⚖️ Neutral",   "desc": "No clear directional momentum"},
    "Weakening": {"color": "#ff9100", "label": "⚠️ Weakening", "desc": "Delivery falling + price rising — distributing into rally"},
    "Lagging":   {"color": "#d50000", "label": "📤 Lagging",   "desc": "Delivery falling + price falling — institutional exit"},
}


# ── Smart Money Quadrant Chart ────────────────────────────────────────────────

def _quadrant_chart(df: pd.DataFrame) -> go.Figure:
    """Smart Money Quadrant: X = 1W cumulative price return, Y = Z-Score vs 100D norm."""
    plot_df = df.dropna(subset=["price_1w", "z_score"]).copy()
    if plot_df.empty:
        return go.Figure()

    x_vals = plot_df["price_1w"]
    y_vals = plot_df["z_score"]
    x_pad = max((x_vals.max() - x_vals.min()) * 0.25, 0.5)
    y_pad = max((y_vals.max() - y_vals.min()) * 0.25, 0.5)
    x0 = x_vals.min() - x_pad
    x1 = x_vals.max() + x_pad
    y0 = y_vals.min() - y_pad
    y1 = y_vals.max() + y_pad

    fig = go.Figure()

    fig.add_shape(type="rect", x0=x0, x1=0, y0=0,  y1=y1,
                  fillcolor="rgba(0,200,83,0.10)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=0, y1=y1,
                  fillcolor="rgba(30,144,255,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0,
                  fillcolor="rgba(255,80,0,0.07)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=y0, y1=0,
                  fillcolor="rgba(213,0,0,0.10)",   line_width=0, layer="below")

    corner_labels = [
        ("🔥 Secret Accum",    x0 + x_pad * 0.3, y1 - y_pad * 0.3, "left",  "top",    "rgba(0,200,83,0.18)"),
        ("✅ Confirmed Buy",     x1 - x_pad * 0.3, y1 - y_pad * 0.3, "right", "top",    "rgba(30,144,255,0.18)"),
        ("❌ Active Selling",    x0 + x_pad * 0.3, y0 + y_pad * 0.3, "left",  "bottom", "rgba(213,0,0,0.18)"),
        ("⚠️ Distribution Trap", x1 - x_pad * 0.3, y0 + y_pad * 0.3, "right", "bottom", "rgba(255,80,0,0.18)"),
    ]
    for label, lx, ly, xanchor, yanchor, bgcolor in corner_labels:
        fig.add_annotation(
            x=lx, y=ly, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=13, color="rgba(255,255,255,0.75)"),
            xanchor=xanchor, yanchor=yanchor,
            bgcolor=bgcolor,
            borderpad=5,
        )

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)

    if y1 > 1.0:
        fig.add_hline(
            y=1.0,
            line_dash="dash", line_width=1.0,
            line_color="rgba(0,200,83,0.45)",
            annotation_text="Surge threshold (Z=+1σ)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="rgba(0,200,83,0.7)"),
        )
    if y0 < -0.5:
        fig.add_hline(
            y=-0.5,
            line_dash="dash", line_width=1.0,
            line_color="rgba(255,80,0,0.45)",
            annotation_text="Weakness threshold (Z=-0.5σ)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="rgba(255,80,0,0.7)"),
        )

    signal_order = [
        "🔥 Secret Accumulation",
        "✅ Confirmed Accumulation",
        "👀 Early Accumulation",
        "📊 Volume Spike",
        "⚖️ Neutral",
        "📉 Weakening",
        "⚠️ Distribution Trap",
        "❌ Active Selling",
    ]
    for signal in signal_order:
        grp = plot_df[plot_df["signal"] == signal]
        if grp.empty:
            continue
        meta  = _SIGNAL_META.get(signal, {})
        color = meta.get("color", "#888888")
        sizes = (grp["accum_score"] / 100 * 22 + 12).clip(12, 34)

        fig.add_trace(go.Scatter(
            x=grp["price_1w"],
            y=grp["z_score"],
            mode="markers",
            name=signal,
            marker=dict(
                color=color,
                size=sizes,
                opacity=0.90,
                line=dict(width=1.5, color="rgba(255,255,255,0.5)"),
            ),
            customdata=grp[["sector", "action", "accum_score",
                             "dv_ratio", "z_score", "breadth", "horizon",
                             "dv_ratio_5d", "z_pct"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"<span style='color:{color}'>{signal}</span><br>"
                "─────────────────────<br>"
                "Score: <b>%{customdata[2]:.0f}</b>/100<br>"
                "Price 1W: <b>%{x:+.2f}%</b><br>"
                "Z-Rank: <b>%{customdata[8]:.0%}ile</b>  (%{y:+.1f}σ)<br>"
                "DV Today: %{customdata[3]:.2f}×  ·  5D Avg: %{customdata[7]:.2f}×<br>"
                "Breadth: %{customdata[5]:.0%}<br>"
                "Horizon: %{customdata[6]}<br>"
                "<i>%{customdata[1]}</i>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(
            title="← Price Falling  |  1-Week Price Return (%)  |  Price Rising →",
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[x0, x1], ticksuffix="%", tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="← Institutions Exiting  |  Z-Score (σ vs 100D norm)  |  Institutions Entering →",
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[y0, y1], ticksuffix="σ", tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h", y=-0.15, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        height=620,
        margin=dict(t=30, b=100, l=100, r=40),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=13, bordercolor="rgba(255,255,255,0.2)"),
        hovermode="closest",
    )
    return fig


# ── Rotation Clock Chart ──────────────────────────────────────────────────────

def _rotation_clock_chart(
    df: pd.DataFrame, period_name: str, nifty_return: float | None = None
) -> go.Figure:
    """RRG-style bubble chart: X = price return, Y = delivery slope z-score.

    nifty_return shifts the quadrant center from 0% to the market return so that
    phases (Leading / Improving / Weakening / Lagging) are market-relative —
    a sector only 'Leading' if it beat Nifty50, not just because it was positive.
    """
    if df.empty:
        return go.Figure()

    x_vals = df["cum_price_ret_pct"]
    y_vals = df["slope_z"]
    x_pad  = max((x_vals.max() - x_vals.min()) * 0.30, 1.0)
    y_pad  = max((y_vals.max() - y_vals.min()) * 0.30, 0.4)
    x0, x1 = x_vals.min() - x_pad, x_vals.max() + x_pad
    y0, y1 = y_vals.min() - y_pad, y_vals.max() + y_pad

    # Quadrant center: use Nifty50 return if available, else 0%
    cx = nifty_return if nifty_return is not None else 0.0

    fig = go.Figure()

    # Quadrant shading — centered on Nifty50 return (not 0%)
    # top-left=Improving (delivery rising, price below Nifty50)
    # top-right=Leading (delivery rising, price above Nifty50)
    # bottom-left=Lagging (delivery falling, price below Nifty50)
    # bottom-right=Weakening (delivery falling, price above Nifty50)
    fig.add_shape(type="rect", x0=x0, x1=cx,  y0=0,  y1=y1, fillcolor="rgba(64,196,255,0.09)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=cx,  x1=x1, y0=0,  y1=y1, fillcolor="rgba(0,200,83,0.09)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x0, x1=cx,  y0=y0, y1=0,  fillcolor="rgba(213,0,0,0.09)",   line_width=0, layer="below")
    fig.add_shape(type="rect", x0=cx,  x1=x1, y0=y0, y1=0,  fillcolor="rgba(255,109,0,0.07)", line_width=0, layer="below")

    corner_labels = [
        ("🔍 CONTRARIAN INFLOW",  x0 + x_pad * 0.35, y1 - y_pad * 0.35, "left",  "top",    "rgba(64,196,255,0.18)"),
        ("💰 MONEY ENTERING",     x1 - x_pad * 0.35, y1 - y_pad * 0.35, "right", "top",    "rgba(0,200,83,0.18)"),
        ("📤 MONEY EXITING",      x0 + x_pad * 0.35, y0 + y_pad * 0.35, "left",  "bottom", "rgba(213,0,0,0.18)"),
        ("⚠️ TOPPING / DIST",    x1 - x_pad * 0.35, y0 + y_pad * 0.35, "right", "bottom", "rgba(255,109,0,0.18)"),
    ]
    for label, lx, ly, xanchor, yanchor, bgcolor in corner_labels:
        fig.add_annotation(
            x=lx, y=ly, text=f"<b>{label}</b>",
            showarrow=False, font=dict(size=12, color="rgba(255,255,255,0.75)"),
            xanchor=xanchor, yanchor=yanchor, bgcolor=bgcolor, borderpad=5,
        )

    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)
    # Primary vertical axis: Nifty50 baseline (gold) if available, else 0% (white)
    if nifty_return is not None:
        fig.add_vline(x=nifty_return, line_color="#ffd600", line_width=2, line_dash="dash")
        fig.add_annotation(
            x=nifty_return, y=y1,
            text=f"<b>Nifty50: {nifty_return:+.2f}%</b>",
            showarrow=False,
            font=dict(size=11, color="#ffd600"),
            xanchor="center", yanchor="top",
            bgcolor="rgba(255,214,0,0.13)", borderpad=4,
            yshift=-4,
        )
        # Zero line as a faint reference
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.20)", line_width=1, line_dash="dot")
    else:
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)
    fig.add_hline(y= 0.25, line_dash="dot", line_width=1.0, line_color="rgba(0,200,83,0.30)")
    fig.add_hline(y=-0.25, line_dash="dot", line_width=1.0, line_color="rgba(213,0,0,0.30)")

    # Log-scale sizes: compress the 100x delivery-value range so Banking/IT
    # don't dwarf smaller sectors. sqrt gives a gentler compression than log.
    max_dv    = max(df["deliv_value_cr"].max(), 1.0)
    _sqrt_max = max_dv ** 0.5

    phase_order = ["Leading", "Improving", "Neutral", "Weakening", "Lagging"]
    for phase in phase_order:
        grp = df[df["phase"] == phase]
        if grp.empty:
            continue
        meta  = _PHASE_META[phase]
        color = meta["color"]
        sizes = ((grp["deliv_value_cr"].clip(lower=0) ** 0.5) / _sqrt_max * 28 + 10).clip(10, 38)

        chg_str = grp["deliv_chg_pct"].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A").values
        dv_str  = grp["deliv_value_cr"].apply(lambda v: f"₹{v:,.0f} Cr").values
        to_str  = grp["turnover_cr"].apply(lambda v: f"₹{v:,.0f} Cr").values

        vs_nifty  = (grp["cum_price_ret_pct"] - cx).round(2).values
        corr_vals = grp["price_deliv_corr"].round(2).values if "price_deliv_corr" in grp.columns else [0.0] * len(grp)
        conf_vals = grp["signal_confidence"].round(2).values if "signal_confidence" in grp.columns else [0.5] * len(grp)
        customdata = list(zip(
            grp["sector"].values,
            grp["flow_signal"].values,
            dv_str,
            grp["slope_z"].round(2).values,
            chg_str,
            to_str,
            grp["avg_deliv_pct"].round(1).values,
            vs_nifty,        # [7] — excess return vs Nifty50
            corr_vals,       # [8] — price-delivery correlation
            conf_vals,       # [9] — signal confidence
        ))

        fig.add_trace(go.Scatter(
            x=grp["cum_price_ret_pct"],
            y=grp["slope_z"],
            mode="markers",
            name=meta["label"],
            marker=dict(
                color=color, size=sizes, opacity=0.85,
                line=dict(width=1.5, color="rgba(255,255,255,0.35)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"<span style='color:{color}'>%{{customdata[1]}}</span><br>"
                "─────────────────────────<br>"
                "Price Return: <b>%{x:+.2f}%</b>  (vs Nifty50: <b>%{customdata[7]:+.2f}%</b>)<br>"
                "Delivery Slope Z: <b>%{y:+.2f}σ</b><br>"
                "Price-Del Correlation: <b>%{customdata[8]:.2f}</b>  "
                "<span style='color:rgba(200,200,200,0.6)'>(−=distribution, +=co-moving)</span><br>"
                "Signal Confidence: <b>%{customdata[9]:.2f}</b><br>"
                "Delivery Value: <b>%{customdata[2]}</b>  ·  Del Chg: <b>%{customdata[4]}</b><br>"
                "Avg Delivery %: %{customdata[6]:.1f}%  ·  Turnover: %{customdata[5]}"
                "<extra></extra>"
            ),
        ))

    # Flag extreme outliers — sectors returning >3× Nifty50 or >40% in a single period
    outlier_threshold = max(abs(cx) * 3, 40.0)
    outliers = df[df["cum_price_ret_pct"] > outlier_threshold]
    for _, row in outliers.iterrows():
        fig.add_annotation(
            x=row["cum_price_ret_pct"], y=row["slope_z"],
            text=f"⚠️ {row['sector']}<br>{row['cum_price_ret_pct']:+.1f}%",
            showarrow=True, arrowhead=2, arrowcolor="#ff9100",
            font=dict(size=10, color="#ff9100"),
            bgcolor="rgba(255,145,0,0.15)", borderpad=3,
            ax=-50, ay=-30,
        )

    nifty_subtitle = f" · Nifty50: {nifty_return:+.2f}%" if nifty_return is not None else ""
    fig.update_layout(
        title=dict(
            text=f"Sector Rotation Clock — {period_name}{nifty_subtitle}",
            font=dict(size=16), x=0.5,
        ),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(
            title="← Below Market  |  Cumulative Price Return (%)  |  Above Market →"
                  if nifty_return is not None else
                  "← Price Falling  |  Cumulative Price Return (%)  |  Price Rising →",
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[x0, x1], ticksuffix="%", tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="← Delivery Momentum Falling  |  Slope Z-Score  |  Delivery Momentum Rising →",
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[y0, y1], tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h", y=-0.15, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        ),
        height=620,
        margin=dict(t=60, b=90, l=90, r=40),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=13, bordercolor="rgba(255,255,255,0.2)"),
    )
    return fig


# ── Trend Chart ───────────────────────────────────────────────────────────────

def _trend_chart(hist: pd.DataFrame, sector: str, signal: str) -> go.Figure:
    """100-day delivery % and delivery value trend for a single sector."""
    if hist.empty:
        return go.Figure()

    color = _SIGNAL_META.get(signal, {}).get("color", "#4c78a8")
    price_colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR
                    for v in hist["avg_price_chg"].fillna(0)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["trade_date"], y=hist["wtd_deliv_per"],
        name="Wtd Delivery %", mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        hovertemplate="<b>%{x|%d %b}</b><br>Wtd Delivery %: %{y:.1f}%<extra></extra>",
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=hist["trade_date"], y=hist["deliv_value_cr"],
        name="Deliv Value (₹ Cr)", mode="lines",
        line=dict(color="#f0b429", width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%d %b}</b><br>Delivery ₹: %{y:.1f} Cr<extra></extra>",
        yaxis="y2",
    ))
    fig.add_trace(go.Bar(
        x=hist["trade_date"], y=hist["avg_price_chg"],
        name="Daily Price Chg %",
        marker_color=price_colors,
        opacity=0.55,
        hovertemplate="<b>%{x|%d %b}</b><br>Price Chg: %{y:+.2f}%<extra></extra>",
        yaxis="y3",
    ))

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(showgrid=False, tickformat="%d %b"),
        yaxis=dict(title="Wtd Delivery %", showgrid=True,
                   gridcolor=GRID_COLOR, side="left", ticksuffix="%"),
        yaxis2=dict(title="Deliv Value (₹ Cr)", overlaying="y", side="left",
                    showgrid=False, anchor="free", position=0.0,
                    tickprefix="₹"),
        yaxis3=dict(title="Daily Price Chg %", overlaying="y", side="right",
                    showgrid=False, zeroline=True,
                    zerolinecolor="rgba(255,255,255,0.25)", ticksuffix="%"),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        height=340, margin=dict(t=30, b=40, l=80, r=60),
        hovermode="x unified",
    )
    return fig


# ── Sector Cards ──────────────────────────────────────────────────────────────

_AVOID_SIGNAL_SET = {"⚠️ Distribution Trap", "❌ Active Selling", "📉 Weakening"}


# Per-stock drill-down list shows only stocks whose 7D turnover-weighted
# delivery % exceeds this floor (user-defined cut for "real delivery").
_MIN_STOCK_WTD_DELIV_PCT = 48.0


def _sector_card(row: pd.Series, selected_date: date, min_turnover: float,
                 deliv_threshold: float = _MIN_STOCK_WTD_DELIV_PCT) -> None:
    meta       = _SIGNAL_META.get(row["signal"], {})
    color      = meta.get("color", "#888")
    score      = row["accum_score"]
    is_avoid   = row["signal"] in _AVOID_SIGNAL_SET
    invest_signal = meta.get("invest", False)

    bar_html = (
        f"<div style='background:rgba(255,255,255,0.1);border-radius:4px;height:6px;margin:4px 0 6px 0'>"
        f"<div style='width:{score}%;background:{color};height:6px;border-radius:4px'></div></div>"
    )

    dv   = row.get("dv_ratio")
    dv5d = row.get("dv_ratio_5d")
    z    = row.get("z_score")
    br   = row.get("breadth")
    p1w  = row.get("price_1w")
    dv1w = row.get("deliv_val_1w_cr")
    action_text = str(row.get("action", ""))
    coverage    = str(row.get("coverage") or "—")
    horizon     = str(row.get("horizon")  or "—")

    def _fmt(v, fmt):
        return fmt.format(v) if (v is not None and not (isinstance(v, float) and pd.isna(v))) else "—"

    dv_str   = _fmt(dv,   "{:.2f}×")
    dv5d_str = _fmt(dv5d, "{:.2f}×")
    # Show the cross-sectional RANK of z (what the gates actually use), with raw
    # σ in parentheses for reference. Raw σ alone is misleading — delivery value
    # trends up so a "+12σ" is routine here, not a once-in-history event. The
    # rank says plainly "Nth percentile among today's sectors".
    z_pct = row.get("z_pct")
    if z is not None and not (isinstance(z, float) and pd.isna(z)):
        if z_pct is not None and not (isinstance(z_pct, float) and pd.isna(z_pct)):
            z_str = f"{z_pct*100:.0f}%ile ({z:+.1f}σ)"
        else:
            z_str = f"{z:+.2f}σ"
    else:
        z_str = "—"
    p1w_str  = _fmt(p1w,  "{:+.2f}%")
    br_str   = f"{br * 100:.0f}%" if (br is not None and not (isinstance(br, float) and pd.isna(br))) else "—"
    dv1w_str = f"₹{dv1w:,.0f} Cr" if (dv1w is not None and not (isinstance(dv1w, float) and pd.isna(dv1w))) else "—"

    # 5D avg color: green if sustained (>=1.15), orange if weak (<=0.9), grey otherwise
    dv5d_color = (
        POSITIVE_COLOR if (dv5d is not None and not pd.isna(dv5d) and dv5d >= 1.15)
        else (NEGATIVE_COLOR if (dv5d is not None and not pd.isna(dv5d) and dv5d <= 0.90)
        else "#888888")
    )
    # Color by z-PERCENTILE (the gate input), not raw z: top-half green,
    # bottom-quartile red, middle grey. Falls back to raw z if rank absent.
    if z_pct is not None and not (isinstance(z_pct, float) and pd.isna(z_pct)):
        z_color = (POSITIVE_COLOR if z_pct >= 0.50
                   else (NEGATIVE_COLOR if z_pct <= 0.25 else "#888888"))
    else:
        z_color = (POSITIVE_COLOR if (z is not None and not pd.isna(z) and z >= 1.0)
                   else (NEGATIVE_COLOR if (z is not None and not pd.isna(z) and z <= -0.5)
                   else "#888888"))
    p1w_color = POSITIVE_COLOR if (p1w is not None and not pd.isna(p1w) and p1w > 0) else NEGATIVE_COLOR

    if is_avoid:
        bottom_row = (
            f"<b style='color:rgba(255,100,80,0.85)'>Avoid for:</b> "
            f"<b style='color:rgba(255,255,255,0.75)'>{coverage}</b>"
            f" &nbsp;|&nbsp; Delivery Value 1W: {dv1w_str}"
        )
    else:
        bottom_row = (
            f"<b style='color:rgba(255,255,255,0.45)'>Coverage:</b> "
            f"<b style='color:rgba(255,255,255,0.75)'>{coverage}</b>"
            + (f" &nbsp;|&nbsp; Horizon: {horizon}" if horizon not in ("—", "") else "")
            + f" &nbsp;|&nbsp; Delivery Value 1W: {dv1w_str}"
        )

    if "—" in action_text:
        action_prefix, action_desc = action_text.split("—", 1)
        action_html = (
            f"<span style='color:{color};font-weight:700'>{action_prefix.strip()}</span>"
            f"<span style='color:rgba(255,255,255,0.55)'> — {action_desc.strip()}</span>"
        )
    else:
        action_html = f"<span style='color:{color}'>{action_text}</span>"

    st.markdown(
        f"<div style='border-left:3px solid {color};padding:8px 12px;margin:4px 0;"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<b style='font-size:14px'>{row['sector']}</b>"
        f"<span style='font-size:11px;color:{color};font-weight:600'>{score:.0f}/100</span></div>"
        f"{bar_html}"
        f"<div style='font-size:11px;margin-bottom:4px'>{row['signal']} &nbsp; {action_html}</div>"
        f"<div style='display:flex;gap:16px;margin-top:4px;font-size:12px'>"
        f"<span>DV Today: <b>{dv_str}</b></span>"
        f"<span>5D Avg: <b style='color:{dv5d_color}'>{dv5d_str}</b></span>"
        f"<span>Z-Rank: <b style='color:{z_color}'>{z_str}</b></span>"
        f"<span>Breadth: <b>{br_str}</b></span>"
        f"<span>1W Price: <b style='color:{p1w_color}'>{p1w_str}</b></span>"
        f"</div>"
        f"<div style='margin-top:4px;font-size:11px;color:rgba(255,255,255,0.5)'>"
        f"{bottom_row}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"📋  View stocks in {row['sector']}", expanded=False):
        stocks = cached_sector_stocks_rotation(row["sector"], selected_date, min_turnover)
        if stocks.empty:
            st.caption("No stock data for this period.")
        else:
            valid_deliv = stocks["wtd_deliv_per"].dropna()
            hi_thresh = float(valid_deliv.quantile(0.67)) if len(valid_deliv) >= 3 else float(valid_deliv.max())
            lo_thresh = float(valid_deliv.quantile(0.33)) if len(valid_deliv) >= 3 else float(valid_deliv.min())

            has_own_history = "avg_deliv_per_100d" in stocks.columns

            def _stock_signal(r: pd.Series) -> str:
                p   = float(r["price_chg_pct"])      if pd.notna(r["price_chg_pct"])      else 0.0
                d   = float(r["wtd_deliv_per"])       if pd.notna(r["wtd_deliv_per"])       else 0.0
                avg = (float(r["avg_deliv_per_100d"]) if has_own_history
                       and pd.notna(r.get("avg_deliv_per_100d")) else None)

                if avg is not None:
                    if invest_signal:
                        if d > avg and p < 0:
                            return "🔥 Strong"
                        elif d > avg:
                            return "✅ Buying"
                        elif d > avg * 0.75:
                            return "👀 Watch"
                        else:
                            return "⚪ Weak"
                    else:
                        if d < avg and p > 0:
                            return "❌ Exit Now"
                        elif d < avg:
                            return "⚠️ Reducing"
                        elif p > 0:
                            return "📉 Fading"
                        else:
                            return "⚪ Neutral"
                else:
                    if invest_signal:
                        if d >= hi_thresh and p < 0:
                            return "🔥 Strong"
                        elif d >= hi_thresh:
                            return "✅ Buying"
                        elif d >= lo_thresh:
                            return "👀 Watch"
                        else:
                            return "⚪ Weak"
                    else:
                        if d <= lo_thresh and p > 0:
                            return "❌ Exit Now"
                        elif d <= lo_thresh:
                            return "⚠️ Reducing"
                        elif p > 0:
                            return "📉 Fading"
                        else:
                            return "⚪ Neutral"

            stocks = stocks.copy()
            stocks["conviction"] = stocks.apply(_stock_signal, axis=1)

            # ── F&O overlay: merge per-symbol futures/options positioning ──────
            # Cached (@st.cache_data), so calling per-card is a cache hit. Non-F&O
            # stocks get NaN → rendered blank, exactly as intended. fut_signal is
            # the OI-price read (or "OI settling (post-expiry)"); opt_signal is the
            # PCR read. These are SHORT-TERM (daily OI) reads — complementary to the
            # swing delivery conviction, not a replacement.
            _fno = cached_fno_positioning_by_symbol(selected_date)
            if not _fno.empty:
                stocks = stocks.merge(
                    _fno[["symbol", "fut_signal", "opt_signal"]],
                    on="symbol", how="left",
                )
            else:
                stocks["fut_signal"] = None
                stocks["opt_signal"] = None

            if invest_signal:
                _rank = {"🔥 Strong": 0, "✅ Buying": 1, "👀 Watch": 2, "⚪ Weak": 3}
            else:
                _rank = {"❌ Exit Now": 0, "⚠️ Reducing": 1, "📉 Fading": 2, "⚪ Neutral": 3}

            stocks["_rank"] = stocks["conviction"].map(_rank).fillna(9)
            stocks = stocks.sort_values(
                ["_rank", "wtd_deliv_per", "deliv_value_cr"],
                ascending=[True, False, False],
            ).drop(columns="_rank")

            # Context strip: stock count + top-3 delivery contributors
            n_stocks = len(stocks)
            total_dv = stocks["deliv_value_cr"].sum()
            total_turnover = stocks["turnover_cr"].sum()
            top3 = stocks.nlargest(3, "deliv_value_cr")[["symbol", "deliv_value_cr"]]
            top3_parts = " + ".join(
                f"{r['symbol']} {r['deliv_value_cr'] / total_dv * 100:.0f}%"
                for _, r in top3.iterrows()
            ) if total_dv > 0 else ""
            top3_total_pct = top3["deliv_value_cr"].sum() / total_dv * 100 if total_dv > 0 else 0

            n_industries = stocks["industry"].nunique() if "industry" in stocks.columns else 0
            industries_list = stocks["industry"].dropna().unique().tolist() if "industry" in stocks.columns else []

            st.markdown(
                f"<div style='font-size:11px;color:#888;margin-bottom:4px'>"
                f"{n_stocks} stocks &nbsp;·&nbsp; "
                f"Top-3 delivery: {top3_parts} = {top3_total_pct:.0f}% of sector"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Mixed-sector warning: ≥4 distinct sub-industries is a quality flag
            if n_industries >= 4:
                industry_summary = ", ".join(sorted(industries_list)[:6])
                if len(industries_list) > 6:
                    industry_summary += f" +{len(industries_list) - 6} more"
                st.markdown(
                    f"<div style='background:rgba(100,100,255,0.10);border-left:3px solid #7986cb;"
                    f"padding:6px 10px;border-radius:0 4px 4px 0;margin-bottom:4px;font-size:12px'>"
                    f"🔀 <b>Mixed sector</b> — {n_industries} sub-industries: {industry_summary}. "
                    f"Sector-level signals may blend unrelated themes."
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Delivery-value dominance warning (institutional metric, not turnover)
            dominant = stocks[stocks["deliv_value_cr"] / total_dv > 0.35] if total_dv > 0 else stocks.iloc[:0]
            if not dominant.empty:
                dom = dominant.iloc[0]
                dom_dv_pct = dom["deliv_value_cr"] / total_dv * 100
                dom_to_pct = dom["turnover_cr"] / total_turnover * 100 if total_turnover > 0 else 0
                dom_conv = stocks.loc[stocks["symbol"] == dom["symbol"], "conviction"].values[0]
                warn_color = "#ff9100" if invest_signal else "#d50000"
                st.markdown(
                    f"<div style='background:rgba(255,145,0,0.12);border-left:3px solid {warn_color};"
                    f"padding:6px 10px;border-radius:0 4px 4px 0;margin-bottom:6px;font-size:12px'>"
                    f"⚠️ <b>{dom['symbol']}</b> drives <b>{dom_dv_pct:.0f}%</b> of sector delivery value "
                    f"(₹{dom['deliv_value_cr']:.0f} Cr · {dom['wtd_deliv_per']:.1f}% del · {dom_to_pct:.0f}% of turnover) "
                    f"with <b>{dom_conv}</b> conviction. "
                    f"Sector signal is driven by this one stock — verify independently."
                    f"</div>",
                    unsafe_allow_html=True,
                )

            display_cols = ["symbol", "company_name", "industry", "ltp", "conviction",
                            "fut_signal", "opt_signal",
                            "wtd_deliv_per", "avg_deliv_per_100d",
                            "deliv_value_cr", "turnover_cr", "price_chg_pct"]
            display_cols = [c for c in display_cols if c in stocks.columns]

            # Show only stocks with 7D turnover-weighted delivery % above the floor.
            # Conviction + dominance/top-3 context above stay on the full sector set.
            shown = stocks[stocks["wtd_deliv_per"] > deliv_threshold]
            n_hidden = len(stocks) - len(shown)
            if n_hidden:
                st.caption(
                    f"Showing {len(shown)} of {len(stocks)} stocks — "
                    f"Wtd Deliv % > {deliv_threshold:.0f}% "
                    f"({n_hidden} below threshold hidden)."
                )

            st.dataframe(
                shown[display_cols],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "symbol":        st.column_config.TextColumn("Symbol", width="small",
                        help="NSE ticker symbol"),
                    "company_name":  st.column_config.TextColumn("Company",
                        help="Company full name from NSE sector master"),
                    "industry":      st.column_config.TextColumn("Sub-Sector",
                        help="Industry classification within the sector"),
                    "ltp":           st.column_config.NumberColumn(
                        "LTP (₹)", format="₹%.2f",
                        help="Last Traded Price\n"
                             "Formula: most recent close_price in the 7-day window\n"
                             "= ARGMAX(close_price, trade_date)"),
                    "conviction":    st.column_config.TextColumn("Conviction",
                        help="Own-history conviction — compares 7D Wtd Deliv % against each stock's own 100D baseline\n\n"
                             "🔥 Strong  = today's delivery ABOVE own 100D avg AND price falling  (institutions buying the dip)\n"
                             "✅ Buying  = today's delivery ABOVE own 100D avg AND price rising  (momentum confirmed)\n"
                             "👀 Watch   = today's delivery ≥ 75% of own 100D avg  (approaching normal)\n"
                             "⚪ Weak    = today's delivery BELOW 75% of own 100D avg  (sub-normal for THIS stock)\n\n"
                             "For AVOID sectors:\n"
                             "❌ Exit Now  = delivery BELOW own avg AND price rising  (institutions selling into retail rally)\n"
                             "⚠️ Reducing  = delivery BELOW own 100D avg\n"
                             "📉 Fading    = price rising but delivery not below avg\n"
                             "⚪ Neutral   = no clear signal\n\n"
                             "Falls back to sector-relative percentile for stocks with no 100D history"),
                    "fut_signal":    st.column_config.TextColumn(
                        "Futures",
                        help="Stock-FUTURES positioning (near-month OI vs price, daily):\n"
                             "🟢 Long Buildup = OI↑ + price↑ (fresh longs)\n"
                             "🔴 Short Buildup = OI↑ + price↓ (fresh shorts)\n"
                             "Short Covering = OI↓ + price↑ | Long Unwinding = OI↓ + price↓\n"
                             "Blank = not an F&O stock. 'OI settling (post-expiry)' = "
                             "2-3 days after monthly expiry, OI still rolling — not reliable.\n"
                             "SHORT-TERM read — confirms/diverges from the swing delivery signal."),
                    "opt_signal":    st.column_config.TextColumn(
                        "Options",
                        help="Stock-OPTIONS positioning (near-month PCR, contrarian):\n"
                             "Put Heavy (PCR>1.3) = downside hedged → contrarian bullish\n"
                             "Call Heavy (PCR<0.6) = complacent calls → contrarian bearish\n"
                             "Neutral = balanced. Blank = not an F&O stock."),
                    "avg_deliv_per_100d": st.column_config.NumberColumn(
                        "100D Avg Del%", format="%.1f%%",
                        help="Stock's own 100-trading-day average delivery %\n\n"
                             "This is the baseline for own-history conviction.\n"
                             "Compare against Wtd Deliv % (7D) to see if today is above or below normal for THIS stock.\n"
                             "A stock at 15% delivery is 'Strong' if its own 100D avg is 8%, even if peers are 40%."),
                    "wtd_deliv_per": st.column_config.NumberColumn(
                        "Wtd Deliv %", format="%.1f%%",
                        help="Turnover-Weighted Delivery %  (last 7 trading days)\n\n"
                             "Formula: Σ(deliv_per × turnover_lacs) / Σ(turnover_lacs)\n\n"
                             "Why weighted: a ₹500 Cr stock at 60% delivery counts more\n"
                             "than a ₹5 Cr stock at 80% delivery.\n"
                             "High % = institutions are taking delivery (holding, not squaring off)"),
                    "deliv_value_cr":st.column_config.NumberColumn(
                        "Deliv Value (₹ Cr)", format="₹%.1f",
                        help="Delivery Value in ₹ Crores  (last 7 trading days)\n\n"
                             "Formula: Σ(deliv_per / 100 × turnover_lacs) / 100\n\n"
                             "= actual ₹ worth of shares taken home (not squared off intraday)\n"
                             "This is the absolute measure of institutional conviction —\n"
                             "retail traders square off intraday, institutions take delivery"),
                    "turnover_cr":   st.column_config.NumberColumn(
                        "Turnover (₹ Cr)", format="₹%.1f",
                        help="Total Traded Value in ₹ Crores  (last 7 trading days)\n\n"
                             "Formula: Σ(turnover_lacs) / 100\n\n"
                             "= total buy + sell value traded\n"
                             "High turnover with low delivery % = speculative / intraday activity\n"
                             "High turnover with high delivery % = institutional accumulation"),
                    "price_chg_pct": st.column_config.NumberColumn(
                        "Price Chg %", format="%+.2f%%",
                        help="Average Daily Price Change %  (last 7 trading days)\n\n"
                             "Formula: AVG((close_price − prev_close) / prev_close × 100)\n\n"
                             "Simple average across all trading days in the window\n"
                             "+ = price rising on average   − = price falling on average"),
                },
            )


# ── Phase Card (Rotation Clock) ───────────────────────────────────────────────

def _phase_card(row: pd.Series, color: str) -> None:
    sector  = row["sector"]
    price   = row["cum_price_ret_pct"]
    dv_cr   = row["deliv_value_cr"]
    dv_chg  = row.get("deliv_chg_pct")
    slope_z = row["slope_z"]
    avg_del = row["avg_deliv_pct"]

    price_c = POSITIVE_COLOR if price > 0 else NEGATIVE_COLOR
    chg_str = f"{dv_chg:+.1f}%" if pd.notna(dv_chg) else "—"
    chg_c   = POSITIVE_COLOR if (pd.notna(dv_chg) and dv_chg > 0) else NEGATIVE_COLOR

    st.markdown(
        f"<div style='border-left:3px solid {color};padding:6px 10px;margin:3px 0;"
        f"background:rgba(255,255,255,0.025);border-radius:0 5px 5px 0'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<b style='font-size:13px'>{sector}</b>"
        f"<span style='font-size:12px;font-weight:600;color:{price_c}'>{price:+.2f}%</span></div>"
        f"<div style='display:flex;gap:14px;margin-top:3px;font-size:11px;color:rgba(255,255,255,0.55)'>"
        f"<span>Del Chg: <b style='color:{chg_c}'>{chg_str}</b></span>"
        f"<span>Slope Z: <b style='color:{color}'>{slope_z:+.2f}σ</b></span>"
        f"<span>DV: <b>₹{dv_cr:,.0f} Cr</b></span>"
        f"<span>Avg Del%: {avg_del:.1f}%</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ── Cross-Period Comparison ───────────────────────────────────────────────────

def _render_cross_period(selected_date: date, min_turnover: float) -> None:
    _WINDOWS = [(5, "1W"), (10, "2W"), (22, "1M"), (65, "3M")]
    _PHASE_ICON = {
        "Leading":   "💰 Lead",
        "Improving": "🔍 Impr",
        "Neutral":   "⚖️ Neut",
        "Weakening": "⚠️ Weak",
        "Lagging":   "📤 Lag",
    }
    _PHASE_COLOR = {
        "Leading":   "#00c853",
        "Improving": "#40c4ff",
        "Neutral":   "#888888",
        "Weakening": "#ff9100",
        "Lagging":   "#d50000",
    }

    period_data: dict[str, pd.DataFrame] = {}
    all_sectors: list[str] = []

    with st.spinner("Loading all 4 time periods…"):
        for w, label in _WINDOWS:
            d = cached_sector_rotation_timeframe(selected_date, w, float(min_turnover))
            if not d.empty:
                period_data[label] = d.set_index("sector")
                if not all_sectors:
                    all_sectors = d["sector"].tolist()

    if not all_sectors:
        st.warning("Insufficient data for cross-period comparison.")
        return

    # Build HTML matrix table
    header_cells = "".join(
        f"<th style='padding:6px 14px;text-align:center;font-size:12px;"
        f"color:rgba(255,255,255,0.5);font-weight:600;letter-spacing:0.5px'>{lbl}</th>"
        for _, lbl in _WINDOWS
    )
    header = (
        f"<tr style='border-bottom:2px solid rgba(255,255,255,0.12)'>"
        f"<th style='padding:6px 10px;text-align:left;font-size:12px;"
        f"color:rgba(255,255,255,0.5);font-weight:600'>SECTOR</th>"
        f"{header_cells}</tr>"
    )

    rows_html = ""
    for sector in all_sectors:
        cells = ""
        for _, lbl in _WINDOWS:
            if lbl in period_data and sector in period_data[lbl].index:
                phase = period_data[lbl].loc[sector, "phase"]
                icon  = _PHASE_ICON.get(phase, "—")
                c     = _PHASE_COLOR.get(phase, "#888")
                pr    = period_data[lbl].loc[sector, "cum_price_ret_pct"]
                pr_c  = POSITIVE_COLOR if pr > 0 else NEGATIVE_COLOR
                cells += (
                    f"<td style='padding:6px 14px;text-align:center'>"
                    f"<div style='font-size:12px;color:{c};font-weight:600'>{icon}</div>"
                    f"<div style='font-size:10px;color:{pr_c}'>{pr:+.1f}%</div>"
                    f"</td>"
                )
            else:
                cells += "<td style='padding:6px 14px;text-align:center;color:#555'>—</td>"

        rows_html += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>"
            f"<td style='padding:6px 10px;font-size:13px;font-weight:500'>{sector}</td>"
            f"{cells}</tr>"
        )

    st.markdown(
        f"<div style='overflow-x:auto'>"
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead>{header}</thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )
    st.caption("Phase icons: 💰 Leading = money entering · 🔍 Improving = contrarian inflow · "
               "⚖️ Neutral = sideways · ⚠️ Weakening = topping · 📤 Lagging = money exiting. "
               "% = cumulative price return for that period.")


# ── Rotation Clock Tab ────────────────────────────────────────────────────────

def _render_custom_range(all_dates: list, min_turnover: float) -> None:
    """Custom date range: sector rotation + per-sector stock drill-down."""
    if not all_dates:
        st.warning("No trading dates available.")
        return

    min_avail = all_dates[-1]   # oldest
    max_avail = all_dates[0]    # most recent

    # Pre-sort once (all_dates is newest-first; we need both orders)
    avail_asc = sorted(all_dates)   # ascending  — for from_snap search
    avail_set = set(all_dates)

    col_from, col_to = st.columns(2)
    # Default from_date = ~1 month before most recent (22 trading days back)
    default_from = all_dates[min(21, len(all_dates) - 1)] if len(all_dates) > 1 else min_avail

    with col_from:
        from_date = st.date_input(
            "From Date",
            value=default_from,
            min_value=min_avail,
            max_value=max_avail,
            key="cr_from_date",
            help="Start of the analysis period",
        )
    with col_to:
        to_date = st.date_input(
            "To Date",
            value=max_avail,
            min_value=min_avail,
            max_value=max_avail,
            key="cr_to_date",
            help="End of the analysis period",
        )

    if from_date >= to_date:
        st.warning("From Date must be before To Date.")
        return

    # Snap to nearest available trading days
    from_snap = next((d for d in avail_asc if d >= from_date), None)
    to_snap   = next((d for d in reversed(avail_asc) if d <= to_date), None)

    if from_snap is None or to_snap is None or from_snap >= to_snap:
        st.warning("No trading data found in the selected range.")
        return

    n_calendar = (to_snap - from_snap).days
    n_trading  = sum(1 for d in avail_set if from_snap <= d <= to_snap)

    st.caption(
        f"**{from_snap.strftime('%d %b %Y')}** → **{to_snap.strftime('%d %b %Y')}**  "
        f"({n_calendar} calendar days · {n_trading} trading days)"
    )

    with st.spinner("Computing sector rotation for custom range…"):
        df = cached_sector_rotation_custom_range(from_snap, to_snap, float(min_turnover))

    if df.empty:
        st.warning("No data found for this date range. The range may be too short or pre-date available history.")
        return

    # KPI pills — Nifty50 first for benchmark context
    nifty_ret = df["nifty_return"].iloc[0] if "nifty_return" in df.columns else None
    pc = df["phase"].value_counts()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "🔵 Nifty50",
        f"{nifty_ret:+.2f}%" if nifty_ret is not None else "N/A",
        help="Nifty50 return for this period. Quadrant center shifts to this value "
             "so phases reflect market-relative performance.",
    )
    k2.metric("💰 Leading",   pc.get("Leading",   0))
    k3.metric("🔍 Improving", pc.get("Improving", 0))
    k4.metric("⚖️ Neutral",   pc.get("Neutral",   0))
    k5.metric("⚠️ Weakening", pc.get("Weakening", 0))
    k6.metric("📤 Lagging",   pc.get("Lagging",   0))

    # Bubble chart
    period_label = f"{from_snap.strftime('%d %b')} → {to_snap.strftime('%d %b %Y')}"
    st.plotly_chart(
        _rotation_clock_chart(df, period_label, nifty_return=nifty_ret),
        use_container_width=True,
        key="cr_clock_chart",
    )
    _render_clock_legend(df)

    # ── Phase cards with stock drill-down ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Sector & Stock Breakdown")
    st.caption(
        "Expand any sector to see how individual stocks performed during this period. "
        "Sorted by delivery value (highest institutional activity first)."
    )

    # Per-stock delivery filter for the drill-down tables below.
    cr_deliv_threshold = st.slider(
        "Min stock Avg Delivery % — filters the per-stock lists below",
        min_value=0, max_value=100, value=int(_MIN_STOCK_WTD_DELIV_PCT), step=1,
        key="cr_stock_deliv_threshold",
        help="Hide stocks whose period turnover-weighted delivery % is at or below this value. "
             "The Top Performers / Laggards summary still uses the full stock set.",
    )

    phase_order = ["Leading", "Improving", "Neutral", "Weakening", "Lagging"]
    for phase in phase_order:
        grp = df[df["phase"] == phase].reset_index(drop=True)
        if grp.empty:
            continue
        meta  = _PHASE_META[phase]
        color = meta["color"]

        st.markdown(
            f"<div style='font-size:13px;font-weight:700;color:{color};"
            f"margin:14px 0 4px 0;letter-spacing:0.3px'>"
            f"{meta['label']} — {meta['desc']} ({len(grp)})</div>",
            unsafe_allow_html=True,
        )

        for _, row in grp.iterrows():
            _phase_card(row, color)

            with st.expander(f"📋 Stocks in {row['sector']} — {from_snap.strftime('%d %b')} to {to_snap.strftime('%d %b %Y')}", expanded=False):
                with st.spinner(f"Loading {row['sector']} stocks…"):
                    stocks = cached_sector_stocks_custom_range(
                        row["sector"], from_snap, to_snap, float(min_turnover)
                    )

                if stocks.empty:
                    st.caption("No stock data for this period.")
                    continue

                # Top/bottom performers summary
                valid = stocks.dropna(subset=["period_ret_pct"])
                if not valid.empty:
                    top3    = valid.nlargest(3, "period_ret_pct")
                    bottom3 = valid.nsmallest(3, "period_ret_pct")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(
                            "<div style='font-size:11px;color:#00c853;font-weight:600;margin-bottom:3px'>🏆 Top Performers</div>",
                            unsafe_allow_html=True,
                        )
                        for _, s in top3.iterrows():
                            ret = s["period_ret_pct"]
                            st.markdown(
                                f"<span style='font-size:12px'><b>{s['symbol']}</b> "
                                f"<span style='color:#00c853'>{ret:+.1f}%</span> "
                                f"<span style='color:rgba(255,255,255,0.45);font-size:10px'>"
                                f"₹{s['price_start']:.0f}→₹{s['price_end']:.0f}</span></span>",
                                unsafe_allow_html=True,
                            )
                    with c2:
                        st.markdown(
                            "<div style='font-size:11px;color:#d50000;font-weight:600;margin-bottom:3px'>📉 Laggards</div>",
                            unsafe_allow_html=True,
                        )
                        for _, s in bottom3.iterrows():
                            ret = s["period_ret_pct"]
                            st.markdown(
                                f"<span style='font-size:12px'><b>{s['symbol']}</b> "
                                f"<span style='color:#d50000'>{ret:+.1f}%</span> "
                                f"<span style='color:rgba(255,255,255,0.45);font-size:10px'>"
                                f"₹{s['price_start']:.0f}→₹{s['price_end']:.0f}</span></span>",
                                unsafe_allow_html=True,
                            )

                # Filter the table by period delivery % (summary above uses full set).
                cr_shown = stocks[stocks["wtd_deliv_per"] > cr_deliv_threshold] \
                    if "wtd_deliv_per" in stocks.columns else stocks
                cr_hidden = len(stocks) - len(cr_shown)
                if cr_hidden:
                    st.caption(
                        f"Showing {len(cr_shown)} of {len(stocks)} stocks — "
                        f"Avg Deliv % > {cr_deliv_threshold:.0f}% "
                        f"({cr_hidden} below threshold hidden)."
                    )

                st.dataframe(
                    cr_shown,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "symbol":       st.column_config.TextColumn("Symbol", width="small"),
                        "company_name": st.column_config.TextColumn("Company"),
                        "industry":     st.column_config.TextColumn("Sub-Sector"),
                        "price_start":  st.column_config.NumberColumn(
                            f"Price on {from_snap.strftime('%d %b')}", format="₹%.2f",
                            help="Price at the start of the period (prev_close of first trading day)"),
                        "price_end":    st.column_config.NumberColumn(
                            f"Price on {to_snap.strftime('%d %b')}", format="₹%.2f",
                            help="Price at the end of the period (close of last trading day)"),
                        "period_ret_pct": st.column_config.NumberColumn(
                            "Period Return %", format="%+.2f%%",
                            help=f"(Price End − Price Start) / Price Start × 100\n"
                                 f"Period: {from_snap.strftime('%d %b')} → {to_snap.strftime('%d %b %Y')}"),
                        "wtd_deliv_per": st.column_config.NumberColumn(
                            "Avg Delivery %", format="%.1f%%",
                            help="Turnover-weighted average delivery % over the period"),
                        "deliv_value_cr": st.column_config.NumberColumn(
                            "Delivery Value (₹ Cr)", format="₹%.1f",
                            help="Total delivery value in ₹ Crores over the period"),
                        "turnover_cr": st.column_config.NumberColumn(
                            "Turnover (₹ Cr)", format="₹%.1f"),
                        "trading_days": st.column_config.NumberColumn(
                            "Trading Days", format="%d"),
                    },
                )


def _render_signal_validation(
    selected_date: date,
    window: int,
    min_turnover: float,
    period_name: str,
) -> None:
    """Show how the rotation clock signals from N days ago actually performed."""
    with st.spinner("Computing signal validation…"):
        bt = cached_rotation_clock_backtest(selected_date, window, float(min_turnover))

    if bt.empty:
        st.info(f"Not enough history to backtest {period_name} signals. Need at least {window * 2} trading days.")
        return

    signal_date   = bt["signal_date"].iloc[0]
    nifty_fwd_ret = bt["forward_nifty_ret"].iloc[0] if "forward_nifty_ret" in bt.columns else None
    using_relative = nifty_fwd_ret is not None

    nifty_note = (
        f"  Nifty50 returned **{nifty_fwd_ret:+.2f}%** over that period — "
        f"'Correct' = sector beat Nifty50 (inflow) or underperformed Nifty50 (outflow)."
        if using_relative else ""
    )
    st.caption(
        f"Signals computed **as of {signal_date.strftime('%d %b %Y')}** "
        f"({window} trading days before {selected_date.strftime('%d %b %Y')}). "
        f"Forward returns measured from that date to today.{nifty_note}"
    )

    # ── Phase-level accuracy summary ──────────────────────────────────────────
    inflow_phases  = ["Leading", "Improving"]
    outflow_phases = ["Weakening", "Lagging"]
    active_phases  = [p for p in ["Leading", "Improving", "Weakening", "Lagging"]
                      if not bt[bt["phase"] == p].empty]

    total_correct    = 0
    total_predicted  = 0

    cols = st.columns(len(active_phases) + 1)

    for i, phase in enumerate(active_phases):
        grp  = bt[(bt["phase"] == phase) & bt["forward_ret_pct"].notna()]
        meta = _PHASE_META[phase]
        if grp.empty:
            cols[i].metric(meta["label"], "—")
            continue

        avg_ret        = grp["forward_ret_pct"].mean()
        avg_vs_nifty   = grp["forward_vs_nifty"].mean() if "forward_vs_nifty" in grp.columns else None
        n_correct      = int(grp["signal_correct"].fillna(False).sum())
        n_total        = len(grp)
        hit_rate       = n_correct / n_total * 100

        total_correct   += n_correct
        total_predicted += n_total

        expected_positive = phase in inflow_phases
        # Color by market-relative avg if available, else absolute
        ref_ret = avg_vs_nifty if avg_vs_nifty is not None else avg_ret
        ret_ok  = (ref_ret > 0) if expected_positive else (ref_ret < 0)

        vs_str = f" (vs Nifty50: {avg_vs_nifty:+.1f}%)" if avg_vs_nifty is not None else ""
        correct_label = (
            "beat market" if expected_positive else "underperform market"
        ) if using_relative else (
            "> 0%" if expected_positive else "< 0%"
        )

        cols[i].metric(
            label=f"{meta['label']} ({n_total})",
            value=f"{avg_ret:+.1f}% avg{vs_str}",
            delta=f"{hit_rate:.0f}% hit  {n_correct}/{n_total}",
            delta_color="normal" if ret_ok else "inverse",
            help=(
                f"{'Inflow' if expected_positive else 'Outflow'} signal.\n"
                f"Correct = forward return {correct_label}.\n"
                + (f"Nifty50 forward: {nifty_fwd_ret:+.2f}%" if nifty_fwd_ret else "")
            ),
        )

    overall_pct = total_correct / total_predicted * 100 if total_predicted else 0
    if   overall_pct >= 65: verdict = "✅ High"
    elif overall_pct >= 50: verdict = "⚖️ Mixed"
    else:                   verdict = "❌ Low"

    cols[-1].metric(
        label="Overall Accuracy",
        value=f"{overall_pct:.0f}%",
        delta=f"{total_correct}/{total_predicted}  {verdict}",
        delta_color="normal" if overall_pct >= 55 else "inverse",
        help="Correct calls ÷ total predicted (excludes Neutral sectors).",
    )

    st.markdown("---")

    # ── Sector detail table ───────────────────────────────────────────────────
    avail_cols = ["sector", "phase", "signal_confidence", "forward_ret_pct",
                  "forward_vs_nifty", "signal_correct",
                  "cum_price_ret_pct", "slope_z", "price_deliv_corr", "deliv_chg_pct"]
    disp = bt[[c for c in avail_cols if c in bt.columns]].copy()

    disp["signal_correct"] = disp["signal_correct"].map(
        lambda v: "✅ Correct" if v is True else ("❌ Wrong" if v is False else "—")
    )

    correct_help = (
        "✅ = sector beat Nifty50 (inflow) or underperformed Nifty50 (outflow)\n"
        "❌ = signal was wrong vs market\n"
        "— = Neutral (no directional prediction)"
    ) if using_relative else (
        "✅ = signal direction matched actual return\n"
        "❌ = signal was wrong\n"
        "— = Neutral (no directional prediction)"
    )

    st.dataframe(
        disp,
        hide_index=True,
        use_container_width=True,
        column_config={
            "sector":            st.column_config.TextColumn("Sector"),
            "phase":             st.column_config.TextColumn("Phase on Signal Date",
                help=f"Rotation phase as of {signal_date.strftime('%d %b %Y')}"),
            "signal_confidence": st.column_config.NumberColumn(
                "Confidence", format="%.2f",
                help="Signal strength 0→1.\n"
                     "Based on slope_z magnitude, price-delivery anti-correlation strength.\n"
                     "High confidence (>0.7) = stronger evidence for the phase call."),
            "forward_ret_pct":   st.column_config.NumberColumn(
                "Forward Return (abs)", format="%+.2f%%",
                help=f"Cumulative sector return from {signal_date.strftime('%d %b')} → {selected_date.strftime('%d %b')}"),
            "forward_vs_nifty":  st.column_config.NumberColumn(
                "vs Nifty50",  format="%+.2f%%",
                help=f"Forward return minus Nifty50 ({nifty_fwd_ret:+.2f}% forward)\n"
                     "Positive = sector outperformed market. Used for accuracy scoring."
                     if nifty_fwd_ret is not None else "Forward return vs Nifty50"),
            "signal_correct":    st.column_config.TextColumn("Correct?", help=correct_help),
            "cum_price_ret_pct": st.column_config.NumberColumn(
                "Price Ret on Signal Date", format="%+.2f%%",
                help=f"Sector return as of {signal_date.strftime('%d %b')} — what triggered classification"),
            "slope_z":           st.column_config.NumberColumn(
                "Delivery Slope Z", format="%+.2f",
                help="Delivery momentum Z-score on the signal date"),
            "price_deliv_corr":  st.column_config.NumberColumn(
                "Price-Del Corr", format="%.2f",
                help="Correlation between daily price return and daily delivery %.\n"
                     "Negative = price rising as delivery falls (distribution confirmed).\n"
                     "Only Weakening signals with corr < -0.15 are shown as Weakening."),
            "deliv_chg_pct":     st.column_config.NumberColumn(
                "Del Chg% on Signal Date", format="%+.1f%%",
                help="Delivery value change vs prior period on the signal date"),
        },
    )


def _render_clock_legend(df: pd.DataFrame) -> None:
    """Compact sector-reference grid below the bubble chart — replaces inline text labels."""
    phase_order  = ["Leading", "Improving", "Neutral", "Weakening", "Lagging"]
    cells_html   = ""

    for phase in phase_order:
        grp = df[df["phase"] == phase].sort_values("deliv_value_cr", ascending=False)
        if grp.empty:
            continue
        meta  = _PHASE_META[phase]
        color = meta["color"]
        label = meta["label"]

        sector_pills = ""
        for _, row in grp.iterrows():
            pr     = row["cum_price_ret_pct"]
            pr_c   = POSITIVE_COLOR if pr > 0 else NEGATIVE_COLOR
            chg    = row.get("deliv_chg_pct")
            chg_s  = f"{chg:+.0f}%" if pd.notna(chg) else "—"
            chg_c  = POSITIVE_COLOR if (pd.notna(chg) and chg > 0) else NEGATIVE_COLOR
            sector_pills += (
                f"<div style='display:inline-flex;align-items:center;gap:5px;"
                f"background:rgba(255,255,255,0.04);border-left:3px solid {color};"
                f"border-radius:0 4px 4px 0;padding:3px 8px;margin:2px;white-space:nowrap'>"
                f"<span style='font-size:12px;font-weight:600'>{row['sector']}</span>"
                f"<span style='font-size:10px;color:{pr_c}'>{pr:+.1f}%</span>"
                f"<span style='font-size:10px;color:{chg_c}'>DV{chg_s}</span>"
                f"</div>"
            )

        cells_html += (
            f"<div style='margin-bottom:6px'>"
            f"<div style='font-size:11px;font-weight:600;color:{color};margin-bottom:3px'>{label}</div>"
            f"<div style='display:flex;flex-wrap:wrap'>{sector_pills}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='background:rgba(255,255,255,0.02);border-radius:6px;"
        f"padding:8px 12px;margin-bottom:12px'>"
        f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin-bottom:6px'>"
        f"SECTOR REFERENCE — hover chart bubbles for detail &nbsp;·&nbsp; "
        f"% = price return &nbsp;·&nbsp; DV% = delivery value change vs prior period</div>"
        f"{cells_html}</div>",
        unsafe_allow_html=True,
    )


def _render_rotation_clock(selected_date: date, min_turnover: float, all_dates: list | None = None) -> None:
    st.caption(
        "Where is institutional money flowing across different time horizons? "
        "Based on **delivery momentum** (slope of daily delivery %) vs **price return** — "
        "like an institutional research firm's sector rotation framework."
    )

    with st.expander("📖 How to read the Rotation Clock", expanded=False):
        st.markdown("""
**RRG-Inspired Framework — 4 Market-Relative Rotation Phases:**

| Phase | Delivery Slope | Price vs Nifty50 | Interpretation | Action |
|-------|---------------|------------------|----------------|--------|
| 💰 **Leading**   | Rising ↑ | Above Nifty50 ↑ | Institutions buying + outperforming market | **BUY / HOLD** |
| 🔍 **Improving** | Rising ↑ | Below Nifty50 ↓ | Institutions accumulating while underperforming — contrarian zone | **ACCUMULATE** |
| ⚠️ **Weakening** | Falling ↓ | Above Nifty50 ↑ | Institutions distributing into outperforming prices | **EXIT / REDUCE** |
| 📤 **Lagging**   | Falling ↓ | Below Nifty50 ↓ | Institutions exiting, price lagging market | **AVOID** |

**Quadrant Center = Nifty50** — The vertical gold dashed line marks the Nifty50 return for the selected period.
Sectors to the RIGHT of the gold line are outperforming the market; sectors to the LEFT are underperforming.
This makes the chart valid in both bull AND bear markets — a sector at +5% is "Improving" not "Lagging" if Nifty50 was +15%.

**Delivery Slope** = Linear regression of daily turnover-weighted delivery % over the period.
Positive slope = institutions are INCREASINGLY committed (building positions).
Negative slope = conviction is FADING (reducing exposure).

**Slope Z-Score** = Cross-sectional z-score across all sectors for the selected period.
Tells you which sectors are gaining or losing institutional interest *relative to each other*.

**Delivery Change %** = Current period delivery value vs the prior equal-length period.
Positive = more institutional money this period than the last one (INFLOW).
Negative = less institutional money (OUTFLOW).

**Bubble Size** = Total delivery value ₹ Cr — larger bubbles = more absolute institutional activity.

**Key insight:** Sectors rotate: Improving → Leading → Weakening → Lagging → Improving.
Ideal entry: sector moving from Improving to Leading (rising delivery + price crossing above Nifty50 baseline).
        """)

    _PERIODS = {
        "1 Week (~5 days)":    5,
        "2 Weeks (~10 days)":  10,
        "1 Month (~22 days)":  22,
        "3 Months (~65 days)": 65,
        "📅 Custom Range":     0,
    }

    sel    = st.radio("Analysis Period", options=list(_PERIODS.keys()), horizontal=True, key="rot_clock_period")
    window = _PERIODS[sel]

    # Custom date range — separate renderer with stock drill-down
    if window == 0:
        _render_custom_range(all_dates or [], min_turnover)
        return

    with st.spinner(f"Computing {sel} sector rotation…"):
        df = cached_sector_rotation_timeframe(selected_date, window, float(min_turnover))

    if df.empty:
        st.warning(f"Insufficient data for {sel} analysis. Need at least {window + 3} trading days of history.")
        return

    # KPI pills
    nifty_ret = df["nifty_return"].iloc[0] if "nifty_return" in df.columns else None
    pc = df["phase"].value_counts()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(
        "🔵 Nifty50",
        f"{nifty_ret:+.2f}%" if nifty_ret is not None else "N/A",
        help="Nifty50 cumulative return for this period — the quadrant center. "
             "Sectors right of this line outperformed the market.",
    )
    k2.metric("💰 Leading",   pc.get("Leading",   0), help="Delivery rising + price beating Nifty50")
    k3.metric("🔍 Improving", pc.get("Improving", 0), help="Delivery rising + price below Nifty50 — contrarian accumulation")
    k4.metric("⚖️ Neutral",   pc.get("Neutral",   0), help="No clear directional bias in delivery momentum")
    k5.metric("⚠️ Weakening", pc.get("Weakening", 0), help="Delivery falling + price beating Nifty50 — distributing into rally")
    k6.metric("📤 Lagging",   pc.get("Lagging",   0), help="Delivery falling + price below Nifty50 — institutional exit")

    # Bubble chart
    st.plotly_chart(
        _rotation_clock_chart(df, sel, nifty_return=nifty_ret),
        use_container_width=True,
        key=f"rot_clock_chart_{window}",
    )

    # Sector reference legend — all sectors in a compact color-coded grid
    _render_clock_legend(df)

    # Two-column phase cards
    leading   = df[df["phase"] == "Leading"].reset_index(drop=True)
    improving = df[df["phase"] == "Improving"].reset_index(drop=True)
    weakening = df[df["phase"] == "Weakening"].reset_index(drop=True)
    lagging   = df[df["phase"] == "Lagging"].reset_index(drop=True)
    neutral   = df[df["phase"] == "Neutral"].reset_index(drop=True)

    col_in, col_out = st.columns(2)

    with col_in:
        st.markdown("#### 🟢 MONEY FLOWING IN")
        if leading.empty and improving.empty:
            st.info("No sectors with strong inflow signal this period.")
        else:
            if not leading.empty:
                st.markdown(
                    f"<div style='font-size:12px;color:#00c853;font-weight:600;margin-bottom:4px'>"
                    f"💰 LEADING — Money Entering ({len(leading)})</div>",
                    unsafe_allow_html=True,
                )
                for _, row in leading.iterrows():
                    _phase_card(row, "#00c853")
            if not improving.empty:
                st.markdown(
                    f"<div style='font-size:12px;color:#40c4ff;font-weight:600;margin:10px 0 4px 0'>"
                    f"🔍 IMPROVING — Contrarian Inflow ({len(improving)})</div>",
                    unsafe_allow_html=True,
                )
                for _, row in improving.iterrows():
                    _phase_card(row, "#40c4ff")

    with col_out:
        st.markdown("#### 🔴 MONEY FLOWING OUT")
        if weakening.empty and lagging.empty:
            st.info("No sectors with strong outflow signal this period.")
        else:
            if not weakening.empty:
                st.markdown(
                    f"<div style='font-size:12px;color:#ff9100;font-weight:600;margin-bottom:4px'>"
                    f"⚠️ WEAKENING — Distribution ({len(weakening)})</div>",
                    unsafe_allow_html=True,
                )
                for _, row in weakening.iterrows():
                    _phase_card(row, "#ff9100")
            if not lagging.empty:
                st.markdown(
                    f"<div style='font-size:12px;color:#d50000;font-weight:600;margin:10px 0 4px 0'>"
                    f"📤 LAGGING — Money Exiting ({len(lagging)})</div>",
                    unsafe_allow_html=True,
                )
                for _, row in lagging.iterrows():
                    _phase_card(row, "#d50000")

    if not neutral.empty:
        with st.expander(f"⚖️ Neutral Sectors — {len(neutral)} with no clear bias", expanded=False):
            for _, row in neutral.iterrows():
                _phase_card(row, "#888888")

    # Cross-period comparison
    with st.expander("📊 Cross-Period Comparison — All 4 Timeframes at Once", expanded=False):
        st.caption("See each sector's rotation phase simultaneously across 1W / 2W / 1M / 3M.")
        _render_cross_period(selected_date, min_turnover)

    # ── Signal Validation ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📊 Signal Validation — Did the {sel} Rotation Clock Call It Right?")
    st.caption(
        f"The rotation clock computed signals **{window} trading days ago**. "
        f"This section shows whether those signals actually predicted what happened since then. "
        f"Inflow signals (Leading/Improving) should have produced positive returns. "
        f"Outflow signals (Weakening/Lagging) should have produced negative returns."
    )
    _render_signal_validation(selected_date, window, min_turnover, sel)

    # Summary table
    st.markdown("---")
    st.markdown(f"#### 📋 All Sectors — {sel} Rotation Summary")

    disp_cols = [
        "sector", "phase", "flow_signal",
        "cum_price_ret_pct", "slope_z", "delivery_slope",
        "deliv_value_cr", "deliv_chg_pct", "avg_deliv_pct", "num_days",
    ]
    disp = df[[c for c in disp_cols if c in df.columns]].copy()

    st.dataframe(
        disp,
        hide_index=True,
        use_container_width=True,
        column_config={
            "sector":            st.column_config.TextColumn("Sector"),
            "phase":             st.column_config.TextColumn("Phase"),
            "flow_signal":       st.column_config.TextColumn("Flow Signal"),
            "cum_price_ret_pct": st.column_config.NumberColumn(
                "Price Return %", format="%+.2f%%",
                help="Cumulative turnover-weighted price return over the selected period\n"
                     "= compound product of daily sector returns"),
            "slope_z":           st.column_config.NumberColumn(
                "Slope Z-Score", format="%+.2f",
                help="Cross-sectional z-score of delivery slope across all sectors\n"
                     "Tells you which sectors are gaining vs losing institutional interest relative to each other\n"
                     "> 0.25σ = rising momentum   < -0.25σ = falling momentum"),
            "delivery_slope":    st.column_config.NumberColumn(
                "Delivery Slope", format="%+.4f",
                help="Linear regression slope of daily weighted delivery %\n"
                     "Positive = institutions increasingly committed over the period\n"
                     "Units: delivery % points per trading day"),
            "deliv_value_cr":    st.column_config.NumberColumn(
                "Deliv Value (₹ Cr)", format="₹%.1f",
                help="Total delivery value over the period in ₹ Crores"),
            "deliv_chg_pct":     st.column_config.NumberColumn(
                "Deliv Chg vs Prior %", format="%+.1f%%",
                help="% change in delivery value vs the prior equal-length period\n"
                     "Positive = more institutional money flowing in this period\n"
                     "Negative = less institutional money (outflow vs prior period)"),
            "avg_deliv_pct":     st.column_config.NumberColumn(
                "Avg Delivery %", format="%.1f%%",
                help="Average turnover-weighted delivery % over the current period"),
            "num_days":          st.column_config.NumberColumn("Trading Days", format="%d"),
        },
    )


# ── Smart Money Tab (existing content) ───────────────────────────────────────

def _render_smart_money(selected_date: date, min_turnover: float) -> None:
    st.caption(
        f"Where are institutions entering and exiting? "
        f"Based on **100 days** of turnover-weighted delivery data "
        f"— as of **{selected_date.strftime('%d %b %Y')}**"
    )

    with st.expander("📖 How to read this page", expanded=False):
        st.markdown("""
**Why delivery % alone misleads you:**
A ₹5,000 stock with 60% delivery and 1L volume = ₹30 Cr delivered.
A ₹50 stock with 80% delivery and 10L volume = ₹4 Cr delivered.
The first stock has *far* more real institutional commitment — which is why we weight by turnover (₹ value traded).

**The signal matrix (X = 1W cumulative price return, Y = Z-Score):**
| | Z-Score ≥ +1σ AND Delivery% above 100D avg | Z-Score ≥ +1σ BUT Delivery% BELOW 100D avg | Z-Score ≤ -0.5σ |
|---|---|---|---|
| **Price UP** | ✅ Confirmed Accumulation — enter/hold | 📊 Volume Spike — CAUTION | ⚠️ Distribution Trap — EXIT |
| **Price DOWN** | 🔥 Secret Accumulation — **best entry** | 📊 Volume Spike — CAUTION | ❌ Active Selling — avoid |
| **Price flat** | 👀 Early Accumulation — watch | 📊 Volume Spike — CAUTION | 📉 Weakening — reduce |

**📊 Volume Spike — the false-positive filter:**
When speculative events (news, results, global macro) explode trading volumes 10–20× normal, absolute
delivery value (₹ Cr) rises *mathematically* even when delivery % falls. Without this check,
that would mislabel the sector as "Confirmed Accumulation." Volume Spike is triggered when:
*Z-Score ≥ +1σ (value surge) AND delivery% fell more than 15% below its own 100D average.*
A marginal dip (e.g. 98% of average) is treated as normal — only a genuine conviction collapse (< 85% of avg) is flagged.

**Secret Accumulation is the most powerful signal.** Institutions buy quietly while retail panics on falling prices. When delivery Z-Score surges above +1σ AND delivery% is above normal despite falling prices, smart money is building a position.

**Distribution Trap is the most dangerous.** Institutions need retail buyers to exit into. If delivery Z-Score collapses (below -0.5σ) while price rises, institutions are selling into the retail FOMO rally.

**DV Today** — today's delivered value ÷ own 100D daily average. 1.5× = 50% above normal. Single-day snapshot — can spike from one large block trade.

**5D Avg DV** — 5-day average DV ratio (1W delivery ÷ 5 days, vs 100D daily mean). Smoothed institutional activity over a week — the primary signal driver. Avoids single-day noise.

**Z-Score (σ)** — how many standard deviations today's delivery VALUE is above its 100D mean. Z ≥ 2.0 = top 2.5% of trading days.

**Breadth** — fraction of stocks in the sector where today's delivery exceeds their own 100D average. 70%+ = broad institutional participation.

**Score (0–100):** 30% RS vs Nifty + 25% 5D Avg DV + 15% DV Today + 15% Breadth + 15% Z-Score. Cross-sectional — ranks sectors relative to each other.
        """)

    with st.spinner("Computing 100-day rotation signals…"):
        rot = cached_sector_rotation(selected_date, min_turnover)

    if rot.empty:
        st.warning("Insufficient data. Need at least 10 trading days of history.")
        return

    if "z_score" not in rot.columns:
        st.cache_data.clear()
        st.rerun()

    _INVEST_SIGNALS   = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation", "👀 Early Accumulation"}
    _CAUTION_SIGNALS  = {"📊 Volume Spike"}
    _AVOID_SIGNALS    = {"⚠️ Distribution Trap", "❌ Active Selling", "📉 Weakening"}

    entering  = rot[rot["signal"].isin(_INVEST_SIGNALS)].copy()
    caution   = rot[rot["signal"].isin(_CAUTION_SIGNALS)].copy()
    exiting   = rot[rot["signal"].isin(_AVOID_SIGNALS)].sort_values("accum_score", ascending=True).copy()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🔥 Secret Accum",   len(rot[rot["signal"] == "🔥 Secret Accumulation"]),
              help="Price falling + Z-Score ≥ +1σ + Delivery% above 100D avg — best entry zone")
    k2.metric("✅ Confirmed Buy",    len(rot[rot["signal"] == "✅ Confirmed Accumulation"]),
              help="Price rising + Z-Score ≥ +1σ + Delivery% above 100D avg — momentum confirmed")
    k3.metric("📊 Volume Spike",    len(rot[rot["signal"] == "📊 Volume Spike"]),
              help="Z-Score ≥ +1σ BUT delivery% BELOW 100D avg — speculative surge, not institutional conviction")
    k4.metric("⚠️ Distribution",    len(rot[rot["signal"] == "⚠️ Distribution Trap"]),
              help="Price rising + Z-Score ≤ -0.5σ — institutions selling into retail rally")
    k5.metric("❌ Active Selling",  len(rot[rot["signal"] == "❌ Active Selling"]),
              help="Price falling + Z-Score ≤ -0.5σ — avoid")

    st.markdown("---")

    st.markdown("### 📊 Smart Money Quadrant")
    st.caption(
        "X = 1-week cumulative price return (%).  "
        "Y = Delivery Z-Score (σ above 100D mean) — bubble size = Score (0–100).  "
        "Hover any bubble for DV Ratio, Z-Score, Breadth detail."
    )
    st.plotly_chart(_quadrant_chart(rot), use_container_width=True)

    with st.expander("🗂️ Sector Reference — full list ranked by score", expanded=False):
        ref_cols = ["sector", "signal", "accum_score", "coverage",
                    "dv_ratio", "z_score", "breadth", "price_1w", "action"]
        ref_df = rot[[c for c in ref_cols if c in rot.columns]].copy()

        def _action_colors(action: str):
            a = str(action).upper()
            if "STRONG BUY" in a or a.startswith("BUY"):
                return "#00c853", "rgba(0,200,83,0.18)"
            if "EXIT" in a or "AVOID" in a:
                return "#ff5252", "rgba(213,0,0,0.22)"
            if "REDUCE" in a:
                return "#ff9100", "rgba(255,109,0,0.22)"
            if "CAUTION" in a:
                return "#ffd600", "rgba(255,214,0,0.18)"
            if "WATCH" in a:
                return "#ffca28", "rgba(255,202,40,0.22)"
            return "#888888", "rgba(120,120,120,0.15)"

        def _score_bar(score, action) -> str:
            score = int(score or 0)
            txt_c, bar_c = _action_colors(action)
            return (
                f"<div style='position:relative;width:100%;height:22px;"
                f"background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden'>"
                f"<div style='position:absolute;left:0;top:0;height:100%;width:{score}%;"
                f"background:{bar_c};border-radius:4px'></div>"
                f"<div style='position:absolute;left:0;top:0;width:100%;height:100%;"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-weight:700;font-size:13px;color:{txt_c}'>{score}</div>"
                f"</div>"
            )

        def _fmt_val(v, fmt, plus=False):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "<span style='color:#555'>—</span>"
            color = "#00c853" if v > 0 else "#ff5252" if v < 0 else "#888"
            prefix = "+" if plus and v > 0 else ""
            return f"<span style='color:{color}'>{prefix}{fmt.format(v)}</span>"

        def _fmt_breadth(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return "<span style='color:#555'>—</span>"
            pct = v * 100
            color = "#00c853" if pct >= 70 else "#64dd17" if pct >= 50 else "#888" if pct >= 30 else "#ff5252"
            return f"<span style='color:{color}'>{pct:.0f}%</span>"

        def _signal_badge(sig: str) -> str:
            meta = _SIGNAL_META.get(sig, {})
            color = meta.get("color", "#888")
            return (
                f"<span style='background:rgba(255,255,255,0.07);border-left:3px solid {color};"
                f"padding:2px 8px;border-radius:0 4px 4px 0;font-size:12px'>{sig}</span>"
            )

        rows_html = ""
        for _, row in ref_df.iterrows():
            action   = str(row.get("action", ""))
            score    = row.get("accum_score", 0)
            signal   = str(row.get("signal", ""))
            coverage = str(row.get("coverage", "—") or "—")
            dv       = row.get("dv_ratio")
            dv5d_t   = row.get("dv_ratio_5d")
            z        = row.get("z_score")
            br       = row.get("breadth")
            p1w      = row.get("price_1w")
            action_short = action.split("—")[0].strip() if "—" in action else action[:18]
            txt_c, _ = _action_colors(action)

            rows_html += (
                f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>"
                f"<td style='padding:6px 10px;font-size:13px;font-weight:500'>{row.get('sector','')}</td>"
                f"<td style='padding:6px 8px'>{_signal_badge(signal)}</td>"
                f"<td style='padding:6px 8px;width:130px'>{_score_bar(score, action)}</td>"
                f"<td style='padding:6px 8px;font-size:12px;color:#aaa'>{coverage}</td>"
                f"<td style='padding:6px 8px;text-align:right'>{_fmt_val(dv, '{:.2f}×')}</td>"
                f"<td style='padding:6px 8px;text-align:right'>{_fmt_val(dv5d_t, '{:.2f}×')}</td>"
                f"<td style='padding:6px 8px;text-align:right'>{_fmt_val(z, '{:+.2f}σ', plus=True)}</td>"
                f"<td style='padding:6px 8px;text-align:right'>{_fmt_breadth(br)}</td>"
                f"<td style='padding:6px 8px;text-align:right'>{_fmt_val(p1w, '{:+.2f}%', plus=True)}</td>"
                f"<td style='padding:6px 8px;font-size:11px;color:{txt_c};font-weight:600'>{action_short}</td>"
                f"</tr>"
            )

        header = (
            "<tr style='border-bottom:2px solid rgba(255,255,255,0.12)'>"
            + "".join(
                f"<th style='padding:6px 8px;font-size:11px;color:rgba(255,255,255,0.5);"
                f"font-weight:600;text-transform:uppercase;letter-spacing:0.5px;"
                f"text-align:{'right' if i >= 4 else 'left'}'>{h}</th>"
                for i, h in enumerate([
                    "Sector", "Signal", "Score", "Coverage",
                    "DV Today", "5D Avg", "Z-Score", "Breadth", "Price 1W%", "Action"
                ])
            )
            + "</tr>"
        )

        st.markdown(
            f"<div style='overflow-x:auto'>"
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<thead>{header}</thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Per-stock delivery filter — controls every "View stocks in …" drill-down below.
    deliv_threshold = st.slider(
        "Min stock Wtd Delivery % — filters the per-stock lists below",
        min_value=0, max_value=100, value=int(_MIN_STOCK_WTD_DELIV_PCT), step=1,
        key="rotation_stock_deliv_threshold",
        help="Hide stocks whose 7-day turnover-weighted delivery % is at or below this value "
             "inside the 'View stocks in …' expanders. Sector-level stats (top-3 contributors, "
             "single-stock dominance warning) still use the full stock set.",
    )

    col_enter, col_avoid = st.columns(2)

    _HIGH_CONV = 70

    with col_enter:
        st.markdown("### 🟢 SECTORS TO INVEST")
        st.caption("All accumulation signals — highest score first. Score = institutional conviction strength.")
        if entering.empty:
            st.info("No sectors with accumulation signal today.")
        else:
            shown_divider = False
            for _, row in entering.iterrows():
                if not shown_divider and row["accum_score"] < _HIGH_CONV:
                    st.markdown(
                        "<div style='margin:10px 0 6px 0;border-top:1px solid rgba(255,255,255,0.08);"
                        "padding-top:6px;font-size:11px;color:rgba(255,255,255,0.35);"
                        "letter-spacing:0.5px'>MODERATE CONVICTION</div>",
                        unsafe_allow_html=True,
                    )
                    shown_divider = True
                _sector_card(row, selected_date, min_turnover, deliv_threshold)

    with col_avoid:
        st.markdown("### 🔴 SECTORS TO AVOID / EXIT")
        st.caption(
            "Active distribution/selling signals first; then relative laggards — "
            "the weakest sectors by score when no genuine distribution exists."
        )

        # Tier 1 — genuine distribution / selling (real institutional exit, red).
        if not exiting.empty:
            st.markdown(
                "<div style='font-size:11px;font-weight:600;color:#d50000;"
                "margin-bottom:4px'>ACTIVE DISTRIBUTION / SELLING</div>",
                unsafe_allow_html=True,
            )
            for _, row in exiting.iterrows():
                _sector_card(row, selected_date, min_turnover, deliv_threshold)

        # Tier 2 — relative laggards: weakest sectors by score, excluding any
        # already shown in the invest / caution / distribution lists. The absolute
        # z<=-0.5 distribution gate cannot fire in a strong-delivery regime (z is
        # positive market-wide), so without this the column would be empty even
        # when clear underweight candidates exist. These are NOT active selling —
        # they are the relative weakest in today's cross-section.
        shown = set(entering["sector"]) | set(caution["sector"]) | set(exiting["sector"])
        laggards = rot[~rot["sector"].isin(shown)].nsmallest(5, "accum_score")

        if laggards.empty and exiting.empty:
            st.info("No distribution signals and no clear laggards today.")
        elif not laggards.empty:
            st.markdown(
                "<div style='font-size:11px;font-weight:600;color:#ff9100;"
                "margin:10px 0 4px 0'>RELATIVE LAGGARDS — weakest by score "
                "(underweight, not active selling)</div>",
                unsafe_allow_html=True,
            )
            for _, row in laggards.iterrows():
                _sector_card(row, selected_date, min_turnover, deliv_threshold)

    if not caution.empty:
        st.markdown("---")
        st.markdown("### 📊 VOLUME SPIKE — DO NOT CONFUSE WITH ACCUMULATION")
        st.caption(
            "Z-Score is high (delivery VALUE surged) BUT delivery % is BELOW its 100D average. "
            "Speculative event-driven trading — not institutional conviction. "
            "Do not buy based on delivery value alone."
        )
        for _, row in caution.iterrows():
            _sector_card(row, selected_date, min_turnover, deliv_threshold)

    st.markdown("---")

    st.markdown("### 📈 100-Day Delivery Trend — Drill Into a Sector")
    st.caption("See exactly how delivery % and delivery value evolved — the trend tells the story")

    sector_options = rot["sector"].tolist()
    chosen = st.selectbox(
        "Select sector to inspect",
        options=sector_options,
        format_func=lambda s: f"{rot.loc[rot['sector']==s,'signal'].values[0]}  {s}",
        key="rotation_sector_select",
    )

    if chosen:
        row = rot[rot["sector"] == chosen].iloc[0]
        meta = _SIGNAL_META.get(row["signal"], {})

        dv_disp   = f"{row['dv_ratio']:.2f}×"     if pd.notna(row.get("dv_ratio"))     else "—"
        dv5d_raw  = row.get("dv_ratio_5d")
        dv5d_disp = f"{dv5d_raw:.2f}×"            if (dv5d_raw is not None and pd.notna(dv5d_raw)) else "—"
        z_disp    = f"{row['z_score']:+.2f}σ"     if pd.notna(row.get("z_score"))      else "—"
        br_raw    = row.get("breadth")
        br_disp   = f"{br_raw*100:.0f}% breadth"  if (br_raw is not None and pd.notna(br_raw)) else "—"

        st.markdown(
            f"<div style='padding:10px 16px;border-left:4px solid "
            f"{meta.get('color','#888')};background:rgba(255,255,255,0.04);"
            f"border-radius:0 8px 8px 0;margin:8px 0'>"
            f"<b style='font-size:16px'>{row['signal']}  —  {chosen}</b><br>"
            f"<span style='color:rgba(255,255,255,0.7)'>{row['action']}</span><br>"
            f"<span style='font-size:12px;color:rgba(255,255,255,0.5)'>"
            f"Score: {row['accum_score']:.0f}/100 &nbsp;|&nbsp; "
            f"DV Today: {dv_disp} &nbsp;|&nbsp; 5D Avg: {dv5d_disp} &nbsp;|&nbsp; "
            f"Z-Score: {z_disp} &nbsp;|&nbsp; {br_disp}"
            f" &nbsp;|&nbsp; Horizon: {row['horizon']}"
            f"</span></div>",
            unsafe_allow_html=True,
        )

        with st.spinner(f"Loading 100-day trend for {chosen}…"):
            hist = cached_sector_rotation_history(chosen, selected_date, min_turnover)

        if not hist.empty:
            st.plotly_chart(_trend_chart(hist, chosen, row["signal"]),
                            use_container_width=True)

    st.markdown("---")

    with st.expander("📋 Full Rotation Table — All Sectors", expanded=False):
        st.caption(
            "All sectors ranked by accumulation score.  "
            "Score = 30% RS vs Nifty + 25% 5D Avg DV + 15% DV Today + 15% Breadth + 15% Z-Score"
        )
        display_cols = ["sector", "signal", "accum_score", "coverage", "horizon",
                        "dv_ratio", "dv_ratio_5d", "z_score", "breadth", "trend_slope",
                        "price_1w", "price_1m", "price_3m",
                        "today_dv_cr", "deliv_val_1w_cr",
                        "today_wtd_deliv_pct", "avg_wtd_deliv_pct_100d"]
        display = rot[[c for c in display_cols if c in rot.columns]].copy()

        st.dataframe(
            display,
            column_config={
                "sector":         st.column_config.TextColumn("Sector"),
                "signal":         st.column_config.TextColumn("Signal"),
                "accum_score":    st.column_config.ProgressColumn(
                    "Score", max_value=100, format="%.0f",
                    help="Score = 30% RS vs Nifty + 25% 5D Avg DV + 15% DV Today + 15% Breadth + 15% Z-Score\n"
                         "Cross-sectional rank: ranks sectors relative to each other on today's data"),
                "coverage":       st.column_config.TextColumn(
                    "Coverage",
                    help="Swing (3–15 days): Z-Score ≥ 2σ + Breadth ≥ 50%\n"
                         "Positional (4–8 weeks): DV Ratio > 1.2 + positive slope + Breadth ≥ 40%\n"
                         "Mid Term (3–4 months): steep 100-day slope + DV Ratio > 1.3 + Breadth ≥ 50%"),
                "horizon":        st.column_config.TextColumn("Horizon"),
                "dv_ratio":       st.column_config.NumberColumn(
                    "DV Today", format="%.2f×",
                    help="Today's delivered value ÷ own 100D daily average\n"
                         "1.0× = exactly average  |  1.5× = 50% above norm\n"
                         "Single-day snapshot — can spike from one large block trade"),
                "dv_ratio_5d":    st.column_config.NumberColumn(
                    "5D Avg DV", format="%.2f×",
                    help="5-day average DV ratio = (1W delivery ÷ 5) ÷ (100D delivery ÷ 100)\n"
                         "1.0× = exactly normal  |  1.3× = 30% above weekly average\n"
                         "Primary signal driver — smooths single-day noise over a week"),
                "z_score":        st.column_config.NumberColumn(
                    "Z-Score (σ)", format="%+.2f",
                    help="(Today's DV − 100D mean) ÷ 100D std deviation\n"
                         "Z ≥ 2.0 = top 2.5% of days  |  Z ≥ 1.0 = top 16%  |  Z ≤ -0.5 = below normal\n"
                         "Statistically grounded — adapts to each sector's own delivery volatility"),
                "breadth":        st.column_config.NumberColumn(
                    "Breadth", format="%.0f%%",
                    help="% of stocks in sector where today's delivery > own 100D avg daily DV\n"
                         "70%+ = broad institutional participation\n"
                         "30% or below = one large-cap driving the sector signal"),
                "trend_slope":    st.column_config.NumberColumn(
                    "Trend Slope", format="%+.3f",
                    help="Linear regression slope of 100-day delivery % series\n"
                         "Normalised by mean — % change per trading day\n"
                         "Positive = delivery trend rising  |  Negative = delivery trend falling"),
                "price_1w":       st.column_config.NumberColumn(
                    "1W Price%", format="%+.2f%%",
                    help="Cumulative 1W price return: (today_close − 5D_ago_close) / 5D_ago_close × 100"),
                "price_1m":       st.column_config.NumberColumn("1M Price%",  format="%+.2f%%"),
                "price_3m":       st.column_config.NumberColumn("3M Price%",  format="%+.2f%%"),
                "today_dv_cr":    st.column_config.NumberColumn(
                    "Today DV (₹ Cr)", format="₹%.1f",
                    help="Today's single-day delivered value in ₹ Crores\n"
                         "Absolute size of today's institutional activity"),
                "deliv_val_1w_cr":st.column_config.NumberColumn(
                    "1W Deliv Val (₹ Cr)", format="₹%.1f",
                    help="₹ value of shares delivered in last 1 week — total institutional conviction"),
                "today_wtd_deliv_pct": st.column_config.NumberColumn(
                    "Today Del%", format="%.1f%%",
                    help="Today's turnover-weighted delivery %\n\n"
                         "Conviction quality check: if this is BELOW the 100D avg, a high Z-Score\n"
                         "is a Volume Spike (speculative), not institutional accumulation.\n"
                         "Formula: Σ(deliv_per × turnover_lacs) / Σ(turnover_lacs)"),
                "avg_wtd_deliv_pct_100d": st.column_config.NumberColumn(
                    "100D Avg Del%", format="%.1f%%",
                    help="Sector's own 100-trading-day average turnover-weighted delivery %\n\n"
                         "Baseline for the conviction quality check.\n"
                         "Today Del% > this → pct_surge = True → genuine institutional activity\n"
                         "Today Del% < this → pct_surge = False → Volume Spike, not accumulation"),
            },
            use_container_width=True,
            hide_index=True,
        )


# ── Relative Strength vs Nifty50 ─────────────────────────────────────────────

_RS_PERIOD_META = {
    "1w":     ("rs_1w",     "nifty_1w",     "price_1w",  "1 Week"),
    "2w":     ("rs_2w",     "nifty_2w",     "price_2w",  "2 Week"),
    "1m":     ("rs_1m",     "nifty_1m",     "price_1m",  "1 Month"),
    "custom": ("rs_custom", "nifty_custom", "price_custom", "Custom"),
}


def _rs_bar_chart(df: pd.DataFrame, rs_col: str, period_label: str, nifty_val: float | None = None) -> go.Figure:
    """Horizontal bar chart: sector RS vs Nifty50. Zero line annotated with Nifty50 return."""
    if rs_col not in df.columns:
        return go.Figure()

    plot = df[["sector", rs_col, "signal"]].dropna(subset=[rs_col]).copy()
    plot = plot.sort_values(rs_col, ascending=True).reset_index(drop=True)

    bar_colors = [
        _SIGNAL_META.get(sig, {}).get("color", "#888888") if rs >= 0 else NEGATIVE_COLOR
        for sig, rs in zip(plot["signal"], plot[rs_col])
    ]

    fig = go.Figure(go.Bar(
        x=plot[rs_col],
        y=plot["sector"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:+.2f}%" for v in plot[rs_col]],
        textposition="outside",
        textfont=dict(size=10),
        customdata=plot[["signal", rs_col]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "RS vs Nifty50: <b>%{x:+.2f}%</b><br>"
            "Signal: %{customdata[0]}"
            "<extra></extra>"
        ),
    ))

    nifty_annotation = (
        f"Nifty50 ({period_label}): {nifty_val:+.2f}%"
        if nifty_val is not None and not (isinstance(nifty_val, float) and pd.isna(nifty_val))
        else "Nifty50 = 0 (benchmark)"
    )
    fig.add_vline(
        x=0,
        line_color="rgba(255,215,0,0.7)",
        line_width=2,
        annotation_text=nifty_annotation,
        annotation_position="top",
        annotation_font=dict(size=11, color="#FFD700"),
    )

    fig.update_layout(
        title=dict(
            text=(
                f"Sector Relative Strength vs Nifty50 — {period_label}  "
                "<span style='font-size:11px;color:rgba(255,255,255,0.4)'>"
                "Positive = outperforming  ·  Negative = lagging  ·  "
                "Gold line = Nifty50 benchmark</span>"
            ),
            font=dict(size=14),
        ),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(
            title="Excess Return vs Nifty50 (%)",
            showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, ticksuffix="%",
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        height=max(440, len(plot) * 26 + 120),
        margin=dict(t=70, b=50, l=190, r=110),
    )
    return fig


def _rs_delivery_scatter(
    df: pd.DataFrame,
    period_label: str,
    rs_col: str,
    nifty_col: str,
    price_col: str,
) -> go.Figure:
    """
    RS × Delivery scatter — the definitive institutional lens.
    X = RS vs Nifty50  ·  Y = Delivery Z-Score
    Hover for detail; no inline labels (use legend panel below chart).
    """
    plot = df.dropna(subset=[rs_col, "z_score"]).copy()
    if plot.empty:
        return go.Figure()

    x_vals = plot[rs_col]
    y_vals = plot["z_score"]
    x_pad = max((x_vals.max() - x_vals.min()) * 0.28, 1.5)
    y_pad = max((y_vals.max() - y_vals.min()) * 0.28, 0.5)
    x0, x1 = x_vals.min() - x_pad, x_vals.max() + x_pad
    y0, y1 = y_vals.min() - y_pad, y_vals.max() + y_pad

    fig = go.Figure()

    fig.add_shape(type="rect", x0=x0, x1=0, y0=0, y1=y1,
                  fillcolor="rgba(64,196,255,0.09)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=x1, y0=0, y1=y1,
                  fillcolor="rgba(0,200,83,0.09)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0,
                  fillcolor="rgba(213,0,0,0.09)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=x1, y0=y0, y1=0,
                  fillcolor="rgba(255,109,0,0.07)", line_width=0, layer="below")

    for label, lx, ly, xanchor, yanchor, bgcolor in [
        ("🔍 HIDDEN ACCUMULATION\n(Best Entry)",  x0 + x_pad*0.3, y1 - y_pad*0.3, "left",  "top",    "rgba(64,196,255,0.18)"),
        ("💰 LEADING\n(Strong Hold / Add)",        x1 - x_pad*0.3, y1 - y_pad*0.3, "right", "top",    "rgba(0,200,83,0.18)"),
        ("📤 LAGGING\n(Avoid)",                   x0 + x_pad*0.3, y0 + y_pad*0.3, "left",  "bottom", "rgba(213,0,0,0.18)"),
        ("⚠️ DISTRIBUTION\n(Exit / Short)",       x1 - x_pad*0.3, y0 + y_pad*0.3, "right", "bottom", "rgba(255,109,0,0.18)"),
    ]:
        fig.add_annotation(
            x=lx, y=ly, text=f"<b>{label}</b>",
            showarrow=False, font=dict(size=11, color="rgba(255,255,255,0.70)"),
            xanchor=xanchor, yanchor=yanchor, bgcolor=bgcolor, borderpad=5, align="center",
        )

    fig.add_hline(y=0,    line_color="rgba(255,255,255,0.35)", line_width=1.5)
    fig.add_hline(y=1.0,  line_dash="dash", line_width=1.0,
                  line_color="rgba(0,200,83,0.45)",
                  annotation_text="Surge threshold (Z=+1σ)",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="rgba(0,200,83,0.7)"))
    if y0 < -0.5:
        fig.add_hline(y=-0.5, line_dash="dash", line_width=1.0,
                      line_color="rgba(255,80,0,0.45)",
                      annotation_text="Weakness threshold (Z=−0.5σ)",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color="rgba(255,80,0,0.7)"))
    fig.add_vline(x=0, line_color="rgba(255,215,0,0.55)", line_width=2,
                  annotation_text="Nifty50",
                  annotation_position="top",
                  annotation_font=dict(size=10, color="#FFD700"))

    safe_price_col = price_col if price_col in plot.columns else rs_col
    safe_nifty_col = nifty_col if nifty_col in plot.columns else None

    for signal in [
        "🔥 Secret Accumulation", "✅ Confirmed Accumulation",
        "👀 Early Accumulation", "📊 Volume Spike", "⚖️ Neutral",
        "📉 Weakening", "⚠️ Distribution Trap", "❌ Active Selling",
    ]:
        grp = plot[plot["signal"] == signal]
        if grp.empty:
            continue
        color = _SIGNAL_META.get(signal, {}).get("color", "#888888")
        sizes = (grp["accum_score"] / 100 * 24 + 14).clip(14, 38)

        nifty_vals = (
            grp[safe_nifty_col].values if safe_nifty_col else
            [float("nan")] * len(grp)
        )
        customdata = list(zip(
            grp["sector"].values,
            grp["signal"].values,
            grp["accum_score"].values,
            grp["z_score"].values,
            grp[rs_col].values,
            grp[safe_price_col].values,
            nifty_vals,
        ))

        fig.add_trace(go.Scatter(
            x=grp[rs_col],
            y=grp["z_score"],
            mode="markers",
            name=signal,
            marker=dict(
                color=color, size=sizes, opacity=0.90,
                line=dict(width=1.5, color="rgba(255,255,255,0.5)"),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"<span style='color:{color}'>%{{customdata[1]}}</span><br>"
                "────────────────────<br>"
                f"RS vs Nifty50 ({period_label}): <b>%{{customdata[4]:+.2f}}%</b><br>"
                "Z-Score (Delivery): <b>%{customdata[3]:+.2f}σ</b><br>"
                "Score: %{customdata[2]:.0f}/100<br>"
                f"Sector {period_label}: %{{customdata[5]:+.2f}}%"
                "  ·  Nifty50: %{customdata[6]:+.2f}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        title=dict(
            text=(
                f"RS vs Nifty50 ({period_label}) × Institutional Delivery  "
                "<span style='font-size:11px;color:rgba(255,255,255,0.4)'>"
                "Top-Left = hidden accumulation (best entry)  ·  "
                "Top-Right = leading (hold/add)  ·  "
                "Gold line = Nifty50 &nbsp;·&nbsp; Hover for sector detail</span>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(
            title=f"← Lagging Nifty50  |  RS vs Nifty50 ({period_label}) %  |  Leading Nifty50 →",
            showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, ticksuffix="%", tickfont=dict(size=11),
            range=[x0, x1],
        ),
        yaxis=dict(
            title="← Institutions Exiting  |  Delivery Z-Score (σ)  |  Institutions Entering →",
            showgrid=True, gridcolor=GRID_COLOR,
            zeroline=False, ticksuffix="σ", tickfont=dict(size=11),
            range=[y0, y1],
        ),
        legend=dict(
            orientation="h", y=-0.14, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        ),
        height=620,
        margin=dict(t=70, b=90, l=100, r=40),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=13,
                        bordercolor="rgba(255,255,255,0.2)"),
        hovermode="closest",
    )
    return fig


def _render_rs_legend(df: pd.DataFrame, rs_col: str) -> None:
    """Compact sector reference grid below the RS scatter — grouped by quadrant."""
    if rs_col not in df.columns or "z_score" not in df.columns:
        return

    quadrants = [
        ("🔍 HIDDEN ACCUM",  df[(df[rs_col] < 0) & (df["z_score"] >= 0)], "#40c4ff"),
        ("💰 LEADING",        df[(df[rs_col] >= 0) & (df["z_score"] >= 0)], "#00c853"),
        ("📤 LAGGING",        df[(df[rs_col] < 0) & (df["z_score"] < 0)],  "#d50000"),
        ("⚠️ DISTRIBUTION",  df[(df[rs_col] >= 0) & (df["z_score"] < 0)], "#ff9100"),
    ]

    cells_html = ""
    for label, grp, color in quadrants:
        if grp.empty:
            continue
        grp = grp.sort_values(rs_col, ascending=False)
        pills = ""
        for _, row in grp.iterrows():
            rs_v  = row[rs_col]
            z_v   = row["z_score"]
            rs_c  = POSITIVE_COLOR if rs_v >= 0 else NEGATIVE_COLOR
            z_c   = "#00c853" if z_v >= 1 else "#ff9100" if z_v >= 0 else "#888"
            pills += (
                f"<div style='display:inline-flex;align-items:center;gap:5px;"
                f"background:rgba(255,255,255,0.04);border-left:3px solid {color};"
                f"border-radius:0 4px 4px 0;padding:3px 8px;margin:2px;white-space:nowrap'>"
                f"<span style='font-size:12px;font-weight:600'>{row['sector']}</span>"
                f"<span style='font-size:10px;color:{rs_c}'>RS{rs_v:+.1f}%</span>"
                f"<span style='font-size:10px;color:{z_c}'>Z{z_v:+.1f}σ</span>"
                f"</div>"
            )
        cells_html += (
            f"<div style='margin-bottom:5px'>"
            f"<div style='font-size:11px;font-weight:600;color:{color};margin-bottom:3px'>{label}</div>"
            f"<div style='display:flex;flex-wrap:wrap'>{pills}</div>"
            f"</div>"
        )

    if cells_html:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.02);border-radius:6px;"
            f"padding:8px 12px;margin:0 0 12px 0'>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin-bottom:6px'>"
            f"SECTOR REFERENCE — hover chart bubbles for detail &nbsp;·&nbsp; "
            f"RS = excess return vs Nifty50 &nbsp;·&nbsp; Z = delivery z-score</div>"
            f"{cells_html}</div>",
            unsafe_allow_html=True,
        )


def _render_rs_charts(df: pd.DataFrame, period: str, period_label: str) -> None:
    """Render scatter + legend + bar for a given period (preset or custom)."""
    rs_col, nifty_col, price_col, _ = _RS_PERIOD_META[period]

    nifty_val = None
    if nifty_col in df.columns:
        v = df[nifty_col].dropna()
        nifty_val = float(v.iloc[0]) if not v.empty else None

    outperformers = int(df[rs_col].gt(0).sum()) if rs_col in df.columns else 0

    # ── Nifty50 return strip for selected period ──────────────────────────────
    _c1, _c2, _c3 = st.columns([2, 2, 3])
    def _pct_html(v, label):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            val_html = "<span style='font-size:1.15rem;font-weight:700;color:#888'>—</span>"
        else:
            c = POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR
            val_html = f"<span style='font-size:1.15rem;font-weight:700;color:{c}'>{v:+.2f}%</span>"
        return (
            f"<div style='text-align:center;padding:10px 14px;"
            f"background:rgba(255,215,0,0.07);border-radius:8px;"
            f"border-top:2px solid #FFD700'>"
            f"<div style='font-size:10px;color:#FFD700;letter-spacing:1px;font-weight:600'>"
            f"NIFTY50 · {label}</div>"
            f"<div style='margin-top:4px'>{val_html}</div></div>"
        )
    _c1.markdown(_pct_html(nifty_val, period_label), unsafe_allow_html=True)
    _c2.markdown(
        f"<div style='text-align:center;padding:10px 14px;"
        f"background:rgba(76,175,80,0.07);border-radius:8px;"
        f"border-top:2px solid #4CAF50'>"
        f"<div style='font-size:10px;color:#4CAF50;letter-spacing:1px;font-weight:600'>"
        f"OUTPERFORMING</div>"
        f"<div style='margin-top:4px'><span style='font-size:1.15rem;font-weight:700;"
        f"color:#4CAF50'>{outperformers}</span>"
        f"<span style='font-size:0.85rem;color:#888'> / {len(df)} sectors</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    underperformers = len(df) - outperformers
    _c3.markdown(
        f"<div style='padding:10px 14px;background:rgba(255,255,255,0.03);"
        f"border-radius:8px;border-left:3px solid #888;font-size:12px;color:rgba(255,255,255,0.55)'>"
        f"RS = Sector Return − Nifty50 Return &nbsp;·&nbsp; "
        f"X-axis zero line = Nifty50 benchmark &nbsp;·&nbsp; "
        f"<span style='color:#d50000'>{underperformers} lagging</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Scatter ───────────────────────────────────────────────────────────────
    st.plotly_chart(
        _rs_delivery_scatter(df, period_label, rs_col, nifty_col, price_col),
        use_container_width=True,
    )
    _render_rs_legend(df, rs_col)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.plotly_chart(
        _rs_bar_chart(df, rs_col, period_label, nifty_val),
        use_container_width=True,
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    with st.expander("📋 RS Summary Table — All Sectors", expanded=False):
        rs_table_cols = ["sector", "signal", "accum_score",
                         f"price_{period}" if period != "custom" else "price_custom",
                         rs_col, "z_score", "dv_ratio"]
        rs_table_cols = [c for c in rs_table_cols if c in df.columns]
        disp = df[rs_table_cols].copy().sort_values(rs_col, ascending=False).reset_index(drop=True)
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "sector":      st.column_config.TextColumn("Sector"),
                "signal":      st.column_config.TextColumn("Signal"),
                "accum_score": st.column_config.NumberColumn("Score", format="%.1f"),
                rs_col:        st.column_config.NumberColumn(f"RS {period_label} %", format="%+.2f%%"),
                "z_score":     st.column_config.NumberColumn("Z-Score",  format="%+.2f"),
                "dv_ratio":    st.column_config.NumberColumn("DV Ratio", format="%.2f×"),
            },
        )


def _render_rs_custom_range(df_base: pd.DataFrame, all_dates: list, min_turnover: float) -> None:
    """Custom date RS tab — pick from/to dates and compare sectors vs Nifty50."""
    if not all_dates:
        st.warning("No trading dates available.")
        return

    avail_asc = sorted(all_dates)
    min_avail, max_avail = avail_asc[0], avail_asc[-1]
    default_from = all_dates[min(21, len(all_dates) - 1)] if len(all_dates) > 1 else min_avail

    c_from, c_to = st.columns(2)
    with c_from:
        from_date = st.date_input("From Date", value=default_from,
                                  min_value=min_avail, max_value=max_avail, key="rs_cr_from")
    with c_to:
        to_date = st.date_input("To Date", value=max_avail,
                                min_value=min_avail, max_value=max_avail, key="rs_cr_to")

    if from_date >= to_date:
        st.warning("From Date must be before To Date.")
        return

    from_snap = next((d for d in avail_asc if d >= from_date), None)
    to_snap   = next((d for d in reversed(avail_asc) if d <= to_date), None)
    if from_snap is None or to_snap is None or from_snap >= to_snap:
        st.warning("No trading data in selected range.")
        return

    n_cal = (to_snap - from_snap).days
    n_td  = sum(1 for d in all_dates if from_snap <= d <= to_snap)
    st.caption(
        f"**{from_snap.strftime('%d %b %Y')}** → **{to_snap.strftime('%d %b %Y')}**  "
        f"({n_cal} calendar days · {n_td} trading days)"
    )

    with st.spinner("Computing custom RS vs Nifty50…"):
        rs_df = cached_sector_rs_custom_range(from_snap, to_snap, float(min_turnover))

    if rs_df.empty:
        st.warning("No sector data for this range.")
        return

    # Merge custom RS into the base df (which has Z-scores / signals)
    plot_df = df_base.merge(
        rs_df[["sector", "rs_custom", "nifty_custom",
               "cum_price_ret_pct"]].rename(columns={"cum_price_ret_pct": "price_custom"}),
        on="sector",
        how="inner",
    )
    if plot_df.empty:
        st.warning("No matching sectors between current signals and custom range data.")
        return

    period_label = f"{from_snap.strftime('%d %b')} → {to_snap.strftime('%d %b %Y')}"
    _render_rs_charts(plot_df, "custom", period_label)


def _render_relative_strength(trade_date: date, min_turnover: float, all_dates: list | None = None) -> None:
    """Relative Strength of every sector vs Nifty50, with 1W / 2W / 1M / Custom periods."""
    df = cached_sector_rotation(trade_date, min_turnover)

    if df.empty:
        st.info("No sector rotation data for this date.")
        return

    has_rs = "rs_1w" in df.columns and df["rs_1w"].notna().any()
    if not has_rs:
        st.warning(
            "Nifty50 benchmark data not found. Run:\n"
            "```\npython -m src.cli backfill-indices 120\n```\n"
            "to populate index data, then refresh."
        )
        return

    st.caption(
        "**RS = Sector Return − Nifty50 Return.** Positive = sector outperforming the benchmark. "
        "Combine with delivery Z-score to find sectors where institutions are accumulating "
        "despite underperformance (Hidden Accumulation — best contrarian entry)."
    )

    # ── Period selector ───────────────────────────────────────────────────────
    period = st.radio(
        "Period",
        options=["1w", "2w", "1m", "custom"],
        format_func=lambda p: {
            "1w": "1 Week", "2w": "2 Week", "1m": "1 Month", "custom": "📅 Custom Date"
        }[p],
        horizontal=True,
        key="rs_period_sel",
        index=2,  # default: 1 Month
    )

    if period == "custom":
        _render_rs_custom_range(df, all_dates or [], min_turnover)
        return

    _render_rs_charts(df, period, _RS_PERIOD_META[period][3])


# ── Entry Point ───────────────────────────────────────────────────────────────

def render(selected_date: date, min_turnover: float, all_dates: list | None = None) -> None:
    st.subheader("🔄 Sector Rotation — Smart Money Tracker")

    tab_smart, tab_clock, tab_rs = st.tabs([
        "🎯 Smart Money (Daily Signal)",
        "📅 Rotation Clock",
        "📈 vs Nifty50",
    ])

    with tab_smart:
        _render_smart_money(selected_date, min_turnover)

    with tab_clock:
        _render_rotation_clock(selected_date, min_turnover, all_dates=all_dates)

    with tab_rs:
        _render_relative_strength(selected_date, min_turnover, all_dates=all_dates)
