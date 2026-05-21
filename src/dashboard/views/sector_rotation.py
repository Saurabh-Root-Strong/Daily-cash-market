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
    cached_sector_stocks_custom_range,
    cached_sector_stocks_rotation,
)
from src.dashboard.constants import NEGATIVE_COLOR, POSITIVE_COLOR, PLOT_BG, PAPER_BG, GRID_COLOR

def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    """Convert #rrggbb hex to rgba(r,g,b,alpha) for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
                             "dv_ratio", "z_score", "breadth", "horizon"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"<span style='color:{color}'>{signal}</span><br>"
                "─────────────────────<br>"
                "Score: <b>%{customdata[2]:.0f}</b>/100<br>"
                "Price 1W: <b>%{x:+.2f}%</b><br>"
                "Z-Score: <b>%{y:+.2f}σ</b><br>"
                "DV Ratio: %{customdata[3]:.2f}×  ·  Breadth: %{customdata[5]:.0%}<br>"
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

def _rotation_clock_chart(df: pd.DataFrame, period_name: str) -> go.Figure:
    """RRG-style bubble chart: X = price return, Y = delivery slope z-score."""
    if df.empty:
        return go.Figure()

    x_vals = df["cum_price_ret_pct"]
    y_vals = df["slope_z"]
    x_pad  = max((x_vals.max() - x_vals.min()) * 0.30, 1.0)
    y_pad  = max((y_vals.max() - y_vals.min()) * 0.30, 0.4)
    x0, x1 = x_vals.min() - x_pad, x_vals.max() + x_pad
    y0, y1 = y_vals.min() - y_pad, y_vals.max() + y_pad

    fig = go.Figure()

    # Quadrant shading — top-left=Improving, top-right=Leading, bottom-left=Lagging, bottom-right=Weakening
    fig.add_shape(type="rect", x0=x0, x1=0,  y0=0,  y1=y1, fillcolor="rgba(64,196,255,0.09)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=0,  y1=y1, fillcolor="rgba(0,200,83,0.09)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x0, x1=0,  y0=y0, y1=0,  fillcolor="rgba(213,0,0,0.09)",   line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=y0, y1=0,  fillcolor="rgba(255,109,0,0.07)", line_width=0, layer="below")

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

        customdata = list(zip(
            grp["sector"].values,
            grp["flow_signal"].values,
            dv_str,
            grp["slope_z"].round(2).values,
            chg_str,
            to_str,
            grp["avg_deliv_pct"].round(1).values,
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
                "Price Return: <b>%{x:+.2f}%</b><br>"
                "Delivery Slope Z: <b>%{y:+.2f}σ</b><br>"
                "Delivery Value: <b>%{customdata[2]}</b><br>"
                "Del Chg vs Prior: <b>%{customdata[4]}</b><br>"
                "Avg Delivery %: %{customdata[6]:.1f}%<br>"
                "Turnover: %{customdata[5]}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=f"Sector Rotation Clock — {period_name}",
            font=dict(size=16), x=0.5,
        ),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        xaxis=dict(
            title="← Price Falling  |  Cumulative Price Return (%)  |  Price Rising →",
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


def _sector_card(row: pd.Series, selected_date: date, min_turnover: float) -> None:
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
    z    = row.get("z_score")
    br   = row.get("breadth")
    p1w  = row.get("price_1w")
    dv1w = row.get("deliv_val_1w_cr")
    action_text = str(row.get("action", ""))
    coverage    = str(row.get("coverage") or "—")
    horizon     = str(row.get("horizon")  or "—")

    def _fmt(v, fmt):
        return fmt.format(v) if (v is not None and not (isinstance(v, float) and pd.isna(v))) else "—"

    dv_str   = _fmt(dv,  "{:.2f}×")
    z_str    = _fmt(z,   "{:+.2f}σ")
    p1w_str  = _fmt(p1w, "{:+.2f}%")
    br_str   = f"{br * 100:.0f}%" if (br is not None and not (isinstance(br, float) and pd.isna(br))) else "—"
    dv1w_str = f"₹{dv1w:,.0f} Cr" if (dv1w is not None and not (isinstance(dv1w, float) and pd.isna(dv1w))) else "—"

    z_color   = (POSITIVE_COLOR if (z is not None and not pd.isna(z) and z >= 1.0)
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
        f"<span>DV Ratio: <b>{dv_str}</b></span>"
        f"<span>Z-Score: <b style='color:{z_color}'>{z_str}</b></span>"
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

            if invest_signal:
                _rank = {"🔥 Strong": 0, "✅ Buying": 1, "👀 Watch": 2, "⚪ Weak": 3}
            else:
                _rank = {"❌ Exit Now": 0, "⚠️ Reducing": 1, "📉 Fading": 2, "⚪ Neutral": 3}

            stocks["_rank"] = stocks["conviction"].map(_rank).fillna(9)
            stocks = stocks.sort_values(
                ["_rank", "wtd_deliv_per", "deliv_value_cr"],
                ascending=[True, False, False],
            ).drop(columns="_rank")

            total_turnover = stocks["turnover_cr"].sum()
            dominant = stocks[stocks["turnover_cr"] / total_turnover > 0.30]
            if not dominant.empty:
                dom = dominant.iloc[0]
                dom_pct = dom["turnover_cr"] / total_turnover * 100
                dom_conv = stocks.loc[stocks["symbol"] == dom["symbol"], "conviction"].values[0]
                warn_color = "#ff9100" if invest_signal else "#d50000"
                st.markdown(
                    f"<div style='background:rgba(255,145,0,0.12);border-left:3px solid {warn_color};"
                    f"padding:6px 10px;border-radius:0 4px 4px 0;margin-bottom:6px;font-size:12px'>"
                    f"⚠️ <b>{dom['symbol']}</b> dominates <b>{dom_pct:.0f}%</b> of sector turnover "
                    f"with <b>{dom_conv}</b> conviction ({dom['wtd_deliv_per']:.1f}% delivery). "
                    f"The sector signal is driven by this one stock — verify independently."
                    f"</div>",
                    unsafe_allow_html=True,
                )

            display_cols = ["symbol", "company_name", "industry", "ltp", "conviction",
                            "wtd_deliv_per", "avg_deliv_per_100d",
                            "deliv_value_cr", "turnover_cr", "price_chg_pct"]
            display_cols = [c for c in display_cols if c in stocks.columns]

            st.dataframe(
                stocks[display_cols],
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

    # KPI pills
    pc = df["phase"].value_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Leading",   pc.get("Leading",   0))
    k2.metric("🔍 Improving", pc.get("Improving", 0))
    k3.metric("⚖️ Neutral",   pc.get("Neutral",   0))
    k4.metric("⚠️ Weakening", pc.get("Weakening", 0))
    k5.metric("📤 Lagging",   pc.get("Lagging",   0))

    # Bubble chart
    period_label = f"{from_snap.strftime('%d %b')} → {to_snap.strftime('%d %b %Y')}"
    st.plotly_chart(
        _rotation_clock_chart(df, period_label),
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

                st.dataframe(
                    stocks,
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

    signal_date = bt["signal_date"].iloc[0]
    n_fwd_days  = bt["forward_ret_pct"].dropna().shape[0]

    st.caption(
        f"Signals computed **as of {signal_date.strftime('%d %b %Y')}** "
        f"({window} trading days before {selected_date.strftime('%d %b %Y')}). "
        f"Forward returns measured from that date to today."
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

        avg_ret   = grp["forward_ret_pct"].mean()
        n_correct = int(grp["signal_correct"].fillna(False).sum())
        n_total   = len(grp)
        hit_rate  = n_correct / n_total * 100

        total_correct   += n_correct
        total_predicted += n_total

        # Color logic: inflow phases want positive avg_ret, outflow want negative
        expected_positive = phase in inflow_phases
        ret_ok = (avg_ret > 0) if expected_positive else (avg_ret < 0)

        cols[i].metric(
            label=f"{meta['label']} ({n_total})",
            value=f"{avg_ret:+.1f}% avg",
            delta=f"{hit_rate:.0f}% hit  {n_correct}/{n_total}",
            delta_color="normal" if ret_ok else "inverse",
            help=(
                f"{'Inflow' if expected_positive else 'Outflow'} signal.\n"
                f"Correct = forward return {'> 0%' if expected_positive else '< 0%'}."
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
    disp = bt[[
        "sector", "phase", "forward_ret_pct", "signal_correct",
        "cum_price_ret_pct", "slope_z", "deliv_chg_pct",
    ]].copy()

    disp["signal_correct"] = disp["signal_correct"].map(
        lambda v: "✅ Correct" if v is True else ("❌ Wrong" if v is False else "—")
    )

    st.dataframe(
        disp,
        hide_index=True,
        use_container_width=True,
        column_config={
            "sector":          st.column_config.TextColumn("Sector"),
            "phase":           st.column_config.TextColumn("Phase on Signal Date",
                help=f"Rotation phase as of {signal_date.strftime('%d %b %Y')}"),
            "forward_ret_pct": st.column_config.NumberColumn(
                "Actual Forward Return", format="%+.2f%%",
                help=f"Cumulative sector return from {signal_date.strftime('%d %b')} → {selected_date.strftime('%d %b')}"),
            "signal_correct":  st.column_config.TextColumn("Correct?",
                help="✅ = signal direction matched actual return\n"
                     "❌ = signal was wrong\n"
                     "— = Neutral (no directional prediction)"),
            "cum_price_ret_pct": st.column_config.NumberColumn(
                "Price Ret on Signal Date", format="%+.2f%%",
                help=f"Sector price return as of {signal_date.strftime('%d %b')} — what triggered the phase classification"),
            "slope_z":         st.column_config.NumberColumn(
                "Delivery Slope Z", format="%+.2f",
                help="Delivery momentum Z-score on the signal date"),
            "deliv_chg_pct":   st.column_config.NumberColumn(
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
**RRG-Inspired Framework — 4 Rotation Phases:**

| Phase | Delivery Slope | Price Return | Interpretation | Action |
|-------|---------------|--------------|----------------|--------|
| 💰 **Leading** | Rising ↑ | Rising ↑ | Institutions buying + price confirming | **BUY / HOLD** |
| 🔍 **Improving** | Rising ↑ | Falling ↓ | Institutions accumulating while retail exits | **ACCUMULATE** |
| ⚠️ **Weakening** | Falling ↓ | Rising ↑ | Institutions distributing into retail FOMO | **EXIT / REDUCE** |
| 📤 **Lagging** | Falling ↓ | Falling ↓ | Institutions exiting, price confirming | **AVOID** |

**Delivery Slope** = Linear regression of daily turnover-weighted delivery % over the period.
Positive slope = institutions are INCREASINGLY committed (building positions).
Negative slope = conviction is FADING (reducing exposure).

**Slope Z-Score** = Cross-sectional z-score across all sectors for the selected period.
Tells you which sectors are gaining or losing institutional interest *relative to each other*.

**Delivery Change %** = Current period delivery value vs the prior equal-length period.
Positive = more institutional money this period than the last one (INFLOW).
Negative = less institutional money (OUTFLOW).

**Bubble Size** = Total delivery value ₹ Cr — larger bubbles = more absolute institutional activity.

**Key insight:** Sectors typically rotate: Improving → Leading → Weakening → Lagging → Improving.
Catching a sector moving from Improving to Leading (rising delivery + turning positive price) is the ideal entry.
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
    pc = df["phase"].value_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Leading",   pc.get("Leading",   0), help="Delivery rising + price rising — institutions and price aligned")
    k2.metric("🔍 Improving", pc.get("Improving", 0), help="Delivery rising + price falling — contrarian accumulation zone")
    k3.metric("⚖️ Neutral",   pc.get("Neutral",   0), help="No clear directional bias in delivery momentum")
    k4.metric("⚠️ Weakening", pc.get("Weakening", 0), help="Delivery falling + price rising — institutions distributing into rally")
    k5.metric("📤 Lagging",   pc.get("Lagging",   0), help="Delivery falling + price falling — institutional exit confirmed")

    # Bubble chart
    st.plotly_chart(
        _rotation_clock_chart(df, sel),
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

**DV Ratio** — today's delivered value vs own 100D daily average. 1.5× = 50% above normal. Removes sector size bias.

**Z-Score (σ)** — how many standard deviations today's delivery VALUE is above its 100D mean. Z ≥ 2.0 = top 2.5% of trading days.

**Breadth** — fraction of stocks in the sector where today's delivery exceeds their own 100D average. 70%+ = broad institutional participation.

**Score (0–100):** 35% DV Ratio + 25% Breadth + 20% Z-Score + 10% 1W Price trend + 10% Trend slope. Cross-sectional — ranks sectors relative to each other on today's data.
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
                    "DV Ratio", "Z-Score", "Breadth", "Price 1W%", "Action"
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
                _sector_card(row, selected_date, min_turnover)

    with col_avoid:
        st.markdown("### 🔴 SECTORS TO AVOID / EXIT")
        st.caption("All distribution/selling signals — highest danger first (lowest score).")
        if exiting.empty:
            st.info("No sectors with distribution or selling signal today.")
        else:
            shown_divider = False
            for _, row in exiting.iterrows():
                if not shown_divider and row["accum_score"] > (100 - _HIGH_CONV):
                    st.markdown(
                        "<div style='margin:10px 0 6px 0;border-top:1px solid rgba(255,255,255,0.08);"
                        "padding-top:6px;font-size:11px;color:rgba(255,255,255,0.35);"
                        "letter-spacing:0.5px'>MILDER SIGNAL</div>",
                        unsafe_allow_html=True,
                    )
                    shown_divider = True
                _sector_card(row, selected_date, min_turnover)

    if not caution.empty:
        st.markdown("---")
        st.markdown("### 📊 VOLUME SPIKE — DO NOT CONFUSE WITH ACCUMULATION")
        st.caption(
            "Z-Score is high (delivery VALUE surged) BUT delivery % is BELOW its 100D average. "
            "Speculative event-driven trading — not institutional conviction. "
            "Do not buy based on delivery value alone."
        )
        for _, row in caution.iterrows():
            _sector_card(row, selected_date, min_turnover)

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

        dv_disp  = f"{row['dv_ratio']:.2f}×"  if pd.notna(row.get("dv_ratio"))  else "—"
        z_disp   = f"{row['z_score']:+.2f}σ"  if pd.notna(row.get("z_score"))   else "—"
        br_raw   = row.get("breadth")
        br_disp  = f"{br_raw*100:.0f}% breadth" if (br_raw is not None and pd.notna(br_raw)) else "—"

        st.markdown(
            f"<div style='padding:10px 16px;border-left:4px solid "
            f"{meta.get('color','#888')};background:rgba(255,255,255,0.04);"
            f"border-radius:0 8px 8px 0;margin:8px 0'>"
            f"<b style='font-size:16px'>{row['signal']}  —  {chosen}</b><br>"
            f"<span style='color:rgba(255,255,255,0.7)'>{row['action']}</span><br>"
            f"<span style='font-size:12px;color:rgba(255,255,255,0.5)'>"
            f"Score: {row['accum_score']:.0f}/100 &nbsp;|&nbsp; "
            f"DV Ratio: {dv_disp} &nbsp;|&nbsp; Z-Score: {z_disp} &nbsp;|&nbsp; {br_disp}"
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
            "Score = 35% DV Ratio + 25% Breadth + 20% Z-Score + 10% 1W Price + 10% Trend slope"
        )
        display_cols = ["sector", "signal", "accum_score", "coverage", "horizon",
                        "dv_ratio", "z_score", "breadth", "trend_slope",
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
                    help="Score = 35% DV Ratio + 25% Breadth + 20% Z-Score + 10% Price 1W + 10% Trend slope\n"
                         "Cross-sectional: ranks sectors relative to each other on today's data"),
                "coverage":       st.column_config.TextColumn(
                    "Coverage",
                    help="Swing (3–15 days): Z-Score ≥ 2σ + Breadth ≥ 50%\n"
                         "Positional (4–8 weeks): DV Ratio > 1.2 + positive slope + Breadth ≥ 40%\n"
                         "Mid Term (3–4 months): steep 100-day slope + DV Ratio > 1.3 + Breadth ≥ 50%"),
                "horizon":        st.column_config.TextColumn("Horizon"),
                "dv_ratio":       st.column_config.NumberColumn(
                    "DV Ratio", format="%.2f×",
                    help="Today's delivered value ÷ own 100D daily average\n"
                         "1.0× = exactly average  |  1.5× = 50% above norm\n"
                         "Removes sector size bias — Banking and Defence on equal footing"),
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

def _rs_bar_chart(df: pd.DataFrame, period: str) -> go.Figure:
    """
    Horizontal bar chart: sector RS vs Nifty50 for a selected period.
    Green = outperforming, red = underperforming. Zero line = Nifty50.
    """
    col = f"rs_{period}"
    if col not in df.columns:
        return go.Figure()

    plot = df[["sector", col, "signal"]].dropna(subset=[col]).copy()
    plot = plot.sort_values(col, ascending=True).reset_index(drop=True)

    colors = [
        _SIGNAL_META.get(sig, {}).get("color", "#888888")
        for sig in plot["signal"]
    ]
    bar_colors = [
        c if rs >= 0 else NEGATIVE_COLOR
        for c, rs in zip(colors, plot[col])
    ]

    fig = go.Figure(go.Bar(
        x=plot[col],
        y=plot["sector"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:+.2f}%" for v in plot[col]],
        textposition="outside",
        textfont=dict(size=10),
        customdata=plot[["signal", col]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "RS vs Nifty50: <b>%{x:+.2f}%</b><br>"
            "Signal: %{customdata[0]}"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color="rgba(255,255,255,0.5)", line_width=2)

    period_label = {"1w": "1 Week", "1m": "1 Month", "3m": "3 Month"}[period]
    fig.update_layout(
        title=dict(
            text=(
                f"Sector Relative Strength vs Nifty50 — {period_label}  "
                "<span style='font-size:11px;color:rgba(255,255,255,0.4)'>"
                "Positive = outperforming benchmark  ·  Negative = lagging benchmark</span>"
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
        height=max(420, len(plot) * 28 + 100),
        margin=dict(t=70, b=50, l=180, r=100),
    )
    return fig


def _rs_delivery_scatter(df: pd.DataFrame, period: str) -> go.Figure:
    """
    The definitive institutional lens:
      X = Relative Strength vs Nifty50 (price outperformance)
      Y = Delivery Z-Score (institutional conviction)

    4 quadrants:
      Top-Right  (RS > 0, Z > 1): Leading + inflow = strongest sectors
      Top-Left   (RS < 0, Z > 1): Lagging price but institutions accumulating = best entry
      Bottom-Right (RS > 0, Z < -0.5): Outperforming but institutions exiting = distribution
      Bottom-Left  (RS < 0, Z < -0.5): Lagging + outflow = avoid
    """
    col = f"rs_{period}"
    plot = df.dropna(subset=[col, "z_score"]).copy()
    if plot.empty:
        return go.Figure()

    x_vals = plot[col]
    y_vals = plot["z_score"]
    x_pad = max((x_vals.max() - x_vals.min()) * 0.28, 1.0)
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
            showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.70)"),
            xanchor=xanchor, yanchor=yanchor,
            bgcolor=bgcolor, borderpad=5,
            align="center",
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
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)

    for signal in [
        "🔥 Secret Accumulation", "✅ Confirmed Accumulation",
        "👀 Early Accumulation", "📊 Volume Spike", "⚖️ Neutral",
        "📉 Weakening", "⚠️ Distribution Trap", "❌ Active Selling",
    ]:
        grp = plot[plot["signal"] == signal]
        if grp.empty:
            continue
        color = _SIGNAL_META.get(signal, {}).get("color", "#888888")
        sizes = (grp["accum_score"] / 100 * 22 + 12).clip(12, 34)

        fig.add_trace(go.Scatter(
            x=grp[col],
            y=grp["z_score"],
            mode="markers+text",
            name=signal,
            text=grp["sector"],
            textposition="top center",
            textfont=dict(size=9, color="rgba(255,255,255,0.75)"),
            marker=dict(
                color=color, size=sizes, opacity=0.90,
                line=dict(width=1.5, color="rgba(255,255,255,0.5)"),
            ),
            customdata=grp[["sector", "signal", "accum_score",
                             "z_score", col, "price_1w",
                             f"nifty_{period}"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"<span style='color:{color}'>%{{customdata[1]}}</span><br>"
                "────────────────────<br>"
                f"RS vs Nifty50: <b>%{{customdata[4]:+.2f}}%</b><br>"
                "Z-Score: <b>%{customdata[3]:+.2f}σ</b><br>"
                "Score: %{customdata[2]:.0f}/100<br>"
                f"Sector 1W: %{{customdata[5]:+.2f}}%  ·  Nifty50: %{{customdata[6]:+.2f}}%"
                "<extra></extra>"
            ),
        ))

    period_label = {"1w": "1 Week", "1m": "1 Month", "3m": "3 Month"}[period]
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        title=dict(
            text=(
                f"RS vs Nifty50 ({period_label}) × Institutional Delivery Flow  "
                "<span style='font-size:11px;color:rgba(255,255,255,0.4)'>"
                "Top-Left = hidden accumulation (best entry)  ·  "
                "Top-Right = leading + confirmed (hold/add)  ·  "
                "Bottom-Right = distributing (exit)</span>"
            ),
            font=dict(size=13),
        ),
        xaxis=dict(
            title=f"← Lagging Nifty50  |  RS vs Nifty50 {period_label} (%)  |  Leading Nifty50 →",
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
            orientation="h", y=-0.15, x=0.5, xanchor="center",
            font=dict(size=11), bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        ),
        height=640,
        margin=dict(t=70, b=100, l=100, r=40),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=13,
                        bordercolor="rgba(255,255,255,0.2)"),
        hovermode="closest",
    )
    return fig


def _render_relative_strength(trade_date: date, min_turnover: float) -> None:
    """
    Relative Strength of every sector vs the Nifty50 benchmark.

    Why this matters:
      Absolute price return is meaningless without benchmark context.
      A sector up +2% when Nifty is +5% is a LAGGER, not a leader.
      RS = sector_return − nifty50_return tells you who is getting
      disproportionate institutional money flows relative to the market.
    """
    df = cached_sector_rotation(trade_date, min_turnover)

    if df.empty:
        st.info("No sector rotation data for this date.")
        return

    # Check if RS columns exist (requires index_data to be populated)
    has_rs = "rs_1w" in df.columns and df["rs_1w"].notna().any()
    if not has_rs:
        st.warning(
            "Nifty50 benchmark data not found. Run:\n"
            "```\npython -m src.cli backfill-indices 120\n```\n"
            "to populate index data, then refresh."
        )
        return

    # ── Benchmark KPI bar ─────────────────────────────────────────────────────
    n_1w = df["nifty_1w"].iloc[0] if "nifty_1w" in df.columns else None
    n_1m = df["nifty_1m"].iloc[0] if "nifty_1m" in df.columns else None
    n_3m = df["nifty_3m"].iloc[0] if "nifty_3m" in df.columns else None

    st.markdown("##### 📌 Nifty50 Benchmark Returns (RS = Sector − Benchmark)")
    c1, c2, c3, c4 = st.columns(4)
    def _fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        color = POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR
        return f"<span style='color:{color};font-size:1.1rem;font-weight:700'>{v:+.2f}%</span>"

    c1.markdown(
        f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,0.04);"
        f"border-radius:8px;border-top:2px solid #FFD700'>"
        f"<div style='font-size:10px;color:#888;letter-spacing:1px'>NIFTY50  1W</div>"
        f"<div style='margin-top:4px'>{_fmt(n_1w)}</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,0.04);"
        f"border-radius:8px;border-top:2px solid #FFD700'>"
        f"<div style='font-size:10px;color:#888;letter-spacing:1px'>NIFTY50  1M</div>"
        f"<div style='margin-top:4px'>{_fmt(n_1m)}</div></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,0.04);"
        f"border-radius:8px;border-top:2px solid #FFD700'>"
        f"<div style='font-size:10px;color:#888;letter-spacing:1px'>NIFTY50  3M</div>"
        f"<div style='margin-top:4px'>{_fmt(n_3m)}</div></div>",
        unsafe_allow_html=True,
    )
    outperformers_1m = int(df["rs_1m"].gt(0).sum()) if "rs_1m" in df.columns else 0
    c4.markdown(
        f"<div style='text-align:center;padding:10px;background:rgba(255,255,255,0.04);"
        f"border-radius:8px;border-top:2px solid #4CAF50'>"
        f"<div style='font-size:10px;color:#888;letter-spacing:1px'>OUTPERFORMERS (1M)</div>"
        f"<div style='margin-top:4px'><span style='color:#4CAF50;font-size:1.1rem;"
        f"font-weight:700'>{outperformers_1m} / {len(df)} sectors</span></div></div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Period selector + charts ──────────────────────────────────────────────
    period = st.radio(
        "Period",
        options=["1w", "1m", "3m"],
        format_func=lambda p: {"1w": "1 Week", "1m": "1 Month", "3m": "3 Month"}[p],
        horizontal=True,
        key="rs_period_sel",
        index=1,  # default: 1 Month
    )

    # RS × Delivery scatter — the primary chart
    st.plotly_chart(
        _rs_delivery_scatter(df, period),
        use_container_width=True,
    )

    # RS bar chart
    st.plotly_chart(
        _rs_bar_chart(df, period),
        use_container_width=True,
    )

    # ── RS Summary Table ──────────────────────────────────────────────────────
    st.markdown("#### Sector RS vs Nifty50 — Full Table")
    with st.expander("📋 Show RS Table", expanded=False):
        rs_cols = ["sector", "signal", "accum_score",
                   "price_1w", "rs_1w", "price_1m", "rs_1m",
                   "price_3m", "rs_3m", "z_score", "dv_ratio"]
        disp = df[[c for c in rs_cols if c in df.columns]].copy()
        disp = disp.sort_values("rs_1m", ascending=False).reset_index(drop=True)
        disp.columns = [
            c.replace("price_1w", "Sector 1W%")
             .replace("price_1m", "Sector 1M%")
             .replace("price_3m", "Sector 3M%")
             .replace("rs_1w",    "RS 1W%")
             .replace("rs_1m",    "RS 1M%")
             .replace("rs_3m",    "RS 3M%")
             .replace("accum_score", "Score")
             .replace("z_score",  "Z-Score")
             .replace("dv_ratio", "DV Ratio")
             .replace("sector",   "Sector")
             .replace("signal",   "Signal")
            for c in disp.columns
        ]
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score":      st.column_config.NumberColumn("Score",      format="%.1f"),
                "Sector 1W%": st.column_config.NumberColumn("Sector 1W%", format="%+.2f%%"),
                "Sector 1M%": st.column_config.NumberColumn("Sector 1M%", format="%+.2f%%"),
                "Sector 3M%": st.column_config.NumberColumn("Sector 3M%", format="%+.2f%%"),
                "RS 1W%":     st.column_config.NumberColumn("RS 1W%",     format="%+.2f%%"),
                "RS 1M%":     st.column_config.NumberColumn("RS 1M%",     format="%+.2f%%"),
                "RS 3M%":     st.column_config.NumberColumn("RS 3M%",     format="%+.2f%%"),
                "Z-Score":    st.column_config.NumberColumn("Z-Score",    format="%+.2f"),
                "DV Ratio":   st.column_config.NumberColumn("DV Ratio",   format="%.2f×"),
            },
        )


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
        _render_relative_strength(selected_date, min_turnover)
