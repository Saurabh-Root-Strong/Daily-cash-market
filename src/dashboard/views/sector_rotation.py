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

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.dashboard.column_help import nc as _hnc, tc as _htc, dc as _hdc, cc as _hcc, pc as _hpc, cfg as _hcfg

from src.dashboard.cache.queries import (
    cached_rotation_clock_backtest,
    cached_rotation_clock_accuracy,
    cached_sector_rotation,
    cached_sector_overlay,
    cached_market_scenario,
    cached_sector_rotation_history,
    cached_sector_rotation_custom_range,
    cached_sector_rotation_timeframe,
    cached_sector_rs_custom_range,
    cached_sector_stocks_custom_range,
    cached_sector_stocks_rotation,
    cached_fno_positioning_by_symbol,
    cached_fno_expiry_breakdown,
    cached_market_regime,
    cached_sector_memory_context,
    cached_price_action,
    cached_forward_tilt,
    cached_clock_replay,
    cached_operator_footprint,
    cached_next_month_context,
    cached_analogues,
    cached_clock_stock_detail,
    cached_tilt_replay,
    cached_sector_month_map,
    cached_month_suggestion,
    cached_seasonality_record,
    cached_sector_defensive,
    cached_market_breadth,
    cached_nifty_breakout,
    cached_mtf_trend,
)
from src.analytics.price_action import (BRK_BREAKOUT, BRK_FALSEBRK, BRK_EXTENDED,
                                        BRK_BOUNCE, BRK_COILING, BRK_NONE, BRK_STATES,
                                        MTF_ALIGN, MTF_CONFIRM_UP, MTF_FALSE_POP,
                                        MTF_PULLBACK, MTF_DOWN_ALGN, MTF_NEUTRAL)
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
    df: pd.DataFrame, period_name: str, nifty_return: float | None = None,
    center: float | None = None,
) -> go.Figure:
    """RRG-style bubble chart: X = price return, Y = delivery slope z-score.

    The quadrant center is the cross-sectional MEDIAN sector return (the typical
    sector), matching the phase classification — a sector is 'Leading' only if it
    beats its PEERS, not just Nifty50. Nifty is drawn as a faint reference line.
    """
    if df.empty:
        return go.Figure()

    x_vals = df["cum_price_ret_pct"]
    y_vals = df["slope_z"]
    x_pad  = max((x_vals.max() - x_vals.min()) * 0.30, 1.0)
    y_pad  = max((y_vals.max() - y_vals.min()) * 0.30, 0.4)
    x0, x1 = x_vals.min() - x_pad, x_vals.max() + x_pad
    y0, y1 = y_vals.min() - y_pad, y_vals.max() + y_pad

    # Quadrant center = cross-sectional MEDIAN sector return (matches the phase
    # classification); fall back to Nifty / 0% only if not supplied.
    cx = (center if center is not None
          else (nifty_return if nifty_return is not None else 0.0))

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
    # Primary vertical axis: the cross-sectional median sector (gold) = quadrant center.
    fig.add_vline(x=cx, line_color="#ffd600", line_width=2, line_dash="dash")
    fig.add_annotation(
        x=cx, y=y1, text=f"<b>Typical sector (median): {cx:+.2f}%</b>",
        showarrow=False, font=dict(size=11, color="#ffd600"),
        xanchor="center", yanchor="top", bgcolor="rgba(255,214,0,0.13)", borderpad=4, yshift=-4,
    )
    # Nifty50 as a faint secondary reference (where the cap-weighted index sits).
    if nifty_return is not None and abs(nifty_return - cx) > 0.05:
        fig.add_vline(x=nifty_return, line_color="rgba(255,255,255,0.30)", line_width=1, line_dash="dot")
        fig.add_annotation(
            x=nifty_return, y=y0, text=f"Nifty50 {nifty_return:+.1f}%",
            showarrow=False, font=dict(size=9.5, color="rgba(255,255,255,0.5)"),
            xanchor="center", yanchor="bottom",
        )
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


def _render_memory_context(mem) -> None:
    """
    Display SectorMemoryContext inside a Streamlit expander.
    Shows median forward returns + RS vs Nifty for 1W / 2W / 1M horizons.
    Silently no-ops when the memory engine has no data yet.
    """
    if getattr(mem, "error", None):
        return   # DB missing or sector not yet recorded — silent skip
    if getattr(mem, "n_filled", 0) == 0:
        note = getattr(mem, "note", "")
        if not note:
            return
        with st.expander("🧠 Memory: building...", expanded=False):
            st.caption(note)
        return

    n     = mem.n_filled
    label = f"🧠 Memory: {n} similar past setup{'s' if n != 1 else ''}"
    with st.expander(label, expanded=False):
        has_any = (
            mem.ret_1w_median is not None or
            mem.ret_2w_median is not None or
            mem.ret_1m_median is not None
        )
        if not has_any:
            st.caption(
                getattr(mem, "note", None)
                or "Outcomes not yet filled for similar setups — check back after 30+ trading days."
            )
            return

        def _ret_row(label_: str, median_, p25_, p75_, pos_pct_, rs_, rs_pos_) -> None:
            if median_ is None:
                return
            mc = POSITIVE_COLOR if median_ > 0 else NEGATIVE_COLOR
            rng = (f"  (p25 {p25_:+.1f}% → p75 {p75_:+.1f}%)"
                   if p25_ is not None else "")
            pos = (f"  ·  {pos_pct_:.0f}% positive" if pos_pct_ is not None else "")
            if rs_ is not None:
                rc = "#69f0ae" if rs_ > 0 else "#ff5252"
                rs_out_str = f" ({rs_pos_:.0f}% outperformed)" if rs_pos_ is not None else ""
                rs_html = (
                    f"  ·  RS vs Nifty: "
                    f"<b style='color:{rc}'>{rs_:+.1f}%</b>{rs_out_str}"
                )
            else:
                rs_html = ""
            st.markdown(
                f"<div style='font-size:12px;margin-bottom:3px'>"
                f"<b>{label_}</b>  median "
                f"<b style='color:{mc}'>{median_:+.2f}%</b>"
                f"{rng}{pos}{rs_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        _ret_row("1W →", mem.ret_1w_median, mem.ret_1w_p25, mem.ret_1w_p75,
                 mem.ret_1w_pos_pct, mem.rs_1w_median, mem.rs_1w_pos_pct)
        _ret_row("2W →", mem.ret_2w_median, mem.ret_2w_p25, mem.ret_2w_p75,
                 mem.ret_2w_pos_pct, mem.rs_2w_median, mem.rs_2w_pos_pct)
        _ret_row("1M →", mem.ret_1m_median, None, None,
                 mem.ret_1m_pos_pct, mem.rs_1m_median, None)

        if getattr(mem, "by_regime", {}):
            parts = "  ·  ".join(f"{k} {v}" for k, v in sorted(mem.by_regime.items()))
            st.markdown(
                f"<div style='font-size:11px;color:rgba(255,255,255,0.45);margin-top:4px'>"
                f"Regimes in sample: {parts}"
                f"</div>",
                unsafe_allow_html=True,
            )

        if getattr(mem, "note", ""):
            st.caption(mem.note)


def _sector_card(row: pd.Series, selected_date: date, min_turnover: float,
                 deliv_threshold: float = _MIN_STOCK_WTD_DELIV_PCT,
                 deliv_vs_100d_pct: float = 0.0,
                 fno_row: pd.Series | None = None,
                 min_price: float = 0.0,
                 max_price: float = 0.0,
                 regime_label: str = "SIDEWAYS",
                 regime: "dict | None" = None,
                 defense_mode: bool = False,
                 score_col: str = "accum_score",
                 fno_filter: "tuple | None" = None,
                 fno_match_all: bool = True,
                 fno_only: bool = False,
                 fno_oi_op: "str | None" = None,
                 fno_oi_threshold: float = 0.0,
                 price_action: "pd.DataFrame | None" = None,
                 pa_filter: "tuple | None" = None) -> None:
    meta       = _SIGNAL_META.get(row["signal"], {})
    color      = meta.get("color", "#888")
    is_avoid   = row["signal"] in _AVOID_SIGNAL_SET
    invest_signal = meta.get("invest", False)

    # In a BEAR/CAUTION regime the headline number is the DEFENSE score (capital
    # protection), not the momentum accum_score — and the card is coloured by the
    # defensive verdict, so a low-beta sector no longer looks like "Active Selling".
    _dscore  = row.get("defense_score")
    _verdict = str(row.get("defensive_verdict", "") or "")
    _has_def = defense_mode and _dscore is not None and not (isinstance(_dscore, float) and pd.isna(_dscore))
    if _has_def:
        score = float(_dscore)
        _elig = bool(row.get("bear_eligible", False))
        color = "#00c853" if (_elig and _verdict.startswith("🛡️ Defensive Leader")) else \
                "#69f0ae" if _elig else \
                "#ff9100" if _verdict.startswith("⚠️") else "#d50000"
    elif (score_col and score_col not in ("accum_score",)
          and score_col in row.index and pd.notna(row.get(score_col))):
        score = float(row[score_col])   # accumulation / reversal / adj score
    else:
        score = row["accum_score"]

    # ── Regime override badge ──────────────────────────────────────────────────
    # When market regime conflicts with or qualifies the sector signal, show a
    # clearly visible badge so the trader cannot misread "BUY" as an outright buy.
    # This is the most important piece for correct equity decision-making.
    _REGIME_INVEST_OVERRIDE = {
        "BEAR": (
            "#ff9100", "rgba(255,109,0,0.13)", "rgba(255,109,0,0.35)",
            "⚠ BEAR MARKET — This sector falls LESS than the index, not UP. "
            "Absolute return likely negative. Use only as: (a) pairs vs Nifty short, "
            "(b) defensive core holding, or (c) staged entry anticipating regime flip. "
            "Do NOT deploy fresh capital expecting positive returns."
        ),
        "CAUTION": (
            "#ffd600", "rgba(255,214,0,0.10)", "rgba(255,214,0,0.30)",
            "⚠ MARKET DETERIORATING — Scale in at half-size with a hard stop. "
            "Market regime is weakening; even strong accumulation can turn negative "
            "if the broader market confirms a break lower."
        ),
    }
    _REGIME_AVOID_OVERRIDE = {
        "BEAR": (
            "#d50000", "rgba(213,0,0,0.15)", "rgba(213,0,0,0.40)",
            "🔴 DOUBLE RISK — Bear market regime + sector distribution = maximum risk. "
            "Exit any existing positions immediately. Do not average down. "
            "No floor visible until regime improves."
        ),
        "CAUTION": (
            "#ff6d00", "rgba(255,109,0,0.12)", "rgba(255,109,0,0.35)",
            "🟠 EXIT EARLY — Market is deteriorating and this sector is already weak. "
            "Reduce or exit before both regime and sector confirm breakdown together."
        ),
    }

    # ── Relative Strength vs Nifty — the quantitative answer to "will I lose money?" ──
    # rs_1w = sector_1W_return − nifty_1W_return (already in row from analytics).
    # In a BEAR regime this is the ONE number that tells you whether delivery
    # signals are translating into real outperformance vs the index.
    # Positive RS in a BEAR = sector genuinely falling LESS or rising while market falls.
    # Negative RS in a BEAR = sector is LAGGING even on a relative basis — worst case.
    _rs_1w = row.get("rs_1w")
    _p1w   = row.get("price_1w")
    _n1w   = row.get("nifty_1w")
    if _rs_1w is not None and not (isinstance(_rs_1w, float) and pd.isna(_rs_1w)):
        _rs_color = "#69f0ae" if _rs_1w > 0 else "#ff5252"
        _rs_sign  = "+" if _rs_1w > 0 else ""
        _rs_label = (
            f"RS vs Nifty (1W): <b style='color:{_rs_color}'>{_rs_sign}{_rs_1w:.1f}%</b>"
            + (f" — Sector {_p1w:+.1f}% vs Nifty {_n1w:+.1f}%"
               if _p1w is not None and _n1w is not None
                  and not pd.isna(_p1w) and not pd.isna(_n1w)
               else "")
        )
    else:
        _rs_label = ""

    regime_badge_html = ""
    if invest_signal and regime_label in _REGIME_INVEST_OVERRIDE:
        rc, rbg, rbord, rtxt = _REGIME_INVEST_OVERRIDE[regime_label]
        regime_badge_html = (
            f"<div style='background:{rbg};border:1px solid {rbord};"
            f"border-radius:4px;padding:5px 9px;margin-bottom:5px;"
            f"font-size:11px;color:{rc};line-height:1.45'>{rtxt}"
            + (f"<br><span style='color:rgba(255,255,255,0.65);font-size:11px'>{_rs_label}</span>" if _rs_label else "")
            + "</div>"
        )
    elif is_avoid and regime_label in _REGIME_AVOID_OVERRIDE:
        rc, rbg, rbord, rtxt = _REGIME_AVOID_OVERRIDE[regime_label]
        regime_badge_html = (
            f"<div style='background:{rbg};border:1px solid {rbord};"
            f"border-radius:4px;padding:5px 9px;margin-bottom:5px;"
            f"font-size:11px;color:{rc};line-height:1.45'>{rtxt}"
            + (f"<br><span style='color:rgba(255,255,255,0.65);font-size:11px'>{_rs_label}</span>" if _rs_label else "")
            + "</div>"
        )

    # NOTE: the sector-level F&O BADGE was removed after Phase-3 IC validation —
    # the sector F&O aggregate showed no measurable edge for sector SELECTION
    # (RS/delivery dominate; aggregate too lumpy with 1-2 F&O stocks/sector). The
    # F&O read lives ONLY at the per-stock level (Futures/Options columns in the
    # drill-down), which is the intended use: confirm a stock AFTER the sector is
    # picked on delivery/RS. fno_row is accepted for signature stability but unused.
    _ = fno_row

    bar_html = (
        f"<div style='background:rgba(255,255,255,0.1);border-radius:4px;height:6px;margin:4px 0 6px 0'>"
        f"<div style='width:{score}%;background:{color};height:6px;border-radius:4px'></div></div>"
    )

    dv   = row.get("dv_ratio")
    dv5d = row.get("dv_ratio_5d")
    z    = row.get("z_score")
    # Prefer the multi-day accumulation breadth (fraction of constituents in
    # genuine multi-day accumulation — the metric the verdict is now gated on).
    # Falls back to the legacy 1-day value breadth for sectors without robust data.
    _ba  = row.get("breadth_accum")
    br   = _ba if (_ba is not None and not (isinstance(_ba, float) and pd.isna(_ba))) else row.get("breadth")
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

    # ── Basket-quality disclosure: how many liquid names + single-stock concentration ──
    # Makes it visible whether a sector signal is a diversified basket or one name.
    _nliq = row.get("n_liquid_stocks")
    _topp = row.get("top_stock_pct")
    _tcr  = row.get("sector_turnover_cr")
    basket_html = ""
    if _nliq is not None and not (isinstance(_nliq, float) and pd.isna(_nliq)):
        _conc_hot = _topp is not None and not (isinstance(_topp, float) and pd.isna(_topp)) and _topp > 40
        _conc_col = "#ff9100" if _conc_hot else "rgba(255,255,255,0.45)"
        basket_html = (
            f"<div style='margin-top:3px;font-size:10.5px;color:rgba(255,255,255,0.4)'>"
            f"Basket: {int(_nliq)} liquid name{'s' if int(_nliq) != 1 else ''}"
            + (f" · top stock <b style='color:{_conc_col}'>{int(_topp)}%</b> of turnover"
               if _topp is not None and not (isinstance(_topp, float) and pd.isna(_topp)) else "")
            + (f" · ₹{int(_tcr):,} Cr"
               if _tcr is not None and not (isinstance(_tcr, float) and pd.isna(_tcr)) else "")
            + "</div>"
        )

    # ── Defensive metrics (β / downside-capture) — the bear-market quality line ─
    _beta = row.get("beta"); _dcap = row.get("down_capture")
    defense_html = ""
    if _beta is not None and not (isinstance(_beta, float) and pd.isna(_beta)):
        _b_col = "#69f0ae" if _beta < 0.9 else ("#ff9100" if _beta > 1.2 else "rgba(255,255,255,0.55)")
        _d_txt = ""
        if _dcap is not None and not (isinstance(_dcap, float) and pd.isna(_dcap)):
            _d_col = "#69f0ae" if _dcap < 0.85 else ("#ff5252" if _dcap > 1.05 else "rgba(255,255,255,0.55)")
            _d_lbl = ("falls less than market" if _dcap < 0.85
                      else "amplifies market falls" if _dcap > 1.05 else "tracks market")
            _d_txt = (f" · downside-capture <b style='color:{_d_col}'>{_dcap:.2f}×</b> "
                      f"<span style='color:rgba(255,255,255,0.4)'>({_d_lbl})</span>")
        defense_html = (
            f"<div style='margin-top:2px;font-size:10.5px;color:rgba(255,255,255,0.45)'>"
            f"β <b style='color:{_b_col}'>{_beta:.2f}</b>{_d_txt}</div>"
        )
    basket_html = basket_html + defense_html

    if "—" in action_text:
        action_prefix, action_desc = action_text.split("—", 1)
        action_html = (
            f"<span style='color:{color};font-weight:700'>{action_prefix.strip()}</span>"
            f"<span style='color:rgba(255,255,255,0.55)'> — {action_desc.strip()}</span>"
        )
    else:
        action_html = f"<span style='color:{color}'>{action_text}</span>"

    # ── Memory conviction badge ────────────────────────────────────────────────
    # Shown only when the memory overlay has an ACTIONABLE opinion, so most cards
    # stay clean and the eye is drawn to setups history strongly confirms — or,
    # critically, contradicts (a footprint that looks strong today but faded in
    # similar past regimes). Neutral / Unproven setups show nothing.
    conviction_badge_html = ""
    _conv = row.get("conviction")
    if _conv in ("HIGH", "CONFIRM", "DISAGREE"):
        _basis = row.get("mem_basis", "")
        _CONV_STYLE = {
            "HIGH":     ("#00c853", "rgba(0,200,83,0.14)",   "rgba(0,200,83,0.40)",   "🔥 High Conviction — history strongly rewards this setup"),
            "CONFIRM":  ("#69f0ae", "rgba(0,200,83,0.08)",   "rgba(0,200,83,0.28)",   "✅ Memory Confirms"),
            "DISAGREE": ("#ff6d00", "rgba(255,109,0,0.14)",  "rgba(255,109,0,0.40)",  "⚠️ History Disagrees — similar past setups underperformed"),
        }
        _cc, _cbg, _cbord, _clabel = _CONV_STYLE[_conv]
        _adj = row.get("adj_score")
        _adj_txt = (f" &nbsp;·&nbsp; adj {float(_adj):.0f}/100"
                    if _adj is not None and not (isinstance(_adj, float) and pd.isna(_adj)) else "")
        conviction_badge_html = (
            f"<div style='background:{_cbg};border:1px solid {_cbord};border-radius:4px;"
            f"padding:4px 8px;margin-bottom:5px;font-size:10.5px;color:{_cc};line-height:1.4'>"
            f"<b>{_clabel}</b>{_adj_txt}"
            + (f"<br><span style='color:rgba(255,255,255,0.55)'>{_basis}</span>" if _basis else "")
            + "</div>"
        )

    # ── Defensive verdict badge (BEAR/CAUTION) — replaces the momentum framing ─
    defense_badge_html = ""
    if _has_def and _verdict:
        _vcol = ("#00c853" if _verdict.startswith("🛡️ Defensive Leader")
                 else "#69f0ae" if _verdict.startswith("🛡️")
                 else "#ff9100" if _verdict.startswith("⚠️") else "#ff5252")
        _vbg = ("rgba(0,200,83,0.12)" if _verdict.startswith("🛡️")
                else "rgba(255,145,0,0.12)" if _verdict.startswith("⚠️")
                else "rgba(213,0,0,0.12)")
        _b = row.get("beta"); _dc = row.get("down_capture")
        _det = ""
        if (_b is not None and not pd.isna(_b) and _dc is not None and not pd.isna(_dc)):
            _det = (f"<br><span style='color:rgba(255,255,255,0.55)'>β {float(_b):.2f} · "
                    f"falls {float(_dc):.2f}× the market on down days</span>")
        defense_badge_html = (
            f"<div style='background:{_vbg};border:1px solid {_vcol};border-radius:4px;"
            f"padding:4px 8px;margin-bottom:5px;font-size:10.5px;color:{_vcol};line-height:1.4'>"
            f"<b>{_verdict}</b>{_det}</div>"
        )
        regime_badge_html = ""   # suppress momentum DOUBLE-RISK framing in defense mode

    _score_tag = "🛡️ " if _has_def else ""
    st.markdown(
        f"<div style='border-left:3px solid {color};padding:8px 12px;margin:4px 0;"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<b style='font-size:14px'>{row['sector']}</b>"
        f"<span style='font-size:11px;color:{color};font-weight:600'>{_score_tag}{score:.0f}/100</span></div>"
        f"{bar_html}"
        f"{defense_badge_html}"
        f"{regime_badge_html}"
        f"{conviction_badge_html}"
        f"<div style='font-size:11px;margin-bottom:4px'>{row['signal']} &nbsp; {action_html}</div>"
        f"<div style='display:flex;flex-wrap:wrap;gap:8px 16px;margin-top:4px;font-size:12px'>"
        f"<span>DV Today: <b>{dv_str}</b></span>"
        f"<span>5D Avg: <b style='color:{dv5d_color}'>{dv5d_str}</b></span>"
        f"<span>Z-Rank: <b style='color:{z_color}'>{z_str}</b></span>"
        f"<span>Accum Breadth: <b>{br_str}</b></span>"
        f"<span>1W Price: <b style='color:{p1w_color}'>{p1w_str}</b></span>"
        f"</div>"
        f"<div style='margin-top:4px;font-size:11px;color:rgba(255,255,255,0.5)'>"
        f"{bottom_row}</div>"
        f"{basket_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    with st.expander(f"📋  View stocks in {row['sector']}", expanded=False):
        stocks = cached_sector_stocks_rotation(row["sector"], selected_date, min_turnover)
        if stocks.empty:
            st.caption("No stock data for this period.")
        else:
            # ── Price-action character overlay (merge by symbol) ──────────────
            if (price_action is not None and not price_action.empty
                    and "symbol" in price_action.columns):
                _pa_keep = [c for c in ["symbol", "pa_class", "pa_gappy", "pa_high_vol",
                                        "efficiency_ratio", "avg_body_pct", "long_wick_freq",
                                        "gap_freq", "atr_pct",
                                        "mtf_align", "breakout", "w_trend", "range_pos"]
                            if c in price_action.columns]
                stocks = stocks.merge(price_action[_pa_keep], on="symbol", how="left")

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

            # % excess of recent 7D delivery vs own 100D baseline
            # +15 = recent delivery 15% above own historical norm; −8 = below norm
            if "avg_deliv_per_100d" in stocks.columns:
                stocks["deliv_vs_100d_pct"] = stocks.apply(
                    lambda r: (r["wtd_deliv_per"] / r["avg_deliv_per_100d"] - 1) * 100
                    if pd.notna(r.get("avg_deliv_per_100d")) and r.get("avg_deliv_per_100d", 0) > 0
                    else float("nan"),
                    axis=1,
                )

            # ── F&O overlay: per-expiry futures OI + options PCR ────────────
            # Label cols carry the display string (e.g. "🟢 LB +39%"); the *_oi_chg_pct
            # cols carry the raw signed OI-change % that feeds the strength gate below.
            _FNO_EXP_COLS = ["near_fut_label", "next_fut_label", "far_fut_label",
                             "near_opt_label", "next_opt_label", "far_opt_label",
                             "near_trend_label", "next_trend_label", "far_trend_label",
                             "near_opt_trend_label", "next_opt_trend_label",
                             "far_opt_trend_label"]
            _FNO_NUM_COLS = ["near_oi_chg_pct", "next_oi_chg_pct", "far_oi_chg_pct"]
            _fno_symbols: set[str] = set()   # F&O underlying universe for this date
            try:
                _fno_exp = cached_fno_expiry_breakdown(selected_date)
                if not _fno_exp.empty:
                    _fno_symbols = set(_fno_exp["symbol"])
                    exp_cols = [c for c in ["symbol"] + _FNO_EXP_COLS + _FNO_NUM_COLS
                            if c in _fno_exp.columns]
                    stocks = stocks.merge(_fno_exp[exp_cols], on="symbol", how="left")
                else:
                    for c in _FNO_EXP_COLS + _FNO_NUM_COLS:
                        stocks[c] = None
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "F&O expiry breakdown failed for %s: %s: %s",
                    selected_date, type(_e).__name__, _e,
                )
                for c in _FNO_EXP_COLS + _FNO_NUM_COLS:
                    stocks[c] = None

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
                            "pa_class", "breakout", "mtf_align", "efficiency_ratio",
                            "near_fut_label", "near_trend_label",
                            "next_fut_label", "next_trend_label",
                            "far_fut_label", "far_trend_label",
                            "near_opt_label", "near_opt_trend_label",
                            "next_opt_label", "next_opt_trend_label",
                            "far_opt_label", "far_opt_trend_label",
                            "wtd_deliv_per", "deliv_vs_100d_pct", "avg_deliv_per_100d",
                            "deliv_value_cr", "turnover_cr", "price_chg_pct"]
            display_cols = [c for c in display_cols if c in stocks.columns]

            # Show only stocks with 7D turnover-weighted delivery % above the absolute floor.
            # Conviction + dominance/top-3 context above stay on the full sector set.
            shown = stocks[stocks["wtd_deliv_per"] > deliv_threshold]

            # Optional second filter: keep only stocks whose 7D wtd delivery exceeds
            # their own 100D average by at least deliv_vs_100d_pct %.
            # Formula: wtd_deliv_per >= avg_deliv_per_100d * (1 + deliv_vs_100d_pct / 100)
            # Stocks with no 100D history (NaN avg) are excluded when filter > 0 —
            # they cannot be verified against their own norm.
            if deliv_vs_100d_pct > 0 and "avg_deliv_per_100d" in shown.columns:
                multiplier = 1 + deliv_vs_100d_pct / 100
                shown = shown[
                    shown.apply(
                        lambda r: (
                            pd.notna(r["avg_deliv_per_100d"])
                            and r["avg_deliv_per_100d"] > 0
                            and r["wtd_deliv_per"] >= r["avg_deliv_per_100d"] * multiplier
                        ),
                        axis=1,
                    )
                ]

            # ── Price filter ─────────────────────────────────────────────────
            # ltp=0 / NaN means no price data; those are only excluded when the
            # min-price filter is active (natural — zero price fails >= threshold).
            # For max-price we guard explicitly so a zero-price stock is not
            # shown as "within range" when a max filter is active.
            if min_price > 0 and "ltp" in shown.columns:
                shown = shown[shown["ltp"].fillna(0) >= min_price]
            if max_price > 0 and "ltp" in shown.columns:
                shown = shown[
                    shown["ltp"].notna() &
                    (shown["ltp"] > 0) &
                    (shown["ltp"] <= max_price)
                ]

            # ── Price-action filter ──────────────────────────────────────────
            # pa_filter = (selected_setups, selected_aligns, hide_risky, efficient).
            # Only backtest-validated axes filter (audits: audit_rotation_filters.py
            # + _v2.py, dual-panel walk-forward): setup + alignment (edges),
            # efficient = ER>=0.25 trend-efficiency gate (the one piece of the old
            # pa_class filter that survived — robust both panels/halves/all regimes,
            # per-date L/S spread t 3.2 DCM / 5.1 4yr-OOS at 20d), and hide_risky =
            # single veto dropping gappy OR high-vol names (flags overlap, V 0.48;
            # valid risk vetoes, not return selectors). Candle-anatomy part of
            # pa_class stays context-only (a column) — zero selection value.
            if pa_filter and "pa_class" in shown.columns:
                _pa_setup, _pa_align, _pa_risky, _pa_eff = pa_filter
                if _pa_setup and "breakout" in shown.columns:
                    shown = shown[shown["breakout"].isin(_pa_setup)]
                if _pa_align and "mtf_align" in shown.columns:
                    shown = shown[shown["mtf_align"].isin(_pa_align)]
                if _pa_eff and "efficiency_ratio" in shown.columns:
                    shown = shown[pd.to_numeric(shown["efficiency_ratio"],
                                                errors="coerce") >= 0.25]
                if _pa_risky:
                    _risk = pd.Series(False, index=shown.index)
                    if "pa_gappy" in shown.columns:
                        _risk |= shown["pa_gappy"].fillna(False).astype(bool)
                    if "pa_high_vol" in shown.columns:
                        _risk |= shown["pa_high_vol"].fillna(False).astype(bool)
                    shown = shown[~_risk]

            # ── F&O universe filter — keep only NSE F&O underlyings ─────────────
            # _fno_symbols is the set of stocks with futures/options for this date.
            # When "F&O stocks only" is selected this drops every cash-only name.
            _fno_only_active = False
            if fno_only and "symbol" in shown.columns:
                shown = shown[shown["symbol"].isin(_fno_symbols)]
                _fno_only_active = True

            # ── F&O positioning filter — PER-EXPIRY (OR within; Any/All across) ──
            # fno_filter = {expiry: [(instrument, [tokens], short), ...]}. Within an
            # expiry ANY selected signal matches (OR). Across expiries the combine is
            # controlled by fno_match_all: True = AND (must hold in every expiry),
            # False = OR (hold in at least one). e.g. {Near:[LB], Next:[SB]} →
            # match_all=True: Near=LB AND Next=SB; match_all=False: Near=LB OR Next=SB.
            # OI-strength gate: |OI Δ%| ≥/≤ threshold on the FUTURES leg only. The
            # number in a futures label is the day-over-day OI change % (same
            # contract); we compare its MAGNITUDE so Short Covering / Long
            # Unwinding (OI falling, value negative) are gated on absolute size.
            _oi_gate_on = bool(fno_oi_op) and fno_oi_threshold > 0
            _fno_part = None
            if fno_filter and not shown.empty:
                import functools as _ft, operator as _op
                _ecode = {"Near month": "near", "Next month": "next", "Far month": "far"}
                _exp_masks, _tags = [], []
                for _exp, _sigs in fno_filter.items():
                    _e = _ecode.get(_exp, "near")
                    _sig_masks = []
                    for _instr, _toks, _short in _sigs:
                        _col = f"{_e}_{'fut' if _instr == 'Futures' else 'opt'}_label"
                        if _col in shown.columns:
                            # .astype(bool): apply() on an empty arrow-str column yields
                            # an empty STR Series; OR/AND on str raises — force boolean.
                            _m = shown[_col].fillna("").apply(
                                lambda s, ts=_toks: any(t in str(s) for t in ts)).astype(bool)
                            # Strength gate only constrains FUTURES signals (the OI%
                            # number lives on the futures label). Options buckets are
                            # categorical and pass through unchanged.
                            if _instr == "Futures" and _oi_gate_on:
                                _numcol = f"{_e}_oi_chg_pct"
                                if _numcol in shown.columns:
                                    _mag = pd.to_numeric(shown[_numcol], errors="coerce").abs()
                                    _g = (_mag >= fno_oi_threshold if fno_oi_op == "ge"
                                          else _mag <= fno_oi_threshold)
                                    _m = _m & _g.fillna(False)   # NaN OI% never passes the gate
                            _sig_masks.append(_m)
                    if _sig_masks:
                        _exp_masks.append(_ft.reduce(_op.or_, _sig_masks))   # OR within expiry
                        _tags.append(f"{_exp.split()[0]}={'/'.join(sorted({sh for _, _, sh in _sigs}))}")
                if _exp_masks:
                    _join = _op.and_ if fno_match_all else _op.or_
                    shown = shown[_ft.reduce(_join, _exp_masks)]             # Any/All across expiries
                    _sep = " & " if fno_match_all else " | "
                    _oi_tag = (f" (OI {'≥' if fno_oi_op == 'ge' else '≤'} {fno_oi_threshold:.0f}%)"
                               if _oi_gate_on else "")
                    _fno_part = "F&O " + _sep.join(_tags) + _oi_tag

            n_hidden = len(stocks) - len(shown)
            _pa_active = pa_filter and any(pa_filter)
            if n_hidden or _fno_part or _fno_only_active or _pa_active:
                filter_parts = [f"Wtd Deliv % > {deliv_threshold:.0f}%"]
                if _pa_active:
                    _pp = []
                    if pa_filter[0]:
                        _pp.append(" / ".join(pa_filter[0]))
                    if pa_filter[1]:
                        _pp.append(" / ".join(pa_filter[1]))
                    if pa_filter[2]:
                        _pp.append("no gappy / high-vol")
                    if pa_filter[3]:
                        _pp.append("efficient movers (ER≥0.25)")
                    filter_parts.append("Price action: " + ", ".join(_pp))
                if _fno_only_active:
                    filter_parts.append("F&O stocks only")
                if deliv_vs_100d_pct > 0:
                    filter_parts.append(f"7D ≥ {deliv_vs_100d_pct:.0f}%+ above own 100D avg")
                if min_price > 0 and max_price > 0:
                    filter_parts.append(f"Price ₹{min_price:,.0f}–₹{max_price:,.0f}")
                elif min_price > 0:
                    filter_parts.append(f"Price ≥ ₹{min_price:,.0f}")
                elif max_price > 0:
                    filter_parts.append(f"Price ≤ ₹{max_price:,.0f}")
                if _fno_part:
                    filter_parts.append(_fno_part)
                st.caption(
                    f"Showing {len(shown)} of {len(stocks)} stocks — "
                    f"{' AND '.join(filter_parts)} "
                    f"({n_hidden} hidden)."
                )

            st.dataframe(
                shown[display_cols],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "symbol":        _htc("Symbol", width="small",
                        help="NSE ticker symbol"),
                    "company_name":  _htc("Company",
                        help="Company full name from NSE sector master"),
                    "industry":      _htc("Sub-Sector",
                        help="Industry classification within the sector"),
                    "ltp":           _hnc(
                        "LTP (₹)", format="₹%.2f",
                        help="Last Traded Price — most recent close_price in the 7-day window.\n\n"
                             "Use the Min / Max Price filters above to narrow the list "
                             "by affordable price range or to exclude penny stocks."),
                    "conviction":    _htc("Conviction",
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
                    "pa_class":      _htc(
                        "Price Action", width="small",
                        help="60-day price-action character — trend efficiency (net move ÷ path "
                             "length) × bar conviction (body vs wick):\n"
                             "📈 Clean Trend   = high efficiency, decisive bodies, contained vol\n"
                             "🌊 Volatile Trend = trends but wide / whippy path (wider stops)\n"
                             "🔀 Choppy / Whipsaw = big bars, no net progress (stop-hunt RISK)\n"
                             "😴 Quiet Range   = small bars, coiling (await a breakout)\n\n"
                             "Filter by character with the 'Price Action character' control above."),
                    "breakout":      _htc(
                        "Setup", width="small",
                        help="Daily × Weekly breakout state (validated on 371 days):\n"
                             "🚀 Breakout = tight-base 20d-high break WITH weekly-up — "
                             "the one validated edge (+1.9%/10d rel, t=3.2, holds both halves)\n"
                             "⚠️ False break = same break AGAINST a weekly downtrend — "
                             "textbook false breakout, near-zero to negative (AVOID)\n"
                             "↗ Break (extended) = broke out with no prior coil (weaker)\n"
                             "💥 Breakdown-bounce = broke the 20d low, but such names BOUNCE "
                             "here — a reversal watch, NOT a short\n"
                             "🧊 Coiling = tight base, no break yet (breakout watchlist)\n\n"
                             "Filter with the '🚀 Multi-timeframe setup' control above."),
                    "mtf_align":     _htc(
                        "D×W", width="small",
                        help="Daily direction × Weekly (EMA20/50) trend — a momentum-quality "
                             "gate. A daily up-move continues mainly when the weekly agrees "
                             "(52.8% vs 46.6% win when it doesn't):\n"
                             "✅ Confirmed Up = daily-up + weekly-up (aligned)\n"
                             "⚠️ False Pop = daily-up but weekly-DOWN → the pop tends to fade\n"
                             "🔵 Pullback = daily-down inside a weekly uptrend (dip)\n"
                             "🔻 Down-trend = both lower (aligned down)"),
                    "efficiency_ratio": _hnc(
                        "Trend Eff", format="%.2f",
                        help="Kaufman Efficiency Ratio over 60 days = |net price change| ÷ sum of "
                             "|daily changes|. 1.0 = perfectly clean one-way trend; ~0 = pure "
                             "chop / range. The single best trend-vs-chop discriminator "
                             "(candle body/wick shape is near-uncorrelated with it)."),
                    # ── Per-expiry Futures columns (Near / Next / Far) ─────────
                    # Format: "🟢 LB +38%"  /  "🔴 SB -12%"  /  "⚪ +1%"  /  "⟳ rolling"
                    # On expiry day: Near shows "⟳ rolling" (rollover noise),
                    # Next shows the REAL fresh positioning — that's the signal to read.
                    "near_fut_label": _htc(
                        "Fut Near",
                        help="FUTURES — Near-month (current expiry) OI signal:\n"
                             "🟢 LB = Long Buildup (OI↑ + price↑ — fresh longs)\n"
                             "🔴 SB = Short Buildup (OI↑ + price↓ — fresh shorts)\n"
                             "🔵 SC = Short Covering (OI↓ + price↑)\n"
                             "🟠 LU = Long Unwinding (OI↓ + price↓)\n"
                             "⚪ = Neutral (small move: |price|<0.1% or |OI Δ|<0.5%)\n"
                             "⟳ rolling = expiry rollover in progress — OI unreliable, see Next month\n\n"
                             "Number = OI change % vs yesterday, SAME contract:\n"
                             "    (today's OI − prev-day OI) ÷ prev-day OI × 100\n"
                             "+ve = OI rose (LB/SB build); −ve = OI fell (SC/LU unwind).\n"
                             "Use the 'Futures OI Δ% strength gate' above to keep only "
                             "moves of a given magnitude (e.g. |OI Δ%| ≥ 10%)."),
                    "next_fut_label": _htc(
                        "Fut Next",
                        help="FUTURES — Next-month OI signal (most informative near expiry):\n"
                             "When near-month shows '⟳ rolling', THIS column shows where\n"
                             "real money is being positioned for the coming month.\n"
                             "🟢 LB = fresh longs in next month → bullish carry-over\n"
                             "🔴 SB = fresh shorts in next month → bearish positioning\n"
                             "OI change matched by the SAME next-month expiry_date — no rollover artifacts"),
                    "far_fut_label": _htc(
                        "Fut Far",
                        help="FUTURES — Far-month OI signal (3rd monthly expiry):\n"
                             "Speculative / longer-term positioning.\n"
                             "Lower liquidity — treat as directional context, not a primary signal.\n"
                             "Large OI build in far month = conviction over the coming 2–3 months"),
                    # ── EXPIRY-CYCLE trend per expiry (the 'Cycle' columns) ────
                    # The Fut Near/Next/Far columns above are a ONE-DAY snapshot:
                    # over 503 sessions that label changes 70.7% of the time and
                    # outright reverses 22.4%. These columns run the same OI-price
                    # matrix CUMULATIVELY from the current monthly expiry cycle's
                    # first session, which flips 30.7% / reverses 2.3% — the most
                    # stable of the three bases tested (a rolling 5-day window sat
                    # in between at 36.6% / 3.4%).
                    "near_trend_label": _htc(
                        "Near Cycle",
                        help="FUTURES — Near-month build for THIS EXPIRY CYCLE, plus the "
                             "day-by-day path. Format:  🟢 LB +8% | +2, -1, +3, +2\n\n"
                             "LEFT of the bar — cumulative since the cycle began:\n"
                             "🟢 LB = Long Buildup (OI building + price up over the cycle)\n"
                             "🔴 SB = Short Buildup (OI building + price down)\n"
                             "🔵 SC = Short Covering (OI falling + price up)\n"
                             "🟠 LU = Long Unwinding (OI falling + price down)\n"
                             "⚪ = no clear build either way\n"
                             "⚠ = TODAY moved AGAINST the cycle build — it is fading\n\n"
                             "RIGHT of the bar — the last 4 daily OI changes %, "
                             "NEWEST FIRST: the leftmost figure is TODAY, then yesterday, "
                             "and so on. This separates a build still running "
                             "(+5, +4, +2, +3) from one that stalled a week ago "
                             "(-1, +0, +4, +9). Both can show the same cumulative number.\n\n"
                             "These 4 figures are NOT what the cumulative is made of. The "
                             "cumulative spans the whole cycle — 13 sessions by mid-month, "
                             "19 by expiry — so the 4 shown are only its most recent slice "
                             "and will not add up to it.\n\n"
                             "The number left of the bar is the raw cumulative OI change on "
                             "the SAME contract since the cycle's first session:\n"
                             "    (today's OI − OI at cycle start) ÷ OI at cycle start × 100\n"
                             "It grows through the month by design — median ~1% on day 1, "
                             "~5% by day 5, ~38% by day 20.\n\n"
                             "The COLOUR is not read off that raw number: it ranks each "
                             "stock against every other stock on the same cycle day, because "
                             "a raw % partly measures how far into the cycle you are.\n\n"
                             "'—' left of the bar = no cumulative yet (expiry day itself), or "
                             "the contract held under a fifth of its current OI at cycle "
                             "start, which is a fill-up ramp rather than a build. The daily "
                             "path is still shown in that case."),
                    "next_trend_label": _htc(
                        "Next Cycle",
                        help="FUTURES — Next-month build for this expiry cycle, same reading "
                             "as 'Near Cycle'.\n\n"
                             "This is the column to watch as expiry approaches: it shows "
                             "where positions are being built for the COMING month while the "
                             "near month is being closed out.\n\n"
                             "Expect the cumulative to blank out ('— | path') late in a "
                             "cycle. A next-month contract is still filling up — by day 20 "
                             "its OI is typically 20x+ what it was at cycle start, which is "
                             "contract age, not conviction, so it is not shown as a build. "
                             "The daily path stays visible throughout."),
                    # ── OPTIONS over the expiry cycle ──────────────────────────
                    # Deliberately NOT the futures method. Over a full cycle a fixed
                    # strike loses ~95-98% of its premium to theta (87-90% of OTM
                    # strikes end lower), so "OI up + premium down" measured across a
                    # cycle would call almost the whole chain 'writing' by the clock.
                    # At ONE day the same measurement is clean (median premium change
                    # +0.0%, 37% lower). So the buy/write verdict is taken DAILY and
                    # the cycle is summarised by counting those verdicts.
                    "near_opt_trend_label": _htc(
                        "Opt Near Cycle",
                        help="OPTIONS — near-month positioning across THIS EXPIRY CYCLE.\n"
                             "Format:  🔴 Bear | CE Wrt15 PE Buy9 /16d\n\n"
                             "Each session the call side and the put side are judged "
                             "separately on open interest vs premium:\n"
                             "  Buy  = OI up + premium up    (buyers accumulating)\n"
                             "  Wrt  = OI up + premium down  (writers/sellers accumulating)\n"
                             "  Cov  = OI down + premium up  (writers buying back)\n"
                             "  Exit = OI down + premium down (buyers closing)\n\n"
                             "The number after each is how many sessions that was the "
                             "dominant verdict; '/16d' is how many sessions were counted.\n\n"
                             "Direction: call WRITING caps upside (bearish) and call "
                             "buying is bullish; put WRITING supports (bullish) and put "
                             "buying is bearish. Those are summed into one score.\n\n"
                             "🟢 Bull / 🔴 Bear / ⚪ Bal is that score RANKED AGAINST "
                             "OTHER STOCKS, not read off the raw counts — writing is the "
                             "dominant activity market-wide (the most common verdict for "
                             "125 of 210 call chains and 114 of 210 put chains), so an "
                             "unranked label would read 'writing' almost everywhere and "
                             "tell you nothing.\n\n"
                             "Sessions close to expiry are EXCLUDED from the count: with "
                             "one session left 95% of the chain reads as unwinding, which "
                             "is everyone closing out, not a view. '—' means too few "
                             "sessions counted yet (early in a cycle)."),
                    "next_opt_trend_label": _htc(
                        "Opt Next Cycle",
                        help="OPTIONS — next-month positioning across this expiry cycle. "
                             "Same reading as 'Opt Near Cycle'.\n\n"
                             "Watch this as expiry approaches: it shows where option "
                             "positions are being built for the coming month while the "
                             "near month is being closed out.\n\n"
                             "Blank early in a cycle for most stocks — next-month options "
                             "barely trade until they are close to becoming the near month "
                             "(median 7 traded strikes vs 22 for the near month). Roughly "
                             "70% of rows are blank on cycle day 5 and 13% by day 15. The "
                             "'/Nd' count is per stock for the same reason: thin sessions "
                             "are dropped rather than counted."),
                    "far_opt_trend_label": _htc(
                        "Opt Far Cycle",
                        help="OPTIONS — far-month (3rd expiry) positioning this cycle.\n\n"
                             "Almost always blank, and that is the honest answer. Far-month "
                             "stock options barely trade: the median far chain trades ZERO "
                             "strikes and ZERO contracts, 82% of stock-side-days have no "
                             "far volume at all, and on a typical session only about 2 of "
                             "200+ stocks clear 100 far contracts. This column renders on "
                             "roughly 2% of rows — the handful of names where someone is "
                             "genuinely positioning 2-3 months out.\n\n"
                             "Counts only, no Bull/Bear tint. The near and next columns rank "
                             "each stock against its peers, which needs a real cross-section; "
                             "with only a few far chains trading on any day there is nobody "
                             "to rank against, so no directional verdict is claimed.\n\n"
                             "Same reading otherwise: Buy / Wrt / Cov / Exit with the number "
                             "of sessions each held, and '/Nd' sessions counted."),
                    "far_trend_label": _htc(
                        "Far Cycle",
                        help="FUTURES — Far-month OI change for this expiry cycle.\n\n"
                             "OI ONLY — no direction. The far month trades a median of 33 "
                             "contracts a day (vs 4,192 near-month), so its closing price is "
                             "stale and cannot support a Long/Short read.\n\n"
                             "Shown only where the contract actually traded (median ≥100 "
                             "contracts/day); otherwise '—'. That blanks most rows most "
                             "days, which is the honest answer — roughly three-quarters of "
                             "far-month rows never clear that bar, and a brand-new far "
                             "contract is pure fill-up for most of its first cycle."),
                    # ── Options OI-Premium Matrix (Near / Next) ─────────────────
                    # Solves the core PCR ambiguity: PCR tells you the RATIO of put/call OI
                    # but NOT whether that OI was built by BUYERS or WRITERS.
                    # Low PCR (Call Heavy) could mean:
                    #   a) Call WRITING (bearish) — the standard contrarian read
                    #   b) Call BUYING  (directly bullish) — the market is going up
                    # The OI-premium matrix disambiguates:
                    #   OI↑ + premium↑ → BUYING  (demand drives both up)
                    #   OI↑ + premium↓ → WRITING (supply: writers push price down)
                    #   OI↓ + premium↑ → SHORT COVERING (writers buying back)
                    #   OI↓ + premium↓ → LONG EXITING (buyers selling out)
                    "near_opt_label": _htc(
                        "Opt Near",
                        help="OPTIONS — Near-month OI+Premium matrix (buying vs writing):\n\n"
                             "🔥 Bull C.Buy+P.Wrt = Calls being BOUGHT + Puts being WRITTEN\n"
                             "  → Smart money net LONG: call buyers are bullish; put writers\n"
                             "    are also bullish (selling puts = accepting downside risk for premium)\n\n"
                             "❄️ Bear C.Wrt+P.Buy = Calls being WRITTEN + Puts being BOUGHT\n"
                             "  → Smart money net SHORT: call writers are capping upside;\n"
                             "    put buyers are hedging/betting on a fall\n\n"
                             "⚡ Vol Bet C+P.Buy = Both calls AND puts being bought\n"
                             "  → Straddle / strangle: big move expected, direction unclear\n\n"
                             "📊 Range C+P.Wrt = Both calls AND puts being written\n"
                             "  → Iron condor / theta play: low volatility expected\n\n"
                             "PCR shown as context — the OI-premium signal is the primary read.\n"
                             "Near expiry: positions rolling → read Opt Next instead"),
                    "next_opt_label": _htc(
                        "Opt Next",
                        help="OPTIONS — Next-month OI+Premium matrix (most reliable near expiry):\n\n"
                             "Same signal logic as Opt Near but for the next monthly expiry.\n"
                             "On expiry day, near-month options are closing/rolling — next-month\n"
                             "shows where FRESH institutional positioning is being built.\n\n"
                             "🔥 Bull C.Buy+P.Wrt → fresh next-month net longs accumulating\n"
                             "❄️ Bear C.Wrt+P.Buy → fresh next-month net shorts building\n\n"
                             "Compare with Fut Next (futures OI direction) for confirmation:\n"
                             "Fut Next = 🟢 LB + Opt Next = 🔥 Bull → HIGH CONVICTION LONG\n"
                             "Fut Next = 🔴 SB + Opt Next = ❄️ Bear → HIGH CONVICTION SHORT"),
                    "far_opt_label": _htc(
                        "Opt Far",
                        help="OPTIONS — Far-month (3rd expiry) PCR-based signal:\n"
                             "Far-month options have thin volume — OI-premium matrix is unreliable.\n"
                             "Shows PCR ratio only: Put↑ (>1.3) | Call↑ (<0.6) | Bal (0.6–1.3)\n\n"
                             "Use as macro/structural sentiment context:\n"
                             "Heavy put buying in far month = institutions hedging for next 2–3 months\n"
                             "Heavy call buying in far month = speculative bullish positioning\n"
                             "— = no far-month options activity for this stock"),
                    "deliv_vs_100d_pct": _hnc(
                        "vs 100D", format="%+.1f%%",
                        help="(7D Wtd Delivery % ÷ 100D avg − 1) × 100\n\n"
                             "+15% = recent delivery is 15% ABOVE own 100D norm → strong conviction\n"
                             "−10% = recent delivery is 10% BELOW own norm → fading interest\n"
                             "0% = exactly at own historical average\n\n"
                             "Use this to instantly compare the two adjacent columns — "
                             "positive = above own norm (bullish quality), negative = below norm."),
                    "avg_deliv_per_100d": _hnc(
                        "100D Avg Del%", format="%.1f%%",
                        help="Stock's own 100-trading-day average delivery %\n\n"
                             "This is the baseline for own-history conviction.\n"
                             "Compare against Wtd Deliv % (7D) to see if today is above or below normal for THIS stock.\n"
                             "A stock at 15% delivery is 'Strong' if its own 100D avg is 8%, even if peers are 40%."),
                    "wtd_deliv_per": _hnc(
                        "Wtd Deliv %", format="%.1f%%",
                        help="Turnover-Weighted Delivery %  (last 7 trading days)\n\n"
                             "Formula: Σ(deliv_per × turnover_lacs) / Σ(turnover_lacs)\n\n"
                             "Why weighted: a ₹500 Cr stock at 60% delivery counts more\n"
                             "than a ₹5 Cr stock at 80% delivery.\n"
                             "High % = institutions are taking delivery (holding, not squaring off)"),
                    "deliv_value_cr":_hnc(
                        "Deliv Value (₹ Cr)", format="₹%.1f",
                        help="Delivery Value in ₹ Crores  (last 7 trading days)\n\n"
                             "Formula: Σ(deliv_per / 100 × turnover_lacs) / 100\n\n"
                             "= actual ₹ worth of shares taken home (not squared off intraday)\n"
                             "This is the absolute measure of institutional conviction —\n"
                             "retail traders square off intraday, institutions take delivery"),
                    "turnover_cr":   _hnc(
                        "Turnover (₹ Cr)", format="₹%.1f",
                        help="Total Traded Value in ₹ Crores  (last 7 trading days)\n\n"
                             "Formula: Σ(turnover_lacs) / 100\n\n"
                             "= total buy + sell value traded\n"
                             "High turnover with low delivery % = speculative / intraday activity\n"
                             "High turnover with high delivery % = institutional accumulation"),
                    "price_chg_pct": _hnc(
                        "Price Chg %", format="%+.2f%%",
                        help="Average Daily Price Change %  (last 7 trading days)\n\n"
                             "Formula: AVG((close_price − prev_close) / prev_close × 100)\n\n"
                             "Simple average across all trading days in the window\n"
                             "+ = price rising on average   − = price falling on average"),
                },
            )

    # ── Sector Memory Context ─────────────────────────────────────────────────
    if regime is not None:
        try:
            _ema20_above    = (regime.get("nifty_vs_ema20", "—") == "ABOVE")
            _ema_cross_bull = (True  if regime.get("nifty_vs_ema50") == "ABOVE"
                               else False if regime.get("nifty_vs_ema50") == "BELOW"
                               else None)
            _rs1w_raw = row.get("rs_1w")
            _rs1w = (float(_rs1w_raw)
                     if _rs1w_raw is not None
                     and not (isinstance(_rs1w_raw, float) and pd.isna(_rs1w_raw))
                     else None)
            mem = cached_sector_memory_context(
                as_of_date     = selected_date,
                sector         = str(row["sector"]),
                signal         = str(row.get("signal", "")),
                regime_label   = regime_label,
                dv_ratio       = float(row.get("dv_ratio", 1.0) or 1.0),
                z_pct          = float(row.get("z_pct", 0.5) or 0.5),
                rs_1w          = _rs1w,
                ema20_above    = _ema20_above,
                ema_cross_bull = _ema_cross_bull,
                vix            = float(regime["vix"]) if regime.get("vix") is not None else None,
                fii_5d_cr      = float(regime["fii_5d_cr"]) if regime.get("fii_5d_cr") is not None else None,
                hmm_state      = regime.get("hmm_state"),
                pcr            = None,
            )
            _render_memory_context(mem)
        except Exception:
            pass


# ── Phase Card (Rotation Clock) ───────────────────────────────────────────────

def _phase_card(row: pd.Series, color: str, selected_date: date | None = None,
                min_turnover: float | None = None, key: str = "",
                window: int = 10) -> None:
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
        f"<div style='display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:3px;font-size:11px;color:rgba(255,255,255,0.55)'>"
        f"<span>Del Chg: <b style='color:{chg_c}'>{chg_str}</b></span>"
        f"<span>Slope Z: <b style='color:{color}'>{slope_z:+.2f}σ</b></span>"
        f"<span>DV: <b>₹{dv_cr:,.0f} Cr</b></span>"
        f"<span>Avg Del%: {avg_del:.1f}%</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if selected_date is None:
        return
    _render_stock_table(sector, selected_date, min_turnover, window, context="clock")


def _render_stock_table(sector: str, selected_date, min_turnover: float, window: int,
                        *, context: str = "clock", top_n: int = 12,
                        title: str | None = None, expanded: bool = False) -> None:
    """The single stock drill-down table, shared by the Rotation Clock and the Tilt.

    ONE implementation on purpose. The two callers differ only in SORT ORDER and
    the caption, and that difference is measured, not cosmetic:

      context="clock"    sorts by `dacc` (delivery vs the stock's own 100-day
                         normal). Pooled across all sectors that is the one
                         within-sector metric that works — IC +0.0186, t +5.94
                         (scripts/audit_within_sector_pick.py; the older
                         +0.032/+9.56 is the same sign on a narrower sample).

      context="tilt_ow"  sorts by RAW DELIVERY % (delivered qty / traded qty).
                         Inside a sector that is already an OVERWEIGHT the dacc
                         edge is GONE (IC +0.0081, t +1.31; identical on the
                         same-day and corrected panels, so it is the CONDITIONING,
                         not a data artifact) while the raw LEVEL survives it.
                         Top-4 within each sector, fwd return vs own sector:

                             rank by        top4      t     >=Rs5Cr floor
                             delivery %   +0.596   +4.58   +0.353 (t +3.15)
                             dacc         +0.121   +1.02   +0.103 (t +0.92)
                             momentum     +0.067   +0.66   +0.031 (t +0.40)
                             delivery Rs  -0.024   -0.42  -0.047 (t -0.76)

                         Era-stable at 10d (+0.41 / +0.65 / +0.68, all t >= 2.1)
                         and NOT a size proxy (corr with turnover -0.32; median
                         delivery % across size deciles 49 / 45 / 45%). Sorting by
                         delivery VALUE is actively backwards — its bottom-4 beat
                         its top-4 by +0.61pp/10d. Horizon-limited: fades by 9-10wk.

    `window` MUST be the caller's own period. Hardcoding it once meant a 3-month
    clock opened 14-day stock data, and dacc moves 4.34/5.20/5.44/5.26 across the
    5/10/22/65-day windows — a different top name each time.
    """
    _t = title or f"🔍 {sector} — which stocks (same {window}-day window as the clock)"
    with st.expander(_t, expanded=expanded):
        _stock_table_body(sector, selected_date, min_turnover, window,
                          context=context, top_n=top_n)


def _since_md(rel_pp, bucket: str) -> str:
    """`since` for an expander label, in markdown (labels reject raw HTML).

    Coloured by SIGN, matching the non-expander branch: up green, down red, on both
    lists. `bucket` is kept in the signature because what a colour MEANS differs by
    list - on the avoid list a red number is the call working - but that belongs in
    the hover text, not in the colour rule.

    Without this the figure rendered as plain text whenever the drill-down toggle
    was on, which is most of the time, because expander labels reject raw HTML.
    """
    if rel_pp is None or pd.isna(rel_pp):
        return ""
    tone = "grey" if abs(rel_pp) < 1e-9 else ("green" if rel_pp > 0 else "red")
    return f"  ·  :{tone}[since {rel_pp:+.1f}pp]"


def _stock_table_body(sector: str, selected_date, min_turnover: float, window: int,
                      *, context: str = "clock", top_n: int = 12,
                      compact: bool = False) -> None:
    """The table itself, WITHOUT an expander around it.

    Split out because Streamlit forbids nesting expanders: on the Tilt tab each
    sector row is itself an expander, so it must call this body directly while the
    Rotation Clock keeps calling the wrapper above.

    `compact` drops the two widest columns (1D % and the free-text Read) for
    rendering inside a half-width st.columns() pane. The thin-name warning that
    lives in Read is NOT lost with it — it moves onto the symbol as a ⚠, because
    that flag is a liquidity safety note, not decoration.
    """
    try:
        # Short end of the horizon band: the tab offers ranges ("1-2 wk", "3-4 wk"),
        # `window` is the far end and this is the near end, so both delivery-value
        # columns can be shown side by side.
        _short = max(5, int(window) - 5)
        det = cached_clock_stock_detail(sector, selected_date,
                                        float(min_turnover or 1.0),
                                        lookback_days=int(window),
                                        short_days=int(_short))
    except Exception as exc:                                      # noqa: BLE001
        st.caption(f"Stock detail unavailable: {exc}")
        return
    if det is None or det.empty:
        st.caption("No liquid stocks in this sector for the current filter.")
        return

    sort_col = "wtd_deliv_per" if context == "tilt_ow" else "dacc"
    if sort_col in det.columns and det[sort_col].notna().any():
        det = det.sort_values(sort_col, ascending=False)
    show = det.head(top_n).copy()
    if compact and "thin" in show.columns:
        show["symbol"] = show["symbol"] + np.where(show["thin"].fillna(False), " ⚠", "")
    _wk = max(1, int(round(window / 5)))
    _swk = max(1, int(round(_short / 5)))
    _c_long, _c_short = f"Deliv Cr {_wk}W", f"Deliv Cr {_swk}W"
    _s_plural = "" if _swk == 1 else "s"
    _dup = _c_short == _c_long

    # ── Flow: is the delivery buying picking up or fading? ───────────────
    # The two delivery columns OVERLAP, so the older stretch is (long - short),
    # which is always exactly one week. Compare like with like by putting both
    # sides on a per-week footing before dividing.
    #
    # CALIBRATION (757,884 stock-days, 2022-2026, >=Rs 1 Cr/day liquidity and
    # >=Rs 25 Cr delivered in the window): a +/-25% deadband splits Rising 29% /
    # Steady 39% / Fading 32% and the label changes on 28.9% of stock-days with
    # only 2.1% outright reversals. Tighter (+/-15%) doubles reversals to 4.9%.
    #
    # IT PREDICTS NOTHING, and that is measured, not assumed: forward 10d return
    # in EXCESS of the stock's own sector is +0.552% Rising vs +0.627% Fading -
    # a -0.075pp spread pointing the WRONG way, every |t| < 1. Descriptive only.
    def _flow(_s, _l):
        if pd.isna(_s) or pd.isna(_l):
            return "—"
        older = float(_l) - float(_s)          # the oldest week in the window
        if older <= 0 or float(_l) < 25.0:     # no older week, or too small to read
            return "—"
        ratio = (float(_s) / _swk) / older     # per-week recent vs per-week older
        return "↑ Rising" if ratio > 1.333 else ("↓ Fading" if ratio < 0.75
                                                 else "→ Steady")
    if not _dup and {"deliv_value_cr", "deliv_value_cr_short"}.issubset(show.columns):
        show["Flow"] = [_flow(a, b) for a, b in
                        zip(show["deliv_value_cr_short"], show["deliv_value_cr"])]

    # At the shortest horizon (window=5) _short collapses onto window, so both
    # delivery columns would rename to the SAME label and `show[cols]` would pull
    # the pair back in as duplicates -> ValueError. Drop the redundant one first.
    if _dup:
        show = show.drop(columns=["deliv_value_cr_short"], errors="ignore")

    show = show.rename(columns={
        "symbol": "Stock", "ltp": "LTP", "chg_1d_pct": "1D %",
        "ret_win_pct": f"{window}D %", "rel_ret_pct": "vs sector",
        "wtd_deliv_per": "Deliv %", "dacc": "Deliv x",
        "deliv_value_cr": _c_long, "deliv_value_cr_short": _c_short,
        "read": "Read",
    })
    cols = (["Stock", "LTP", f"{window}D %", "Deliv %", "Deliv x"]
            + ([] if _dup else [_c_short]) + [_c_long]
            + ([] if _dup else ["Flow"])
            if compact else
            ["Stock", "LTP", "1D %", f"{window}D %", "vs sector",
             "Deliv %", "Deliv x"] + ([] if _dup else [_c_short]) + [_c_long]
            + ([] if _dup else ["Flow"]) + ["Read"])
    cols = [c for c in cols if c in show.columns]
    _sorted_by = ("delivery % — the share of traded volume actually delivered"
                  if context == "tilt_ow" else
                  "delivery vs the stock's own 100-day normal")
    # Explicit column_config: without it the last columns get clipped and render
    # blank (that is what happened to "Share of sector delivery %"). Share of
    # sector delivery is dropped — "Deliv Cr" says the same thing more legibly.
    _cfg = {
            "Stock": st.column_config.TextColumn(width="small"),
            "LTP": st.column_config.NumberColumn(
                "LTP ₹", format="%.2f", width="small",
                help="Last traded (close) price on the selected date."),
            "1D %": st.column_config.NumberColumn(
                "1D %", format="%+.2f", width="small",
                help="Close-to-close move vs the previous session."),
            f"{window}D %": st.column_config.NumberColumn(
                f"{_wk}W %", format="%+.2f", width="small",
                help=f"Actual return over the {window}-day window (close now vs "
                     f"close {window} days ago). NOT an average of daily moves."),
            "vs sector": st.column_config.NumberColumn(
                "vs sector", format="%+.2f", width="small",
                help="This stock's window return minus the sector's median "
                     "stock. Positive = leading its own sector."),
            "Deliv %": st.column_config.NumberColumn(
                "Deliv %", format="%.1f", width="small",
                help="TRUE DELIVERY: of everything traded in this stock over the "
                     "window, this much was actually delivered (taken into a demat "
                     "account) rather than squared off intraday. High = real "
                     "ownership changing hands; low = churn.\n\n"
                     "Measured on 1.32M stock-days (2018-2026): ranking a sector's "
                     "stocks by this beats every alternative INSIDE an overweight "
                     "sector — top-4 earn +0.60pp over the next 10 sessions vs their "
                     "own sector (t +4.58), positive in all three eras, and still "
                     "+0.35pp (t +3.15) with a Rs 5 Cr delivery floor. It is not a "
                     "size proxy: correlation with turnover is -0.32."),
            "Deliv x": st.column_config.NumberColumn(
                "Deliv ×", format="%.2f", width="small",
                help="Delivery %% over the window divided by this stock's OWN "
                     "100-day normal. Above 1 = heavier real buying than usual."
                     + ("  Shown as context only here: inside an OVERWEIGHT "
                        "sector this measured t +1.31, i.e. nothing."
                        if context == "tilt_ow" else
                        "  This is the column the list is sorted by, and the "
                        "only within-sector metric that measured predictive.")),
            _c_long: st.column_config.NumberColumn(
                f"Deliv ₹Cr {_wk}W", format="%.0f", width="small",
                help=f"**Money that actually bought and KEPT this stock** over the "
                     f"last {window} trading days ({_wk} weeks) - the full horizon you "
                     f"picked above.\n\n"
                     f"Delivery means the shares were taken into a demat account "
                     f"instead of being sold again the same day. Day traders square "
                     f"off; real buyers take delivery. So this is the size of the "
                     f"genuine money behind the move, in ₹ crore.\n\n"
                     f"Worked out as each day's delivery % times that day's traded "
                     f"value, added up over the window.\n\n"
                     f"Example: `158` means about ₹158 crore of this stock was "
                     f"bought and held over these {_wk} weeks.\n\n"
                     f"This number **includes** the {_swk}W column on its left. "
                     f"Subtract one from the other to see whether the buying is "
                     f"picking up or fading - the **Flow** column does it for you."),
            _c_short: st.column_config.NumberColumn(
                f"Deliv ₹Cr {_swk}W", format="%.0f", width="small",
                help=f"The same money-taken-home figure, but only the last "
                     f"{_short} trading days ({_swk} week{_s_plural}).\n\n"
                     f"**The two columns overlap.** {_wk}W already contains everything "
                     f"in this {_swk}W. So do not read one as a share of the other - "
                     f"subtract:\n\n"
                     f"`{_wk}W - {_swk}W` = what was delivered in the OLDEST week of "
                     f"the window.\n\n"
                     f"Example at the 1-2 wk horizon, `1W = 50` and `2W = 80`:\n\n"
                     f"older week = `80 - 50` = `30`, recent week = `50`, so it went "
                     f"30 then 50 and the buying is **increasing**.\n\n"
                     f"The other way round, `1W = 30` and `2W = 80`: older week = "
                     f"`50`, recent week = `30`, so the buying is **fading**.\n\n"
                     f"The **Flow** column applies this rule for you."),
            "Flow": st.column_config.TextColumn(
                "Flow", width="small",
                help=f"**Is the delivery buying picking up or dying down?**\n\n"
                     f"It compares the two money columns on a per-week basis:\n\n"
                     f"- recent = the {_swk}W figure spread over {_swk} week{_s_plural}\n"
                     f"- older = `{_wk}W - {_swk}W`, the earliest week\n\n"
                     f"- **Rising** - the recent week is running 33% or more ABOVE the "
                     f"older one; buying is stepping up.\n"
                     f"- **Fading** - 25% or more BELOW; the buying is drying up.\n"
                     f"- **Steady** - in between, no real change.\n"
                     f"- **—** (dash) - too small to judge (under ₹25 crore delivered "
                     f"in the window), or there is no older week to compare with.\n\n"
                     f"Example: `1W = 50` and `2W = 80` gives older = `30` against "
                     f"recent `50`, so Rising.\n\n"
                     f"**Honest warning.** This only describes what already happened. "
                     f"Measured on 757,884 stock-days (2022-2026), Rising names went "
                     f"on to make +0.55% over the next 10 days versus their own sector "
                     f"and Fading names +0.63% - the gap is 0.08pp the WRONG way and is "
                     f"not significant. Do not buy something because it says Rising."),
            "Read": st.column_config.TextColumn(width="large"),
    }
    # Same collapse as above: with one delivery column the _c_short entry would
    # have silently overwritten _c_long in this dict and attached "only the last
    # N days" help to the only column there is. Restate it for the single column.
    if _dup:
        _cfg[_c_long] = st.column_config.NumberColumn(
            f"Deliv ₹Cr {_wk}W", format="%.0f", width="small",
            help=f"**Money that actually bought and KEPT this stock** over the "
                 f"last {window} trading days ({_wk} week{_s_plural}).\n\n"
                 f"Delivery means the shares were taken into a demat account "
                 f"instead of being sold again the same day. Day traders square "
                 f"off; real buyers take delivery. So this is the size of the "
                 f"genuine money behind the move, in ₹ crore.\n\n"
                 f"Worked out as each day's delivery %% times that day's traded "
                 f"value, added up over the window.\n\n"
                 f"At this horizon the window is a single week, so there is no "
                 f"older week to compare against and no Flow column.")

    st.dataframe(show[cols], hide_index=True, use_container_width=True,
                 column_config=_cfg)

    if context == "tilt_ow":
        st.caption(
            f"**Sorted by {_sorted_by}** — of everything traded, how much was actually "
            "delivered rather than squared off intraday.\n\n"
            "**This is the one stock-level ranking that measured positive inside an "
            "overweight sector.** Top-4 within each sector, forward return vs their own "
            "sector, 2018-2026: **+0.60pp / 10 sessions (t +4.58)**, positive and "
            "significant in all three eras (+0.41 / +0.65 / +0.68), and still "
            "**+0.35pp (t +3.15)** once a ₹5 Cr delivery floor makes it tradeable. "
            "It is not a size proxy — correlation with turnover −0.32, and median "
            "delivery % is flat across size deciles (49 / 45 / 45%).\n\n"
            "Everything else tested in the same universe failed: **Deliv ×** "
            "(vs own 100-day normal) +0.12pp t +1.02, **momentum** +0.07pp t +0.66, "
            "and **Deliv ₹Cr** −0.02pp t −0.42 — ranking by rupee size is actively "
            "backwards, its *bottom*-4 beat its top-4 by +0.61pp. Deliv × works across "
            "the market as a whole (t +5.94) but its edge lives in the sectors you are "
            "**not** buying, which is why it is a context column here, not the sort.\n\n"
            "⚠️ Horizon-limited: the effect is strong at 1-2 wk (t +4.58) and 3-4 wk "
            "(t +3.62) and fades by 9-10 wk (t +1.53, and the ₹5 Cr floor kills it). "
            "Incremental cost is zero — you have already decided to buy the sector and "
            "must hold something — but standalone it would not clear a round trip. "
            "`scripts/audit_within_sector_pick.py`")
        return

    drivers = det.nlargest(3, "contrib_pct")["symbol"].tolist() if "contrib_pct" in det else []
    st.caption(
        "**Sorted by delivery vs the stock's own 100-day normal** - measured over "
        "1.19M stock-days (2018-2026) that is the one within-sector metric that "
        "works: IC +0.032, t +9.6, quintiles monotonic (-0.25 to +0.25 %/10d), "
        "stable across all three eras, and it survives controlling for the stock's "
        "own momentum.\n\n"
        + (f"**Drove the sector move:** {', '.join(drivers)} - shown for attribution "
           f"only. Ranking by who drove it measured **-0.34%/10d (t -4.8)**, i.e. the "
           f"names that already moved tend to give it back. Same for momentum "
           f"(t -4.8) and delivery-value share (t -4.8).\n\n" if drivers else "")
        + "⚠️ Standalone this does not pay: top-3 by this metric nets **-4.9%/yr** "
          "after a 0.5% round trip. It is a **selection rule for a sector you have "
          "already decided to buy** - where you must hold something anyway and the "
          "incremental cost is zero - not a signal in its own right.\n\n"
        + "⚠️ Conditional caveat: that edge is measured POOLED. Split out, it is "
          "absent inside sectors that are already momentum leaders (t +1.31), so "
          "read this ranking as informative on weak/improving phases and as pure "
          "attribution on a leading one."
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
        _rotation_clock_chart(df, period_label, nifty_return=nifty_ret,
                              center=(df["sector_median_ret"].iloc[0] if "sector_median_ret" in df.columns else None)),
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
            # Custom Range has neither selected_date nor window in scope, and already
            # renders its own '📋 Stocks in ...' expander below — no drill-down here.
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
                        "symbol":       _htc("Symbol", width="small"),
                        "company_name": _htc("Company"),
                        "industry":     _htc("Sub-Sector"),
                        "price_start":  _hnc(
                            f"Price on {from_snap.strftime('%d %b')}", format="₹%.2f",
                            help="Price at the start of the period (prev_close of first trading day)"),
                        "price_end":    _hnc(
                            f"Price on {to_snap.strftime('%d %b')}", format="₹%.2f",
                            help="Price at the end of the period (close of last trading day)"),
                        "period_ret_pct": _hnc(
                            "Period Return %", format="%+.2f%%",
                            help=f"(Price End − Price Start) / Price Start × 100\n"
                                 f"Period: {from_snap.strftime('%d %b')} → {to_snap.strftime('%d %b %Y')}"),
                        "wtd_deliv_per": _hnc(
                            "Avg Delivery %", format="%.1f%%",
                            help="Turnover-weighted average delivery % over the period"),
                        "deliv_value_cr": _hnc(
                            "Delivery Value (₹ Cr)", format="₹%.1f",
                            help="Total delivery value in ₹ Crores over the period"),
                        "turnover_cr": _hnc(
                            "Turnover (₹ Cr)", format="₹%.1f"),
                        "trading_days": _hnc(
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
    using_relative = "forward_vs_peers" in bt.columns

    nifty_note = (
        "  **'Correct' = sector beat its PEER sectors (inflow) or underperformed peers "
        "(outflow)** — the cross-sectional benchmark the clock classifies against. "
        + (f"(Nifty50 moved {nifty_fwd_ret:+.2f}% over the window — shown as a reference column only.)"
           if nifty_fwd_ret is not None else "")
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
        avg_vs_peers   = grp["forward_vs_peers"].mean() if "forward_vs_peers" in grp.columns else None
        n_correct      = int(grp["signal_correct"].fillna(False).sum())
        n_total        = len(grp)
        hit_rate       = n_correct / n_total * 100

        total_correct   += n_correct
        total_predicted += n_total

        expected_positive = phase in inflow_phases
        # Color by the PEER-relative avg (the benchmark "Correct?" uses), else absolute
        ref_ret = avg_vs_peers if avg_vs_peers is not None else avg_ret
        ret_ok  = (ref_ret > 0) if expected_positive else (ref_ret < 0)

        vs_str = f" (vs peers: {avg_vs_peers:+.1f}%)" if avg_vs_peers is not None else ""
        correct_label = (
            "beat peer sectors" if expected_positive else "underperform peers"
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
                f"Correct = forward return {correct_label} (cross-sectional benchmark).\n"
                + (f"Nifty50 forward (reference): {nifty_fwd_ret:+.2f}%" if nifty_fwd_ret else "")
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
                  "forward_vs_peers", "forward_vs_nifty", "signal_correct",
                  "cum_price_ret_pct", "slope_z", "price_deliv_corr", "deliv_chg_pct"]
    disp = bt[[c for c in avail_cols if c in bt.columns]].copy()

    disp["signal_correct"] = disp["signal_correct"].map(
        lambda v: "✅ Correct" if v is True else ("❌ Wrong" if v is False else "—")
    )

    correct_help = (
        "✅ = sector beat its PEER sectors (inflow) or underperformed peers (outflow)\n"
        "❌ = signal was wrong vs peers\n"
        "— = Neutral (no directional prediction)\n"
        "(Scored vs the cross-sectional peer median — NOT vs Nifty50.)"
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
            "sector":            _htc("Sector"),
            "phase":             _htc("Phase on Signal Date",
                help=f"Rotation phase as of {signal_date.strftime('%d %b %Y')}"),
            "signal_confidence": _hnc(
                "Confidence", format="%.2f",
                help="Signal strength 0→1.\n"
                     "Based on slope_z magnitude, price-delivery anti-correlation strength.\n"
                     "High confidence (>0.7) = stronger evidence for the phase call."),
            "forward_ret_pct":   _hnc(
                "Forward Return (abs)", format="%+.2f%%",
                help=f"Cumulative sector return from {signal_date.strftime('%d %b')} → {selected_date.strftime('%d %b')}"),
            "forward_vs_peers":  _hnc(
                "vs Peers", format="%+.2f%%",
                help="Forward return MINUS the median sector's forward return. Positive = "
                     "this sector beat the OTHER sectors. ⭐ THIS is what 'Correct?' is scored on "
                     "(the rotation question is which sector beats which, not vs the index)."),
            "forward_vs_nifty":  _hnc(
                "vs Nifty50",  format="%+.2f%%",
                help=f"Forward return minus Nifty50 ({nifty_fwd_ret:+.2f}% forward) — shown as a "
                     "REFERENCE only. The clock is scored vs peers, not vs the index, so this can "
                     "be negative while 'Correct?' is ✅ (the sector beat peers but not the index)."
                     if nifty_fwd_ret is not None else "Forward return vs Nifty50 (reference only)"),
            "signal_correct":    _htc("Correct?", help=correct_help),
            "cum_price_ret_pct": _hnc(
                "Price Ret on Signal Date", format="%+.2f%%",
                help=f"Sector return as of {signal_date.strftime('%d %b')} — what triggered classification"),
            "slope_z":           _hnc(
                "Delivery Slope Z", format="%+.2f",
                help="Delivery momentum Z-score on the signal date"),
            "price_deliv_corr":  _hnc(
                "Price-Del Corr", format="%.2f",
                help="Correlation between daily price return and daily delivery %.\n"
                     "Negative = price rising as delivery falls (distribution confirmed).\n"
                     "Only Weakening signals with corr < -0.15 are shown as Weakening."),
            "deliv_chg_pct":     _hnc(
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
| 💰 **Leading**   | Rising ↑ | Above peer-median ↑ | Institutions buying + outperforming peer sectors | ✅ **ACT — BUY / HOLD** |
| 🔍 **Improving** | Rising ↑ | Below peer-median ↓ | Accumulating but price not confirming — contrarian zone | 👀 **WATCH** (await price confirmation) |
| ⚠️ **Weakening** | Falling ↓ | Above peer-median ↑ | Distributing into outperforming prices | **EXIT / REDUCE** |
| 📤 **Lagging**   | Falling ↓ | Below peer-median ↓ | Institutions exiting, price lagging peers | **AVOID** |

> **Walk-forward evidence (peer-relative, honest):** on the **1-month** clock, Leading modestly beat peers (~+0.8%/mo, ~63% hit) and Lagging underperformed (~−0.9%/mo, ~67% hit); **Improving and Weakening showed no reliable edge**. The **1-week** clock is essentially coin-flip. Treat this as a *flow map* (where institutional delivery is moving now), not a precise alpha signal — use the 1M view, and confirm with the daily Smart-Money signal + FII/DII flow. (Sample is small: ~14 independent monthly windows.)

**Quadrant Center = Typical Sector (cross-sectional median)** — The gold dashed line marks the MEDIAN sector return for the period; sectors to the RIGHT are outperforming their peers, to the LEFT underperforming. This is the correct, apples-to-apples benchmark for *sector rotation* (and is consistent with the cross-sectional slope-Z on the Y-axis). Nifty50 is shown as a faint dotted reference only — benchmarking equal-weight median sector returns against the cap-weighted index made every sector look like it lagged when mega-caps led.

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

    # Bubble chart — collapsed by default. The quadrant counts above and the
    # sector cards below carry the same information in less space; the chart is
    # for when you want to see WHERE in a quadrant a sector sits (a bubble just
    # inside "Leading" is a different call from one deep in it). Plotly is also
    # the heaviest thing on the tab, so not rendering it until asked keeps the
    # page responsive when switching Analysis Period.
    _n_plot = len(df)
    with st.expander(f"📈 Rotation Clock chart — {sel} · {_n_plot} sectors "
                     f"(click to show)", expanded=False):
        st.plotly_chart(
            _rotation_clock_chart(df, sel, nifty_return=nifty_ret,
                                  center=(df["sector_median_ret"].iloc[0]
                                          if "sector_median_ret" in df.columns else None)),
            use_container_width=True,
            key=f"rot_clock_chart_{window}",
        )
        st.caption(
            "Vertical dashed line = the TYPICAL sector (cross-sectional median), which is "
            "the quadrant centre — not Nifty50. Nifty is the thin grey reference only. "
            "Bubble size = delivery value ₹Cr. A sector's DISTANCE from the centre matters: "
            "a name just over the line is a marginal call, deep in a quadrant is a clear one."
        )

    # ── RESULTS: what the clock said on a past date, and what followed ────────
    _rc1, _rc2 = st.columns([3, 1])
    with _rc2:
        _show_cres = st.toggle("📊 Results", value=False, key="clock_results",
                               help="Turn this on to look back: pick any past "
                                    "date, see which phase each sector was in "
                                    "that day, and find out what happened next.")
    if _show_cres:
        _render_clock_replay(selected_date, min_turnover, int(window), sel)
        st.markdown("---")

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
            # ACT — Leading is the only inflow phase with a validated forward edge
            # (walk-forward: +1.4%/mo relative). Money entering WITH price confirming.
            if not leading.empty:
                st.markdown(
                    f"<div style='font-size:12px;color:#00c853;font-weight:600;margin-bottom:2px'>"
                    f"💰 LEADING — Money Entering · ✅ ACT ({len(leading)})</div>"
                    f"<div style='font-size:10.5px;color:rgba(255,255,255,0.5);margin-bottom:5px'>"
                    f"Delivery rising AND price beating peer sectors — the most reliable "
                    f"phase, but the edge is modest and only on the 1-MONTH clock "
                    f"(~+0.8% vs peers, ~63% hit); the 1-week clock is near coin-flip.</div>",
                    unsafe_allow_html=True,
                )
                for _, row in leading.iterrows():
                    _phase_card(row, "#00c853", selected_date, min_turnover, window=window)
            # WATCH — Improving (contrarian inflow) had NO forward edge on its own in
            # the walk-forward (~-0.1%/mo). Accumulation without price confirmation —
            # a watchlist, not a buy. Wait for it to migrate into Leading.
            if not improving.empty:
                st.markdown(
                    f"<div style='font-size:12px;color:#40c4ff;font-weight:600;margin:12px 0 2px 0'>"
                    f"🔍 IMPROVING — Contrarian Inflow · 👀 WATCH ({len(improving)})</div>"
                    f"<div style='font-size:10.5px;color:rgba(255,255,255,0.5);margin-bottom:5px'>"
                    f"Delivery accumulating but price NOT yet confirming. No standalone "
                    f"forward edge — a watchlist. Buy only once it crosses into Leading.</div>",
                    unsafe_allow_html=True,
                )
                for _, row in improving.iterrows():
                    _phase_card(row, "#40c4ff", selected_date, min_turnover, window=window)

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
                    _phase_card(row, "#ff9100", selected_date, min_turnover, window=window)
            if not lagging.empty:
                st.markdown(
                    f"<div style='font-size:12px;color:#d50000;font-weight:600;margin:10px 0 4px 0'>"
                    f"📤 LAGGING — Money Exiting ({len(lagging)})</div>",
                    unsafe_allow_html=True,
                )
                for _, row in lagging.iterrows():
                    _phase_card(row, "#d50000", selected_date, min_turnover, window=window)

    if not neutral.empty:
        with st.expander(f"⚖️ Neutral Sectors — {len(neutral)} with no clear bias", expanded=False):
            for _, row in neutral.iterrows():
                _phase_card(row, "#888888", selected_date, min_turnover, window=window)

    # Cross-period comparison
    with st.expander("📊 Cross-Period Comparison — All 4 Timeframes at Once", expanded=False):
        st.caption("See each sector's rotation phase simultaneously across 1W / 2W / 1M / 3M.")
        _render_cross_period(selected_date, min_turnover)

    # ── Signal Validation ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📊 Signal Validation — Did the {sel} Rotation Clock Call It Right?")

    # ── Walk-forward reliability across timeframes (1W / 2W / 1M) ─────────────
    st.caption(
        "**Walk-forward reliability — judged vs PEER sectors across many past windows.** "
        "Compare timeframes: this is the meaningful read; the single window further down "
        "is just the latest worked example."
    )
    _WINS = [(5, "1W (~5d)"), (10, "2W (~10d)"), (22, "1M (~22d)")]
    with st.spinner("Computing walk-forward reliability across 1W / 2W / 1M…"):
        comp = []
        for w, lbl in _WINS:
            a = cached_rotation_clock_accuracy(selected_date, w, float(min_turnover))
            if not a or not a.get("n_predictions"):
                continue
            bp = a["by_phase"]; le = bp.get("Leading", {}); lg = bp.get("Lagging", {})
            im = bp.get("Improving", {})
            comp.append({
                "Timeframe": lbl, "Windows": a["n_signals"],
                "Overall hit %": a["overall_hit"],
                "Inflow−Outflow %/win": a["inflow_outflow_spread"],
                "Leading hit %": le.get("hit_rate"), "Leading edge %": le.get("avg_excess"),
                "Improving edge %": im.get("avg_excess"),
                "Lagging hit %": lg.get("hit_rate"), "Lagging edge %": lg.get("avg_excess"),
            })
    if comp:
        st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True, column_config=_hcfg(pd.DataFrame(comp)))
        st.caption(
            "⚖️ **Read across timeframes:** the **1-week** clock is essentially coin-flip; "
            "reliability rises with horizon — the **1-month** clock carries the edge "
            "(Leading = act, Lagging = avoid; Improving/Weakening unreliable). A flow map, "
            "not a standalone alpha signal — confirm with the daily Smart-Money signal + "
            "FII/DII flow. (Limited samples: ~14–40 windows each.)"
        )
    st.markdown("###### 🔬 Latest window — worked example")

    st.caption(
        f"The rotation clock computed signals **{window} trading days ago**; below shows how "
        f"those specific sectors did since (vs peers). NOTE: one window is a tiny sample — "
        f"read it as an illustration, not a verdict (use the aggregate above)."
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
            "sector":            _htc("Sector"),
            "phase":             _htc("Phase"),
            "flow_signal":       _htc("Flow Signal"),
            "cum_price_ret_pct": _hnc(
                "Price Return %", format="%+.2f%%",
                help="Cumulative turnover-weighted price return over the selected period\n"
                     "= compound product of daily sector returns"),
            "slope_z":           _hnc(
                "Slope Z-Score", format="%+.2f",
                help="Cross-sectional z-score of delivery slope across all sectors\n"
                     "Tells you which sectors are gaining vs losing institutional interest relative to each other\n"
                     "> 0.25σ = rising momentum   < -0.25σ = falling momentum"),
            "delivery_slope":    _hnc(
                "Delivery Slope", format="%+.4f",
                help="Linear regression slope of daily weighted delivery %\n"
                     "Positive = institutions increasingly committed over the period\n"
                     "Units: delivery % points per trading day"),
            "deliv_value_cr":    _hnc(
                "Deliv Value (₹ Cr)", format="₹%.1f",
                help="Total delivery value over the period in ₹ Crores"),
            "deliv_chg_pct":     _hnc(
                "Deliv Chg vs Prior %", format="%+.1f%%",
                help="% change in delivery value vs the prior equal-length period\n"
                     "Positive = more institutional money flowing in this period\n"
                     "Negative = less institutional money (outflow vs prior period)"),
            "avg_deliv_pct":     _hnc(
                "Avg Delivery %", format="%.1f%%",
                help="Average turnover-weighted delivery % over the current period"),
            "num_days":          _hnc("Trading Days", format="%d"),
        },
    )


# ── Smart Money Tab (existing content) ───────────────────────────────────────

def _render_smart_money(selected_date: date, min_turnover: float) -> None:
    st.caption(
        f"Where are institutions entering and exiting? "
        f"Based on **100 days** of turnover-weighted delivery data "
        f"— as of **{selected_date.strftime('%d %b %Y')}**"
    )
    st.info(
        "**Which tab to trade for sector returns?** Head-to-head backtest (137 days, both "
        "engines walked forward — scripts/backtest_smartmoney_vs_tilt.py): for **picking which "
        "sector rises next**, the **🎯 1–2 Wk Forward Tilt (momentum) wins clearly** — its BUY "
        "sectors returned **+2.7%/10d vs +1.3%** here, rank-IC t+7.0 vs ~0, and its top pick "
        "+3.0% vs −0.3%. This delivery signal is best as a **confirmation overlay**, not a "
        "standalone sector picker: when **both** tabs flag the same sector as BUY, forward "
        "return was the highest of all (**+3.7%/10d, 60% hit**). So use this page to see *where "
        "institutional money is flowing* and to **confirm** a Forward-Tilt pick — not to rank "
        "sectors by return on its own. (Bull-only sample — delivery data is 2024-12 onward.)",
        icon="🧭")
    st.warning(
        "**Picking the STOCK inside a sector? (re-measured 2026-08-09 — "
        "scripts/audit_stock_pick_in_clock.py, 1.19M stock-days 2018-2026, forward 10d "
        "EXCESS OVER THE STOCK'S OWN SECTOR, Newey-West t.)**\n\n"
        "**It depends entirely on how you measure delivery.** Against the stock's OWN "
        "100-day normal, delivery accumulation is the single best stock-level signal here: "
        "IC **+0.032, t +9.6**, quintiles monotonic (−0.25 → +0.25 %/10d), stable in all "
        "three eras, and it survives controlling for the stock's own momentum. Measured as "
        "a *sector-relative percentile* it looks anti-predictive — that framing, not "
        "accumulation itself, was the problem.\n\n"
        "**What is genuinely anti-predictive:** price momentum vs the sector (t −4.8), "
        "share of the sector's delivery value (t −4.8), turnover surge (t −3.5) and "
        "\"who drove the sector move\" (t −4.9). So the biggest / most-active name is a "
        "**bad** default pick.\n\n"
        "**Correction to earlier copy:** this banner used to say the laggard beats the "
        "extended leader inside a strong sector. Measured, the opposite holds — in a "
        "top-30%-momentum sector the leader returns **+0.377%** vs the laggard's "
        "**+0.128%** (laggard−leader −0.25pp, t −2.70). The decile curve is **U-shaped**: "
        "both extremes beat the middle. Outside strong sectors the laggard edges ahead, "
        "but not significantly (t +1.05).\n\n"
        "**Cost decides how to use it:** top-3 by delivery-vs-own-normal nets **−4.9%/yr** "
        "standalone (0.5% round trip vs +0.31%/10d gross). It pays only as a **selection "
        "rule inside a sector you have already decided to buy**, where you must hold "
        "something anyway. The sector call remains the validated edge.",
        icon="⚠️")

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
        regime = cached_market_regime(selected_date)
        rot    = cached_sector_overlay(selected_date, min_turnover)

    if rot is None or rot.empty:
        st.warning("Insufficient data. Need at least 10 trading days of history.")
        return

    # ── Memory-sharpened ranking (default ON) ────────────────────────────────
    # adj_score = accum_score tilted by historical forward-outcome conviction in
    # similar past setups (regime-aware, shrunk by evidence quality). The toggle
    # lets the user fall back to the raw cross-sectional accum_score for compare.
    _has_overlay = "adj_score" in rot.columns
    if _has_overlay:
        memory_on = st.toggle(
            "🧠 Memory-sharpened ranking",
            value=True,
            key="sector_memory_on",
            help="Re-rank sectors by accumulation score tilted with how this exact "
                 "footprint actually performed in similar past regimes. Conviction "
                 "badges: 🔥 High · ✅ Confirm · ⚠️ History Disagrees · ❔ Unproven. "
                 "Off = raw cross-sectional score only.",
        )
    else:
        memory_on = False

    # ── Regime-conditional ranking ────────────────────────────────────────────
    # MOMENTUM (accum_score / adj_score) predicts forward returns in up-trends,
    # but in a BEAR/CAUTION regime the momentum leaders are the HIGH-BETA sectors
    # that amplify the fall — buying them is how a "good signal" still loses money.
    # So in down-markets we rank by DEFENSE (low beta, low downside-capture,
    # positive relative strength) instead. This is the core fix for "I bought your
    # pick in a downtrend and still lost".
    # The scenario engine classifies WHERE the market is + where it's HEADED
    # (7 scenarios incl. transitions) and recommends the ranking factor:
    # momentum (uptrend/breakout) · accumulation (range) · defense (downtrend) ·
    # reversal (bottoming). The dashboard ranks accordingly.
    try:
        scenario = cached_market_scenario(selected_date)
    except Exception:
        scenario = {"label": "—", "playbook": "", "ranking_factor":
                    ("defense" if regime.get("regime") in ("BEAR", "CAUTION") else "momentum"),
                    "nifty_5d": 0, "nifty_20d": 0, "breadth": 0.5, "breadth_trend": 0,
                    "vix": regime.get("vix"), "fii_5d_cr": regime.get("fii_5d_cr") or 0}
    _factor = scenario.get("ranking_factor", "momentum")
    _has_defense = "defense_score" in rot.columns and rot["defense_score"].notna().any()
    _defense_mode = bool(_factor == "defense" and _has_defense)

    # ── Scenario playbook banner ──────────────────────────────────────────────
    _SCEN_COL = {"momentum": "#00c853", "accumulation": "#40c4ff",
                 "defense": "#ff9100", "reversal": "#69f0ae"}
    _sc = _SCEN_COL.get(_factor, "#888")
    _vix_s = scenario.get("vix"); _vix_s = f"{_vix_s:.1f}" if isinstance(_vix_s, (int, float)) else "—"
    st.markdown(
        f"<div style='border-left:4px solid {_sc};background:rgba(255,255,255,0.03);"
        f"padding:8px 12px;margin:6px 0;border-radius:0 6px 6px 0'>"
        f"<b style='font-size:14px;color:{_sc}'>{scenario.get('label','—')}</b> "
        f"<span style='font-size:11px;color:rgba(255,255,255,0.45)'>· Nifty 5d "
        f"{scenario.get('nifty_5d',0):+.1f}% / 20d {scenario.get('nifty_20d',0):+.1f}% · "
        f"breadth {scenario.get('breadth',0)*100:.0f}% ({scenario.get('breadth_trend',0)*100:+.0f}) · "
        f"VIX {_vix_s} · {scenario.get('fii_flow_source','FII')} 5d: FII "
        f"₹{scenario.get('fii_cash_5d') if scenario.get('fii_cash_5d') is not None else scenario.get('fii_5d_cr',0):+,.0f} / "
        f"DII ₹{(scenario.get('dii_cash_5d') or 0):+,.0f} Cr</span>"
        f"<div style='font-size:11.5px;color:rgba(255,255,255,0.72);margin-top:3px'>"
        f"{scenario.get('playbook','')}</div></div>",
        unsafe_allow_html=True,
    )

    _momentum_col = "adj_score" if (memory_on and _has_overlay) else "accum_score"
    _FACTOR_COL = {
        "momentum":     _momentum_col,
        "defense":      "defense_score",
        "accumulation": "accumulation_score",
        "reversal":     "reversal_score",
    }
    _rank_col = _FACTOR_COL.get(_factor, _momentum_col)
    if _rank_col not in rot.columns:
        _rank_col = "accum_score"
    _accum_mode = _factor in ("accumulation", "reversal")
    rot = rot.sort_values(_rank_col, ascending=False).reset_index(drop=True)

    if "z_score" not in rot.columns:
        st.cache_data.clear()
        st.rerun()

    # ── Market Regime Banner ─────────────────────────────────────────────────
    _REGIME_STYLE = {
        "BULL":          ("🟢", "#00c853", "rgba(0,200,83,0.10)",  "rgba(0,200,83,0.30)"),
        "CAUTIOUS BULL": ("🟡", "#ffd600", "rgba(255,214,0,0.08)", "rgba(255,214,0,0.30)"),
        "SIDEWAYS":      ("🔵", "#40c4ff", "rgba(64,196,255,0.08)","rgba(64,196,255,0.30)"),
        "CAUTION":       ("🟠", "#ff9100", "rgba(255,109,0,0.10)", "rgba(255,109,0,0.35)"),
        "BEAR":          ("🔴", "#d50000", "rgba(213,0,0,0.12)",   "rgba(213,0,0,0.40)"),
    }
    r_label = regime.get("regime", "SIDEWAYS")
    r_score = regime.get("score", 5.0)
    r_icon, r_color, r_bg, r_border = _REGIME_STYLE.get(r_label, _REGIME_STYLE["SIDEWAYS"])

    # Build signal pills (max 5 for banner readability)
    r_signals = regime.get("signals", [])
    _pill_css  = "display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px 4px 2px 0"
    _bull_pill = f"background:rgba(0,200,83,0.15);color:#69f0ae;border:1px solid rgba(0,200,83,0.3)"
    _bear_pill = f"background:rgba(213,0,0,0.15);color:#ff5252;border:1px solid rgba(213,0,0,0.3)"
    _neut_pill = f"background:rgba(100,100,100,0.15);color:#aaa;border:1px solid rgba(100,100,100,0.3)"
    pills_html = " ".join(
        f"<span style='{_pill_css};{_bull_pill if b=='bull' else _bear_pill if b=='bear' else _neut_pill}'>"
        f"{txt}</span>"
        for b, txt in r_signals[:5]
    )

    # Regime guidance text — sourced from analytics (VIX-differentiated for BEAR)
    guidance = regime.get("banner_guidance", regime.get("invest_caption", ""))

    # Score bar
    score_bar_pct = int(r_score / 10 * 100)
    _score_grad = (
        "linear-gradient(90deg,#d50000,#ff9100)" if r_score < 4.0 else
        "linear-gradient(90deg,#ff9100,#ffd600)" if r_score < 5.5 else
        "linear-gradient(90deg,#ffd600,#00c853)"
    )

    _vix_val = regime.get("vix")
    _fii_val = regime.get("fii_5d_cr")
    _hmm     = regime.get("hmm_state", "—")
    _ema20   = regime.get("nifty_vs_ema20", "—")
    _ema_x   = regime.get("nifty_vs_ema50", "—")   # now = EMA20 vs EMA50 cross
    _n1m     = regime.get("nifty_1m_pct")

    # ── Volume Spike context ──────────────────────────────────────────────────
    # High Volume Spike fraction = many sectors showing speculative trading
    # (turnover surged but delivery % FELL below norm).  This is a structural
    # bearish signal: institutions are trading actively WITHOUT conviction.
    # In a downtrend context it often indicates distribution / churning.
    n_total     = len(rot)
    n_vol_spike = int((rot["signal"] == "📊 Volume Spike").sum())
    vs_frac     = n_vol_spike / n_total if n_total else 0.0
    vs_pill     = ""
    if vs_frac >= 0.35:     # ≥35% of sectors = structural warning
        vs_pill = (
            f"<span style='{_pill_css};background:rgba(255,214,0,0.15);"
            f"color:#ffd600;border:1px solid rgba(255,214,0,0.3)'>"
            f"📊 {n_vol_spike}/{n_total} Vol Spikes — speculative, no conviction</span>"
        )

    vix_str  = f"VIX {_vix_val:.1f}" if _vix_val is not None else ""
    fii_str  = (f"FII all-idx 5D ₹{_fii_val:+,.0f} Cr" if _fii_val is not None else "")
    hmm_str  = f"HMM {_hmm}" if _hmm and _hmm != "—" else ""
    ema_str  = f"Nifty {_ema20} 20D EMA" if _ema20 != "—" else ""
    cross_str = (
        "Golden Cross ✓" if _ema_x == "ABOVE" else
        "Death Cross ✗"  if _ema_x == "BELOW" else ""
    )
    n1m_str  = f"Nifty 1M {_n1m:+.1f}%" if _n1m is not None else ""
    vs_str   = f"Vol Spikes {n_vol_spike}/{n_total} sectors" if vs_frac >= 0.25 else ""
    meta_row = "  ·  ".join(x for x in [ema_str, cross_str, vix_str, fii_str, hmm_str, n1m_str, vs_str] if x)

    st.markdown(
        f"<div style='padding:12px 16px;border-radius:8px;margin-bottom:12px;"
        f"background:{r_bg};border:1px solid {r_border}'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>"
        f"<span style='font-size:15px;font-weight:800;color:{r_color}'>"
        f"{r_icon} MARKET REGIME: {r_label}</span>"
        f"<span style='font-size:12px;color:{r_color};font-weight:600'>"
        f"Score {r_score:.1f}/10</span></div>"
        f"<div style='background:rgba(0,0,0,0.2);border-radius:4px;height:5px;margin-bottom:8px'>"
        f"<div style='width:{score_bar_pct}%;background:{_score_grad};height:5px;border-radius:4px'></div></div>"
        f"<div style='margin-bottom:6px'>{pills_html}{vs_pill}</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.55);margin-bottom:4px'>{meta_row}</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.80);font-style:italic'>"
        f"⚑ {guidance}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _INVEST_SIGNALS   = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation", "👀 Early Accumulation"}
    _CAUTION_SIGNALS  = {"📊 Volume Spike"}
    _AVOID_SIGNALS    = {"⚠️ Distribution Trap", "❌ Active Selling", "📉 Weakening"}

    # Tradability gate: keep thin / single-name / illiquid baskets OUT of the
    # ranked invest/avoid lists — they are single-stock momentum dressed up as
    # sector rotation (e.g. "Oil & Gas" = 7 small-caps, top name 43% of turnover;
    # the real majors sit in "Energy"). They remain visible, flagged, in the full
    # Sector Reference below so nothing is hidden.
    if "is_thin" in rot.columns:
        _thin_mask = rot["is_thin"].fillna(False).astype(bool)
        _tradable  = rot[~_thin_mask].copy()
        _n_thin    = int(_thin_mask.sum())
    else:
        _tradable, _n_thin = rot.copy(), 0

    if _defense_mode:
        # Bear/Caution: show ONLY sectors that are ELIGIBLE for this market — the
        # ones that genuinely cushion a downtrend AND still have institutional flow
        # (bear_eligible). Everything else (amplifiers that fall MORE than the
        # market, and value-traps institutions are exiting) goes to the avoid
        # column. The best bear sector (e.g. Pharma) is often "Neutral" on the
        # momentum signal, so we rank by defense, not by accumulation label.
        _elig = (_tradable["bear_eligible"].fillna(False)
                 if "bear_eligible" in _tradable.columns else pd.Series(False, index=_tradable.index))
        entering = _tradable[_elig].sort_values("defense_score", ascending=False).copy()
        caution  = _tradable.iloc[0:0].copy()   # no separate "caution" tier in defense mode
        exiting  = _tradable[~_elig].sort_values("defense_score", ascending=True).head(8).copy()
    elif _accum_mode:
        # Sideways (accumulation) / Bottoming (reversal): rank by the scenario's
        # score. The buy column is the top sectors on quiet delivery accumulation
        # (and, for reversal, accumulation INTO weakness); the avoid column is the
        # weakest on that score (distribution / no flow).
        entering = _tradable.sort_values(_rank_col, ascending=False).head(8).copy()
        caution  = _tradable.iloc[0:0].copy()
        exiting  = _tradable.sort_values(_rank_col, ascending=True).head(6).copy()
    else:
        caution  = _tradable[_tradable["signal"].isin(_CAUTION_SIGNALS)].copy()
        entering = _tradable[_tradable["signal"].isin(_INVEST_SIGNALS)].copy()
        exiting  = _tradable[_tradable["signal"].isin(_AVOID_SIGNALS)].sort_values(_rank_col, ascending=True).copy()

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
                    "dv_ratio", "dv_ratio_5d", "z_score", "breadth", "price_1w", "action",
                    "is_thin", "thin_reason"]
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

            _is_thin = bool(row.get("is_thin", False))
            _thin_reason = str(row.get("thin_reason", "") or "")
            _thin_badge = (
                f"<span title='{_thin_reason}' style='margin-left:6px;font-size:10px;"
                f"color:#ff9100;border:1px solid rgba(255,145,0,0.4);border-radius:3px;"
                f"padding:0 4px'>🔒 thin</span>" if _is_thin else ""
            )

            rows_html += (
                f"<tr style='border-bottom:1px solid rgba(255,255,255,0.05)'>"
                f"<td style='padding:6px 10px;font-size:13px;font-weight:500'>{row.get('sector','')}{_thin_badge}</td>"
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

    # Per-stock filters — control every "View stocks in …" drill-down below.
    # Sector-level stats (top-3, dominance warning, conviction) always use the
    # full stock set; only the displayed list inside the expander is filtered.
    _fcol1, _fcol2, _fcol3, _fcol4 = st.columns(4)
    with _fcol1:
        deliv_threshold = st.slider(
            "Min stock Wtd Delivery % — filters the per-stock lists below",
            min_value=0, max_value=100, value=int(_MIN_STOCK_WTD_DELIV_PCT), step=1,
            key="rotation_stock_deliv_threshold",
            help="Hide stocks whose 7-day turnover-weighted delivery % is at or below this value "
                 "inside the 'View stocks in …' expanders. Sector-level stats (top-3 contributors, "
                 "single-stock dominance warning) still use the full stock set.",
        )
    with _fcol2:
        deliv_vs_100d_pct = st.slider(
            "Min 7D vs 100D excess % — filters the per-stock lists below",
            min_value=0, max_value=100, value=0, step=5,
            key="rotation_stock_deliv_vs_100d_pct",
            help=(
                "Show only stocks where the 7-day turnover-weighted delivery % is at least X% "
                "ABOVE the stock's own 100-trading-day average delivery %.\n\n"
                "Example — 10% filter: a stock with a 100D avg of 40% must show ≥ 44% recent "
                "delivery to appear. A stock with a 100D avg of 25% must show ≥ 27.5%.\n\n"
                "0 = no filter (all stocks above the base Wtd Deliv % threshold are shown).\n"
                "Set 10–20% to isolate stocks showing abnormally high recent delivery "
                "relative to their own norm — the strongest own-history conviction reads."
            ),
        )
    with _fcol3:
        min_price = float(st.number_input(
            "Min Price ₹ — filters the per-stock lists below",
            min_value=0, max_value=50_000, value=0, step=50,
            key="rotation_stock_min_price",
            help=(
                "Hide stocks priced BELOW this value in the drill-down lists.\n\n"
                "0 = no filter (all prices shown).\n\n"
                "Practical thresholds:\n"
                "  ₹50   — excludes micro-cap / penny stocks\n"
                "  ₹100  — minimum for liquid options strategies\n"
                "  ₹200  — typical retail lot affordability floor\n\n"
                "Stocks with no recent price data (LTP = 0) are excluded whenever "
                "this filter is above 0."
            ),
        ))
    with _fcol4:
        max_price = float(st.number_input(
            "Max Price ₹ — filters the per-stock lists below",
            min_value=0, max_value=200_000, value=0, step=100,
            key="rotation_stock_max_price",
            help=(
                "Hide stocks priced ABOVE this value in the drill-down lists.\n\n"
                "0 = no filter (no upper cap).\n\n"
                "Practical thresholds:\n"
                "  ₹500   — small-capital intraday traders\n"
                "  ₹2,000 — F&O-friendly range (lower margin requirement per lot)\n"
                "  ₹5,000 — mid-cap positional range\n\n"
                "Combine with Min Price to isolate a specific price band, e.g., "
                "₹200–₹2,000 for liquid, affordable swing trades."
            ),
        ))

    # ── F&O positioning filter — multi-select instrument / expiry / signal ────
    # Each signal maps to (instrument, token-found-in-label). The signal options
    # shown depend on which instruments are picked. The chosen signals AUTO-APPLY
    # to every selected expiry; within an expiry signals are OR'd, and the Any/All
    # toggle controls how expiries combine.
    # Each signal → (instrument, [label tokens to match — any], short caption tag).
    # Futures are granular (one code each). Options are DIRECTIONAL BUCKETS that also
    # catch single-sided labels (C.Buying, C.Writing, C.LE…) — verified: full coverage,
    # zero cross-bucket overlap, so no false matches.
    _SIG_MAP = {
        "🟢 Long Buildup (LB)":   ("Futures", ["LB"], "LB"),
        "🔴 Short Buildup (SB)":  ("Futures", ["SB"], "SB"),
        "🔵 Short Covering (SC)": ("Futures", ["SC"], "SC"),
        "🟠 Long Unwinding (LU)": ("Futures", ["LU"], "LU"),
        "🔥 Bullish (C.Buy / P.Wrt / C.SC)": ("Options", ["Bull", "C.Buying", "C.SC", "P.SC"], "Opt↑"),
        "❄️ Bearish (C.Wrt / P.Buy / C.LE)": ("Options", ["Bear", "C.Writing", "C.LE", "P.Buying"], "Opt↓"),
        "📊 Range (both written)":           ("Options", ["Range"], "Opt-Range"),
        "⚡ Vol Bet (both bought)":          ("Options", ["Vol"], "Opt-Vol"),
    }
    # ── Auto-apply model: signals apply to EVERY selected expiry ──────────────
    # Expiry box drives the filter; the plain (non-tagged) signal box says WHAT to
    # look for. fno_filter = {expiry: [(instrument, [tokens], short), ...]} built by
    # crossing every selected expiry with every selected signal.
    _gcol0, _gcol1, _gcol2, _gcol3 = st.columns([1, 1, 1, 2])
    with _gcol0:
        fno_universe = st.radio(
            "Stock universe — filters the per-stock lists below",
            ["All stocks", "F&O stocks only"],
            key="rotation_stock_universe",
            help="All stocks = every name in the sector. F&O stocks only = restrict "
                 "the 'View stocks in …' drill-down lists to NSE F&O underlyings "
                 "(stocks that have futures/options). Sector-level stats (top-3 "
                 "contributors, single-stock dominance warning) always use the full "
                 "stock set. The F&O instrument/expiry/signal filters on the right "
                 "further narrow this universe by positioning.",
        )
        fno_only = fno_universe.startswith("F&O")
    with _gcol1:
        fno_instruments = st.multiselect(
            "F&O filter — instrument", ["Futures", "Options"], key="rot_fno_instr",
            help="Pick one or both. Futures = price+OI buildup (LB/SB/SC/LU). "
                 "Options = call/put OI-premium bias. Empty = no F&O filter.")
    _off = not fno_instruments
    with _gcol2:
        fno_expiries = st.multiselect(
            "Expiry (one or more)", ["Near month", "Next month", "Far month"],
            key="rot_fno_exp", disabled=_off,
            help="Which expiries the signals must hold in. The Any/All toggle controls "
                 "how multiple expiries combine. Read 'Next' on expiry day (Near is mid-rollover).")
    with _gcol3:
        # Plain signal list (no expiry tag) — the chosen signals auto-apply to EVERY
        # expiry picked in the Expiry box. Options are filtered to the chosen instruments.
        _avail = [k for k, v in _SIG_MAP.items() if v[0] in fno_instruments]
        _sel = st.multiselect(
            "F&O signal", _avail, key="rot_fno_sig", disabled=_off,
            help="Pick one or more signals (e.g. Long Buildup + Short Covering). They apply "
                 "to every expiry selected on the left. Within an expiry, signals are OR'd.")
        fno_filter = {}
        for _e in fno_expiries:
            for _s in _sel:
                _instr, _toks, _short = _SIG_MAP[_s]
                if _instr in fno_instruments:
                    fno_filter.setdefault(_e, []).append((_instr, _toks, _short))
        # ── Cross-expiry combine mode ──────────────────────────────────────────
        # Enabled once 2+ expiries carry signals. With one expiry the choice is a
        # no-op (defaults to All so behaviour is unchanged).
        _multi_exp = len(fno_filter) >= 2
        _mode = st.radio(
            "Across expiries", ["Any expiry (OR)", "All expiries (AND)"],
            index=1, horizontal=True, key="rot_fno_mode",
            disabled=_off or not _multi_exp,
            help="ANY = a stock passes if it shows the signal in AT LEAST ONE selected "
                 "expiry (e.g. LB/SC in Near OR Next). ALL = it must show the signal in "
                 "EVERY selected expiry simultaneously (stricter — confirmation across months).")
        fno_match_all = _mode.startswith("All")

    # ── Futures OI-strength gate ──────────────────────────────────────────────
    # The number in each futures label (e.g. "🟢 LB +39%") is the day-over-day OI
    # change % on the SAME contract:  (today_OI − prev_OI) / prev_OI × 100.
    # This gate keeps only STRONG positioning moves (≥ X%) or nascent ones (≤ X%),
    # compared on |OI Δ%| so SC/LU (OI falling, shown negative) are judged on size.
    _fut_in_filter = "Futures" in fno_instruments
    _scol1, _scol2, _scol3 = st.columns([1, 1, 2])
    with _scol1:
        _oi_mode = st.radio(
            "Futures OI Δ% strength gate",
            ["Off", "≥ strong", "≤ weak"],
            horizontal=True, key="rot_fno_oi_mode", disabled=not _fut_in_filter,
            help="Gate the selected FUTURES signals (LB/SB/SC/LU) by the size of the "
                 "day-over-day OI change %.\n\n"
                 "• ≥ strong — keep only big positioning moves (high conviction)\n"
                 "• ≤ weak — isolate early / nascent moves\n\n"
                 "Compared on MAGNITUDE |OI Δ%|, so Short Covering / Long Unwinding "
                 "(OI falling, shown negative) are judged on absolute size. Applies "
                 "per expiry; options buckets are unaffected.")
    with _scol2:
        _oi_thr = float(st.number_input(
            "OI Δ% threshold", min_value=0.0, max_value=500.0, value=10.0, step=5.0,
            key="rot_fno_oi_thr",
            disabled=(not _fut_in_filter or _oi_mode == "Off"),
            help="The OI change % cutoff. Example: 10 with '≥ strong' keeps only "
                 "futures signals where open interest moved at least 10% vs the "
                 "previous trading day (same contract)."))
    fno_oi_op = (None if (not _fut_in_filter or _oi_mode == "Off")
                 else ("ge" if _oi_mode.startswith("≥") else "le"))
    fno_oi_threshold = _oi_thr

    fno_filter = fno_filter or None

    # ── Price-action filters — only the backtest-validated axes ────────────────
    # Filter audits (scripts/audit_rotation_filters.py + audit_rotation_filters_v2.py:
    # dual-panel walk-forward — DCM broad 372d/118.9k obs + Tradebot 4yr F&O OOS
    # 1018d/38k obs — horizons 5/10/15/20d, non-overlapping t, event-time decay,
    # regime/universe splits, MC null, net-of-cost, interactions):
    #   • 🚀 setup — the breakout edge BUILDS over 2-3 weeks on the broad universe:
    #     +1.5%/10d → +3.9%/20d rel (MC p 0.98/1.00, net of 0.3% cost +3.6%/20d),
    #     decay curve peaks ~day 18-20 (no fade). BUT dead/negative on large-cap
    #     F&O (4yr OOS f20 −1.0%, t −2.2 on the largest tercile) and NEGATIVE in
    #     BEAR (f10 −1.6%, win 27%) — the regime gate below is load-bearing.
    #   • 🧭 alignment — ConfirmUp plateaus ~day 13-15 (+1.4%/15d, robust both
    #     panels); FalsePop = dead money the full 20d (fade confirmed); spread is
    #     strongest in CHOP (+1.0..1.3pp), inverts only in thin BEAR samples.
    #   • 📈 efficient movers (ER≥0.25 = old Clean∪Volatile) — the ONE piece of the
    #     removed pa_class filter that survived at the 2-3wk horizon: per-date L/S
    #     spread +0.62pp/20d t=3.2 (DCM) and +1.03pp t=5.1 (4yr OOS), positive in
    #     BOTH halves and ALL THREE regimes, stacks on ConfirmUp (win 58.9%).
    #     Candle-anatomy half of pa_class stays context-only (zero value).
    #   • 🛡️ hide risky — gappy/high-vol merged (V 0.48). On breakout names the
    #     veto keeps the median trade (+1.3% vs +1.0%) but cuts the RIGHT tail
    #     (p90 +12% vs +41%) — risk control, costs mean return; warned below when
    #     stacked with 🚀.
    _scol1, _scol2, _scol3, _scol4 = st.columns([2, 2, 1, 1])
    with _scol1:
        pa_setups = st.multiselect(
            "🚀 Multi-timeframe setup — filters the per-stock lists below",
            list(BRK_STATES[:-1]), default=[], key="rotation_stock_pa_setup",
            help=(
                "Daily × Weekly price-action state (validated on the broad universe + a "
                "4-yr out-of-window audit). Empty = no filter.\n\n"
                "• 🚀 Breakout — tight-base 20d-high break WITH the weekly trend UP → the ONE "
                "validated edge, and it BUILDS over 2-3 weeks: +1.5%/10d → +3.9%/20d rel "
                "on the broad universe (peaks ~day 18-20, net of cost +3.6%/20d, MC "
                "p≈1.00). CAVEAT: broad/small-mid-cap effect — dead-to-NEGATIVE on "
                "large-cap F&O names, and NEGATIVE in a bear tape (f10 −1.6%, win 27%) — "
                "obey the regime line below.\n"
                "• ⚠️ False break — the SAME tight-base break but AGAINST a weekly "
                "downtrend → the textbook false breakout. Measured near-zero to negative: "
                "+0.5%/10d net win 48% on the broad universe, −0.6%/10d net win 41% on 4yr "
                "F&O (win 32% in bull large-caps). Split out so 🚀 isn't diluted — an "
                "AVOID, not a short.\n"
                "• ↗ Break (extended) — broke out with no prior coil → weaker follow-through\n"
                "• 💥 Breakdown-bounce — broke the 20d low, but such names BOUNCE here → "
                "watch for a reversal, NOT a short (validated out-of-window)\n"
                "• 🧊 Coiling — tight base, no break yet → the breakout watchlist"
            ),
        )
    with _scol2:
        pa_aligns = st.multiselect(
            "🧭 Daily×Weekly alignment",
            list(MTF_ALIGN), default=[], key="rotation_stock_pa_align",
            help=(
                "The always-on BACKDROP: every stock sits in one daily×weekly quadrant "
                "(vs 🚀 Setup, which is a rare structural EVENT). A daily up-move only "
                "tends to continue when the WEEKLY trend agrees.\n\n"
                "• ✅ Confirmed Up — daily-up + weekly-up (best win-rate; edge accrues to "
                "~day 13-15 then plateaus — a 2-3wk hold captures it)\n"
                "• ⚠️ False Pop — daily-up but weekly-DOWN → dead money for the full 20d "
                "ahead (audited both panels) — an avoid, not a short\n"
                "• 🔵 Pullback — daily-down inside a weekly uptrend (dip)\n"
                "• 🔻 Down-trend — both lower (aligned down)\n\n"
                "OVERLAP: a 🚀 Breakout is ALWAYS ✅ Confirmed Up, and a ⚠️ False break is "
                "ALWAYS ⚠️ False Pop (audit: 100% nested) — so those pairs don't stack. "
                "Use this to select the broad grind names that have NO setup, or the "
                "🔵/🔻 quadrants 🚀 doesn't cover. Empty = no filter."
            ),
        )
    with _scol3:
        pa_efficient = st.checkbox(
            "📈 Efficient", value=False, key="rotation_stock_pa_eff",
            help="Keep only EFFICIENT movers — Kaufman ER ≥ 0.25 over 60d (net move ÷ "
                 "path length; the old 📈 Clean + 🌊 Volatile Trend classes). The one "
                 "trend-quality gate that survived the dual-panel audit at the 2-3wk "
                 "horizon: +0.6pp/20d over the rest on the broad universe (t 3.2) and "
                 "+1.0pp (t 5.1) on 4yr of F&O history — positive in both halves and "
                 "in bull, chop AND bear. Clean movers keep moving; direction alone "
                 "doesn't.",
        )
    with _scol4:
        pa_hide_risky = st.checkbox(
            "🛡️ Hide risky", value=False, key="rotation_stock_pa_risky",
            help="Drop ⚡ gappy (opens >1% from prior close on ≥35% of days — "
                 "overnight/event risk) and 🔥 high-vol (avg daily range top "
                 "quartile, ATR% ≥ 4.5) names. The two flags catch largely the "
                 "same stocks (audit: Cramér's V 0.48) and each strongly predicts "
                 "its own FORWARD risk, so they act as one veto. Risk filter, not a "
                 "return signal: on 🚀 breakout names it keeps the median trade but "
                 "cuts the big right-tail winners (p90 +12% vs +41%/20d).",
        )
    if pa_hide_risky and BRK_BREAKOUT in (pa_setups or []):
        st.caption(
            "⚠️ 🛡️ + 🚀 stacked: the risky veto keeps the median breakout (+1.3% vs "
            "+1.0%/20d) but drops the lottery tail (p90 +12% vs +41%). Fine for risk "
            "control — just know the biggest winners live in the names it hides."
        )
    # Nested-pair guard — the setup axis (event) and alignment axis (state) share the
    # weekly leg, so two cells are 100% nested (audit: scripts/audit_setup_vs_alignment.py,
    # Cramér's V 0.14 overall = independent, but these pairs co-occur exactly). Stacking
    # them narrows nothing; the 🚀 setup is already the sharper cut (a breakout adds
    # +1.2pp/10d over a ConfirmUp grind, while ConfirmUp adds 0.0 over a breakout).
    _nested = []
    if BRK_BREAKOUT in (pa_setups or []) and MTF_CONFIRM_UP in (pa_aligns or []):
        _nested.append("🚀 Breakout ⊆ ✅ Confirmed Up")
    if BRK_FALSEBRK in (pa_setups or []) and MTF_FALSE_POP in (pa_aligns or []):
        _nested.append("⚠️ False break ⊆ ⚠️ False Pop")
    if _nested:
        st.caption(
            "ℹ️ Redundant stack: " + " · ".join(_nested) + " (100% nested). The 🧭 "
            "alignment pick isn't narrowing the list — it's already implied by the 🚀 "
            "setup. Drop it, or use 🧭 for the 🔵/🔻 quadrants or the no-setup grind names."
        )
    # Contradiction guard — some setup×alignment pairs can NEVER co-occur (structural, not
    # data: a 20d-high break is provably daily-up so 🚀/⚠️/↗ only live in the up quadrants;
    # a 20d-low break is daily-down so 💥 only lives in the down quadrants). Picking an
    # impossible pair returns an empty list with no explanation — warn instead. Map verified
    # on 4,753 live rows (0 off-quadrant). 🧊 Coiling / — span all quadrants (never conflict).
    _valid_aligns = {
        BRK_BREAKOUT: {MTF_CONFIRM_UP},
        BRK_FALSEBRK: {MTF_FALSE_POP},
        BRK_EXTENDED: {MTF_CONFIRM_UP, MTF_FALSE_POP},
        BRK_BOUNCE:   {MTF_PULLBACK, MTF_DOWN_ALGN},
        BRK_COILING:  set(MTF_ALIGN),
        BRK_NONE:     set(MTF_ALIGN),
    }
    if pa_setups and pa_aligns and not _nested:
        _picked_aligns = set(pa_aligns)
        _reachable = any(_valid_aligns.get(s, set(MTF_ALIGN)) & _picked_aligns for s in pa_setups)
        if not _reachable:
            st.caption(
                "🚫 Impossible combination: none of your 🚀 setup picks ever occur in the "
                "chosen 🧭 alignment quadrant — a 20d-high break is always daily-up (✅/⚠️ up "
                "quadrants), a 20d-low break always daily-down (🔵/🔻). This filter returns "
                "nothing. Match the quadrant to the setup, or use just one dropdown."
            )
    # One-or-both decision guide — the two axes nest differently per setup cell, so the
    # right answer isn't uniform (audit: scripts/audit_setup_vs_alignment.py). Spell it out.
    with st.expander("❓ Which should I pick? (full simple guide)"):
        st.markdown(
            "**Rule of thumb: most of the time, use just ONE box.** Leave the other on "
            "“Choose options”. Below is what every choice means and whether to use it.\n\n"

            "**① First box — 🚀 Setup** *(what just happened to the price)*\n\n"
            "| Choice | In plain words | Use it? |\n"
            "|---|---|---|\n"
            "| 🚀 **Breakout** | Jumped above its recent high after a quiet patch, and the bigger trend is up | ✅ **Best buys** — use alone |\n"
            "| ⚠️ **False break** | Same jump, but the bigger trend is DOWN — usually a trap | ❌ **Avoid** — this is your skip list |\n"
            "| ↗ **Break (extended)** | Broke higher, but was already running fast (no quiet patch first) — weaker | 🤔 Only with ✅ Confirmed Up |\n"
            "| 💥 **Breakdown-bounce** | Fell below its recent low — tends to bounce back, so NOT a short | 👀 Watch only |\n"
            "| 🧊 **Coiling** | Very quiet and tight, no jump yet — often just before a move | 👀 Watchlist — add ✅ Confirmed Up |\n\n"

            "**② Second box — 🧭 Alignment** *(the background trend)*\n\n"
            "| Choice | In plain words | Use it? |\n"
            "|---|---|---|\n"
            "| ✅ **Confirmed Up** | Up on the day AND up on the bigger trend — healthy uptrend | ✅ **Best background** |\n"
            "| ⚠️ **False Pop** | Up on the day but the bigger trend is down — fake strength | ❌ Avoid |\n"
            "| 🔵 **Pullback** | Down on the day but the bigger trend is still up — a dip | 🤔 Dip-buyers only |\n"
            "| 🔻 **Down-trend** | Down on the day AND down on the bigger trend — weak | ❌ Avoid for buying |\n"
            "| • **Neutral** | No clear direction | — Skip |\n\n"

            "**③ Do I need BOTH boxes?** *(only matters if you picked something in box ①)*\n\n"
            "| If box ① is… | Second box? |\n"
            "|---|---|\n"
            "| 🚀 Breakout | ❌ No — leave 2nd box empty (it changes nothing) |\n"
            "| ⚠️ False break | ❌ No — leave 2nd box empty |\n"
            "| ↗ Break (extended) | ✅ Yes — add ✅ Confirmed Up to keep the good ones |\n"
            "| 💥 Breakdown-bounce | ➕ Optional — adds nothing tradable |\n"
            "| 🧊 Coiling | ✅ Yes — add ✅ Confirmed Up (coils inside an uptrend) |\n\n"

            "**④ Ready-made picks**\n"
            "- **Best buys right now** → box ①: 🚀 Breakout *(nothing else)*\n"
            "- **Solid uptrends not moving yet** → box ②: ✅ Confirmed Up *(nothing else)*\n"
            "- **Watchlist, about to pop** → box ①: 🧊 Coiling  +  box ②: ✅ Confirmed Up\n"
            "- **Dip-buys in an uptrend** → box ②: 🔵 Pullback *(nothing else)*\n"
            "- **What to steer clear of** → ⚠️ False break (box ①) or ⚠️ False Pop (box ②)\n\n"

            "**⑤ Shorting / downtrends? — this tool has NO short signal**\n"
            "None of these choices is a sell-short trigger. We tested every down/weak "
            "state over 4 years and **none paid to short**:\n"
            "- 🔻 **Down-trend** — actually drifts UP a little (oversold bounce), so shorting loses.\n"
            "- 💥 **Breakdown-bounce** — a fresh low tends to BOUNCE, not keep falling.\n"
            "- ⚠️ **False break** / ⚠️ **False Pop** — weak, but not reliably down enough to short.\n\n"
            "So treat the down/weak choices as **“don’t buy / get out” flags, not short trades**: "
            "🔻 Down-trend and ⚠️ False Pop = avoid buying; 💥 Breakdown = watch for a bounce; "
            "⚠️ False break = a breakout to skip. To find weak names to EXIT or avoid, put "
            "🔻 Down-trend in box ②."
        )
    # ── Live regime gate on the breakout edge ──────────────────────────────────
    # The 4yr OOS audit showed the breakout edge is BULL-CONCENTRATED (win ~58% in
    # bull vs ~43% in chop, dead in bear). Rather than bury that in a tooltip, state
    # it against TODAY'S regime so a 🚀 is trusted only in the tape where it works.
    _rg_label = str(regime.get("regime", "—")) if isinstance(regime, dict) else "—"
    _rg_up = _rg_label in ("BULL", "CAUTIOUS BULL")
    _rg_mid = _rg_label == "SIDEWAYS"
    if _rg_up:
        _bo_state, _bo_col, _bo_msg = (
            "ACTIVE", "#26a69a",
            f"Nifty regime **{_rg_label}** — the breakout edge's home tape "
            "(backtest win ~58% in bull). Trust the 🚀 setups here.")
    elif _rg_mid:
        _bo_state, _bo_col, _bo_msg = (
            "MUTED", "#ff9100",
            f"Nifty regime **{_rg_label}** — breakouts are weak in chop "
            "(backtest win ~43%). Treat 🚀 as a watchlist, not a trigger.")
    else:
        _bo_state, _bo_col, _bo_msg = (
            "OFF", "#ef5350",
            f"Nifty regime **{_rg_label}** — breakouts go NEGATIVE in a bear tape "
            "(audit: −1.6%/10d rel, win 27%). Stand aside on 🚀 until the tape turns.")
    st.markdown(
        f"<div style='background:rgba(120,120,120,0.10);border-left:3px solid {_bo_col};"
        f"padding:5px 10px;border-radius:0 4px 4px 0;margin:2px 0 8px;font-size:12px'>"
        f"🚀 <b>Breakout edge: <span style='color:{_bo_col}'>{_bo_state}</span></b> — {_bo_msg}"
        f"</div>",
        unsafe_allow_html=True,
    )
    try:
        _pa_df = cached_price_action(selected_date, 60, min_turnover)
    except Exception as _pe:
        import logging as _log
        _log.getLogger(__name__).warning("price_action failed (non-fatal): %s", _pe)
        _pa_df = pd.DataFrame()
    pa_filter = (tuple(pa_setups), tuple(pa_aligns), bool(pa_hide_risky),
                 bool(pa_efficient))

    if _n_thin:
        st.caption(
            f"🔒 {_n_thin} thin / single-name / illiquid bucket(s) excluded from the "
            f"ranked lists below (e.g. a basket where one stock is >50% of turnover, "
            f"fewer than 5 liquid names, or <₹500 Cr total). They are single-stock "
            f"momentum, not sector rotation — still listed, flagged, in the Sector "
            f"Reference above."
        )

    col_enter, col_avoid = st.columns(2)

    _HIGH_CONV = 70

    with col_enter:
        st.markdown(f"### {regime.get('invest_label', '🟢 SECTORS TO INVEST')}")
        st.caption(regime.get("invest_caption", "All accumulation signals — highest score first."))
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
                _sector_card(row, selected_date, min_turnover, deliv_threshold, deliv_vs_100d_pct,
                             min_price=min_price, max_price=max_price,
                             regime_label=r_label, regime=regime, defense_mode=_defense_mode,
                             score_col=_rank_col, fno_filter=fno_filter,
                             fno_match_all=fno_match_all, fno_only=fno_only,
                             fno_oi_op=fno_oi_op, fno_oi_threshold=fno_oi_threshold,
                             price_action=_pa_df, pa_filter=pa_filter)

    with col_avoid:
        st.markdown(f"### {regime.get('avoid_label', '🔴 SECTORS TO AVOID / EXIT')}")
        st.caption(regime.get("avoid_caption",
            "Active distribution/selling signals first; then relative laggards — "
            "the weakest sectors by score when no genuine distribution exists."
        ))

        # Tier 1 — genuine distribution / selling (real institutional exit, red).
        if not exiting.empty:
            st.markdown(
                "<div style='font-size:11px;font-weight:600;color:#d50000;"
                "margin-bottom:4px'>ACTIVE DISTRIBUTION / SELLING</div>",
                unsafe_allow_html=True,
            )
            for _, row in exiting.iterrows():
                _sector_card(row, selected_date, min_turnover, deliv_threshold, deliv_vs_100d_pct,
                             min_price=min_price, max_price=max_price,
                             regime_label=r_label, regime=regime, defense_mode=_defense_mode,
                             score_col=_rank_col, fno_filter=fno_filter,
                             fno_match_all=fno_match_all, fno_only=fno_only,
                             fno_oi_op=fno_oi_op, fno_oi_threshold=fno_oi_threshold,
                             price_action=_pa_df, pa_filter=pa_filter)

        # Tier 2 — relative laggards: weakest sectors by score, excluding any
        # already shown in the invest / caution / distribution lists. The absolute
        # z<=-0.5 distribution gate cannot fire in a strong-delivery regime (z is
        # positive market-wide), so without this the column would be empty even
        # when clear underweight candidates exist. These are NOT active selling —
        # they are the relative weakest in today's cross-section.
        shown = set(entering["sector"]) | set(caution["sector"]) | set(exiting["sector"])
        laggards = _tradable[~_tradable["sector"].isin(shown)].nsmallest(5, "accum_score")

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
                _sector_card(row, selected_date, min_turnover, deliv_threshold, deliv_vs_100d_pct,
                             min_price=min_price, max_price=max_price,
                             regime_label=r_label, regime=regime, defense_mode=_defense_mode,
                             score_col=_rank_col, fno_filter=fno_filter,
                             fno_match_all=fno_match_all, fno_only=fno_only,
                             fno_oi_op=fno_oi_op, fno_oi_threshold=fno_oi_threshold,
                             price_action=_pa_df, pa_filter=pa_filter)

    if not caution.empty:
        st.markdown("---")
        st.markdown("### 📊 VOLUME SPIKE — DO NOT CONFUSE WITH ACCUMULATION")
        st.caption(
            "Z-Score is high (delivery VALUE surged) BUT delivery % is BELOW its 100D average. "
            "Speculative event-driven trading — not institutional conviction. "
            "Do not buy based on delivery value alone."
        )
        for _, row in caution.iterrows():
            _sector_card(row, selected_date, min_turnover, deliv_threshold, deliv_vs_100d_pct,
                         min_price=min_price, max_price=max_price,
                         regime_label=r_label, regime=regime, defense_mode=_defense_mode,
                         score_col=_rank_col, fno_filter=fno_filter,
                         fno_match_all=fno_match_all, fno_only=fno_only,
                         fno_oi_op=fno_oi_op, fno_oi_threshold=fno_oi_threshold)

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
                "sector":         _htc("Sector"),
                "signal":         _htc("Signal"),
                "accum_score":    _hpc(
                    "Score", max_value=100, format="%.0f",
                    help="Score = 30% RS vs Nifty + 25% 5D Avg DV + 15% DV Today + 15% Breadth + 15% Z-Score\n"
                         "Cross-sectional rank: ranks sectors relative to each other on today's data"),
                "coverage":       _htc(
                    "Coverage",
                    help="Swing (3–15 days): Z-Score ≥ 2σ + Breadth ≥ 50%\n"
                         "Positional (4–8 weeks): DV Ratio > 1.2 + positive slope + Breadth ≥ 40%\n"
                         "Mid Term (3–4 months): steep 100-day slope + DV Ratio > 1.3 + Breadth ≥ 50%"),
                "horizon":        _htc("Horizon"),
                "dv_ratio":       _hnc(
                    "DV Today", format="%.2f×",
                    help="Today's delivered value ÷ own 100D daily average\n"
                         "1.0× = exactly average  |  1.5× = 50% above norm\n"
                         "Single-day snapshot — can spike from one large block trade"),
                "dv_ratio_5d":    _hnc(
                    "5D Avg DV", format="%.2f×",
                    help="5-day average DV ratio = (1W delivery ÷ 5) ÷ (100D delivery ÷ 100)\n"
                         "1.0× = exactly normal  |  1.3× = 30% above weekly average\n"
                         "Primary signal driver — smooths single-day noise over a week"),
                "z_score":        _hnc(
                    "Z-Score (σ)", format="%+.2f",
                    help="(Today's DV − 100D mean) ÷ 100D std deviation\n"
                         "Z ≥ 2.0 = top 2.5% of days  |  Z ≥ 1.0 = top 16%  |  Z ≤ -0.5 = below normal\n"
                         "Statistically grounded — adapts to each sector's own delivery volatility"),
                "breadth":        _hnc(
                    "Breadth", format="%.0f%%",
                    help="% of stocks in sector where today's delivery > own 100D avg daily DV\n"
                         "70%+ = broad institutional participation\n"
                         "30% or below = one large-cap driving the sector signal"),
                "trend_slope":    _hnc(
                    "Trend Slope", format="%+.3f",
                    help="Linear regression slope of 100-day delivery % series\n"
                         "Normalised by mean — % change per trading day\n"
                         "Positive = delivery trend rising  |  Negative = delivery trend falling"),
                "price_1w":       _hnc(
                    "1W Price%", format="%+.2f%%",
                    help="Cumulative 1W price return: (today_close − 5D_ago_close) / 5D_ago_close × 100"),
                "price_1m":       _hnc("1M Price%",  format="%+.2f%%"),
                "price_3m":       _hnc("3M Price%",  format="%+.2f%%"),
                "today_dv_cr":    _hnc(
                    "Today DV (₹ Cr)", format="₹%.1f",
                    help="Today's single-day delivered value in ₹ Crores\n"
                         "Absolute size of today's institutional activity"),
                "deliv_val_1w_cr":_hnc(
                    "1W Deliv Val (₹ Cr)", format="₹%.1f",
                    help="₹ value of shares delivered in last 1 week — total institutional conviction"),
                "today_wtd_deliv_pct": _hnc(
                    "Today Del%", format="%.1f%%",
                    help="Today's turnover-weighted delivery %\n\n"
                         "Conviction quality check: if this is BELOW the 100D avg, a high Z-Score\n"
                         "is a Volume Spike (speculative), not institutional accumulation.\n"
                         "Formula: Σ(deliv_per × turnover_lacs) / Σ(turnover_lacs)"),
                "avg_wtd_deliv_pct_100d": _hnc(
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
                "sector":      _htc("Sector"),
                "signal":      _htc("Signal"),
                "accum_score": _hnc("Score", format="%.1f"),
                rs_col:        _hnc(f"RS {period_label} %", format="%+.2f%%"),
                "z_score":     _hnc("Z-Score",  format="%+.2f"),
                "dv_ratio":    _hnc("DV Ratio", format="%.2f×"),
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

_TILT_STYLE = {
    "OVERWEIGHT":  ("🟢", POSITIVE_COLOR, "BUY list — rotate in"),
    "UNDERWEIGHT": ("🔴", NEGATIVE_COLOR, "AVOID / trim"),
    "WATCH":       ("🟡", "#d9a441",       "WAIT — not moved yet"),
    "NEUTRAL":     ("⚪", "#8a8f98",       "no clear signal"),
}
# plain-English hover tooltips — "buy or avoid, and which?"
_TILT_HELP = {
    "OVERWEIGHT": ("BUY CANDIDATES. Strongest sectors — money and price are flowing in faster "
                   "than the market. In a green backdrop, these are the ones to buy / add for the "
                   "next 1-2 weeks. It's a lean (~55-60% right over many trades), not a sure thing."),
    "UNDERWEIGHT": ("AVOID. Weakest sectors — money is leaving. Don't buy these; trim if you hold "
                    "them. NOT a short signal (you can't cheaply short a whole sector)."),
    "WATCH": ("WAIT, don't buy yet. Quiet buying is showing up but the price hasn't turned. Buy "
              "only once it strengthens into the green BUY list — early entries here often just sit."),
    "NEUTRAL": ("SKIP. Middle of the pack — no clear buy or avoid edge right now."),
}


def _render_operator_footprint(selected_date: date) -> None:
    """
    Unusual single-stock F&O positioning — where size is being built, and what
    the OI/price combination says it was.

    DESCRIPTIVE BY DESIGN. scripts/backtest_operator_footprint.py walked 105k
    symbol-days over 500 sessions and found no forward edge, so this tab reports
    what is happening and refuses to imply what happens next.
    """
    st.markdown("#### 🕵️ Operator Footprint — unusual F&O positioning")
    st.caption(
        "Someone building real size in a stock leaves a trace: open interest "
        "appearing where it normally does not, at a size that is abnormal **for "
        "that stock**, with a price direction that says whether it was bought or "
        "written. This tab finds those traces across every F&O stock."
    )

    st.warning(
        "**This is a description, not a forecast — and that is a measured "
        "statement, not caution.** A 105,000 symbol-day backtest over 500 "
        "sessions (2024-07 → 2026-08) found **no forward edge**: every "
        "directional information coefficient sat inside noise (t −0.80 to +0.97), "
        "and once benchmarked correctly every basket collapsed to ~0 alongside a "
        "random control. Call BUYING and call WRITING — which imply opposite "
        "directions — scored **identically** (+0.337 vs +0.334), the signature of "
        "a shared exposure rather than information.\n\n"
        "**Scope of that test, stated precisely.** It measured the INGREDIENTS — "
        "call-buying share, put-writing share, ITM buildup, futures OI — not the "
        "composite ranking this tab now shows, which was rebuilt afterwards "
        "(per-moneyness event bars, money-in-bucket-units ranking). No directional "
        "claim is made for the new ranking either; it has simply not been tested, "
        "and nothing here should be read as one.\n\n"
        "**And there is a reason for that.** An option's premium moves with the "
        "underlying, so the BUYING / WRITING label is mostly restating the day's "
        "price direction rather than reading order flow — measured, a call's "
        "premium moved with spot **83%** of the time and a put's against spot "
        "**94%**. Separating real demand needs the premium move net of delta, "
        "which this dataset does not carry. Use this to see WHERE size showed up, "
        "not to predict the stock. `scripts/backtest_operator_footprint.py`",
        icon="⚠️")

    c1, c2 = st.columns([1, 2])
    with c1:
        min_cr = st.slider("Minimum strike size (₹ Cr notional)", 1.0, 50.0, 5.0, 1.0,
                           key="opf_min_cr",
                           help="Ignore strikes smaller than this. Percentage jumps "
                                "in a tiny strike are noise, not a footprint.")
        # Price band. A linear slider is useless here — the F&O universe runs from
        # ~Rs 20 to Rs 43,000+, so 95% of names would sit in the first 5% of the
        # track. These breakpoints are roughly log-spaced so every decade gets
        # usable travel.
        _STEPS = [0, 50, 100, 250, 500, 1000, 2000, 3000, 5000,
                  7500, 10000, 20000, 50000, 1000000]
        # Labels MUST be unique. select_slider resolves the selection by its
        # formatted label, so mapping both 0 and 1000000 to "any" collapsed the two
        # ends onto one option — both handles returned 1000000, the filter became
        # between(1e6, 1e6), and the tab showed "no stock between Rs 1,000,000 and
        # Rs 1,000,000" while 16 stocks actually qualified.
        _lo, _hi = st.select_slider(
            "Stock price band (₹)", options=_STEPS, value=(0, 1000000),
            format_func=lambda v: ("no min" if v == 0
                                   else "no max" if v == 1000000 else f"{v:,}"),
            key="opf_price_v2",
            help="Filter by the stock's own price. The F&O list spans about Rs 20 "
                 "to Rs 43,000, and a Rs 40,000 stock is not actionable for most "
                 "position sizes.\n\n"
                 "The filter is applied BEFORE the top-25 cut, so you get the 25 "
                 "strongest footprints WITHIN your band — not the overall top 25 "
                 "trimmed down to whatever survives.")
    with c2:
        st.caption(
            "Ranked by **money that arrived today**, compared with what counts as "
            "a lot *for that kind of strike*. At-the-money strikes are always busy, "
            "in-the-money ones rarely are, so each is judged on its own scale — "
            "otherwise the at-the-money names would fill the whole list.\n\n"
            "In-the-money strikes count for more: everyday trading happens "
            "out-of-the-money, so big money showing up in-the-money is the harder "
            "thing to explain.\n\n"
            "Open interest that is just *sitting* there scores nothing. The "
            "question is what changed today.")

    try:
        rep = cached_operator_footprint(selected_date, float(min_cr))
    except Exception as exc:                                    # noqa: BLE001
        st.error(f"Footprint unavailable: {exc}")
        return
    if not rep.get("ok"):
        st.info(rep.get("error", "No F&O data for that date."))
        return

    meta = rep["meta"]
    if rep["as_of"] != selected_date:
        st.caption(f"↪ using the last F&O session on or before {selected_date}: "
                   f"**{rep['as_of']}**")
    if not meta.get("has_norm"):
        st.caption("⚠️ Not enough prior expiry history to normalise — showing raw "
                   "buildup only.")

    stocks = rep["stocks"]
    if stocks is None or stocks.empty:
        st.info("No strike cleared the size filter on this date.")
        return

    # Only stocks that actually cleared the event bar. Without this the table pads
    # itself to 25 rows with zero-footprint names — at a Rs 10 Cr strike filter only
    # 16 stocks qualified, so 9 rows would have been stocks where nothing happened.
    if "footprint" in stocks.columns:
        stocks = stocks[stocks["footprint"] > 0]
    if stocks.empty:
        st.info("No stock cleared the event bar on this date. Lower the strike-size "
                "filter to widen the search.")
        return

    # price band first, THEN the top-25 cut (see the slider's help text)
    _n_all = len(stocks)
    if "spot" in stocks.columns:
        _band = stocks["spot"].between(_lo, _hi) | stocks["spot"].isna()
        stocks = stocks[_band]
    stocks = stocks.head(25)
    if stocks.empty:
        st.info(f"None of the {_n_all} stocks with a footprint on this date are "
                f"priced between ₹{_lo:,} and ₹{_hi:,}. Widen the price band.")
        return

    _band_txt = ("" if (_lo, _hi) == (0, 1000000)
                 else f" · priced ₹{_lo:,}–₹{_hi:,} "
                      f"({len(stocks)} of {_n_all} stocks match)")
    st.markdown(f"##### Where size showed up on {rep['as_of']:%d %b %Y}{_band_txt}")
    # Never suppress a name silently — say which and why.
    _ca = rep.get("meta", {}).get("corp_action_symbols") or []
    _nl = rep.get("meta", {}).get("new_listing_symbols") or []
    if _ca:
        st.caption(
            f"⚠️ **Excluded today: {', '.join(_ca)}** — NSE re-priced the whole "
            "strike ladder (ex-dividend, bonus or split), so every position looks "
            "like it arrived this morning when it is the same money under a new "
            "contract name. Flow is not measurable for these on such a day. "
            "Happens roughly 66 times a year across the universe, mostly in "
            "dividend season.")
    if _nl:
        st.caption(
            f"⚠️ **Excluded today: {', '.join(_nl)}** — listed in F&O within the "
            "last 30 days. Every strike is new because the contract itself is new, "
            "so there is no positioning history to be abnormal against.")
    show = stocks.copy()
    show["Spot"] = show["spot"].round(2)
    show["Biggest strike"] = (show["top_type"] + " " +
                              show["top_strike"].round(1).astype(str))
    show["₹ Cr added"] = show["top_add_cr"].round(0)
    show["vs own normal"] = show["top_add_vs_norm"].round(1)
    show["₹ Cr"] = show["top_notional_cr"].round(0)
    # A lone "call share %" forces the reader to work out the put side and to hold
    # a 50% threshold in their head. Name the side that actually won instead.
    def _side_txt(v):
        if v != v:
            return "—"
        return ("CALLS %.0f%%" % v if v >= 55 else
                "PUTS %.0f%%" % (100 - v) if v <= 45 else
                "even %.0f/%.0f" % (v, 100 - v))
    show["New OI went to"] = [_side_txt(v) for v in show["call_share_of_adds_pct"]]
    # Spell the direction out next to the act. "PUT WRITING" is bullish and
    # "CALL WRITING" is bearish — not obvious at a glance, and getting it backwards
    # inverts the whole read of the row.
    from src.analytics.operator_footprint import ACTION_LEAN as _LEAN
    show["What happened"] = [
        (f"{a} ({_LEAN[a]})" if a in _LEAN else a) + (" · ROLLED" if rl else "")
        for a, rl in zip(show["top_action"], show["top_is_roll"])]
    cols = {"symbol": "Stock", "Spot": "Spot", "Biggest strike": "Biggest strike",
            "top_moneyness": "Where", "What happened": "What happened",
            "₹ Cr added": "₹ Cr added", "vs own normal": "vs own normal",
            "₹ Cr": "₹ Cr",
            "New OI went to": "New OI went to",
            "fut_action": "Futures", "n_unusual": "Unusual strikes"}
    have = [c for c in cols if c in show.columns]
    # Explicit height so every row renders without a NESTED scrollbar. Streamlit
    # caps a dataframe at ~10 rows by default and scrolls internally; the mouse
    # wheel then scrolls the PAGE instead, leaving rows 11-25 unreachable.
    st.dataframe(
        show[have].rename(columns=cols), hide_index=True, use_container_width=True,
        height=int(min(len(show), 25)) * 35 + 45,
        column_config={
            "Stock": _htc(
                help="Pick this name in the 'Strike-by-strike detail' box below to "
                     "see every strike behind the row."),
            "Spot": _hnc(
                format="%.2f", help="The stock's closing price on this date."),
            "Biggest strike": _htc(
                help="The single strike where the most notable money arrived today. "
                     "CE = call, PE = put. Every other column on this row that says "
                     "'strike' refers to this one."),
            "Where": _htc(
                help="Where that strike sits against the current price.\n\n"
                     "**ITM** (in the money) — already has real value: a call below "
                     "the price, or a put above it.\n"
                     "**ATM** (at the money) — within 2% of the price.\n"
                     "**OTM** (out of the money) — only pays off if the stock moves "
                     "further.\n\n"
                     "Most everyday trading is OTM, so big money appearing ITM is "
                     "the more unusual event and is weighted higher here."),
            "vs own normal": _hnc(
                format="%.1fx",
                help="How big today's build is next to a normal day's build at this "
                     "kind of strike. 3x = three times normal.\n\n"
                     "'Normal' comes from the last 3 expiry cycles that have already "
                     "finished.\n\n"
                     "Blank = this strike is normally too quiet to have a reliable "
                     "normal, so no multiple is shown and the row is ranked on money "
                     "alone."),
            "What happened": _htc(
                help="What was done to the option, and what it means for the STOCK "
                     "(in brackets).\n\n"
                     "Open interest up + premium up = someone BOUGHT it.\n"
                     "Open interest up + premium down = someone SOLD (wrote) it.\n\n"
                     "**Bullish:** CALL BUYING, PUT WRITING.\n"
                     "**Bearish:** PUT BUYING, CALL WRITING.\n\n"
                     "Writing a put is bullish — the seller keeps the premium as "
                     "long as the stock stays above the strike.\n\n"
                     "**· ROLLED** = an old position moved into a later expiry, not "
                     "new conviction. Mostly seen in expiry week.\n\n"
                     "**NEW STRIKE** = listed today, so there is no earlier premium "
                     "to compare and we cannot say bought or sold.\n\n"
                     "⚠️ Treat bought-vs-sold as a hint, not a fact. An option's "
                     "premium mostly follows the stock, so on an up day almost any "
                     "call with rising open interest looks 'bought'. The ₹ Cr added "
                     "is the number to trust."),
            "New OI went to": _htc(
                help="Which side today's new open interest went to, across ALL "
                     "strikes and expiries of this stock. 'CALLS 72%' means 72% of "
                     "the new open interest was in calls and 28% in puts.\n\n"
                     "This is a whole-stock number — it does not describe the "
                     "'Biggest strike' next to it.\n\n"
                     "⚠️ Calls busy does NOT mean bullish. Those calls may be "
                     "getting SOLD, which is bearish. Check 'What happened' for the "
                     "side."),
            "Unusual strikes": _hnc(
                help="How many separate strikes in this stock saw big new money "
                     "today.\n\n"
                     "This is about SPREAD, not size. A low number means the money "
                     "landed in a few places, which is more interesting — it looks "
                     "like one deliberate position. A high number is usually just a "
                     "busy day."),
            "₹ Cr added": _hnc(
                format="%.0f",
                help="Rupee value of the new open interest that appeared TODAY in "
                     "that one strike. The most reliable number here — this money "
                     "really did arrive. The list is ranked on it."),
            "Futures": _htc(
                help="The same read on the near-month future. Here LONG BUILDUP and "
                     "SHORT BUILDUP mean what they normally do — long or short the "
                     "STOCK — because a future has no buying/selling asymmetry."),
            "₹ Cr": _hnc(
                format="%.0f", help="Total money already sitting at that strike, "
                                    "built up over time — not today's activity."),
        })

    st.markdown("##### Strike-by-strike detail")
    pick = st.selectbox("Stock", show["symbol"].tolist(), key="opf_symbol",
                        help="Every liquid strike for this stock, both expiries.")
    det = rep["strikes"]
    det = det[det["symbol"] == pick].copy()
    if det.empty:
        st.caption("No strikes above the size filter for this stock.")
    else:
        det = det.sort_values("footprint", ascending=False).head(20)
        det["Strike"] = det["option_type"] + " " + det["strike_price"].round(1).astype(str)
        det["Premium"] = det["close_price"].round(2)
        det["Prem chg"] = det["prem_chg"].round(2)
        det["OI"] = det["open_interest"].astype("int64")
        det["OI added"] = det["chg_in_oi"].astype("int64")
        # "vs normal" is the BUILD vs a normal day's build, not standing OI vs a
        # bucket median — the latter printed >=10x on 5.5% of all strike-days and
        # fired for RELIANCE on every session, so it never meant anything.
        from src.analytics.operator_footprint import ACTION_LEAN as _LEAN
        det["action"] = [(f"{a} ({_LEAN[a]})" if a in _LEAN else a)
                         + (" · ROLLED" if rl else "")
                         for a, rl in zip(det["action"], det["is_roll"])]
        det["₹ Cr added"] = det["add_cr"].round(0)
        det["vs normal"] = det["add_vs_norm"].round(1)
        det["₹ Cr"] = det["notional_cr"].round(0)
        det["% of book"] = det["book_share_pct"].round(1)
        # Expiry rendered as "2026-08-25 00:00:00" — a midnight timestamp on a
        # contract that has no time component. Format it as a date.
        det["Expiry"] = pd.to_datetime(det["expiry_date"]).dt.strftime("%d %b %Y")
        st.dataframe(
            det[["Expiry", "Strike", "moneyness", "action", "Premium",
                 "Prem chg", "OI", "OI added", "₹ Cr added", "vs normal",
                 "₹ Cr", "% of book"]]
            .rename(columns={"moneyness": "Where", "action": "What happened"}),
            hide_index=True, use_container_width=True,
            height=int(min(len(det), 20)) * 35 + 45,
            column_config={
                "Expiry": _htc(
                    help="When this contract settles. Stock options are monthly, so "
                         "up to three are live at once — near, next and far."),
                "Strike": _htc(
                    help="CE = call, PE = put, followed by the strike price."),
                "Where": _htc(
                    help="Where the strike sits against the current price.\n\n"
                         "**ITM** (in the money) — already has real value: a call "
                         "below the price, or a put above it.\n"
                         "**ATM** (at the money) — within 2% of the price.\n"
                         "**OTM** (out of the money) — only pays off if the stock "
                         "moves further.\n\n"
                         "Most everyday trading is OTM, so big money appearing ITM "
                         "is the more unusual event."),
                "What happened": _htc(
                    help="What was done to this option, and what it means for the "
                         "STOCK (in brackets).\n\n"
                         "**Bullish:** CALL BUYING, PUT WRITING.\n"
                         "**Bearish:** PUT BUYING, CALL WRITING.\n\n"
                         "⚠️ Bought-vs-sold is a hint, not a fact — an option's "
                         "premium mostly just follows the stock."),
                "Premium": _hnc(
                    format="%.2f", help="What one unit of this option costs today."),
                "Prem chg": _hnc(
                    format="%.2f",
                    help="How much the premium moved today. Its direction is what "
                         "decides the bought-vs-sold label."),
                "OI": _hnc(
                    format="%,d",
                    help="Open interest: how many shares' worth of this option are "
                         "currently held open. Total size, built up over time."),
                "OI added": _hnc(
                    format="%,d",
                    help="How much open interest appeared TODAY, in shares."),
                "₹ Cr added": _hnc(
                    format="%.0f",
                    help="The same as 'OI added' but in rupees. **The most reliable "
                         "number here** — this money really did arrive today, and "
                         "the ranking is built on it."),
                "vs normal": _hnc(
                    format="%.1fx",
                    help="How big today's build is next to a normal day's build at "
                         "this kind of strike. 3x = three times normal.\n\n"
                         "Blank means this strike is normally too quiet to have a "
                         "reliable normal, so no multiple is shown."),
                "₹ Cr": _hnc(
                    format="%.0f",
                    help="Total money sitting at this strike, built up over time — "
                         "not today's activity."),
                "% of book": _hnc(
                    format="%.1f%%",
                    help="This strike's share of ALL the open interest in this "
                         "stock's options. A high number means the stock's "
                         "positioning is concentrated right here."),
            })
        st.caption(
            "**How to read it:** bullish = `CALL BUYING` or `PUT WRITING`. "
            "Bearish = `PUT BUYING` or `CALL WRITING`. Selling a put is bullish — "
            "the seller keeps the premium while the stock holds above the strike.\n\n"
            "⚠️ **Bought-vs-sold is a hint, not a fact.** It is worked out from the "
            "premium's direction, and a premium mostly just follows the stock — so "
            "on a day the stock rallied, `PUT WRITING` is close to simply repeating "
            "that the stock rallied. **Trust the ₹ Cr added**: that money really "
            "arrived. Treat the side as a lean.")

    st.caption(
        "**This list leans towards CALM stocks, not busy ones.** A quiet stock has "
        "a steady baseline, so anything unusual stands out; a wild stock is noisy "
        "enough that nothing looks unusual. Big money in a quiet stock is arguably "
        "the more interesting event — but 'where size showed up' is not the same "
        "as 'where the action is'.\n\n"
        f"Scanned {meta['n_symbols']} F&O stocks · {meta['n_strikes_liquid']:,} "
        f"strikes above ₹{meta['min_notional_cr']:.0f} Cr · compared with the last "
        f"{meta['prior_cycles']} finished expiry cycles at the same point in the "
        f"cycle and the same distance from the price. Stock options are monthly, so "
        f"3 cycles ≈ 3 months. F&O data starts 2024-07-24.")


def _render_clock_replay(selected_date: date, min_turnover: float,
                         window: int, window_label: str) -> None:
    from src.analytics.sector_rotation import _CLOCK_ACTION
    """
    Results panel for the Rotation Clock — what each phase said, and what followed.

    Scored SYMMETRICALLY: an N-session clock is judged over the next N sessions.
    Phases are not all calls, so they are not all ticked the same way:
      Leading   buy-like  -> beating the benchmark is the win
      Weakening } avoid    -> LAGGING the benchmark is the win
      Lagging   }
      Improving WATCH, never a buy on this tab, and measured negative at both
                ends of the horizon range - shown, never scored as a buy
      Neutral   no call    -> context only
    """
    st.markdown("##### 📊 Results — what the clock said, and what followed")

    _c1, _c2 = st.columns([1, 2])
    with _c1:
        _as_of = st.date_input(
            "Signal date", value=selected_date - timedelta(days=90),
            max_value=selected_date, key="clock_replay_date",
            help="The day you want to check. Pick a weekend or a holiday and it "
                 "moves back to the last day the market was open.")
    with _c2:
        st.caption(
            f"Scored over the **same length** as the analysis period you picked "
            f"above — the {window_label} clock, judged over the next {window} "
            f"sessions. Change the period to re-run both the phases and the "
            f"scoring window together.")

    try:
        rep = cached_clock_replay(_as_of, int(window), float(min_turnover), selected_date)
    except Exception as exc:                                    # noqa: BLE001
        st.error(f"Replay unavailable: {exc}")
        return
    if not rep.get("ok"):
        st.info(rep.get("error", "Replay unavailable for that date."))
        return

    anchor = rep["anchor"]
    if rep.get("snapped"):
        st.caption(f"↪ {_as_of} was not a trading day — using **{anchor}**.")

    _bm = st.radio("Judge each sector against", ["Nifty 50", "All-sector basket"],
                   index=0, horizontal=True, key="clock_replay_bench",
                   help="Which yardstick the ✅/❌ marks use.\n\n"
                        "Nifty 50 — could you have just bought the index instead? "
                        "The practical test.\n\n"
                        "All-sector basket — did these phases beat owning every "
                        "sector? The fair test of the sorting itself.")
    _use_nifty = _bm == "Nifty 50"

    _open = rep["horizon"].get("status") == "OPEN"
    _src = rep["horizon"] if not _open else rep.get("to_today", {})
    _ready = _src.get("status") == "DONE"
    if not _ready:
        st.info("No forward data for this date yet.")
        return

    _bask, _nif = _src["basket_abs"], _src["nifty_abs"]
    _fell = _use_nifty and (_nif is None or _nif != _nif)
    _base = _bask if (not _use_nifty or _fell) else _nif
    _lbl = "all-sector basket" if (not _use_nifty or _fell) else "Nifty 50"
    _ps = _src["per_sector"]

    if _open:
        _ela, _rem = rep["sessions_elapsed"], rep["sessions_remaining"]
        st.markdown(f"#### ⏳ Running — {_ela} of {window} sessions done, {_rem} to go"
                    f" · as at the close of {rep['today']:%d %b %Y}")
        st.caption(
            f"The {window_label} clock read on **{anchor:%d %b %Y}** has not "
            f"finished its window, so these are **running totals to the last "
            f"close**. They show ▲ahead / ▼behind versus **{_lbl}** "
            f"({_base:+.2f}% so far) rather than ✅/❌ — nothing is scored until "
            f"the window closes.")
    else:
        _he = rep["horizon_end"]
        st.markdown(f"#### 📅 Result at the close of {_he:%d %b %Y} ({_he:%a})")
        _other = _bask if _use_nifty else _nif
        st.caption(
            f"Clock read **{anchor:%d %b %Y}** at the close, scored to the close of "
            f"**{_he:%d %b %Y}** — {_src['sessions']} sessions. Marks judge each "
            f"sector against **{_lbl}** ({_base:+.2f}%)"
            + (f"; the other benchmark returned {_other:+.2f}%."
               if _other is not None and _other == _other else ".")
            + " For AVOID phases, lagging the benchmark is the win.")

    _ICON = {"Leading": "💰", "Improving": "🔍", "Neutral": "⚖️",
             "Weakening": "⚠️", "Lagging": "📤"}
    for phase in ("Leading", "Improving", "Weakening", "Lagging", "Neutral"):
        names = rep["phases"].get(phase, [])
        if not names:
            continue
        act, want_beat, why = _CLOCK_ACTION[phase]
        pd_ = _src["per_phase"][phase]
        head = (f"{_ICON[phase]} **{phase}** · {act} · {pd_['n']} sector"
                f"{'s' if pd_['n'] != 1 else ''} — _{why}_")
        if pd_["ret"] == pd_["ret"]:
            rel = pd_["ret"] - _base
            if want_beat is None:
                verdict = f"{pd_['ret']:+.2f}% ({rel:+.2f}pp vs {_lbl})"
            elif _open:
                verdict = (f"{pd_['ret']:+.2f}% so far "
                           f"({'▲ahead' if (rel > 0) == want_beat else '▼behind'})")
            else:
                ok = (rel > 0) if want_beat else (rel < 0)
                verdict = f"{pd_['ret']:+.2f}% ({rel:+.2f}pp) {'✅' if ok else '❌'}"
            head += f" → **{verdict}**"
        st.markdown(head)
        bits = []
        for n in names:
            v = _ps.get(n, float("nan"))
            if v != v:
                bits.append(f"{n} _(no data)_"); continue
            col = "#16a34a" if v >= 0 else "#dc2626"
            if want_beat is None:
                mk = ""
            elif _open:
                mk = " ▲" if ((v - _base > 0) == want_beat) else " ▼"
            else:
                mk = " ✅" if (((v > _base) if want_beat else (v < _base))) else " ❌"
            bits.append(f"{n} <span style='color:{col}'>({v:+.2f}%"
                        f"{' so far' if _open else ''})</span>{mk}")
        st.markdown("&nbsp;&nbsp;" + " · ".join(bits), unsafe_allow_html=True)

    _eq = _src.get("basket_eq_abs", float("nan"))
    st.caption(
        f"Benchmarks over the same window: all-sector basket **{_bask:+.2f}%**, "
        f"Nifty 50 **{_nif:+.2f}%**."
        + (f"  Sector returns here are the **median stock** in each sector — the "
           f"same definition the clock's own price axis uses, so the marks judge "
           f"the call on the basis it was made. An equal-weight basket of those "
           f"stocks would have returned **{_eq:+.2f}%** on average; the two differ "
           f"because a few large movers pull a mean around and the median ignores "
           f"them." if _eq == _eq else "")
        + (" _(Nifty unavailable — fell back to the basket for the marks.)_" if _fell else "")
        + "  **Improving is deliberately not ticked.** This tab calls it WATCH, not a "
        "buy, and a 2018-2026 backtest measured it NEGATIVE at both ends of the "
        "range (−10.6%/yr at 1-2wk, t −2.64; −11.7%/yr at 11-12wk, t −2.54), so a "
        "green mark there would be actively misleading. Neutral is not a call either."
        "\n\n**One date is one observation.** The Signal Validation section below "
        "carries the walk-forward record across many past windows — that is the "
        "number to judge the clock on, not this one."
    )


def _render_tilt_replay(selected_date: date, min_turnover: float,
                        horizon_days: int, horizon_label: str) -> None:
    """
    Replay panel: what the tilt said on a chosen past date, and what followed.

    Two outcome blocks are shown side by side and NEVER merged:
      - over the call's OWN horizon  -> the only fair scorecard
      - from then until now          -> a P&L question the signal never claimed
    Scoring a 2-week call over 6 months is a category error; the panel says so
    rather than letting the bigger number flatter the signal.
    """
    st.markdown("##### 📊 Results — what this tab said, and what followed")

    _c1, _c2 = st.columns([1, 2])
    with _c1:
        _default = selected_date - timedelta(days=90)
        _as_of = st.date_input(
            "Signal date", value=_default, max_value=selected_date,
            key="tilt_replay_date",
            help="The day you want to check. Pick a weekend or a holiday and it "
                 "moves back to the last day the market was open.")
    with _c2:
        st.caption(
            f"Scored at the **{horizon_label}** horizon selected above. Change the "
            f"horizon radio to re-score the same date over a different window — the "
            f"suggestion itself changes too, because the ranking lookback scales "
            f"with the horizon.")

    try:
        rep = cached_tilt_replay(_as_of, int(horizon_days), float(min_turnover),
                                 selected_date)
    except Exception as exc:                                    # noqa: BLE001
        st.error(f"Replay unavailable: {exc}")
        return
    if not rep.get("ok"):
        st.info(rep.get("error", "Replay unavailable for that date."))
        return

    anchor = rep["anchor"]
    if rep.get("snapped"):
        st.caption(f"↪ {_as_of} was not a trading day — using **{anchor}**.")

    st.markdown(
        f"**On {anchor} this tab said:** `{rep['verdict']}` · backdrop "
        f"`{rep['state']}` · suggested size **{int(round((rep['size_hint'] or 0)*100))}%**")
    # Per-sector outcome inline, measured over the CALL'S OWN horizon (not to
    # today). A bare "+1.1%" cannot be judged on its own — in a tape where every
    # sector gained 3% it is a miss — so each name also carries a tick against the
    # all-sector basket. For an OVERWEIGHT, beating the basket is the win; for an
    # UNDERWEIGHT, LAGGING it is the win, because the call was "avoid this".
    _hb = rep["horizon"]
    _open = _hb.get("status") == "OPEN"
    # When the window is still running, fall back to the RUNNING numbers (anchor
    # -> last EOD) so the user can see where it stands. These are shown WITHOUT a
    # win/lose tick: ticking an unfinished call would flatter or damn it by
    # accident, which is the whole reason the scorecard waits for the close.
    _src = _hb if not _open else rep.get("to_today", {})
    _ps = _src.get("per_sector") if _src.get("status") == "DONE" else None

    # WHICH BENCHMARK the tick marks judge against. The two answer different
    # questions and a rotation call can pass one and fail the other:
    #   Nifty 50   - what you could actually have bought instead. The investable
    #                alternative, and the honest "was this worth doing at all".
    #   basket     - equal-weight across every sector. Not investable, but it IS
    #                what the tilt's published edge is measured against, so it
    #                answers "was the ROTATION right, among sectors".
    _bm_choice = st.radio(
        "Judge each sector against", ["Nifty 50", "All-sector basket"],
        index=0, horizontal=True, key="tilt_replay_bench",
        help="Which yardstick the ✅/❌ marks use.\n\n"
             "Nifty 50 — could you have just bought the index instead? The "
             "practical test.\n\n"
             "All-sector basket — did picking these sectors beat owning all of "
             "them? The fair test of the picking itself, and the one this tab's "
             "track record is based on.\n\n"
             "A sector can pass one and fail the other, so it is worth checking "
             "both.")
    _use_nifty = _bm_choice == "Nifty 50"
    _bask = _src.get("basket_abs") if _src.get("status") == "DONE" else None
    _nif = _src.get("nifty_abs") if _src.get("status") == "DONE" else None
    # Nifty can be missing (no index rows in range) — fall back rather than mark
    # every sector against NaN, which would silently turn every call into a miss.
    _bm_fellback = _use_nifty and (_nif is None or _nif != _nif)
    _base = (_bask if (not _use_nifty or _bm_fellback) else _nif)
    _base_lbl = ("all-sector basket" if (not _use_nifty or _bm_fellback) else "Nifty 50")

    def _inline(names: list, want_beat: bool) -> str:
        if not names:
            return "_none_"
        if _ps is None:
            return ", ".join(f"{n} _(no data yet)_" for n in names)
        out = []
        for n in names:
            v = _ps.get(n, float("nan")) if hasattr(_ps, "get") else float("nan")
            if v != v:                                   # NaN → sector had no data
                out.append(f"{n} _(no data)_"); continue
            col = "#16a34a" if v >= 0 else "#dc2626"
            if _open:
                # running: direction vs benchmark only, never a verdict
                ahead = (v > _base) if want_beat else (v < _base)
                mark = ("<span style='color:#8a8f98'>▲ahead</span>" if ahead
                        else "<span style='color:#8a8f98'>▼behind</span>")
                out.append(f"**{n}** <span style='color:{col}'>({v:+.2f}% so far)</span> {mark}")
            else:
                mark = "✅" if ((v > _base) if want_beat else (v < _base)) else "❌"
                out.append(f"**{n}** <span style='color:{col}'>({v:+.2f}%)</span> {mark}")
        return " · ".join(out)

    # Headline the OUTCOME DATE. "10-session window" is precise but nobody reads a
    # session count as a date; the user needs to see WHEN this was settled. When
    # the window is still running, show how far in it is instead of a fake date.
    if _open:
        _rem = rep.get("sessions_remaining", 0)
        _ela = rep.get("sessions_elapsed", 0)
        _run_end = rep.get("today")
        st.markdown(
            f"#### ⏳ Running — {_ela} of {horizon_days} sessions done"
            f"{f', {_rem} to go' if _rem else ''}"
            + (f" · as at the close of {_run_end:%d %b %Y}" if _run_end else ""))
        st.caption(
            f"The {horizon_label} call made on **{anchor:%d %b %Y}** has not finished "
            f"its window, so the numbers below are **running totals to the last "
            f"close**, not the result. They carry ▲ahead / ▼behind versus "
            f"**{_base_lbl}** ({_base:+.2f}% so far) rather than ✅/❌ — a call is "
            f"only scored once its window closes, because a mid-window read flatters "
            f"or damns it by accident. Expect these to move."
            if _ps is not None else
            f"The {horizon_label} call made on **{anchor:%d %b %Y}** has not finished "
            f"its window, and no sessions have elapsed yet — nothing to show.")
    else:
        _he = rep["horizon_end"]
        _sess = _hb.get("sessions", horizon_days)
        st.markdown(f"#### 📅 Result at the close of {_he:%d %b %Y} ({_he:%a})")
        _gap = (f" — {_sess} sessions of data across the {horizon_days}-session "
                f"window (one session had no sector data)"
                if _sess != horizon_days else
                f" — {_sess} sessions")
        _other = (_bask if _use_nifty else _nif)
        _bits = [
            f"Signal given **{anchor:%d %b %Y}** at the close, held to the close of "
            f"**{_he:%d %b %Y}**{_gap}.",
            f" Tick marks judge each sector against **{_base_lbl}** ({_base:+.2f}%)",
        ]
        if _bm_fellback:
            _bits.append(" _(Nifty unavailable for this window - fell back to the basket)_")
        if _other is not None and _other == _other:
            _bits.append(f"; the other benchmark returned {_other:+.2f}%")
        _bits.append(f". For an avoid-call, lagging the {_base_lbl} is the win.")
        if _use_nifty and not _bm_fellback:
            _bits.append(" Beating Nifty is the practical test - it is what you could",
                         )
            _bits.append(" have held instead. Beating the basket is the rotation test,")
            _bits.append(" and it is the one this tab's headline edge is measured on.")
        st.caption("".join(_bits))
    _o1, _o2 = st.columns(2)
    with _o1:
        st.markdown("🟢 **Overweight (the buy list)**")
        st.markdown(_inline(rep["ow"], want_beat=True), unsafe_allow_html=True)
    with _o2:
        st.markdown("🔴 **Underweight (avoid / trim)**")
        st.markdown(_inline(rep["uw"], want_beat=False), unsafe_allow_html=True)
    if _ps is not None and _open:
        _ow_ah = sum(1 for n in rep["ow"] if _ps.get(n, float("nan")) > _base)
        _uw_ah = sum(1 for n in rep["uw"] if _ps.get(n, float("nan")) < _base)
        _n = len(rep["ow"]) + len(rep["uw"])
        st.caption(
            f"**Running position vs {_base_lbl}** (NOT a score — {_rem} session"
            f"{'s' if _rem != 1 else ''} still to go): buy list {_ow_ah}/"
            f"{len(rep['ow'])} ahead · avoid list {_uw_ah}/{len(rep['uw'])} behind "
            f"(which is the win) · {_ow_ah + _uw_ah}/{_n} currently on the right "
            f"side. This will change before the window closes.")
    elif not _open and _ps is not None:
        _ow_hit = sum(1 for n in rep["ow"] if _ps.get(n, float("nan")) > _base)
        _uw_hit = sum(1 for n in rep["uw"] if _ps.get(n, float("nan")) < _base)
        _n_ow, _n_uw = len(rep["ow"]), len(rep["uw"])
        st.caption(
            f"**Scorecard vs {_base_lbl}:** buy list {_ow_hit}/{_n_ow} beat it · "
            f"avoid list {_uw_hit}/{_n_uw} lagged it (which is the win) · "
            f"{_ow_hit + _uw_hit}/{_n_ow + _n_uw} calls correct overall. "
            f"With {_n_ow + _n_uw} names, coin-flip expectation is "
            f"{(_n_ow + _n_uw) / 2:.1f} — one date cannot separate skill from luck. "
            f"Switching the benchmark above can move individual marks, because the "
            f"two differ by "
            f"{abs((_bask or 0) - (_nif or 0)):.2f}pp over this window.")

    def _block(b: dict, primary: bool) -> None:
        if b["status"] == "OPEN":
            st.info(f"**{b['label'].capitalize()}** — still open. The "
                    f"{horizon_days}-session window has not finished yet, so there "
                    f"is no result to show. A partial window is not a result.")
            return
        if b["status"] != "DONE":
            st.caption(f"{b['label']}: no data"); return
        tag = "the fair scorecard" if primary else "P&L only — NOT the signal's claim"
        st.markdown(f"**{b['label'].capitalize()}** · {b['sessions']} sessions · _{tag}_")
        m1, m2, m3 = st.columns(3)
        m1.metric("Buy list", f"{b['ow_abs']:+.2f}%",
                  delta=f"{b['ow_vs_basket']:+.2f}pp vs all sectors",  # buy list MINUS basket
                  help="What the sectors it told you to BUY returned over this "
                       "window, splitting your money equally between them.\n\n"
                       "The small number below compares that to buying every "
                       "sector instead. Green means the picking helped; red means "
                       "you'd have done better not picking at all.")
        m2.metric("All-sector basket", f"{b['basket_abs']:+.2f}%",
                  help="What you'd have made splitting your money equally across "
                       "ALL sectors — the 'don't pick anything' result.\n\n"
                       "This is the fair test of the picking itself. If the buy "
                       "list can't beat this, choosing sectors added nothing.")
        # The delta is buy-list MINUS Nifty. Labelling it "vs buy list" under the
        # Nifty tile read as "Nifty is -1.22pp vs the buy list", i.e. the opposite
        # of the truth. Name the subject explicitly instead.
        m3.metric("Nifty 50", f"{b['nifty_abs']:+.2f}%",
                  delta=f"buy list {b['ow_vs_nifty']:+.2f}pp vs this", delta_color="off",
                  help="What the Nifty 50 index did over the same window — what "
                       "you'd have made just buying an index fund and doing "
                       "nothing else.\n\n"
                       "The number below says how the buy list did against it. "
                       "Negative means the index beat your picks.\n\n"
                       "All three figures are price only — dividends are not "
                       "included in any of them, so they are compared fairly.")
        # One plain sentence stating who won, because three tiles + two deltas is
        # a lot to parse and the sign of a relative number is easy to misread.
        _win = max([("the buy list", b["ow_abs"]), ("the all-sector basket", b["basket_abs"]),
                    ("Nifty 50", b["nifty_abs"])], key=lambda x: x[1] if x[1] == x[1] else -1e9)
        _bl = "beat" if b["ow_vs_basket"] > 0 else "lagged"
        _nl = "beat" if b["ow_vs_nifty"] > 0 else "lagged"
        st.caption(
            f"**Over this window the buy list returned {b['ow_abs']:+.2f}%, the "
            f"all-sector basket {b['basket_abs']:+.2f}% and Nifty 50 "
            f"{b['nifty_abs']:+.2f}% — so the buy list {_bl} the basket by "
            f"{abs(b['ow_vs_basket']):.2f}pp and {_nl} Nifty by "
            f"{abs(b['ow_vs_nifty']):.2f}pp. Best of the three: {_win[0]}.**")
        st.caption(
            f"Buy-minus-avoid spread **{b['ow_minus_uw']:+.2f}pp** "
            f"(underweight basket {b['uw_abs']:+.2f}%). "
            + (f"⚠️ {len(b['missing'])} suggested sector(s) had no forward data and are "
               f"excluded: {', '.join(b['missing'])}. " if b.get("missing") else "")
            + "Gross of cost.")

    st.markdown("")
    _b1, _b2 = st.columns(2)
    with _b1:
        _block(rep["horizon"], True)
    with _b2:
        _block(rep["to_today"], False)

    ev = rep.get("evidence") or {}
    st.caption(
        "**One date is one observation — do not generalise from it.** For scale, the "
        f"{horizon_label} horizon's full record is "
        + (f"**{ev.get('net_yr', float('nan')):+.1f}%/yr** net of cost "
           f"(t {ev.get('net_t', float('nan')):.2f}) across the whole 2018-2026 sample"
           if ev else "shown in the evidence box above")
        + ". A single good or bad replay says almost nothing; the aggregate does.\n\n"
        "**Two caveats on these numbers.** They are GROSS of cost — a real basket pays "
        "~0.5% a round trip. And `v_sector_master` has no as-of column, so each sector's "
        "history is measured on the stocks in it *today*; a name that joined later is "
        "treated as always having been there, which flatters every historical sector "
        "return shown here."
    )


def _render_forward_tilt(selected_date: date, min_turnover: float) -> None:
    """Cross-sectional momentum sector tilt, regime-gated, at a user-selected
    forward horizon (1-2 .. 11-12 weeks). The RS lookback scales with the horizon;
    1-2wk is the originally validated build and is bit-identical to it."""
    st.markdown("#### 🎯 Forward Sector Tilt &nbsp; <sub>β</sub>", unsafe_allow_html=True)
    st.caption(
        "Cross-sectional **momentum** - relative strength vs Nifty - is the sector call "
        "that has held up best here. Pick a forward horizon below; the ranking window "
        "scales with it, and each horizon shows its **own** measured record.\n\n"
        "A **causal sector-persistence gate** drops sectors that historically fade right "
        "after looking strong (Realty / Banking / Consumer Durables), which lifts "
        "overweight accuracy from ~55% to ~60% of the time vs the median sector.\n\n"
        "⚠️ Two corrections to earlier copy on this tab. (1) The old headline "
        "'daily-IC t≈9' was a **naive** t-statistic - forward windows overlap, so "
        "consecutive days are not independent observations. Overlap-corrected "
        "(Newey-West) the same reproduction gives **t≈1.6** at 1-2wk; the long/short IC "
        "is the sturdier read at **t 2.15**. (2) The 1-2wk horizon has been **negative "
        "in 2025-26**. Treat this as a statistical lean, never a per-call oracle."
    )

    # ── horizon selector ──────────────────────────────────────────────────────
    # The RS lookbacks scale with the chosen window (long = h, short = h/2), so
    # 1-2 wk stays bit-identical to the validated build. Each horizon carries its
    # OWN measured evidence — the 1-2wk validation is NOT reused for the others.
    from src.analytics.sector_forward_tilt import (TILT_HORIZONS, _BUCKET_EVIDENCE,
                                                   _HORIZON_EVIDENCE)
    _labels = [lbl for lbl, _ in TILT_HORIZONS]
    _pick = st.radio(
        "Forward horizon", _labels, index=0, horizontal=True,
        key="tilt_horizon",
        help="How far ahead the tilt is aimed. The momentum lookback scales with it "
             "(a 12-week call is ranked on 12-week relative strength, not 2-week). "
             "1-2 wk is the originally validated build.",
    )
    _hd = dict(TILT_HORIZONS)[_pick]
    _ev = _HORIZON_EVIDENCE.get(_hd, {})

    # ── RESULTS: what did this tab say on a past date, and what followed? ──────
    _rc1, _rc2 = st.columns([3, 1])
    with _rc2:
        _show_res = st.toggle("📊 Results", value=False, key="tilt_results",
                              help="Turn this on to look back: pick any past date, "
                                   "see which sectors this tab suggested that day, "
                                   "and find out whether they actually went up.")
    if _show_res:
        _render_tilt_replay(selected_date, min_turnover, _hd, _pick)
        st.markdown("---")

    # Every factor label below must follow the radio. The VALUES were already
    # horizon-scaled; leaving the labels fixed at "rs2w / 2-week / 10-day" meant a
    # 12-week call was displayed under 2-week captions — numbers moving while the
    # words stayed put, which is worse than not offering the horizon at all.
    _wkL = max(1, int(round(_hd / 5)))          # long RS lookback, in weeks
    _wkS = max(1, int(round((_hd // 2) / 5)))   # short RS lookback, in weeks
    _rsL_lbl = f"rs{_wkL}w"
    _rsS_lbl = f"rs{_wkS}w"
    _edge_lbl = f"~{_wkL}-week"
    # The delivery factor scales with the horizon too (it did not until the
    # _dv_windows fix), so its caption must move with it — "dv5d" is correct ONLY
    # at 1-2wk. Falls back to 5/100 if the engine did not publish the windows.
    from src.analytics.sector_forward_tilt import _dv_windows
    _dvF, _dvB = _dv_windows(_hd)
    _dv_lbl = f"dv{_dvF}d"

    try:
        df, regime = cached_forward_tilt(selected_date, min_turnover, horizon_days=_hd)
    except Exception as exc:                                  # noqa: BLE001
        st.error(f"Forward tilt unavailable: {exc}")
        return

    if _ev:
        from src.analytics.sector_forward_tilt import _HORIZON_BREAKEVEN_BPS
        _era = _ev.get("era", {})
        _recent = _era.get("2025-26")
        _net = _ev["net_yr"]
        _be = _HORIZON_BREAKEVEN_BPS.get(_hd)
        _pos = _ev.get("pct_pos")
        # Colour is driven by the measured result, not by a `validated` flag — no
        # horizon currently qualifies as validated, so nothing here renders green.
        _col = "#dc2626" if _net <= 0 else "#d97706"
        _lean = ("**NOT TRADEABLE** — negative before you place a trade"
                 if _net <= 0 else
                 "a lean, not a validated edge — the median rebalance calendar "
                 "does not clear |t|≥2")
        st.markdown(
            f"<div style='border:1px solid {_col}55;border-left:4px solid {_col};"
            f"border-radius:6px;padding:9px 13px;margin:2px 0 10px 0;background:{_col}0d;"
            f"font-size:0.9rem;color:#c9ced6'>"
            f"<b style='color:{_col}'>{_pick} evidence</b> — long-only top-4, excess vs the "
            f"equal-weight sector basket, non-overlapping, net of 25bps/side, "
            f"<b>averaged over all {_hd} rebalance calendars</b>: "
            f"<b>{_net:+.1f}%/yr</b> (median t {_ev['net_t']:+.2f}), rebalances "
            f"{_ev['reb_yr']:.1f}×/yr"
            + (f", positive on <b>{_pos * 100:.0f}%</b> of calendars" if _pos else "")
            + f". Long/short IC t (Newey-West) <b>{_ev['ls_ic_t']:+.2f}</b>. {_lean}.<br>"
            f"By era: 2018-21 <b>{_era.get('2018-21', float('nan')):+.1f}%</b> · "
            f"2022-24 <b>{_era.get('2022-24', float('nan')):+.1f}%</b> · "
            f"2025-26 <b>{_recent:+.1f}%</b>"
            + (f"<br>⚠️ <b>Breakeven cost {_be} bps/side.</b> Realistic all-in retail "
               f"cost on a 4-sector stock basket is ~20-40 bps/side, so this horizon "
               f"is at or below its own breakeven — the turnover eats it."
               if (_be and _be <= 40) else
               f"<br>Breakeven cost {_be} bps/side (realistic retail ~20-40)."
               if _be else "")
            + "</div>", unsafe_allow_html=True)
        st.caption(
            "These numbers were **regenerated 2026-08-17** (`scripts/gen_tilt_evidence.py`). "
            "The previous table on this tab quoted +19.6%/yr at 1-2wk and cited a script "
            "that never computed a %/yr figure; the repo's own backtest gives +0.4%/yr for "
            "the same spec. Two method fixes moved the answer more than any parameter: the "
            "rebalance **calendar** is now averaged over all offsets (one offset alone swings "
            "a horizon 10-16pp/yr in either direction), and the panel's liquidity filter is "
            "now **lagged** — a same-day turnover floor admitted a stock *because* it moved "
            "that day."
        )
    if df is None or df.empty:
        st.info("Not enough history / liquid sectors on this date to compute the tilt.")
        return

    # ── the ONE decision surface: verdict · size · action (frozen posture matrix) ─
    mult    = float(regime.get("confidence_mult", 1.0))
    banner  = regime.get("banner", "")
    state   = regime.get("state", "UNKNOWN")
    posture = regime.get("posture", "")
    verdict = regime.get("verdict", "SELECTIVE")
    size    = float(regime.get("size_hint", 0.5))
    action  = regime.get("action", "")
    inverts = bool(regime.get("momentum_inverts", False))
    n_sup   = int(regime.get("ow_suppressed", 0))

    _V_STYLE = {
        "ACT":         ("🟢", "#16a34a", "TRADE THE TILT"),
        "SELECTIVE":   ("🟡", "#d97706", "SELECTIVE — HALF SIZE"),
        "STAND-ASIDE": ("🔴", "#dc2626", "STAND ASIDE"),
    }
    icon, vcol, vlabel = _V_STYLE.get(verdict, _V_STYLE["SELECTIVE"])
    pct = int(round(size * 100))
    _card_tip = ("What to do today, in one line. "
                 "🟢 TRADE THE TILT = market's healthy, OK to buy the green list at full size. "
                 "🟡 SELECTIVE = weak/unclear market (chop or downtrend) — buy only the top names "
                 "at reduced size; the list still holds up but a long-only book falls with the "
                 "market. 🔴 STAND ASIDE = preserve capital. 'Suggested size' = how much of a "
                 "normal position to put on.")
    st.markdown(
        f"<div title=\"{_card_tip}\" style='border:1px solid {vcol}55;border-left:5px solid {vcol};"
        f"border-radius:8px;padding:12px 16px;margin:4px 0 10px 0;background:{vcol}0d'>"
        f"<div style='font-size:1.15rem;font-weight:700;color:{vcol}'>{icon} {vlabel} ⓘ"
        f"<span style='float:right;font-size:0.95rem;color:#8a8f98'>backdrop: {state} · "
        f"suggested size {pct}%</span></div>"
        f"<div style='margin-top:4px;color:#c9ced6'>{action}</div></div>",
        unsafe_allow_html=True)

    # ── MULTI-TIMEFRAME TREND — "is the market trend changing?" (for stock-entry timing) ─
    try:
        mtf = cached_mtf_trend(selected_date)
    except Exception:                                            # noqa: BLE001
        mtf = {"ok": False}
    if mtf.get("ok"):
        _SICON = {"UP": "🟢▲", "DOWN": "🔴▼", "FLAT": "⚪●"}
        # Descriptive labels only. A 13.6yr re-audit (scripts/audit_mtf_entry_claims.py)
        # found NO alignment state carries forward edge that survives base-rate,
        # overlap and multiplicity correction, so none of these are colour-coded as
        # a trade instruction any more — neutral slate, not green/red.
        _ENTRY = {"ALIGNED_UP": ("#64748b", "ALL 4 UP — MATURE TREND"),
                  "MOSTLY_UP": ("#64748b", "3 OF 4 UP"),
                  "MIXED": ("#64748b", "MIXED / TRANSITION"),
                  "MOSTLY_DOWN": ("#64748b", "MOSTLY DOWN"),
                  "ALL_DOWN": ("#64748b", "ALL 4 DOWN — BOTTOM-ISH, NOT A SHORT")}
        ec, elabel = _ENTRY.get(mtf["entry"], ("#8a8f98", mtf["entry"]))
        cells = "".join(
            f"<span style='display:inline-block;min-width:118px'>"
            f"<b>{k.upper()}</b> <span style='color:#8a8f98'>({v['horizon']})</span> "
            f"{_SICON.get(v['state'],'')}{' 🔄' if v['flipped'] else ''}</span>"
            for k, v in mtf["bands"].items())
        _mtf_tip = ("Market trend at 4 horizons (Nifty vs its EMA + slope). This is a "
                    "CONCURRENT description of the trend, not a forecast. 13.6yr re-audit: no "
                    "alignment state has forward edge that survives base-rate, overlap and "
                    "multiplicity correction. All-4-down is the best of the five over ~3 months "
                    "(+2.2pp vs drift) and is a bottom-ish read, never a short. 4/4-up is the "
                    "most common state (35% of sessions). 'Turn' = a band just flipped."
                    ).replace('"', "'")
        st.markdown(
            f"<div title=\"{_mtf_tip}\" style='border:1px solid {ec}55;border-radius:8px;"
            f"padding:10px 14px;margin:2px 0 10px 0;background:{ec}0d'>"
            f"<b style='color:{ec}'>🧭 Multi-timeframe trend: {mtf['n_up']}/4 up · {elabel} ⓘ</b>"
            f"<div style='margin-top:5px'>{cells}</div>"
            f"<div style='margin-top:5px;color:#c9ced6;font-size:0.9rem'>{mtf['posture']}</div></div>",
            unsafe_allow_html=True)
        # NOTE: named _mtf_ev, not _ev. `_ev` holds this horizon's tilt evidence and is
        # read further down by _bucket(); shadowing it here silently fed the MTF dict
        # into the bucket's cost-drag line.
        _mtf_ev = mtf.get("evidence") or {}
        _det = mtf.get("detail") or ""
        st.caption(
            "**Re-audited on 13.6 years (3,165 sessions) — this panel is descriptive, not a timer.** "
            "Nifty rises in 59.9% of 10-day and 67.5% of 65-day windows *unconditionally*, so every "
            "state is scored as excess over that drift."
            + (f"  Current state ({mtf['n_up']}/4 up): **{_mtf_ev.get('ex10', float('nan')):+.2f}pp** vs "
               f"drift at 10d (t {_mtf_ev.get('t10', float('nan')):+.2f}), "
               f"**{_mtf_ev.get('ex65', float('nan')):+.2f}pp** at 65d "
               f"(t {_mtf_ev.get('t65', float('nan')):+.2f}); seen {_mtf_ev.get('freq', float('nan')):.0f}% of sessions."
               if _mtf_ev else "")
            + (f"  This reading is **{_det}**." if _det else "")
            + "  The earlier '3/4-up = best entry, t+3.4' claim came from a **naive** t-statistic: "
              "forward windows overlap, and once corrected it falls to t≈1.2 and fails a "
              "multiplicity test against the other four states (p=0.59). In a non-overlapping "
              "sample the ordering actually reverses. Use the alignment as context for sizing, "
              "not as a buy trigger.")

    # ── the WHY (evidence banner) + medium-term / divergence line ────────────────
    if inverts:
        st.error(f"**Why — {state}:** {banner}")
    elif mult >= 0.95:
        st.success(f"**Why — {state}:** {banner}")
    else:
        st.warning(f"**Why — {state}:** {banner}")

    med = regime.get("med_trend", "UNKNOWN")
    dv = regime.get("divergence", "n/a")
    _DV_TXT = {
        "BULLTRAP":  "⚠ **Bull-trap** — 1-2wk up but 1-2mo down (weakest measured state; reduce size)",
        "DIP_IN_UP": "✓ **Dip-in-uptrend** — 1-2wk down inside a 1-2mo up (best entry timing)",
        "ALIGNED_UP":  "1-2wk and 1-2mo both up (aligned)",
        "ALIGNED_DN":  "⚠ 1-2wk and 1-2mo both down (aligned bearish — sideline)",
        "MIXED": "short/medium trend mixed", "n/a": "medium trend unavailable",
    }
    ts = regime.get("trend_strength", "moderate"); er = regime.get("er20", float("nan"))
    _TS = {"strong": "🟢 strong clean trend", "moderate": "🟡 moderate trend",
           "choppy": "🟠 choppy/grinding (not a clean trend)"}
    er_txt = f" (ER {er:.2f})" if er == er else ""
    st.markdown(
        f"<span style='color:#8a8f98'>Two-horizon read → short (1-2wk) vs medium (1-2mo) Nifty "
        f"trend: medium is <b>{med}</b> · {_DV_TXT.get(dv, dv)}<br>Trend quality: "
        f"<b>{_TS.get(ts, ts)}</b>{er_txt} — 8yr: clean trends persist (77% hit) vs choppy (55%); "
        f"size nudged accordingly.</span>", unsafe_allow_html=True)
    # (the regime-inversion suppression caption was removed with the dead engine path —
    #  `momentum_inverts` was False in every branch, so this could never fire.)

    # ── market-breadth NOWCAST — "are we in a sustained downtrend?" (context, not forecast) ─
    try:
        bd = cached_market_breadth(selected_date)
    except Exception:                                            # noqa: BLE001
        bd = {"ok": False}
    if bd.get("ok"):
        _BD = {  # state → (emoji, colour, plain one-liner)
            "BULL":       ("🟢", "#16a34a", "Broad uptrend — most large-caps above their trend lines and the index above its 200-day line."),
            "RECOVERING": ("🟡", "#2f9e6f", "Recovering — breadth is strong again but the index hasn't reclaimed its 200-day line yet (bounce, not confirmed bull)."),
            "NEUTRAL":    ("⚪", "#8a8f98", "Mixed — no broad trend either way."),
            "WEAKENING":  ("🟠", "#d97706", "Weakening — the index still looks OK but fewer and fewer stocks are holding up (leadership narrowing)."),
            "BEAR":       ("🔴", "#dc2626", "Broad downtrend — most large-caps below their trend lines and the index below its 200-day line."),
        }
        emo, bcol, one = _BD.get(bd["state"], _BD["NEUTRAL"])
        b50 = bd["b50"]; b200 = bd["b200"]; dur = bd.get("band_days", bd.get("dur_days", 0))
        _bd_tip = ("A snapshot of where the WHOLE market is now — not a forecast. It counts how many "
                   "of ~400 large-caps are above their own 50-day and 200-day lines, and whether the "
                   "Nifty is above/below its 200-day line. IMPORTANT (from 4yr backtest): a 'death "
                   "cross' / below-200-day does NOT predict a fall here — those signals usually mark "
                   "bottoms, not tops. So use this to know the mood, not to time a sell.").replace('"',"'")
        st.markdown(
            f"<div title=\"{_bd_tip}\" style='border:1px solid {bcol}44;border-radius:8px;"
            f"padding:8px 14px;margin:8px 0 4px 0;background:{bcol}0a'>"
            f"<b style='color:{bcol}'>{emo} Market regime nowcast: {bd['state']} ⓘ</b> "
            f"<span style='color:#8a8f98'>· {one}</span><br>"
            f"<span style='color:#8a8f98;font-size:0.9rem'>{bd['caption']}"
            f"{' · death cross (long-term lines crossed down)' if bd.get('death_cross') else ''}"
            f" · breadth has held this band {dur} day{'s' if dur != 1 else ''}"
            f"</span></div>",
            unsafe_allow_html=True)
        if bd.get("narrowing"):
            st.caption("⚠ Early caution (low confidence): index up but breadth quietly falling — "
                       "leadership is narrowing. The one signal with a mild forward-down lean in "
                       "backtest (P≈56%), so treat as 'tighten', not 'sell'.")

    # ── Nifty breakout flag — the sharpest 8yr signal (held vs fake), context only ─────
    try:
        bk = cached_nifty_breakout(selected_date)
    except Exception:                                            # noqa: BLE001
        bk = {"ok": False}
    if bk.get("ok") and bk.get("state") in ("HELD", "FAILED"):
        ds = bk["days_since"]; ago = "today" if ds == 0 else f"{ds}d ago"
        if bk["state"] == "HELD":
            st.caption(f"🚀 **Nifty breakout held** ({ago}, above {bk['level']:.0f}) — the single "
                       f"strongest state in the 8yr audit: a 20-day-high break that *sticks* → "
                       f"+1%/2wk (t+3.5) and +8.7%/6mo (t+4.0, 92% hit). Confirmed continuation.")
        else:
            st.caption(f"⚠ **Failed breakout / fakeout** ({ago}) — Nifty broke a 20-day high then "
                       f"fell back below {bk['level']:.0f}. 8yr: fakeouts dip short-term (t−1.7/2wk) "
                       f"but tend to recover later — don't chase the break, wait for a clean hold.")

    c1, c2, c3 = st.columns(3)
    # Plain-English mood. The engine states are EMA20/EMA50 reads; the raw names
    # ("TRENDING_UP") were being read as a call on the whole market, which they are
    # not — they only set how big to trade the buy list.
    # Say what the market is DOING, in words that need no finance vocabulary.
    # ("Risk-on" was replaced — it is macro cross-asset jargon and does not describe
    # what this reads, which is simply Nifty vs its 20/50-day averages.)
    _MOOD = {
        "TRENDING_UP":   ("Uptrend",     "🟢", "Nifty is above its 20- and 50-day averages and rising."),
        "HIGH_VOL":      ("Uptrend, wild", "🟡", "Still trending up, but daily swings are unusually large."),
        "CHOPPY":        ("Sideways",    "🟡", "No clear direction — the averages are tangled."),
        "TRENDING_DOWN": ("Downtrend",   "🟠", "Nifty is below its 20- and 50-day averages and falling."),
        "REVERSAL":      ("Sharp drop",  "🔴", "A hard fall inside what had been an up-run."),
    }
    _mood, _icon, _mood_what = _MOOD.get(state, (state.replace("_", " ").title(), "⚪", ""))
    _sz = int(round(float(regime.get("size_hint", 0.5)) * 100))
    c1.metric("Market mood", f"{_icon} {_mood}",
              delta=f"buy list at {_sz}% size", delta_color="off",
              help=f"{_mood_what} This is a description of the last few weeks, NOT a "
                   f"forecast of where the index goes next.\n\n"
                   f"What it does: it sets the STARTING size for today's buy list. That "
                   f"start is then trimmed by how clean the trend is (a grinding, choppy "
                   f"uptrend gets x0.8) and by a few other checks, which is why the final "
                   f"number shown is {_sz}%, not 100%.\n\n"
                   f"Why trust the dial: measured 2018-2026, the buy list beat the sell "
                   f"list by +0.71% over 10 days while the trend was up (t 2.7), and by "
                   f"+0.08% (t 0.4 - i.e. nothing) when it was not.\n\n"
                   f"Engine state: {state}.")

    n_rev = int(df["revert"].sum()) if "revert" in df.columns else 0
    c2.metric("Reverting sectors", n_rev,
              help="How many top sectors have a habit of fading right after they look strong. We "
                   "drop these from the buy list so you're not buying the top just before it turns.")
    disp = regime.get("dispersion", float("nan"))
    c3.metric("Sector dispersion", f"{disp:.1f}" if disp == disp else "—",
              help="How far apart the sectors are today. High = clear leaders and laggards, so "
                   "picking is worth it. Low = everything moving together, so there's little to "
                   "gain from rotating (the tool trims size).")

    # ── OW / UW / WATCH buckets ───────────────────────────────────────────────
    # Drill-down is toggle-gated, and the gate is load-bearing rather than tidy:
    # Streamlit evaluates expander BODIES eagerly, so turning every sector row into
    # an expander fires one per-stock query per sector on every rerun (~11 sectors,
    # ~2.5s). Off = zero extra queries. This is the trap that had this page at 61s
    # before the lazy-panel work.
    _show_stk = st.toggle(
        "🔍 Click a sector to see its stocks", value=False, key="tilt_stocks",
        help="Turns every sector below into a clickable row. Opening one shows the "
             "names inside it, ranked by delivery value. ATTRIBUTION, not a pick "
             "order — no stock-level ranking measured to work inside a strong sector.")
    if _show_stk:
        st.caption(
            "⚠️ **Attribution, not a pick order.** On 1.32M stock-days (2018-2026), "
            "target = each stock's forward return **minus its own sector's** (so the "
            "sector call is stripped out), inside OVERWEIGHT sectors: "
            "delivery-vs-own-normal **t +1.31**, momentum-vs-sector **t −3.30** "
            "(leaders *under*perform at short horizons), turnover surge **t +0.26**. "
            "A top-4 Deliv × basket is **+0.03pp (t +0.14)** and decays to **−0.07pp** "
            "at a ₹25 Cr floor. Deliv × *does* work market-wide (**IC +0.019, t +5.94**) "
            "— its edge lives in the sectors you are **not** buying. Rows are therefore "
            "sorted by delivery value: where the money actually is. "
            "`scripts/audit_within_sector_pick.py`")

    def _bucket(name: str):
        g = df[df["tilt"] == name].copy()
        icon, color, sub = _TILT_STYLE[name]
        tip = _TILT_HELP.get(name, "").replace('"', "'")
        st.markdown(f"<b style='color:{color}' title=\"{tip}\">{icon} {name} ⓘ</b> "
                    f"<span style='color:#8a8f98'>· {sub}</span>", unsafe_allow_html=True)
        # Measured bucket record. This is the granularity the data supports — the old
        # per-sector "est +145bps" was a straight line through rank (corr 1.0 with it)
        # off a spread that does not reproduce, so it is not quoted any more.
        _be = _BUCKET_EVIDENCE.get(_hd, {})
        if _be and name in ("OVERWEIGHT", "UNDERWEIGHT"):
            _k = "ow" if name == "OVERWEIGHT" else "uw"
            _pp, _t, _n = _be[f"{_k}_pp"], _be[f"{_k}_t"], _be[f"{_k}_n"]
            _sig = abs(_t) >= 2
            _c = "#8a8f98" if not _sig else (POSITIVE_COLOR if _pp > 0 else NEGATIVE_COLOR)
            _line = (f"Measured 2018-2026: sectors in this bucket averaged "
                     f"<b style='color:{_c}'>{_pp:+.2f}pp</b> over the next {_hd} sessions "
                     f"vs the average sector (t {_t:+.2f}, n {_n:,}) — <b>before cost</b>.")
            if name == "OVERWEIGHT":
                _line += (f" The {_be['gross_yr']:+.1f}%/yr that implies is offset by a "
                          f"<b>{_be['cost_drag']:.1f}pp/yr</b> cost drag at "
                          f"{_ev.get('reb_yr', 0):.0f} rebalances/yr"
                          + (", which is why the net is negative even though the ranking works."
                             if _ev.get("net_yr", 0) <= 0 else "."))
            elif not _sig:
                _line += (" Not distinguishable from neutral at this horizon — the "
                          "'AVOID / trim' label is unsupported here (it only clears "
                          "significance at 7-12 wk).")
            st.markdown(f"<div style='color:#8a8f98;font-size:0.85rem;margin:-2px 0 8px 0'>"
                        f"{_line}</div>", unsafe_allow_html=True)
        if g.empty:
            st.caption("— none —"); return
        for _, r in g.iterrows():
            flag = ""
            if r["divergence"] >= 0.30:
                flag = " · 🔼 buying ahead of price (early)"
            elif r["divergence"] <= -0.30:
                flag = " · 🔽 price ahead of buying (late)"
            if r.get("revert", False):
                flag += " · ↩ tends to fade after looking strong"
            if r["thin"]:
                flag += " · ⚠ too few stocks (noisy)"
            # `est_rel_bps` is deliberately NaN now — the tercile spread it was derived
            # from does not reproduce once the rebalance calendar is averaged over, so
            # a per-sector expected return is no longer quoted. Show it only if a future
            # calibration actually populates the column.
            _est = r.get("est_rel_bps", float("nan"))
            _est_txt = (f" · <span title='Rough expected move vs the average sector over "
                        f"{_edge_lbl}. A lean, wide error bars.'>est "
                        f"{int(_est):+d}bps</span>") if pd.notna(_est) else ""
            # Tenure badge. 0 = not established (an overlay changed the call, or no
            # history) → render nothing rather than "0 days". At the lookback edge
            # the true streak is longer than measured, so it reads "90+".
            _d = int(r.get("days_in_tilt", 0) or 0)
            _cap = int(regime.get("tenure_lookback", 90))
            _d_txt = ""
            if _d > 0:
                _d_tip = (
                    f"Consecutive sessions this sector has been on the {name} list "
                    f"at the {_pick} horizon. Measured, 2018-2026: a third of all "
                    f"list entries last exactly ONE session and the median is about "
                    f"three, so a name dropping off soon after appearing is normal, "
                    f"not a signal. IMPORTANT: dropping off is NOT a sell — holding "
                    f"to the horizon beat rotating into the replacement in 23 of 24 "
                    f"era x horizon tests, once the 50bps round trip is paid."
                ).replace('"', "'")
                _d_txt = (f" &nbsp;<span title=\"{_d_tip}\" style='background:#8a8f9822;"
                          f"border-radius:4px;padding:1px 7px;font-size:0.85rem;"
                          f"color:#c9ced6'>Days → <b>{_d}{'+' if _d >= _cap else ''}</b>"
                          f"</span>")
                if _d == 1:
                    _d_txt += " <span style='color:#d9a441;font-size:0.85rem'>NEW</span>"
                # Move since the call appeared. The RELATIVE figure leads, because a
                # sector up 3% while every sector rose 4% has lost ground.
                _sa = r.get("ret_since_tilt_pct", float("nan"))
                _sr = r.get("ret_since_tilt_rel_pp", float("nan"))
                if pd.notna(_sr):
                    # Colour is the SIGN: up green, down red, on both lists. What a
                    # colour MEANS differs by list, so the hover spells that out.
                    #
                    # The title attribute MUST stay on ONE LINE. A blank line inside
                    # it ends the markdown block, and Streamlit then prints the raw
                    # <span ...> as visible text instead of rendering it.
                    _s_col = ("#8a8f98" if abs(_sr) < 1e-9
                              else ("#3fb950" if _sr > 0 else "#f85149"))
                    _mean = ("a red (negative) number here is the call working - the "
                             "sector you were told to skip has lagged"
                             if name == "UNDERWEIGHT" else
                             "a green (positive) number here is the call working - the "
                             "sector has beaten the average")
                    _abs_txt = (f" Its own move over the same window was {_sa:+.1f}%."
                                if pd.notna(_sa) else "")
                    _s_tip = (
                        f"Move since this sector joined the {name} list, over the "
                        f"{_d - 1} session(s) since - the list publishes after the "
                        f"close, so the window starts at that day's close, the "
                        f"earliest you could have acted. "
                        f"Shown versus the equal-weight sector basket, which is what "
                        f"this engine tries to beat: a sector up 3% while every sector "
                        f"rose 4% has lost ground.{_abs_txt} "
                        f"On this list {_mean}. "
                        f"A one- or two-session figure is noise, not evidence: a third "
                        f"of entries last exactly one session. Gross of the ~50bps "
                        f"round trip."
                    ).replace(chr(34), chr(39)).replace(chr(10), " ")
                    _d_txt += (f" &nbsp;<span title=\"{_s_tip}\" "
                               f"style='font-size:0.85rem;color:{_s_col}'>"
                               f"since {_sr:+.1f}pp</span>")
            if _show_stk:
                # Same row, now the click target. Expander labels take markdown but
                # not raw HTML, so the badges are rebuilt in plain markdown — the
                # hover tooltips are traded for the drill-down, which is the point.
                _lbl = (f"**{r['sector']}**"
                        + (f"  ·  Days → {_d}{'+' if _d >= _cap else ''}"
                           + ("  · NEW" if _d == 1 else "") if _d else "")
                        + _since_md(r.get("ret_since_tilt_rel_pp", float("nan")), name)
                        + f"  ·  {_rsL_lbl} {r['rs_2w']:+.1f}%"
                        + f"  ·  {_dv_lbl} {r['dv5d']:.2f}"
                        + flag)          # `flag` is plain text/emoji, no HTML to strip
                with st.expander(_lbl, expanded=False):
                    # compact: these render inside a half-width st.columns() pane,
                    # so the two widest columns are dropped and the thin-name ⚠
                    # rides on the symbol instead of the free-text Read column.
                    _stock_table_body(r["sector"], selected_date, min_turnover, _hd,
                                      context="tilt_ow", top_n=8, compact=True)
            else:
                st.markdown(
                    f"**{r['sector']}**{_d_txt} &nbsp; <span title='Strength vs Nifty over the last "
                    f"{_wkL} weeks — higher = leading the market'>{_rsL_lbl} {r['rs_2w']:+.1f}%</span> · "
                    f"<span title='Recent delivery-buying vs its own normal — above 1 = more real "
                    f"buying than usual'>{_dv_lbl} {r['dv5d']:.2f}</span>"
                    f"{_est_txt}{flag}", unsafe_allow_html=True)

    st.caption(
        f"Each row: **Days →** = consecutive sessions the sector has held this call · "
        f"**since** = move since the call appeared, versus the equal-weight sector "
        f"basket - a sector up 3% while every sector rose 4% has lost ground. "
        f"Green is simply up, red is down; on the AVOID list a red number is "
        f"the call working · "
        f"**{_rsL_lbl}** = {_wkL}-week strength vs the market (higher = leading) · "
        f"**{_dv_lbl}** = real buying vs its own normal (>1 = heavier), measured over the "
        f"last {_dvF} sessions against its own {_dvB}-session average. All follow the horizon "
        f"you picked above. Hover any label for the plain meaning.\n\n"
        f"**Reading 'since'.** It is measured from the close of the session the call "
        f"first appeared — the list publishes after the close, so that is the earliest "
        f"you could have acted — and it is quoted RELATIVE to the average sector, "
        f"because a sector up 3% while every sector rose 4% has lost ground. It is a "
        f"record of what happened, not evidence the call worked: a third of entries "
        f"last one session, and at this horizon the buy list is negative net of the "
        f"50bps round trip (see the bucket note above). Gross of costs.\n\n"
        f"**Reading the Days badge.** Measured over 2018-2026: the buy list is a fixed "
        f"top-6-of-24 quota, so something enters only when something else leaves — churn "
        f"is structural, not a verdict. **A third of all entries last exactly one session** "
        f"and the median is ~3. **A sector dropping off is not a sell:** holding to the "
        f"horizon beat rotating into its replacement in 23 of 24 era × horizon tests once "
        f"the 50bps round trip is paid. Reacting to every daily change costs 20-38%/yr in "
        f"friction and is negative at every horizon.")
    col_ow, col_uw = st.columns(2)
    with col_ow:
        _bucket("OVERWEIGHT")
        # transparency: top-ranked sectors dropped by the persistence gate
        if "revert" in df.columns:
            demoted = df[(df["rank"] >= 0.75) & (df["revert"]) & (df["tilt"] != "OVERWEIGHT")]
            if not demoted.empty:
                names = ", ".join(demoted["sector"])
                st.caption(f"↩ kept OFF the buy list (look strong now but usually fade next): {names}")
        st.write("")
        _bucket("WATCH")
    with col_uw:
        _bucket("UNDERWEIGHT")


    # ── defensive lens (DESCRIPTIVE context — NOT part of the alpha ranking) ────
    # Audited (2026-07-03): a regime-conditional defensive BLEND degraded the tilt
    # walk-forward — the down-market edge needs forward knowledge of market direction.
    # So down-capture/beta are shown as descriptive context only; the engine never
    # trades on them.
    try:
        _defn = cached_sector_defensive(selected_date, min_turnover)
    except Exception:                                        # noqa: BLE001
        _defn = pd.DataFrame()

    # ── full ranked table ─────────────────────────────────────────────────────
    with st.expander("Full ranked table — all sectors, factors + defensive lens"):
        t = df[["sector", "tilt", "rank", "rs_2w", "rs_1w", "dv5d", "accum_breadth",
                "persistence", "n_liq", "est_rel_bps", "confidence"]].copy()
        t["rank"] = (t["rank"] * 100).round(0)
        t["accum_breadth"] = (t["accum_breadth"] * 100).round(0)
        if not _defn.empty:
            t = t.merge(_defn[["sector", "down_capture", "beta"]], on="sector", how="left")
        cfg = {
            "sector": "Sector",
            "tilt": st.column_config.TextColumn(
                "Tilt", help="🟢 OVERWEIGHT = buy list · 🔴 UNDERWEIGHT = avoid/trim · "
                             "🟡 WATCH = wait, not yet · ⚪ NEUTRAL = skip."),
            "rank": st.column_config.NumberColumn(
                "Strength rank", format="%d",
                help="Where this sector sits vs all others today, 0 (weakest) to 100 "
                     "(strongest). 75+ is the buy zone, 25 or below is the avoid zone."),
            "rs_2w": st.column_config.NumberColumn(
                f"{_wkL}wk vs mkt %", format="%.1f",
                help=f"How much it beat (+) or lagged (−) the Nifty over the last {_wkL} weeks. "
                     f"This is the main strength signal, and its window follows the forward "
                     f"horizon selected at the top of the tab."),
            "rs_1w": st.column_config.NumberColumn(
                f"{_wkS}wk vs mkt %", format="%.1f",
                help=f"Same as the {_wkL}-week column, but over the last {_wkS} week(s)."),
            "dv5d": st.column_config.NumberColumn(
                "Buying vs normal ×", format="%.2f",
                help="Recent real (delivery) buying compared to its own usual level. "
                     "Above 1 = heavier buying than normal; below 1 = lighter."),
            "accum_breadth": st.column_config.NumberColumn(
                "Stocks accumulating %", format="%d",
                help="Share of the sector's stocks showing quiet accumulation. Higher = "
                     "the strength is broad, not just one or two names."),
            "persistence": st.column_config.NumberColumn(
                "Follow-through", format="%.1f",
                help="Track record of this sector AFTER it looks strong. Positive = strength "
                     "usually continues (trust the buy). Negative = it usually fades, so we "
                     "keep it off the buy list even when it ranks high."),
            "n_liq": st.column_config.NumberColumn(
                "# tradable", format="%d",
                help="How many liquid stocks the sector has. Too few = the signal is noisy."),
            "est_rel_bps": st.column_config.NumberColumn(
                "Rough 10d edge (bps)", format="%d",
                help="Ballpark out/under-performance vs the average sector over ~10 days "
                     "(100 bps = 1%). A lean with wide error bars; shows 0 when the backdrop "
                     "says stand aside."),
            "confidence": st.column_config.NumberColumn(
                "Confidence", format="%.2f",
                help="How much to trust this row, 0 to 1. Lower when the market backdrop is "
                     "weak, the sector fades historically, or it has too few stocks."),
        }
        if "down_capture" in t.columns:
            cfg["down_capture"] = st.column_config.NumberColumn(
                "Falls vs mkt 🛡", format="%.2f",
                help="On days the market fell, how much this sector fell versus Nifty. Below 1 "
                     "= it dropped LESS than the market (more defensive). Just context for "
                     "drawdown risk — it does NOT go into the buy/avoid call.")
            cfg["beta"] = st.column_config.NumberColumn(
                "Swinginess", format="%.2f",
                help="How much it moves for a given Nifty move. Below 1 = calmer than the "
                     "market; above 1 = bigger swings both ways.")
        st.dataframe(t, hide_index=True, use_container_width=True, column_config=cfg)
        if "down_capture" in t.columns:
            st.caption(
                "🛡 **Defensive lens** answers *'if the market falls, which of these held up'* "
                "— down-capture <1 = fell less than Nifty on down days. It is **descriptive, "
                "not predictive**: blending it into the tilt degraded accuracy in backtest, so "
                "the ranking ignores it. Use it to sanity-check drawdown risk, not to pick.")
    st.caption(
        "⚠ Alpha validated on a bull sample; **regime behaviour** now calibrated on DCM's own "
        "**real-bear data 2018–2026** (2018 midcap crisis / 2020 COVID / 2022 bear). Measured: "
        "in a Nifty **downtrend** the sector overweight does **not** reliably underperform "
        "(OW−UW ~+1.6%, not significant — the earlier 'inverts ~0.8%' was a stock-level read that "
        "doesn't transfer to sectors). So downtrend/reversal **reduce size and keep top names** "
        "(the signal is unreliable and a long-only book carries market beta) rather than switch "
        "off; chop mutes; only uptrend/high-vol keep full size. A **bull-trap** (1-2wk up inside "
        "a 1-2mo downtrend) is the weakest state → extra size cut. There is **no buy-the-crash "
        "long**: smart-money accumulation FAILED to lead the recovery in backtest. UNDERWEIGHT = "
        "reduce/avoid, not a short. WATCH = accumulation, momentum not yet turned. "
        "↩ = persistence gate demoted a historically mean-reverting sector."
    )


_MONTHS_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _season_cell(mean: float, tier: str, cross, clamp: float = 5.0) -> str:
    """One heatmap cell. Colour = effect, opacity = confidence, ring = significance."""
    v = max(-1.0, min(1.0, (mean or 0.0) / clamp))
    mult = {"STRONG": 1.0, "WEAK": 0.90, "NOISE": 0.26}.get(tier, 0.26)
    alpha = round(0.06 + 0.55 * abs(v) * mult, 3)
    rgb = "22,163,74" if v >= 0 else "220,38,38"
    ring = ""
    if tier == "STRONG":
        ring = "box-shadow:inset 0 0 0 2px #eab308;"
    elif tier == "WEAK":
        ring = "box-shadow:inset 0 0 0 1px rgba(234,179,8,.55);"
    mark = "<sup style='color:#38bdf8'>✓</sup>" if cross else ""
    dim = "opacity:.45;" if tier == "NOISE" else "font-weight:600;"
    return (f"<td style='background:rgba({rgb},{alpha});{ring}text-align:center;"
            f"padding:5px 3px;font-size:11px;font-variant-numeric:tabular-nums'>"
            f"<span style='{dim}'>{mean:+.1f}</span>{mark}</td>")


def _render_month_seasonality(selected_date: date) -> None:
    """Month-wise best/worst sectors — descriptive calendar map + causal suggestion."""
    st.markdown("#### 🗓️ Month-Wise Best / Worst Sectors")
    st.caption(
        "How each sector has behaved in each **calendar month**, measured as excess "
        "return vs the equal-weight sector basket and de-meaned per sector (so this "
        "shows the *month* effect, not the sector's own drift). Built only from months "
        "that completed before the selected date."
    )

    try:
        grid, meta = cached_sector_month_map(selected_date)
        sugg = cached_month_suggestion(selected_date, top_k=4)
        rec_dcm = cached_seasonality_record(selected_date, top_k=3, lens="dcm")
        rec_nse = cached_seasonality_record(selected_date, top_k=3, lens="nse")
    except Exception as exc:                                  # noqa: BLE001
        st.error(f"Seasonality unavailable: {exc}")
        return
    if grid is None or grid.empty:
        st.info("Not enough month history on this date.")
        return

    # ── the honest headline: the rule's sign depends on the sector definition ──
    d_ann = rec_dcm.get("net_annual") if "error" not in rec_dcm else None
    n_ann = rec_nse.get("net_annual") if "error" not in rec_nse else None
    if d_ann is not None and n_ann is not None and (d_ann > 0) != (n_ann > 0):
        st.markdown(
            "<div style='border:1px solid #dc262655;border-left:5px solid #dc2626;"
            "border-radius:8px;padding:12px 16px;margin:4px 0 12px 0;background:#dc26260d'>"
            "<div style='font-size:1.05rem;font-weight:700;color:#dc2626'>"
            "⚠️ DO NOT TRADE THIS TAB STANDALONE</div>"
            "<div style='margin-top:5px;color:#c9ced6;font-size:.92rem'>"
            "Walked forward over the <b>same window</b>, this rule returns "
            f"<b>{d_ann:+.1f}%/yr</b> on these 24 buckets but <b>{n_ann:+.1f}%/yr</b> on the "
            "22 NSE sector indices. Same rule, same period, <b>opposite sign</b> — so the "
            "result is a property of the sector <i>definition</i>, not of the market. "
            "Across the full grid <b>zero</b> cells survive a permutation control, and the "
            "naive-significant count is <i>lower</i> than pure noise produces. "
            "Use this as a context//timing overlay on the Forward-Tilt call, never as a "
            "standalone sector picker.</div></div>",
            unsafe_allow_html=True)

    # ── suggestion for the month about to start ──
    if "error" not in sugg:
        when = f"for **{sugg['target_period']}**" + (" (starts soon)" if sugg["is_next"] else " (current month)")
        st.markdown(f"##### 📌 {sugg['month']} playbook {when}")
        c1, c2 = st.columns(2)
        for col, key, title, arrow in ((c1, "best", "Historically strongest", "▲"),
                                       (c2, "worst", "Historically weakest", "▼")):
            with col:
                rows = []
                for r in sugg[key]:
                    tier = r["tier"]
                    badge = ("🟡 weak" if tier == "WEAK" else
                             "🟢 strong" if tier == "STRONG" else "⚪ noise")
                    rows.append({
                        "Sector": r["sector"], "Avg %": round(r["mean"], 2),
                        "Hit %": round(r["hit"]), "Yrs": r["n"],
                        "Evidence": badge,
                        "2-lens": "✓" if r.get("cross") else "",
                    })
                st.markdown(f"**{arrow} {title}**")
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)
        st.caption(
            f"Evidence tier: 🟢 survives the grid-wide permutation control "
            f"(|t| ≥ {meta.get('crit_t', float('nan')):.1f}) · 🟡 nominally significant only · "
            f"⚪ indistinguishable from noise. **2-lens ✓** = the matching NSE sector index "
            f"shows the same sign over its longer 13.6-yr history — the single most useful "
            f"column here. This month: {sugg['n_strong']} strong, {sugg['n_weak']} weak, "
            f"{sugg['n_cross']} two-lens confirmed."
        )

    # ── what the rule has actually been worth, WITH the control ──
    # The control is mandatory. Ranking by a sector's mean excess in a month also
    # picks up that sector's persistent drift, so a pure-persistence rule that
    # never looks at the calendar reproduces most of the "seasonal" return. Only
    # the DIFFERENCE between the two rows is a seasonal claim. Windows are matched
    # (2018+) so the DCM-vs-NSE comparison is lens-vs-lens, not era-vs-era.
    st.markdown("##### 📉 What this rule has actually returned (walk-forward, net of cost)")
    SINCE = "2018-01-01"
    tr = []
    for lens_lbl, lens_key in (("DCM 24 buckets", "dcm"), ("NSE 22 indices", "nse")):
        for mode_lbl, mode_key in (("seasonal (by month)", "month"),
                                   ("month effect only", "demeaned"),
                                   ("CONTROL — ignores month", "persistence")):
            try:
                rec = cached_seasonality_record(selected_date, top_k=3, lens=lens_key,
                                                mode=mode_key, since=SINCE)
            except Exception:                                 # noqa: BLE001
                rec = {"error": "n/a"}
            if "error" in rec:
                tr.append({"Sector definition": lens_lbl, "Rule": mode_lbl,
                           "Months": "—", "Net %/mo": "—", "Net t": "—",
                           "Hit %": "—", "Net %/yr": "—"})
                continue
            tr.append({
                "Sector definition": lens_lbl, "Rule": mode_lbl,
                "Months": rec["n_months"],
                "Net %/mo": round(rec["net_pm"], 2),
                "Net t": round(rec["net_t"], 2),
                "Hit %": round(rec["hit"]),
                "Net %/yr": round(rec["net_annual"], 1),
            })
    st.dataframe(pd.DataFrame(tr), hide_index=True, use_container_width=True)
    st.caption(
        f"Rule: each month, rank sectors by their mean excess in that calendar month using "
        f"**only prior data**, hold the top 3, rebalance monthly. 25bps/side on ~80% monthly "
        f"turnover. **Windows matched from {SINCE[:4]}** so the two lenses are comparable.\n\n"
        "**Read the CONTROL row first.** It ranks sectors by their overall mean and never "
        "looks at the calendar. On the 24 buckets it earns almost as much as the seasonal "
        "rule — so most of that number is sector *persistence*, not seasonality. Only the gap "
        "between 'seasonal' and 'CONTROL' is a seasonal claim, and it does not clear |t| ≥ 2. "
        "Note the control **reverses sign** on NSE indices, which have point-in-time "
        "constituents; `v_sector_master` applies today's sector membership backwards over all "
        "history, so DCM's persistence is likely a constituent artifact."
    )

    # ── the full grid ──
    st.markdown("##### 🔥 Full grid — every sector × month")
    piv = grid.pivot(index="sector", columns="m", values="mean")
    tiers = grid.pivot(index="sector", columns="m", values="tier")
    crosses = grid.pivot(index="sector", columns="m", values="cross")
    order = piv.max(axis=1).sort_values(ascending=False).index

    html = ["<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%'>",
            "<thead><tr><th style='text-align:left;padding:6px 10px;font-size:11px;"
            "color:#8a8f98;position:sticky;left:0;background:#0e1117'>Sector</th>"]
    for mo in _MONTHS_ABBR:
        html.append(f"<th style='padding:6px 3px;font-size:10px;color:#8a8f98'>{mo}</th>")
    html.append("</tr></thead><tbody>")
    for s in order:
        html.append("<tr><th style='text-align:left;padding:5px 10px;font-size:11.5px;"
                    "font-weight:500;white-space:nowrap;position:sticky;left:0;"
                    f"background:#0e1117'>{s}</th>")
        for m in range(1, 13):
            if m not in piv.columns or pd.isna(piv.loc[s, m]):
                html.append("<td style='text-align:center;color:#3a3f48;font-size:11px'>·</td>")
            else:
                html.append(_season_cell(float(piv.loc[s, m]), str(tiers.loc[s, m]),
                                         bool(crosses.loc[s, m]) if pd.notna(crosses.loc[s, m]) else False))
        html.append("</tr>")
    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption(
        "Green = outperformed that month, red = underperformed. **Faint cells fail the "
        "significance control — which is almost all of them, by design.** Yellow ring = "
        "nominally significant. <sup style='color:#38bdf8'>✓</sup> = confirmed by the "
        "independent NSE index lens.",
        unsafe_allow_html=True)

    with st.expander("⚠️ What would break this — read before acting"):
        st.markdown(
            f"""
- **Sample size.** {meta.get('months', 0)} months = about **{meta.get('months', 0)//12} observations
  per sector-month cell**. Monthly returns are fat-tailed; 8 points cannot establish a
  seasonal effect. Only waiting fixes this.
- **Multiple testing.** {meta.get('n_cells', 0)} cells are tested at once. At p<0.05 you
  *expect* ~{int(0.05*meta.get('n_cells',0))} false positives. The permutation control
  (|t| ≥ {meta.get('crit_t', float('nan')):.1f}) is what a cell must beat to mean anything;
  currently **{meta.get('n_strong', 0)}** do.
- **Lens instability.** The same rule flips sign between the 24-bucket and 22-index
  definitions over the identical window. That alone disqualifies it as a standalone signal.
- **Dividends.** These are price series. Indian ex-dividend dates cluster **Jun–Aug**, and
  high-yield sectors carry ~2% of mechanical drag in an ex-date-heavy month — the same order
  as the effects shown. Treat Jun–Aug readings as suspect.
- **Constituent look-ahead.** `v_sector_master` holds *current* tickers, applied backwards
  over all history, so a sector's past is measured on the stocks that are in it today.
- **Costs.** ~80% monthly turnover. Gross edges of ~1%/month are largely eaten by the
  0.4%/month round trip.
- **The one durable pattern** found in the 13.6-yr study is **financials over defensives in
  October** (+5.13%/mo, t=3.84, 11 of 13 years) and its **July mirror** — visible above as
  Banking/Oct and the Jul column.
"""
        )


def _plain_level(pct: float) -> tuple[str, str]:
    """Percentile -> words. A z-score is a statistician's readout; the person
    reading this wants to know if it is high, low, or ordinary."""
    if pct != pct:
        return "—", ""
    if pct >= 90:
        return "🔴 Very high", "near the top of its 2-year range"
    if pct >= 70:
        return "🟠 High", "above its usual range"
    if pct >= 30:
        return "⚪ Normal", "middle of its usual range"
    if pct >= 10:
        return "🔵 Low", "below its usual range"
    return "🟣 Very low", "near the bottom of its 2-year range"


def _render_market_next_month(selected_date: date) -> None:
    """Conditions monitor for the next 1-4 weeks. Emits NO direction — see
    src/analytics/market_context.py for the audit trail behind that choice."""
    try:
        ctx = cached_next_month_context(selected_date)
    except Exception as exc:                                    # noqa: BLE001
        st.error(f"Market context unavailable: {exc}")
        return
    if not ctx.get("ok"):
        st.info(ctx.get("error", "No data."))
        return

    meta = ctx["meta"]
    br = ctx["base_rates"]
    one_mo = br.loc[br["horizon"] == "1 month"]
    up1m = float(one_mo["up_rate_pct"].iloc[0]) if len(one_mo) else float("nan")
    mv1m = float(one_mo["mean_pct"].iloc[0]) if len(one_mo) else float("nan")

    # ── ONE plain sentence, before any table ────────────────────────────────
    st.markdown("#### 🧭 Market Next Month")
    st.markdown(
        f"### Since 2018, Nifty has been higher a month later **{up1m:.0f}% of the time**, "
        f"averaging **{mv1m:+.1f}%**.")
    st.caption(
        "That is what happens by default, with no view at all. It is the number "
        "any forecast has to beat — and nothing tested here beats it. So this page "
        "shows you **where conditions stand**, not where the market is going.")

    st.divider()

    # ── conditions, in words ────────────────────────────────────────────────
    st.markdown("##### Where things stand right now")
    stt = ctx["state"]
    if stt.empty:
        st.caption("No FII positioning available for this date.")
    else:
        cols = st.columns(len(stt))
        for c, (_, r) in zip(cols, stt.iterrows()):
            level, phrase = _plain_level(r["pct_2y"])
            moved = ""
            if r["z60"] == r["z60"] and r["z60_5d_ago"] == r["z60_5d_ago"]:
                dz = r["z60"] - r["z60_5d_ago"]
                if abs(dz) >= 0.4:
                    moved = "rising" if dz > 0 else "falling"
            with c:
                st.metric(
                    label=r["feature"].replace("FII ", ""),
                    value=level.split(" ", 1)[1] if " " in level else level,
                    delta=moved if moved else None,
                    delta_color="off",
                    help=r["why"])
                st.caption(f"{level.split(' ')[0]} {phrase} · {r['pct_2y']:.0f}th percentile")

    st.caption(
        "**Reading it:** these describe FII positioning only — how they are "
        "leaning, not what will happen. FIIs run structurally short index futures "
        "as a hedge, so a low reading is normal there, not bearish.")

    st.divider()

    _render_trend_survival(selected_date)

    st.divider()

    # ── the RANGE — the honest answer to "what can happen next" ─────────────
    st.markdown("##### What Nifty and Bank Nifty have actually done from here")

    # where both indices closed and what they did today — shown for BOTH, not just
    # whichever is selected below, since the pair is the comparison that matters
    lv = ctx.get("levels")
    if lv is not None and not lv.empty:
        lcols = st.columns(len(lv))
        for c, (_, r) in zip(lcols, lv.iterrows()):
            c.metric(
                r["index"],
                f"{r['close']:,.0f}",
                f"{r['chg_pts']:+,.0f} pts ({r['chg_pct']:+.2f}%)",
                help=f"Close on {r['date']:%d %b %Y} versus the previous session "
                     f"({r['prev_close']:,.0f}).")
        _stale = lv[lv["is_stale"]]
        if len(_stale):
            st.caption("⚠️ " + ", ".join(
                f"**{r['index']}** last traded {r['date']:%d %b %Y}"
                for _, r in _stale.iterrows())
                + " — no session on the selected date, so the close shown is that "
                  "day's, not the selected one's.")

    rg = ctx.get("ranges")
    if rg is None or rg.empty:
        st.caption("Range history unavailable.")
    else:
        which = st.radio("Index", ["Nifty 50", "Nifty Bank"], horizontal=True,
                         key="mnm_idx", label_visibility="collapsed")
        sub = rg[rg["index"] == which]
        if len(sub):
            st.caption(f"**{which} is at {sub['spot'].iloc[0]:,.0f} today.**")
        for _, r in sub.iterrows():
            st.markdown(f"**{r['horizon']}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Likely range (1 in 2)",
                      f"{r['lvl_p25']:,.0f} – {r['lvl_p75']:,.0f}",
                      f"{r['pts_p25']:+,.0f} to {r['pts_p75']:+,.0f} pts",
                      delta_color="off")
            c2.metric("Wider range (8 in 10)",
                      f"{r['lvl_p10']:,.0f} – {r['lvl_p90']:,.0f}",
                      f"{r['pts_p10']:+,.0f} to {r['pts_p90']:+,.0f} pts",
                      delta_color="off")
            c3.metric("Typical swing", f"±{r['typical_swing_pts']:,.0f} pts",
                      f"rose {r['up_rate_pct']:.0f}% of the time",
                      delta_color="off")
        st.caption(
            "**This is the reliable part.** Direction is not forecastable from "
            "anything tested here, but the range is — and it is what you can "
            "actually plan around: position size, stop distance, whether an option "
            "premium is rich or cheap.")
        st.caption(
            "**These bands adapt to today's volatility.** Each past outcome is "
            "scaled by the volatility known before it, then rescaled to now — so a "
            "calm market gives a tighter band than a violent one. Checked over "
            "~2,800 out-of-sample forecasts: the 8-in-10 band actually contained "
            "81% of Nifty outcomes and 82% of Bank Nifty's. Raw historical "
            "percentiles were both wider AND less accurate (84% and 88%), because "
            "they blend the COVID crash into a quiet week.")
        st.caption(
            "⚠️ **The band is not a promise.** 8 in 10 means one month in five "
            "finishes outside it, and the worst month in this history was about "
            "−37% for Nifty and −45% for Bank Nifty. Size for the tail, not for "
            "the middle.")

    st.divider()

    # ── ANALOGUES — past setups that looked like today, and what followed ───
    st.markdown("##### Days that looked like today, and what happened next")
    # label_visibility must stay VISIBLE: Streamlit hangs the help "?" icon off the
    # label, so collapsing the label also hides the tooltip.
    _mode_lbl = st.radio(
        "Find past days where…", key="mnm_ana_mode", horizontal=True,
        options=["Price structure (2013+)", "Price + FII positioning (2018+)",
                 "FII positioning only (2018+)"],
        help="Pick what has to look like today for a past day to count as a "
             "match.\n\n"
             "**Price structure** — the chart looked like today's. Where the "
             "market sits vs its 50- and 200-day averages, how much it has moved, "
             "how jumpy it is, how far below its 12-month high.\n\n"
             "**Price + FII** — the chart AND FII positioning both looked like "
             "today's.\n\n"
             "**FII positioning only** — the chart is ignored. Only FII behaviour "
             "has to match: their futures and options positions, and how much they "
             "traded.\n\n"
             "They give different answers on purpose. Today, the chart matches "
             "2015 and 2019; FII positioning matches late 2020 instead.")
    st.caption(
        "**Price structure** = the chart looked like today. "
        "**Price + FII** = the chart and FII positions both looked like today. "
        "**FII positioning only** = ignore the chart, match only on what FIIs were "
        "doing. Each finds different days — that is the point.")
    _mode = ("fii" if _mode_lbl.startswith("FII")
             else "price+fii" if "FII" in _mode_lbl else "price")
    try:
        ana = cached_analogues(selected_date, _mode)
    except Exception as exc:                                    # noqa: BLE001
        ana = {"ok": False, "error": str(exc)}
    if not ana.get("ok"):
        st.caption(ana.get("error", "Analogue matching unavailable."))
    else:
        am = ana["meta"]
        _cal, _sh = am.get("cal") or {}, am.get("shared") or {}
        adf = ana["analogues"].copy()
        summ = ana["summary"]
        # The base rate goes on the SAME line as the up-share, not four captions
        # below it. Nifty rises ~57/59/64% of all windows, so an up-share is
        # meaningless read on its own and this is the line people actually read.
        _base_by_h = dict(zip(("1 week", "2 weeks", "1 month"),
                              _sh.get("base_hit", ())))
        for _, r in summ.iterrows():
            _b = _base_by_h.get(r["horizon"])
            st.markdown(
                f"**{r['horizon']}** — of the {int(r['n'])} closest past setups, "
                f"**{r['up_share_pct']:.0f}%** were higher after this long"
                + (f" (Nifty rises **{_b:.0f}%** of the time anyway)" if _b else "")
                + f". Middle outcome **{r['median_pct']:+.1f}%**, "
                f"spanning **{r['worst_pct']:+.1f}%** to **{r['best_pct']:+.1f}%**.")
        show = adf.copy()
        show["Date"] = pd.to_datetime(show["date"]).dt.strftime("%d %b %Y")
        show["Nifty then"] = show["close"].round(0)
        show["Match"] = show["distance"].round(2)

        # ── footer row, inside the same table ────────────────────────────────
        # The three outcome columns are pre-formatted as TEXT so a summary row can
        # live in them. Trade-off accepted deliberately: sorting those columns
        # becomes alphabetical rather than numeric. Date / Nifty then / Match stay
        # numeric, and those are what the table is actually sorted by.
        # No SUM is shown: these are separate episodes years apart, not a run of
        # trades, so adding the returns would produce a meaningless figure.
        # NaN, not None: these two stay NUMERIC columns (see above), and a None
        # renders as the literal string "None" under a NumberColumn while a NaN
        # renders blank and leaves the dtype — and therefore the sorting — intact.
        _foot = {"Date": f"▬ {len(show)} matches",
                 "Nifty then": np.nan, "Match": np.nan}
        for _hn in ("1 week", "2 weeks", "1 month"):
            if _hn not in show:
                continue
            _v = pd.to_numeric(show[_hn], errors="coerce")
            _ok = _v.dropna()
            if _ok.empty:
                show[_hn] = ""
                _foot[_hn] = ""
                continue
            # a flat close is neither up nor down; folding it into "down" (which
            # `100 - up%` silently did) overstates the bearish share
            _up = int((_ok > 0).sum())
            _dn = int((_ok < 0).sum())
            _fl = len(_ok) - _up - _dn
            _foot[_hn] = (f"up {_up / len(_ok) * 100:.0f}%, "
                          f"down {_dn / len(_ok) * 100:.0f}%, "
                          + (f"flat {_fl / len(_ok) * 100:.0f}%, " if _fl else "")
                          + f"median {_ok.median():+.2f}%")
            show[_hn] = _v.map(lambda x: f"{x:+.2f}%" if x == x else "")
        show = pd.concat([show, pd.DataFrame([_foot])], ignore_index=True)

        st.dataframe(
            show[["Date", "Nifty then", "Match", "1 week", "2 weeks", "1 month"]],
            hide_index=True, use_container_width=True,
            height=int(len(show)) * 35 + 45,
            column_config={
                "Match": _hnc(format="%.2f",
                    help="How closely that day resembled today. **Lower = more "
                         "alike.**\n\n"
                         "⚠️ **Compare these numbers only within one button.** The "
                         "score is a distance through however many things that "
                         "button matches on — 7, 10 or 6 — so the same figure means "
                         "different things on different buttons. On 'Price + FII' "
                         "the closest match today scores around 0.8; on 'FII only' "
                         "the *furthest* one scores about 0.7. Neither is 'better'.\n\n"
                         "⚠️ **A closer match is NOT a more reliable one.** Checked "
                         "against what actually happened afterwards, the closest "
                         "matches did not call direction any better than the "
                         "loosest. Use this only to see WHICH days are being "
                         "compared, and how tightly the set holds together."),
                "Nifty then": _hnc(format="%,.0f",
                    help="Where Nifty closed on that day.\n\n"
                         "The level itself is ignored when matching — only the "
                         "SHAPE of the setup counts. That is why 2015 at 8,400 can "
                         "match today at 24,400: both can be, say, 2% above their "
                         "50-day average and 6% off their 12-month high."),
                "1 week": _htc(
                    help="How much Nifty moved in the 5 trading days after that "
                         "day. Example: +2.11% means someone buying that close was "
                         "up 2.11% a week later.\n\n"
                         "The last row sums up the column: what share went up, what "
                         "share went down, and the middle outcome."),
                "2 weeks": _htc(
                    help="The move over the 10 trading days after that day. The "
                         "last row summarises the column."),
                "1 month": _htc(
                    help="The move over the 21 trading days after that day.\n\n"
                         "Example: 2020-10-20 matched today's setup and Nifty was "
                         "+8.75% a month later — while 2024-12-09 also matched and "
                         "was −3.78%. Same-looking setups, opposite outcomes: that "
                         "spread is the point of this table.\n\n"
                         "⚠️ Compare the last row's 'up' share against how often "
                         "Nifty rises ANYWAY — about 58% over a week and 64% over a "
                         "month. Beating 50% means nothing."),
            })

        if am.get("is_stale"):
            st.warning(
                f"⚠️ **This is {am['formation_date']:%d %b %Y}'s positioning, not "
                f"{ana['as_of']:%d %b %Y}'s.** The FII "
                "participant file was not published for the selected date — NSE "
                "skips it on Budget and other special sessions — so the match is "
                "built from the last date that has one. The matched days and their "
                "outcomes below are real; the *starting point* is one session old.")

        if am.get("dropped_gap_candidates"):
            st.caption(
                f"{am['dropped_gap_candidates']} candidate day(s) were skipped "
                f"because index history is missing sessions immediately after them "
                f"({am['n_session_breaks']} such holes exist, the largest in "
                f"May–June 2017). Their '1 week' would have spanned several weeks.")

        if _cal:
            st.caption(
                f"**These are {am['k']} separate episodes, not {am['k']} days from "
                f"the same week.** Matches must be at least {am['min_sep']} sessions "
                f"apart. On this setting, without that rule the closest matches land "
                f"a median of **{_cal['gap_noguard']} session(s)** apart — one "
                f"episode counted {am['k']} times, which looks like strong agreement "
                f"but is a single observation. With the rule the typical gap is "
                f"**{_cal['gap_guard']} sessions**."
                + ("  \n(On this particular button the guard barely binds — the "
                   "matches were already far apart.)"
                   if _cal["gap_noguard"] >= 20 else ""))
        # ── calibration, PER MODE ────────────────────────────────────────────
        # Every number below is keyed to the button that is actually selected.
        # This block used to emit the PRICE model's walk-forward figures under all
        # three buttons; two of the three readings were therefore wrong. If a mode
        # has no measurement it says so instead of borrowing another mode's.
        if not _cal:
            st.caption("No walk-forward measurement exists for this matching mode. "
                       "Read the table as description only.")
        else:
            st.warning(
                "⚠️ **A green table does not mean the market is going up.** "
                f"Measured on {_sh['window_days']:,} sessions "
                f"({_sh['window']}), betting on this button's {_sh['rule']}: Nifty "
                f"went on to rise **{_cal['bull_hit'][0]:.0f}% / "
                f"{_cal['bull_hit'][1]:.0f}% / {_cal['bull_hit'][2]:.0f}%** of the "
                "time over 1 week / 2 weeks / 1 month. It rises **"
                f"{_sh['base_hit'][0]:.0f}% / {_sh['base_hit'][1]:.0f}% / "
                f"{_sh['base_hit'][2]:.0f}%** of the time anyway — so the table is "
                f"worth **{_cal['bull_edge'][0]:+.1f} / {_cal['bull_edge'][1]:+.1f} "
                f"/ {_cal['bull_edge'][2]:+.1f} points**, on "
                f"{_cal['bull_n'][0]}/{_cal['bull_n'][1]}/{_cal['bull_n'][2]} days."
                "\n\n"
                "Allowing for the fact that those windows overlap, that is worth "
                f"**t = {_cal['bull_t'][0]:+.2f} / {_cal['bull_t'][1]:+.2f} / "
                f"{_cal['bull_t'][2]:+.2f}**. Anything under 2 is indistinguishable "
                "from luck, and this whole panel was searched hard enough to need "
                "about 3.6.")
            st.caption(
                "**Compared against picking 12 past days at random** (20 draws, same "
                f"count and spacing rules): random scores "
                f"{_cal['ctl_edge'][0]:+.1f} ± {_cal['ctl_sd'][0]:.1f} / "
                f"{_cal['ctl_edge'][1]:+.1f} ± {_cal['ctl_sd'][1]:.1f} / "
                f"{_cal['ctl_edge'][2]:+.1f} ± {_cal['ctl_sd'][2]:.1f} points. "
                "The spread on that control is the size of the effect being claimed, "
                "so treat anything inside roughly ±4 points as noise. "
                f"Rank correlation between the table's middle outcome and what "
                f"actually happened: {_cal['ic'][0]:+.3f} / {_cal['ic'][1]:+.3f} / "
                f"{_cal['ic'][2]:+.3f}.")
            st.caption(
                "When past matches leaned **BEARISH** instead (this button's bottom "
                f"20%), Nifty rose {_cal['bear_hit'][0]:.0f}% / "
                f"{_cal['bear_hit'][1]:.0f}% / {_cal['bear_hit'][2]:.0f}% of the "
                f"time — {_cal['bear_edge'][0]:+.1f} / {_cal['bear_edge'][1]:+.1f} / "
                f"{_cal['bear_edge'][2]:+.1f} points against base, on "
                f"{_cal['bear_n'][0]}/{_cal['bear_n'][1]}/{_cal['bear_n'][2]} days. "
                "Mildly contrarian, consistent across buttons, and still inside the "
                "noise band above. Worth knowing, not worth trading.")
            st.caption(
                f"**The exact percentage moves if you change the settings.** Asking "
                f"for 8 or 20 matches instead of {am['k']}, or spacing them "
                f"differently, moves today's one-month figure between "
                f"**{_cal['sens_lo']}%** and **{_cal['sens_hi']}%** on this button. "
                f"That is a {_cal['sens_hi'] - _cal['sens_lo']}-point range from "
                "settings alone, so read the headline as a rough impression, not a "
                "number.")
            st.caption(
                f"**Why none of this clears the bar.** Testing all "
                f"{_sh['n_candidates']} button × horizon combinations together gives "
                f"**p = {_sh['perm_p']:.2f}** — the best real result "
                f"({_sh['perm_obs_best']:.1f} points) is the size of thing shuffled "
                f"data throws up as a matter of course at this search intensity "
                f"({_sh['perm_null_median']:.1f}). The full "
                f"audit tried about {_sh['n_candidates_full']} variants in all. "
                "Searching harder raises the noise floor; it does not find an edge.")
            st.caption(
                "**On the spread: honest width, but no new information.** The "
                f"low-to-high range covers {_cal['span_cov_1m']:.0f}% of one-month "
                "outcomes, close to the 85% that 12 independent draws imply — so it "
                "is not overstating itself. But raced against plain 20-day "
                "volatility for predicting how big the next move will be, it adds "
                f"nothing (rank correlation {_cal['span_ic'][2]:+.3f} for the "
                f"spread versus {_sh['vol20_ic'][2]:+.3f} for volatility). **The "
                "volatility-scaled range higher up this tab is the better tool for "
                f"that** — it is narrower ({_cal['span_width_1m']:.1f} points wide "
                "here) at comparable coverage.")

    st.divider()

    # ── the analogue, compressed to one line per horizon ────────────────────
    st.markdown("##### What has followed conditions like today's")
    an = ctx["analogue"]
    if an.empty:
        st.caption("Not enough comparable history.")
    else:
        agg = (an.groupby("horizon")
                 .agg(rose=("up_rate_pct", "mean"), base=("base_up_pct", "mean"),
                      diff=("excess_pp", "mean"), n=("n_similar", "sum"))
                 .reindex([h for h in ("1 week", "2 weeks", "1 month")
                           if h in an["horizon"].unique()]))
        for h, r in agg.iterrows():
            gap = r["rose"] - r["base"]
            verdict = ("no different from normal" if abs(gap) < 3
                       else "slightly better than normal" if gap > 0
                       else "slightly worse than normal")
            st.markdown(
                f"**{h}** — after conditions like today's, Nifty rose "
                f"**{r['rose']:.0f}%** of the time. Normally it rises "
                f"**{r['base']:.0f}%** of the time. → *{verdict}.*")
        st.caption(
            "The gap between those two numbers is the only part that is about "
            "positioning. Everything else is the market's habit of drifting up.")

    # ── methodology, folded away ────────────────────────────────────────────
    with st.expander("Why there is no up/down call here (the testing)"):
        st.markdown(f"""
A next-week / next-month direction call from FII positioning was tested over
**{meta['fii_sessions']:,} sessions** ({meta['fii_start']} onward). It is not in
the data:

* **Continuous signal is noise.** 12 FII features × 3 horizons, Newey-West
  t-statistics at lag = horizon. Best was **t = 2.13**; with 36 tests you need
  about **3.2** before it means anything.
* **The extremes look good until tested honestly.** Using real-time (not
  hindsight) thresholds, lagging the signal one session because NSE publishes the
  participant file *after* the close, and running a permutation that prices in
  having searched **{meta['n_candidates_searched']} candidates**:
  **p = {meta['reality_check_p']:.3f} — it does not survive.**
* **The scale of the illusion.** Searching that many candidates on *shuffled*
  data typically throws up a **{meta['null_max_excess_pp']:.2f}pp** "edge" —
  larger than the real one that was found.
* **It decays.** FII index-option positioning in its top decile was worth
  +1.54pp per 2 weeks in 2018-21, +0.89pp in 2022-24, and **+0.33pp** from
  Nov-2024 on (51.2% hit rate against a 49.4% base).
* An earlier 8.5-year re-test of the classic FII long-share read had already
  returned **IC ≈ 0**.

**Widening the data did not help — it made it worse.** The test was rerun with
FII open interest *and* FII volume *and* FII derivative rupee flows *and* market
breadth *and* delivery acceleration *and* sector trend and dispersion:
**{meta['wide_candidates']} candidates**. Best result **{meta['wide_best_pp']:.2f}pp**, against a
noise floor whose *median* is **{meta['wide_null_median_pp']:.2f}pp** —
**p = {meta['wide_p']:.3f}**. The strongest real finding was smaller than what shuffled
data typically hands you once you search that hard. More inputs mean a bigger
search, not a better answer.

**A data trap this page avoids.** SEBI raised F&O lot sizes from Nov-2024, so FII
contract counts fell several-fold (median index-futures long 156,356 → 27,460).
Any multi-year z-score on raw counts would partly be measuring that rule change,
so everything here is scored against a rolling {meta['z_window']}-session window
and never compared across the break.
""")
        st.caption(
            f"FII participant data from {meta['fii_start']} · "
            f"{meta['fii_sessions']:,} sessions · positioning publishes after the "
            "close, so the earliest it could be acted on is the next session.")


_SMART = "🎯 Smart Money (Daily Signal)"
_TILT = "🧭 Forward Tilt"
_CLOCK = "📅 Rotation Clock"
_RS = "📈 vs Nifty50"
_SEASON = "🗓️ Month-Wise Best/Worst"
_OPER = "🕵️ Operator Footprint"
_NEXT = "🧭 Market Next Month"
_ILC  = "🏛️ Index & Large Cap"
_CMD  = "🛢️ Commodity vs Index"
_VAL  = "📐 Index P/E & P/B"

_PANELS = (_SMART, _TILT, _CLOCK, _RS, _SEASON, _OPER, _NEXT, _ILC, _CMD, _VAL)


def render(selected_date: date, min_turnover: float, all_dates: list | None = None) -> None:
    st.subheader("🔄 Sector Rotation — Smart Money Tracker")

    # NOT st.tabs. `with tab:` is a plain context manager, so st.tabs runs EVERY
    # panel body on every rerun and lets the browser hide six of them. Measured
    # 2026-08-16: ~33s to display one panel, 25.6s of it Operator Footprint,
    # which you pay even while reading Market Next Month (0.8s of real work).
    # A selector runs one panel, so the page costs what you are looking at.
    # Streamlit exposes no way to read which st.tabs tab is active server-side,
    # so laziness requires owning the selection ourselves.
    panel = st.segmented_control(
        "Panel", _PANELS, default=_SMART, key="sr_panel",
        label_visibility="collapsed")
    # segmented_control returns None when the active chip is clicked again.
    # Without this the page would blank out on a stray second click.
    panel = panel or _SMART

    if panel == _SMART:
        _render_smart_money(selected_date, min_turnover)
    elif panel == _TILT:
        _render_forward_tilt(selected_date, min_turnover)
    elif panel == _CLOCK:
        _render_rotation_clock(selected_date, min_turnover, all_dates=all_dates)
    elif panel == _RS:
        _render_relative_strength(selected_date, min_turnover, all_dates=all_dates)
    elif panel == _SEASON:
        _render_month_seasonality(selected_date)
    elif panel == _OPER:
        _render_operator_footprint(selected_date)
    elif panel == _NEXT:
        _render_market_next_month(selected_date)
    elif panel == _ILC:
        _render_index_largecap(selected_date, min_turnover)
    elif panel == _CMD:
        _render_commodity_index(selected_date)
    elif panel == _VAL:
        _render_index_valuation(selected_date)


def _render_index_valuation(selected_date: date) -> None:
    from src.dashboard.views.index_valuation import render_index_valuation
    render_index_valuation(selected_date)


def _render_commodity_index(selected_date: date) -> None:
    # Lives in its own module: this file is already ~400 KB.
    from src.dashboard.views.commodity_index import render_commodity_index
    render_commodity_index(selected_date)



# ═══════════════════════════════════════════════════════════════════════════
# TREND SURVIVAL — how much longer does the current trend have?
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def _trend_survival_board(d: date) -> pd.DataFrame:
    from src.analytics.trend_survival import get_trend_survival
    return get_trend_survival(d)


_GRADE_ICON = {"STRONG": "🟩 STRONG", "INTACT": "🟨 INTACT", "FADING": "🟥 FADING"}


def _render_trend_survival(selected_date: date) -> None:
    """Duration, never direction. Returns proved unpredictable across this
    project's testing; trend LENGTH did not, so this is the one forecast shown."""
    from src.dashboard.column_help import cfg as _hcfg

    try:
        bd = _trend_survival_board(selected_date)
    except Exception as exc:                      # never take the page down
        st.caption(f"Trend survival unavailable: {exc}")
        return
    bd = bd[bd["Error"] == ""] if "Error" in bd.columns else bd
    if bd.empty:
        st.caption("No trend data for this date.")
        return

    st.markdown("##### How much longer does this move have?")
    st.caption(
        "Nobody can tell you which way the market goes next — this page says so "
        "at the top. But **how long a move keeps going** is something the data "
        "does answer, so that is all this table does. "
        "**Hover any column heading** to see what it means.")

    def _pct(v):
        return "–" if v is None or v != v else f"{v * 100:.0f}%"

    disp = pd.DataFrame({
        "Index": bd["Index"],
        "Direction": bd["Trend"].map({"UP": "▲ Up", "DOWN": "▼ Down"}).fillna(bd["Trend"]),
        "Days so far": bd["Age"].astype(int).astype(str) + "d",
        "Strength": bd["Grade"].map(_GRADE_ICON).fillna(bd["Grade"]),
        "Gap to average": bd["Into trend"].map(lambda v: f"{v * 100:.1f}%"),
        "Still going in 5 days": bd["P(+5d)"].map(_pct),
        "Still going in 10 days": bd["P(+10d)"].map(_pct),
        "Usually lasts": bd["Median left"].map(
            lambda v: "–" if v is None or v != v
            else f"{v:.0f} more day" + ("" if abs(v - 1) < 1e-9 else "s")),
        "Older than": bd["Longer than"].map(_pct),
    })
    st.dataframe(disp, hide_index=True, use_container_width=True,
                 column_config=_hcfg(disp))

    # one row spelled out in plain words, so the table teaches itself
    lead = bd[bd["Index"] == "Nifty 50"]
    lead = lead if len(lead) else bd.head(1)
    if len(lead):
        r = lead.iloc[0]
        way = "below" if r["Trend"] == "DOWN" else "above"
        p5 = r["P(+5d)"]
        odds = ("a coin flip" if p5 == p5 and 0.45 <= p5 <= 0.55
                else "likely" if p5 == p5 and p5 > 0.55 else "unlikely")
        older = f"{r['Longer than'] * 100:.0f}%" if r["Longer than"] == r["Longer than"] else "?"
        st.caption(
            f"**Reading the {r['Index']} row:** it has closed {way} its 20-day "
            f"average for **{int(r['Age'])} days straight** — longer than {older} "
            f"of its past moves in this direction. The strength markers read "
            f"**{r['Grade']}**, but it is only {abs(r['Into trend']) * 100:.1f}% away "
            f"from that average line, so whether the move is still going a week "
            f"from now is **{odds}** ({_pct(p5)}).")

    st.caption(
        "⚠️ **This is about TIME, not direction.** \"Still going in 5 days: 90%\" "
        "is not a reason to bet that way — betting along with a down move lost "
        "money at every age tested. What it is good for: picking an option expiry "
        "that matches how long the move is likely to live, setting a stop far "
        "enough away, sizing the position, and telling whether a reversal idea is "
        "simply **early** rather than wrong.")
    st.caption(
        "Most of these odds come from the **Gap to average** column, which is "
        "partly just arithmetic — a move sitting close to the line crosses back "
        "sooner. The **Strength** grade adds roughly 10 points on top of that.")


# ═══════════════════════════════════════════════════════════════════════════
# INDEX & LARGE CAP — who actually moved the index today
# ═══════════════════════════════════════════════════════════════════════════

# Per-index verdicts for the non-Nifty indices. MEASURED on each index's own
# constituents, not inherited from Nifty: scripts/ilc_multi_index_backtest.py,
# same construction as the live panel, 2022-01 .. 2026-09 (Midcap Select from
# 2024-11-29). The script reproduces Nifty's shipped null first (composite gap
# t +3.12 alone -> +0.02 beside close-strength), so the harness is calibrated.
_ILC_VERDICT = {
    "BANKNIFTY": (
        "**A decomposition, not a forecast — re-measured on Bank Nifty's own 14 "
        "names, not borrowed from Nifty.** `scripts/ilc_multi_index_backtest.py`, "
        "1,178 sessions 2022-2026.\n\n"
        "- **Same answer as Nifty: every signal is close-strength in disguise.** "
        "Over the 51-candidate family the strongest are Top-2 breadth → next gap "
        "(|t| 3.76) and the Top-2 futures read (|t| 3.48), family p=0.008 — but "
        "both correlate +0.44 to +0.57 with the index's own close-strength, and "
        "beside it they fall to **t=+1.36 and +1.71**. The futures read's sign "
        "comes from the PRICE move (long build and short cover both need price "
        "up), so it is partly a price signal with an OI label. The rest fall to "
        "-0.06 .. +1.08.\n"
        "- **The index got less top-heavy in 2026, and history mixes both.** "
        "Regressing the index on its three bucket returns, the Top 2 carried "
        "**~0.55** of the moves every quarter from 2022 to 2025, then **0.47** "
        "(Mar-2026) and **0.39** (Jun-2026) — the SEBI concentration cap (top "
        "name ≤20%, top three ≤45%) biting.\n\n"
        "So no arrow. The state table is a **base rate** from a more top-heavy "
        "index than the one on screen."),
    "FINNIFTY": (
        "**A decomposition, not a forecast — re-measured on Fin Nifty's own 20 "
        "names.** `scripts/ilc_multi_index_backtest.py`, 1,178 sessions "
        "2022-2026.\n\n"
        "- **Its top 6 are banks plus Bajaj Finance (about 70% of the index).** "
        "Top 2 (HDFCBANK, ICICIBANK) is the same pair as Bank Nifty's, so on "
        "most days the two tabs read the same heavyweights.\n"
        "- **Every signal dies beside close-strength.** Family p=0.004, led by "
        "Top-2 and Rest-14 breadth → gap (|t| 3.91 each) and the Top-2 futures "
        "read (|t| 3.68); beside the index's own close-strength they fall to "
        "**t=+1.37, +1.87 and +1.73**.\n"
        "- **One lead that did not survive.** The carry spread → next day reads "
        "t=+2.21 and holds beside close-strength (+2.24), with the same sign on "
        "Nifty (+1.71) and Bank Nifty (+1.51). But it is one of 51 tests, the "
        "family's null 95th percentile is 3.18, and the three indices share "
        "their biggest stocks, so they are not three independent confirmations.\n"
        "- **Same weight cap as Bank Nifty:** Top 2 carried ~0.55-0.63 of the "
        "moves through 2025, **0.33** by Jun-2026.\n\n"
        "So no arrow. The state table is a **base rate**, not a call."),
    "MIDCPNIFTY": (
        "**A decomposition, not a forecast — and the weakest-evidenced of the "
        "four.** `scripts/ilc_multi_index_backtest.py --asof 2026-09-29`, **457 "
        "sessions only**: most of this basket's heavyweights (BSE, PAYTM, "
        "POLICYBZR) had no stock F&O before 29 Nov 2024, and Midcap Select only "
        "holds F&O stocks, so earlier history is a basket the index could not "
        "have owned. It is cut there.\n\n"
        "- **Nothing survives.** Family p=0.044 on 51 candidates; the best "
        "(Rest-15 futures → gap, |t| 3.26) falls to t=+1.31 beside close-strength.\n"
        "- **The composite flips sign out of sample:** next-day −9.2 bps, 47.1% "
        "hit after Oct-2025.\n"
        "- **The bucket boundary is arbitrary.** BSE is 9.6%; ranks 2-10 all sit "
        "between 5.2% and 4.1%, and rank 5 vs rank 6 is 4.63% vs 4.61%.\n"
        "- **BSE is NOT the carry spread.** Checked: BSE minus the basket "
        "explains R² 0.02 of it. The spread is broad, not one stock.\n\n"
        "So no arrow. Treat every number here as thin."),
    # the basket in force from 30 Sep 2026 (BSE promoted to Nifty 50)
    "MIDCPNIFTY_2609": (
        "**A decomposition, not a forecast — on a basket that barely has "
        "heavyweights.** From 30 Sep 2026 BSE (9.6%) moves to Nifty 50, and the "
        "largest names left are ~6%. `scripts/ilc_multi_index_backtest.py`, "
        "today's 25 applied to **457 sessions** (Nov-2024+), buckets per NSE's "
        "30 Sep 2026 factsheet.\n\n"
        "- **The Top 2 barely move the index:** regressed on the three bucket "
        "returns they carry **0.03-0.14** of its moves; the Rest 18 carries "
        "0.62-0.84. \"Heavyweights carried it\" means little here.\n"
        "- **The composite is null.** Beside close-strength the net score is "
        "t=+0.16 at the gap and −0.76 next day; out of sample next day runs "
        "−12.2 bps at 48.0% after Oct-2025. Rest-18 breadth → gap (|t| 3.25) "
        "falls to **t=+1.15** beside close-strength.\n"
        "- **One lead, not shipped.** When the Rest 18's put OI grows faster "
        "than call OI (net of the expiry cycle), the next morning opened "
        "**+25 bps** on average against **−0 bps** for the reverse; it holds "
        "beside close-strength (t=+2.68) and has the same sign in both halves "
        "of the history. But it is fragile: that t runs from +1.25 to +2.85 "
        "depending on one setting (how strongly the cycle is adjusted), the "
        "later half alone is t≈+1.6, the family p is 0.032 across 51 tests, and "
        "there are only 457 sessions.\n\n"
        "So no arrow."),
}

_ILC_SCORE_VERDICT = {
    "BANKNIFTY": (
        "**This score does not forecast Bank Nifty — measured, 1,178 sessions "
        "2022-2026.**\n\n"
        "| | next open (gap) | next day | next 5 sessions |\n|---|---|---|---|\n"
        "| **net score hit rate** | 54.9% | **50.0%** | **49.4%** |\n"
        "| base rate (up days) | 57.0% | 52.8% | 54.9% |\n"
        "| information coefficient | +0.081 | +0.000 | −0.004 |\n\n"
        "The gap IC is close-strength again (net score t=+0.47 beside it). Out "
        "of sample after May-2024 the next-day read runs **−5.8 bps at 46.0%**, "
        "below a coin flip. On disjoint weeks the five weekday offsets span −18.9 "
        "to +26.4 bps. Read it as **what positioning is doing**."),
    "FINNIFTY": (
        "**This score does not forecast Fin Nifty — measured, 1,178 sessions "
        "2022-2026.**\n\n"
        "| | next open (gap) | next day | next 5 sessions |\n|---|---|---|---|\n"
        "| **net score hit rate** | 54.5% | **51.7%** | **51.6%** |\n"
        "| base rate (up days) | 56.7% | 53.0% | 53.6% |\n"
        "| information coefficient | +0.106 | +0.033 | +0.027 |\n\n"
        "Beside close-strength the score is t=+1.34 at the gap and +1.87 next "
        "day — not significant. Out of sample after May-2024 next day runs "
        "**−0.9 bps at 49.7%**. Read it as **what positioning is doing**."),
    "MIDCPNIFTY": (
        "**This score does not forecast Midcap Select — measured, 457 sessions "
        "(Nov-2024+).**\n\n"
        "| | next open (gap) | next day | next 5 sessions |\n|---|---|---|---|\n"
        "| **net score hit rate** | 54.2% | **49.6%** | **51.3%** |\n"
        "| base rate (up days) | 69.8% | 53.8% | 52.1% |\n"
        "| information coefficient | +0.130 | −0.016 | −0.013 |\n\n"
        "Beside close-strength: t=+0.47 at the gap, −0.66 next day. Out of "
        "sample after Oct-2025 next day runs **−9.2 bps at 47.1%**. Read it as "
        "**what positioning is doing**."),
    "MIDCPNIFTY_2609": (
        "**This score does not forecast Midcap Select — measured on the "
        "30-Sep-2026 basket, 457 sessions (Nov-2024+).**\n\n"
        "| | next open (gap) | next day | next 5 sessions |\n|---|---|---|---|\n"
        "| **net score hit rate** | 56.5% | **49.7%** | **51.9%** |\n"
        "| base rate (up days) | 69.8% | 53.8% | 52.1% |\n"
        "| information coefficient | +0.145 | −0.023 | +0.003 |\n\n"
        "Beside close-strength: t=+0.16 at the gap, −0.76 next day. Out of "
        "sample after Oct-2025 next day runs **−12.2 bps at 48.0%**. Read it as "
        "**what positioning is doing**."),
}


def _ilc_verdict_key(sym: str, meta: dict) -> str:
    """Midcap Select's verdict depends on WHICH basket is in force."""
    eff = meta.get("effective")
    if sym == "MIDCPNIFTY" and eff is not None and eff >= date(2026, 9, 30):
        return "MIDCPNIFTY_2609"
    return sym


def _render_ilc_index_notes(sym: str, selected_date: date) -> None:
    """Which basket is in force on the date, and how its buckets were drawn."""
    from src.analytics.index_largecap import index_meta
    meta = index_meta(sym, selected_date)
    box = st.expander("How the buckets are drawn", expanded=False)
    if meta["effective"] is not None:
        box.caption(
            f"**Basket in force from {meta['effective']:%d %b %Y}** — "
            f"{meta['source']}. Dates before that use the previous basket.")
    if meta["provisional"]:
        st.warning(
            "**Provisional buckets.** NSE publishes the new weights in the "
            "30 Sep factsheet. Until then they are ESTIMATED: Midcap Select "
            "weights are exactly proportional to Nifty Midcap 100's (BSE/PAYTM "
            "= 1.855 in both), which places FEDERALBNK ~6.3% and MCX ~6.2% "
            "ahead of PAYTM 5.2%. INDHOTEL (top 10 per NSE, rank unknown) and "
            "GVT&D sit in the last bucket until the factsheet confirms.")
    if sym == "NIFTY":
        box.caption(
            "**Buckets are NSE's own ranking:** Top 10 = the members of NSE's "
            "*Nifty Top 10 Equal Weight* index, Next 10 = *Top 20* minus *Top 10*. "
            "NSE picks them by 6-month average free-float cap every March and "
            "September, so a month of price moves cannot reshuffle them — the "
            "spot factsheet ranks differ at the edge (31 Aug: SBIN #6, M&M #10), "
            "and the September review moves SBIN into the Top 10 from 30 Sep.")
        return
    parts =" · ".join(f"**{k}**: {', '.join(v)}" for k, v in meta["buckets"].items())
    why = {
        "BANKNIFTY": "Cut on the two widest gaps in the published weights: "
                     "17.0, 14.9 | 10.3, 9.9, 9.2 | 7.2 and below.",
        "FINNIFTY": "Cut on the two widest gaps in the published weights: "
                    "17.3, 15.1 | 10.4, 9.6, 9.5, 8.8 | 4.9 and below.",
        "MIDCPNIFTY": ("With BSE gone the index is close to equal weight "
                       "(30 Sep 2026 factsheet: MCX 6.4, FEDERALBNK 6.0 | PAYTM "
                       "5.5, HEROMOTOCO 5.2, INDHOTEL 4.9, BHEL 4.6, INDUSINDBK "
                       "4.5 | AUBANK 4.5 and below). Only the gap after rank 2 "
                       "(0.46pp) is real; Next 5 vs Rest 18 is a layout choice. "
                       "The Top 2 hold ~12%, so expect a small carry spread and "
                       "a weak heavyweight story."
                       if meta["effective"] is not None else
                       "No natural cut past BSE (9.6%) — ranks 2-10 span only "
                       "5.2% to 4.1%, so Top 5 / Next 5 is a layout choice, "
                       "not structure."),
    }[sym]
    box.caption(
        f"**Buckets**: {parts}. {why} "
        "Only the top 10 weights are published, so every cut sits inside the top "
        "10 and the last bucket is simply everyone else. A cut on a wide gap "
        "keeps membership stable as prices move; a cut between near-equal "
        "weights would reshuffle every month.")
    if selected_date < date(2025, 12, 1) and sym in ("BANKNIFTY", "FINNIFTY"):
        st.caption(
            "⚠ The date on screen predates the SEBI weight cap. The buckets are "
            "TODAY's, and the index weighted its Top 2 more heavily then (~55% "
            "of its moves against ~35% now).")
    if sym == "MIDCPNIFTY" and selected_date < meta["history_from"]:
        st.caption(
            f"⚠ Before {meta['history_from']:%d %b %Y} most of today's Midcap "
            "Select heavyweights had no stock F&O, and the index only holds F&O "
            "names — this date shows a basket the index could not have held. "
            "Base rates and matches start at that date.")


def _ilc_oi_cell(v, expiry_opt: bool = False) -> str:
    """OI % change as text. Streamlit renders a missing number as a grey 'None',
    which reads as a broken feed; an empty OI cell always has a reason."""
    if v is not None and not pd.isna(v):
        return f"{v:+.2f}"
    return "off · expiry day" if expiry_opt else "—"


_ILC_FUT_READ = {
    "long build": "🟢 long build", "short covering": "🟢 short covering",
    "long build + covering": "🟢 long build + covering",
    "short build": "🔴 short build", "long unwind": "🔴 long unwind",
    "short build + unwind": "🔴 short build + unwind",
    "mixed": "⚪ mixed", None: "—",
}


def _ilc_num(v, fmt: str = "{:+.2f}") -> str:
    """A number as text, or a dash. Used where one row (today) has no outcome
    yet: a NaN in a numeric column renders as a grey 'None'."""
    return "—" if v is None or pd.isna(v) else fmt.format(v)


_ILC_QUAD = {
    "long build": "🟢 long build", "short covering": "🟢 short covering",
    "short build": "🔴 short build", "long unwind": "🔴 long unwind",
    "flat": "⚪ flat", None: "—",
    "added, price flat": "⚪ positions added · price flat",
    "closed, price flat": "⚪ positions closed · price flat",
}
_ILC_OPT = {
    "put writing": "🟢 put writing", "put side heavier": "🟢 put side heavier",
    "balanced": "⚪ balanced", "call side heavier": "🔴 call side heavier",
    "call writing": "🔴 call writing", None: "—",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_stock_flow(trade_date: date, fno_symbol: str) -> pd.DataFrame:
    from src.analytics.index_largecap import get_stock_flow
    return get_stock_flow(trade_date, fno_symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_cycle_flow(trade_date: date, fno_symbol: str) -> dict:
    from src.analytics.index_largecap import get_cycle_flow
    return get_cycle_flow(trade_date, fno_symbol)


def _ilc_cycle_cell(x: dict, off: bool = False) -> str:
    """'+41% · usual +56% · below 11 of 12' -- a cumulative OI figure is only
    readable next to what is normal on that day of a cycle."""
    if off:
        return "off · expiry day"
    if not x or x.get("now") is None:
        return "—"
    out = f"{x['now']:+.1f}%"
    if x.get("usual") is None:
        return out + " · no usual yet"
    out += f" · usual {x['usual']:+.1f}%"
    n, pct = int(x.get("n") or 0), x.get("pct")
    if pct is not None and n:
        below = int(round(pct / 100 * n))          # past cycles this one exceeds
        if pct >= 75:
            out += f" · ▲ above {below} of {n}"
        elif pct <= 25:
            out += f" · ▼ below {n - below} of {n}"
        else:
            out += " · in line"
    return out


def _render_ilc_cycle_flow(d, sym: str, _name: str, selected_date: date) -> None:
    """Bucket OI since the expiry cycle started, against the usual for that day."""
    try:
        c = _cached_cycle_flow(selected_date, sym)
    except Exception as exc:                                   # noqa: BLE001
        st.caption(f"Cycle view unavailable: {exc}")
        return
    if not c or not c.get("buckets"):
        return
    a, k = c["anchor"], c["day"]
    st.markdown(f"##### Since the expiry cycle started — {a:%d %b %Y}, "
                f"session {k} of the cycle")
    st.caption(
        f"Today against the **{a:%d %b} expiry close**, not against yesterday. "
        "Cumulative OI is only meaningful beside its **usual** level for this "
        "day of a cycle, so each cell shows both.")
    st.expander("How to read this table", expanded=False).caption(
        "The table above compares today with **the previous session**. This one "
        f"compares today with **the close of the last monthly expiry ({a:%d %b})**: "
        "has the bucket added positions over the cycle, or shed them? "
        "**A cumulative OI figure means nothing on its own** — a new options "
        "series starts almost empty and fills through the month, so call and put "
        "OI are up sharply in *every* cycle (Nifty Top 10, typical: +56% calls and "
        "+44% puts by session 3, +154% and +112% by session 15; futures only "
        "+1.3% and +4.7%). So each figure is shown beside the bucket's **usual** "
        f"level on session {k} of its last {c['n_cycles']} cycles, and where "
        "today ranks among them. Descriptive: not tested as a signal.")
    rows = []
    for lab, r in c["buckets"].items():
        ce, pe = r["CE"], r["PE"]
        pace = "—"
        if not c["is_expiry"] and None not in (ce.get("now"), ce.get("usual"),
                                               pe.get("now"), pe.get("usual")):
            pace = (f"calls {ce['now'] - ce['usual']:+.0f} · "
                    f"puts {pe['now'] - pe['usual']:+.0f}")
        rows.append({
            "Bucket": lab,
            "Price %": _ilc_num(r.get("price")),
            "Futures OI": _ilc_cycle_cell(r["FUT"]),
            "Futures read": _ilc_lab(_ILC_QUAD, r.get("fut_read")),
            "Call OI": _ilc_cycle_cell(ce, c["is_expiry"]),
            "Put OI": _ilc_cycle_cell(pe, c["is_expiry"]),
            "Pace vs usual (pts)": "off · expiry day" if c["is_expiry"] else pace,
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            "Bucket": st.column_config.TextColumn(
                "Bucket", help="The same weight buckets as above."),
            "Price %": st.column_config.TextColumn(
                "Price %",
                help=f"The bucket's median stock move since the {a:%d %b} expiry "
                     "close, adjusted for splits, bonuses and demergers."),
            "Futures OI": st.column_config.TextColumn(
                "Futures OI",
                help="Total futures open interest now, across every live expiry "
                     f"month, against the book that survived the {a:%d %b} expiry. "
                     "Bucket MEDIAN of its stocks. Positive = the cycle has added "
                     "positions on balance; negative = it has shed them. 'usual' "
                     "is the same bucket's median on the same session of its "
                     "earlier cycles; '▲ above 10 of 12' means only 2 of the last "
                     "12 cycles had built more by this session."),
            "Futures read": st.column_config.TextColumn(
                "Futures read",
                help="The four-way read on the whole cycle: price up + OI up = "
                     "long build, price up + OI down = short covering, price down "
                     "+ OI up = short build, price down + OI down = long unwind. "
                     "'positions added / closed · price flat' = OI clearly moved "
                     "but the price did not. The flat zone widens as the cycle "
                     "ages (0.5% x the square root of sessions elapsed)."),
            "Call OI": st.column_config.TextColumn(
                "Call OI",
                help="Call open interest, every strike and expiry, against the "
                     f"{a:%d %b} post-expiry book. It is ALWAYS up strongly - the "
                     "new series is filling - so read it against 'usual', never "
                     "against zero. Switched off on an expiry day."),
            "Put OI": st.column_config.TextColumn(
                "Put OI",
                help="The same for puts. Below 'usual' means puts are being "
                     "opened more slowly than in a typical cycle, not that puts "
                     "are being closed."),
            "Pace vs usual (pts)": st.column_config.TextColumn(
                "Pace vs usual (pts)",
                help="Each side's cumulative OI minus its usual level, in "
                     "percentage points. 'calls -15 · puts -17' = both building "
                     "more slowly than usual, by about the same amount. A wide "
                     "gap between the two says one side is running ahead of the "
                     "other for this point in the cycle."),
        })
    if c["n_cycles"] < 6:
        st.caption(
            f"Only {c['n_cycles']} earlier cycle(s) ran to session {k}, so there is "
            "no 'usual' to compare with — this cycle is unusually long. The "
            "cumulative figures stand alone.")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_bucket_trends(trade_date: date, fno_symbol: str) -> dict:
    from src.analytics.index_largecap import get_bucket_trends
    return get_bucket_trends(trade_date, fno_symbol)


_ILC_TREND_NAME = {
    "delivery": "Delivery", "futures": "Futures OI", "call": "Call OI",
    "put": "Put OI", "putcall": "Put minus call",
}
_ILC_TREND_FMT = {
    "delivery": "{:+.2f}", "futures": "{:+.1f}%", "call": "{:+.0f}",
    "put": "{:+.0f}", "putcall": "{:+.0f}",
}


def _ilc_trend_reading(r: dict) -> str:
    """One row of the trend table in words. A level, never a call."""
    v, rk, m = r.get("s"), r.get("rank"), r["measure"]
    if v is None:
        return "—"
    hi = rk is not None and rk >= 70
    lo = rk is not None and rk <= 30
    if m == "delivery":
        if rk is None:
            return "🟢 heavy" if v > 0.3 else "🔴 light" if v < -0.3 else "⚪ normal"
        return "🟢 heavy" if hi else "🔴 light" if lo else "⚪ normal"
    if m == "futures":
        return _ilc_lab(_ILC_QUAD, r.get("quadrant"))
    if m in ("call", "put"):
        return ("⚪ above usual" if hi else "⚪ below usual" if lo
                else "⚪ near usual") if rk is not None else "—"
    if rk is None:
        return "—"
    return "🟢 puts ahead" if hi else "🔴 calls ahead" if lo else "⚪ even"


def _ilc_trend_direction(r: dict) -> str:
    d, c = r.get("direction"), r.get("cross")
    arrow = {"rising": "▲ rising", "falling": "▼ falling", "steady": "● steady"}.get(d, "—")
    if c:
        ago, way = c
        when = "today" if ago == 0 else f"{ago} session{'s' if ago != 1 else ''} ago"
        arrow += f" · crossed {'above' if way > 0 else 'below'} its 22-session pace {when}"
    return arrow


def _render_ilc_trends(sym: str, _name: str, selected_date: date) -> None:
    """Delivery and positioning over 10 and 22 sessions, beside the index's trend."""
    try:
        t = _cached_bucket_trends(selected_date, sym)
    except Exception as exc:                                   # noqa: BLE001
        st.caption(f"Trend view unavailable: {exc}")
        return
    st.markdown("##### Data trend — the last 10 and 22 sessions")
    if not t.get("ok"):
        st.caption(t.get("note") or "No trend available for this date.")
        return
    S, L = t["short"], t["long"]
    st.caption(
        f"The tables above are one session. This is the **run**: each bucket's "
        f"delivery and positioning added up over the last **{S}** and **{L}** "
        "sessions, where that sits in its own history, and whether it is "
        "rising or fading.")

    ix = t.get("index") or {}
    if ix:
        word = {"up": "🟢 UP", "down": "🔴 DOWN", "mixed": "⚪ mixed", "unclear": "—"}
        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"{_name} short trend ({S} sessions)",
            f"{word[ix['short']]}  {ix['short_ret']:+.1f}%",
            (f"{'above' if ix['short_above'] else 'below'} its {S}-day average "
             f"for {ix['short_run']} session{'s' if ix['short_run'] != 1 else ''}"
             if ix.get("short_run") else None),
            delta_color="off",
            help=f"UP = higher than {S} sessions ago AND above its {S}-day "
                 "average. DOWN = lower and below. Mixed = the two disagree, "
                 "which is what a turn looks like while it is happening.")
        c2.metric(
            f"{_name} long trend ({L} sessions)",
            f"{word[ix['long']]}  {ix['long_ret']:+.1f}%",
            (f"{'above' if ix['long_above'] else 'below'} its 50-day average "
             f"for {ix['long_run']} session{'s' if ix['long_run'] != 1 else ''}"
             if ix.get("long_run") else None),
            delta_color="off",
            help=f"UP = higher than {L} sessions ago (about a month) AND above "
                 "its 50-day average. DOWN = lower and below.")
        c3.metric("From its 60-session high", f"{ix['from_high']:+.1f}%",
                  help="How far the close sits below the highest close of the "
                       "last 60 sessions. 0 = at the high.")

    rows = []
    for m in ("delivery", "futures", "call", "put", "putcall"):
        for r in [x for x in t["rows"] if x["measure"] == m]:
            f = _ILC_TREND_FMT[m]
            rows.append({
                "Measure": _ILC_TREND_NAME[m],
                "Bucket": r["bucket"],
                f"{S} sessions": _ilc_num(r["s"], f),
                f"{L} sessions": _ilc_num(r["l"], f),
                "Rank": None if r["rank"] is None else round(r["rank"]),
                "Direction": _ilc_trend_direction(r),
                "Path": r["path"] if len(r["path"]) >= 5 else None,
                "Reading": _ilc_trend_reading(r),
            })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            "Measure": st.column_config.TextColumn(
                "Measure",
                help="**Delivery** = average daily delivery z (each stock against "
                     "its own 21-session normal). **Futures OI** = % change in "
                     "open futures positions, added up over the window. **Call "
                     "OI** and **Put OI** = % change in open option positions "
                     "added up, MINUS what is usual for those sessions of the "
                     "expiry cycle - a new series fills from near empty every "
                     "month, so a raw total would rise in every cycle. **Put "
                     "minus call** = how much faster puts were built than calls, "
                     "net of the cycle."),
            "Bucket": st.column_config.TextColumn(
                "Bucket", help="The same weight buckets as above, as they stand "
                               "on the Session date, applied back through the "
                               "window so like is compared with like."),
            f"{S} sessions": st.column_config.TextColumn(
                f"{S} sessions",
                help=f"The short trend: the last {S} sessions, about two weeks. "
                     "Delivery +0.40 = delivery ran 0.4 standard deviations above "
                     "normal on an average day. Futures +8.0% = open positions "
                     "grew 8% in total. Call +30 = call OI grew 30 points more "
                     "than it usually does over these sessions of a cycle."),
            f"{L} sessions": st.column_config.TextColumn(
                f"{L} sessions",
                help=f"The long trend: the last {L} sessions, about a month. "
                     f"Compare with the {S}-session column: if the short one is "
                     "running faster than the long one, the trend is building; "
                     "slower, it is fading."),
            "Rank": st.column_config.ProgressColumn(
                "Rank", format="%d", min_value=0, max_value=100,
                help=f"Where the {S}-session reading sits among the same reading "
                     "on up to 500 EARLIER sessions. 50 = ordinary. 90 = higher "
                     "than nine days in ten. 10 = lower than nine in ten. Blank "
                     "= fewer than 120 earlier sessions to compare with."),
            "Direction": st.column_config.TextColumn(
                "Direction",
                help=f"Is the {S}-session reading rising or falling against five "
                     "sessions ago - judged against how much it normally moves "
                     "in five sessions, so a wobble is 'steady'. 'Crossed' = the "
                     f"{S}-session pace moved through the {L}-session pace in the "
                     "last five sessions: the trend is turning."),
            "Path": st.column_config.LineChartColumn(
                "Path (30 sessions)",
                help=f"The {S}-session reading on each of the last 30 sessions, "
                     "oldest on the left. The shape is the trend; the last point "
                     "is today."),
            "Reading": st.column_config.TextColumn(
                "Reading",
                help="The level in words. Delivery heavy / light = Rank above 70 "
                     "/ below 30. Futures uses the same four names as the Flow "
                     f"read, over the whole {S} sessions: OI up + price up = long "
                     "build, OI up + price down = short build, OI down + price "
                     "up = short covering, OI down + price down = long unwind; "
                     "'price flat' when the bucket moved less than 1.6% over the "
                     "window. Call / put = above or below what is usual for the "
                     "cycle. Put minus call: 'puts ahead' = Rank above 70."),
        })
    notes = [
        "**What is left out, and why.** The monthly settlement session is left "
        "out of every futures and options window - positions roll into the next "
        "month that day, which reads as a +18% build for a mechanical reason. "
        "Weekend special sessions (Budget, Muhurat) are left out of everything: "
        "they are short and thin. A window that loses a session is scaled back "
        f"up to {S} or {L}, so windows stay comparable."]
    if not t.get("have_fno"):
        notes.append("**No stock F&O data in this window** - only the delivery "
                     "rows can be shown.")
    elif t.get("n_hist", 0) < 120:
        notes.append(f"**Only {t.get('n_hist', 0)} earlier sessions on file**, so "
                     "no Rank is shown yet - the readings are real, the "
                     "yardstick is not ready.")
    st.expander("How these trends are built", expanded=False).caption(
        "\n\n".join(notes))
    st.caption("**A trend in this table does not forecast the index.** It "
               "describes what delivery and positioning have been doing. "
               "Measured, below.")
    _ev = st.expander("Does the data trend say how long the index trend lasts? "
                      "— measured", expanded=False)
    _ev.error(
        "**No - and that is measured on Nifty 50, not assumed.** 1,141 sessions "
        "(Mar-2022 to Oct-2026), 87 trend measures built exactly as in this "
        "table (each bucket, 10 and 22 sessions, and the turn between them) "
        "against the next 5, 10 and 22 sessions, each one also tested beside "
        "the index's own momentum.\n\n"
        "- **As a group, nothing.** The best of the 261 tests reads t=3.50; "
        "repeating the same search with the outcomes shifted in time gives "
        "3.29 on a typical run and 4.48 at the 95th percentile. p=0.29.\n"
        "- **Turning points, nothing.** Eighteen kinds of turn (delivery turns "
        "up, put OI turns down, and so on): none moved the next 5, 10 or 22 "
        "sessions (every |t| under 1.5).\n"
        "- **The one that looked real, and why it is not shown as a rule.** "
        "Inside a downtrend, futures OI being *shed* was followed by +3.60% "
        "over 22 sessions (93% up) against -0.48% (36% up) while OI was still "
        "being *built*. But those 41 days are only 12 separate downtrends - one "
        "vote per downtrend gives t=+1.35; the gap is absent in 2025-2026 "
        "(+0.09% vs +0.25%); and in the slide it was found during, the ten "
        "OI-shed days of 08-24 Sep 2026 were followed by -2.33% over the next "
        "10 sessions, none of the ten higher.\n"
        "- **Tops do not share a signature.** Before the Aug-2026 fall, futures "
        "OI built over 22 sessions was at its lowest reading on record and "
        "Top-10 delivery at its 98th percentile. Across the seven falls of 5% "
        "or more since 2022 no reading repeats - the Sep-2024 top had futures "
        "OI at its 90th percentile, the opposite.\n\n"
        "A month-long horizon has only about 51 independent windows in this "
        "history, so an effect smaller than roughly 2% a month could not be "
        "seen here even if it existed. Read the table as the state of "
        "positioning, not as where the index goes."
        + ("" if sym == "NIFTY" else
           f"\n\nThe full search was run on Nifty 50. On {_name} only the "
           "futures-in-a-downtrend cell was re-run: same direction on Bank "
           "Nifty and Midcap Select, absent on Fin Nifty - and the three share "
           "their largest names, so that is not three confirmations."))


def _ilc_lab(table: dict, v) -> str:
    """Label lookup that treats any missing value (None or NaN) as a dash: pandas
    turns a None in a mostly-string column into NaN, which is not a dict key."""
    return table[None] if v is None or (isinstance(v, float) and pd.isna(v)) else table[v]


def _ilc_shares(v: float) -> str:
    """OI change in shares, in lakh/crore so a 28-million figure is readable."""
    if v is None or pd.isna(v):
        return "—"
    a = abs(v)
    if a >= 1e7:
        return f"{v / 1e7:+.2f} Cr"
    return f"{v / 1e5:+.1f} L"


def _render_ilc_stock_flow(d, sym: str, _name: str, selected_date: date) -> None:
    """Stock by stock: how each vote was cast, the last 4 sessions, the cycle."""
    try:
        sf = _cached_stock_flow(selected_date, sym)
    except Exception as exc:                                   # noqa: BLE001
        st.caption(f"Stock-by-stock view unavailable: {exc}")
        return
    if sf.empty:
        return
    anchor = sf["cyc_from"].iloc[0]
    since = f"since {anchor:%d %b}" if anchor else "this cycle"
    st.markdown("##### Stock by stock — how each vote was cast")
    st.caption(
        "Every stock's vote: **today**, the **last 4 sessions**, and the "
        f"**cycle {since}**. OI up = positions added; OI down = positions closed.")
    st.expander("How each vote is measured", expanded=False).caption(
        "**Which contracts:** every live expiry month **added together** — "
        "current + next + far — with each contract compared against itself on "
        "the earlier day, so a position rolled into next month nets to zero. "
        "Measured on 89,103 stock-days: the current month holds 83.8% of the OI, "
        "the next 15.1%, the far 1.1%; the **current month alone** gives a "
        "different label on 27% of stock-days and the *opposite* one on 11%, "
        "because it bleeds into the next contract through the roll. "
        "**OI up = positions were added; OI down = positions were closed** — "
        "that part is arithmetic. *Which side* added them is read from price. "
        f"Three horizons: **today**, the **last 4 sessions**, and **{since}** "
        "(the close of the last monthly expiry).")
    out = pd.DataFrame({
        "Bucket": sf["bucket"],
        "Stock": sf["symbol"],
        "Price %": [_ilc_num(v) for v in sf["price_pct"]],
        "OI % today": [_ilc_num(v) for v in sf["oi_pct"]],
        "OI change": [_ilc_shares(v) for v in sf["oi_chg"]],
        "Today": [_ilc_lab(_ILC_QUAD, v) for v in sf["read"]],
        "Cur · Next · Far %": [
            "—" if pd.isna(c) else f"{c:.0f} · {n:.0f} · {f:.0f}"
            for c, n, f in zip(sf["cur_share"], sf["next_share"], sf["far_share"])],
        "OI % last 4": [", ".join(f"{x:+.1f}" for x in t) if t else "—"
                        for t in sf["trail"]],
        "Cycle price %": [_ilc_num(v) for v in sf["cyc_price_pct"]],
        "Cycle OI %": [_ilc_num(v) for v in sf["cyc_oi_pct"]],
        "Cycle": [_ilc_lab(_ILC_QUAD, v) for v in sf["cyc_read"]],
        "Activity %": [_ilc_num(v, "{:+.0f}") for v in sf["activity_pct"]],
        "Delivery %": [_ilc_num(v, "{:.1f}") for v in sf["deliv_pct"]],
        "Call OI %": [_ilc_oi_cell(v, d.is_expiry_session) for v in sf["ce_pct"]],
        "Put OI %": [_ilc_oi_cell(v, d.is_expiry_session) for v in sf["pe_pct"]],
        "Options": [("off · expiry day" if d.is_expiry_session
                     else _ilc_lab(_ILC_OPT, v)) for v in sf["opt_read"]],
    })
    st.dataframe(
        out, hide_index=True, use_container_width=True,
        height=min(38 * (len(out) + 1) + 4, 560),
        column_config={
            "Bucket": st.column_config.TextColumn(
                "Bucket", help="Which weight bucket the stock sits in."),
            "Stock": st.column_config.TextColumn(
                "Stock", help="NSE symbol. Every index member is listed."),
            "Price %": st.column_config.TextColumn(
                "Price %",
                help="The stock's cash-market close against its previous close, "
                     "adjusted for splits and bonuses on the ex-date (a 1:2 "
                     "bonus is not a 33% fall). The futures price is not used: "
                     "it breaks across a roll."),
            "OI % today": st.column_config.TextColumn(
                "OI % today",
                help="Change in the stock's total futures open interest - all "
                     "expiry months added - against the previous session. "
                     "Positive = positions were ADDED today. Negative = "
                     "positions were CLOSED. Under 0.5% either way casts no "
                     "vote ('flat')."),
            "OI change": st.column_config.TextColumn(
                "OI change",
                help="The same change in shares (L = lakh, Cr = crore). A big "
                     "percentage on a thin stock can be a small number of "
                     "shares; this shows the size."),
            "Today": st.column_config.TextColumn(
                "Today",
                help="One session. Price up + OI up = long build (positions "
                     "added on a rise). Price up + OI down = short covering "
                     "(positions closed on a rise). Price down + OI up = short "
                     "build. Price down + OI down = long unwind. These are the "
                     "exact votes counted in the 'Stock votes' column above."),
            "Cur · Next · Far %": st.column_config.TextColumn(
                "Cur · Next · Far %",
                help="How today's open interest is split across the current, "
                     "next and far expiry months. '96 · 4 · 0' = almost all in "
                     "the current month. As expiry nears, the first number "
                     "falls and the second rises - that is the roll, and it is "
                     "why the three are added before measuring any change."),
            "OI % last 4": st.column_config.TextColumn(
                "OI % last 4",
                help="The daily OI change for the last four sessions, TODAY "
                     "FIRST. '+2.0, -0.4, +0.2, +13.9' = +2.0% today, -0.4% the "
                     "session before. Shows whether today continues a run or "
                     "reverses it. An entry that falls on a monthly expiry day "
                     "counts the settling contract too."),
            "Cycle price %": st.column_config.TextColumn(
                "Cycle price %",
                help=f"The stock's price move {since}, the close of the last "
                     "monthly expiry."),
            "Cycle OI %": st.column_config.TextColumn(
                "Cycle OI %",
                help=f"Total futures open interest now against the book that "
                     f"survived the last expiry ({since}). Positive = the "
                     "cycle has ADDED positions on balance. Withheld ('—') if "
                     "a single day moved more than 50%: that is a split or "
                     "bonus re-basing the contract, not positioning."),
            "Cycle": st.column_config.TextColumn(
                "Cycle",
                help=f"The same four-way read on the whole cycle {since}, "
                     "instead of on one day. A stock can read 'short covering' "
                     "today and 'short build' for the cycle: shorts built over "
                     "several sessions and some were bought back today. The "
                     "'positions added / closed · price flat' means OI clearly "
                     "moved but price did not, so the side cannot be read. The "
                     "'flat' zone widens as the cycle ages (0.5% x the square "
                     "root of the sessions elapsed, on both price and OI), "
                     "because after three weeks almost everything has moved "
                     "more than half a percent. Splits and bonuses are adjusted "
                     "out using the exchange's own factors."),
            "Activity %": st.column_config.TextColumn(
                "Activity %",
                help="The volume leg: the stock's CASH turnover against its own "
                     "21-session norm. Futures volume is deliberately not used "
                     "- measured, it reads +204% three sessions before every "
                     "expiry (each roll trades two contracts) and about -60% "
                     "for eight sessions after, so it shows the calendar, not "
                     "conviction. A lean on heavy activity carries more "
                     "weight than the same lean on a quiet day."),
            "Delivery %": st.column_config.TextColumn(
                "Delivery %",
                help="Share of today's cash volume that was taken to delivery "
                     "rather than squared off intraday."),
            "Call OI %": st.column_config.TextColumn(
                "Call OI %",
                help="Change in the stock's call open interest, every strike "
                     "and expiry added, against the previous session."),
            "Put OI %": st.column_config.TextColumn(
                "Put OI %",
                help="Change in the stock's put open interest, every strike "
                     "and expiry added, against the previous session."),
            "Options": st.column_config.TextColumn(
                "Options",
                help="Which side grew faster for this stock, judged against "
                     "what is normal for this point in the expiry cycle (calls "
                     "build faster early in a cycle and close faster into "
                     "expiry). 'Put side heavier' reads bullish ONLY if the new "
                     "puts were sold rather than bought - open interest cannot "
                     "tell which."),
        })


@st.cache_data(ttl=600, show_spinner=False)
def _cached_ilc_forward_record(fno_symbol: str) -> dict:
    from src.analytics.ilc_forward import forward_record
    return forward_record(fno_symbol)


def _render_ilc_forward(sym: str, _name: str, selected_date: date) -> None:
    """The one lead, frozen and logged before each open. A record, not a call."""
    if sym not in ("NIFTY", "BANKNIFTY"):
        return
    from src.analytics.ilc_forward import load_model
    try:
        model = load_model()
        rec = _cached_ilc_forward_record(sym)
    except Exception as exc:                                   # noqa: BLE001
        st.caption(f"Forward test unavailable: {exc}")
        return
    pre = model["preregistration"]
    box = st.expander(f"🧪 Forward test — the one lead, frozen {model['frozen_on']}: "
                      f"{rec['n_scored']} of 300 sessions scored", expanded=False)
    box.caption(
        "**Not a signal yet.** The backtest found one weak lead on this tab: the "
        "numbers above (options side beyond the price move, delivery, futures "
        f"lean) against the NEXT session's open-to-close — {model['backtest_before_freezing'][sym]}. "
        "A backtest can fool itself, so the model was **frozen** and is now "
        "scored only on sessions it has never seen. Each evening's forecast is "
        "written once, before the next open, and never edited. "
        f"**Decision at {pre['decision_at']}**: {pre['pass']}. "
        "Nothing before then is evidence — at its backtest strength the record "
        "needs ~280 sessions just to separate from luck."
        + (" Bank Nifty is shown beside Nifty, not counted as a second test: the "
           "two share their biggest banks." if sym == "BANKNIFTY" else ""))
    R = rec["rows"]
    if R is None or len(R) == 0:
        box.info("No forward rows yet — the first is written by tonight's daily job.")
        return
    c1, c2, c3, c4 = box.columns(4)
    c1.metric("Sessions logged", rec["n_logged"],
              help="Rows written by the nightly job. One per session; never rewritten.")
    c2.metric("On time, scored", rec["n_scored"],
              help="Written before 09:15 on the next session AND that session has "
                   f"closed. Late rows ({rec['late']}) are kept but never scored.")
    c3.metric("Rank corr (forecast vs next open-to-close)",
              "-" if rec["ic"] is None else f"{rec['ic']:+.2f}",
              None if rec["ic_t"] is None else f"t {rec['ic_t']:+.2f}", delta_color="off",
              help="The primary test. Backtest +0.12. Needs t >= 2.0 at 300 sessions.")
    c4.metric("Fifth trades, net of 0.05%",
              "-" if rec["trade_net"] is None else f"{rec['trade_net']:+.3f}%",
              (f"{rec['trades']} trades · {rec['trade_hit']:.0f}% won"
               if rec["trades"] else None), delta_color="off",
              help="LONG when the forecast is in the top fifth of its history, SHORT "
                   "in the bottom fifth, else no trade. Next session open to close.")
    show = R.tail(15).iloc[::-1]
    box.dataframe(pd.DataFrame({
        "Session": [d.strftime("%d %b %Y") for d in show.trade_date],
        "Logged": [t.strftime("%d %b %H:%M") for t in show.logged_at],
        "On time": ["yes" if v is True else "LATE" if v is False else "pending"
                    for v in show.on_time],
        "Forecast %": [f"{v:+.3f}" for v in show.score],
        "Side": show.side,
        "Next open→close %": [_ilc_num(v) for v in show.next_oc],
        "Net %": [_ilc_num(v, "{:+.3f}") for v in show.pnl_net],
    }), hide_index=True, use_container_width=True)


def _ilc_chip(text: str, colour: str) -> str:
    return (f"<span style='background:{colour}22;border:1px solid {colour}55;"
            f"color:{colour};padding:2px 8px;border-radius:10px;"
            f"font-size:0.82rem;white-space:nowrap'>{text}</span>")



# Dead bands for the live Delivery/Activity read. The delivery band is the SAME
# +-0.3 the Delivery column uses to print heavy/light, so the tooltip can never
# call a bucket "delivery up" while the column beside it says "normal". Without a
# band, 09 Sep 2026 Rest 30 (z +0.11, Activity +4%) was labelled "Real - more of it
# delivered" while its rupee delivery actually FELL 1%.
_PAIR_Z_BAND = 0.3      # delivery z
_PAIR_ACT_BAND = 5.0    # Activity %
_PAIR_RS_BAND = 5.0     # rupee delivery change %, for the verdict


def _ilc_pair_help(rows: list, when: date) -> str:
    """Tooltip for the Delivery z + Activity pair, with a LIVE worked example.

    The rule is fixed. The example is computed from the selected session, because
    any hard-coded example ("Rest 30 today is the trap") is only true on one date
    and reads as nonsense on every other. The verdict on each bucket is anchored
    on the RUPEES actually delivered, which is the ground truth the share and the
    activity are only two halves of.
    """
    def _dz(v):
        return "↑" if v > _PAIR_Z_BAND else "↓" if v < -_PAIR_Z_BAND else "→"

    def _da(v):
        return "↑" if v > _PAIR_ACT_BAND else "↓" if v < -_PAIR_ACT_BAND else "→"

    rule = {
        ("↑", "↑"): "**Real** — more money, and more of it delivered",
        ("↑", "↓"): "**Don't trust it** — the day was quiet, not the buying strong",
        ("↓", "↑"): "Busy, but mostly day-trading",
        ("↓", "↓"): "Quiet, nothing happening",
    }
    word_d = {"↑": "heavy", "↓": "light", "→": "normal"}
    word_a = {"↑": "busier", "↓": "quieter", "→": "normal"}

    parts = [
        "**Activity tells you whether a delivery reading is real.** Delivery % is a "
        "SHARE of the day's trading, so it can go up for two very different "
        "reasons: more people really took delivery (real buying), or the day was "
        "just quiet, so the same delivered stock became a bigger share of a smaller "
        "total (looks like buying, isn't).",
        "**The one rule to remember:**",
        "- Delivery ↑ + Activity ↑ → " + rule[("↑", "↑")],
        "- Delivery ↑ + Activity ↓ → " + rule[("↑", "↓")],
        "- Delivery ↓ + Activity ↑ → " + rule[("↓", "↑")],
        "- Delivery ↓ + Activity ↓ → " + rule[("↓", "↓")],
        f"(Delivery counts as ↑/↓ beyond ±{_PAIR_Z_BAND} z, the same line the "
        f"Delivery column uses; Activity beyond ±{_PAIR_ACT_BAND:.0f}%.)",
        f"**On {when:%d %b %Y}:**",
    ]
    traps = []
    for b in rows:
        if b.deliv_z is None or b.turnover_vs_norm is None:
            parts.append(f"- {b.label}: not enough history to read")
            continue
        kd, ka = _dz(b.deliv_z), _da(b.turnover_vs_norm)
        line = (f"- {b.label}: delivery {word_d[kd]} (z {b.deliv_z:+.2f}), "
                f"{word_a[ka]} day (Activity {b.turnover_vs_norm:+.0f}%)")
        if (kd, ka) in rule:
            line += f" → {kd}{ka} " + rule[(kd, ka)].replace("**", "")
        if b.deliv_value_cr is not None and b.deliv_value_norm_cr:
            chg = b.deliv_value_cr / b.deliv_value_norm_cr * 100 - 100
            verdict = ("more rupees delivered" if chg > _PAIR_RS_BAND else
                       "fewer rupees delivered" if chg < -_PAIR_RS_BAND else
                       "about the usual rupees delivered")
            line += (f". In rupees: Rs {b.deliv_value_cr:,.0f} Cr against a normal "
                     f"Rs {b.deliv_value_norm_cr:,.0f} Cr ({chg:+.0f}%) — {verdict}.")
            # the illusion: the share reads above normal, the rupees are down
            if b.deliv_z > 0 and chg < -_PAIR_RS_BAND:
                traps.append((b.label, b.deliv_z, b.turnover_vs_norm,
                              b.deliv_value_cr, b.deliv_value_norm_cr, chg))
        parts.append(line)
    for lab, z, act, v, n, chg in traps:
        parts.append(
            f"**{lab} is the trap today.** Its delivery z is positive ({z:+.2f}), "
            f"so the share looks firm — but Activity is {act:+.0f}%, so the day was "
            f"much "
            f"quieter. In rupees it actually delivered Rs {v:,.0f} Cr against a "
            f"normal Rs {n:,.0f} Cr — {abs(chg):.0f}% LESS. The positive delivery "
            f"z is an illusion created by the quiet day.")
    # paragraphs get a blank line between them; bullet lines stay tight
    return "\n\n".join(parts).replace("\n\n- ", "\n- ")

def _ilc_set_date(sessions: list, pick: date | None = None) -> None:
    """Callback: move the tab to `pick` (or to what the date box now holds)."""
    from src.analytics.index_largecap import resolve_session
    ss = st.session_state
    raw = ss.get("ilc_date") if pick is None else pick
    got = resolve_session(sessions, raw) if raw else None
    if got is None:
        return
    ss["ilc_date_keep"] = got
    # remembered only so the tab can say WHY the box shows a different day
    ss["ilc_snapped_from"] = raw if raw != got else None


def _ilc_pick_date(sidebar_date: date) -> tuple[date, dict]:
    """The session this tab shows, and the session calendar it was picked from.

    The sidebar's Trading Date only lists the last 90 sessions; this tab has
    cash history back to 2018 and stock F&O back to 2022, so it carries its own
    picker over every session on file.

    Two rules. (1) The sidebar still drives the tab: changing it moves this
    picker with it, so the two never disagree by accident. (2) The tab's own
    choice lives in a plain session key, not the widget's -- Streamlit drops a
    widget's state when its panel is not on screen, and the date would silently
    reset every time another panel was opened.
    """
    from src.dashboard.cache.queries import cached_ilc_calendar
    try:
        cal = cached_ilc_calendar()
    except Exception:                                          # noqa: BLE001
        cal = {}
    sessions = cal.get("sessions") or []
    if not sessions:
        return sidebar_date, cal
    from src.analytics.index_largecap import resolve_session
    ss = st.session_state
    if ss.get("ilc_sidebar_seen") != sidebar_date:
        ss["ilc_sidebar_seen"] = sidebar_date
        ss["ilc_date_keep"] = sidebar_date
        ss["ilc_snapped_from"] = None
    view = resolve_session(sessions, ss.get("ilc_date_keep", sidebar_date))
    ss["ilc_date_keep"] = view
    ss["ilc_date"] = view          # before the widget is built, so it shows `view`
    i = sessions.index(view)

    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.date_input(
            "Session date", key="ilc_date", format="DD/MM/YYYY",
            min_value=sessions[0], max_value=sessions[-1], width=190,
            on_change=_ilc_set_date, args=(sessions,),
            help="Read this tab as it stood at the close of any session on "
                 f"file ({sessions[0]:%d %b %Y} to {sessions[-1]:%d %b %Y}). "
                 "Pick a weekend or a holiday and it shows the last session "
                 "before it — never the one after, which had not happened yet. "
                 "Changing the sidebar Trading Date moves this with it.")
        st.button("Previous session", icon=":material/chevron_left:",
                  key="ilc_prev", disabled=i == 0, on_click=_ilc_set_date,
                  args=(sessions, sessions[max(i - 1, 0)]))
        st.button("Next session", icon=":material/chevron_right:",
                  icon_position="right", key="ilc_next",
                  disabled=i == len(sessions) - 1, on_click=_ilc_set_date,
                  args=(sessions, sessions[min(i + 1, len(sessions) - 1)]))
        st.button("Latest", icon=":material/last_page:", key="ilc_latest",
                  disabled=view == sessions[-1], on_click=_ilc_set_date,
                  args=(sessions, sessions[-1]))
        if view != sidebar_date:
            st.button(f"Sidebar date ({sidebar_date:%d %b %Y})",
                      icon=":material/undo:", key="ilc_to_sidebar",
                      on_click=_ilc_set_date, args=(sessions, sidebar_date))

    snapped = ss.get("ilc_snapped_from")
    if snapped and snapped != view:
        st.caption(
            f"{snapped:%d %b %Y (%a)} was not a trading session — showing the "
            f"last one before it, **{view:%d %b %Y (%a)}**.")
    if view != sessions[-1]:
        behind = len(sessions) - 1 - i
        st.caption(
            f"🕰️ **Past session: {view:%d %b %Y (%a)}** — {behind} "
            f"session{'s' if behind != 1 else ''} before the latest "
            f"({sessions[-1]:%d %b %Y}). Every table below is computed only "
            "from data up to that close. Where the text says *today* it means "
            "this session.")
    return view, cal


def _render_ilc_replay_notes(sym: str, d, meta: dict, cal: dict,
                             view: date, n_all: int, name: str) -> None:
    """What a PAST date can and cannot be trusted for on this tab."""
    from src.analytics.index_largecap import BASKET_AS_OF, reviews_since
    notes = []
    n_rev = reviews_since(view)
    if meta.get("effective") is None and n_rev:
        missing = n_all - d.n_present
        notes.append(
            f"**The basket is not the one {name} held that day.** Members and "
            f"bucket ranks are NSE's {BASKET_AS_OF:%d %b %Y} list, and this "
            f"session is about **{n_rev} index review"
            f"{'s' if n_rev != 1 else ''}** earlier (NSE reconstitutes every "
            "March and September; this project holds no dated membership "
            "before 2026). The index return is the real index of that day; "
            "the equal-weight basket, breadth and buckets are the later list. "
            "So the carry spread is **not** an exact identity here, and names "
            "that were promoted later *because* they outperformed flatter the "
            "breadth."
            + (f" {missing} of the {n_all} have no price row on this date at "
               "all (not yet listed, or suspended)." if missing > 0 else ""))
        if view < date(2026, 3, 1):
            notes.append(
                f"**The ±{d.carry_band:.2f} pp top-heavy / broad band is "
                "today's.** It is calibrated on sessions since March 2026; "
                "earlier spreads ran wider on Bank and Fin Nifty, so the chip "
                "under the carry spread under-calls old top-heavy days there.")
    fno_from = cal.get("fno_from")
    if fno_from and view < fno_from:
        notes.append(
            f"**Cash market only.** Stock futures and options history starts "
            f"{fno_from:%d %b %Y}, so futures OI, options OI, the cycle table, "
            "the stock votes and the historical matches are empty for this "
            "date. Return, breadth, carry spread and delivery are real.")
    if notes:
        st.warning("\n\n".join(notes))


def _render_ilc_hindsight(sym: str, name: str, view: date, cal: dict) -> None:
    """What the index did next -- folded away, so the tab is read blind first."""
    sessions = cal.get("sessions") or []
    if not sessions or view >= sessions[-1]:
        return
    from src.dashboard.cache.queries import cached_realised_forward
    try:
        rz = cached_realised_forward(view, sym)
    except Exception:                                          # noqa: BLE001
        return
    if rz.get("gap") is None and rz.get("d1") is None:
        return
    with st.expander(f"🔎 Hindsight — what {name} did after this close",
                     expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Next open (gap)",
                  "-" if rz["gap"] is None else f"{rz['gap']:+.2f}%",
                  help="Next session's open against this session's close.")
        c2.metric("Next session",
                  "-" if rz["d1"] is None else f"{rz['d1']:+.2f}%",
                  help=(f"Close of {rz['d1_date']:%d %b %Y} against this close."
                        if rz.get("d1_date") else None))
        c3.metric("Next 5 sessions",
                  "not yet" if rz["d5"] is None else f"{rz['d5']:+.2f}%",
                  help=(f"Close of {rz['d5_date']:%d %b %Y} against this close."
                        if rz.get("d5_date") else
                        "Five sessions have not passed since this date."))
        st.caption(
            "Not known at that close, and nothing below uses it. It is here so "
            "a past read can be checked against what followed — one session is "
            "an anecdote, the measured hit rates further down are the record.")


def _ilc_no_fno_note(view: date) -> str:
    """Why a date has no futures / options columns -- three different reasons."""
    from src.dashboard.cache.queries import cached_ilc_calendar
    try:
        cal = cached_ilc_calendar()
    except Exception:                                          # noqa: BLE001
        cal = {}
    lo, hi, have = cal.get("fno_from"), cal.get("fno_to"), cal.get("fno") or set()
    if lo and view < lo:
        return (f"**No stock F&O data for {view:%d %b %Y}.** That history starts "
                f"{lo:%d %b %Y}; before it the futures and options columns are "
                "empty and the score runs on delivery alone.")
    if lo and view == lo:
        return (f"**{view:%d %b %Y} is the first F&O session on file.** An OI "
                "change needs a prior session to compare against, so the "
                "futures and options columns start from the next one.")
    if hi and view <= hi:
        which = "this session" if view not in have else "the session before it"
        return (f"**F&O file missing for {which}.** The cash-market file for "
                f"{view:%d %b %Y} is loaded but the futures and options one is "
                "not, so those columns are empty and the score runs on delivery "
                "alone. A hole in the stored history, not a quiet day.")
    return (f"**No F&O data for {view:%d %b %Y} yet.** The cash-market "
            "file is loaded but the futures and options file for this session "
            "(or the one before it) is not, so the futures and options columns "
            "are empty and the score runs on delivery alone. This is a data gap, "
            "not a quiet day - NSE publishes the F&O file later in the evening.")


def _render_index_largecap(selected_date: date, min_turnover: float) -> None:
    from src.dashboard.cache.queries import (cached_index_largecap,
                                             cached_concentration_trend,
                                             cached_state_base_rates,
                                             cached_flow_analogues,
                                             cached_bucket_analogues)
    from src.analytics.index_largecap import INDEX_BUCKETS, index_meta
    sym = st.segmented_control(
        "Index", list(INDEX_BUCKETS), default="NIFTY", key="ilc_index",
        format_func=lambda s: INDEX_BUCKETS[s]["display"],
        help="Each index is split on the widest gaps in its OWN published "
             "weights, so the bucket sizes differ by index — and by date, since "
             "NSE reconstitutes every March and September.") or "NIFTY"
    # From here on `selected_date` is the TAB's date, which starts as the sidebar's.
    selected_date, _cal = _ilc_pick_date(selected_date)
    meta = index_meta(sym, selected_date)
    _labels = list(meta["buckets"])
    _n_all = sum(len(v) for v in meta["buckets"].values())
    _name = meta["display"]
    st.markdown(
        f"{_name} is free-float **cap weighted**, so its {_n_all} members do not "
        "move it equally. This panel splits the index into weight buckets and "
        "shows which one actually carried it - and how often the index and its "
        "own basket disagree.")
    st.caption(
        "The sidebar **Min Traded Value** filter does not apply here: the basket is "
        "fixed index membership, not a liquidity screen. Everything below is "
        "anchored on the **Session date** above.")

    try:
        d = cached_index_largecap(selected_date, sym)
    except Exception as exc:                                   # noqa: BLE001
        st.error(f"Index & Large Cap unavailable: {exc}")
        return
    if not d.data_ok:
        st.info(d.note or "No constituent data for this date.")
        return

    _render_ilc_index_notes(sym, selected_date)
    _render_ilc_replay_notes(sym, d, meta, _cal, selected_date, _n_all, _name)
    _render_ilc_hindsight(sym, _name, selected_date, _cal)
    # The verdict is the honesty rail of this tab, so its CONCLUSION stays on
    # screen; only the evidence folds away. Hiding the whole box would let the
    # tables below read as signals.
    st.markdown(
        f"🧭 **No arrow on this tab.** Nothing here forecasts {_name} once the "
        "day's own close-strength is accounted for — measured, not caution. "
        "Read every table as *what happened*.")
    with st.expander("Why there is no forecast — the evidence", expanded=False):
        if sym != "NIFTY":
            st.warning(_ILC_VERDICT[_ilc_verdict_key(sym, meta)])
        else:
            _render_ilc_nifty_verdict()

    _render_ilc_body(d, sym, meta, _labels, _n_all, _name, selected_date,
                     cached_concentration_trend, cached_state_base_rates,
                     cached_flow_analogues)


def _render_ilc_nifty_verdict() -> None:
    # ── honesty rail: this panel describes, it does not forecast ─────────────
    # Re-measured 29 Sep 2026 on NSE's own Top 10 / Top 20 buckets. The first
    # version of this panel measured a mis-bucketed list (3 of the Top 10 and 5
    # of the Next 10 wrong); every script was re-run with the corrected lists and
    # the verdict held -- or, for the three-way gate, got weaker.
    st.warning(
        "**This is a decomposition, not a forecast - and that is measured, not "
        "caution.** Re-tested end to end on NSE's own Top-10 / Top-20 buckets "
        "(`scripts/nifty_bucket_backtest_v2.py` + stages 2-4): a 1,322-session "
        "cash panel (2021+) and a 1,173-session F&O panel (2022+), 200 "
        "candidates.\n\n"
        "- **Futures and options OI add nothing as a family.** Reality check over "
        "all 140 F&O candidates: **p=0.39** (best |t| 3.14 against a null 95th "
        "percentile of 4.42).\n"
        "- **Breadth is close-strength wearing a hat.** Top-10 breadth predicts "
        "the overnight gap at t=+2.65 alone - but in one regression with the "
        "index's own close-strength it goes to **t=-0.49** while CLR holds "
        "**t=+3.81** (they correlate +0.661). Next-10 breadth is the cash panel's "
        "best at |t| 4.17 (p=0.041) and dies the same way: **t=+1.83** beside CLR.\n"
        "- **The week horizon is an overlap artifact.** On strictly disjoint "
        "weeks Top-10 breadth is t=+0.65, and the top-vs-rest divergence swings "
        "-5.2 to +18.6 bps across the five weekday phases.\n"
        "- **The three-way gate is gone.** Heavy top-10 delivery AND rising "
        "futures OI AND price up now pays +27.6bps/5d over base across 38 "
        "episodes; random blocks of matched size beat it **20% of the time** "
        "and dropping its largest episode leaves +7.5bps. On the old mis-bucketed "
        "list it read +57.5 - a result that halves when three stocks change "
        "bucket is not structure.\n"
        "- **One lead, not shipped.** When the Top 10's put OI grows faster than "
        "call OI - measured against what is normal for that point in the expiry "
        "cycle - NIFTY's next 5 sessions averaged **+31 bps (58% up)** against "
        "**-9 bps (51% up)** for the reverse, positive in every year 2022-2026, "
        "and it survives close-strength, momentum and expiry-cycle phase "
        "(t=+3.46). Two cautions. The outcome is itself calendar-driven: the 5 "
        "sessions starting 1-2 days before expiry averaged +57 bps against -8 "
        "bps early in a cycle, and the raw put/call gap swings with the same "
        "calendar, so this lead is only as good as that adjustment. And across "
        "the whole F&O family it is not significant (|t| 3.19 against a null "
        "95th percentile of 3.28); at ~0.11 sd per week, confirming it on fresh "
        "data would take **~6.5 years**. It stays a lead.\n\n"
        "So no arrow is shown. The state table below is a **base rate**, not a "
        "call.")


def _render_ilc_body(d, sym, meta, _labels, _n_all, _name, selected_date,
                     cached_concentration_trend, cached_state_base_rates,
                     cached_flow_analogues) -> None:
    _top, _tail = _labels[0], _labels[-1]
    _mid = _labels[1] if len(_labels) > 2 else None
    _top_names = ", ".join(meta["buckets"][_top][:3])
    _bucket_help = (
        f"Which slice of the {_name}. {_top} = the "
        f"{len(meta['buckets'][_top])} biggest by index weight ({_top_names}"
        f"{'...' if len(meta['buckets'][_top]) > 3 else ''})."
        + (f" {_mid} = the next {len(meta['buckets'][_mid])}." if _mid else "")
        + f" {_tail} = everything else.")

    # ── today's decomposition ────────────────────────────────────────────────
    st.markdown(f"#### 📐 {d.display} - {selected_date:%d %b %Y}")
    if d.carry_spread is None:
        st.info(d.note or "Index return unavailable - carry spread hidden.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Index (cap-weighted)",
              f"{d.index_ret:+.2f}%" if d.index_ret is not None else "-")
    c2.metric("Equal-weight basket",
              f"{d.equal_ret:+.2f}%" if d.equal_ret is not None else "-",
              help=f"Plain mean of the {_n_all} constituents. No weights needed "
                   "- exact.")
    c3.metric("Carry spread",
              f"{d.carry_spread:+.2f} pp" if d.carry_spread is not None else "-",
              help="Index minus equal-weight. Positive = heavyweights carrying it; "
                   "negative = the broad basket is beating the index. Called "
                   f"top-heavy / broad beyond ±{d.carry_band:.2f} pp for "
                   f"{_name} — the same percentile of its own history that "
                   "Nifty's ±0.15 pp sits at, because a small index's spread is "
                   "naturally wider.")
    c4.metric("Advancing", f"{d.adv_all} / {d.n_present}")

    _col = ("#00C853" if "carried" in d.regime else
            "#FF5252" if "dragged" in d.regime else "#78909C")
    st.markdown(_ilc_chip(d.regime, _col), unsafe_allow_html=True)

    # ── bucket table ─────────────────────────────────────────────────────────
    st.markdown("##### Weight buckets")
    st.expander("Why these are membership buckets, not index weights",
                expanded=False).caption(
        "Buckets are **membership by published index weight**, not the weights "
        "themselves - the DB holds no free-float data and weights are *not* "
        "recoverable from the index series (a returns fit reproduces Nifty at "
        "OOS R-squared 0.957 yet drops HDFCBANK out of the top 20; a price-levels "
        "fit gets rank correlation -0.06). Returns below are **equal-weighted "
        "inside each bucket**, so they are not index points.")
    rows = []
    for b in d.rows:
        lean = _ILC_FUT_READ[b.fut_read]
        rows.append({
            "Bucket": b.label + (" ⚠" if b.thin else ""),
            "Return %": None if b.ret_pct is None else round(b.ret_pct, 2),
            "Adv": f"{b.adv}/{b.n_present}",
            "Adv %": None if b.adv_pct is None else round(b.adv_pct, 0),
            "Delivery % (last 4)": (", ".join(f"{v:.1f}" for v in b.deliv_trail)
                                    if b.deliv_trail else
                                    ("-" if b.deliv_pct is None
                                     else f"{b.deliv_pct:.1f}")),
            "Deliv z": None if b.deliv_z is None else round(b.deliv_z, 2),
            "Futures OI %": _ilc_oi_cell(b.fut_oi_pct),
            "Futures read": lean,
            "F&O names": f"{b.fut_valid}/{b.n_members}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={
                     "Bucket": st.column_config.TextColumn(
                         "Bucket", help=_bucket_help),
                     "Return %": st.column_config.NumberColumn(
                         "Return %", format="%.2f",
                         help="Plain average of the bucket's stock returns today. "
                              "If 5 stocks rose 1% and 5 fell 1%, this is 0.00. "
                              "It is NOT the bucket's contribution to the index."),
                     "Adv": st.column_config.TextColumn(
                         "Adv",
                         help="How many of the bucket closed UP. '4/10' means 4 of "
                              "the 10 rose and 6 fell."),
                     "Adv %": st.column_config.NumberColumn(
                         "Adv %", format="%.0f",
                         help="The same thing as a percentage. 4 of 10 up = 40. "
                              "Above 50 means most of the bucket rose."),
                     "Delivery % (last 4)": st.column_config.TextColumn(
                         "Delivery % (last 4)",
                         help="Of all shares traded, the share actually DELIVERED "
                              "to buyers instead of bought and sold the same day. "
                              "65 means 65% was real buying, not intraday churn. "
                              "FOUR sessions, TODAY FIRST then going back, so the "
                              "direction is visible: '60.4, 65.6, 57.5, 63.7' means "
                              "today 60.4 against 65.6 yesterday. Big stocks always "
                              "sit higher than small ones, so judge the LEVEL with "
                              "Deliv z, not with these raw numbers."),
                     "Futures OI %": st.column_config.TextColumn(
                         "Futures OI %",
                         help="Change in the number of open futures contracts vs "
                              "yesterday, averaged across the bucket. +2 means 2% "
                              "more open positions (new money coming in); -2 means "
                              "positions were closed."),
                     "Deliv z": st.column_config.NumberColumn(
                         "Deliv z", format="%.2f",
                         help="Mean of each stock's delivery z against ITS OWN "
                              "21-session normal. Per-symbol, not a z of the bucket "
                              "mean: the Rest-30 mean had correlation +0.674 with "
                              "how many of its names reported that day, so a "
                              "bucket-level z was partly a coverage signal."),
                     "Futures read": st.column_config.TextColumn(
                         "Futures read",
                         help="How each bucket's label is built, one session at a time. "
                              "For every stock: total futures OI across ALL live "
                              "expiry months, today vs the previous session, "
                              "against the stock's cash close-to-close move. OI "
                              "must move more than 0.5% to count. Price up + OI up "
                              "= long build; price up + OI down = short covering; "
                              "price down + OI up = short build; price down + OI "
                              "down = long unwind. The bucket is bullish when "
                              "(long + cover) outnumber (short + unwind) by more "
                              "than 1.5x, bearish the other way, else mixed - and "
                              "it is NAMED by whichever of the two dominates, "
                              "because fresh positions and closed positions are "
                              "different things (measured: a bullish bucket is "
                              "mostly short covering on 37-60% of such days). "
                              "Limits: OI cannot show who initiated a trade, so "
                              "the four names are a desk convention; volume is not "
                              "used; and 14% of votes come from a price move under "
                              "0.25% (ignoring them never flips a bucket's side, "
                              "it only moves 10-22% of days between a lean and "
                              "'mixed')."),
                     "F&O names": st.column_config.TextColumn(
                         "F&O names",
                         help="How many of the bucket's stocks actually have "
                              "futures trading. '10/10' means all ten - a solid "
                              "read. '4/30' would mean only 4 of 30 had data, so "
                              "the futures columns beside it are thin, not calm."),
                 })

    for b in d.rows:
        if b.movers_up or b.movers_dn:
            up = " · ".join(f"{s} {v:+.1f}%" for s, v in b.movers_up) or "-"
            dn = " · ".join(f"{s} {v:+.1f}%" for s, v in b.movers_dn) or "-"
            st.markdown(
                f"**{b.label}** &nbsp; 🟢 {up} &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 {dn}",
                unsafe_allow_html=True)

    # ── flow summary: delivery + futures + options, per bucket ──────────────
    if d.unusual_session and d.unusual_kind == "heavy":
        st.warning(
            f"**Heavy, basket-wide session — {d.unusual_session}.** Index funds "
            "and ETFs trade a rebalance at the closing price, and those trades are "
            "delivery by construction, so every bucket reads 'heavy delivery' and "
            "the score leans long for a mechanical reason, not a conviction one. "
            "Measured: every NSE rebalance close checked sits beyond the ordinary-"
            "day 95th percentile of both basket delivery z (0.62) and activity "
            "(+41%) — 27 Sep 2024 z +0.96 / +70%, 29 Sep 2025 +0.79 / +94%. Read "
            "the delivery column today as the rebalance, not as investors.")
    elif d.unusual_session:
        st.warning(
            f"**This is a special session — {d.unusual_session}.** NSE runs live "
            "sessions on Budget days and for Muhurat trading, and they are short "
            "and thin. The delivery reads below describe the SESSION TYPE more "
            "than investor behaviour: the 2026 Budget Sunday printed delivery z "
            "of -1.75 / -2.30 / -2.36 across the three buckets, while Muhurat "
            "2023 ran the other way at 64% delivery against ~53% on its "
            "neighbours. The numbers are correct; read them with that in mind.")
    st.markdown("##### Flow read — delivery, futures OI, options OI")
    if not d.is_expiry_session and d.rows and all(b.fut_valid == 0 for b in d.rows):
        st.info(_ilc_no_fno_note(selected_date))
    if d.is_expiry_session:
        st.info(
            "**Monthly expiry day.** Futures OI counts every contract **including "
            "the one settling today**, on both days — NSE still reports its OI at "
            "the close, so the total runs straight through the roll (settlement-"
            "day bucket leans match ordinary days: Nifty long/mixed/short 38/22/40 "
            "vs 36/30/34). Its level runs ~+1.5pp above an ordinary day, so read "
            "the lean, not the size. **Options OI is switched off today**: "
            "expiring calls drop away faster than puts, and the read would come "
            "out 'put side heavier' on 72-86% of expiry days against 42-46% "
            "normally — a fake bullish signal every month.")
    if d.front_expiry:
        _dte = d.days_to_expiry
        _when = ("settles today" if _dte == 0 else
                 "settles tomorrow" if _dte == 1 else f"{_dte} days away")
        st.expander("Which expiry months these OI numbers use",
                    expanded=False).caption(
            f"**Which expiry these OI numbers use: all of them.** Front month is "
            f"**{d.front_expiry:%d %b %Y}** ({_when}), holding "
            f"{d.near_oi_share:.0f}% of forward open interest. Every live expiry "
            f"month is counted, with each contract compared against ITSELF on the "
            f"prior session — so a position rolling from this month to the next "
            f"nets to zero instead of reading as an exit. Measured, the front "
            f"contract alone sheds **-37.7% a day at 2 days to expiry** and does "
            f"not settle until 9-12 days out, so a near-month read with a "
            f"late roll would print basket-wide unwinding every month "
            f"(`scripts/ilc_expiry_choice.py`).")
    _flow = []
    for b in d.rows:
        _sc = b.flow_score
        _flow.append({
            "Bucket": b.label,
            "Delivery": ("—" if b.deliv_z is None else
                         "🟢 heavy" if b.deliv_z > 0.3 else
                         "🔴 light" if b.deliv_z < -0.3 else "⚪ normal"),
            "Delivery % (last 4)": (", ".join(f"{v:.1f}" for v in b.deliv_trail)
                                    if b.deliv_trail else "-"),
            "Deliv z": None if b.deliv_z is None else round(b.deliv_z, 2),
            "Activity": (None if b.turnover_vs_norm is None
                         else round(b.turnover_vs_norm, 0)),
            "Futures": _ILC_FUT_READ[b.fut_read],
            "Stock votes": b.fut_votes,
            "Fut OI %": _ilc_oi_cell(b.fut_oi_pct),
            "Call OI %": _ilc_oi_cell(b.ce_oi_pct, d.is_expiry_session),
            "Put OI %": _ilc_oi_cell(b.pe_oi_pct, d.is_expiry_session),
            "Options": {"put writing": "🟢 put writing",
                        "put side heavier": "🟢 put side heavier",
                        "balanced": "⚪ balanced",
                        "call side heavier": "🔴 call side heavier",
                        "call writing": "🔴 call writing",
                        None: ("off · expiry day" if d.is_expiry_session
                               else "—")}[b.opt_read],
            "Bucket score": None if _sc is None else round(_sc, 2),
        })
    _pair_help = _ilc_pair_help(d.rows, selected_date)
    st.dataframe(pd.DataFrame(_flow), hide_index=True, use_container_width=True,
                 column_config={
                     "Bucket": st.column_config.TextColumn(
                         "Bucket", help=_bucket_help),
                     "Delivery": st.column_config.TextColumn(
                         "Delivery",
                         help="Was real buying unusually HEAVY or LIGHT today, "
                              "against these same stocks' own last 21 sessions? "
                              "Heavy = investors taking delivery rather than day "
                              "trading. Marked heavy above +0.3, light below -0.3."),
                     "Delivery % (last 4)": st.column_config.TextColumn(
                         "Delivery % (last 4)",
                         help="The raw delivery share for FOUR sessions, TODAY "
                              "FIRST then going back. '60.4, 65.6, 57.5, 63.7' "
                              "means today 60.4 against 65.6 yesterday - falling. "
                              "The Delivery word beside it judges the LEVEL; this "
                              "shows the DIRECTION."),
                     "Deliv z": st.column_config.NumberColumn(
                         "Deliv z", format="%.2f",
                         help=_pair_help),
                     "Activity": st.column_config.NumberColumn(
                         "Activity %", format="%+d",
                         help=_pair_help),
                     "Futures": st.column_config.TextColumn(
                         "Futures",
                         help="How each bucket's label is built, one session at a time. "
                              "For every stock: total futures OI across ALL live "
                              "expiry months, today vs the previous session, "
                              "against the stock's cash close-to-close move. OI "
                              "must move more than 0.5% to count. Price up + OI up "
                              "= long build; price up + OI down = short covering; "
                              "price down + OI up = short build; price down + OI "
                              "down = long unwind. The bucket is bullish when "
                              "(long + cover) outnumber (short + unwind) by more "
                              "than 1.5x, bearish the other way, else mixed - and "
                              "it is NAMED by whichever of the two dominates, "
                              "because fresh positions and closed positions are "
                              "different things (measured: a bullish bucket is "
                              "mostly short covering on 37-60% of such days). "
                              "Limits: OI cannot show who initiated a trade, so "
                              "the four names are a desk convention; volume is not "
                              "used; and 14% of votes come from a price move under "
                              "0.25% (ignoring them never flips a bucket's side, "
                              "it only moves 10-22% of days between a lean and "
                              "'mixed')."),
                     "Stock votes": st.column_config.TextColumn(
                         "Stock votes",
                         help="The count behind the Futures label, stock by stock. "
                              "'▲ 2 long · 5 cover | ▼ 3 short · 0 unwind' means 7 "
                              "stocks voted bullish (2 fresh longs, 5 shorts "
                              "closing) against 3 bearish - bullish, and mostly "
                              "covering. 'flat' = OI moved less than 0.5%."),
                     "Fut OI %": st.column_config.TextColumn(
                         "Fut OI %",
                         help="How much the number of open futures contracts "
                              "changed vs yesterday. +0.25 = 0.25% more open "
                              "positions. Negative = positions were closed. "
                              "Counted across ALL expiry months, so a position "
                              "rolling to next month is not mistaken for an exit."),
                     "Call OI %": st.column_config.TextColumn(
                         "Call OI %",
                         help="Change in open CALL option positions vs yesterday. "
                              "Rising calls usually means sellers writing calls "
                              "above the market - they expect a ceiling."),
                     "Put OI %": st.column_config.TextColumn(
                         "Put OI %",
                         help="Change in open PUT option positions vs yesterday. "
                              "Rising puts usually means sellers writing puts "
                              "below the market - they expect a floor."),
                     "Options": st.column_config.TextColumn(
                         "Options",
                         help="Which option side grew faster today. For each "
                              "stock, call OI is summed across every strike and "
                              "every expiry, and put OI the same, each compared "
                              "with the previous session; the bucket averages "
                              "them. Example: calls +4.3% and puts +2.0% = 'call "
                              "side heavier'. Within 0.25pp = 'balanced'. 'Put "
                              "writing' / 'call writing' only when one side rose "
                              "and the other fell. Price is not used. "
                              "The gap is judged against what is NORMAL for this point "
                              "in the expiry cycle, not against zero: calls are "
                              "opened faster on a new series and closed faster "
                              "into expiry, so the raw gap runs about -3.6 points "
                              "twenty sessions out and +3.3 the day before expiry. "
                              "Unadjusted, this column said 'call side heavier' on "
                              "70% of days in a cycle's first week and 'put side "
                              "heavier' on 75% the day before expiry. Each index "
                              "gets its own strength of that pattern (Nifty 1.14x, "
                              "Bank 0.93x, Fin 0.95x, Midcap Select 0.62x). "
                              "The bullish "
                              "reading of 'put side heavier' ASSUMES the new puts "
                              "were sold, not bought - OI cannot tell which."),
                     "Bucket score": st.column_config.NumberColumn(
                         "Bucket score", format="%.2f",
                         help="The three columns above averaged into one number, "
                              "from -1 (all bearish) to +1 (all bullish). Example: "
                              "delivery heavy (+1), futures short build (-1), calls "
                              "heavier (-1) averages to -0.33. It SUMMARISES "
                              "positioning; it does not predict the index - the "
                              "measured hit rate is right below this table."),
                 })
    _render_ilc_cycle_flow(d, sym, _name, selected_date)
    _render_ilc_trends(sym, _name, selected_date)
    _render_ilc_stock_flow(d, sym, _name, selected_date)
    _ns = d.net_score
    if _ns is not None:
        _lbl = ("flow leans long" if _ns > 0.25 else
                "flow leans short" if _ns < -0.25 else "flow is mixed")
        _c = "#00C853" if _ns > 0.25 else "#FF5252" if _ns < -0.25 else "#78909C"
        st.markdown(_ilc_chip(f"Net flow score {_ns:+.2f} — {_lbl}", _c),
                    unsafe_allow_html=True)
    if d.is_expiry_session:
        st.caption("Expiry day — the options leg is off (see above), so the score "
                   "runs on delivery and futures only.")
    st.caption("**The score summarises positioning; it does not forecast the "
               "index.** Its measured hit rate is below.")
    _sc = st.expander("Does this score forecast the index? — measured hit rates",
                      expanded=False)
    if sym != "NIFTY":
        _sc.error(_ILC_SCORE_VERDICT[_ilc_verdict_key(sym, meta)])
    else:
      _sc.error(
        "**This score does not forecast the index, and that is measured on the "
        "full four years.** `scripts/ilc_multi_index_backtest.py`, 1,178 sessions "
        "2022-2026, on the exact rule this table uses (NSE's Top-10 / Top-20 "
        "buckets; splits, bonuses and demergers adjusted; options read net of "
        "the expiry cycle):\n\n"
        "| | next open (gap) | next day | next 5 sessions |\n|---|---|---|---|\n"
        "| **net score hit rate** | 54.4% | **52.3%** | **52.1%** |\n"
        "| base rate (up days) | 61.0% | 52.0% | 54.1% |\n"
        "| information coefficient | +0.091 | +0.034 | +0.030 |\n\n"
        "Beside the index's own close-strength the score is t=+0.35 at the gap, "
        "+1.05 next day and +1.02 over 5 sessions - nothing. Out of sample after "
        "May-2024 the next-day read runs **+1.1 bps at 50.4%**, a coin flip. One "
        "fragile spot, stated rather than hidden: the score's *sign* against the "
        "next gap reads t=+1.97 beside close-strength while its *value* reads "
        "+0.35 - two versions of one test disagreeing is noise, not an edge. An "
        "earlier build of this score, on mis-drawn buckets, hit 49.3% next day "
        "(`scripts/nifty_bucket_composite.py`). Read this block as **what "
        "positioning is doing**, never as where the index is going.")
    _render_ilc_forward(sym, _name, selected_date)

    # ── historical match: the actual sessions, and what NIFTY did next ──────
    st.markdown(f"##### When these numbers occurred before — what {_name} did next")
    _fa = cached_flow_analogues(selected_date, sym, 25)
    if not _fa.get("ok"):
        st.info(_fa.get("note") or "No historical match available.")
    else:
        _spot = _fa.get("spot")
        _m = _fa["matches"].copy()

        def _trip(row, leg):
            """Top / Next / Rest for one leg, as one compact cell."""
            return " · ".join(
                f"{row[f'{b} {leg}']:+.2f}" for b in _labels)

        def _opt_word(v):
            return "call" if v < -0.25 else "put" if v > 0.25 else "bal"

        def _optrip(row):
            return " · ".join(
                _opt_word(row[f"{b} options"]) for b in _labels)

        _today = pd.Series(_fa["today"])
        _disp = [{
            "Session": f"{selected_date:%d %b %Y}  ← today",
            "Alike": None,
            "Delivery z": _trip(_today, "delivery"),
            "Futures lean": _trip(_today, "futures"),
            "Call vs put": _optrip(_today),
            # NaN, not None: a None here makes the whole column object-typed and
            # Streamlit prints the literal text "None" in today's row.
            "Next day %": "— not yet", "Next day pts": "—",
            "Next 5 sessions %": "— not yet", "Next 5d pts": "—",
        }]
        _far = float(_m["distance"].max()) or 1.0
        for _, r in _m.iterrows():
            _disp.append({
                "Session": pd.Timestamp(r["date"]).strftime("%d %b %Y"),
                "Alike": round(100 * (1 - r["distance"] / (_far * 1.15)), 0),
                "Delivery z": _trip(r, "delivery"),
                "Futures lean": _trip(r, "futures"),
                "Call vs put": _optrip(r),
                "Next day %": _ilc_num(r["d1"]),
                "Next day pts": _ilc_num(r.get("d1_pts"), "{:+,.0f}"),
                "Next 5 sessions %": _ilc_num(r["d5"]),
                "Next 5d pts": _ilc_num(r.get("d5_pts"), "{:+,.0f}"),
            })
        st.dataframe(
            pd.DataFrame(_disp), hide_index=True, use_container_width=True,
            column_config={
                "Session": st.column_config.TextColumn(
                    "Session",
                    help="A past trading day whose nine flow numbers looked like "
                         "today's. The first row is today itself, for comparison - "
                         "read across it, then down."),
                "Alike": st.column_config.ProgressColumn(
                    "Alike", format="%d%%", min_value=0, max_value=100,
                    help="How closely that day matched today across all nine "
                         "numbers. 100% would be an identical day. These are the "
                         "closest days on record, not perfect twins - the top "
                         "match is typically 30-45%."),
                "Delivery z": st.column_config.TextColumn(
                    f"Delivery z  ({' · '.join(_labels)})",
                    help=f"Three numbers: {', then '.join(_labels)}. Each "
                         "is that bucket's delivery against its own 21-session "
                         "normal. Example: today reads +1.13 / +0.58 / +0.42, and "
                         "a row showing +1.29 / +0.63 / +0.44 is a close match."),
                "Futures lean": st.column_config.TextColumn(
                    f"Futures lean  ({' · '.join(_labels)})",
                    help="The same lean the Flow read table names, as one "
                         "number per bucket: the share of its stocks voting "
                         "bullish (long build or short covering) minus the "
                         "share voting bearish (short build or long unwind). "
                         "+1.00 = every stock bullish, -1.00 = every stock "
                         "bearish, 0 = split or flat. Example: 4 covering, 1 "
                         "long, 1 short and 4 flat out of 10 reads +0.40. A "
                         "matched row with the same sign as today had the same "
                         "side of futures lean as today. Until 10 Oct 2026 "
                         "this column was OI change signed by price, which "
                         "printed short covering and short build with the SAME "
                         "sign - one matched row in four had the opposite lean "
                         "to the day it was matched to; now one in twenty."),
                "Call vs put": st.column_config.TextColumn(
                    f"Call vs put  ({' · '.join(_labels)})",
                    help="Which option side grew faster in each bucket. "
                         "'call' = calls grew faster (leans bearish), "
                         "'put' = puts grew faster (leans bullish), "
                         "'bal' = neither. Example: 'call · call · bal'."),
                "Next day pts": st.column_config.TextColumn(
                    "Next day pts",
                    help="That session's % move applied to TODAY'S level"
                         + (f" ({_spot:,.0f})" if _spot else "")
                         + ". The index level has moved a long way since 2022, "
                           "so raw historical points are not comparable across "
                           "the window."),
                "Next day %": st.column_config.TextColumn(
                    "Next day %",
                    help=f"What {_name} actually did on the session AFTER that day. "
                         "+0.25 means it closed 0.25% higher the next day."),
                "Next 5 sessions %": st.column_config.TextColumn(
                    "Next 5 sessions %",
                    help=f"What {_name} did over the following FIVE sessions - "
                         "roughly the next week - measured from that day's close."),
                "Next 5d pts": st.column_config.TextColumn(
                    "Next 5d pts",
                    help=f"The same week move in {_name} points at today's level, so "
                         "old and recent rows are directly comparable."),
            })
        _s1, _s5 = _fa["summary"]["d1"], _fa["summary"]["d5"]
        _up1, _up5 = int(round(_s1["up"] * _s1["n"] / 100)), int(round(_s5["up"] * _s5["n"] / 100))
        # Built in pieces on purpose: an inline `... if cond else "."` here binds
        # to the WHOLE concatenation, so a missing index close collapsed the entire
        # sentence to a single full stop.
        _txt = (
            f"**Of these {_s1['n']} sessions: {_up1} rose the next day "
            f"({_s1['up']:.0f}%, base {_s1['base_up']:.0f}%) and {_up5} rose over "
            f"the next 5 ({_s5['up']:.0f}%, base {_s5['base_up']:.0f}%).** "
            f"Next day averaged {_s1['mean']:+.2f}%")
        if _s1.get("mean_pts") is not None:
            _txt += f" ({_s1['mean_pts']:+,.0f} pts)"
        _txt += f", the week {_s5['mean']:+.2f}%"
        if _s5.get("mean_pts") is not None:
            _txt += f" ({_s5['mean_pts']:+,.0f} pts)"
        _txt += "."
        if _s5.get("best_pts") is not None and _s5.get("worst_pts") is not None:
            _txt += (f" Range on the week: {_s5['worst_pts']:+,.0f} to "
                     f"{_s5['best_pts']:+,.0f} pts.")
        st.markdown(_txt)
        _mq = _fa["summary"]["match_quality"]
        st.caption(
            f"Every session back to 2022, using only days BEFORE "
            f"{selected_date:%d %b %Y} and skipping the last {_fa['purge']} "
            f"(a neighbour from last week shares most of its forward window with "
            f"this one). The 25th match sits at distance {_mq['kth']:.2f} against "
            f"{_mq['median_random']:.2f} for a typical day.")
        # Three things the headline sentence hides, each measured on this table.
        _md = sorted(pd.to_datetime(_m["date"]).dt.date)
        _indep = 1 + sum(1 for a, b in zip(_md[:-1], _md[1:])
                         if np.busday_count(a, b) > 5)
        _kr1, _kr5 = _s1.get("k_range"), _s5.get("k_range")
        _bits = [
            f"**{len(_md)} rows, but {_indep} separate episodes** — matches "
            "within a week of each other share most of their next-5-session "
            "window, so the week figure rests on fewer independent cases than "
            "it looks."]
        if _kr1 and _kr5:
            _bits.append(
                "**The averages move with how many matches are taken** (10 to "
                f"60): next day {_kr1[0]:+.2f}% to {_kr1[1]:+.2f}%, the week "
                f"{_kr5[0]:+.2f}% to {_kr5[1]:+.2f}%"
                + (" — the week figure changes SIGN, so it says nothing."
                   if _s5.get("k_sign_flips") else ".")
                + (" The next-day figure changes sign too."
                   if _s1.get("k_sign_flips") else ""))
        _bits.append(
            "**'Alike' is a relative scale**, set so the 25th match always "
            "reads about 13% — it ranks these rows against each other and is "
            "not a percentage of similarity.")
        _bits.append(
            "**Every row uses the buckets in force on "
            f"{selected_date:%d %b %Y}**, applied back through history, so "
            "like is compared with like. Open an old row with the Session "
            "date picker and its own table can differ slightly, because that "
            "day is shown with the buckets it had then.")
        st.caption("\n\n".join(_bits))
        st.caption("**These matches are a record, not a forecast** — similar "
                   "past days did not predict the next move.")
        _an = st.expander("Do similar past days predict the next move? — the evidence",
                          expanded=False)
        if sym != "NIFTY":
            _an.caption(
                f"**Measured on {_name} too: the matches do not forecast it.** "
                "Walk-forward on 664 sessions (Dec-2023 to Oct-2026), the "
                "average next day of the 25 matches against what actually "
                "followed: rank correlation +0.01 on Bank Nifty and +0.02 on "
                "Fin Nifty, neither distinguishable from zero. Nifty 50 reads "
                "+0.16 on the same test and is the only one that does, which "
                "is why it is not believed there either. Read the rows as "
                "what happened, not as what will.")
        else:
          _an.error(
            "**A lead on Nifty, not a forecast.** Walk-forward on 664 sessions "
            "(Dec-2023 to Oct-2026): the average next day of the 25 matches "
            "lined up with what followed - rank correlation **+0.16**, right "
            "direction 55.1% against 51.8% for 25 random past days, positive in "
            "2024, 2025 and 2026, and it holds beside the index's own "
            "close-strength and the day's return (t +3.63). Three reasons it "
            "is not shown as a call:\n\n"
            "- **It does not repeat.** Same code, same window: Bank Nifty "
            "+0.01, Fin Nifty +0.02.\n"
            "- **It depends on the match count.** +0.07 with 10 matches, +0.16 "
            "with 25, +0.14 with 50, +0.04 with 100. A real effect should not "
            "peak at the one setting this table uses.\n"
            "- **Most of it cannot be traded.** The F&O file arrives after the "
            "close. From the next open to the next close it is +0.09 "
            "(t +1.99).\n\n"
            "Before 10 Oct 2026 this table matched on a futures number that "
            "could not tell short covering from short build, and on that "
            "version the nearest days forecast no better than the farthest "
            "(1.389% vs 1.470% error). The +0.16 appeared when that was "
            "corrected, on one index out of three - a result to watch on fresh "
            "sessions, not to act on.")

    # ── state base rates ─────────────────────────────────────────────────────
    st.markdown("##### What has followed this state, historically")
    _st = d.state
    if _st is None:
        st.caption("Today's state is unavailable — a bucket is too thin to "
                   "classify, so no base rate is shown.")
    br = cached_state_base_rates(sym, 5, selected_date)
    if br.empty:
        st.caption("Not enough history for state base rates.")
    else:
        if _st:
            _r = br[br["state"] == _st]
            _a = br[br["state"].str.startswith("—")]
            if not _r.empty and not _a.empty:
                r0, a0 = _r.iloc[0], _a.iloc[0]
                st.markdown(_ilc_chip(
                    f"Today: {_st} &nbsp;·&nbsp; {int(r0['days'])} prior sessions",
                    "#42A5F5"), unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                for col, lab, kb, ku in [
                        (m1, "Next open (gap)", "gap_bps", "gap_up"),
                        (m2, "Next session", "d1_bps", "d1_up"),
                        (m3, "Next 5 sessions", "d5_bps", "d5_up")]:
                    col.metric(lab, f"{r0[kb]:+.1f} bps",
                               f"{r0[kb]-a0[kb]:+.1f} vs all sessions",
                               delta_color="off",
                               help=f"Up {r0[ku]:.1f}% of the time, against "
                                    f"{a0[ku]:.1f}% unconditionally. A BASE RATE "
                                    f"over {int(a0['days'])} sessions, not a "
                                    f"forecast.")
        st.dataframe(
            br.rename(columns={
                "state": "State", "days": "Days", "gap_bps": "Gap bps",
                "gap_up": "Gap up %", "d1_bps": "1d bps", "d1_up": "1d up %",
                "d5_bps": "5d bps", "d5_up": "5d up %"}),
            hide_index=True, use_container_width=True,
            column_config={
                "State": st.column_config.TextColumn(
                    "State",
                    help=f"Which end of the index rose. 'Broad up' = both the "
                         f"{_top} and the {_tail} mostly rose. 'Heavyweights "
                         f"carried' = the {_top} rose while most of the {_tail} "
                         "fell - the narrow rally. 'Heavyweights lagged' is the "
                         "reverse. Every session falls in exactly one."),
                "Days": st.column_config.NumberColumn(
                    "Days", format="%d",
                    help="How many past sessions were in this state. The bottom "
                         "row is every session, for comparison."),
                "Gap bps": st.column_config.NumberColumn(
                    "Gap bps", format="%.1f",
                    help="Average OVERNIGHT move: from that day's close to the "
                         "next morning's open. 100 bps = 1.00%, so 16.1 means "
                         "about 0.16% of the index."),
                "Gap up %": st.column_config.NumberColumn(
                    "Gap up %", format="%.1f",
                    help="How often the next morning opened HIGHER. Compare with "
                         "the bottom row: if a state shows 69% against a 62% "
                         "all-sessions rate, it opened up more often than usual."),
                "1d bps": st.column_config.NumberColumn(
                    "1d bps", format="%.1f",
                    help="Average close-to-close move on the NEXT session. "
                         "100 bps = 1.00%."),
                "1d up %": st.column_config.NumberColumn(
                    "1d up %", format="%.1f",
                    help="How often the next session closed higher. The "
                         "all-sessions rate is about 52%, so anything near 52 is "
                         "no better than a coin flip."),
                "5d bps": st.column_config.NumberColumn(
                    "5d bps", format="%.1f",
                    help="Average move over the next FIVE sessions - about a week. "
                         "Treat this column as unmeasurable: this data cannot "
                         "resolve anything smaller than roughly 36 bps at a week."),
                "5d up %": st.column_config.NumberColumn(
                    "5d up %", format="%.1f",
                    help=f"How often {_name} was higher a week later."),
            })
        st.caption(
            "Every session lands in exactly one state, so this is the record, "
            "not a fitted rule. **The gap column is the only one this data can "
            "resolve** — its minimum detectable edge is ~5bps, against ~36bps for "
            "the 5-day column on non-overlapping windows. Read the 5d numbers as "
            "*not measurable here*, not as *real*. And the gap column's ordering "
            "is close-strength: control for CLR and the breadth signal goes to "
            + ("t=-0.49." if sym == "NIFTY" else
               "no longer significant (see the verdict at the top of this tab).")
            + f" Survivorship applies — this is today's {_n_all} applied to "
            "history."
            + ("" if sym == "NIFTY" else
               f" The **{_top}** held ~55% of {_name}'s moves until 2025 and ~35% "
               "since the SEBI weight cap bit in 2026, so older rows describe a "
               "more top-heavy index than today's."
               if sym in ("BANKNIFTY", "FINNIFTY") else
               f" History starts 29 Nov 2024 (when most of today's {_top} got "
               "stock F&O), so each row rests on few sessions — the 5d column "
               "especially."))

    # ── the measured trend ───────────────────────────────────────────────────
    with st.expander("📈 How often does the index disagree with its own basket?",
                     expanded=False):
        tr = cached_concentration_trend(sym, 5, selected_date)
        if tr.empty:
            st.info("Not enough history for the concentration trend.")
        else:
            st.dataframe(
                tr.rename(columns={
                    "year": "Year", "sessions": "Sessions",
                    "abs_spread": "|Carry spread| pp",
                    "carried_days": "Carried (days)", "carried_pct": "Carried %",
                    "dragged_days": "Dragged (days)", "dragged_pct": "Dragged %"}),
                hide_index=True, use_container_width=True,
                column_config={
                    "Year": st.column_config.NumberColumn(
                        "Year", format="%d",
                        help="Calendar year. Only full years are shown."),
                    "Sessions": st.column_config.NumberColumn(
                        "Sessions", format="%d",
                        help="Trading days counted that year. Part-years are "
                              "dropped - a 100-day stub is not comparable with a "
                              "full year."),
                    "|Carry spread| pp": st.column_config.NumberColumn(
                        "|Carry spread| pp", format="%.3f",
                        help="How far the index typically drifted from a simple "
                             f"average of its {_n_all} stocks, ignoring direction. "
                             "Bigger = the heavyweights and the rest went their "
                             "own ways more often."),
                    "Carried (days)": st.column_config.NumberColumn(
                        "Carried (days)", format="%d",
                        help="Days the index CLOSED UP while fewer than half its "
                             f"own {_n_all} stocks rose - the heavyweights dragged it "
                             "green on their own."),
                    "Carried %": st.column_config.NumberColumn(
                        "Carried %", format="%.1f",
                        help="The same as a share of that year's sessions. Do NOT "
                             "read the column as a trend - see the note below the "
                             "table; it does not replicate on a basket that is not "
                             "today's index list."),
                    "Dragged (days)": st.column_config.NumberColumn(
                        "Dragged (days)", format="%d",
                        help="The opposite: the index closed DOWN even though more "
                             "than half its stocks rose."),
                    "Dragged %": st.column_config.NumberColumn(
                        "Dragged %", format="%.1f",
                        help="The same as a share of that year's sessions."),
                })
            if sym != "NIFTY":
                st.caption(
                    "**Carried** = the index rose while under half its own "
                    "members did. **Dragged** = it fell while over half "
                    f"advanced. **A record of today's {_n_all}, not a trend** — "
                    "the same survivorship that broke Nifty's apparent rise "
                    "applies here, and was not re-tested per index. Partial "
                    "calendar years are dropped.")
            else:
              st.caption(
                "**Carried** = the index rose while under half its own members did. "
                "**Dragged** = it fell while over half advanced.\n\n"
                "**Read this as a record of today's 50, not as a trend.** On this "
                "basket the carried share looks monotone (2.4% to 7.2%), and that "
                "was this panel's original headline - but it does **not** replicate. "
                "Restricted to the 46 names with full history it is 2.4 / 3.7 / 7.3 / "
                "5.6 / 6.6 (not monotone), and on a **membership-independent** basket "
                "of ~186 liquid names a day it is 8.5 / 9.1 / 14.2 / 8.1 / 8.4 - a "
                "2024 spike that reverts, not a rise. The likely cause is "
                "survivorship: this is TODAY's list, and names promoted into the "
                "index *because* they outperformed inflate the early years' breadth. "
                "Extending the window back settles it - 2021 reads **11.7%**, higher "
                "than 2026, so the 'rise' was an artifact of starting at 2022, the "
                "series minimum. "
                "A point-in-time membership table would settle it. Partial calendar "
                "years are dropped - a 100-session stub read 14.0% beside a 2.4% "
                "full year, a truncated denominator, not a reversal.")
