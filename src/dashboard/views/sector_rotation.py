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
                             "near_opt_label", "next_opt_label", "far_opt_label"]
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
                            "near_fut_label", "next_fut_label", "far_fut_label",
                            "near_opt_label", "next_opt_label", "far_opt_label",
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
        f"<div style='display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:3px;font-size:11px;color:rgba(255,255,255,0.55)'>"
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

    # Bubble chart
    st.plotly_chart(
        _rotation_clock_chart(df, sel, nifty_return=nifty_ret,
                              center=(df["sector_median_ret"].iloc[0] if "sector_median_ret" in df.columns else None)),
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
                    _phase_card(row, "#00c853")
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


def _render_forward_tilt(selected_date: date, min_turnover: float) -> None:
    """Validated 1–2 week cross-sectional momentum tilt (regime-gated)."""
    st.markdown("#### 🎯 1–2 Week Forward Tilt &nbsp; <sub>β</sub>", unsafe_allow_html=True)
    st.caption(
        "The **only** sector call that survived deep validation: cross-sectional "
        "**momentum** (relative strength vs Nifty) predicts 1–2wk forward returns — "
        "daily-IC t≈9, Monte-Carlo p<0.002 vs 600 random portfolios, cost- and "
        "sub-period-robust. Measured accuracy: an OVERWEIGHT sector beats the median "
        "sector **~55%** of the time on its own; a **causal sector-persistence gate** "
        "(drops historically mean-reverting sectors like Realty / Banking / Consumer "
        "Durables) lifts that to **~60%** and the OW-vs-UW basket wins ~64% of 5-day "
        "rebalances. A statistical lean (~1–2% relative / 2wk), not a per-call oracle."
    )

    try:
        df, regime = cached_forward_tilt(selected_date, min_turnover)
    except Exception as exc:                                  # noqa: BLE001
        st.error(f"Forward tilt unavailable: {exc}")
        return
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
        _ENTRY = {"BEST": ("#16a34a", "BEST ENTRY WINDOW"), "EXTENDED": ("#d97706", "EXTENDED — DON'T CHASE"),
                  "MIXED": ("#d97706", "MIXED / TRANSITION"), "WEAK": ("#dc2626", "WEAK — RISKY"),
                  "STANDDOWN": ("#dc2626", "BROAD DOWNTREND — WAIT")}
        ec, elabel = _ENTRY.get(mtf["entry"], ("#8a8f98", mtf["entry"]))
        cells = "".join(
            f"<span style='display:inline-block;min-width:118px'>"
            f"<b>{k.upper()}</b> <span style='color:#8a8f98'>({v['horizon']})</span> "
            f"{_SICON.get(v['state'],'')}{' 🔄' if v['flipped'] else ''}</span>"
            for k, v in mtf["bands"].items())
        _mtf_tip = ("Market trend at 4 horizons (Nifty vs its EMA + slope). 8yr backtest: 3/4-up = "
                    "the best entry (leaders building, not exhausted); 4/4-up = EXTENDED, often near "
                    "a top (don't chase); all-down = wait/bottom-fish, don't short (bounces). "
                    "Longer-timeframe turns can't be forecast — this is a concurrent read. 🔄 = a "
                    "band just flipped.").replace('"', "'")
        st.markdown(
            f"<div title=\"{_mtf_tip}\" style='border:1px solid {ec}55;border-radius:8px;"
            f"padding:10px 14px;margin:2px 0 10px 0;background:{ec}0d'>"
            f"<b style='color:{ec}'>🧭 Multi-timeframe trend: {mtf['n_up']}/4 up · {elabel} ⓘ</b>"
            f"<div style='margin-top:5px'>{cells}</div>"
            f"<div style='margin-top:5px;color:#c9ced6;font-size:0.9rem'>{mtf['posture']}</div></div>",
            unsafe_allow_html=True)
        st.caption(
            "Accuracy (8yr, matched horizon): **swing / short** trend has a *mild* forward edge "
            "(~+1-2pp vs drift) — use them for entry timing. **Long / very-long** bands DESCRIBE "
            "the current trend but are **not predictive** — the market mean-reverts, so a long-term "
            "🔴▼ DOWN is a *bottom* signal (all-4-down → ~76% up over ~3mo), **not** a reason to "
            "short. Up-calls barely beat the bull drift; the real value is the alignment read above.")

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
    if n_sup:
        st.caption(f"🚫 {n_sup} overweight call(s) suppressed — in this regime the top-momentum "
                   f"basket is measured to UNDERPERFORM (OOS 4yr). Shown as NEUTRAL, not a buy.")

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
        b50 = bd["b50"]; b200 = bd["b200"]; dur = bd["dur_days"]
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
            f" · held {dur} day{'s' if dur != 1 else ''}</span></div>",
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
    c1.metric("Market backdrop", state,
              help="The overall market mood, which sets how big to trade the buy list today. "
                   "Uptrend = full size. Chop = small, top names only. Downtrend / sharp fall = "
                   "reduced size, top names only — the buy list still holds up in bears (tested "
                   "2018-2026), but the signal is less reliable and a long-only book still falls "
                   "with the market, so don't deploy full.")
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
    def _bucket(name: str):
        g = df[df["tilt"] == name].copy()
        icon, color, sub = _TILT_STYLE[name]
        tip = _TILT_HELP.get(name, "").replace('"', "'")
        st.markdown(f"<b style='color:{color}' title=\"{tip}\">{icon} {name} ⓘ</b> "
                    f"<span style='color:#8a8f98'>· {sub}</span>", unsafe_allow_html=True)
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
            st.markdown(
                f"**{r['sector']}** &nbsp; <span title='Strength vs Nifty over the last 2 weeks — "
                f"higher = leading the market'>rs₂w {r['rs_2w']:+.1f}%</span> · "
                f"<span title='Recent delivery-buying vs its own normal — above 1 = more real "
                f"buying than usual'>dv5d {r['dv5d']:.2f}</span> · "
                f"<span title='Rough expected move vs the average sector over ~10 days. A lean, "
                f"wide error bars — 0 when the backdrop says stand aside'>est {int(r['est_rel_bps']):+d}bps</span>"
                f"{flag}", unsafe_allow_html=True)

    st.caption("Each row: **rs₂w** = 2-week strength vs the market (higher = leading) · **dv5d** = "
               "real buying vs its own normal (>1 = heavier) · **est** = rough 10-day edge vs the "
               "average sector. Hover any label for the plain meaning.")
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
                "2wk vs mkt %", format="%.1f",
                help="How much it beat (+) or lagged (−) the Nifty over the last 2 weeks. "
                     "This is the main strength signal."),
            "rs_1w": st.column_config.NumberColumn(
                "1wk vs mkt %", format="%.1f", help="Same as 2wk, but over the last 1 week."),
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


def render(selected_date: date, min_turnover: float, all_dates: list | None = None) -> None:
    st.subheader("🔄 Sector Rotation — Smart Money Tracker")

    tab_smart, tab_tilt, tab_clock, tab_rs = st.tabs([
        "🎯 Smart Money (Daily Signal)",
        "🧭 1–2 Wk Forward Tilt",
        "📅 Rotation Clock",
        "📈 vs Nifty50",
    ])

    with tab_smart:
        _render_smart_money(selected_date, min_turnover)

    with tab_tilt:
        _render_forward_tilt(selected_date, min_turnover)

    with tab_clock:
        _render_rotation_clock(selected_date, min_turnover, all_dates=all_dates)

    with tab_rs:
        _render_relative_strength(selected_date, min_turnover, all_dates=all_dates)
