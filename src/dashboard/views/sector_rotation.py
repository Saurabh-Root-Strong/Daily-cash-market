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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.cache.queries import (
    cached_sector_rotation,
    cached_sector_rotation_history,
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


def _quadrant_chart(df: pd.DataFrame) -> go.Figure:
    """Smart Money Quadrant: X = 1W cumulative price return, Y = Z-Score vs 100D norm.

    Y-axis is Z-Score (σ above 100D daily-DV mean) — statistically grounded.
    Z ≥ 1.0 = delivery surge (top ~16% of days) = institutions entering.
    Z ≤ -0.5 = delivery below normal = institutions reducing exposure.
    """
    plot_df = df.dropna(subset=["price_1w", "z_score"]).copy()
    if plot_df.empty:
        return go.Figure()

    # ── Axis ranges: natural padding ─────────────────────────────────────────
    x_vals = plot_df["price_1w"]
    y_vals = plot_df["z_score"]
    x_pad = max((x_vals.max() - x_vals.min()) * 0.25, 0.5)
    y_pad = max((y_vals.max() - y_vals.min()) * 0.25, 0.5)
    x0 = x_vals.min() - x_pad
    x1 = x_vals.max() + x_pad
    y0 = y_vals.min() - y_pad
    y1 = y_vals.max() + y_pad

    fig = go.Figure()

    # ── Quadrant shading (split at X=0, Y=0) ─────────────────────────────────
    fig.add_shape(type="rect", x0=x0, x1=0, y0=0,  y1=y1,
                  fillcolor="rgba(0,200,83,0.10)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=0, y1=y1,
                  fillcolor="rgba(30,144,255,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0,
                  fillcolor="rgba(255,80,0,0.07)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=y0, y1=0,
                  fillcolor="rgba(213,0,0,0.10)",   line_width=0, layer="below")

    # ── Quadrant labels ───────────────────────────────────────────────────────
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

    # ── Crosshairs at Z=0 and price=0 ────────────────────────────────────────
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)

    # ── Reference line at Z=1.0 (delivery surge threshold) ───────────────────
    if y1 > 1.0:
        fig.add_hline(
            y=1.0,
            line_dash="dash", line_width=1.0,
            line_color="rgba(0,200,83,0.45)",
            annotation_text="Surge threshold (Z=+1σ)",
            annotation_position="top right",
            annotation_font=dict(size=10, color="rgba(0,200,83,0.7)"),
        )
    # ── Reference line at Z=-0.5 (weakness threshold) ────────────────────────
    if y0 < -0.5:
        fig.add_hline(
            y=-0.5,
            line_dash="dash", line_width=1.0,
            line_color="rgba(255,80,0,0.45)",
            annotation_text="Weakness threshold (Z=-0.5σ)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="rgba(255,80,0,0.7)"),
        )

    # ── One trace per signal so legend shows signal names ────────────────────
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

    # Bottom row: context-aware labels
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

    # Split action into prefix (BUY/AVOID/WATCH etc.) and description
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

    # ── Click to expand: per-stock breakdown (collapsed by default) ───────────
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
                    # Own-history: compare 7D wtd delivery % against stock's own 100D baseline
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
                    # Fallback: sector-relative percentile (new stocks without 100D history)
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
                # Prominent warning — gray caption was easy to miss
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


def render(selected_date: date, min_turnover: float) -> None:
    st.subheader("🔄 Sector Rotation — Smart Money Tracker")
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

    # Guard: if z_score column is missing the cache has stale pre-rewrite data.
    # Clear it and rerun so the fresh analytics result is used.
    if "z_score" not in rot.columns:
        st.cache_data.clear()
        st.rerun()

    _INVEST_SIGNALS   = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation", "👀 Early Accumulation"}
    _CAUTION_SIGNALS  = {"📊 Volume Spike"}
    _AVOID_SIGNALS    = {"⚠️ Distribution Trap", "❌ Active Selling", "📉 Weakening"}

    # Show ALL sectors with each signal type — sorted by score (no hidden sectors)
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

    # ── Smart Money Quadrant Chart ────────────────────────────────────────────
    st.markdown("### 📊 Smart Money Quadrant")
    st.caption(
        "X = 1-week cumulative price return (%).  "
        "Y = Delivery Z-Score (σ above 100D mean) — bubble size = Score (0–100).  "
        "Hover any bubble for DV Ratio, Z-Score, Breadth detail."
    )
    st.plotly_chart(_quadrant_chart(rot), use_container_width=True)

    # ── Sector reference table — identify every bubble ────────────────────────
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

    # ── Enter / Avoid columns ─────────────────────────────────────────────────
    col_enter, col_avoid = st.columns(2)

    _HIGH_CONV = 70   # visual separator threshold only — nothing is hidden

    with col_enter:
        st.markdown("### 🟢 SECTORS TO INVEST")
        st.caption("All accumulation signals — highest score first. Score = institutional conviction strength.")
        if entering.empty:
            st.info("No sectors with accumulation signal today.")
        else:
            shown_divider = False
            for _, row in entering.iterrows():
                # Visual separator between high and moderate conviction
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

    # ── Volume Spike caution section ──────────────────────────────────────────
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

    # ── Sector drill-down trend ───────────────────────────────────────────────
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

    # ── Full rotation table ───────────────────────────────────────────────────
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
