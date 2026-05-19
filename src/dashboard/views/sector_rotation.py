"""
Sector Rotation — Smart Money / Institutional Activity Tracker.

Answers: WHERE are institutions putting money (short + long term)?
         WHERE are they quietly exiting before retail notices?

Signal logic: Delivery % weighted by turnover (₹ value traded) tells us
institutional conviction. Rising delivery + any price direction = smart money
accumulating. Falling delivery + rising price = dangerous distribution trap.
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
    "🔥 Secret Accumulation":  {"color": "#00c853", "rank": 0, "invest": True},
    "✅ Confirmed Accumulation": {"color": "#69f0ae", "rank": 1, "invest": True},
    "👀 Early Accumulation":     {"color": "#b9f6ca", "rank": 2, "invest": True},
    "⚖️ Neutral":               {"color": "#888888", "rank": 3, "invest": False},
    "📉 Weakening":             {"color": "#ffab40", "rank": 4, "invest": False},
    "⚠️ Distribution Trap":     {"color": "#ff6d00", "rank": 5, "invest": False},
    "❌ Active Selling":        {"color": "#d50000", "rank": 6, "invest": False},
}


def _quadrant_chart(df: pd.DataFrame) -> go.Figure:
    """Smart Money Quadrant: X = price momentum (1M), Y = delivery momentum.

    One Plotly trace per signal group so the legend is clean and readable.
    No text labels on bubbles — hover shows full detail, reference table below.
    """
    plot_df = df.dropna(subset=["price_1w", "deliv_momentum"]).copy()
    if plot_df.empty:
        return go.Figure()

    # ── Axis ranges: natural padding, NOT forced-symmetric ───────────────────
    x_vals = plot_df["price_1w"]
    y_vals = plot_df["deliv_momentum"]
    x_pad = max((x_vals.max() - x_vals.min()) * 0.25, 0.5)
    y_pad = max((y_vals.max() - y_vals.min()) * 0.25, 8.0)
    x0 = x_vals.min() - x_pad
    x1 = x_vals.max() + x_pad
    y0 = y_vals.min() - y_pad
    y1 = y_vals.max() + y_pad

    fig = go.Figure()

    # ── Quadrant shading ─────────────────────────────────────────────────────
    fig.add_shape(type="rect", x0=x0, x1=0, y0=0,  y1=y1,
                  fillcolor="rgba(0,200,83,0.10)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=0, y1=y1,
                  fillcolor="rgba(30,144,255,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0,
                  fillcolor="rgba(255,80,0,0.07)",  line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0,  x1=x1, y0=y0, y1=0,
                  fillcolor="rgba(213,0,0,0.10)",   line_width=0, layer="below")

    # ── Quadrant labels with bgcolor box for readability ─────────────────────
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

    # ── Crosshairs ───────────────────────────────────────────────────────────
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1.5)

    # ── One trace per signal so legend shows signal names ────────────────────
    signal_order = [
        "🔥 Secret Accumulation",
        "✅ Confirmed Accumulation",
        "👀 Early Accumulation",
        "⚖️ Neutral",
        "📉 Weakening",
        "⚠️ Distribution Trap",
        "❌ Active Selling",
    ]
    for signal in signal_order:
        grp = plot_df[plot_df["signal"] == signal]
        if grp.empty:
            continue
        meta   = _SIGNAL_META.get(signal, {})
        color  = meta.get("color", "#888888")
        sizes  = (grp["accum_score"] / 100 * 22 + 12).clip(12, 34)

        fig.add_trace(go.Scatter(
            x=grp["price_1w"],
            y=grp["deliv_momentum"],
            mode="markers",
            name=signal,
            marker=dict(
                color=color,
                size=sizes,
                opacity=0.90,
                line=dict(width=1.5, color="rgba(255,255,255,0.5)"),
            ),
            customdata=grp[["sector", "action", "accum_score",
                             "deliv_1w", "deliv_3m", "horizon"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"<span style='color:{color}'>{signal}</span><br>"
                "─────────────────────<br>"
                "Score: <b>%{customdata[2]:.0f}</b>/100<br>"
                "Price 1W: <b>%{x:+.2f}%</b><br>"
                "Delivery Momentum: <b>%{y:+.1f}%</b><br>"
                "Wtd Deliv 1W: %{customdata[3]:.1f}%  ·  3M baseline: %{customdata[4]:.1f}%<br>"
                "Horizon: %{customdata[5]}<br>"
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
            title="← Institutions Exiting  |  Delivery Momentum (%)  |  Institutions Entering →",
            showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
            range=[y0, y1], ticksuffix="%", tickfont=dict(size=11),
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
                    for v in hist["wtd_price_chg"].fillna(0)]

    fig = go.Figure()
    # Left axis: Wtd Delivery % (line + fill)
    fig.add_trace(go.Scatter(
        x=hist["trade_date"], y=hist["wtd_deliv_per"],
        name="Wtd Delivery %", mode="lines",
        line=dict(color=color, width=2.5),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        hovertemplate="<b>%{x|%d %b}</b><br>Wtd Delivery %: %{y:.1f}%<extra></extra>",
        yaxis="y1",
    ))
    # Left axis: Delivery Value ₹ Cr (dotted line, same axis — same unit domain)
    fig.add_trace(go.Scatter(
        x=hist["trade_date"], y=hist["deliv_value_cr"],
        name="Deliv Value (₹ Cr)", mode="lines",
        line=dict(color="#f0b429", width=1.5, dash="dot"),
        hovertemplate="<b>%{x|%d %b}</b><br>Delivery ₹: %{y:.1f} Cr<extra></extra>",
        yaxis="y2",
    ))
    # Right axis: Daily Price Change % (bars, green/red)
    fig.add_trace(go.Bar(
        x=hist["trade_date"], y=hist["wtd_price_chg"],
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


def _sector_card(row: pd.Series, selected_date: date, min_turnover: float) -> None:
    meta  = _SIGNAL_META.get(row["signal"], {})
    color = meta.get("color", "#888")
    score = row["accum_score"]

    bar_html = (
        f"<div style='background:rgba(255,255,255,0.1);border-radius:4px;height:6px;margin:4px 0 8px 0'>"
        f"<div style='width:{score}%;background:{color};height:6px;border-radius:4px'></div></div>"
    )
    # All values stored as Python float or None — never the string "—"
    dm   = row.get("deliv_momentum")
    p1w  = row.get("price_1w")
    d1w  = row.get("deliv_1w")
    d3m_ = row.get("deliv_3m")
    dv1w = row.get("deliv_val_1w_cr")

    def _fmt(v, fmt):
        return fmt.format(v) if (v is not None and not (isinstance(v, float) and pd.isna(v))) else "—"

    dm_str    = _fmt(dm,   "{:+.1f}%")
    p1w_str   = _fmt(p1w,  "{:+.2f}%")
    d1w_str   = _fmt(d1w,  "{:.1f}%")
    d3m_str   = _fmt(d3m_, "{:.1f}%")
    dv1w_str  = f"₹{dv1w:,.0f} Cr" if (dv1w is not None and not (isinstance(dv1w, float) and pd.isna(dv1w))) else "—"
    dm_color  = POSITIVE_COLOR if (dm is not None and dm > 0) else NEGATIVE_COLOR
    p1w_color = POSITIVE_COLOR if (p1w is not None and p1w > 0) else NEGATIVE_COLOR

    st.markdown(
        f"<div style='border-left:3px solid {color};padding:8px 12px;margin:4px 0;"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<b style='font-size:14px'>{row['sector']}</b>"
        f"<span style='font-size:11px;color:{color};font-weight:600'>{score:.0f}/100</span></div>"
        f"{bar_html}"
        f"<div style='font-size:11px;color:rgba(255,255,255,0.6)'>{row['signal']}</div>"
        f"<div style='display:flex;gap:16px;margin-top:6px;font-size:12px'>"
        f"<span>Deliv chg: <b style='color:{dm_color}'>{dm_str}</b></span>"
        f"<span>1W price: <b style='color:{p1w_color}'>{p1w_str}</b></span>"
        f"<span>1W deliv: <b>{d1w_str}</b></span>"
        f"<span>3M avg: <b>{d3m_str}</b></span>"
        f"</div>"
        f"<div style='margin-top:4px;font-size:11px;color:rgba(255,255,255,0.5)'>"
        f"Coverage: <b style='color:rgba(255,255,255,0.75)'>{row.get('coverage','—')}</b>"
        f" &nbsp;|&nbsp; Horizon: {row['horizon']}"
        f" &nbsp;|&nbsp; Delivery Value 1W: {dv1w_str}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Click to expand: per-stock breakdown ─────────────────────────────────
    with st.expander(f"📋  View stocks in {row['sector']}"):
        stocks = cached_sector_stocks_rotation(row["sector"], selected_date, min_turnover)
        if stocks.empty:
            st.caption("No stock data for this period.")
        else:
            invest_signal = meta.get("invest", False)

            # Sector-relative thresholds: top/bottom tercile of this sector's delivery %
            valid_deliv = stocks["wtd_deliv_per"].dropna()
            hi_thresh = float(valid_deliv.quantile(0.67)) if len(valid_deliv) >= 3 else float(valid_deliv.max())
            lo_thresh = float(valid_deliv.quantile(0.33)) if len(valid_deliv) >= 3 else float(valid_deliv.min())

            def _stock_signal(r: pd.Series) -> str:
                p = float(r["price_chg_pct"]) if pd.notna(r["price_chg_pct"]) else 0.0
                d = float(r["wtd_deliv_per"])  if pd.notna(r["wtd_deliv_per"])  else 0.0
                if invest_signal:
                    # Secret/Confirmed: highest delivery stocks with falling price = strongest conviction
                    if d >= hi_thresh and p < 0:
                        return "🔥 Strong"
                    elif d >= hi_thresh:
                        return "✅ Buying"
                    elif d >= lo_thresh:
                        return "👀 Watch"
                    else:
                        return "⚪ Weak"
                else:
                    # Distribution/Selling: lowest delivery stocks still rising = most dangerous
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

            # Sort by conviction strength first, then by delivery % within each group.
            # Sorting by ₹ delivery value alone was wrong — it put high-cap Weak stocks
            # at the top and buried Strong/Buying stocks that had smaller market caps.
            if invest_signal:
                _rank = {"🔥 Strong": 0, "✅ Buying": 1, "👀 Watch": 2, "⚪ Weak": 3}
            else:
                _rank = {"❌ Exit Now": 0, "⚠️ Reducing": 1, "📉 Fading": 2, "⚪ Neutral": 3}

            stocks["_rank"] = stocks["conviction"].map(_rank).fillna(9)
            stocks = stocks.sort_values(
                ["_rank", "wtd_deliv_per", "deliv_value_cr"],
                ascending=[True, False, False],
            ).drop(columns="_rank")

            # Flag if a single stock dominates sector turnover (>30%) — misleading context
            total_turnover = stocks["turnover_cr"].sum()
            dominant = stocks[stocks["turnover_cr"] / total_turnover > 0.30]
            if not dominant.empty:
                dom = dominant.iloc[0]
                dom_pct = dom["turnover_cr"] / total_turnover * 100
                dom_conv = stocks.loc[stocks["symbol"] == dom["symbol"], "conviction"].values[0]
                st.caption(
                    f"⚠️ **{dom['symbol']}** dominates {dom_pct:.0f}% of sector turnover "
                    f"with **{dom_conv}** conviction ({dom['wtd_deliv_per']:.1f}% delivery) "
                    f"— sector signal is heavily influenced by this single stock."
                )

            st.dataframe(
                stocks[["symbol", "company_name", "industry", "ltp", "conviction",
                         "wtd_deliv_per", "deliv_value_cr", "turnover_cr", "price_chg_pct"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "symbol":        st.column_config.TextColumn("Symbol", width="small"),
                    "company_name":  st.column_config.TextColumn("Company"),
                    "industry":      st.column_config.TextColumn("Sub-Sector"),
                    "ltp":           st.column_config.NumberColumn(
                        "LTP (₹)", format="₹%.2f",
                        help="Last traded price (most recent close in the period)"),
                    "conviction":    st.column_config.TextColumn("Conviction",
                        help="Strong/Buying = top-third delivery % (institutional accumulation)\n"
                             "Watch = mid-third delivery %\n"
                             "Weak = bottom-third delivery % (relative to sector peers)"),
                    "wtd_deliv_per": st.column_config.NumberColumn(
                        "Wtd Deliv %", format="%.1f%%",
                        help="Turnover-weighted delivery % — last 7 days"),
                    "deliv_value_cr":st.column_config.NumberColumn(
                        "Deliv Value (₹ Cr)", format="₹%.1f",
                        help="₹ value of shares delivered — absolute institutional conviction"),
                    "turnover_cr":   st.column_config.NumberColumn("Turnover (₹ Cr)", format="₹%.1f"),
                    "price_chg_pct": st.column_config.NumberColumn("Price Chg %", format="%+.2f%%"),
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

**The four quadrants:**
| | Delivery RISING | Delivery FALLING |
|---|---|---|
| **Price UP** | ✅ Confirmed Accumulation — enter/hold | ⚠️ Distribution Trap — EXIT immediately |
| **Price DOWN** | 🔥 Secret Accumulation — **best entry** | ❌ Active Selling — strict avoid |

**Secret Accumulation is the most powerful signal.** Institutions buy quietly while retail panics on falling prices. When delivery rises despite falling prices, smart money is building a position — the price reversal usually follows.

**Distribution Trap is the most dangerous.** Institutions need retail buyers to exit. They let price rise (creating FOMO), then dump their holdings into the rally. If delivery falls while price rises — institutions are exiting and retail is their exit liquidity.

**Score (0–100):** Composite of delivery momentum (40%), 100-day trend slope (30%), and recent acceleration (20%) + signal bonus/penalty.
        """)

    with st.spinner("Computing 100-day rotation signals…"):
        rot = cached_sector_rotation(selected_date, min_turnover)

    if rot.empty:
        st.warning("Insufficient data. Need at least 10 trading days of history.")
        return

    # ── Score thresholds ──────────────────────────────────────────────────────
    # Scores are normalized 0–100 across all sectors. 78+ = top conviction
    # signals; 22 and below = high-danger avoid zone.
    _INVEST_THRESHOLD = 78
    _AVOID_THRESHOLD  = 22

    _INVEST_SIGNALS = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation", "👀 Early Accumulation"}
    _AVOID_SIGNALS  = {"⚠️ Distribution Trap", "❌ Active Selling", "📉 Weakening"}

    entering = rot[rot["signal"].isin(_INVEST_SIGNALS) & (rot["accum_score"] >= _INVEST_THRESHOLD)]
    exiting  = rot[rot["signal"].isin(_AVOID_SIGNALS)  & (rot["accum_score"] <= _AVOID_THRESHOLD)]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🔥 Secret Accum",   len(rot[rot["signal"] == "🔥 Secret Accumulation"]),
              help="Price falling + delivery rising — best entry zone")
    k2.metric("✅ Confirmed Buy",    len(rot[rot["signal"] == "✅ Confirmed Accumulation"]),
              help="Price rising + delivery rising — momentum confirmed")
    k3.metric("⚠️ Distribution",    len(rot[rot["signal"] == "⚠️ Distribution Trap"]),
              help="Price rising + delivery falling — institutions exiting into retail rally")
    k4.metric("❌ Active Selling",  len(rot[rot["signal"] == "❌ Active Selling"]),
              help="Price falling + delivery falling — avoid")

    st.markdown("---")

    # ── Smart Money Quadrant Chart ────────────────────────────────────────────
    st.markdown("### 📊 Smart Money Quadrant")
    st.caption(
        "Bubble size = Accumulation Score (0–100). "
        "Hover any bubble for full sector detail. "
        "Use the legend below the chart to identify signals."
    )
    st.plotly_chart(_quadrant_chart(rot), use_container_width=True)

    # ── Sector reference table — identify every bubble ────────────────────────
    with st.expander("🗂️ Sector Reference — full list ranked by score", expanded=False):
        ref_cols = ["sector", "signal", "accum_score", "coverage",
                    "deliv_momentum", "price_1w", "deliv_1w", "deliv_3m", "action"]
        ref_df = rot[[c for c in ref_cols if c in rot.columns]].copy()
        ref_df.columns = ["Sector", "Signal", "Score", "Coverage",
                          "Deliv Momentum %", "Price 1W %",
                          "Wtd Deliv 1W %", "Wtd Deliv 3M %", "Action"][: len(ref_df.columns)]
        st.dataframe(
            ref_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Score":           st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                "Coverage":        st.column_config.TextColumn("Coverage", help="Swing=3–15 days · Positional=4–8 weeks · Mid Term=3–4 months"),
                "Deliv Momentum %":st.column_config.NumberColumn(format="%.1f%%"),
                "Price 1W %":      st.column_config.NumberColumn(format="%+.2f%%"),
                "Wtd Deliv 1W %":  st.column_config.NumberColumn(format="%.1f%%"),
                "Wtd Deliv 3M %":  st.column_config.NumberColumn(format="%.1f%%"),
            },
        )
    st.markdown("---")

    # ── Enter / Avoid columns ─────────────────────────────────────────────────
    col_enter, col_avoid = st.columns(2)

    with col_enter:
        st.markdown("### 🟢 SECTORS TO INVEST")
        n_hidden_enter = len(rot[rot["signal"].isin(_INVEST_SIGNALS)]) - len(entering)
        st.caption(
            f"Score ≥ {_INVEST_THRESHOLD}/100 only — highest conviction first. "
            + (f"{n_hidden_enter} weaker signal sector(s) hidden." if n_hidden_enter > 0 else "")
        )
        if entering.empty:
            st.info(f"No sectors with accumulation signal and score ≥ {_INVEST_THRESHOLD} today.")
        else:
            for _, row in entering.iterrows():
                _sector_card(row, selected_date, min_turnover)

    with col_avoid:
        st.markdown("### 🔴 SECTORS TO AVOID / EXIT")
        n_hidden_avoid = len(rot[rot["signal"].isin(_AVOID_SIGNALS)]) - len(exiting)
        st.caption(
            f"Score ≤ {_AVOID_THRESHOLD}/100 only — highest danger first. "
            + (f"{n_hidden_avoid} milder signal sector(s) hidden." if n_hidden_avoid > 0 else "")
        )
        avoid_sorted = exiting.sort_values("accum_score", ascending=True)
        if avoid_sorted.empty:
            st.info(f"No sectors with distribution signal and score ≤ {_AVOID_THRESHOLD} today.")
        else:
            for _, row in avoid_sorted.iterrows():
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

        st.markdown(
            f"<div style='padding:10px 16px;border-left:4px solid "
            f"{meta.get('color','#888')};background:rgba(255,255,255,0.04);"
            f"border-radius:0 8px 8px 0;margin:8px 0'>"
            f"<b style='font-size:16px'>{row['signal']}  —  {chosen}</b><br>"
            f"<span style='color:rgba(255,255,255,0.7)'>{row['action']}</span><br>"
            f"<span style='font-size:12px;color:rgba(255,255,255,0.5)'>"
            f"Horizon: {row['horizon']} &nbsp;|&nbsp; Score: {row['accum_score']:.0f}/100"
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
            "All sectors ranked by accumulation score. "
            "Delivery Momentum = (1W delivery − 3M baseline) ÷ 3M baseline × 100"
        )
        display_cols = ["sector", "signal", "accum_score", "coverage", "horizon",
                        "deliv_momentum", "deliv_1w", "deliv_1m", "deliv_3m",
                        "price_1w", "price_1m", "price_3m", "deliv_val_1w_cr"]
        display = rot[[c for c in display_cols if c in rot.columns]].copy()

        st.dataframe(
            display,
            column_config={
                "sector":         st.column_config.TextColumn("Sector"),
                "signal":         st.column_config.TextColumn("Signal"),
                "accum_score":    st.column_config.ProgressColumn(
                    "Score", max_value=100, format="%.0f"),
                "coverage":       st.column_config.TextColumn(
                    "Coverage",
                    help="Swing (3–15 days): Elder impulse + Weinstein Stage 1→2 breakout\n"
                         "Positional (4–8 weeks): Weinstein Stage 2 + Pring weekly KST\n"
                         "Mid Term (3–4 months): Murphy sector leadership + steep 100-day slope\n"
                         "BTST (1–2 days): not shown here — use Signals page for single-day delivery spikes"),
                "horizon":        st.column_config.TextColumn("Horizon"),
                "deliv_momentum": st.column_config.NumberColumn(
                    "Deliv Momentum %", format="%+.1f%%",
                    help="(1W delivery − 3M avg) ÷ 3M avg × 100"),
                "deliv_1w":       st.column_config.NumberColumn("1W Deliv%",  format="%.1f%%"),
                "deliv_1m":       st.column_config.NumberColumn("1M Deliv%",  format="%.1f%%"),
                "deliv_3m":       st.column_config.NumberColumn("3M Deliv%",  format="%.1f%%"),
                "price_1w":       st.column_config.NumberColumn("1W Price%",  format="%+.2f%%"),
                "price_1m":       st.column_config.NumberColumn("1M Price%",  format="%+.2f%%"),
                "price_3m":       st.column_config.NumberColumn("3M Price%",  format="%+.2f%%"),
                "deliv_val_1w_cr":st.column_config.NumberColumn(
                    "1W Deliv Val (Cr)", format="₹%.1f",
                    help="₹ value of shares delivered in last 1 week — absolute conviction"),
            },
            use_container_width=True,
            hide_index=True,
        )
