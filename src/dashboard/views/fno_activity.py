"""
F&O Activity page — daily F&O bhavcopy data with expiry structure.

Shows:
  - KPI summary (symbols, expiries, OI, PCR)
  - Expiry calendar with Near/Mid/Far classification + Weekly/Monthly type
  - Index F&O tab: per-expiry OI breakdown for NIFTY/BANKNIFTY etc.
  - Stock F&O tab: top stocks by OI with futures/call/put split
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard.cache.queries import (
    cached_fno_dates_available,
    cached_fno_expiry_calendar,
    cached_fno_expiry_oi_history,
    cached_fno_index_expiry_oi,
    cached_index_futures_rollover,
    cached_index_options_chain,
    cached_fno_index_symbols,
    cached_fno_stock_leaders,
    cached_fno_summary,
)

_RANK_COLORS = {
    # Monthly (futures structure layer)
    "Near Month": "#2196F3",
    "Mid Month":  "#FF9800",
    "Far Month":  "#9C27B0",
    "Far+":       "#607D8B",
    # Weekly (options gamma layer)
    "Near Week":  "#4CAF50",
    "Mid Week":   "#8BC34A",
    "Far Week":   "#CDDC39",
}
_TYPE_COLORS = {
    "Weekly":  "#4CAF50",
    "Monthly": "#FF5722",
}


def render(selected_date: date) -> None:
    st.subheader("📊 F&O Daily Activity")

    fno_dates = cached_fno_dates_available()
    if not fno_dates:
        st.warning(
            "No F&O Bhavcopy data loaded yet. Run:\n"
            "```\npython -m src.cli backfill-fno\n```"
        )
        return

    # Date picker scoped to available FNO dates
    fno_date = st.selectbox(
        "F&O Date",
        options=fno_dates,
        index=0,
        format_func=lambda d: d.strftime("%d %b %Y (%a)"),
        key="fno_date_select",
    )

    stats = cached_fno_summary(fno_date)
    if not stats:
        st.info(f"No F&O data available for {fno_date.strftime('%d %b %Y')}.")
        return

    _render_kpi_bar(stats)
    st.divider()

    tab_expiry, tab_index, tab_stocks = st.tabs([
        "📅 Expiry Calendar",
        "📈 Index F&O",
        "🏭 Stock F&O",
    ])

    with tab_expiry:
        _render_expiry_calendar(fno_date)

    with tab_index:
        _render_index_fao(fno_date)

    with tab_stocks:
        _render_stock_fao(fno_date)


# ── KPI bar ───────────────────────────────────────────────────────────────────

def _render_kpi_bar(stats: dict) -> None:
    cols = st.columns(7)
    kpis = [
        ("Symbols",          f"{stats.get('total_symbols', 0):,}",
         f"Index: {stats.get('index_symbols',0)} | Stock: {stats.get('stock_symbols',0)}"),
        ("Active Expiries",  f"{stats.get('total_expiries', 0)}",     ""),
        ("Total OI",         _fmt_cr(stats.get("total_oi", 0) / 1_000), "In thousands of contracts"),
        ("Futures OI",       _fmt_cr(stats.get("fut_oi", 0)  / 1_000), "Contracts (K)"),
        ("Call OI",          _fmt_cr(stats.get("call_oi", 0) / 1_000), "Contracts (K)"),
        ("Put OI",           _fmt_cr(stats.get("put_oi",  0) / 1_000), "Contracts (K)"),
        ("Overall PCR",      _fmt_pcr(stats.get("overall_pcr")),
         _pcr_label(stats.get("overall_pcr"))),
    ]
    for col, (label, value, delta) in zip(cols, kpis):
        col.metric(label, value, delta if delta else None)


def _fmt_cr(v) -> str:
    if v is None:
        return "—"
    v = float(v)
    if v >= 1_000:
        return f"{v/1_000:.1f}M"
    if v >= 100:
        return f"{v:.0f}K"
    return f"{v:.1f}K"


def _fmt_pcr(v) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _pcr_label(pcr) -> str:
    if pcr is None:
        return ""
    if pcr > 1.3:
        return "Contrarian Bullish"
    if pcr < 0.7:
        return "Contrarian Bearish"
    return "Neutral"


# ── Expiry Calendar tab ───────────────────────────────────────────────────────

def _render_expiry_calendar(trade_date: date) -> None:
    df = cached_fno_expiry_calendar(trade_date)

    if df.empty:
        st.info("No active expiry data for this date.")
        return

    # Drop far-out expiries with no open interest (noise from long-dated instrument listings)
    df = df[df["total_oi"] > 0].reset_index(drop=True)
    if df.empty:
        st.info("No active expiry data for this date.")
        return

    st.markdown("#### Upcoming Expiry Dates")

    # Summary cards: first Near Month and first Near Week (most actionable pair)
    _all_near = ["Near Month", "Mid Month", "Far Month", "Near Week", "Mid Week", "Far Week"]
    rank_rows = df[df["expiry_rank"].isin(_all_near)].head(6)
    if not rank_rows.empty:
        cols = st.columns(min(len(rank_rows), 6))
        for col, (_, row) in zip(cols, rank_rows.iterrows()):
            color = _RANK_COLORS.get(row["expiry_rank"], "#607D8B")
            type_badge = "🗓️ Monthly" if row["expiry_type"] == "Monthly" else "📅 Weekly"
            pcr_txt = f"PCR {row['pcr']:.2f}" if row["pcr"] is not None else "PCR —"
            col.markdown(
                f"<div style='border-left:4px solid {color};padding:8px 12px;"
                f"background:#1e1e1e;border-radius:4px'>"
                f"<div style='color:{color};font-weight:bold;font-size:0.8rem'>"
                f"{row['expiry_rank']}</div>"
                f"<div style='font-size:1.2rem;font-weight:bold'>{row['expiry_label']}</div>"
                f"<div style='color:#aaa;font-size:0.8rem'>"
                f"{row['days_to_expiry']}d away &nbsp; {type_badge}</div>"
                f"<div style='font-size:0.85rem'>OI: {row['total_oi']:,.0f} &nbsp; {pcr_txt}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Full expiry calendar chart
    fig = _expiry_calendar_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### All Active Expiries")
    disp = df[[
        "expiry_label", "expiry_type", "expiry_rank", "days_to_expiry",
        "symbols", "total_oi", "call_oi", "put_oi", "fut_oi", "pcr", "value_cr"
    ]].copy()
    disp.columns = [
        "Expiry", "Type", "Month", "Days", "Symbols",
        "Total OI", "Call OI", "Put OI", "Fut OI", "PCR", "Value (Cr)"
    ]
    for c in ["Total OI", "Call OI", "Put OI", "Fut OI"]:
        disp[c] = disp[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    disp["Value (Cr)"] = disp["Value (Cr)"].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—")
    disp["PCR"] = disp["PCR"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    st.dataframe(disp, use_container_width=True, hide_index=True)


def _expiry_calendar_chart(df: pd.DataFrame) -> go.Figure:
    monthly = df[df["expiry_type"] == "Monthly"]
    weekly  = df[df["expiry_type"] == "Weekly"]

    fig = go.Figure()

    if not weekly.empty:
        fig.add_trace(go.Bar(
            x=weekly["expiry_label"],
            y=weekly["total_oi"],
            name="Weekly Expiry",
            marker_color=_TYPE_COLORS["Weekly"],
            text=weekly["total_oi"].apply(lambda v: f"{v/1000:.0f}K"),
            textposition="outside",
        ))

    if not monthly.empty:
        fig.add_trace(go.Bar(
            x=monthly["expiry_label"],
            y=monthly["total_oi"],
            name="Monthly Expiry",
            marker_color=_TYPE_COLORS["Monthly"],
            text=monthly["total_oi"].apply(lambda v: f"{v/1000:.0f}K"),
            textposition="outside",
        ))

    fig.update_layout(
        title="Open Interest by Expiry Date",
        xaxis_title="Expiry Date",
        yaxis_title="Open Interest (Contracts)",
        barmode="group",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    return fig


# ── Options Chain Charts ──────────────────────────────────────────────────────

def _options_chain_chart(
    chain: pd.DataFrame, spot: float, max_pain: float | None,
    symbol: str, expiry_label: str,
) -> go.Figure:
    """
    Mirrored waterfall: Call OI below zero (resistance) | Put OI above zero (support).
    ATM strike highlighted gold. Max Pain and Spot marked as vertical lines.
    """
    chain = chain.sort_values("strike_price").reset_index(drop=True)
    strikes = chain["strike_price"].tolist()

    ce_colors = ["#FFD700" if atm else "#EF5350" for atm in chain["is_atm"]]
    pe_colors = ["#FFD700" if atm else "#66BB6A" for atm in chain["is_atm"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="✋ Call OI  (resistance ceiling)",
        x=strikes,
        y=(-chain["ce_oi"]).tolist(),
        marker_color=ce_colors,
        marker_line_width=0,
        hovertemplate=(
            "<b>Strike %{x:,}</b><br>"
            "Call OI: <b>%{customdata[0]:,}</b><br>"
            "OI Chg: %{customdata[1]:+,}<br>"
            "Price: ₹%{customdata[2]:.2f} · Vol: %{customdata[3]:,}"
            "<extra></extra>"
        ),
        customdata=chain[["ce_oi", "ce_chg_oi", "ce_close", "ce_vol"]].values,
    ))

    fig.add_trace(go.Bar(
        name="🛡️ Put OI  (support floor)",
        x=strikes,
        y=chain["pe_oi"].tolist(),
        marker_color=pe_colors,
        marker_line_width=0,
        hovertemplate=(
            "<b>Strike %{x:,}</b><br>"
            "Put OI: <b>%{customdata[0]:,}</b><br>"
            "OI Chg: %{customdata[1]:+,}<br>"
            "Price: ₹%{customdata[2]:.2f} · Vol: %{customdata[3]:,}"
            "<extra></extra>"
        ),
        customdata=chain[["pe_oi", "pe_chg_oi", "pe_close", "pe_vol"]].values,
    ))

    # Zero dividing line (call/put axis separator)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.4)", line_width=1.5)

    # Spot vline
    if spot and spot > 0:
        fig.add_vline(
            x=spot, line_color="#FFD700", line_width=2, line_dash="dash",
            annotation_text=f"Spot {spot:,.0f}",
            annotation_position="top right",
            annotation_font=dict(size=11, color="#FFD700"),
        )

    # Max pain vline
    if max_pain and not np.isnan(max_pain):
        fig.add_vline(
            x=max_pain, line_color="#FF9800", line_width=2, line_dash="dot",
            annotation_text=f"Max Pain {int(max_pain):,}",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#FF9800"),
        )

    fig.update_layout(
        title=dict(
            text=f"{symbol} Options Chain — {expiry_label}  "
                 f"<span style='font-size:12px;color:rgba(255,255,255,0.5)'>"
                 f"↓ Call OI = resistance &nbsp;|&nbsp; ↑ Put OI = support &nbsp;|&nbsp; "
                 f"🟡 = ATM strike</span>",
            font=dict(size=14),
        ),
        barmode="overlay",
        height=480,
        template="plotly_dark",
        xaxis=dict(title="Strike Price", tickformat=",", dtick=None),
        yaxis=dict(title="Open Interest (Contracts)", tickformat=","),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=70, b=90, l=70, r=40),
        hovermode="x unified",
    )
    return fig


def _oi_change_chart(chain: pd.DataFrame, symbol: str) -> go.Figure:
    """OI buildup / unwinding by strike: CE below zero (red), PE above zero (green)."""
    chain = chain.sort_values("strike_price").reset_index(drop=True)
    strikes = chain["strike_price"].tolist()

    # CE change: positive change → fresh call writing (bear) = below 0
    ce_vals   = (-chain["ce_chg_oi"]).tolist()
    pe_vals   = chain["pe_chg_oi"].tolist()
    ce_colors = ["#B71C1C" if v < 0 else "#EF5350" for v in ce_vals]
    pe_colors = ["#1B5E20" if v < 0 else "#66BB6A" for v in pe_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Call OI Change",
        x=strikes, y=ce_vals,
        marker_color=ce_colors,
        hovertemplate="<b>Strike %{x:,}</b><br>Call OI Δ: %{customdata:+,}<extra></extra>",
        customdata=chain["ce_chg_oi"].values,
    ))
    fig.add_trace(go.Bar(
        name="Put OI Change",
        x=strikes, y=pe_vals,
        marker_color=pe_colors,
        hovertemplate="<b>Strike %{x:,}</b><br>Put OI Δ: %{customdata:+,}<extra></extra>",
        customdata=chain["pe_chg_oi"].values,
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.4)", line_width=1.5)

    fig.update_layout(
        title=f"{symbol} OI Build / Unwind Today  (↓ new call writing · ↑ new put writing)",
        barmode="overlay",
        height=280,
        template="plotly_dark",
        xaxis=dict(title="Strike Price", tickformat=","),
        yaxis=dict(title="OI Change", tickformat=","),
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(t=50, b=80, l=70, r=40),
        hovermode="x unified",
    )
    return fig


def _render_options_interpretation(
    chain: pd.DataFrame, spot: float, atm: float,
    max_pain: float | None, support: float | None, resistance: float | None,
    pcr: float | None, implied_move: float | None, implied_move_pct: float | None,
    exp_low: float | None, exp_high: float | None, symbol: str,
    expiry_type: str = "Monthly", days_to_expiry: int = 0, expiry_label: str = "",
) -> None:
    """
    Institutional-grade options interpretation.

    Key principle: OI = WRITING (selling), not buying. Buyers leave volume traces
    but rarely build persistent OI. We detect writing dominance via OI structure and
    infer directional buying from elevated vol/OI ratios.
    """

    # ── Step 1: Score-based directional verdict ──────────────────────────────────
    score = 0.0
    score_reasons: list[str] = []

    if pcr is not None:
        if pcr >= 1.5:
            score += 1.5
            score_reasons.append(f"PCR {pcr:.2f} → very heavy put writing = strong floor")
        elif pcr >= 1.2:
            score += 1.0
            score_reasons.append(f"PCR {pcr:.2f} → put-writing dominant = floor support")
        elif pcr >= 0.9:
            score += 0.3
            score_reasons.append(f"PCR {pcr:.2f} → mild put-side lean")
        elif pcr <= 0.6:
            score -= 1.5
            score_reasons.append(f"PCR {pcr:.2f} → very heavy call writing = strong ceiling")
        elif pcr <= 0.8:
            score -= 1.0
            score_reasons.append(f"PCR {pcr:.2f} → call-writing dominant = ceiling cap")
        else:
            score_reasons.append(f"PCR {pcr:.2f} → balanced")

    if max_pain is not None and spot > 0:
        dist_pct = (spot - max_pain) / max_pain * 100
        if dist_pct > 1.5:
            score -= 0.5
            score_reasons.append(f"Spot {dist_pct:.1f}% above max pain → downward gravity")
        elif dist_pct < -1.5:
            score += 0.5
            score_reasons.append(f"Spot {abs(dist_pct):.1f}% below max pain → upward pull")
        else:
            score_reasons.append(f"Spot near max pain (Δ{abs(dist_pct):.1f}%) → pin risk")

    if support is not None and resistance is not None and spot > 0:
        dist_up = max(float(resistance) - spot, 1)
        dist_dn = max(spot - float(support), 1)
        asym = (dist_up - dist_dn) / (dist_up + dist_dn)
        if asym > 0.2:
            score += 0.3
            score_reasons.append(f"Asymmetric: +{dist_up:.0f} pts room up vs {dist_dn:.0f} pts down")
        elif asym < -0.2:
            score -= 0.3
            score_reasons.append(f"Asymmetric: {dist_dn:.0f} pts room down vs +{dist_up:.0f} pts up")

    ce_build = float(chain[chain["ce_chg_oi"] > 0]["ce_chg_oi"].sum())
    pe_build = float(chain[chain["pe_chg_oi"] > 0]["pe_chg_oi"].sum())
    total_build = ce_build + pe_build
    if total_build > 0:
        pe_frac = pe_build / total_build
        if pe_frac >= 0.60:
            score += 0.4
            score_reasons.append(f"Today: {pe_frac*100:.0f}% new OI = put writing → fresh floor")
        elif pe_frac <= 0.40:
            score -= 0.4
            score_reasons.append(f"Today: {(1-pe_frac)*100:.0f}% new OI = call writing → fresh ceiling")

    if score >= 1.5:
        verdict, v_color, v_emoji = "STRONGLY BULLISH", "#4CAF50", "🟢"
    elif score >= 0.7:
        verdict, v_color, v_emoji = "BULLISH", "#66BB6A", "📈"
    elif score >= 0.2:
        verdict, v_color, v_emoji = "MILDLY BULLISH", "#CDDC39", "↗️"
    elif score <= -1.5:
        verdict, v_color, v_emoji = "STRONGLY BEARISH", "#EF5350", "🔴"
    elif score <= -0.7:
        verdict, v_color, v_emoji = "BEARISH", "#EF5350", "📉"
    elif score <= -0.2:
        verdict, v_color, v_emoji = "MILDLY BEARISH", "#FF9800", "↘️"
    else:
        verdict, v_color, v_emoji = "NEUTRAL / RANGEBOUND", "#9E9E9E", "↔️"

    exp_ctx = f"{'📅 Weekly' if expiry_type == 'Weekly' else '🗓️ Monthly'} expiry · {days_to_expiry}d remaining"

    st.markdown("#### 🔍 Options Intelligence — Market Verdict")
    st.markdown(
        f"<div style='padding:18px 24px;background:rgba(255,255,255,0.03);"
        f"border:2px solid {v_color};border-radius:10px;margin-bottom:16px'>"
        f"<div style='font-size:11px;color:rgba(255,255,255,0.4);letter-spacing:1px;"
        f"text-transform:uppercase'>{symbol}  ·  {expiry_label}  ·  {exp_ctx}</div>"
        f"<div style='font-size:30px;font-weight:900;color:{v_color};margin:6px 0 4px'>"
        f"{v_emoji} {verdict}</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.5);line-height:1.7'>"
        f"{'  ·  '.join(score_reasons)}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Step 2: Three-range expected move grid ────────────────────────────────────
    if implied_move is not None and exp_low is not None and exp_high is not None:
        st.markdown("#### 📐 Expected Range to Expiry")
        straddle_low, straddle_high = float(exp_low), float(exp_high)

        if support is not None and resistance is not None:
            inst_low, inst_high = float(support), float(resistance)
            # Weekly: gamma dominates → weight straddle more; Monthly: writer OI matters more
            w_s = 0.65 if expiry_type == "Weekly" else 0.40
            w_i = 1.0 - w_s
            cons_low  = straddle_low  * w_s + inst_low  * w_i
            cons_high = straddle_high * w_s + inst_high * w_i
            weight_note = ("65% straddle + 35% institutional"
                           if expiry_type == "Weekly" else
                           "40% straddle + 60% institutional")

            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px'>"
                # Institutional range
                f"<div style='padding:12px;background:rgba(255,255,255,0.04);border-radius:8px;"
                f"border-top:3px solid #2196F3'>"
                f"<div style='font-size:10px;color:#2196F3;font-weight:700;letter-spacing:0.5px'>"
                f"🏦 INSTITUTIONAL RANGE</div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin:2px 0 6px'>"
                f"Put Wall → Call Wall (writers' defended zone)</div>"
                f"<div style='font-size:20px;font-weight:800'>"
                f"<span style='color:#EF5350'>{inst_low:,.0f}</span>"
                f"<span style='color:rgba(255,255,255,0.25)'> — </span>"
                f"<span style='color:#4CAF50'>{inst_high:,.0f}</span></div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin-top:4px'>"
                f"{inst_high - inst_low:,.0f} pts band · option sellers are positioned here</div>"
                f"</div>"
                # Straddle range
                f"<div style='padding:12px;background:rgba(255,255,255,0.04);border-radius:8px;"
                f"border-top:3px solid #FFD700'>"
                f"<div style='font-size:10px;color:#FFD700;font-weight:700;letter-spacing:0.5px'>"
                f"📐 STRADDLE IMPLIED</div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin:2px 0 6px'>"
                f"ATM CE + ATM PE (market-priced move)</div>"
                f"<div style='font-size:20px;font-weight:800'>"
                f"<span style='color:#EF5350'>{straddle_low:,.0f}</span>"
                f"<span style='color:rgba(255,255,255,0.25)'> — </span>"
                f"<span style='color:#4CAF50'>{straddle_high:,.0f}</span></div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin-top:4px'>"
                f"±{implied_move:,.0f} pts  (±{implied_move_pct:.1f}%)</div>"
                f"</div>"
                # Consolidated range
                f"<div style='padding:12px;background:rgba(255,255,255,0.05);border-radius:8px;"
                f"border-top:3px solid {v_color}'>"
                f"<div style='font-size:10px;color:{v_color};font-weight:700;letter-spacing:0.5px'>"
                f"🎯 CONSOLIDATED TARGET</div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin:2px 0 6px'>"
                f"{weight_note}</div>"
                f"<div style='font-size:20px;font-weight:800'>"
                f"<span style='color:#EF5350'>{cons_low:,.0f}</span>"
                f"<span style='color:rgba(255,255,255,0.25)'> — </span>"
                f"<span style='color:#4CAF50'>{cons_high:,.0f}</span></div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.4);margin-top:4px'>"
                f"{'Weekly: gamma-dominant pricing' if expiry_type == 'Weekly' else 'Monthly: OI positioning matters more'}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding:12px;background:rgba(255,255,255,0.04);border-radius:8px;"
                f"border-top:3px solid #FFD700;margin-bottom:12px'>"
                f"<div style='font-size:10px;color:#FFD700;font-weight:700'>📐 STRADDLE IMPLIED RANGE</div>"
                f"<div style='font-size:20px;font-weight:800;margin:6px 0'>"
                f"<span style='color:#EF5350'>{straddle_low:,.0f}</span>"
                f"<span style='color:rgba(255,255,255,0.25)'> — </span>"
                f"<span style='color:#4CAF50'>{straddle_high:,.0f}</span></div>"
                f"<div style='font-size:11px;color:rgba(255,255,255,0.4)'>"
                f"±{implied_move:,.0f} pts  (±{implied_move_pct:.1f}%)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Step 3: OI composition — writing vs buying context ───────────────────────
    ce_total = float(chain["ce_oi"].sum())
    pe_total = float(chain["pe_oi"].sum())
    total_oi = ce_total + pe_total
    ce_vol_oi = chain["ce_vol"].sum() / max(ce_total, 1)
    pe_vol_oi = chain["pe_vol"].sum() / max(pe_total, 1)

    if total_oi > 0:
        pe_pct = pe_total / total_oi * 100
        ce_pct = 100.0 - pe_pct
        vol_note = ""
        if max(ce_vol_oi, pe_vol_oi) > 0.12:
            high_side = "Call" if ce_vol_oi > pe_vol_oi else "Put"
            vol_note = (
                f"  ·  ⚡ Elevated {high_side} vol/OI ({max(ce_vol_oi, pe_vol_oi):.1%}) "
                "→ possible directional BUYING on top of writing"
            )
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.03);padding:12px 16px;"
            f"border-radius:8px;margin-bottom:12px'>"
            f"<div style='font-size:11px;color:rgba(255,255,255,0.45);margin-bottom:8px'>"
            f"📌 <b>OI = Option WRITING (selling)</b>, not buying  ·  "
            f"<span style='color:#EF5350'>Call Writers {ce_pct:.0f}%  ({ce_total/1000:.0f}K contracts)</span>"
            f"  vs  "
            f"<span style='color:#4CAF50'>Put Writers {pe_pct:.0f}%  ({pe_total/1000:.0f}K contracts)</span>"
            f"  ·  Vol/OI → CE: {ce_vol_oi:.1%} | PE: {pe_vol_oi:.1%}{vol_note}"
            f"</div>"
            f"<div style='height:12px;background:#EF5350;border-radius:6px;overflow:hidden'>"
            f"<div style='height:100%;width:{pe_pct:.1f}%;background:#4CAF50;float:right'></div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Step 4: Actionable signals ───────────────────────────────────────────────
    st.markdown("#### 📡 Actionable Signals")
    signals: list[tuple[str, str, str, str]] = []

    if pcr is not None:
        if pcr > 1.3:
            signals.append(("✅", "#4CAF50", f"Bullish Bias — PCR {pcr:.2f}: Heavy Put WRITING",
                "Institutions are SELLING puts in large size → they expect spot to STAY ABOVE put "
                "strikes. This is floor-building activity, not defensive hedging. PCR > 1.3 = "
                "contrarian bullish (sellers have unlimited loss if market falls, so they only sell "
                "when they're very confident in the floor)."))
        elif pcr > 1.0:
            signals.append(("🔍", "#40c4ff", f"Mild Bullish — PCR {pcr:.2f}: Moderate Put Writing",
                "More puts being sold than calls → mild floor support. Watch for further PCR "
                "expansion as confirmation."))
        elif pcr < 0.7:
            signals.append(("❌", "#EF5350", f"Bearish Bias — PCR {pcr:.2f}: Heavy Call WRITING",
                "Institutions are SELLING calls in large size → they expect spot to STAY BELOW "
                "call strikes. This actively caps any rally. PCR < 0.7 = contrarian bearish "
                "(call sellers have unlimited loss on breakout, so they only sell if very "
                "confident about the ceiling holding)."))
        else:
            signals.append(("⚖️", "#888888", f"Neutral — PCR {pcr:.2f}: Balanced Writing",
                "Call and put writing roughly equal. No strong institutional directional bet from PCR."))

    if max_pain is not None and spot > 0:
        dist = spot - max_pain
        dist_pct = dist / max_pain * 100
        if days_to_expiry <= 2:
            pin_note = (f"⚠️ EXPIRY IN {days_to_expiry}d — max pain gravity is EXTREME. "
                        f"High probability of convergence toward {int(max_pain):,}.")
        elif days_to_expiry <= 5:
            pin_note = (f"Moderate pin risk ({days_to_expiry}d to expiry). "
                        f"Gravity toward {int(max_pain):,} intensifies each session.")
        else:
            pin_note = (f"Low pin risk now ({days_to_expiry}d left). "
                        f"Max pain at {int(max_pain):,} becomes dominant in final 2–3 sessions.")

        if abs(dist_pct) < 0.5:
            signals.append(("📌", "#FFD700", f"Spot Pinned at Max Pain {int(max_pain):,}",
                pin_note))
        elif dist > 0:
            signals.append(("⬇️", "#FF9800",
                f"Spot {spot:,.0f} is {dist:,.0f} pts ABOVE Max Pain ({dist_pct:+.1f}%)",
                f"Downside gravity active — writers maximize profit if spot falls to {int(max_pain):,}. "
                f"{pin_note}"))
        else:
            signals.append(("⬆️", "#4CAF50",
                f"Spot {spot:,.0f} is {abs(dist):,.0f} pts BELOW Max Pain ({dist_pct:+.1f}%)",
                f"Upside gravity active — writers maximize profit if spot rises to {int(max_pain):,}. "
                f"{pin_note}"))

    if resistance is not None:
        try:
            res_oi = int(chain.loc[chain["strike_price"] == resistance, "ce_oi"].iloc[0])
            dist_r = float(resistance) - spot
            signals.append(("✋", "#EF5350",
                f"Call Wall (Resistance) at {int(resistance):,}  [+{dist_r:,.0f} pts from spot]",
                f"{res_oi:,} contracts of call WRITING defend this ceiling. "
                "These are sellers — they'll add short calls aggressively on any rally, "
                "capping upside until this OI is closed or rolled higher."))
        except (IndexError, KeyError):
            pass

    if support is not None:
        try:
            sup_oi = int(chain.loc[chain["strike_price"] == support, "pe_oi"].iloc[0])
            dist_s = spot - float(support)
            signals.append(("🛡️", "#4CAF50",
                f"Put Wall (Support) at {int(support):,}  [{dist_s:,.0f} pts below spot]",
                f"{sup_oi:,} contracts of put WRITING defend this floor. "
                "Put sellers buy back aggressively on dips here to protect their short puts — "
                "creates a strong elastic support floor."))
        except (IndexError, KeyError):
            pass

    top_ce = chain[chain["ce_chg_oi"] > 0].nlargest(1, "ce_chg_oi")
    if not top_ce.empty:
        r = top_ce.iloc[0]
        pos_note = ("above spot → fresh CEILING written"
                    if r["strike_price"] > spot else "below spot → bearish hedge added")
        signals.append(("📈", "#FF9800",
            f"Fresh Call Writing Today: +{int(r['ce_chg_oi']):,} OI at {int(r['strike_price']):,}",
            f"New call SELLING at {int(r['strike_price']):,} ({pos_note}). "
            "These are WRITERS, not buyers — they collect premium betting market stays below "
            "this strike. Rising OI + stable/falling premium = writing confirmation."))

    top_pe = chain[chain["pe_chg_oi"] > 0].nlargest(1, "pe_chg_oi")
    if not top_pe.empty:
        r = top_pe.iloc[0]
        pos_note = ("below spot → fresh FLOOR support written"
                    if r["strike_price"] < spot else "above spot → bearish buffer")
        signals.append(("📉", "#40c4ff",
            f"Fresh Put Writing Today: +{int(r['pe_chg_oi']):,} OI at {int(r['strike_price']):,}",
            f"New put SELLING at {int(r['strike_price']):,} ({pos_note}). "
            "Institutions betting spot stays above this level — adds fresh elastic support."))

    ce_unwind = float(chain[chain["ce_chg_oi"] < 0]["ce_chg_oi"].sum())
    pe_unwind = float(chain[chain["pe_chg_oi"] < 0]["pe_chg_oi"].sum())
    if ce_total > 0 and abs(ce_unwind) > ce_total * 0.05:
        signals.append(("⚠️", "#FF9800",
            f"Call OI Unwinding: {ce_unwind:,.0f} contracts closed today",
            "Significant call writing being CLOSED → ceiling resistance weakening. "
            "Could precede an upside breakout if put writers simultaneously hold/increase."))
    if pe_total > 0 and abs(pe_unwind) > pe_total * 0.05:
        signals.append(("⚠️", "#FF7043",
            f"Put OI Unwinding: {pe_unwind:,.0f} contracts closed today",
            "Significant put writing being CLOSED → floor support weakening. "
            "Watch for breakdown if put OI isn't re-established at lower strikes."))

    if ce_vol_oi > 0.15:
        signals.append(("👀", "#9C27B0",
            f"Elevated Call Vol/OI: {ce_vol_oi:.1%} (threshold >15%)",
            "Volume unusually high relative to outstanding call OI — suggests directional CALL "
            "BUYING on top of standard writing. When buyers are active, premium expands and "
            "options can become expensive. Monitor: if OI also builds → confirms fresh longs."))
    if pe_vol_oi > 0.15:
        signals.append(("👀", "#CE93D8",
            f"Elevated Put Vol/OI: {pe_vol_oi:.1%} (threshold >15%)",
            "Volume unusually high relative to outstanding put OI — possible directional PUT "
            "BUYING or large institutional hedge. Bearish bets or tail-risk hedging active "
            "on top of standard writing."))

    for icon, color, title, detail in signals:
        st.markdown(
            f"<div style='padding:8px 12px;margin:4px 0;"
            f"background:rgba(255,255,255,0.03);border-left:3px solid {color};"
            f"border-radius:0 6px 6px 0'>"
            f"<span style='font-size:13px'>{icon} <b style='color:{color}'>{title}</b></span><br>"
            f"<span style='font-size:12px;color:rgba(255,255,255,0.65)'>{detail}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_index_options_panel(
    fno_date: date,
    selected_idx: str,
    expiry_df: pd.DataFrame,
) -> None:
    """Full institutional-grade options chain analysis for a selected index + expiry."""
    # Only expiries with actual options OI
    opts_df = expiry_df[expiry_df["call_oi"] + expiry_df["put_oi"] > 0].copy()
    if opts_df.empty:
        st.info("No options data available for this index / date.")
        return

    # Expiry selector radio
    exp_options: dict[str, date] = {}
    for _, row in opts_df.iterrows():
        wm = "Weekly" if row["expiry_type"] == "Weekly" else "Monthly"
        label = f"{row['expiry_label']}  ·  {wm}  ·  {row['days_to_expiry']}d"
        exp_options[label] = row["expiry_date"]

    sel_exp_label = st.radio(
        "Select Expiry",
        options=list(exp_options.keys()),
        horizontal=True,
        key="idx_opts_expiry_sel",
    )
    expiry_date = exp_options[sel_exp_label]
    _erow = opts_df[opts_df["expiry_date"] == expiry_date].iloc[0]
    expiry_type_sel   = str(_erow["expiry_type"])    if "expiry_type"    in opts_df.columns else "Monthly"
    days_to_exp_sel   = int(_erow["days_to_expiry"]) if "days_to_expiry" in opts_df.columns else 0

    with st.spinner(f"Loading {selected_idx} options chain…"):
        chain = cached_index_options_chain(fno_date, selected_idx, expiry_date, n_strikes=15)

    if chain.empty:
        st.info(f"No strike-level data for {selected_idx} — {sel_exp_label}.")
        return

    # Extract metadata
    spot     = float(chain["spot_price"].iloc[0] or 0)
    atm      = float(chain["atm_strike"].iloc[0] or 0)
    max_pain = chain["max_pain"].iloc[0]
    max_pain = float(max_pain) if max_pain is not None else None

    # Compute levels
    ce_total = float(chain["ce_oi"].sum())
    pe_total = float(chain["pe_oi"].sum())
    pcr      = round(pe_total / ce_total, 2) if ce_total > 0 else None

    resistance = float(chain.loc[chain["ce_oi"].idxmax(), "strike_price"]) if ce_total > 0 else None
    support    = float(chain.loc[chain["pe_oi"].idxmax(), "strike_price"]) if pe_total > 0 else None

    # Implied move from ATM straddle
    atm_row = chain[chain["is_atm"]]
    if not atm_row.empty:
        atm_ce = float(atm_row["ce_close"].iloc[0])
        atm_pe = float(atm_row["pe_close"].iloc[0])
        implied_move     = atm_ce + atm_pe
        implied_move_pct = implied_move / spot * 100 if spot > 0 else 0
        exp_low, exp_high = spot - implied_move, spot + implied_move
    else:
        atm_ce = atm_pe = implied_move = implied_move_pct = exp_low = exp_high = None

    # ── Key Levels KPI row ────────────────────────────────────────────────────
    exp_label_short = sel_exp_label.split("·")[0].strip()
    st.markdown(f"#### {selected_idx} · Options Chain — {exp_label_short}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📍 Spot (Fut)", f"{spot:,.0f}")
    c2.metric("🎯 ATM Strike", f"{int(atm):,}")
    mp_dist = round((spot - max_pain) / max_pain * 100, 1) if max_pain else None
    c3.metric("⚖️ Max Pain", f"{int(max_pain):,}" if max_pain else "—",
              f"{mp_dist:+.1f}% from spot" if mp_dist else None, delta_color="off")
    c4.metric("🛡️ Put Wall", f"{int(support):,}" if support else "—", "↑ Support", delta_color="off")
    c5.metric("✋ Call Wall", f"{int(resistance):,}" if resistance else "—", "↓ Resistance", delta_color="off")
    c6.metric("📊 PCR", f"{pcr:.2f}" if pcr else "—", _pcr_label(pcr))

    # Implied move info bar
    if implied_move is not None and implied_move > 0:
        st.info(
            f"**ATM Straddle (Implied Move):** ±{implied_move:,.0f} pts  (±{implied_move_pct:.1f}%)  "
            f"→ Expected range: **{exp_low:,.0f}** – **{exp_high:,.0f}**  "
            f"· ATM CE ₹{atm_ce:.2f}  ATM PE ₹{atm_pe:.2f}"
        )

    st.divider()

    # ── Waterfall chart ───────────────────────────────────────────────────────
    st.plotly_chart(
        _options_chain_chart(chain, spot, max_pain, selected_idx, exp_label_short),
        use_container_width=True,
        key=f"opts_chain_{selected_idx}_{expiry_date}",
    )

    # ── OI change chart ───────────────────────────────────────────────────────
    if chain[["ce_chg_oi", "pe_chg_oi"]].abs().values.sum() > 0:
        st.plotly_chart(
            _oi_change_chart(chain, selected_idx),
            use_container_width=True,
            key=f"opts_chg_{selected_idx}_{expiry_date}",
        )
    else:
        st.caption("OI change data not available for this session.")

    st.divider()

    # ── Interpretation + signals ──────────────────────────────────────────────
    _render_options_interpretation(
        chain, spot, atm, max_pain, support, resistance,
        pcr, implied_move, implied_move_pct, exp_low, exp_high, selected_idx,
        expiry_type=expiry_type_sel,
        days_to_expiry=days_to_exp_sel,
        expiry_label=exp_label_short,
    )

    # ── Full chain table ──────────────────────────────────────────────────────
    with st.expander("📋 Full Options Chain Table (ATM ± 15 strikes)", expanded=False):
        disp = chain[[
            "strike_price", "ce_oi", "ce_chg_oi", "ce_vol", "ce_close",
            "pe_oi", "pe_chg_oi", "pe_vol", "pe_close",
            "total_oi", "pcr_at_strike", "is_atm", "is_max_pain",
        ]].copy()
        disp.columns = [
            "Strike", "CE OI", "CE Chg OI", "CE Vol", "CE Price",
            "PE OI", "PE Chg OI", "PE Vol", "PE Price",
            "Total OI", "PCR", "ATM?", "Max Pain?",
        ]
        st.dataframe(
            disp,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Strike":   st.column_config.NumberColumn("Strike", format="%,.0f"),
                "CE OI":    st.column_config.NumberColumn("CE OI", format="%,d"),
                "CE Chg OI":st.column_config.NumberColumn("CE Chg OI", format="%+,d"),
                "CE Vol":   st.column_config.NumberColumn("CE Vol", format="%,d"),
                "CE Price": st.column_config.NumberColumn("CE Price (₹)", format="₹%.2f"),
                "PE OI":    st.column_config.NumberColumn("PE OI", format="%,d"),
                "PE Chg OI":st.column_config.NumberColumn("PE Chg OI", format="%+,d"),
                "PE Vol":   st.column_config.NumberColumn("PE Vol", format="%,d"),
                "PE Price": st.column_config.NumberColumn("PE Price (₹)", format="₹%.2f"),
                "Total OI": st.column_config.NumberColumn("Total OI", format="%,d"),
                "PCR":      st.column_config.NumberColumn("PCR", format="%.2f"),
                "ATM?":     st.column_config.CheckboxColumn("ATM?"),
                "Max Pain?":st.column_config.CheckboxColumn("Max Pain?"),
            },
        )


# ── Index Futures Rollover panel ─────────────────────────────────────────────

_CARRY_FAIR_ANN = 6.5  # India repo-dividend spread ≈ 6–7% pa → fair value benchmark

def _futures_price_curve(roll_df: pd.DataFrame, symbol: str) -> go.Figure:
    """Futures price curve: spot curve across monthly expiries (contango vs backwardation)."""
    fig = go.Figure()

    # Color the line by slope direction
    prices = roll_df["settle_price"].tolist()
    labels = roll_df["expiry_label"].tolist()

    # Segments colored by contango (+) vs backwardation (-)
    for i in range(len(roll_df) - 1):
        color = "#4CAF50" if prices[i + 1] >= prices[i] else "#EF5350"
        fig.add_trace(go.Scatter(
            x=[labels[i], labels[i + 1]],
            y=[prices[i], prices[i + 1]],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Dots with carry annotation
    carry_text = []
    for _, row in roll_df.iterrows():
        if pd.notna(row["carry_pct_ann"]):
            sign = "+" if row["carry_pct_ann"] >= 0 else ""
            carry_text.append(f"{sign}{row['carry_pct_ann']:.1f}% ann")
        else:
            carry_text.append("Near")

    dot_colors = []
    for _, row in roll_df.iterrows():
        if pd.notna(row["carry_pct_ann"]):
            if row["carry_pct_ann"] >= _CARRY_FAIR_ANN:
                dot_colors.append("#4CAF50")
            elif row["carry_pct_ann"] >= 3.0:
                dot_colors.append("#CDDC39")
            elif row["carry_pct_ann"] >= 0:
                dot_colors.append("#FF9800")
            else:
                dot_colors.append("#EF5350")
        else:
            dot_colors.append("#2196F3")

    fig.add_trace(go.Scatter(
        x=labels,
        y=prices,
        mode="markers+text",
        marker=dict(size=14, color=dot_colors, line=dict(width=2, color="white")),
        text=[
            f"<b>{p:,.0f}</b><br><span style='font-size:10px'>{c}</span>"
            for p, c in zip(prices, carry_text)
        ],
        textposition="top center",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Settle: <b>%{y:,.0f}</b><br>"
            "OI: %{customdata[0]:,}  (%{customdata[1]:.1f}%)<br>"
            "OI Chg: %{customdata[2]:+,}<br>"
            "Carry: %{customdata[3]}"
            "<extra></extra>"
        ),
        customdata=[
            [
                int(row["open_interest"]),
                float(row["oi_pct"]),
                int(row["chg_in_oi"]),
                carry_text[i],
            ]
            for i, (_, row) in enumerate(roll_df.iterrows())
        ],
        showlegend=False,
    ))

    # Fair value band annotation
    fig.add_hrect(
        y0=prices[0] * (1 + 3.0 / 100 * roll_df.iloc[0]["days_to_expiry"] / 365),
        y1=prices[0] * (1 + _CARRY_FAIR_ANN / 100 * (roll_df.iloc[-1]["days_to_expiry"] if len(roll_df) > 1 else 30) / 365),
        fillcolor="rgba(255,255,255,0.04)",
        layer="below",
        line_width=0,
        annotation_text="Fair value band",
        annotation_position="top right",
        annotation_font=dict(size=10, color="rgba(255,255,255,0.3)"),
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{symbol} Futures Term Structure  "
                "<span style='font-size:11px;color:rgba(255,255,255,0.4)'>"
                "🟢 Contango (premium) = normal/bullish  "
                "🔴 Backwardation (discount) = stress/bearish</span>"
            ),
            font=dict(size=14),
        ),
        height=360,
        template="plotly_dark",
        xaxis_title="Expiry",
        yaxis=dict(title="Settle Price", tickformat=","),
        margin=dict(t=70, b=50, l=70, r=40),
        hovermode="x unified",
    )
    return fig


def _render_index_futures_panel(trade_date: date, symbol: str, expiry_df: pd.DataFrame) -> None:
    """
    Futures Rollover & Cost-of-Carry analysis.

    Index futures have ONLY monthly expiries (Near / Mid / Far month).
    Weekly rows in expiry_df are OPTIDX only — filtered out here.
    """
    roll_df = cached_index_futures_rollover(trade_date, symbol)

    if roll_df.empty:
        st.info(f"No FUTIDX data available for {symbol} on this date.")
        return

    near  = roll_df.iloc[0]
    total_oi     = int(roll_df["open_interest"].sum())
    near_oi      = int(near["open_interest"])
    near_pct     = float(near["oi_pct"])
    net_chg_oi   = int(roll_df["chg_in_oi"].sum())

    # Near-to-mid carry (most important spread)
    carry_ann = None
    carry_pts = None
    if len(roll_df) >= 2:
        carry_ann = float(roll_df.iloc[1]["carry_pct_ann"])
        carry_pts = float(roll_df.iloc[1]["carry_pts"])

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown(f"#### {symbol} — Futures Rollover & Term Structure")
    c1, c2, c3, c4, c5 = st.columns(5)

    # Near month status
    near_label = near["expiry_label"]
    days_left  = int(near["days_to_expiry"])
    if near_pct > 50:
        roll_status, roll_color = "Early / Not Rolling", "#FF9800"
    elif near_pct > 20:
        roll_status, roll_color = "Rollover Active", "#FFD700"
    else:
        roll_status, roll_color = "Mostly Rolled", "#4CAF50"

    c1.metric("Near Month", near_label, f"{days_left}d left", delta_color="off")
    c2.metric("Near Month OI", f"{near_oi:,}",
              f"{near_pct:.0f}% of total · {roll_status}", delta_color="off")
    c3.metric("Total FUTIDX OI", f"{total_oi:,}",
              f"Net Δ: {net_chg_oi:+,} today",
              delta_color="normal" if net_chg_oi >= 0 else "inverse")

    if carry_ann is not None:
        if carry_ann >= _CARRY_FAIR_ANN:
            carry_label, carry_delta_color = "Bullish Premium", "normal"
        elif carry_ann >= 3.0:
            carry_label, carry_delta_color = "Near Fair Value", "off"
        elif carry_ann >= 0:
            carry_label, carry_delta_color = "Slight Discount", "inverse"
        else:
            carry_label, carry_delta_color = "Backwardation ⚠️", "inverse"

        mid_label = roll_df.iloc[1]["expiry_label"]
        c4.metric(
            f"Carry (Near→{mid_label})",
            f"{carry_ann:+.1f}% ann",
            f"{carry_pts:+.0f} pts  ·  {carry_label}",
            delta_color=carry_delta_color,
        )
    else:
        c4.metric("Carry", "—", "Only one expiry")

    # Active month (largest OI)
    active_row = roll_df.loc[roll_df["open_interest"].idxmax()]
    c5.metric(
        "Active Month",
        active_row["expiry_label"],
        f"OI: {int(active_row['open_interest']):,}  ({active_row['oi_pct']:.0f}% of total)",
        delta_color="off",
    )

    st.divider()

    # ── Term structure chart + OI bar ─────────────────────────────────────────
    col_curve, col_oi = st.columns([3, 2])
    with col_curve:
        st.plotly_chart(_futures_price_curve(roll_df, symbol), use_container_width=True)
    with col_oi:
        # OI bar — monthly futures only
        st.markdown(f"##### {symbol} Futures OI Distribution")
        fig_oi = go.Figure(go.Bar(
            x=roll_df["expiry_label"],
            y=roll_df["open_interest"],
            marker_color=[_RANK_COLORS.get(r, "#607D8B") for r in roll_df["expiry_rank"]],
            text=roll_df["open_interest"].apply(lambda v: f"{v/1000:.0f}K"),
            textposition="outside",
            customdata=roll_df[["chg_in_oi", "oi_pct"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "OI: %{y:,}<br>"
                "OI Chg: %{customdata[0]:+,}<br>"
                "% of Total: %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        ))
        fig_oi.update_layout(
            height=340, template="plotly_dark",
            yaxis_title="Open Interest",
            margin=dict(t=30, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_oi, use_container_width=True)

    # ── Rollover table ────────────────────────────────────────────────────────
    with st.expander("📋 Futures Expiry Detail Table", expanded=False):
        disp = roll_df[[
            "expiry_label", "expiry_rank", "days_to_expiry",
            "settle_price", "open_interest", "chg_in_oi",
            "oi_pct", "carry_pts", "carry_pct_ann",
        ]].copy()
        disp.columns = [
            "Expiry", "Month", "Days",
            "Settle Price", "OI", "OI Chg",
            "% of Total", "Carry (pts)", "Carry (% ann)",
        ]
        st.dataframe(disp, hide_index=True, use_container_width=True,
            column_config={
                "Settle Price":   st.column_config.NumberColumn(format="%,.2f"),
                "OI":             st.column_config.NumberColumn(format="%,d"),
                "OI Chg":         st.column_config.NumberColumn(format="%+,d"),
                "% of Total":     st.column_config.NumberColumn(format="%.1f%%"),
                "Carry (pts)":    st.column_config.NumberColumn(format="%+.2f"),
                "Carry (% ann)":  st.column_config.NumberColumn(format="%+.2f%%"),
            })

    st.divider()

    # ── Directional interpretation ────────────────────────────────────────────
    st.markdown("#### 📡 Rollover Signals — Directional Reading")
    signals: list[tuple[str, str, str, str]] = []

    # 1. Carry signal
    if carry_ann is not None:
        if carry_ann >= _CARRY_FAIR_ANN:
            signals.append(("🟢", "#4CAF50",
                f"Bullish Premium: {carry_ann:+.1f}% ann carry (above fair value ~{_CARRY_FAIR_ANN:.0f}%)",
                f"Futures are trading at a PREMIUM to fair value. Traders are paying extra to stay "
                f"long into the next expiry ({roll_df.iloc[1]['expiry_label']}). "
                "This indicates net bullish positioning — longs rolling up, shorts unwinding."))
        elif carry_ann >= 3.0:
            signals.append(("🟡", "#CDDC39",
                f"Near Fair Value: {carry_ann:+.1f}% ann carry (fair value ~{_CARRY_FAIR_ANN:.0f}%)",
                "Futures carry is within normal range. No strong directional bet embedded in "
                "the futures curve — market is fairly priced. Watch for carry expansion (bullish) "
                "or contraction (bearish)."))
        elif carry_ann >= 0:
            signals.append(("🟠", "#FF9800",
                f"Slight Discount: {carry_ann:+.1f}% ann carry (below fair value)",
                "Futures trading at a small discount to fair value. Could mean longs are "
                "not enthusiastic about rolling — mild bearish lean or wait-and-watch mode."))
        else:
            signals.append(("🔴", "#EF5350",
                f"Backwardation: {carry_ann:+.1f}% ann carry ⚠️",
                "Far month futures are CHEAPER than near month — market pricing in a DECLINE. "
                "This is a strong bearish structural signal. Typically seen during panic, "
                "high volatility, or when large players are shorting far-month futures."))

    # 2. Rollover completion signal
    if near_pct > 50 and days_left <= 7:
        signals.append(("⚠️", "#FF9800",
            f"High Near-Month OI ({near_pct:.0f}%) with only {days_left}d left — Expiry Squeeze Risk",
            f"{near_oi:,} contracts still in {near_label}. Heavy rollover expected in next "
            f"{days_left} sessions. Expect sharp intraday moves and potential gamma squeeze as "
            "these positions are forcibly closed or rolled at expiry."))
    elif near_pct < 15:
        signals.append(("✅", "#4CAF50",
            f"Rollover Nearly Complete — Near Month at {near_pct:.0f}% of total",
            "Most longs/shorts have already rolled to the next expiry. "
            "Near-month risk is low. Active month pricing now drives market direction."))
    else:
        signals.append(("🔄", "#40c4ff",
            f"Rollover In Progress — {near_pct:.0f}% still in near month ({days_left}d left)",
            "Mid-phase rollover. Expect gradual OI transfer to mid/far months over next few sessions. "
            "Near-month pricing still relevant but active month is where the real positions are."))

    # 3. Net OI trend
    if net_chg_oi > 0:
        signals.append(("📈", "#4CAF50",
            f"New Money Entering: Net OI +{net_chg_oi:+,} today",
            "Overall FUTIDX OI is BUILDING. Fresh positions are being created — not just rolling. "
            "If carried with premium (contango), this confirms directional conviction "
            "(long build = bullish; monitor in context of carry)."))
    elif net_chg_oi < 0:
        signals.append(("📉", "#EF5350",
            f"Money Leaving: Net OI {net_chg_oi:+,} today",
            "Overall FUTIDX OI is SHRINKING. Positions are being closed, not just rolled. "
            "Could signal uncertainty, profit-taking, or institutional exit. "
            "Watch: if mid-month OI is rising while near falls = rolling (healthy). "
            "If all months declining = net exit (bearish)."))

    # 4. Active month vs near month flow
    if len(roll_df) >= 2:
        mid_chg = int(roll_df.iloc[1]["chg_in_oi"])
        near_chg = int(roll_df.iloc[0]["chg_in_oi"])
        if near_chg < 0 and mid_chg > 0:
            signals.append(("🔄", "#40c4ff",
                f"Classic Roll: Near {near_chg:+,} → Active Month {mid_chg:+,}",
                "Near month OI declining while mid month builds — textbook rollover flow. "
                "No net position change, just moving forward in time. "
                "Direction of carry tells you whether longs or shorts are dominant rollers."))
        elif near_chg > 0 and mid_chg < 0:
            signals.append(("❓", "#FF9800",
                f"Reverse Flow: Near +{near_chg:,} | Mid {mid_chg:,}",
                "Unusual: near-month building while mid-month declining. "
                "Could be shorts opening near-term positions (bearish into expiry) "
                "or longs cutting far exposure. Monitor closely."))

    for icon, color, title, detail in signals:
        st.markdown(
            f"<div style='padding:8px 12px;margin:4px 0;"
            f"background:rgba(255,255,255,0.03);border-left:3px solid {color};"
            f"border-radius:0 6px 6px 0'>"
            f"<span style='font-size:13px'>{icon} <b style='color:{color}'>{title}</b></span><br>"
            f"<span style='font-size:12px;color:rgba(255,255,255,0.65)'>{detail}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Index F&O tab ─────────────────────────────────────────────────────────────

def _render_index_fao(trade_date: date) -> None:
    index_symbols = cached_fno_index_symbols(trade_date)
    if not index_symbols:
        st.info("No index F&O data for this date.")
        return

    # Priority order for default selection
    _priority = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]
    sorted_symbols = sorted(
        index_symbols,
        key=lambda s: (_priority.index(s) if s in _priority else 999),
    )

    ctrl_col, view_col = st.columns([2, 3])
    with ctrl_col:
        selected_idx = st.selectbox(
            "Select Index",
            options=sorted_symbols,
            key="fno_index_select",
        )
    with view_col:
        idx_view_mode = st.radio(
            "Show",
            ["All", "Futures Only", "Options Only"],
            horizontal=True,
            key="idx_fao_view_mode",
        )

    df = cached_fno_index_expiry_oi(trade_date, selected_idx)

    if df.empty:
        st.info(f"No expiry data for {selected_idx} on this date.")
        return

    _render_index_expiry_cards(df, selected_idx)
    st.markdown("---")

    if idx_view_mode == "Options Only":
        _render_index_options_panel(trade_date, selected_idx, df)
    elif idx_view_mode == "Futures Only":
        _render_index_futures_panel(trade_date, selected_idx, df)
    else:
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig = _index_oi_chart(df, selected_idx, idx_view_mode)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown(f"##### {selected_idx} — OI by Expiry")
            disp = df[[
                "expiry_label", "expiry_type", "expiry_rank", "days_to_expiry",
                "fut_oi", "call_oi", "put_oi", "total_oi", "pcr",
            ]].copy()
            disp.columns = ["Expiry", "Type", "Month", "Days", "Fut OI", "Call OI", "Put OI", "Total OI", "PCR"]
            for c in ["Fut OI", "Call OI", "Put OI", "Total OI"]:
                disp[c] = disp[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
            disp["PCR"] = disp["PCR"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
            st.dataframe(disp, use_container_width=True, hide_index=True)

    # OI buildup history (last 45 days) — shown for all modes
    st.markdown(f"##### {selected_idx} — OI Buildup History")
    to_d = trade_date
    from_d = trade_date - timedelta(days=45)
    hist_df = cached_fno_expiry_oi_history(selected_idx, from_d, to_d)
    if not hist_df.empty:
        fig_hist = _oi_buildup_chart(hist_df, selected_idx)
        st.plotly_chart(fig_hist, use_container_width=True)


def _render_index_expiry_cards(df: pd.DataFrame, symbol: str) -> None:
    """
    Two-section expiry structure display.

    Monthly section (🗓️): futures structure layer — Near/Mid/Far Month.
      Each index has exactly 3 monthly futures; this is where OI and carry live.

    Weekly section (📅): options gamma layer — Near/Mid/Far Week.
      Weekly expiries are options-only (no futures OI). PCR here is
      short-term gamma positioning, different from the structural monthly PCR.
    """
    st.markdown(f"#### {symbol} — Expiry Structure")

    monthly_rows = df[df["expiry_rank"].isin(["Near Month", "Mid Month", "Far Month"])].copy()
    weekly_rows  = df[df["expiry_rank"].isin(["Near Week",  "Mid Week",  "Far Week"])].copy()

    def _expiry_card(col, row) -> None:
        color    = _RANK_COLORS.get(row["expiry_rank"], "#607D8B")
        pcr_txt  = f"PCR {row['pcr']:.2f}" if row["pcr"] is not None else "PCR —"
        fut_oi   = row["fut_oi"]
        fut_line = f"Fut: {fut_oi:,.0f}" if fut_oi > 0 else "Fut: — (options only)"
        col.markdown(
            f"<div style='border-left:4px solid {color};padding:8px 12px;"
            f"background:#1a1a2e;border-radius:4px'>"
            f"<div style='color:{color};font-weight:bold;font-size:0.75rem'>"
            f"{row['expiry_rank']}</div>"
            f"<div style='font-size:1.1rem;font-weight:bold'>{row['expiry_label']}</div>"
            f"<div style='color:#aaa;font-size:0.78rem'>{row['days_to_expiry']}d to expiry</div>"
            f"<div style='font-size:0.82rem'>"
            f"{fut_line}<br>"
            f"Call: {row['call_oi']:,.0f} &nbsp; Put: {row['put_oi']:,.0f}<br>"
            f"{pcr_txt}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Monthly futures structure ──────────────────────────────────────────────
    if not monthly_rows.empty:
        st.markdown(
            "<div style='font-size:0.7rem;color:#FF9800;letter-spacing:1px;"
            "text-transform:uppercase;margin-bottom:4px'>"
            "🗓️ Monthly Expiries — Futures & Options Structure</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(monthly_rows), 3))
        for col, (_, row) in zip(cols, monthly_rows.iterrows()):
            _expiry_card(col, row)

    # ── Weekly options gamma layer ─────────────────────────────────────────────
    if not weekly_rows.empty:
        st.markdown(
            "<div style='font-size:0.7rem;color:#4CAF50;letter-spacing:1px;"
            "text-transform:uppercase;margin:8px 0 4px'>"
            "📅 Weekly Expiries — Options Gamma Layer (no futures OI)</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(weekly_rows), 3))
        for col, (_, row) in zip(cols, weekly_rows.iterrows()):
            _expiry_card(col, row)


def _index_oi_chart(df: pd.DataFrame, symbol: str, view_mode: str = "All") -> go.Figure:
    fig = go.Figure()

    if view_mode in ("All", "Futures Only"):
        fig.add_trace(go.Bar(
            name="Futures OI",
            x=df["expiry_label"],
            y=df["fut_oi"],
            marker_color="#607D8B",
        ))

    if view_mode in ("All", "Options Only"):
        fig.add_trace(go.Bar(
            name="Call OI",
            x=df["expiry_label"],
            y=df["call_oi"],
            marker_color="#4CAF50",
        ))
        fig.add_trace(go.Bar(
            name="Put OI",
            x=df["expiry_label"],
            y=df["put_oi"],
            marker_color="#F44336",
        ))
        # PCR line on secondary y-axis — only meaningful for options
        pcr_vals = df["pcr"].fillna(0)
        fig.add_trace(go.Scatter(
            name="PCR",
            x=df["expiry_label"],
            y=pcr_vals,
            mode="lines+markers",
            line=dict(color="#FFD700", width=2),
            yaxis="y2",
        ))

    _title_suffix = {
        "All": "Open Interest by Expiry",
        "Futures Only": "Futures OI by Expiry",
        "Options Only": "Options OI by Expiry (Call + Put)",
    }.get(view_mode, "Open Interest by Expiry")

    layout_kwargs: dict = dict(
        title=f"{symbol} — {_title_suffix}",
        barmode="group",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(title="Contracts"),
        margin=dict(t=60, b=40),
    )
    if view_mode in ("All", "Options Only"):
        layout_kwargs["yaxis2"] = dict(title="PCR", overlaying="y", side="right", showgrid=False)

    fig.update_layout(**layout_kwargs)
    return fig


def _oi_buildup_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    expiries = df["expiry_label"].unique()[:4]  # show at most 4 expiries
    colors   = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

    fig = go.Figure()
    for exp, color in zip(expiries, colors):
        sub = df[df["expiry_label"] == exp].sort_values("trade_date")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["trade_date"],
            y=sub["total_oi"],
            name=exp,
            mode="lines",
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=f"{symbol} — OI Buildup by Expiry (last 45 days)",
        height=320,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="OI Value (₹ Cr)",
        margin=dict(t=60, b=40),
    )
    return fig


# ── Stock F&O tab ─────────────────────────────────────────────────────────────

def _render_stock_fao(trade_date: date) -> None:
    df = cached_fno_stock_leaders(trade_date, top_n=30)

    if df.empty:
        st.info("No stock F&O data for this date.")
        return

    view_mode = st.radio(
        "Show",
        ["All", "Futures Only", "Options Only"],
        horizontal=True,
        key="stock_fao_view_mode",
    )

    # Re-sort based on what the user cares about
    if view_mode == "Futures Only":
        df = df.sort_values("fut_oi", ascending=False).reset_index(drop=True)
    elif view_mode == "Options Only":
        df = df.copy()
        df["_opts_oi"] = df["call_oi"].fillna(0) + df["put_oi"].fillna(0)
        df = df.sort_values("_opts_oi", ascending=False).reset_index(drop=True)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        top20 = df.head(20)
        fig = _stock_oi_chart(top20, view_mode)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        _render_stock_table(df, view_mode)

    # PCR distribution only makes sense for options
    if view_mode in ("All", "Options Only"):
        st.markdown("##### PCR Distribution — Top 20 Stocks")
        top20_pcr = df.head(20).dropna(subset=["pcr"])
        if not top20_pcr.empty:
            fig_pcr = _stock_pcr_chart(top20_pcr)
            st.plotly_chart(fig_pcr, use_container_width=True)


def _render_stock_table(df: pd.DataFrame, view_mode: str) -> None:
    st.markdown("##### Top Stocks by OI")
    if view_mode == "Futures Only":
        disp = df[["symbol", "fut_oi", "total_volume", "value_cr"]].copy()
        disp.columns = ["Symbol", "Fut OI", "Volume", "Value (Cr)"]
        disp["Fut OI"] = disp["Fut OI"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        disp["Volume"] = disp["Volume"].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        disp["Value (Cr)"] = disp["Value (Cr)"].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—")
    elif view_mode == "Options Only":
        disp = df[["symbol", "call_oi", "put_oi", "total_oi", "pcr", "value_cr"]].copy()
        disp.columns = ["Symbol", "Call OI", "Put OI", "Total OI", "PCR", "Value (Cr)"]
        for c in ["Call OI", "Put OI", "Total OI"]:
            disp[c] = disp[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        disp["PCR"] = disp["PCR"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        disp["Value (Cr)"] = disp["Value (Cr)"].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—")
    else:  # All
        disp = df[[
            "symbol", "fut_oi", "call_oi", "put_oi", "total_oi",
            "total_volume", "value_cr", "pcr",
        ]].copy()
        disp.columns = ["Symbol", "Fut OI", "Call OI", "Put OI", "Total OI", "Volume", "Value (Cr)", "PCR"]
        for c in ["Fut OI", "Call OI", "Put OI", "Total OI", "Volume"]:
            disp[c] = disp[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        disp["Value (Cr)"] = disp["Value (Cr)"].apply(lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—")
        disp["PCR"] = disp["PCR"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    st.dataframe(disp, use_container_width=True, hide_index=True)


def _stock_oi_chart(df: pd.DataFrame, view_mode: str = "All") -> go.Figure:
    fig = go.Figure()

    if view_mode in ("All", "Futures Only"):
        fig.add_trace(go.Bar(
            name="Futures OI",
            x=df["symbol"],
            y=df["fut_oi"],
            marker_color="#607D8B",
        ))

    if view_mode in ("All", "Options Only"):
        fig.add_trace(go.Bar(
            name="Call OI",
            x=df["symbol"],
            y=df["call_oi"],
            marker_color="#4CAF50",
        ))
        fig.add_trace(go.Bar(
            name="Put OI",
            x=df["symbol"],
            y=df["put_oi"],
            marker_color="#F44336",
        ))

    _title_suffix = {
        "All": "Open Interest Breakdown",
        "Futures Only": "Futures Open Interest",
        "Options Only": "Options Open Interest (Call + Put)",
    }.get(view_mode, "Open Interest Breakdown")

    fig.update_layout(
        title=f"Top 20 Stocks — {_title_suffix}",
        barmode="stack",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Contracts",
        xaxis_tickangle=-45,
        margin=dict(t=60, b=80),
    )
    return fig


def _stock_pcr_chart(df: pd.DataFrame) -> go.Figure:
    colors = ["#4CAF50" if v > 1.0 else "#F44336" for v in df["pcr"]]

    fig = go.Figure(go.Bar(
        x=df["symbol"],
        y=df["pcr"],
        marker_color=colors,
        text=df["pcr"].apply(lambda v: f"{v:.2f}"),
        textposition="outside",
    ))

    fig.add_hline(y=1.3, line_dash="dash", line_color="green",
                  annotation_text="1.3 Contrarian Bull", annotation_position="top right")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="1.0 Neutral")
    fig.add_hline(y=0.7, line_dash="dash", line_color="red",
                  annotation_text="0.7 Contrarian Bear", annotation_position="bottom right")

    fig.update_layout(
        title="PCR by Stock (Put OI ÷ Call OI)",
        height=340,
        template="plotly_dark",
        yaxis_title="PCR",
        xaxis_tickangle=-45,
        margin=dict(t=60, b=80),
    )
    return fig
