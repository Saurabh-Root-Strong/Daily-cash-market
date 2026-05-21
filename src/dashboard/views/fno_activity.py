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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard.cache.queries import (
    cached_fno_dates_available,
    cached_fno_expiry_calendar,
    cached_fno_expiry_oi_history,
    cached_fno_index_expiry_oi,
    cached_fno_index_symbols,
    cached_fno_stock_leaders,
    cached_fno_summary,
)

_RANK_COLORS = {
    "Near Month": "#2196F3",
    "Mid Month":  "#FF9800",
    "Far Month":  "#9C27B0",
    "Far+":       "#607D8B",
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

    # Near/Mid/Far summary cards
    rank_rows = df[df["expiry_rank"].isin(["Near Month", "Mid Month", "Far Month"])]
    if not rank_rows.empty:
        cols = st.columns(len(rank_rows))
        for col, (_, row) in zip(cols, rank_rows.iterrows()):
            color = _RANK_COLORS.get(row["expiry_rank"], "#607D8B")
            type_badge = (
                "🗓️ Monthly" if row["expiry_type"] == "Monthly" else "📅 Weekly"
            )
            pcr_txt = f"PCR {row['pcr']:.2f}" if row["pcr"] is not None else "PCR —"
            col.markdown(f"""
<div style="border-left:4px solid {color};padding:8px 12px;background:#1e1e1e;border-radius:4px">
<div style="color:{color};font-weight:bold;font-size:0.8rem">{row['expiry_rank']}</div>
<div style="font-size:1.2rem;font-weight:bold">{row['expiry_label']}</div>
<div style="color:#aaa;font-size:0.8rem">{row['days_to_expiry']}d away &nbsp; {type_badge}</div>
<div style="font-size:0.85rem">OI: {row['total_oi']:,.0f} &nbsp; {pcr_txt}</div>
</div>
""", unsafe_allow_html=True)

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

    selected_idx = st.selectbox(
        "Select Index",
        options=sorted_symbols,
        key="fno_index_select",
    )

    df = cached_fno_index_expiry_oi(trade_date, selected_idx)

    if df.empty:
        st.info(f"No expiry data for {selected_idx} on this date.")
        return

    _render_index_expiry_cards(df, selected_idx)
    st.markdown("---")

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig = _index_oi_chart(df, selected_idx)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown(f"##### {selected_idx} — OI by Expiry")
        disp = df[[
            "expiry_label", "expiry_type", "expiry_rank", "days_to_expiry",
            "fut_oi", "call_oi", "put_oi", "total_oi", "pcr"
        ]].copy()
        disp.columns = ["Expiry", "Type", "Month", "Days", "Fut OI", "Call OI", "Put OI", "Total OI", "PCR"]
        for c in ["Fut OI", "Call OI", "Put OI", "Total OI"]:
            disp[c] = disp[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        disp["PCR"] = disp["PCR"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # OI buildup history (last 30 days)
    st.markdown(f"##### {selected_idx} — OI Buildup History")
    to_d = trade_date
    from_d = trade_date - timedelta(days=45)
    hist_df = cached_fno_expiry_oi_history(selected_idx, from_d, to_d)
    if not hist_df.empty:
        fig_hist = _oi_buildup_chart(hist_df, selected_idx)
        st.plotly_chart(fig_hist, use_container_width=True)


def _render_index_expiry_cards(df: pd.DataFrame, symbol: str) -> None:
    st.markdown(f"#### {symbol} — Expiry Structure")
    near_rows = df[df["expiry_rank"].isin(["Near Month", "Mid Month", "Far Month"])]
    if near_rows.empty:
        return
    cols = st.columns(min(len(near_rows), 4))
    for col, (_, row) in zip(cols, near_rows.iterrows()):
        color = _RANK_COLORS.get(row["expiry_rank"], "#607D8B")
        type_badge = "🗓️ Monthly" if row["expiry_type"] == "Monthly" else "📅 Weekly"
        pcr_txt = f"PCR {row['pcr']:.2f}" if row["pcr"] is not None else "—"
        col.markdown(f"""
<div style="border-left:4px solid {color};padding:8px 12px;background:#1a1a2e;border-radius:4px">
<div style="color:{color};font-weight:bold;font-size:0.75rem">{row['expiry_rank']} &nbsp; {type_badge}</div>
<div style="font-size:1.1rem;font-weight:bold">{row['expiry_label']}</div>
<div style="color:#aaa;font-size:0.78rem">{row['days_to_expiry']}d to expiry</div>
<div style="font-size:0.82rem">
  Fut: {row['fut_oi']:,.0f} &nbsp; Call: {row['call_oi']:,.0f}<br>
  Put: {row['put_oi']:,.0f} &nbsp; {pcr_txt}
</div>
</div>
""", unsafe_allow_html=True)


def _index_oi_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Futures OI",
        x=df["expiry_label"],
        y=df["fut_oi"],
        marker_color="#607D8B",
    ))
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

    # PCR line on secondary y-axis
    pcr_vals = df["pcr"].fillna(0)
    fig.add_trace(go.Scatter(
        name="PCR",
        x=df["expiry_label"],
        y=pcr_vals,
        mode="lines+markers",
        line=dict(color="#FFD700", width=2),
        yaxis="y2",
    ))

    fig.update_layout(
        title=f"{symbol} — Open Interest by Expiry",
        barmode="group",
        height=380,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(title="Contracts"),
        yaxis2=dict(title="PCR", overlaying="y", side="right", showgrid=False),
        margin=dict(t=60, b=40),
    )
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
        yaxis_title="Total OI (Contracts)",
        margin=dict(t=60, b=40),
    )
    return fig


# ── Stock F&O tab ─────────────────────────────────────────────────────────────

def _render_stock_fao(trade_date: date) -> None:
    df = cached_fno_stock_leaders(trade_date, top_n=30)

    if df.empty:
        st.info("No stock F&O data for this date.")
        return

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        top20 = df.head(20)
        fig = _stock_oi_chart(top20)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("##### Top Stocks by OI")
        disp = df[[
            "symbol", "fut_oi", "call_oi", "put_oi", "total_oi",
            "total_volume", "value_cr", "pcr"
        ]].copy()
        disp.columns = [
            "Symbol", "Fut OI", "Call OI", "Put OI",
            "Total OI", "Volume", "Value (Cr)", "PCR"
        ]
        for c in ["Fut OI", "Call OI", "Put OI", "Total OI", "Volume"]:
            disp[c] = disp[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        disp["Value (Cr)"] = disp["Value (Cr)"].apply(
            lambda v: f"₹{v:,.0f}" if pd.notna(v) else "—"
        )
        disp["PCR"] = disp["PCR"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # PCR distribution chart
    st.markdown("##### PCR Distribution — Top 20 Stocks")
    top20_pcr = df.head(20).dropna(subset=["pcr"])
    if not top20_pcr.empty:
        fig_pcr = _stock_pcr_chart(top20_pcr)
        st.plotly_chart(fig_pcr, use_container_width=True)


def _stock_oi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Futures OI",
        x=df["symbol"],
        y=df["fut_oi"],
        marker_color="#607D8B",
    ))
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

    fig.update_layout(
        title="Top 20 Stocks — Open Interest Breakdown",
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
