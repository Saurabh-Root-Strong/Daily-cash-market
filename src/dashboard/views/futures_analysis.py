"""
Futures Analysis — expiry structure, index term structure, stock OI signals.

  Tab 1 — Overview      : Expiry Calendar · Top Stocks by Futures OI
  Tab 2 — Index Futures : Term structure · Rollover · Cost of Carry
  Tab 3 — Stock Signals : OI-Matrix composite scores · Sector · Scanner
"""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from src.dashboard.cache.queries import cached_fno_dates_available, cached_fno_summary


def render(selected_date: date) -> None:
    st.subheader("📈 Futures Analysis")

    fno_dates = cached_fno_dates_available()
    if not fno_dates:
        st.warning(
            "No F&O Bhavcopy data loaded yet.  Run:\n"
            "```\npython -m src.cli backfill-fno\n```"
        )
        return

    fno_date = st.selectbox(
        "F&O Date",
        options=fno_dates,
        index=0,
        format_func=lambda d: d.strftime("%d %b %Y (%a)"),
        key="fut_date_select",
    )

    stats = cached_fno_summary(fno_date)
    if not stats:
        st.info(f"No F&O data available for {fno_date.strftime('%d %b %Y')}.")
        return

    _render_futures_kpi(stats)
    st.divider()

    tab_ov, tab_idx, tab_sig = st.tabs([
        "📊 Overview",
        "📈 Index Futures",
        "🎯 Stock Signals",
    ])

    with tab_ov:
        _render_overview_tab(fno_date)

    with tab_idx:
        _render_index_futures_tab(fno_date)

    with tab_sig:
        _render_signals_tab(fno_date)


# ── KPI bar — futures columns only ────────────────────────────────────────────

def _render_futures_kpi(stats: dict) -> None:
    from src.dashboard.views.fno_activity import _fmt_cr
    cols = st.columns(4)
    kpis = [
        ("Symbols",        f"{stats.get('total_symbols', 0):,}",
         f"Index: {stats.get('index_symbols', 0)} | Stock: {stats.get('stock_symbols', 0)}"),
        ("Active Expiries", f"{stats.get('total_expiries', 0)}", ""),
        ("Total OI",        _fmt_cr(stats.get("total_oi", 0) / 1_000), "Thousands of contracts"),
        ("Futures OI",      _fmt_cr(stats.get("fut_oi",  0) / 1_000), "Contracts (K)"),
    ]
    for col, (label, value, delta) in zip(cols, kpis):
        col.metric(label, value, delta if delta else None)


# ── Tab 1: Overview ────────────────────────────────────────────────────────────

def _render_overview_tab(fno_date: date) -> None:
    from src.dashboard.views.fno_activity import _render_expiry_calendar, _render_stock_fao
    _render_expiry_calendar(fno_date)
    st.divider()
    st.markdown("#### 🏭 Top F&O Stocks by Futures OI")
    _render_stock_fao(fno_date)


# ── Tab 2: Index Futures ───────────────────────────────────────────────────────

def _render_index_futures_tab(fno_date: date) -> None:
    from src.dashboard.cache.queries import (
        cached_fno_index_symbols,
        cached_fno_index_expiry_oi,
        cached_fno_expiry_oi_history,
    )
    from src.dashboard.views.fno_activity import (
        _render_index_expiry_cards,
        _render_index_futures_panel,
        _oi_buildup_chart,
    )

    index_symbols = cached_fno_index_symbols(fno_date)
    if not index_symbols:
        st.info("No index F&O data for this date.")
        return

    _priority = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"]
    sorted_symbols = sorted(
        index_symbols,
        key=lambda s: (_priority.index(s) if s in _priority else 999),
    )

    selected_idx = st.selectbox(
        "Select Index",
        options=sorted_symbols,
        key="fut_index_select",
    )

    df = cached_fno_index_expiry_oi(fno_date, selected_idx)
    if df.empty:
        st.info(f"No expiry data for {selected_idx} on this date.")
        return

    _render_index_expiry_cards(df, selected_idx)
    st.markdown("---")
    _render_index_futures_panel(fno_date, selected_idx, df)

    # OI buildup history — last 45 days
    hist_df = cached_fno_expiry_oi_history(selected_idx, fno_date - timedelta(days=45), fno_date)
    if not hist_df.empty:
        st.markdown(f"##### {selected_idx} — OI Buildup History (45d)")
        st.plotly_chart(_oi_buildup_chart(hist_df, selected_idx), use_container_width=True)


# ── Tab 3: Stock Signals ───────────────────────────────────────────────────────

def _render_signals_tab(fno_date: date) -> None:
    from src.dashboard.cache.queries import (
        cached_fno_composite_signals,
        cached_stock_expiry_matrix,
        cached_stock_monthly_expiries,
    )
    from src.dashboard.views.fno_stocks import (
        _ALL_SIGNALS,
        _EXP_NAMES,
        _EXP_PREFIXES,
        _render_all_stocks,
        _render_kpi_bar as _signals_kpi,
        _render_scanner,
        _render_sector_view,
    )

    with st.expander("📖 How to Read — Stock Signals", expanded=False):
        st.markdown("""
**OI-Price Matrix (35%)** — the core futures signal:

| Price | OI | Signal | Meaning |
|-------|----|--------|---------|
| ↑ | ↑ | **Long Buildup** 🟢 | Fresh longs entering — strong conviction |
| ↓ | ↑ | **Short Buildup** 🔴 | Fresh shorts entering — strong conviction |
| ↑ | ↓ | **Short Cover** 🟩 | Shorts exiting — weaker bullish |
| ↓ | ↓ | **Long Unwind** 🟠 | Longs exiting — weaker bearish |

**5-Factor Score:**

| Factor | Weight | Bullish when |
|--------|--------|-------------|
| 🔄 OI Matrix | 35% | Long Buildup (OI↑ + Price↑) |
| 🏦 Cost of Carry | 20% | Annualised basis > 7% fair value |
| 📊 PCR Contrarian | 20% | PCR > 1.5 — panic puts = floor |
| 📊 Rollover | 15% | Near OI building OR forward roll with rising price |
| 🎯 Max Pain | 10% | Only material in final 7 days to expiry |
        """)

    col_oi, col_sig, col_sec = st.columns([1, 2, 2])
    with col_oi:
        min_oi_k = st.number_input(
            "Min Futures OI (K)", min_value=0, max_value=5000, value=50, step=50,
            key="fut_sig_min_oi",
        )
    min_oi = int(min_oi_k * 1000)

    expiries = cached_stock_monthly_expiries(fno_date)
    exp_labels = (
        [f"{_EXP_NAMES[i]} ({e.strftime('%d %b')})" for i, e in enumerate(expiries[:3])]
        if expiries else _EXP_NAMES[:]
    )
    sel_exp    = st.radio("Expiry", options=exp_labels, horizontal=True, key="fut_sig_exp")
    exp_idx    = exp_labels.index(sel_exp)
    exp_prefix = _EXP_PREFIXES[exp_idx]

    df_sig = cached_fno_composite_signals(fno_date, min_fut_oi=min_oi)
    if df_sig.empty:
        st.info("No F&O stock data for this date / OI filter.")
        return

    df_mat = cached_stock_expiry_matrix(fno_date, min_fut_oi=min_oi)
    if not df_mat.empty:
        mat_cols = (
            ["symbol", "spot_price", "roll_signal"]
            + [
                f"{p}_{c}"
                for p in _EXP_PREFIXES
                for c in ["fut_oi", "chg_oi", "basis_pct", "pcr", "max_pain", "mp_dist_pct"]
            ]
        )
        mat_cols = [c for c in mat_cols if c in df_mat.columns]
        df = df_sig.merge(df_mat[mat_cols], on="symbol", how="left")
    else:
        df = df_sig.copy()

    all_sectors = sorted(df["sector"].dropna().unique().tolist())
    with col_sig:
        sig_filter = st.multiselect(
            "Signal Filter", options=_ALL_SIGNALS, default=_ALL_SIGNALS, key="fut_sig_filter",
        )
    with col_sec:
        sec_filter = st.multiselect(
            "Sector Filter", options=all_sectors, default=all_sectors, key="fut_sec_filter",
        )
    if not sig_filter:
        with col_sig:
            st.caption("ℹ️ No signal selected → showing all")
    if not sec_filter:
        with col_sec:
            st.caption("ℹ️ No sector selected → showing all")

    active_sigs = sig_filter if sig_filter else _ALL_SIGNALS
    active_secs = sec_filter if sec_filter else all_sectors
    df_filtered = df[
        df["signal_label"].isin(active_sigs) & df["sector"].isin(active_secs)
    ].copy()

    _signals_kpi(df, exp_prefix)
    st.divider()

    t_sec, t_all, t_scan = st.tabs(["🏗️ By Sector", "📋 All Stocks", "🔥 Scanner"])
    with t_sec:
        _render_sector_view(df_filtered, exp_prefix)
    with t_all:
        _render_all_stocks(df_filtered, exp_prefix)
    with t_scan:
        _render_scanner(df)
