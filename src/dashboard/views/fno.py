"""
F&O Analysis — single sidebar entry with 4 focused tabs.

  Tab 1 — Overview       : Market KPIs · Expiry Calendar · Stock OI Leaders
  Tab 2 — Index F&O      : Futures term structure · Options chain · OI buildup
  Tab 3 — Stock Signals  : OI-Matrix composite scores · Sector view · Scanner
  Tab 4 — Expiry & Chain : Stock near/next/far matrix · Chain lookup

The three former standalone pages (fno_activity, fno_stocks, fno_expiry) are
retained as internal implementation modules; their render() functions are no
longer registered in app.py.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from src.dashboard.cache.queries import cached_fno_dates_available, cached_fno_summary


def render(selected_date: date) -> None:
    st.subheader("📊 F&O Analysis")

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
        key="fno_unified_date",
    )

    stats = cached_fno_summary(fno_date)
    if not stats:
        st.info(f"No F&O data available for {fno_date.strftime('%d %b %Y')}.")
        return

    # Market-wide KPI bar shown above all tabs
    from src.dashboard.views.fno_activity import _render_kpi_bar as _market_kpi
    _market_kpi(stats)
    st.divider()

    tab_ov, tab_idx, tab_sig, tab_exp = st.tabs([
        "📊 Overview",
        "📈 Index F&O",
        "🎯 Stock Signals",
        "🔬 Expiry & Chain",
    ])

    with tab_ov:
        _render_overview_tab(fno_date)

    with tab_idx:
        from src.dashboard.views.fno_activity import _render_index_fao
        _render_index_fao(fno_date)

    with tab_sig:
        _render_signals_tab(fno_date)

    with tab_exp:
        _render_expiry_chain_tab(fno_date)


# ── Tab 1: Overview ────────────────────────────────────────────────────────────

def _render_overview_tab(fno_date: date) -> None:
    from src.dashboard.views.fno_activity import (
        _render_expiry_calendar,
        _render_stock_fao,
    )
    _render_expiry_calendar(fno_date)
    st.divider()
    st.markdown("#### 🏭 Top F&O Stocks by OI")
    _render_stock_fao(fno_date)


# ── Tab 3: Stock Signals ───────────────────────────────────────────────────────

def _render_signals_tab(fno_date: date) -> None:
    """Date-shared controls + data load — mirrors fno_stocks.render() minus date picker."""
    import pandas as pd  # noqa: F401  (used in closures below)

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

**Roll Signal** (near vs next month OI comparison):
- 🔄 **Rolling Fwd** — Near OI↓ + Next OI↑: rolling forward, normal pre-expiry
- 📈 **Building** — Near OI rising >50K: fresh positions added
- 📉 **Unwinding** — Both OI falling: positions being closed
- ⚖️ **Neutral** — Within normal range
        """)

    col_oi, col_sig, col_sec = st.columns([1, 2, 2])
    with col_oi:
        min_oi_k = st.number_input(
            "Min Futures OI (K)", min_value=0, max_value=5000, value=50, step=50,
            key="fno_sig_min_oi",
        )
    min_oi = int(min_oi_k * 1000)

    expiries = cached_stock_monthly_expiries(fno_date)
    exp_labels = (
        [f"{_EXP_NAMES[i]} ({e.strftime('%d %b')})" for i, e in enumerate(expiries[:3])]
        if expiries else _EXP_NAMES[:]
    )
    sel_exp    = st.radio("Expiry", options=exp_labels, horizontal=True, key="fno_sig_exp")
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
            "Signal Filter", options=_ALL_SIGNALS, default=_ALL_SIGNALS, key="fno_sig_filter",
        )
    with col_sec:
        sec_filter = st.multiselect(
            "Sector Filter", options=all_sectors, default=all_sectors, key="fno_sec_filter",
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


# ── Tab 4: Expiry & Chain ──────────────────────────────────────────────────────

def _render_expiry_chain_tab(fno_date: date) -> None:
    with st.expander("ℹ️ How to Read — Expiry & Chain", expanded=False):
        st.markdown("""
**Stock Matrix** — All F&O stocks across Near / Next / Far month in parallel tabs.

| Column | Meaning |
|--------|---------|
| Fut OI | Open futures contracts for that expiry |
| Chg OI | Change vs previous session (+ve = new positions, −ve = closing) |
| Basis% | (Futures − Spot) ÷ Spot × 100. Positive = premium (bullish) |
| PCR | Put/Call OI ratio. >1 = put-heavy, <0.7 = call-heavy |
| Max Pain | Strike where option writers lose least — price gravitates here near expiry |

**Roll Signal** (near vs next OI):
- 🔄 Rolling Fwd — Near OI↓ + Next OI↑ (rollover in progress)
- 📈 Building — Near OI rising >50K (fresh positions)
- 📉 Unwinding — Both OI falling (exit signal)
- ⚖️ Neutral — Normal range

**Chain Lookup** — Horizontal butterfly: CE (calls) on the left, PE (puts) on the right.
High CE OI at a strike = call writing resistance ceiling.
High PE OI at a strike = put writing support floor.
Max Pain (orange dashed) = expiry settlement magnet.
        """)

    sub_matrix, sub_chain = st.tabs(["📋 Stock Matrix", "🔍 Chain Lookup"])
    with sub_matrix:
        from src.dashboard.views.fno_expiry import _render_stock_matrix
        _render_stock_matrix(fno_date)
    with sub_chain:
        from src.dashboard.views.fno_expiry import _render_standalone_chain
        _render_standalone_chain(fno_date)
