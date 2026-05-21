"""
Options Analysis — index options chain, PCR intelligence, stock options.

  Tab 1 — Index Options  : Options chain · Max Pain · Options Intelligence
  Tab 2 — Stock Options  : Top stocks by options OI · PCR distribution
  Tab 3 — Expiry & Chain : Stock matrix · Chain lookup
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from src.dashboard.cache.queries import cached_fno_dates_available, cached_fno_summary


def render(selected_date: date) -> None:
    st.subheader("📊 Options Analysis")

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
        key="opts_date_select",
    )

    stats = cached_fno_summary(fno_date)
    if not stats:
        st.info(f"No F&O data available for {fno_date.strftime('%d %b %Y')}.")
        return

    _render_options_kpi(stats)
    st.divider()

    tab_idx, tab_stk, tab_exp = st.tabs([
        "📈 Index Options",
        "🏭 Stock Options",
        "🔬 Expiry & Chain",
    ])

    with tab_idx:
        _render_index_options_tab(fno_date)

    with tab_stk:
        _render_stock_options_tab(fno_date)

    with tab_exp:
        _render_expiry_chain_tab(fno_date)


# ── KPI bar — options columns only ────────────────────────────────────────────

def _render_options_kpi(stats: dict) -> None:
    from src.dashboard.views.fno_activity import _fmt_cr, _fmt_pcr, _pcr_label
    cols = st.columns(4)
    kpis = [
        ("Call OI",      _fmt_cr(stats.get("call_oi", 0) / 1_000), "Contracts (K)"),
        ("Put OI",       _fmt_cr(stats.get("put_oi",  0) / 1_000), "Contracts (K)"),
        ("Overall PCR",  _fmt_pcr(stats.get("overall_pcr")),        _pcr_label(stats.get("overall_pcr"))),
        ("Total OI",     _fmt_cr(stats.get("total_oi", 0) / 1_000), "Thousands of contracts"),
    ]
    for col, (label, value, delta) in zip(cols, kpis):
        col.metric(label, value, delta if delta else None)


# ── Tab 1: Index Options ───────────────────────────────────────────────────────

def _render_index_options_tab(fno_date: date) -> None:
    from src.dashboard.cache.queries import (
        cached_fno_index_symbols,
        cached_fno_index_expiry_oi,
    )
    from src.dashboard.views.fno_activity import (
        _render_index_expiry_cards,
        _render_index_options_panel,
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
        key="opts_index_select",
    )

    df = cached_fno_index_expiry_oi(fno_date, selected_idx)
    if df.empty:
        st.info(f"No expiry data for {selected_idx} on this date.")
        return

    _render_index_expiry_cards(df, selected_idx)
    st.markdown("---")
    _render_index_options_panel(fno_date, selected_idx, df)


# ── Tab 2: Stock Options ───────────────────────────────────────────────────────

def _render_stock_options_tab(fno_date: date) -> None:
    from src.dashboard.cache.queries import cached_fno_stock_leaders
    from src.dashboard.views.fno_activity import (
        _render_stock_table,
        _stock_oi_chart,
        _stock_pcr_chart,
    )

    df = cached_fno_stock_leaders(fno_date, top_n=30)
    if df.empty:
        st.info("No stock F&O data for this date.")
        return

    df = df.copy()
    df["_opts_oi"] = df["call_oi"].fillna(0) + df["put_oi"].fillna(0)
    df = df.sort_values("_opts_oi", ascending=False).reset_index(drop=True)

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(_stock_oi_chart(df.head(20), "Options Only"), use_container_width=True)
    with col_table:
        _render_stock_table(df, "Options Only")

    st.markdown("##### PCR Distribution — Top 20 Stocks")
    top20_pcr = df.head(20).dropna(subset=["pcr"])
    if not top20_pcr.empty:
        st.plotly_chart(_stock_pcr_chart(top20_pcr), use_container_width=True)


# ── Tab 3: Expiry & Chain ──────────────────────────────────────────────────────

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

**Chain Lookup** — CE (calls) on the left, PE (puts) on the right.
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
