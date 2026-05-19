"""
Sector Performance page — multi-period summary with expandable tree table.

Imports:
  - cache.queries  → all SQL calls (cached 5 min)
  - components.kpi → performance_kpi_strip
  - components.filters → render_filter_builder / apply_filters / render_filter_summary
  - components.charts  → outlook_bar_chart, period_comparison_chart
  - state          → expand/collapse session state helpers
  - constants      → shared column keys, widths, tooltips
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard import state as ss
from src.dashboard.cache.queries import (
    cached_sector_master_performance,
    cached_subsector_master_performance,
    cached_subsector_stocks_performance,
    cached_all_stocks,
    cached_search_stocks,
)
from src.dashboard.components.charts import outlook_bar_chart, period_comparison_chart
from src.dashboard.components.filters import (
    apply_filters,
    render_filter_builder,
    render_filter_summary,
)
from src.dashboard.components.kpi import performance_kpi_strip
from src.dashboard.constants import (
    COL_TOOLTIPS,
    DELIV_KEYS,
    METRIC_LABELS,
    PRICE_KEYS,
    SECTOR_COL_WIDTHS,
    SORT_COL_MAP,
    SUBSECTOR_COL_WIDTHS,
)

# ── Stock-level column config (period performance drilldown) ──────────────────
_STOCK_COL_CONFIG = {
    "symbol":           st.column_config.TextColumn("Symbol"),
    "company_name":     st.column_config.TextColumn("Company"),
    "category":         st.column_config.TextColumn("Category",
                            help="Specific product/business category within the sub-sector"),
    "close_price":      st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
    "1W_price_chg_pct": st.column_config.NumberColumn("1W Price%",  format="%.2f%%",
                            help="Cumulative price change over last 1 week"),
    "2W_price_chg_pct": st.column_config.NumberColumn("2W Price%",  format="%.2f%%",
                            help="Cumulative price change over last 2 weeks"),
    "1M_price_chg_pct": st.column_config.NumberColumn("1M Price%",  format="%.2f%%",
                            help="Cumulative price change over last 1 month"),
    "3M_price_chg_pct": st.column_config.NumberColumn("3M Price%",  format="%.2f%%",
                            help="Cumulative price change over last 3 months"),
    "1W_deliv_pct":     st.column_config.NumberColumn("1W Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 1 week"),
    "2W_deliv_pct":     st.column_config.NumberColumn("2W Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 2 weeks"),
    "1M_deliv_pct":     st.column_config.NumberColumn("1M Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 1 month"),
    "3M_deliv_pct":     st.column_config.NumberColumn("3M Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 3 months"),
    "deliv_per":        st.column_config.NumberColumn("Today Deliv%", format="%.1f%%",
                            help="Today's delivery % — compare with period averages to spot change"),
    "deliv_ratio":      st.column_config.NumberColumn("Deliv Ratio",  format="%.2f",
                            help=">1.2 = accumulating above norm, <0.8 = distributing below norm"),
}
_STOCK_SHOW = [
    "symbol", "company_name", "category", "close_price",
    "1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct",
    "1W_deliv_pct",     "2W_deliv_pct",     "1M_deliv_pct",     "3M_deliv_pct",
    "deliv_per", "deliv_ratio",
]

# ── Search results column config (adds sector + industry to stock config) ─────
_SEARCH_COL_CONFIG = {
    "symbol":           st.column_config.TextColumn("Symbol"),
    "company_name":     st.column_config.TextColumn("Company"),
    "sector":           st.column_config.TextColumn("Sector"),
    "industry":         st.column_config.TextColumn("Sub-Sector"),
    "category":         st.column_config.TextColumn("Category"),
    "close_price":      st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
    "price_change_pct": st.column_config.NumberColumn("Today Chg%", format="%.2f%%",
                            help="Price change vs previous close"),
    "1W_price_chg_pct": st.column_config.NumberColumn("1W Price%",  format="%.2f%%"),
    "2W_price_chg_pct": st.column_config.NumberColumn("2W Price%",  format="%.2f%%"),
    "1M_price_chg_pct": st.column_config.NumberColumn("1M Price%",  format="%.2f%%"),
    "3M_price_chg_pct": st.column_config.NumberColumn("3M Price%",  format="%.2f%%"),
    "deliv_per":        st.column_config.NumberColumn("Today Deliv%", format="%.1f%%"),
    "1W_deliv_pct":     st.column_config.NumberColumn("1W Deliv%",  format="%.1f%%"),
    "1M_deliv_pct":     st.column_config.NumberColumn("1M Deliv%",  format="%.1f%%"),
    "3M_deliv_pct":     st.column_config.NumberColumn("3M Deliv%",  format="%.1f%%"),
    "deliv_ratio":      st.column_config.NumberColumn("Deliv Ratio", format="%.2f",
                            help=">1.2 = accumulating, <0.8 = distributing"),
    "vol_ratio":        st.column_config.NumberColumn("Vol Ratio",   format="%.2f",
                            help="Today's volume vs 20-day avg"),
}
_SEARCH_SHOW = [
    "symbol", "company_name", "sector", "industry", "category",
    "close_price", "price_change_pct",
    "1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct",
    "deliv_per", "1W_deliv_pct", "1M_deliv_pct", "3M_deliv_pct",
    "deliv_ratio", "vol_ratio",
]


# ── Local formatting helpers ──────────────────────────────────────────────────
def _color(val) -> str:
    if pd.isna(val):
        return "#888"
    return "#2ca02c" if val > 0 else ("#d62728" if val < 0 else "#888")


def _fmt(val, dec: int = 2) -> str:
    if pd.isna(val):
        return "—"
    return f"{'+'if val>0 else ''}{val:.{dec}f}%"


def _normalize(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


# ── Column header with hover tooltip ─────────────────────────────────────────
def _header_cell(col, label: str, tooltip: str | None = None, bold: bool = True) -> None:
    if tooltip:
        style = (
            "cursor:help;font-weight:600;"
            "border-bottom:1px dashed rgba(255,255,255,0.4);font-size:13px"
        )
        col.markdown(
            f'<span title="{tooltip}" style="{style}">{label}</span>',
            unsafe_allow_html=True,
        )
    else:
        col.markdown(f"**{label}**" if (bold and label) else label)


def _price_cell(col, val) -> None:
    col.markdown(
        f"<div style='color:{_color(val)};font-weight:600;font-size:13px'>{_fmt(val)}</div>",
        unsafe_allow_html=True,
    )


def _deliv_cell(col, val) -> None:
    col.markdown(
        f"<div style='font-size:13px'>{_fmt(val, 1)}</div>",
        unsafe_allow_html=True,
    )


# ── Master expandable tree table ──────────────────────────────────────────────
def _master_table(
    sector_df: pd.DataFrame,
    subsector_df: pd.DataFrame,
    selected_date: date,
    min_turnover: float,
) -> None:
    # Header row
    h = st.columns(SECTOR_COL_WIDTHS)
    _header_cell(h[0], "")
    _header_cell(h[1], "Sector")
    for col, key in zip(h[2:], ["1W Price%", "2W Price%", "1M Price%", "3M Price%",
                                  "1W Deliv%",  "2W Deliv%",  "1M Deliv%",  "3M Deliv%"]):
        _header_cell(col, key, tooltip=COL_TOOLTIPS.get(key))

    st.markdown(
        "<hr style='margin:3px 0 5px 0;border-color:rgba(255,255,255,0.2)'>",
        unsafe_allow_html=True,
    )

    for _, sec_row in sector_df.iterrows():
        sector   = sec_row["sector"]
        sec_open = ss.is_sector_open(sector)

        # ── Sector row ────────────────────────────────────────────────────────
        r = st.columns(SECTOR_COL_WIDTHS)
        r[0].button(
            "▼" if sec_open else "▶",
            key=f"s_{sector}",
            on_click=ss.toggle_sector, args=(sector,),
            use_container_width=True,
        )
        r[1].markdown(f"**{sector}**")
        for c, k in zip(r[2:6], PRICE_KEYS):
            _price_cell(c, sec_row.get(k, float("nan")))
        for c, k in zip(r[6:10], DELIV_KEYS):
            _deliv_cell(c, sec_row.get(k, float("nan")))

        # ── Sub-sector rows ───────────────────────────────────────────────────
        if sec_open:
            sub_df = subsector_df[subsector_df["sector"] == sector].copy()

            sh = st.columns(SUBSECTOR_COL_WIDTHS)
            for col, lbl in zip(sh, ["", "", "**Sub-Sector**",
                                      "1W Price%", "2W Price%", "1M Price%", "3M Price%",
                                      "1W Deliv%",  "2W Deliv%",  "1M Deliv%",  "3M Deliv%"]):
                col.markdown(
                    f"<small style='color:#aaa'>{lbl}</small>",
                    unsafe_allow_html=True,
                )

            for _, sub_row in sub_df.iterrows():
                industry    = sub_row["industry"]
                stock_count = int(sub_row.get("stock_count", 0))
                sub_key     = f"{sector}|{industry}"
                sub_open    = ss.is_subsector_open(sub_key)

                sr = st.columns(SUBSECTOR_COL_WIDTHS)
                sr[0].write("")
                sr[1].button(
                    "▼" if sub_open else "▶",
                    key=f"ss_{sub_key}",
                    on_click=ss.toggle_subsector, args=(sub_key,),
                    use_container_width=True,
                )
                count_label = f"({stock_count})" if stock_count else ""
                sr[2].markdown(
                    f"<span style='font-size:13px'>{industry} "
                    f"<span style='color:#888;font-size:11px'>{count_label}</span></span>",
                    unsafe_allow_html=True,
                )
                for c, k in zip(sr[3:7], PRICE_KEYS):
                    _price_cell(c, sub_row.get(k, float("nan")))
                for c, k in zip(sr[7:11], DELIV_KEYS):
                    _deliv_cell(c, sub_row.get(k, float("nan")))

                # ── Stock rows ────────────────────────────────────────────────
                if sub_open:
                    with st.spinner(f"Loading {industry} stocks…"):
                        stocks = cached_subsector_stocks_performance(
                            selected_date, sector, industry, min_turnover
                        )
                    if stocks.empty:
                        st.info("No stock data for this sub-sector.")
                    else:
                        show_cols = [c for c in _STOCK_SHOW if c in stocks.columns]
                        cfg = {k: v for k, v in _STOCK_COL_CONFIG.items() if k in show_cols}
                        st.dataframe(stocks[show_cols], column_config=cfg,
                                     use_container_width=True, hide_index=True)

            st.markdown(
                "<hr style='margin:4px 0;border-color:rgba(255,255,255,0.08)'>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='border-bottom:1px solid rgba(255,255,255,0.05);margin:2px 0'></div>",
            unsafe_allow_html=True,
        )


# ── Outlook scoring ───────────────────────────────────────────────────────────
def _compute_outlook(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_dm"]  = out["1W_deliv_pct"] - out["3M_deliv_pct"]
    out["_pm"]  = out.get("2W_price_chg_pct", out["1W_price_chg_pct"])
    out["_bc"]  = out["3M_deliv_pct"]
    out["_acc"] = (
        (out["1W_deliv_pct"] > out.get("2W_deliv_pct", out["1M_deliv_pct"])).astype(float) +
        (out.get("2W_deliv_pct", out["1M_deliv_pct"]) > out["1M_deliv_pct"]).astype(float)
    )
    out["Score"] = (
        _normalize(out["_dm"]) * 35 + _normalize(out["_bc"]) * 25 +
        _normalize(out["_pm"]) * 25 + _normalize(out["_acc"]) * 15
    ).round(1)

    def _sig(row) -> str:
        if row["_dm"] > 0 and row["_pm"] > 0: return "🟢 Accumulating"
        if row["_dm"] > 0:                     return "🟡 Buying Dips"
        if row["_pm"] > 0:                     return "🟠 Weak Rally"
        return "🔴 Distributing"

    out["Signal"] = out.apply(_sig, axis=1)
    drop_cols = [c for c in out.columns if c.startswith("_")]
    return out.drop(columns=drop_cols).sort_values("Score", ascending=False).reset_index(drop=True)


# ── Main render ───────────────────────────────────────────────────────────────
def render(selected_date: date, min_turnover: float) -> None:
    st.subheader("Sector Performance — Multi-Period Summary")
    st.caption(
        f"Cumulative price change & turnover-weighted delivery — as of "
        f"**{selected_date.strftime('%d %b %Y')}**"
    )

    sector_df    = cached_sector_master_performance(selected_date, min_turnover)
    subsector_df = cached_subsector_master_performance(selected_date, min_turnover)

    if sector_df.empty:
        st.warning("No data available. Run a backfill first.")
        return

    performance_kpi_strip(sector_df)
    st.markdown("---")

    # ── 1-2 Week Outlook ──────────────────────────────────────────────────────
    st.markdown("### 📈 1–2 Week Sector Outlook")
    st.caption(
        "Score = 35% delivery momentum + 25% base conviction "
        "+ 25% price momentum + 15% acceleration"
    )
    scored = _compute_outlook(sector_df)
    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1, c2, c3], scored.head(3).iterrows()):
        col.metric(
            f"{row['Signal']} — {row['sector']}",
            f"Score: {row['Score']:.0f}/100",
            f"Deliv {row['1W_deliv_pct']:.1f}% vs {row['3M_deliv_pct']:.1f}% (3M avg)",
        )
    st.plotly_chart(outlook_bar_chart(scored), use_container_width=True)
    st.markdown("---")

    # ── Master Performance Table ──────────────────────────────────────────────
    st.markdown("### 📋 Master Performance Table")

    with st.expander("❓ How to read this table", expanded=False):
        st.markdown("""
**Price % columns (1W / 2W / 1M / 3M)**
: Cumulative return from that many days ago to today.
  Formula: *(today's close − start close) ÷ start close × 100*, weighted by today's turnover.
  🟢 green = sector gained &nbsp;|&nbsp; 🔴 red = sector fell over that window.

**Delivery % columns (1W / 2W / 1M / 3M)**
: Turnover-weighted average of daily delivery % over the period.
  Higher delivery % = more shares taken home = investor conviction.
  Compare short-period to long-period: rising delivery over time → accumulation.

**Expanding rows**
: Click ▶ next to a **Sector** to see its sub-sectors.
  Click ▶ next to a **Sub-Sector** to see individual stock performance.
  The number in **(  )** is the stock count passing your turnover filter.

| Price % | Delivery % | What it likely means |
|---------|------------|----------------------|
| ↑ Up    | ↑ Rising   | Institutions buying on strength — conviction rally |
| ↓ Down  | ↑ Rising   | Smart money absorbing weakness — watch for bounce |
| ↑ Up    | ↓ Falling  | Retail-driven rally, low conviction — may not sustain |
| ↓ Down  | ↓ Falling  | Broad selling — avoid or reduce exposure |
        """)

    st.caption(
        "**▶ click** to expand a sector and see sub-sectors &nbsp;|&nbsp; "
        "**▶ click** a sub-sector to see individual stocks &nbsp;|&nbsp; "
        "Numbers in **(  )** = stock count in that sub-sector"
    )

    # ── Stock search — native selectbox typeahead ─────────────────────────────
    all_stocks_df = cached_all_stocks()
    stock_options = (
        [f"{r['symbol']} — {r['company_name']}" for _, r in all_stocks_df.iterrows()]
        if not all_stocks_df.empty else []
    )

    chosen = st.selectbox(
        "stock_search",
        options=stock_options,
        index=None,
        placeholder="🔍  Search by stock name or symbol…",
        key="stock_search_select",
        label_visibility="collapsed",
    )

    if chosen:
        selected_symbol = chosen.split(" — ")[0].strip()
        with st.spinner(f"Loading {selected_symbol}…"):
            results = cached_search_stocks(selected_date, selected_symbol, min_turnover)
            results = results[results["symbol"] == selected_symbol]

        if results.empty:
            st.info(
                f"No trading data for **{selected_symbol}** on "
                f"{selected_date.strftime('%d %b %Y')}. Try a different date."
            )
        else:
            show_cols = [c for c in _SEARCH_SHOW if c in results.columns]
            cfg = {k: v for k, v in _SEARCH_COL_CONFIG.items() if k in show_cols}
            st.dataframe(results[show_cols], column_config=cfg,
                         use_container_width=True, hide_index=True)
    else:
        # ── Sort + Collapse controls ──────────────────────────────────────────
        sc1, sc2, sc3 = st.columns([2.5, 2.5, 1.2])
        with sc1:
            sort_by = st.selectbox("Sort by", METRIC_LABELS, index=0, key=ss.MASTER_SORT_COL)
        with sc2:
            sort_dir = st.radio(
                "Order", ["Highest first", "Lowest first"],
                horizontal=True, key=ss.MASTER_SORT_DIR,
            )
        with sc3:
            st.write("")
            st.write("")
            if st.button("Collapse All", use_container_width=True, key="collapse_all_btn"):
                ss.collapse_all()
                st.rerun()

        # ── Custom filter builder ─────────────────────────────────────────────
        active_filters = render_filter_builder()

        # ── Apply sort + filters ──────────────────────────────────────────────
        display_df = apply_filters(sector_df, active_filters)
        render_filter_summary(active_filters, len(display_df))

        sort_col  = SORT_COL_MAP[sort_by]
        ascending = sort_dir == "Lowest first"
        display_df = display_df.sort_values(
            sort_col, ascending=ascending, na_position="last"
        ).reset_index(drop=True)

        if display_df.empty:
            st.info("No sectors match the current filters. Use **✕ Clear All Filters** to reset.")
        else:
            _master_table(display_df, subsector_df, selected_date, min_turnover)

    st.markdown("---")

    # ── Visual comparison ─────────────────────────────────────────────────────
    st.markdown("### Visual Comparison")
    metric = st.radio("Compare by", ["Delivery %", "Price Change %"], horizontal=True)
    st.plotly_chart(period_comparison_chart(sector_df, metric), use_container_width=True)

    with st.expander("ℹ️ Signal Legend", expanded=False):
        st.markdown("""
        | Signal | Meaning | Next 1-2W Implication |
        |--------|---------|----------------------|
        | 🟢 **Accumulating** | Delivery ↑ + Price ↑ | Institutions buying on strength — likely to continue |
        | 🟡 **Buying Dips**  | Delivery ↑ + Price ↓ | Smart money absorbing weakness — watch for bounce |
        | 🟠 **Weak Rally**   | Delivery ↓ + Price ↑ | Price up but no conviction — rally may not sustain |
        | 🔴 **Distributing** | Delivery ↓ + Price ↓ | Avoid or reduce exposure |
        """)
