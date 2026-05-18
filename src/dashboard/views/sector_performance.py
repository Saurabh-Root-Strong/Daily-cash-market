from datetime import date
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.analytics.sector_aggregator import (
    get_sector_master_performance,
    get_subsector_master_performance,
    get_subsector_stocks_performance,
)

# ── Column layout constants ───────────────────────────────────────────────────
# [btn | name | 1W% | 2W% | 1M% | 3M% | 1WD | 2WD | 1MD | 3MD]
_SEC_COLS  = [0.28, 2.0, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
_SUB_COLS  = [0.28, 0.28, 1.9, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

_PRICE_KEYS = ["1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct"]
_DELIV_KEYS = ["1W_deliv_pct",     "2W_deliv_pct",     "1M_deliv_pct",     "3M_deliv_pct"]

_STOCK_COL_CONFIG = {
    "symbol":           st.column_config.TextColumn("Symbol"),
    "company_name":     st.column_config.TextColumn("Company"),
    "close_price":      st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
    # Price change by period
    "1W_price_chg_pct": st.column_config.NumberColumn("1W Price%",  format="%.2f%%",
                            help="Cumulative price change over last 1 week"),
    "2W_price_chg_pct": st.column_config.NumberColumn("2W Price%",  format="%.2f%%",
                            help="Cumulative price change over last 2 weeks"),
    "1M_price_chg_pct": st.column_config.NumberColumn("1M Price%",  format="%.2f%%",
                            help="Cumulative price change over last 1 month"),
    "3M_price_chg_pct": st.column_config.NumberColumn("3M Price%",  format="%.2f%%",
                            help="Cumulative price change over last 3 months"),
    # Avg delivery % by period
    "1W_deliv_pct":     st.column_config.NumberColumn("1W Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 1 week — conviction of buyers"),
    "2W_deliv_pct":     st.column_config.NumberColumn("2W Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 2 weeks"),
    "1M_deliv_pct":     st.column_config.NumberColumn("1M Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 1 month"),
    "3M_deliv_pct":     st.column_config.NumberColumn("3M Deliv%",  format="%.1f%%",
                            help="Average delivery % over last 3 months (long-term baseline)"),
    # Today's signals
    "deliv_per":        st.column_config.NumberColumn("Today Deliv%", format="%.1f%%",
                            help="Today's delivery % — compare with period averages to spot change"),
    "deliv_ratio":      st.column_config.NumberColumn("Deliv Ratio",  format="%.2f",
                            help=">1.2 = accumulating above norm, <0.8 = distributing below norm"),
}
_STOCK_SHOW = [
    "symbol", "company_name", "close_price",
    "1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct",
    "1W_deliv_pct",     "2W_deliv_pct",     "1M_deliv_pct",     "3M_deliv_pct",
    "deliv_per", "deliv_ratio",
]


# ── Formatting helpers ────────────────────────────────────────────────────────
def _color(val):
    if pd.isna(val): return "#888"
    return "#2ca02c" if val > 0 else ("#d62728" if val < 0 else "#888")

def _fmt(val, dec=2):
    if pd.isna(val): return "—"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.{dec}f}%"

def _normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)


# ── Session state toggle helpers ──────────────────────────────────────────────
def _toggle_sector(key):
    s = st.session_state.setdefault("exp_sectors", set())
    if key in s: s.discard(key)
    else: s.add(key)

def _toggle_subsector(key):
    s = st.session_state.setdefault("exp_subsectors", set())
    if key in s: s.discard(key)
    else: s.add(key)

def _is_sec_open(key):
    return key in st.session_state.get("exp_sectors", set())

def _is_sub_open(key):
    return key in st.session_state.get("exp_subsectors", set())


# ── Column header tooltips ────────────────────────────────────────────────────
_COL_TOOLTIPS = {
    "1W Price%": "Cumulative price return over the last 7 days.\n(end close − start close) ÷ start close × 100, weighted by today's turnover.",
    "2W Price%": "Cumulative price return over the last 14 days.",
    "1M Price%": "Cumulative price return over the last 30 days.",
    "3M Price%": "Cumulative price return over the last 90 days (one quarter).",
    "1W Deliv%": "Turnover-weighted avg delivery % over last 7 days.\nHigher = more shares held overnight = investor conviction buying.",
    "2W Deliv%": "Turnover-weighted avg delivery % over last 14 days.",
    "1M Deliv%": "Turnover-weighted avg delivery % over last 30 days.",
    "3M Deliv%": "Turnover-weighted avg delivery % over last 90 days — long-term conviction baseline.",
}

def _header_cell(col, label, tooltip=None, bold=True):
    if tooltip:
        style = "cursor:help;font-weight:600;border-bottom:1px dashed rgba(255,255,255,0.4);font-size:13px"
        col.markdown(
            f'<span title="{tooltip}" style="{style}">{label}</span>',
            unsafe_allow_html=True,
        )
    else:
        col.markdown(f"**{label}**" if (bold and label) else label)


# ── Write a styled metric cell ────────────────────────────────────────────────
def _price_cell(col, val):
    col.markdown(
        f"<div style='color:{_color(val)};font-weight:600;font-size:13px'>{_fmt(val)}</div>",
        unsafe_allow_html=True,
    )

def _deliv_cell(col, val):
    col.markdown(
        f"<div style='font-size:13px'>{_fmt(val,1)}</div>",
        unsafe_allow_html=True,
    )


# ── Master table renderer ─────────────────────────────────────────────────────
def _master_table(sector_df, subsector_df, selected_date, min_turnover):
    # ── Header row ─────────────────────────────────────────────────────────
    h = st.columns(_SEC_COLS)
    _header_cell(h[0], "")
    _header_cell(h[1], "Sector")
    for col, key in zip(h[2:], ["1W Price%","2W Price%","1M Price%","3M Price%",
                                  "1W Deliv%","2W Deliv%","1M Deliv%","3M Deliv%"]):
        _header_cell(col, key, tooltip=_COL_TOOLTIPS.get(key))
    st.markdown("<hr style='margin:3px 0 5px 0;border-color:rgba(255,255,255,0.2)'>",
                unsafe_allow_html=True)

    for _, sec_row in sector_df.iterrows():
        sector = sec_row["sector"]
        sec_open = _is_sec_open(sector)

        # ── Sector row ────────────────────────────────────────────────────
        r = st.columns(_SEC_COLS)
        icon = "▼" if sec_open else "▶"
        r[0].button(icon, key=f"s_{sector}",
                    on_click=_toggle_sector, args=(sector,),
                    use_container_width=True)
        r[1].markdown(f"**{sector}**")
        for c, k in zip(r[2:6], _PRICE_KEYS):
            _price_cell(c, sec_row.get(k, float("nan")))
        for c, k in zip(r[6:10], _DELIV_KEYS):
            _deliv_cell(c, sec_row.get(k, float("nan")))

        # ── Sub-sector rows (visible when sector is open) ─────────────────
        if sec_open:
            sub_df = subsector_df[subsector_df["sector"] == sector].copy()

            # Sub-sector header
            sh = st.columns(_SUB_COLS)
            for col, lbl in zip(sh, ["","","**Sub-Sector**",
                                      "1W Price%","2W Price%","1M Price%","3M Price%",
                                      "1W Deliv%","2W Deliv%","1M Deliv%","3M Deliv%"]):
                col.markdown(f"<small style='color:#aaa'>{lbl}</small>", unsafe_allow_html=True)

            for _, sub_row in sub_df.iterrows():
                industry = sub_row["industry"]
                stock_count = int(sub_row.get("stock_count", 0))
                sub_key = f"{sector}|{industry}"
                sub_open = _is_sub_open(sub_key)

                sr = st.columns(_SUB_COLS)
                sr[0].write("")   # indent spacer
                sub_icon = "▼" if sub_open else "▶"
                sr[1].button(sub_icon, key=f"ss_{sub_key}",
                             on_click=_toggle_subsector, args=(sub_key,),
                             use_container_width=True)
                count_label = f"({stock_count})" if stock_count else ""
                sr[2].markdown(
                    f"<span style='font-size:13px'>{industry} "
                    f"<span style='color:#888;font-size:11px'>{count_label}</span></span>",
                    unsafe_allow_html=True,
                )
                for c, k in zip(sr[3:7], _PRICE_KEYS):
                    _price_cell(c, sub_row.get(k, float("nan")))
                for c, k in zip(sr[7:11], _DELIV_KEYS):
                    _deliv_cell(c, sub_row.get(k, float("nan")))

                # ── Stock rows (visible when sub-sector is open) ──────────
                if sub_open:
                    with st.spinner(f"Loading {industry} stocks…"):
                        stocks = get_subsector_stocks_performance(
                            selected_date, sector, industry, min_turnover
                        )
                    if stocks.empty:
                        st.info("No stock data for this sub-sector.")
                    else:
                        show_cols = [c for c in _STOCK_SHOW if c in stocks.columns]
                        cfg = {k: v for k, v in _STOCK_COL_CONFIG.items() if k in show_cols}
                        st.dataframe(
                            stocks[show_cols],
                            column_config=cfg,
                            use_container_width=True,
                            hide_index=True,
                        )

            st.markdown(
                "<hr style='margin:4px 0;border-color:rgba(255,255,255,0.08)'>",
                unsafe_allow_html=True,
            )

        # thin separator between sectors
        st.markdown(
            "<div style='border-bottom:1px solid rgba(255,255,255,0.05);margin:2px 0'></div>",
            unsafe_allow_html=True,
        )


# ── Outlook scoring ───────────────────────────────────────────────────────────
def _compute_outlook(df):
    out = df.copy()
    out["_dm"]  = out["1W_deliv_pct"] - out["3M_deliv_pct"]
    out["_pm"]  = out.get("2W_price_chg_pct", out["1W_price_chg_pct"])
    out["_bc"]  = out["3M_deliv_pct"]
    out["_acc"] = (
        (out["1W_deliv_pct"] > out.get("2W_deliv_pct", out["1M_deliv_pct"])).astype(float) +
        (out.get("2W_deliv_pct", out["1M_deliv_pct"]) > out["1M_deliv_pct"]).astype(float)
    )
    out["Score"] = (
        _normalize(out["_dm"])  * 35 + _normalize(out["_bc"]) * 25 +
        _normalize(out["_pm"])  * 25 + _normalize(out["_acc"]) * 15
    ).round(1)

    def _sig(row):
        if row["_dm"] > 0 and row["_pm"] > 0: return "🟢 Accumulating"
        if row["_dm"] > 0:                     return "🟡 Buying Dips"
        if row["_pm"] > 0:                     return "🟠 Weak Rally"
        return "🔴 Distributing"

    out["Signal"] = out.apply(_sig, axis=1)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")]) \
              .sort_values("Score", ascending=False).reset_index(drop=True)


def _kpi_row(df):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Top Deliv (1W)",  df.nlargest(1,"1W_deliv_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'1W_deliv_pct').iloc[0]['1W_deliv_pct']:.1f}%")
    c2.metric("Top Deliv (1M)",  df.nlargest(1,"1M_deliv_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'1M_deliv_pct').iloc[0]['1M_deliv_pct']:.1f}%")
    c3.metric("Top Deliv (3M)",  df.nlargest(1,"3M_deliv_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'3M_deliv_pct').iloc[0]['3M_deliv_pct']:.1f}%")
    c4.metric("Best Price (1W)", df.nlargest(1,"1W_price_chg_pct").iloc[0]["sector"],
              f"{df.nlargest(1,'1W_price_chg_pct').iloc[0]['1W_price_chg_pct']:+.2f}%")
    if "2W_price_chg_pct" in df.columns:
        c5.metric("Best Price (2W)", df.nlargest(1,"2W_price_chg_pct").iloc[0]["sector"],
                  f"{df.nlargest(1,'2W_price_chg_pct').iloc[0]['2W_price_chg_pct']:+.2f}%")


def _outlook_chart(scored):
    top = scored.head(10)
    colors = top["Signal"].map({
        "🟢 Accumulating":"#2ca02c","🟡 Buying Dips":"#f0b429",
        "🟠 Weak Rally":"#f58518","🔴 Distributing":"#d62728",
    }).tolist()
    fig = go.Figure(go.Bar(
        x=top["Score"], y=top["sector"], orientation="h",
        marker_color=colors, text=top["Signal"], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Composite Score (0–100)", range=[0,115]),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160,r=200,t=20,b=40), height=360,
    )
    return fig


def _comparison_chart(df, metric):
    col_map = {
        "Delivery %":     ("1W_deliv_pct","2W_deliv_pct","1M_deliv_pct","3M_deliv_pct"),
        "Price Change %": ("1W_price_chg_pct","2W_price_chg_pct","1M_price_chg_pct","3M_price_chg_pct"),
    }
    cols   = [c for c in col_map[metric] if c in df.columns]
    labels = ["1W","2W","1M","3M"]
    colors = ["#4c78a8","#72b7b2","#f58518","#54a24b"]
    sdf = df.sort_values(cols[-1], ascending=True).fillna(0)
    fig = go.Figure()
    for col, name, color in zip(cols, labels, colors):
        fig.add_trace(go.Bar(
            y=sdf["sector"], x=sdf[col], name=name, orientation="h",
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.2f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        xaxis=dict(title=metric, zeroline=True, zerolinecolor="rgba(255,255,255,0.3)", ticksuffix="%"),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160,r=40,t=40,b=50),
        height=max(440, len(df)*28+120), bargap=0.18, bargroupgap=0.06,
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────
def render(selected_date: date, min_turnover: float) -> None:
    st.subheader("Sector Performance — Multi-Period Summary")
    st.caption(
        f"Cumulative price change & turnover-weighted delivery — as of "
        f"**{selected_date.strftime('%d %b %Y')}**"
    )

    sector_df    = get_sector_master_performance(selected_date, min_turnover_lacs=min_turnover)
    subsector_df = get_subsector_master_performance(selected_date, min_turnover_lacs=min_turnover)

    if sector_df.empty:
        st.warning("No data available. Run a backfill first.")
        return

    _kpi_row(sector_df)
    st.markdown("---")

    # ── 1-2 Week Outlook ─────────────────────────────────────────────────────
    st.markdown("### 📈 1–2 Week Sector Outlook")
    st.caption("Score = 35% delivery momentum + 25% base conviction + 25% price momentum + 15% acceleration")
    scored = _compute_outlook(sector_df)
    top3 = scored.head(3)
    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1,c2,c3], top3.iterrows()):
        col.metric(f"{row['Signal']} — {row['sector']}",
                   f"Score: {row['Score']:.0f}/100",
                   f"Deliv {row['1W_deliv_pct']:.1f}% vs {row['3M_deliv_pct']:.1f}% (3M avg)")
    st.plotly_chart(_outlook_chart(scored), use_container_width=True)

    st.markdown("---")

    # ── Master expandable table ───────────────────────────────────────────────
    st.markdown("### 📋 Master Performance Table")

    with st.expander("❓ How to read this table", expanded=False):
        st.markdown("""
**Price % columns (1W / 2W / 1M / 3M)**
: Cumulative return from that many days ago to today.
  Formula: *(today's close − start close) ÷ start close × 100*, weighted by today's turnover.
  🟢 green = sector gained &nbsp;|&nbsp; 🔴 red = sector fell over that window.

**Delivery % columns (1W / 2W / 1M / 3M)**
: Turnover-weighted average of daily delivery % over the period.
  Higher delivery % = more shares taken home (not squared off intraday) = investor conviction.
  Compare short-period to long-period: rising delivery over time → accumulation.

**Expanding rows**
: Click ▶ next to a **Sector** to see its sub-sectors.
  Click ▶ next to a **Sub-Sector** to see individual stock performance.
  The number in **(  )** is the stock count passing your turnover filter.

**Reading signals together**

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

    _SORT_COL_MAP = {
        "1W Price%":  "1W_price_chg_pct",
        "2W Price%":  "2W_price_chg_pct",
        "1M Price%":  "1M_price_chg_pct",
        "3M Price%":  "3M_price_chg_pct",
        "1W Deliv%":  "1W_deliv_pct",
        "2W Deliv%":  "2W_deliv_pct",
        "1M Deliv%":  "1M_deliv_pct",
        "3M Deliv%":  "3M_deliv_pct",
    }

    _FILTER_MAP = {
        "Price Up (1W)":                       lambda df: df[df["1W_price_chg_pct"] > 0],
        "Price Down (1W)":                     lambda df: df[df["1W_price_chg_pct"] < 0],
        "Price Up (1M)":                       lambda df: df[df["1M_price_chg_pct"] > 0],
        "Price Down (1M)":                     lambda df: df[df["1M_price_chg_pct"] < 0],
        "High Delivery 1W (> 45%)":            lambda df: df[df["1W_deliv_pct"] > 45],
        "Low Delivery 1W (< 30%)":             lambda df: df[df["1W_deliv_pct"] < 30],
        "Accumulating (1W Deliv > 3M Deliv)":  lambda df: df[df["1W_deliv_pct"] > df["3M_deliv_pct"]],
        "Distributing (1W Deliv < 3M Deliv)":  lambda df: df[df["1W_deliv_pct"] < df["3M_deliv_pct"]],
    }

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 3, 1.2])
    with ctrl1:
        sort_by = st.selectbox(
            "Sort by",
            options=list(_SORT_COL_MAP.keys()),
            index=0,
            key="master_sort_col",
        )
    with ctrl2:
        sort_dir = st.radio(
            "Order",
            ["Highest first", "Lowest first"],
            horizontal=True,
            key="master_sort_dir",
        )
    with ctrl3:
        filters = st.multiselect(
            "Filter (combine multiple — AND logic)",
            options=list(_FILTER_MAP.keys()),
            key="master_filter",
            placeholder="Select one or more filters…",
        )
    with ctrl4:
        st.write("")
        st.write("")
        col_ca, col_cf = st.columns(2)
        if col_ca.button("Collapse All", use_container_width=True):
            st.session_state["exp_sectors"] = set()
            st.session_state["exp_subsectors"] = set()
            st.rerun()
        if col_cf.button("Clear Filters", use_container_width=True):
            st.session_state["master_filter"] = []
            st.rerun()

    display_df = sector_df.copy()
    for f in filters:
        if f in _FILTER_MAP:
            display_df = _FILTER_MAP[f](display_df)

    sort_col = _SORT_COL_MAP[sort_by]
    ascending = sort_dir == "Lowest first"
    display_df = display_df.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)

    if display_df.empty:
        st.info("No sectors match the selected filters. Use **Clear Filters** to reset.")
    else:
        _master_table(display_df, subsector_df, selected_date, min_turnover)

    st.markdown("---")

    # ── Visual comparison ─────────────────────────────────────────────────────
    st.markdown("### Visual Comparison")
    metric = st.radio("Compare by", ["Delivery %", "Price Change %"], horizontal=True)
    st.plotly_chart(_comparison_chart(sector_df, metric), use_container_width=True)

    with st.expander("ℹ️ Signal Legend", expanded=False):
        st.markdown("""
        | Signal | Meaning | Next 1-2W Implication |
        |--------|---------|----------------------|
        | 🟢 **Accumulating** | Delivery ↑ + Price ↑ | Institutions buying on strength — likely to continue |
        | 🟡 **Buying Dips**  | Delivery ↑ + Price ↓ | Smart money absorbing weakness — watch for bounce |
        | 🟠 **Weak Rally**   | Delivery ↓ + Price ↑ | Price up but no conviction — rally may not sustain |
        | 🔴 **Distributing** | Delivery ↓ + Price ↓ | Avoid or reduce exposure |
        """)
