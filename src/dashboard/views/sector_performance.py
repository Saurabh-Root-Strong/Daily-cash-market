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
from src.dashboard.components.charts import outlook_bar_chart, period_comparison_chart, signal_bar_chart
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
    "1W_deliv_cr":      st.column_config.NumberColumn("1W Deliv (Cr)", format="₹%.2f",
                            help="Total delivered value (₹ Cr) over last 1 week — real money flow"),
    "2W_deliv_cr":      st.column_config.NumberColumn("2W Deliv (Cr)", format="₹%.2f",
                            help="Total delivered value (₹ Cr) over last 2 weeks"),
    "1M_deliv_cr":      st.column_config.NumberColumn("1M Deliv (Cr)", format="₹%.2f",
                            help="Total delivered value (₹ Cr) over last 1 month"),
    "3M_deliv_cr":      st.column_config.NumberColumn("3M Deliv (Cr)", format="₹%.2f",
                            help="Total delivered value (₹ Cr) over last 3 months"),
    "deliv_per":        st.column_config.NumberColumn("Today Deliv%", format="%.1f%%",
                            help="Today's delivery % — compare with period averages to spot change"),
    "deliv_ratio":      st.column_config.NumberColumn("Deliv Ratio",  format="%.2f",
                            help=">1.2 = accumulating above norm, <0.8 = distributing below norm"),
}
_STOCK_SHOW = [
    "symbol", "company_name", "category", "close_price",
    "1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct",
    "1W_deliv_cr",      "2W_deliv_cr",      "1M_deliv_cr",      "3M_deliv_cr",
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
    "1W_deliv_cr":      st.column_config.NumberColumn("1W Deliv (Cr)", format="₹%.2f",
                            help="Delivered value (₹ Cr) over last 1 week"),
    "1M_deliv_cr":      st.column_config.NumberColumn("1M Deliv (Cr)", format="₹%.2f",
                            help="Delivered value (₹ Cr) over last 1 month"),
    "3M_deliv_cr":      st.column_config.NumberColumn("3M Deliv (Cr)", format="₹%.2f",
                            help="Delivered value (₹ Cr) over last 3 months"),
    "deliv_ratio":      st.column_config.NumberColumn("Deliv Ratio", format="%.2f",
                            help=">1.2 = accumulating, <0.8 = distributing"),
    "vol_ratio":        st.column_config.NumberColumn("Vol Ratio",   format="%.2f",
                            help="Today's volume vs 20-day avg"),
}
_SEARCH_SHOW = [
    "symbol", "company_name", "sector", "industry", "category",
    "close_price", "price_change_pct",
    "1W_price_chg_pct", "2W_price_chg_pct", "1M_price_chg_pct", "3M_price_chg_pct",
    "deliv_per", "1W_deliv_cr", "1M_deliv_cr", "3M_deliv_cr",
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


def _dv_ratio_cell(col, val) -> None:
    if pd.isna(val):
        col.markdown("<div style='font-size:13px'>—</div>", unsafe_allow_html=True)
        return
    if val >= 2.0:
        color, arrow = "#00c853", "↑↑"
    elif val >= 1.5:
        color, arrow = "#64dd17", "↑"
    elif val >= 0.75:
        color, arrow = "#888888", "→"
    elif val >= 0.5:
        color, arrow = "#ff6d00", "↓"
    else:
        color, arrow = "#d50000", "↓↓"
    col.markdown(
        f"<div style='color:{color};font-weight:700;font-size:13px'>{arrow} {val:.2f}x</div>",
        unsafe_allow_html=True,
    )


def _z_score_cell(col, val) -> None:
    if pd.isna(val):
        col.markdown("<div style='font-size:13px'>—</div>", unsafe_allow_html=True)
        return
    if val >= 2.0:
        color, tag = "#00c853", f"⚡ {val:+.1f}σ"
    elif val >= 1.0:
        color, tag = "#64dd17", f"↑ {val:+.1f}σ"
    elif val >= -1.0:
        color, tag = "#888888", f"{val:+.1f}σ"
    elif val >= -2.0:
        color, tag = "#ff6d00", f"↓ {val:+.1f}σ"
    else:
        color, tag = "#d50000", f"↓↓ {val:+.1f}σ"
    col.markdown(
        f"<div style='color:{color};font-weight:700;font-size:13px'>{tag}</div>",
        unsafe_allow_html=True,
    )


def _breadth_cell(col, val) -> None:
    if pd.isna(val):
        col.markdown("<div style='font-size:13px'>—</div>", unsafe_allow_html=True)
        return
    pct = val * 100
    if val >= 0.70:   color = "#00c853"
    elif val >= 0.50: color = "#64dd17"
    elif val >= 0.30: color = "#888888"
    else:             color = "#d50000"
    col.markdown(
        f"<div style='color:{color};font-weight:700;font-size:13px'>{pct:.0f}%</div>",
        unsafe_allow_html=True,
    )


def _deliv_cell(col, val) -> None:
    if pd.isna(val):
        col.markdown("<div style='font-size:13px'>—</div>", unsafe_allow_html=True)
    else:
        col.markdown(
            f"<div style='font-size:13px'>₹{val:.0f} Cr</div>",
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
                                  "1W Deliv Cr", "2W Deliv Cr", "1M Deliv Cr", "3M Deliv Cr",
                                  "DV Ratio", "Z-Score", "Breadth"]):
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
        _dv_ratio_cell(r[10], sec_row.get("dv_ratio", float("nan")))
        _z_score_cell(r[11], sec_row.get("z_score", float("nan")))
        _breadth_cell(r[12], sec_row.get("breadth", float("nan")))

        # ── Sub-sector rows ───────────────────────────────────────────────────
        if sec_open:
            sub_df = subsector_df[subsector_df["sector"] == sector].copy()

            sh = st.columns(SUBSECTOR_COL_WIDTHS)
            for col, lbl in zip(sh, ["", "", "**Sub-Sector**",
                                      "1W Price%", "2W Price%", "1M Price%", "3M Price%",
                                      "1W Deliv Cr", "2W Deliv Cr", "1M Deliv Cr", "3M Deliv Cr",
                                      "DV Ratio", "Z-Score", "Breadth"]):
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
                _dv_ratio_cell(sr[11], sub_row.get("dv_ratio", float("nan")))
                _z_score_cell(sr[12], sub_row.get("z_score", float("nan")))
                _breadth_cell(sr[13], sub_row.get("breadth", float("nan")))

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

    # Participation acceleration: is the normalized rate INCREASING toward the present?
    daily_100d = (out["100D_deliv_cr"] / 100).replace(0, float("nan"))
    norm_1w = (out["1W_deliv_cr"] / 5)  / daily_100d
    norm_2w = (out["2W_deliv_cr"] / 10) / daily_100d
    norm_1m = (out["1M_deliv_cr"] / 22) / daily_100d

    out["_dv"]  = out["dv_ratio"].fillna(1.0)          # RelativeDV  (35%)
    out["_br"]  = out["breadth"].fillna(0.5)            # Breadth     (25%)
    out["_z"]   = out["z_score"].fillna(0.0)            # RelativeStr (20%)
    out["_pm"]  = out.get("2W_price_chg_pct", out["1W_price_chg_pct"])  # Trend (10%)
    out["_acc"] = (                                     # Participation (10%)
        (norm_1w > norm_2w).astype(float) +
        (norm_2w > norm_1m).astype(float)
    )

    out["Score"] = (
        _normalize(out["_dv"])  * 35 +   # RelativeDV: today vs own 100D mean
        _normalize(out["_br"])  * 25 +   # Breadth: how many stocks are above norm
        _normalize(out["_z"])   * 20 +   # Z-Score: statistical abnormality
        _normalize(out["_pm"])  * 10 +   # Trend: price direction confirmation
        _normalize(out["_acc"]) * 10     # Participation: delivery acceleration
    ).round(1)

    def _sig(row) -> str:
        z  = row["_z"]
        dv = row["_dv"]
        br = row["_br"]
        pm = row["_pm"] if not pd.isna(row["_pm"]) else 0
        # Abnormal flow = Z >= 1.5 OR DV >= 1.5x OR both moderate (Z>=1.0 + DV>=1.2)
        # Broad = majority of stocks above their norm, OR signal is extreme
        abnormal = z >= 1.5 or dv >= 1.5 or (z >= 1.0 and dv >= 1.2)
        broad    = br >= 0.5 or z >= 2.0 or dv >= 2.0
        strong   = abnormal and broad
        if strong and pm > 0:      return "🟢 Accumulating"
        if strong:                  return "🟡 Buying Dips"
        if dv >= 0.75 and pm > 0:  return "🟠 Weak Rally"
        # Neutral: delivery near normal (0.6–0.75×) OR price flat — not actively distributing
        # Distributing: delivery clearly below normal (<0.6×) OR price down with low delivery
        if dv >= 0.60 and abs(pm) <= 1.5:   return "⚪ Neutral"
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
    st.markdown(
        "### 📈 1–2 Week Sector Outlook "
        "<span title='Score = 35% DV Ratio + 25% Breadth + 20% Z-Score + 10% Price Trend + 10% Delivery Acceleration. "
        "Score is a RELATIVE rank within today's cross-section — the best sector gets highest score. "
        "Accumulating = abnormal+broad delivery + price up. "
        "Buying Dips = abnormal+broad + price down (smart money absorbing). "
        "Weak Rally = normal delivery + price up (retail-driven, low conviction). "
        "Neutral = near-normal delivery, flat price. "
        "Distributing = below-normal delivery or price falling with weak flow.' "
        "style='cursor:help;color:#ffd600;font-size:16px'>ℹ️</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Score = **35% DV Ratio** (relative flow) + **25% Breadth** (stock participation) "
        "+ **20% Z-Score** (statistical abnormality) + **10% Price Trend** + **10% Delivery Acceleration** "
        "— relative rank within today's sector universe"
    )
    scored = _compute_outlook(sector_df)

    def _kpi_help(row) -> str:
        dv = row.get("dv_ratio", float("nan"))
        z  = row.get("z_score",  float("nan"))
        br = row.get("breadth",  float("nan"))
        sc = row.get("Score",    float("nan"))
        lines = []
        if not pd.isna(dv):
            dv_note = ("significantly above norm — institutional surge" if dv >= 1.5
                       else "above norm" if dv >= 1.0
                       else "below norm — reduced participation")
            lines.append(f"DV Ratio {dv:.2f}× — delivery today vs own 100D avg. {dv_note}.")
        if not pd.isna(z):
            z_note  = ("rare extreme event (top ~2.5% of days)" if abs(z) >= 2.0
                       else "elevated (top ~16% of days)" if abs(z) >= 1.0
                       else "normal range")
            lines.append(f"Z-Score {z:+.1f}σ — {z_note}.")
        if not pd.isna(br):
            br_note = ("broad — sector-wide, not single-stock" if br >= 0.5
                       else "narrow — dominated by 1–2 large-caps, not the whole sector")
            lines.append(f"Breadth {br*100:.0f}% — {br_note}.")
        if not pd.isna(sc):
            lines.append(f"Composite Score {sc:.0f}/100 — relative rank today.")
        return "\n\n".join(lines)

    # Top 3 — strongest buy signals
    st.markdown(
        "<div style='font-size:13px;font-weight:600;color:#00c853;margin-bottom:4px'>"
        "🏆 Strongest Signals (Buy / Accumulate)</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1, c2, c3], scored.head(3).iterrows()):
        dv = row.get("dv_ratio", float("nan"))
        z  = row.get("z_score",  float("nan"))
        br = row.get("breadth",  float("nan"))
        dv_str = f"{dv:.2f}x"     if not pd.isna(dv) else "—"
        z_str  = f"{z:+.1f}σ"     if not pd.isna(z)  else "—"
        br_str = f"{br*100:.0f}%" if not pd.isna(br) else "—"
        col.metric(
            label=f"{row['Signal']} — {row['sector']}",
            value=f"Score: {row['Score']:.0f}/100",
            delta=f"DV {dv_str}  |  Z {z_str}  |  Breadth {br_str}",
            help=_kpi_help(row),
        )

    # Bottom 3 — avoid / exit signals (only Distributing sectors)
    distrib = scored[scored["Signal"] == "🔴 Distributing"].tail(3)
    if not distrib.empty:
        st.markdown(
            "<div style='font-size:13px;font-weight:600;color:#d62728;margin:10px 0 4px 0'>"
            "⚠️ Weakest Signals (Avoid / Exit)</div>",
            unsafe_allow_html=True,
        )
        d1, d2, d3 = st.columns(3)
        for col, (_, row) in zip([d1, d2, d3], distrib.iterrows()):
            dv = row.get("dv_ratio", float("nan"))
            z  = row.get("z_score",  float("nan"))
            br = row.get("breadth",  float("nan"))
            dv_str = f"{dv:.2f}x"     if not pd.isna(dv) else "—"
            z_str  = f"{z:+.1f}σ"     if not pd.isna(z)  else "—"
            br_str = f"{br*100:.0f}%" if not pd.isna(br) else "—"
            col.metric(
                label=f"{row['Signal']} — {row['sector']}",
                value=f"Score: {row['Score']:.0f}/100",
                delta=f"DV {dv_str}  |  Z {z_str}  |  Breadth {br_str}",
                delta_color="inverse",
                help=_kpi_help(row),
            )

    st.plotly_chart(outlook_bar_chart(scored), use_container_width=True)
    st.markdown("---")

    # ── Master Performance Table ──────────────────────────────────────────────
    st.markdown("### 📋 Master Performance Table")

    with st.expander("❓ How to read this table", expanded=False):
        st.markdown("""
**Price % columns (1W / 2W / 1M / 3M)**
: Cumulative return from that many calendar days ago to today.
  Formula: *(today's close − start close) ÷ start close × 100*, weighted by today's turnover.
  🟢 green = sector gained &nbsp;|&nbsp; 🔴 red = sector fell over that window.

**Delivered Value columns (1W / 2W / 1M / 3M) — in ₹ Crore**
: Total ₹ actually delivered (taken home, not squared intraday) across all stocks in the sector over each period.
  Formula: Σ(daily turnover × delivery%) for every stock, every day in the window.
  This is **real money flow** — not a ratio or %. Higher = more institutional commitment.

---

**DV Ratio** — *Relative Flow Strength*
: Compares today's sector delivery to that sector's own 100-day daily average.
  Formula: **Today's DV ÷ (100-day total DV ÷ 100)**
  - `1.00x` = exactly average — normal participation
  - `2.84x` = today is 2.84× the daily norm — sector is surging
  - `0.40x` = well below normal — money flowing out

  **Why not just use raw ₹ Crore?** Banking does ₹7,000 Cr on a slow day. Defence averages ₹300 Cr.
  Raw ₹ always makes Banking look stronger. DV Ratio removes that size bias — each sector competes against *itself*.
  Defence at 2.84x is a bigger signal than Banking at 1.11x.

  Color: 🟢 ↑↑ ≥ 2x &nbsp;|&nbsp; 🟢 ↑ ≥ 1.5x &nbsp;|&nbsp; ⚪ → normal &nbsp;|&nbsp; 🟠 ↓ weak &nbsp;|&nbsp; 🔴 ↓↓ very weak

**Z-Score (σ)** — *Statistical Abnormality*
: Measures how many standard deviations today's delivery is above or below the sector's own 100-day mean.
  Formula: **(Today's DV − 100D mean) ÷ 100D std-dev**
  - `0σ` = exactly average
  - `+1σ` = above average (happens ~16% of trading days)
  - `+2σ` ⚡ = extreme surge (happens ~2.5% of days — roughly 2–3 times a year per sector)
  - `+3.5σ` = historically rare institutional event
  - Negative = below-average participation

  **Why Z-Score on top of DV Ratio?** DV Ratio only compares to the mean.
  Z-Score also accounts for each sector's *own volatility* — a calm sector needs a smaller move to be unusual.
  Metals bounces around a lot, so +1.5x is unremarkable. FMCG is steady, so +1.5x is a real event.
  Z-Score captures that difference.

  Color: ⚡ green ≥ +2σ &nbsp;|&nbsp; ↑ light-green ≥ +1σ &nbsp;|&nbsp; ⚪ normal &nbsp;|&nbsp; 🟠 ↓ weak &nbsp;|&nbsp; 🔴 ↓↓ < −2σ

**Breadth** — *Participation Width*
: Fraction of stocks in the sector where today's delivery value exceeds that stock's own 100-day daily average.
  Formula: **Stocks with today DV > own 100D avg ÷ total stocks with history**
  - `70%+` = most stocks are above norm — genuine sector-wide buying
  - `50%` = broad but not dominant
  - `17%` = only 2–3 stocks are above norm — the sector move is being driven by one large-cap, not the sector

  **Why Breadth matters:** HDFC Bank alone can push Banking's DV Ratio to 1.5x and make it look like a sector signal.
  Breadth exposes that. If breadth is 17%, it's a *stock* story, not a *sector* story — don't rotate into the whole sector.

  Color: 🟢 ≥ 70% &nbsp;|&nbsp; 🟡 ≥ 50% &nbsp;|&nbsp; ⚪ 30–50% &nbsp;|&nbsp; 🔴 < 30%

---

**Expanding rows**
: Click ▶ next to a **Sector** to see its sub-sectors.
  Click ▶ next to a **Sub-Sector** to see individual stock performance.
  The number in **(  )** is the stock count passing your turnover filter.

| Price % | DV Ratio + Breadth | What it likely means |
|---------|-------------------|----------------------|
| ↑ Up    | DV > 1.5x, Breadth > 50% | Strong institutional accumulation — high conviction |
| ↓ Down  | DV > 1.5x, Breadth > 50% | Smart money absorbing weakness — watch for bounce |
| ↑ Up    | DV normal, Breadth low  | Retail-driven rally, low conviction — may not sustain |
| ↓ Down  | DV < 1x, Breadth low    | Broad distribution — avoid or reduce exposure |
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
    st.caption(
        "Signal modes (DV Ratio / Z-Score / Breadth) remove size bias — each sector vs its own history. "
        "Sorted by signal strength, not sector size."
    )
    metric = st.radio(
        "Compare by",
        ["DV Ratio", "Z-Score", "Breadth", "Price Change %", "Delivered Value (Cr)"],
        horizontal=True,
    )

    if metric == "DV Ratio":
        fig = signal_bar_chart(
            sector_df, "dv_ratio",
            x_title="DV Ratio (today ÷ 100D daily avg)",
            fmt="{:.2f}x",
            ref_val=1.0,
            thresholds=[
                (2.0,  "#00c853"),
                (1.5,  "#64dd17"),
                (1.0,  "#888888"),
                (0.75, "#ff6d00"),
                (-999, "#d50000"),
            ],
        )
    elif metric == "Z-Score":
        fig = signal_bar_chart(
            sector_df, "z_score",
            x_title="Z-Score  σ  (standard deviations above 100D mean)",
            fmt="{:+.2f}σ",
            ref_val=0.0,
            thresholds=[
                (2.0,  "#00c853"),
                (1.0,  "#64dd17"),
                (0.0,  "#888888"),
                (-1.0, "#ff6d00"),
                (-999, "#d50000"),
            ],
        )
    elif metric == "Breadth":
        bdf = sector_df.copy()
        bdf["breadth_pct"] = bdf["breadth"] * 100
        fig = signal_bar_chart(
            bdf, "breadth_pct",
            x_title="Breadth  (% of stocks above own 100D daily avg)",
            fmt="{:.0f}%",
            ref_val=50.0,
            thresholds=[
                (70.0, "#00c853"),
                (50.0, "#64dd17"),
                (30.0, "#888888"),
                (0.0,  "#d50000"),
            ],
        )
    else:
        fig = period_comparison_chart(sector_df, metric)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Signal Legend", expanded=False):
        st.markdown("""
        | Signal | Trigger | Next 1-2W Implication |
        |--------|---------|----------------------|
        | 🟢 **Accumulating** | Abnormal flow (Z ≥ 1.5σ or DV ≥ 1.5×) + Broad (≥50% stocks) + Price rising | High-conviction institutional buying — follow with conviction |
        | 🟡 **Buying Dips**  | Abnormal + Broad flow + Price falling or flat | Smart money absorbing weakness — watch for bounce |
        | 🟠 **Weak Rally**   | Normal delivery (DV 0.75–1.5×) + Price rising | Price moving without institutional backing — lower conviction, may not sustain |
        | ⚪ **Neutral**      | Near-normal delivery (DV 0.60–0.75×) + flat price (within ±1.5%) | No clear signal in either direction — wait and watch |
        | 🔴 **Distributing** | Delivery clearly below normal (DV < 0.60×) OR price falling with weak flow | Institutional money leaving — avoid or reduce exposure |

        ---

        **Score interpretation (relative rank within today's sector universe):**
        | Score Range | Meaning |
        |-------------|---------|
        | 80–100 | Very strong signal — top institutional interest today |
        | 60–80  | Moderate positive — above average participation |
        | 40–60  | Neutral — no strong directional lean |
        | 20–40  | Below average — weak participation |
        | 0–20   | Weakest sector today — active outflow or very low delivery |

        ⚠️ **Score is RELATIVE**: even a score of 80/100 means "best in today's universe" — not absolute strength.
        On a broad market selloff day, the top scorer may still be distributing in absolute terms.

        **Key principle:** High absolute delivery ≠ strong signal. *Abnormal + Broad* delivery = strong signal.
        Banking doing ₹7,000 Cr is normal. Defence doing ₹800 Cr when it averages ₹300 Cr AND 70% of Defence stocks are above their own norm — that is the real signal.
        """)
