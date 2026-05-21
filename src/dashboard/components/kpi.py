"""
KPI metric strip components — reused across multiple views.

Each function renders one horizontal row of st.metric cards.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def market_kpi_strip(sector_df: pd.DataFrame) -> None:
    """5-metric overview strip for the Sector Overview page."""
    total_stocks  = int(sector_df["stock_count"].sum())
    total_to_lacs = sector_df["total_turnover_lacs"].sum()
    avg_deliv     = (
        (sector_df["wtd_deliv_per"] * sector_df["stock_count"]).sum() / total_stocks
    )
    avg_price_chg = (
        (sector_df["wtd_price_change_pct"] * sector_df["stock_count"]).sum() / total_stocks
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Sectors",           len(sector_df),
              help="Number of broad market sectors with data today")
    k2.metric("Stocks",            total_stocks,
              help="Total stocks in today's universe (min turnover filter applied)")
    k3.metric("Total Traded Value", f"₹{total_to_lacs / 100:.0f} Cr",
              help="Combined traded value across all sectors today")
    k4.metric("Avg Delivery %",    f"{avg_deliv:.1f}%",
              help="Turnover-weighted average delivery % — higher = more conviction buying")
    k5.metric("Avg Price Chg",     f"{avg_price_chg:+.2f}%",
              help="Turnover-weighted average price change across all stocks today")


def sector_kpi_strip(row: pd.Series) -> None:
    """5-metric drilldown strip for a single sector inside Sector Overview."""
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Stocks",             int(row["stock_count"]),
              help="Number of stocks in this sector today")
    d2.metric("Wtd Delivery %",     f"{row['wtd_deliv_per']:.1f}%",
              help="Turnover-weighted average delivery % for this sector")
    d3.metric("Wtd Price Chg",      f"{row['wtd_price_change_pct']:+.2f}%",
              help="Turnover-weighted average price change for this sector")
    d4.metric("Traded Value",       f"₹{row['total_turnover_lacs'] / 100:.0f} Cr",
              help="Total traded value (turnover) in this sector today")
    top_sym = row.get("top_delivery_symbol")
    d5.metric("Top Delivery Stock", str(top_sym) if top_sym else "—",
              help="Stock with the highest delivery % in this sector today")


def performance_kpi_strip(sector_df: pd.DataFrame) -> None:
    """5-metric top-performers strip for the Sector Performance page."""
    def _top(col: str) -> tuple[str, float]:
        r = sector_df.nlargest(1, col).iloc[0]
        return r["sector"], float(r[col])

    c1, c2, c3, c4, c5 = st.columns(5)
    s, v = _top("dv_ratio");          c1.metric("Top DV Ratio",    s, f"{v:.2f}x vs 100D avg",
                                                  help="Highest relative flow strength — sector surging vs its own 100-day baseline")
    if "breadth" in sector_df.columns:
        s, v = _top("breadth")
        c2.metric("Top Breadth", s, f"{v*100:.0f}% stocks above norm",
                  help="Most stocks surging above their own 100D delivery baseline — broad vs narrow rally")
    else:
        s, v = _top("1W_deliv_cr")
        c2.metric("Top Deliv (1W)", s, f"₹{v:.0f} Cr")
    if "z_score" in sector_df.columns:
        s, v = _top("z_score")
        c3.metric("Top Z-Score", s, f"{v:+.1f}σ",
                  help="Most statistically abnormal delivery today — how many std-devs above its own 100-day norm")
    s, v = _top("1W_price_chg_pct");  c4.metric("Best Price (1W)", s, f"{v:+.2f}%")
    if "2W_price_chg_pct" in sector_df.columns:
        s, v = _top("2W_price_chg_pct")
        c5.metric("Best Price (2W)", s, f"{v:+.2f}%")


def stock_kpi_strip(row: pd.Series) -> None:
    """5-metric strip for a single stock on the Stock Detail page."""
    close       = float(row.get("close_price",      0) or 0)
    chg         = float(row.get("price_change_pct",  0) or 0)
    deliv       = float(row.get("deliv_per",          0) or 0)
    deliv_avg   = float(row.get("deliv_per_10d_avg",  0) or 0)
    deliv_ratio = float(row.get("deliv_ratio",        0) or 0)
    sector      = str(row.get("sector", "—"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Close Price",        f"₹{close:.2f}", f"{chg:+.2f}%")
    c2.metric("Delivery %",         f"{deliv:.1f}%")
    c3.metric("10d Avg Delivery %", f"{deliv_avg:.1f}%")
    c4.metric("Delivery Ratio",     f"{deliv_ratio:.2f}")
    c5.metric("Sector",             sector)
