import pandas as pd
import streamlit as st

STOCK_TABLE_COLUMNS = {
    "symbol": st.column_config.TextColumn(
        "Symbol",
        help="NSE ticker symbol",
    ),
    "company_name": st.column_config.TextColumn(
        "Company",
        help="Full company name",
    ),
    "sector": st.column_config.TextColumn(
        "Sector",
        help="Broad market sector (Banking, IT, Healthcare, etc.)",
    ),
    "industry": st.column_config.TextColumn(
        "Sub-Sector",
        help="Industry sub-sector within the broad sector",
    ),
    "close_price": st.column_config.NumberColumn(
        "Close (₹)",
        format="₹%.2f",
        help="Day's closing price in Rupees",
    ),
    "price_change_pct": st.column_config.NumberColumn(
        "Chg %",
        format="%.2f%%",
        help="Price change % vs previous close. Positive = stock gained, Negative = stock fell",
    ),
    "turnover_cr": st.column_config.NumberColumn(
        "Traded Val (Cr)",
        format="%.2f",
        help="Total value traded today in Crores (₹). Higher = more market activity",
    ),
    "turnover_share_pct": st.column_config.ProgressColumn(
        "Sector Traded Val %",
        max_value=100,
        format="%.1f%%",
        help="This stock's % share of total sector traded value today — who dominates trading in this sector",
    ),
    "deliv_per": st.column_config.NumberColumn(
        "Delivery %",
        format="%.1f%%",
        help="% of traded quantity taken as delivery (not squared off intraday). Higher = more conviction buying",
    ),
    "deliv_per_10d_avg": st.column_config.NumberColumn(
        "10d Avg Deliv %",
        format="%.1f%%",
        help="Average delivery % over the past 10 trading days — the stock's normal baseline",
    ),
    "deliv_ratio": st.column_config.NumberColumn(
        "Deliv Ratio",
        format="%.2f",
        help="Today's delivery % ÷ 10-day avg. >1.2 = unusual accumulation, <0.8 = possible distribution",
    ),
    "vol_ratio": st.column_config.NumberColumn(
        "Vol Ratio",
        format="%.2f",
        help="Today's volume ÷ 20-day avg volume. >1.5 = unusually high activity",
    ),
    "deliv_value_cr": st.column_config.NumberColumn(
        "Deliv Val (Cr)",
        format="%.2f",
        help="₹ value of shares actually delivered (taken home). Delivery % × Traded Value",
    ),
    "deliv_value_share_pct": st.column_config.ProgressColumn(
        "Sector Deliv Share %",
        max_value=100,
        format="%.1f%%",
        help="This stock's % share of total sector delivery value today — who is driving real buying in this sector",
    ),
}

SECTOR_TABLE_COLUMNS = {
    "sector": st.column_config.TextColumn(
        "Sector",
        help="Broad market sector",
    ),
    "stock_count": st.column_config.NumberColumn(
        "Stocks",
        help="Number of stocks in this sector passing the turnover filter today",
    ),
    "wtd_price_change_pct": st.column_config.NumberColumn(
        "Wtd Price Chg %",
        format="%.2f%%",
        help="Turnover-weighted average price change % — heavier weight to larger stocks",
    ),
    "simple_price_change_pct": st.column_config.NumberColumn(
        "Avg Price Chg %",
        format="%.2f%%",
        help="Simple (equal-weight) average price change % across all stocks in the sector",
    ),
    "wtd_deliv_per": st.column_config.NumberColumn(
        "Wtd Delivery %",
        format="%.1f%%",
        help="Turnover-weighted average delivery % — heavier weight to larger stocks",
    ),
    "simple_deliv_per": st.column_config.NumberColumn(
        "Avg Delivery %",
        format="%.1f%%",
        help="Simple (equal-weight) average delivery % across all stocks in the sector",
    ),
    "top_delivery_symbol": st.column_config.TextColumn(
        "Top Deliv Stock",
        help="Stock with the highest delivery % in this sector today",
    ),
    "accumulation_count": st.column_config.NumberColumn(
        "Accumulating",
        help="Stocks with delivery ratio ≥ 1.2 (today's delivery significantly above their 10-day avg)",
    ),
    "distribution_count": st.column_config.NumberColumn(
        "Distributing",
        help="Stocks with delivery ratio < 0.8 (today's delivery significantly below their 10-day avg)",
    ),
    "total_turnover_cr": st.column_config.NumberColumn(
        "Traded Val (Cr)",
        format="%.1f",
        help="Total traded value across all stocks in this sector today (in Crores ₹)",
    ),
    "total_deliv_value_cr": st.column_config.NumberColumn(
        "Deliv Val (Cr)",
        format="%.1f",
        help="Total delivery value across all stocks in this sector today (in Crores ₹)",
    ),
}


def to_display_df(df: pd.DataFrame, col_config: dict = None) -> pd.DataFrame:
    """Convert lacs columns to crores and optionally filter to only configured columns."""
    df = df.copy()
    if "turnover_lacs" in df.columns:
        df["turnover_cr"] = df["turnover_lacs"] / 100
    if "deliv_value_lacs" in df.columns:
        df["deliv_value_cr"] = df["deliv_value_lacs"] / 100
    if "total_turnover_lacs" in df.columns:
        df["total_turnover_cr"] = df["total_turnover_lacs"] / 100
    if "total_deliv_value_lacs" in df.columns:
        df["total_deliv_value_cr"] = df["total_deliv_value_lacs"] / 100
    if col_config is not None:
        keep = [c for c in col_config if c in df.columns]
        df = df[keep]
    return df
