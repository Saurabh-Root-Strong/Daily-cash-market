import streamlit as st

STOCK_TABLE_COLUMNS = {
    "symbol": st.column_config.TextColumn("Symbol"),
    "company_name": st.column_config.TextColumn("Company"),
    "sector": st.column_config.TextColumn("Sector"),
    "close_price": st.column_config.NumberColumn("Close", format="₹%.2f"),
    "price_change_pct": st.column_config.NumberColumn("Chg %", format="%.2f%%"),
    "turnover_lacs": st.column_config.NumberColumn("Turnover (L)", format="%.0f"),
    "deliv_per": st.column_config.NumberColumn("Deliv %", format="%.1f%%"),
    "deliv_per_10d_avg": st.column_config.NumberColumn("10d Avg Deliv %", format="%.1f%%"),
    "deliv_ratio": st.column_config.NumberColumn("Deliv Ratio", format="%.2f"),
    "vol_ratio": st.column_config.NumberColumn("Vol Ratio", format="%.2f"),
    "deliv_value_lacs": st.column_config.NumberColumn("Deliv Val (L)", format="%.0f"),
    "turnover_share_pct": st.column_config.ProgressColumn("TO Share %", max_value=100, format="%.1f%%"),
    "deliv_value_share_pct": st.column_config.ProgressColumn("Deliv Share %", max_value=100, format="%.1f%%"),
}

SECTOR_TABLE_COLUMNS = {
    "sector": st.column_config.TextColumn("Sector"),
    "stock_count": st.column_config.NumberColumn("Stocks"),
    "wtd_price_change_pct": st.column_config.NumberColumn("Wtd Price Chg %", format="%.2f%%"),
    "simple_price_change_pct": st.column_config.NumberColumn("Avg Price Chg %", format="%.2f%%"),
    "wtd_deliv_per": st.column_config.NumberColumn("Wtd Deliv %", format="%.1f%%"),
    "simple_deliv_per": st.column_config.NumberColumn("Avg Deliv %", format="%.1f%%"),
    "top_delivery_symbol": st.column_config.TextColumn("Top Deliv Stock"),
    "accumulation_count": st.column_config.NumberColumn("Acc Stocks"),
    "distribution_count": st.column_config.NumberColumn("Dist Stocks"),
    "total_turnover_lacs": st.column_config.NumberColumn("Total TO (L)", format="%.0f"),
    "total_deliv_value_lacs": st.column_config.NumberColumn("Deliv Val (L)", format="%.0f"),
}
