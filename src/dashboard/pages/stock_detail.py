from datetime import date
import streamlit as st

from src.analytics.delivery_signals import get_stock_metrics, get_stock_history
from src.dashboard.components.charts import stock_price_chart
from src.dashboard.components.tables import STOCK_TABLE_COLUMNS


def render(selected_date: date, min_turnover: float) -> None:
    st.header("Stock Detail")

    metrics_df = get_stock_metrics(selected_date, min_turnover_lacs=min_turnover)

    if metrics_df.empty:
        st.warning("No stock data for selected date. Run a backfill first.")
        return

    symbols = sorted(metrics_df["symbol"].unique().tolist())
    selected_symbol = st.selectbox("Select stock", symbols)

    if not selected_symbol:
        return

    row = metrics_df[metrics_df["symbol"] == selected_symbol]
    if row.empty:
        st.info("No data for this stock on selected date.")
        return

    r = row.iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    close = r.get("close_price", 0) or 0
    chg = r.get("price_change_pct", 0) or 0
    deliv = r.get("deliv_per", 0) or 0
    deliv_avg = r.get("deliv_per_10d_avg", 0) or 0
    deliv_ratio = r.get("deliv_ratio", 0) or 0
    sector = r.get("sector", "—")

    col1.metric("Close Price", f"₹{close:.2f}", f"{chg:+.2f}%")
    col2.metric("Delivery %", f"{deliv:.1f}%")
    col3.metric("10d Avg Delivery %", f"{deliv_avg:.1f}%")
    col4.metric("Delivery Ratio", f"{deliv_ratio:.2f}")
    col5.metric("Sector", sector)

    history_df = get_stock_history(selected_symbol, days=60)
    if not history_df.empty:
        fig = stock_price_chart(history_df, selected_symbol)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        show_cols = {k: v for k, v in STOCK_TABLE_COLUMNS.items() if k in row.columns}
        st.dataframe(row, column_config=show_cols, use_container_width=True)
