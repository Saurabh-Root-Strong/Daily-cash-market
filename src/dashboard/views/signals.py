from datetime import date
import streamlit as st

from src.analytics.delivery_signals import get_top_accumulation, get_top_distribution
from src.dashboard.components.tables import STOCK_TABLE_COLUMNS, to_display_df


def render(selected_date: date, min_turnover: float) -> None:
    st.header("Accumulation / Distribution Signals")

    tab1, tab2 = st.tabs(["Top Accumulation", "Top Distribution"])

    with tab1:
        st.markdown("""
        **Accumulation**: Stocks with delivery ratio significantly above their 10-day average.
        High delivery means buyers are taking delivery rather than squaring off intraday — conviction buy.
        """)
        acc_df = get_top_accumulation(selected_date, limit=20)
        if acc_df.empty:
            st.info("No accumulation signals for selected date.")
        else:
            acc_df = to_display_df(acc_df, STOCK_TABLE_COLUMNS)
            show_cols = {k: v for k, v in STOCK_TABLE_COLUMNS.items() if k in acc_df.columns}
            st.dataframe(acc_df, column_config=show_cols, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("""
        **Distribution**: Stocks with price up >1% but delivery ratio below average.
        Price rising on low delivery often indicates selling into strength / weak hands buying.
        """)
        dist_df = get_top_distribution(selected_date, limit=20)
        if dist_df.empty:
            st.info("No distribution signals for selected date.")
        else:
            dist_df = to_display_df(dist_df, STOCK_TABLE_COLUMNS)
            show_cols = {k: v for k, v in STOCK_TABLE_COLUMNS.items() if k in dist_df.columns}
            st.dataframe(dist_df, column_config=show_cols, use_container_width=True, hide_index=True)
