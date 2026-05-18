import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="NSE Daily Cash Market",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def _get_connection():
    from src.data.connection import get_raw_connection
    return get_raw_connection()


def _get_available_dates():
    from src.data.repository import get_available_dates
    return get_available_dates(limit=90)


def main():
    st.title("NSE Daily Cash Market Dashboard")

    available_dates = _get_available_dates()

    if not available_dates:
        st.error("No data found. Run `setup.bat` or `python -m src.cli backfill 60` to load data.")
        st.stop()

    from src.config_loader import load_config
    cfg = load_config()

    with st.sidebar:
        st.header("Controls")

        selected_date = st.selectbox(
            "Trading Date",
            options=available_dates,
            format_func=lambda d: d.strftime("%d %b %Y (%a)"),
        )

        default_cr = cfg["analytics"]["min_turnover_lacs"] / 100
        min_turnover_cr = st.slider(
            "Min Traded Value Filter (Cr)",
            min_value=0.0,
            max_value=10.0,
            value=float(default_cr),
            step=0.25,
            help="Hide stocks with traded value below this threshold. 1 Cr = ₹1 Crore",
        )
        min_turnover = min_turnover_cr * 100  # convert back to lakhs for analytics

        page = st.radio(
            "Page",
            options=["Sector Overview", "Sector Performance", "Stock Detail", "Signals"],
            index=0,
        )

        st.divider()
        st.caption(f"Data: {len(available_dates)} trading days")

    from src.dashboard.views import sector_overview, sector_performance, stock_detail, signals

    if page == "Sector Overview":
        sector_overview.render(selected_date, float(min_turnover))
    elif page == "Sector Performance":
        sector_performance.render(selected_date, float(min_turnover))
    elif page == "Stock Detail":
        stock_detail.render(selected_date, float(min_turnover))
    elif page == "Signals":
        signals.render(selected_date, float(min_turnover))


if __name__ == "__main__":
    main()
