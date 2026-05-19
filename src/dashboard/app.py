"""
NSE Dashboard — Streamlit entry point.

dashboard.bat runs:  streamlit run src/dashboard/streamlit_app.py
streamlit_app.py is a thin shim that calls main() here.

Responsibilities:
  - Page config (must be first Streamlit call)
  - Sidebar: date picker, turnover filter, page selector
  - Route to the correct view's render()
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

_LAST_UPDATED_FILE = PROJECT_ROOT / "logs" / "last_updated.txt"


def _read_last_updated() -> str | None:
    """Return formatted fetch time from the marker file, or None if absent."""
    try:
        ts = datetime.fromisoformat(_LAST_UPDATED_FILE.read_text().strip())
        return ts.strftime("%d %b %Y %I:%M %p")
    except Exception:
        return None


def main() -> None:
    st.set_page_config(
        page_title="NSE Daily Cash Market",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    from src.core.config import get_config
    from src.data.repository import get_available_dates

    available_dates = get_available_dates(limit=90)

    if not available_dates:
        st.error(
            "No data found. Run `setup.bat` or "
            "`python -m src.cli backfill 60` to load data."
        )
        st.stop()

    cfg = get_config()
    default_cr = cfg.analytics.min_turnover_lacs / 100  # lakhs → crores for display

    with st.sidebar:
        st.header("Controls")

        selected_date = st.selectbox(
            "Trading Date",
            options=available_dates,
            format_func=lambda d: d.strftime("%d %b %Y (%a)"),
        )

        min_turnover_cr = st.slider(
            "Min Traded Value Filter (Cr)",
            min_value=0.0,
            max_value=10.0,
            value=float(default_cr),
            step=0.25,
            help="Hide stocks with traded value below this threshold. 1 Cr = ₹1 Crore",
        )
        min_turnover = min_turnover_cr * 100  # crores → lakhs for analytics layer

        page = st.radio(
            "Page",
            options=["Sector Overview", "Sector Performance", "Stock Detail", "🔄 Sector Rotation"],
            index=0,
        )

        st.divider()

        # Data freshness info
        last_updated = _read_last_updated()
        if last_updated:
            st.caption(f"Last fetched: {last_updated}")
        st.caption(f"History: {len(available_dates)} trading days")

        # One-click cache clear so user sees fresh data immediately after 7:30 PM job
        if st.button("Refresh Data", help="Clear cached queries and reload latest data from DB"):
            st.cache_data.clear()
            st.rerun()

    from src.dashboard.views import sector_overview, sector_performance, stock_detail

    if page == "Sector Overview":
        sector_overview.render(selected_date, float(min_turnover))
    elif page == "Sector Performance":
        sector_performance.render(selected_date, float(min_turnover))
    elif page == "Stock Detail":
        stock_detail.render(selected_date, float(min_turnover))
    elif page == "🔄 Sector Rotation":
        from src.dashboard.views import sector_rotation
        sector_rotation.render(selected_date, float(min_turnover))


if __name__ == "__main__":
    main()
