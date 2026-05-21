"""
NSE Dashboard — Streamlit entry point.

dashboard.bat runs:  streamlit run src/dashboard/streamlit_app.py
streamlit_app.py is a thin shim that calls main() here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

_LAST_UPDATED_FILE = PROJECT_ROOT / "logs" / "last_updated.txt"


def _read_last_updated() -> str | None:
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
    from src.dashboard.cache.queries import cached_available_dates

    all_dates = cached_available_dates(limit=500)   # full history for backtest

    if not all_dates:
        st.error("No data found. Run `setup.bat` or `python -m src.cli backfill 60` to load data.")
        st.stop()

    available_dates = all_dates[:90]  # most recent 90 for normal date selector

    cfg = get_config()
    default_cr = cfg.analytics.min_turnover_lacs / 100

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
        min_turnover = min_turnover_cr * 100

        page = st.radio(
            "Page",
            options=[
                "Sector Performance",
                "🔄 Sector Rotation",
                "🎯 Big Players (F&O)",
                "📊 F&O Activity",
                "📋 F&O Stock Signals",
                "🗓️ F&O Expiry Structure",
                "📈 Index Tracker",
                "🔬 Backtest",
            ],
            index=0,
        )

        st.divider()

        last_updated = _read_last_updated()
        if last_updated:
            st.caption(f"Last fetched: {last_updated}")
        st.caption(f"History: {len(all_dates)} trading days")

        if st.button("Refresh Data", help="Clear cached queries and reload latest data from DB"):
            st.cache_data.clear()
            st.rerun()

    from src.dashboard.views import sector_performance

    if page == "Sector Performance":
        sector_performance.render(selected_date, float(min_turnover))
    elif page == "🔄 Sector Rotation":
        from src.dashboard.views import sector_rotation
        sector_rotation.render(selected_date, float(min_turnover))
    elif page == "🎯 Big Players (F&O)":
        from src.dashboard.views import fao_tracker
        fao_tracker.render(selected_date)
    elif page == "📊 F&O Activity":
        from src.dashboard.views import fno_activity
        fno_activity.render(selected_date)
    elif page == "📋 F&O Stock Signals":
        from src.dashboard.views import fno_stocks
        fno_stocks.render(selected_date)
    elif page == "🗓️ F&O Expiry Structure":
        from src.dashboard.views import fno_expiry
        fno_expiry.render(selected_date)
    elif page == "📈 Index Tracker":
        from src.dashboard.views import index_tracker
        index_tracker.render(selected_date)
    elif page == "🔬 Backtest":
        from src.dashboard.views import backtest
        backtest.render(all_dates)   # full history for signal/check date pickers


if __name__ == "__main__":
    main()
