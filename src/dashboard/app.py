"""
NSE Dashboard — Streamlit entry point.

dashboard.bat runs:  streamlit run src/dashboard/streamlit_app.py
streamlit_app.py is a thin shim that calls main() here.
"""
from __future__ import annotations

import os
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
        raw = _LAST_UPDATED_FILE.read_text().strip()
        # File now contains latest trade_date (YYYY-MM-DD) from DB, not a wall-clock timestamp
        try:
            from datetime import date as _date
            d = _date.fromisoformat(raw[:10])
            suffix = " (Today)" if d == _date.today() else ""
            return d.strftime("%d %b %Y") + suffix
        except ValueError:
            pass
        # Fallback: old format with full timestamp
        ts = datetime.fromisoformat(raw)
        return ts.strftime("%d %b %Y %I:%M %p")
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_health():
    from src.analytics.data_health import run_health_check
    return run_health_check(lookback_days=10)


def _render_data_health_widget() -> None:
    """Compact sidebar widget showing data completeness for all tables."""
    try:
        h = _cached_health()
    except Exception:
        return

    if h.has_errors:
        overall = "❌ Data Gaps Detected"
        color   = "#FF4444"
    elif h.warn_sources:
        overall = "⚠️ Minor Lag (Expected)"
        color   = "#FF9800"
    else:
        overall = "✅ All Data Current"
        color   = "#4CAF50"

    with st.expander(overall, expanded=h.has_errors):
        for s in h.sources.values():
            latest_str = s.latest_date.strftime("%d %b") if s.latest_date else "—"
            if s.level == "ok":
                st.caption(f"✅ {s.label}: {latest_str}")
            elif s.level == "warn":
                st.caption(f"⚠️ {s.label}: {latest_str} *(NSDL lag — normal)*")
            else:
                missing_str = ", ".join(d.strftime("%d %b") for d in s.critical_missing)
                st.markdown(
                    f"<span style='color:#FF4444'>❌ **{s.label}**: missing {missing_str}</span>",
                    unsafe_allow_html=True,
                )
                st.caption("Run: `python -m src.cli fill-gaps 10`")


def _inject_streamlit_secrets() -> None:
    """
    Push Streamlit secrets into os.environ so the rest of the code can read them
    via os.environ.get() regardless of whether it's cloud or local.
    Called once before any other code runs.
    """
    try:
        for key, val in st.secrets.items():
            if isinstance(val, str) and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass   # No secrets file — local mode


def _cloud_startup() -> None:
    """
    Cloud mode initialisation — runs once per cold start.
    Downloads DuckDB snapshot from GitHub Releases before anything else loads.
    Shows a spinner so mobile users know what's happening.
    """
    from src.core.cloud import is_cloud, ensure_database, get_snapshot_info
    if not is_cloud():
        return

    if not st.session_state.get("_cloud_db_ready"):
        with st.spinner("Downloading latest market data snapshot..."):
            ok = ensure_database()
        if not ok:
            st.error(
                "Could not download market data from GitHub Releases. "
                "Check that GITHUB_REPO and GITHUB_TOKEN are set in Streamlit secrets."
            )
            st.stop()
        st.session_state["_cloud_db_ready"] = True

    # Cloud info banner
    info = get_snapshot_info()
    if info["last_updated"] is not None:
        last = info["last_updated"]
        last_str = last.strftime("%d %b %Y") if hasattr(last, "strftime") else str(last)
        st.info(
            f"**Cloud mode** — data as of **{last_str}** ({info['db_size_mb']:.0f} MB snapshot). "
            "Updated after each evening market fetch on the host machine.",
            icon="📱",
        )


def main() -> None:
    st.set_page_config(
        page_title="NSE Daily Cash Market",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_streamlit_secrets()   # push secrets → os.environ before anything reads them
    _cloud_startup()               # no-op in local mode; downloads DB snapshot in cloud

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
            help=(
                "Select a trading date to analyse. "
                "Top of the list = most recent trading day. "
                "All signals, charts and sector data update for the selected date. "
                "Use older dates to review past predictions and see if they were correct."
            ),
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
                "📈 Futures Analysis",
                "📊 Options Analysis",
                "📈 Index Tracker",
                "🔮 Index Prediction",
                "🔬 Backtest",
                "🌍 FPI Capital Flow",
                "🧠 Prediction Memory",
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

        # ── Data Health Widget ─────────────────────────────────────────────
        _render_data_health_widget()

    from src.dashboard.views import sector_performance

    if page == "Sector Performance":
        sector_performance.render(selected_date, float(min_turnover))
    elif page == "🔄 Sector Rotation":
        from src.dashboard.views import sector_rotation
        sector_rotation.render(selected_date, float(min_turnover), all_dates=all_dates)
    elif page == "🎯 Big Players (F&O)":
        from src.dashboard.views import fao_tracker
        fao_tracker.render(selected_date)
    elif page == "📈 Futures Analysis":
        from src.dashboard.views import futures_analysis
        futures_analysis.render(selected_date)
    elif page == "📊 Options Analysis":
        from src.dashboard.views import options_analysis
        options_analysis.render(selected_date)
    elif page == "📈 Index Tracker":
        from src.dashboard.views import index_tracker
        index_tracker.render(selected_date)
    elif page == "🔮 Index Prediction":
        from src.dashboard.views import index_prediction
        index_prediction.render(selected_date)
    elif page == "🔬 Backtest":
        from src.dashboard.views import backtest
        backtest.render(all_dates)   # full history for signal/check date pickers
    elif page == "🌍 FPI Capital Flow":
        from src.dashboard.views import fpi_flows
        fpi_flows.render(selected_date)
    elif page == "🧠 Prediction Memory":
        from src.dashboard.views import prediction_memory
        prediction_memory.render(selected_date)


if __name__ == "__main__":
    main()
