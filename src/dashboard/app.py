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


_CLOUD_CHECK_INTERVAL = 1800   # re-check for new snapshot every 30 minutes


def _cloud_startup() -> None:
    """
    Cloud mode: ensure the DuckDB snapshot is downloaded and up to date.

    Runs on every app load but only hits GitHub API once per 30 minutes.
    When a new snapshot is detected (your laptop uploaded after the 6:30 PM fetch),
    it re-downloads automatically — no manual "Refresh Data" needed.
    """
    import time as _time
    from src.core.cloud import is_cloud, ensure_database, get_snapshot_info, _CLOUD_DB_PATH
    if not is_cloud():
        return

    now = _time.time()
    last_check = st.session_state.get("_cloud_last_check", 0)
    needs_check = (now - last_check) >= _CLOUD_CHECK_INTERVAL

    if needs_check:
        # Snapshot mtime before — if it changes, a new DB was downloaded
        db_mtime_before = (
            _CLOUD_DB_PATH.stat().st_mtime if _CLOUD_DB_PATH.exists() else 0
        )
        with st.spinner("Checking for latest market data..."):
            ok = ensure_database()
        if not ok:
            st.error(
                "Could not download market data from GitHub Releases. "
                "Check that GITHUB_REPO and GITHUB_TOKEN are set in Streamlit secrets."
            )
            st.stop()
        st.session_state["_cloud_last_check"] = now
        st.session_state["_cloud_db_ready"]   = True
        db_mtime_after = (
            _CLOUD_DB_PATH.stat().st_mtime if _CLOUD_DB_PATH.exists() else 0
        )
        if db_mtime_after != db_mtime_before:
            # New snapshot was downloaded → clear all caches so pages show fresh data
            st.cache_data.clear()
            st.toast("Market data updated! Showing latest data.", icon="✅")

    # Cloud info banner
    info = get_snapshot_info()
    if info["last_updated"] is not None:
        last = info["last_updated"]
        last_str = last.strftime("%d %b %Y") if hasattr(last, "strftime") else str(last)
        st.info(
            f"**Cloud mode** — data as of **{last_str}** ({info['db_size_mb']:.0f} MB snapshot). "
            "Auto-refreshes within 30 min of each evening fetch on the host machine.",
            icon="📱",
        )


def main() -> None:
    st.set_page_config(
        page_title="NSE Daily Cash Market",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Responsive CSS ─────────────────────────────────────────────────────────
    st.markdown("""
<style>
/* ── Phone (≤ 640px): single column, compact padding ───────────────── */
@media screen and (max-width: 640px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"],
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 100% !important;
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    .main .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    [data-testid="stMetricValue"] > div { font-size: 1.05rem !important; }
    [data-testid="stMetricLabel"] > div { font-size: 0.75rem !important; }
}

/* ── Phablet / small tablet (641–900px): stack everything ──────────── */
@media screen and (min-width: 641px) and (max-width: 900px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"],
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    [data-testid="stMetricValue"] > div { font-size: 1.1rem !important; }
}

/* ── Tablet / laptop with sidebar (901–1200px): 3-per-row max ───────── *
 *  Covers 1024px laptop + open sidebar (~1030px content area),           *
 *  1280px laptop + open sidebar (~1030px content area).                  *
 *  30% min-width → 5-col KPI wraps to 3+2 rows (not 2+2+1).            */
@media screen and (min-width: 901px) and (max-width: 1200px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"],
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 30% !important;
        flex: 1 1 30% !important;
    }
}

/* ── Standard laptop / desktop (1201px+): Streamlit defaults apply ──── *
 *  Wide layout + sidebar ≈ 950–1200px content area. No overrides needed. */
</style>
""", unsafe_allow_html=True)

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

        if st.button(
            "🔄 Refresh Data",
            help=(
                "Clears cached queries and reloads from DB.\n"
                "On Cloud: also re-downloads the latest market snapshot from GitHub Releases "
                "so you always get the most recent data without rebooting."
            ),
        ):
            st.cache_data.clear()
            # Cloud: force re-download of the latest snapshot so new data
            # (fno_bhavcopy, daily_data, etc.) is picked up without a manual reboot.
            from src.core.cloud import is_cloud, _CLOUD_DB_PATH, _CLOUD_VER_PATH
            if is_cloud():
                st.session_state.pop("_cloud_db_ready", None)
                st.session_state.pop("_cloud_last_check", None)
                for p in [_CLOUD_DB_PATH, _CLOUD_VER_PATH]:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
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
