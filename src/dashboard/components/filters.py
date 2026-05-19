"""
Custom filter builder component for the Master Performance Table.

Usage:
    from src.dashboard.components.filters import render_filter_builder, apply_filters, render_filter_summary

    active = render_filter_builder()          # renders UI, returns current filters
    df     = apply_filters(df, active)        # applies AND logic
    render_filter_summary(active, len(df))    # shows "N sectors match" badge
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import state as ss
from src.dashboard.constants import SORT_COL_MAP, METRIC_LABELS

_OP_LABELS = ["> greater than", "< less than"]


def _read_widgets() -> list[dict]:
    """Snapshot current widget values back into dicts (called before any rerun)."""
    result = []
    for i, flt in enumerate(st.session_state.get(ss.CUSTOM_FILTERS, [])):
        label  = st.session_state.get(f"fm_{i}", flt.get("label", METRIC_LABELS[0]))
        op_raw = st.session_state.get(f"fo_{i}", _OP_LABELS[0])
        value  = float(st.session_state.get(f"fv_{i}", flt.get("value", 0.0)))
        result.append({
            "label": label,
            "op":    ">" if op_raw.startswith(">") else "<",
            "value": value,
        })
    return result


def render_filter_builder() -> list[dict]:
    """
    Render the Add-a-filter UI.

    Returns the current list of active filter dicts so the caller can
    apply them with `apply_filters()`.  Each dict has keys:
        label (str), op (">" | "<"), value (float)
    """
    st.markdown(
        "<div style='font-size:13px;font-weight:600;margin-bottom:4px'>"
        "Custom Filters &nbsp;<span style='font-weight:400;color:#aaa'>"
        "— type your own thresholds, all rows must match (AND logic)</span></div>",
        unsafe_allow_html=True,
    )

    if ss.CUSTOM_FILTERS not in st.session_state:
        st.session_state[ss.CUSTOM_FILTERS] = []

    # ── Header row ────────────────────────────────────────────────────────────
    _, fh2, fh3, fh4, _ = st.columns([0.12, 2.5, 1.5, 1.8, 0.5])
    fh2.markdown("<small style='color:#888'>Metric</small>",    unsafe_allow_html=True)
    fh3.markdown("<small style='color:#888'>Condition</small>", unsafe_allow_html=True)
    fh4.markdown("<small style='color:#888'>Value (%)</small>", unsafe_allow_html=True)

    # ── One row per active filter ─────────────────────────────────────────────
    delete_idx = None
    for i, flt in enumerate(st.session_state[ss.CUSTOM_FILTERS]):
        fc0, fc1, fc2, fc3, fc4 = st.columns([0.12, 2.5, 1.5, 1.8, 0.5])
        fc0.markdown(
            f"<div style='font-size:11px;color:#888;padding-top:8px'>#{i+1}</div>",
            unsafe_allow_html=True,
        )
        cur_lbl = flt.get("label", METRIC_LABELS[0])
        with fc1:
            st.selectbox(
                "m", METRIC_LABELS,
                index=METRIC_LABELS.index(cur_lbl) if cur_lbl in METRIC_LABELS else 0,
                key=f"fm_{i}", label_visibility="collapsed",
            )
        cur_op = flt.get("op", ">")
        with fc2:
            st.selectbox(
                "op", _OP_LABELS,
                index=0 if cur_op == ">" else 1,
                key=f"fo_{i}", label_visibility="collapsed",
            )
        with fc3:
            st.number_input(
                "val", value=float(flt.get("value", 0.0)),
                min_value=-100.0, max_value=100.0, step=0.5, format="%.1f",
                key=f"fv_{i}", label_visibility="collapsed",
            )
        with fc4:
            if st.button("✕", key=f"fdel_{i}", use_container_width=True):
                delete_idx = i

    # ── Delete: persist widget values first, then pop ─────────────────────────
    if delete_idx is not None:
        st.session_state[ss.CUSTOM_FILTERS] = _read_widgets()
        st.session_state[ss.CUSTOM_FILTERS].pop(delete_idx)
        st.rerun()

    # ── Add / Clear buttons ───────────────────────────────────────────────────
    ba, bc = st.columns([1.5, 1.5])
    with ba:
        if st.button("＋ Add Filter", use_container_width=True, key="add_filter_btn"):
            st.session_state[ss.CUSTOM_FILTERS].append(
                {"label": "1W Price%", "op": ">", "value": 0.0}
            )
            st.rerun()
    with bc:
        if st.button("✕ Clear All Filters", use_container_width=True, key="clear_filters_btn"):
            st.session_state[ss.CUSTOM_FILTERS] = []
            st.rerun()

    return _read_widgets()


def apply_filters(df: pd.DataFrame, active_filters: list[dict]) -> pd.DataFrame:
    """Apply a list of filter dicts (from render_filter_builder) to a DataFrame."""
    result = df.copy()
    for flt in active_filters:
        col = SORT_COL_MAP.get(flt["label"])
        if not col or col not in result.columns:
            continue
        val  = float(flt["value"])
        mask = result[col] > val if flt["op"] == ">" else result[col] < val
        result = result[mask]
    return result


def render_filter_summary(active_filters: list[dict], match_count: int) -> None:
    """Show 'Active filters (n): ... → N sector(s) match' caption."""
    if not active_filters:
        return
    parts = [f"**{f['label']}** {f['op']} {f['value']:.1f}%" for f in active_filters]
    st.caption(
        f"Active filters ({len(parts)}): " + " &nbsp;AND&nbsp; ".join(parts)
        + f" → **{match_count} sector(s)** match"
    )
