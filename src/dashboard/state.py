"""
Streamlit session-state key constants and toggle helpers.

All views import from here — a single place to rename a key without hunting
across four view files.
"""
from __future__ import annotations

import streamlit as st

# ── Key constants ─────────────────────────────────────────────────────────────
EXP_SECTORS     = "exp_sectors"
EXP_SUBSECTORS  = "exp_subsectors"
CUSTOM_FILTERS  = "custom_filters"
MASTER_SORT_COL = "master_sort_col"
MASTER_SORT_DIR = "master_sort_dir"


# ── Expand / collapse helpers ─────────────────────────────────────────────────
def toggle_sector(key: str) -> None:
    s: set = st.session_state.setdefault(EXP_SECTORS, set())
    s.discard(key) if key in s else s.add(key)


def toggle_subsector(key: str) -> None:
    s: set = st.session_state.setdefault(EXP_SUBSECTORS, set())
    s.discard(key) if key in s else s.add(key)


def is_sector_open(key: str) -> bool:
    return key in st.session_state.get(EXP_SECTORS, set())


def is_subsector_open(key: str) -> bool:
    return key in st.session_state.get(EXP_SUBSECTORS, set())


def collapse_all() -> None:
    st.session_state[EXP_SECTORS]    = set()
    st.session_state[EXP_SUBSECTORS] = set()
