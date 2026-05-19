"""
Dashboard-level UI constants — single source of truth for all views and components.

Import from here rather than repeating literals across files.
"""
from __future__ import annotations

# ── Color palette ─────────────────────────────────────────────────────────────
POSITIVE_COLOR = "#2ca02c"   # green
NEGATIVE_COLOR = "#d62728"   # red
NEUTRAL_COLOR  = "#888888"   # grey
ACCENT_COLOR   = "rgba(255,165,0,0.4)"  # orange (dotted line overlays)

SIGNAL_COLORS: dict[str, str] = {
    "🟢 Accumulating": "#2ca02c",
    "🟡 Buying Dips":  "#f0b429",
    "🟠 Weak Rally":   "#f58518",
    "🔴 Distributing": "#d62728",
}

# ── Chart theme ───────────────────────────────────────────────────────────────
PLOT_BG    = "rgba(0,0,0,0)"
PAPER_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.08)"

# ── Period definitions ────────────────────────────────────────────────────────
PERIOD_LABELS = ["1W", "2W", "1M", "3M"]

PRICE_KEYS = [
    "1W_price_chg_pct",
    "2W_price_chg_pct",
    "1M_price_chg_pct",
    "3M_price_chg_pct",
]

DELIV_KEYS = [
    "1W_deliv_pct",
    "2W_deliv_pct",
    "1M_deliv_pct",
    "3M_deliv_pct",
]

# ── Filter / sort label → DataFrame column map ───────────────────────────────
SORT_COL_MAP: dict[str, str] = {
    "1W Price%": "1W_price_chg_pct",
    "2W Price%": "2W_price_chg_pct",
    "1M Price%": "1M_price_chg_pct",
    "3M Price%": "3M_price_chg_pct",
    "1W Deliv%": "1W_deliv_pct",
    "2W Deliv%": "2W_deliv_pct",
    "1M Deliv%": "1M_deliv_pct",
    "3M Deliv%": "3M_deliv_pct",
}

METRIC_LABELS: list[str] = list(SORT_COL_MAP.keys())

# ── Column header tooltips (hover titles) ─────────────────────────────────────
COL_TOOLTIPS: dict[str, str] = {
    "1W Price%": "Cumulative price return over the last 7 days.\n"
                 "(end close − start close) ÷ start close × 100, weighted by today's turnover.",
    "2W Price%": "Cumulative price return over the last 14 days.",
    "1M Price%": "Cumulative price return over the last 30 days.",
    "3M Price%": "Cumulative price return over the last 90 days (one quarter).",
    "1W Deliv%": "Turnover-weighted avg delivery % over last 7 days.\n"
                 "Higher = more shares held overnight = investor conviction buying.",
    "2W Deliv%": "Turnover-weighted avg delivery % over last 14 days.",
    "1M Deliv%": "Turnover-weighted avg delivery % over last 30 days.",
    "3M Deliv%": "Turnover-weighted avg delivery % over last 90 days — long-term conviction baseline.",
}

# ── Master table column-width ratios ─────────────────────────────────────────
# [expand_btn | name | 1W% | 2W% | 1M% | 3M% | 1WD | 2WD | 1MD | 3MD]
SECTOR_COL_WIDTHS:    list[float] = [0.28, 2.0,  0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
SUBSECTOR_COL_WIDTHS: list[float] = [0.28, 0.28, 1.9,  0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

# ── Comparison chart bar colors (1W → 2W → 1M → 3M) ─────────────────────────
PERIOD_COLORS: list[str] = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
