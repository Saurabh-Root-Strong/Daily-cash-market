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
    "⚪ Neutral":      "#888888",
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
    "1W_deliv_cr",
    "2W_deliv_cr",
    "1M_deliv_cr",
    "3M_deliv_cr",
]

# ── Filter / sort label → DataFrame column map ───────────────────────────────
SORT_COL_MAP: dict[str, str] = {
    "1W Price%":   "1W_price_chg_pct",
    "2W Price%":   "2W_price_chg_pct",
    "1M Price%":   "1M_price_chg_pct",
    "3M Price%":   "3M_price_chg_pct",
    "1W Deliv Cr": "1W_deliv_cr",
    "2W Deliv Cr": "2W_deliv_cr",
    "1M Deliv Cr": "1M_deliv_cr",
    "3M Deliv Cr": "3M_deliv_cr",
    "DV Ratio":    "dv_ratio",
    "Z-Score":     "z_score",
    "Breadth":     "breadth",
}

METRIC_LABELS: list[str] = list(SORT_COL_MAP.keys())

# ── Column header tooltips (hover titles) ─────────────────────────────────────
COL_TOOLTIPS: dict[str, str] = {
    "1W Price%": "Cumulative price return over the last 7 days.\n"
                 "(end close − start close) ÷ start close × 100, weighted by today's turnover.",
    "2W Price%": "Cumulative price return over the last 14 days.",
    "1M Price%": "Cumulative price return over the last 30 days.",
    "3M Price%": "Cumulative price return over the last 90 days (one quarter).",
    "1W Deliv Cr": "Total delivered value (₹ Cr) over last 7 days.\n"
                   "= Σ(daily turnover × delivery%) for every stock in the sector.\n"
                   "Real money flow — how much ₹ was taken home, not traded intraday.",
    "2W Deliv Cr": "Total delivered value (₹ Cr) over last 14 days.",
    "1M Deliv Cr": "Total delivered value (₹ Cr) over last 30 days.",
    "3M Deliv Cr": "Total delivered value (₹ Cr) over last 90 days — long-term money flow baseline.",
    "DV Ratio":    "Relative Flow Strength — removes sector size bias entirely.\n"
                   "= today's delivered value ÷ (100-day daily delivery avg)\n"
                   "Each sector is compared to its own history, not to other sectors.\n"
                   "Banking 1.04x = normal  |  Defence 2.91x = surging → Defence is the real signal.",
    "Z-Score":     "Statistical Abnormality — how unusual is today's institutional participation?\n"
                   "= (today's DV − 100D mean daily DV) ÷ 100D std-dev of daily DV\n"
                   "Accounts for each sector's own volatility — a steady sector needs a smaller move.\n"
                   "Z > 2: extreme surge (top ~2.5% of days)  |  Z 1–2: above normal\n"
                   "Z −1 to 1: normal range  |  Z < −2: very weak participation.",
    "Breadth":     "Participation Breadth — fraction of stocks where today's DV > its own 100D daily avg.\n"
                   "= stocks with today_DV_i > avg_100D_daily_DV_i (each stock vs its own history)\n"
                   "Prevents one large-cap from masking a narrow rally.\n"
                   "70%+ = broad surge (most stocks above norm)  |  50-70% = moderate\n"
                   "30-50% = narrow  |  <30% = very few stocks above their norm.",
}

# ── Master table column-width ratios ─────────────────────────────────────────
# [expand | name | 1W% | 2W% | 1M% | 3M% | 1WD | 2WD | 1MD | 3MD | DVRatio | Z-Score | Breadth]
SECTOR_COL_WIDTHS:    list[float] = [0.28, 1.8,  0.65, 0.65, 0.65, 0.65, 0.70, 0.70, 0.70, 0.70, 0.72, 0.65, 0.60]
SUBSECTOR_COL_WIDTHS: list[float] = [0.28, 0.28, 1.6,  0.65, 0.65, 0.65, 0.65, 0.70, 0.70, 0.70, 0.70, 0.72, 0.65, 0.60]

# ── Comparison chart bar colors (1W → 2W → 1M → 3M) ─────────────────────────
PERIOD_COLORS: list[str] = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
