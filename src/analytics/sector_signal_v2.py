"""
Bottom-up ROBUST sector delivery signals.

Design rationale (validated in scripts/backtest_sector_signal_v2.py over the full
371-day history):

  The legacy sector delivery signal was TOP-DOWN: it summed raw delivery VALUE (₹)
  across a sector's stocks, then computed a single 1-day mean/std z-score on that
  sum. Three EOD-data facts make that structurally wrong:

    1. Delivery ₹-value is violently right-skewed (skew ≈ 4.2, std ≈ 3× MAD, one
       day ≈ 3× the median). mean/std z chases spike days; median/MAD does not.
    2. ₹-value drifts with price → a persistent positive-z bias ("+3.6σ median"),
       the reason the old absolute gates never fired. Delivery PERCENTAGE
       (deliv_per) is the drift-free primitive.
    3. A single mega-cap can be ~50% of a sector's delivery value (Telecom =
       BHARTIARTL), so the sector "signal" was really one stock. Measured
       coherence of the old signal with its own constituents' accumulation was
       ρ ≈ −0.05 (i.e. NONE).

  This module computes the signal BOTTOM-UP: robust, %-based, multi-day signals
  per STOCK first, then aggregates equal-weight (trimmed) across constituents, so
  the sector verdict is by-construction a summary of its stocks (measured coherence
  ρ ≈ +0.49, contradictions 26% → 10%), with forward IC no worse than the legacy
  factor (delivery IC ≈ 0.03–0.07 either way — RS momentum remains the real alpha).

Public entry point:
  get_robust_delivery_signals(as_of_date, min_turnover_lacs) -> DataFrame with one
  row per sector and columns:
    robz            robust, %-based, trimmed-mean delivery abnormality (σ-like)
    deliv_ratio5    median constituent 5d/100d delivery-% ratio (sustained flow)
    breadth_accum   fraction of constituents in genuine multi-day accumulation
    deliv_slope     median constituent 15d delivery-% slope (normalised)
    n_liq_v2        liquid constituents used
"""
from __future__ import annotations

import warnings
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.base import get_min_turnover_filter
from src.data.repository import query_dataframe
from src.logging_setup import get_logger

log = get_logger(__name__)

_TRIM_FRAC   = 0.10   # trim 10% each tail of per-stock robz before averaging
_MIN_NAMES   = 5      # need >=5 liquid constituents to aggregate a sector
_MIN_HIST    = 40     # per-stock min days of delivery history for a robust baseline
_ROBZ_CLIP   = 4.0    # clip robust z to +/- this
_ACCUM_R5    = 1.05   # per-stock: 5d/100d delivery ratio above this ...
# ... AND positive slope  => that stock is "accumulating" (multi-day, not 1-day)


def _slope_per_col(mat: pd.DataFrame, med: pd.Series) -> pd.Series:
    """OLS slope of each column over its rows (x = 0..n-1), normalised by `med`."""
    y = mat.values.astype(float)
    n = y.shape[0]
    if n < 5:
        return pd.Series(0.0, index=mat.columns)
    x = np.arange(n, dtype=float)
    xd = x - x.mean()
    denom = (xd ** 2).sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        ym = np.nanmean(y, axis=0)
        num = np.nansum((y - ym) * xd[:, None], axis=0)
    slope = num / denom
    return pd.Series(slope, index=mat.columns) / med.replace(0, np.nan)


def get_robust_delivery_signals(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """Per-sector robust bottom-up delivery signals as of `as_of_date`."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    # ~150 calendar days of the delivery-% panel for stocks liquid on the as-of day.
    panel = query_dataframe(
        """
        WITH liq AS (
            SELECT b.symbol
            FROM daily_data b
            WHERE b.trade_date = ?
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
        )
        SELECT b.symbol, s.sector, b.trade_date, b.deliv_per
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
        INNER JOIN liq ON liq.symbol = b.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector NOT IN ('ETF', 'Others')
          AND b.trade_date > (?::date - 210)
          AND b.trade_date <= ?
        ORDER BY b.symbol, b.trade_date
        """,
        [as_of_date, min_turnover_lacs, as_of_date, as_of_date],
    )
    if panel.empty:
        return pd.DataFrame(columns=[
            "sector", "robz", "deliv_ratio5", "breadth_accum", "deliv_slope", "n_liq_v2",
        ])

    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.date
    sym_sector = panel.drop_duplicates("symbol").set_index("symbol")["sector"]

    # dates x symbols delivery-% matrix
    dp = panel.pivot_table("deliv_per", "trade_date", "symbol").sort_index()
    if len(dp) < _MIN_HIST + 1:
        return pd.DataFrame(columns=[
            "sector", "robz", "deliv_ratio5", "breadth_accum", "deliv_slope", "n_liq_v2",
        ])

    today = dp.iloc[-1]
    hist  = dp.iloc[:-1]                       # today excluded from the baseline
    hist100 = hist.tail(100)

    enough = hist100.notna().sum() >= _MIN_HIST
    med = hist100.median()
    mad = (hist100 - med).abs().median()

    robz = ((today - med) / (1.4826 * mad.replace(0, np.nan))).clip(-_ROBZ_CLIP, _ROBZ_CLIP)
    ratio5 = dp.tail(5).mean() / hist100.mean().replace(0, np.nan)
    slope  = _slope_per_col(dp.tail(15), med)
    accum  = ((ratio5 > _ACCUM_R5) & (slope > 0)).astype(float)

    stk = pd.DataFrame({
        "sector": sym_sector.reindex(dp.columns).values,
        "robz":   robz.values,
        "ratio5": ratio5.values,
        "slope":  slope.values,
        "accum":  accum.values,
        "enough": enough.values,
        "has_today": today.notna().values,
    }, index=dp.columns)
    stk = stk[stk["enough"] & stk["has_today"]].dropna(subset=["robz", "ratio5"])

    def _trimmed_mean(s: pd.Series) -> float:
        s = s.sort_values()
        k = int(len(s) * _TRIM_FRAC)
        core = s.iloc[k:len(s) - k] if len(s) > 2 * k + 1 else s
        return float(core.mean())

    g = stk.groupby("sector")
    out = pd.DataFrame({
        "n_liq_v2":      g.size(),
        "robz":          g["robz"].apply(_trimmed_mean),
        "deliv_ratio5":  g["ratio5"].median(),
        "breadth_accum": g["accum"].mean(),
        "deliv_slope":   g["slope"].median(),
    })
    out = out[out["n_liq_v2"] >= _MIN_NAMES].reset_index()
    return out
