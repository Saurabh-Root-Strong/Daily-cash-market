"""
Price-Action character — per-stock candle-anatomy + trend-efficiency classifier.

Motivation (measured on the live DB, 1,008 liquid stocks, 60d window):
  Candle anatomy (body %, wick %, large-body / long-wick frequency, gap frequency)
  and TREND are near-INDEPENDENT in this market — every candle metric correlates
  ~0 with directional efficiency (|corr| < 0.10). So "big body ⇒ trending" is
  false; you cannot read trend off candle shape. A useful classifier therefore
  needs TWO orthogonal axes:

    A. Trend efficiency — Kaufman Efficiency Ratio (ER) = |net move| / path length
       over the window. ER→1 clean directional trend, ER→0 chop/range. This is the
       ONLY reliable trend/chop discriminator (validated: sorting by ER separates
       ~50% movers from ~17% non-movers at equal bar size).
    B. Bar conviction — average body % / large-body frequency (cross-sectionally
       ranked, because the absolute spread across stocks is tiny: p10 0.38→p90 0.48).

  Plus two orthogonal RISK overlays: gap frequency (overnight/event risk) and
  ATR % (volatility scale).

Four characters (A × B) + flags:
    📈 Clean Trend     — high ER, decisive bars, contained vol  (most tradable)
    🌊 Volatile Trend  — high ER but wide/whippy path           (trend, wider stops)
    🔀 Choppy/Whipsaw  — low ER, big bars, no net progress      (RISK: stop-hunt)
    😴 Quiet Range     — low ER, small bars, coiling            (await breakout)
  flags: ⚡ Gappy (gap-heavy) · 🔥 High-Vol (ATR top quartile)

Public entry point:
  get_price_action(as_of_date, lookback_days=60, min_turnover_lacs) -> DataFrame,
  one row per liquid stock with the raw metrics, the class, and the risk flags.
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

_LOOKBACK_DAYS   = 60      # trading days of candle history for the character
_MIN_DAYS        = 40      # min bars required to classify a stock
_MIN_GOOD_RANGE  = 30      # min bars with a non-zero high-low range

# Thresholds calibrated to the live cross-sectional distribution (see the probe in
# the module docstring). ER is already 0..1 normalised, so its cuts are absolute;
# body conviction is judged cross-sectionally (median split) because its absolute
# spread is tiny. Frequencies/vol use absolute cuts anchored at ~top-decile/quartile.
_ER_TREND        = 0.25    # ER >= this  ≈ top quartile → directional trend
_ER_RANGE        = 0.12    # ER <= this  ≈ bottom third → rangebound
_LARGE_BODY      = 0.60    # a bar's body% above this is a "large body"
_LONG_WICK       = 0.55    # a bar's combined wick% above this is a "long wick"
_GAP_PCT         = 1.0     # |open−prev_close| beyond this % is a gap
_GAPPY_FREQ      = 0.35    # gap on >=35% of bars → ⚡ Gappy (top-decile)
_HIVOL_ATR       = 4.5     # avg range/close % above this → 🔥 High-Vol (top-quartile)

# User-facing class labels (stable — the view filters on these exact strings).
CLEAN_TREND    = "📈 Clean Trend"
VOLATILE_TREND = "🌊 Volatile Trend"
CHOPPY         = "🔀 Choppy / Whipsaw"
QUIET_RANGE    = "😴 Quiet Range"
PA_CLASSES     = (CLEAN_TREND, VOLATILE_TREND, CHOPPY, QUIET_RANGE)


def get_price_action(
    as_of_date: date,
    lookback_days: int = _LOOKBACK_DAYS,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """Per-stock price-action character as of `as_of_date` (stocks liquid that day)."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    # ~2× the window in calendar days to guarantee `lookback_days` trading bars.
    cal_span = int(lookback_days * 2 + 20)
    P = query_dataframe(
        """
        WITH liq AS (
            SELECT symbol FROM daily_data
            WHERE trade_date = ? AND series IN ('EQ','SM','ST')
              AND turnover_lacs >= ?
        )
        SELECT b.symbol, b.trade_date,
               b.open_price AS o, b.high_price AS h, b.low_price AS l,
               b.close_price AS c, b.prev_close AS pc
        FROM daily_data b
        INNER JOIN liq ON liq.symbol = b.symbol
        WHERE b.series IN ('EQ','SM','ST')
          AND b.trade_date > (?::date - INTERVAL '1 day' * ?)
          AND b.trade_date <= ?
        ORDER BY b.symbol, b.trade_date
        """,
        [as_of_date, min_turnover_lacs, as_of_date, cal_span, as_of_date],
    )
    if P.empty:
        return pd.DataFrame(columns=[
            "symbol", "avg_body_pct", "avg_upper_wick_pct", "avg_lower_wick_pct",
            "large_body_freq", "long_wick_freq", "gap_freq", "efficiency_ratio",
            "atr_pct", "wick_bias", "net_ret_pct", "n_days",
            "pa_class", "pa_gappy", "pa_high_vol",
        ])

    P["trade_date"] = pd.to_datetime(P["trade_date"]).dt.date

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for sym, g in P.groupby("symbol", sort=False):
            g = g.sort_values("trade_date").tail(lookback_days)
            if len(g) < _MIN_DAYS:
                continue
            o = g["o"].to_numpy(float); h = g["h"].to_numpy(float)
            l = g["l"].to_numpy(float); c = g["c"].to_numpy(float)
            pc = g["pc"].to_numpy(float)
            rng = h - l
            good = rng > 0
            if good.sum() < _MIN_GOOD_RANGE:
                continue

            body = np.abs(c - o)
            upper = h - np.maximum(o, c)
            lower = np.minimum(o, c) - l
            bp = np.where(good, body / rng, np.nan)
            up = np.where(good, upper / rng, np.nan)
            lo = np.where(good, lower / rng, np.nan)
            wick = up + lo
            gap = np.where(pc > 0, np.abs(o - pc) / pc * 100, np.nan)

            er_den = np.sum(np.abs(np.diff(c)))
            er = float(abs(c[-1] - c[0]) / er_den) if er_den > 0 else 0.0
            atr_pct = float(np.nanmean(np.where(c > 0, rng / c * 100, np.nan)))
            net = float((c[-1] - c[0]) / c[0] * 100) if c[0] > 0 else 0.0

            rows.append({
                "symbol":             sym,
                "avg_body_pct":       float(np.nanmean(bp)),
                "avg_upper_wick_pct": float(np.nanmean(up)),
                "avg_lower_wick_pct": float(np.nanmean(lo)),
                "large_body_freq":    float(np.nanmean(bp > _LARGE_BODY)),
                "long_wick_freq":     float(np.nanmean(wick > _LONG_WICK)),
                "gap_freq":           float(np.nanmean(gap > _GAP_PCT)),
                "efficiency_ratio":   er,
                "atr_pct":            atr_pct,
                "wick_bias":          float(np.nanmean(lo) - np.nanmean(up)),  # + demand / − supply
                "net_ret_pct":        net,
                "n_days":             int(len(g)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Bar-conviction split is cross-sectional (absolute spread is tiny).
    body_median = float(df["avg_body_pct"].median())

    def _classify(r) -> str:
        trending = r["efficiency_ratio"] >= _ER_TREND
        ranging  = r["efficiency_ratio"] <= _ER_RANGE
        decisive = r["avg_body_pct"] >= body_median
        high_vol = r["atr_pct"] >= _HIVOL_ATR
        if trending:
            return VOLATILE_TREND if (high_vol or not decisive) else CLEAN_TREND
        # not a clean trend → chop / range, split by bar conviction.
        # (mid-ER names between the range and trend cuts land here too.)
        return CHOPPY if decisive else QUIET_RANGE

    df["pa_class"]    = df.apply(_classify, axis=1)
    df["pa_gappy"]    = df["gap_freq"] >= _GAPPY_FREQ
    df["pa_high_vol"] = df["atr_pct"]  >= _HIVOL_ATR
    return df.reset_index(drop=True)
