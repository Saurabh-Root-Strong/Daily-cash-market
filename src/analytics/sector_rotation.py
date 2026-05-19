"""
Sector Rotation — Smart Money / Institutional Activity Detector.

Core insight: Retail traders square off intraday (no delivery). Institutions
and smart money take DELIVERY. So rising delivery % = rising institutional
conviction, falling delivery % = institutions exiting.

But delivery % alone is misleading — we weight by turnover (price × qty) so
a ₹5000 stock at 60% delivery outweighs a ₹50 stock at 80% delivery.

Signal matrix (price direction × delivery momentum):
  ┌──────────────────────────────────────────────────────────┐
  │          │  Delivery RISING   │  Delivery FALLING        │
  │──────────┼────────────────────┼──────────────────────────│
  │ Price UP │ Confirmed Buy ✅   │ Distribution Trap ⚠️     │
  │Price DOWN│ Secret Accum 🔥   │ Active Selling ❌         │
  └──────────┴────────────────────┴──────────────────────────┘

"Secret Accumulation" (price down + delivery up) is the STRONGEST signal:
institutions are quietly buying while retail panics — this precedes big rallies.
"Distribution Trap" is the most DANGEROUS: institutions exit into a retail-driven
price rally. Retail is the exit liquidity for smart money.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.base import get_min_turnover_filter
from src.data.repository import query_dataframe
from src.logging_setup import get_logger

log = get_logger(__name__)

_LOOKBACK_DAYS = 100


def _slope(series: pd.Series) -> float:
    """Linear regression slope, normalised by series mean so units are comparable."""
    y = series.dropna().values
    if len(y) < 8:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    mean = y.mean() if y.mean() != 0 else 1.0
    return float(slope / mean * 100)   # % change per trading day


def _normalize_0_100(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100


def get_sector_rotation(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    lookback_days: int = _LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Returns one row per sector with smart-money rotation signals.

    Columns
    -------
    sector, signal, horizon, accum_score,
    delivery_momentum,        # (1W_deliv - 3M_deliv) / 3M_deliv × 100
    slope,                    # delivery trend direction (100-day linear reg)
    deliv_1w/1m/3m,           # turnover-weighted delivery % per period
    price_1w/1m/3m,           # cumulative price return per period
    deliv_val_1w_cr/3m_cr,    # actual ₹ value delivered (Crores)
    action                    # plain-English recommendation
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    start = as_of_date - timedelta(days=lookback_days)

    # One row per sector per trading day — turnover-weighted aggregation
    sql = """
        SELECT
            s.sector,
            b.trade_date,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)              AS wtd_deliv_per,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100  AS deliv_value_cr,
            SUM(b.turnover_lacs) / 100                         AS turnover_cr,
            SUM(
                CASE WHEN b.prev_close > 0
                THEN (b.close_price - b.prev_close) / b.prev_close * 100
                     * b.turnover_lacs END
            ) / NULLIF(SUM(CASE WHEN b.prev_close > 0 THEN b.turnover_lacs END), 0)
                AS wtd_price_chg
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector NOT IN ('ETF', 'Others')
          AND b.trade_date > ?
          AND b.trade_date <= ?
          AND b.turnover_lacs >= ?
        GROUP BY s.sector, b.trade_date
        ORDER BY s.sector, b.trade_date
    """
    raw = query_dataframe(sql, [start, as_of_date, min_turnover_lacs])
    if raw.empty:
        return pd.DataFrame()

    records = []

    for sector, grp in raw.groupby("sector"):
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        if len(grp) < 10:
            continue

        def _wavg(days: int, col: str) -> float:
            """Turnover-weighted average across days in the period.

            Each day already has a within-day turnover-weighted value (from SQL).
            We further weight those daily values by the day's total turnover so that
            a high-volume budget day doesn't count the same as a thin holiday session.
            """
            cutoff = pd.Timestamp(as_of_date - timedelta(days=days))
            sub = grp.loc[grp["trade_date"] > cutoff].dropna(subset=[col, "turnover_cr"])
            if len(sub) < 2:
                return float("nan")
            total_w = sub["turnover_cr"].sum()
            if total_w == 0:
                return float("nan")
            return float((sub[col] * sub["turnover_cr"]).sum() / total_w)

        def _sum(days: int, col: str) -> float:
            """Sum a value column over the period (used for ₹ delivery value)."""
            cutoff = pd.Timestamp(as_of_date - timedelta(days=days))
            sub = grp.loc[grp["trade_date"] > cutoff, col].dropna()
            return float(sub.sum()) if len(sub) >= 2 else float("nan")

        # ── Period delivery % — double-weighted: stocks within day (SQL) + days by turnover
        d1w = _wavg(7,  "wtd_deliv_per")
        d2w = _wavg(14, "wtd_deliv_per")
        d1m = _wavg(30, "wtd_deliv_per")
        d3m = _wavg(90, "wtd_deliv_per")

        # ── Period price returns — turnover-weighted across days
        p1w = _wavg(7,  "wtd_price_chg")
        p1m = _wavg(30, "wtd_price_chg")
        p3m = _wavg(90, "wtd_price_chg")

        # ── Delivery value (₹ Cr) — total delivered ₹ over the period
        dv1w = _sum(7,  "deliv_value_cr")
        dv3m = _sum(90, "deliv_value_cr")

        # ── 100-day linear trend of delivery %
        trend_slope = _slope(grp["wtd_deliv_per"])

        # ── Guard: need valid 1W data to compute any signal ─────────────────────
        if pd.isna(d1w):
            continue

        # ── Delivery momentum: how much has recent delivery shifted vs baseline
        baseline = d3m if (not pd.isna(d3m) and d3m > 0) else d1m
        if pd.isna(baseline) or baseline == 0:
            continue
        deliv_momentum = (d1w - baseline) / baseline * 100  # % change

        # ── Delivery acceleration: is it speeding up recently?
        deliv_accel = (d1w - d2w) if (not pd.isna(d1w) and not pd.isna(d2w)) else 0.0

        # ── Signal classification — the four quadrants ────────────────────────
        p_up   = (not pd.isna(p1w)) and p1w >  0.5   # price rising
        p_down = (not pd.isna(p1w)) and p1w < -0.5   # price falling
        d_up   = deliv_momentum >  8    # delivery meaningfully rising (>8%)
        d_down = deliv_momentum < -8    # delivery meaningfully falling (<-8%)

        if p_down and d_up:
            signal = "🔥 Secret Accumulation"
            action = "STRONG BUY — Institutions loading while retail panics"
        elif p_up and d_up:
            signal = "✅ Confirmed Accumulation"
            action = "BUY — Smart money buying on strength, momentum confirmed"
        elif p_up and d_down:
            signal = "⚠️ Distribution Trap"
            action = "EXIT / AVOID — Institutions selling into retail rally"
        elif p_down and d_down:
            signal = "❌ Active Selling"
            action = "AVOID — Broad institutional exit, no floor visible yet"
        elif d_up:
            signal = "👀 Early Accumulation"
            action = "WATCH — Delivery rising, price not confirmed yet"
        elif d_down:
            signal = "📉 Weakening"
            action = "REDUCE — Delivery conviction fading"
        else:
            signal = "⚖️ Neutral"
            action = "HOLD — No clear institutional directional bias"

        # ── Investment horizon ────────────────────────────────────────────────
        short_term = deliv_accel > 1 and (not pd.isna(d1w)) and (not pd.isna(d2w)) and d1w > d2w
        long_term  = trend_slope > 0.05 and (not pd.isna(d1m)) and (not pd.isna(d3m)) and d1m > d3m

        if short_term and long_term:
            horizon = "Short + Long Term"
        elif long_term:
            horizon = "Long Term"
        elif short_term:
            horizon = "Short Term"
        else:
            horizon = "—"

        # ── Trading timeframe coverage ────────────────────────────────────────
        # Grounded in:
        #   Weinstein (4-stage volume/price model — Stage 1→2 breakout criteria)
        #   Elder     (Triple Screen — weekly trend + daily impulse confirmation)
        #   Murphy    (Sector rotation leadership phases, typically 4–12 weeks)
        #   Pring     (KST multi-period Rate-of-Change: daily/weekly/monthly alignment)
        #   O'Neil    (CANSLIM "I" — ≥40% volume surge signals institutional entry)
        #
        # BTST (1-2 days) is NOT computed here — sector-level 7-day windows are too
        # coarse. Use the Signals page (single-day delivery spikes) for BTST setups.

        _buy_signals = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation",
                        "👀 Early Accumulation"}
        cov = []

        if signal in _buy_signals:
            # Swing (3–15 days)
            # Weinstein: volume 2-3× avg at Stage 1→2 breakout.
            # Elder: short-term impulse (Force Index > 0 on daily screen) aligns with
            #        weekly trend already up.
            # Condition: delivery acceleration this week + momentum shift > 10%
            swing_buy = (short_term and deliv_momentum > 10)
            if swing_buy:
                cov.append("Swing")

            # Positional (1–2 months)
            # Weinstein Stage 2: price above rising 30-week SMA, sustained volume.
            # Elder: weekly MACD above zero, medium-term Force Index positive.
            # Pring KST weekly: ROC alignment across 10/13/15/20-day periods.
            # Condition: 1-month delivery above 3-month baseline + modest 100-day upslope
            #            + delivery momentum still meaningful (>5%)
            positional_buy = (
                not pd.isna(d1m) and not pd.isna(d3m)
                and d1m > d3m
                and trend_slope > 0.03
                and deliv_momentum > 5
            )
            if positional_buy:
                cov.append("Positional")

            # Mid Term (3–4 months)
            # Murphy: confirmed sector leadership phase (4–12 weeks per phase).
            # Weinstein: Stage 2 mature — 30-week SMA steeply sloped upward.
            # Pring long-term KST: 9/12/18/24-month ROC all aligned.
            # O'Neil: dominant institutional sponsorship, multi-month accumulation.
            # Condition: steep 100-day slope + 1-month delivery still above 3-month
            #            + momentum confirms ongoing conviction (>10%)
            mid_term_buy = (
                trend_slope > 0.12
                and not pd.isna(d1m) and not pd.isna(d3m)
                and d1m > d3m
                and deliv_momentum > 10
            )
            if mid_term_buy:
                cov.append("Mid Term")

        else:
            # Avoid/exit signals — coverage = how long the weakness is likely to persist
            # Same framework applied in reverse (delivery contracting, slope falling)
            weak_accel = (not pd.isna(d1w) and not pd.isna(d2w) and d1w < d2w)

            # Swing exit: fast delivery deterioration this week
            if weak_accel and deliv_momentum < -10:
                cov.append("Swing")

            # Positional exit: 1-month delivery below 3-month + negative 100-day slope
            if (not pd.isna(d1m) and not pd.isna(d3m)
                    and d1m < d3m and trend_slope < -0.03):
                cov.append("Positional")

            # Mid Term exit: steep downward slope + sustained multi-month weakness
            if (trend_slope < -0.12
                    and not pd.isna(d1m) and not pd.isna(d3m)
                    and d1m < d3m and deliv_momentum < -10):
                cov.append("Mid Term")

        coverage = " + ".join(cov) if cov else "—"

        # ── Accumulation score (0–100) ────────────────────────────────────────
        # Components: momentum (40%) + trend slope (30%) + acceleration (20%)
        # + bonus/penalty for signal type (10%)
        mom_score   = min(max(deliv_momentum, -60), 60) / 60 * 40   # –40 to +40
        slope_score = min(max(trend_slope, -1.5), 1.5) / 1.5 * 30  # –30 to +30
        accel_score = min(max(deliv_accel, -5), 5) / 5 * 20        # –20 to +20
        bonus = (
            15  if "Secret"      in signal else
            10  if "Confirmed"    in signal else
            5   if "Early"        in signal else
            -15 if "Distribution" in signal else
            -20 if "Active"       in signal else
            0
        )
        raw_score = mom_score + slope_score + accel_score + bonus
        accum_score = round(max(0.0, min(100.0, raw_score + 50)), 1)

        records.append({
            "sector":            sector,
            "signal":            signal,
            "action":            action,
            "horizon":           horizon,
            "coverage":          coverage,
            "accum_score":       accum_score,
            "deliv_momentum":    round(deliv_momentum, 1) if not pd.isna(deliv_momentum) else None,
            "trend_slope":       round(trend_slope,    3),
            "deliv_1w":          round(d1w, 1) if not pd.isna(d1w) else None,
            "deliv_1m":          round(d1m, 1) if not pd.isna(d1m) else None,
            "deliv_3m":          round(d3m, 1) if not pd.isna(d3m) else None,
            "price_1w":          round(p1w, 2) if not pd.isna(p1w) else None,
            "price_1m":          round(p1m, 2) if not pd.isna(p1m) else None,
            "price_3m":          round(p3m, 2) if not pd.isna(p3m) else None,
            "deliv_val_1w_cr":   round(dv1w, 1) if not pd.isna(dv1w) else None,
            "deliv_val_3m_cr":   round(dv3m, 1) if not pd.isna(dv3m) else None,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # Normalize accum_score across sectors so scores spread 0–100
    result["accum_score"] = _normalize_0_100(result["accum_score"]).round(1)
    return result.sort_values("accum_score", ascending=False).reset_index(drop=True)


def get_sector_stocks_rotation(
    sector: str,
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Per-stock rotation metrics for a sector over the last `lookback_days` trading days.

    Same double-weighting methodology as sector-level:
    - Within a day: turnover-weighted delivery %
    - Across days: weighted by each day's turnover

    Returns one row per stock, sorted by delivery value (₹ Cr) descending
    — highest ₹ delivered = highest institutional conviction.

    Columns: symbol, company_name, industry, wtd_deliv_per, deliv_value_cr,
             turnover_cr, price_chg_pct
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    start = as_of_date - timedelta(days=lookback_days)
    sql = """
        SELECT
            b.symbol,
            s.company_name,
            s.industry,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)              AS wtd_deliv_per,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100  AS deliv_value_cr,
            SUM(b.turnover_lacs) / 100                         AS turnover_cr,
            SUM(
                CASE WHEN b.prev_close > 0
                THEN (b.close_price - b.prev_close) / b.prev_close * 100
                     * b.turnover_lacs END
            ) / NULLIF(SUM(CASE WHEN b.prev_close > 0 THEN b.turnover_lacs END), 0)
                AS price_chg_pct
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector = ?
          AND b.trade_date > ?
          AND b.trade_date <= ?
          AND b.turnover_lacs >= ?
        GROUP BY b.symbol, s.company_name, s.industry
        ORDER BY deliv_value_cr DESC
    """
    return query_dataframe(sql, [sector, start, as_of_date, min_turnover_lacs])


def get_sector_rotation_history(
    sector: str,
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    lookback_days: int = _LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Weekly delivery % trend for a single sector over lookback period."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    start = as_of_date - timedelta(days=lookback_days)
    sql = """
        SELECT
            b.trade_date,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)              AS wtd_deliv_per,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100  AS deliv_value_cr,
            SUM(b.turnover_lacs) / 100                         AS turnover_cr,
            SUM(
                CASE WHEN b.prev_close > 0
                THEN (b.close_price - b.prev_close) / b.prev_close * 100
                     * b.turnover_lacs END
            ) / NULLIF(SUM(CASE WHEN b.prev_close > 0 THEN b.turnover_lacs END), 0)
                AS wtd_price_chg
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector = ?
          AND b.trade_date > ?
          AND b.trade_date <= ?
          AND b.turnover_lacs >= ?
        GROUP BY b.trade_date
        ORDER BY b.trade_date
    """
    return query_dataframe(sql, [sector, start, as_of_date, min_turnover_lacs])
