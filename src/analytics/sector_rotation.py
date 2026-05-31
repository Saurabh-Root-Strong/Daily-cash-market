"""
Sector Rotation — Smart Money / Institutional Activity Detector.

Core insight: Retail traders square off intraday (no delivery). Institutions
and smart money take DELIVERY. So rising delivery = rising institutional
conviction, falling delivery = institutions exiting.

Signal matrix (cumulative 1W price direction × Z-Score of today's delivery):
  ┌──────────────────────────────────────────────────────────┐
  │              │  Z-Score HIGH (≥ +1σ)  │  Z-Score LOW (≤ -0.5σ) │
  │──────────────┼────────────────────────┼─────────────────────────│
  │ Price 1W UP  │ Confirmed Buy ✅       │ Distribution Trap ⚠️    │
  │ Price 1W DOWN│ Secret Accum 🔥        │ Active Selling ❌        │
  └──────────────┴────────────────────────┴─────────────────────────┘

This module delegates all delivery metric computation (DV Ratio, Z-Score,
Breadth, price returns) to get_sector_master_performance(), which has a
proven pure-history baseline (today excluded from denominator). Only the
100-day delivery % time-series (for trend slope) is fetched here.
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


def _get_nifty50_period_returns(as_of_date: date) -> dict:
    """
    Fetch Nifty50 1W / 1M / 3M returns from the index_data table.

    Uses the same calendar-day offsets as sector_aggregator so the RS
    calculation is apples-to-apples: rs = sector_return − nifty50_return.
    Returns {1w, 1m, 3m} as floats, or None when index data is absent.
    """
    df = query_dataframe(
        """
        SELECT trade_date, close_val
        FROM index_data
        WHERE index_name = 'Nifty 50'
          AND trade_date <= ?
          AND trade_date >= (? - INTERVAL 100 DAY)
        ORDER BY trade_date
        """,
        [as_of_date, as_of_date],
    )
    if df.empty:
        return {"1w": None, "2w": None, "1m": None, "3m": None}

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    latest_close = float(df.iloc[-1]["close_val"])

    def _ret(cal_days: int) -> float | None:
        cutoff = as_of_date - timedelta(days=cal_days)
        past = df[df["trade_date"] <= cutoff]
        if past.empty:
            return None
        start = float(past.iloc[-1]["close_val"])
        return round((latest_close - start) / start * 100, 2) if start > 0 else None

    return {"1w": _ret(7), "2w": _ret(14), "1m": _ret(30), "3m": _ret(90)}


def _slope(series: pd.Series) -> float:
    """Linear regression slope normalised by series mean (% change per trading day)."""
    y = series.dropna().values
    if len(y) < 8:
        return 0.0
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    mean = y.mean() if y.mean() != 0 else 1.0
    return float(slope / mean * 100)


# Shared normalizers live in analytics.base (deduped — were copied here and in
# the Sector Performance view). Aliased to keep existing call sites unchanged.
from src.analytics.base import minmax01 as _normalize, rank01 as _rank01  # noqa: E402


def get_sector_rotation(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    lookback_days: int = _LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Returns one row per sector with smart-money rotation signals.

    Columns
    -------
    sector, signal, action, horizon, coverage, accum_score,
    dv_ratio      -- today's DV / 100D daily avg (size-bias-free relative strength)
    z_score       -- std-devs above 100D mean (volatility-adjusted abnormality)
    breadth       -- fraction of stocks above own 100D daily avg DV
    trend_slope   -- 100-day linear slope of daily delivery % (direction)
    price_1w/1m/3m  -- cumulative price return over each period
    today_dv_cr   -- today's single-day delivered value (₹ Cr)
    deliv_val_1w_cr -- 1W total delivered value (₹ Cr)
    """
    from src.analytics.sector_aggregator import get_sector_master_performance

    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    # ── All correctly computed metrics from the analytics layer ────────────────
    perf = get_sector_master_performance(as_of_date, min_turnover_lacs)
    if perf.empty:
        return pd.DataFrame()

    # Actual trading days in the trailing 1-week window — the divisor for the
    # 5-day delivery ratio, instead of a hardcoded 5. On a normal week this is 5;
    # on a holiday-shortened week it is 3-4, where dividing by a fixed 5 would
    # understate the recent daily average by up to ~20% and could miss the
    # dv5d>=1.15 accumulation gate. The 100D side already divides by an exact
    # trading-day count (100) by construction, so this makes both sides honest.
    _n1w = query_dataframe(
        "SELECT COUNT(DISTINCT trade_date) AS n FROM daily_data "
        "WHERE trade_date > ? AND trade_date <= ?",
        [as_of_date - timedelta(days=7), as_of_date],
    )
    n_1w_days = int(_n1w["n"].iloc[0]) if not _n1w.empty and _n1w["n"].iloc[0] else 5

    # ── 100-day daily delivery % series — for trend slope only ────────────────
    start = as_of_date - timedelta(days=lookback_days)
    hist_sql = """
        SELECT
            s.sector,
            b.trade_date,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)             AS wtd_deliv_per,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100 AS deliv_value_cr,
            SUM(b.turnover_lacs * (b.close_price - b.prev_close)
                    / NULLIF(b.prev_close, 0) * 100)
                / NULLIF(SUM(CASE WHEN b.prev_close > 0 THEN b.turnover_lacs END), 0)
                                                              AS wtd_daily_ret_pct
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
    hist = query_dataframe(hist_sql, [start, as_of_date, min_turnover_lacs])

    slope_map    = {}
    dv1w_cr_map  = {}
    price_2w_map = {}

    for sector, grp in hist.groupby("sector"):
        grp = grp.sort_values("trade_date").reset_index(drop=True)
        slope_map[sector] = _slope(grp["wtd_deliv_per"])
        cutoff_1w = pd.Timestamp(as_of_date - timedelta(days=7))
        sub_1w    = grp[grp["trade_date"] > cutoff_1w]
        dv1w_cr_map[sector] = float(sub_1w["deliv_value_cr"].sum()) if len(sub_1w) >= 1 else float("nan")
        cutoff_2w = pd.Timestamp(as_of_date - timedelta(days=14))
        sub_2w    = grp[grp["trade_date"] > cutoff_2w]
        if "wtd_daily_ret_pct" in sub_2w.columns and len(sub_2w) >= 1:
            daily_2w = sub_2w["wtd_daily_ret_pct"].fillna(0) / 100
            price_2w_map[sector] = round(float((1 + daily_2w).prod() - 1) * 100, 2)
        else:
            price_2w_map[sector] = float("nan")

    # ── Cross-sectional z-percentile (regime-proof gate input) ──────────────────
    # Raw z is INFLATED in a trending-delivery regime (integrity audit: median
    # ~+3.6, 79% of sectors above z=1, |z|>2 for 74% — vs a healthy ~5%). So the
    # old absolute gates (z>=1 surge, z<=-0.5 weak) fired for almost everything
    # and NEVER flagged weakness (empty "Avoid"). Percentile rank across today's
    # tradable sectors always discriminates and is immune to the trend. The dv5d
    # ratio (well-behaved) stays as the absolute strength/weakness anchor.
    _tradable = perf[~perf["sector"].isin(("ETF", "Others"))]
    _z_pct_map = dict(zip(
        _tradable["sector"], _tradable["z_score"].rank(pct=True),
    ))

    # ── Build one record per sector ────────────────────────────────────────────
    records = []
    for _, row in perf.iterrows():
        sector = str(row["sector"])
        if sector in ("ETF", "Others"):
            continue
        z_pct = float(_z_pct_map.get(sector, 0.5))   # 0..1 rank of this sector's z

        dv_ratio  = float(row.get("dv_ratio",          float("nan")))
        z_score   = float(row.get("z_score",            float("nan")))
        breadth   = float(row.get("breadth",            float("nan")))
        p1w       = float(row.get("1W_price_chg_pct",   float("nan")))
        p1m       = float(row.get("1M_price_chg_pct",   float("nan")))
        p3m       = float(row.get("3M_price_chg_pct",   float("nan")))
        today_dv  = float(row.get("today_dv_cr",        float("nan")))
        deliv_1w  = float(row.get("1W_deliv_cr",        float("nan")))
        deliv_100d = float(row.get("100D_deliv_cr",     float("nan")))

        if pd.isna(dv_ratio) or pd.isna(z_score):
            continue

        # ── 5-day average delivery ratio ──────────────────────────────────────
        # dv_ratio_5d = (1W total delivery / 5 days) / (100D total delivery / 100 days)
        # This is the 5-day average relative to the sector's own daily norm.
        # Purpose: prevents single-day spikes or dips from flipping the signal category.
        # A sector that was distributing yesterday cannot show "Confirmed Accumulation"
        # today just because of one large block trade. It needs sustained flow.
        if not pd.isna(deliv_1w) and not pd.isna(deliv_100d) and deliv_100d > 0:
            dv_ratio_5d = round((deliv_1w / n_1w_days) / (deliv_100d / 100), 3)
        else:
            dv_ratio_5d = float("nan")
        dv5d = dv_ratio_5d if not pd.isna(dv_ratio_5d) else 1.0  # fallback: neutral

        trend_slope = slope_map.get(sector, 0.0)
        dv1w_cr     = dv1w_cr_map.get(sector, float("nan"))

        # ── Signal classification — two-layer evidence model ──────────────────
        #
        # Layer 1 (today's snapshot): z_score >= 1.0 = delivery value statistically above norm.
        # Layer 2 (sustained evidence): 5-day avg delivery ratio must also support the signal.
        #
        # ACCUMULATION requires BOTH today's z AND sustained 5-day flow:
        #   Confirmed:  z >= 1.0  AND  dv5d >= 1.15  (today elevated + recent week elevated)
        #   Extreme:    z >= 2.0  AND  dv5d >= 0.9   (exceptional event, background neutral+)
        #   → Prevents: sector distributing for 5 days, single spike → "Confirmed Accumulation"
        #
        # DISTRIBUTION requires BOTH today's weakness AND sustained weakness:
        #   Confirmed:  z <= -0.5 AND  dv5d <= 0.90  (today weak + recent week weak)
        #   → Prevents: sector accumulating for 5 days, quiet session → "Distribution Trap"
        #
        # WEAKENING (mild warning) stays single-day — it's a caution, not a trade signal.
        # VOLUME SPIKE stays single-day — by definition it's a one-session anomaly.
        #
        # pct_surge: delivery PERCENTAGE vs 100D avg — catches speculative intraday volume.
        # When turnover surges but delivery% falls, institutions are NOT accumulating.
        today_wtd_pct = float(row.get("today_wtd_deliv_pct",   float("nan")))
        avg_wtd_pct   = float(row.get("avg_wtd_deliv_pct_100d", float("nan")))

        # Gates use the cross-sectional z-PERCENTILE, not raw z (see note above).
        #   z_pct >= 0.50  ≈ old "z >= 1.0"   (above-median delivery abnormality)
        #   z_pct >= 0.90  ≈ old "z >= 2.0"   (extreme — top 10% of sectors today)
        #   z_pct <= 0.25  ≈ old "z <= -0.5"  (bottom quartile — now actually fires)
        d_surge = z_pct >= 0.50
        d_weak  = z_pct <= 0.25
        p_up    = (not pd.isna(p1w)) and p1w > 1.0
        p_down  = (not pd.isna(p1w)) and p1w < -1.0

        if not pd.isna(today_wtd_pct) and not pd.isna(avg_wtd_pct) and avg_wtd_pct > 0:
            pct_surge = today_wtd_pct >= avg_wtd_pct * 0.85
        else:
            pct_surge = True

        # Two-layer gate: percentile z (relative strength) AND dv5d (absolute,
        # well-behaved ratio — the genuine sustained-flow anchor, unchanged).
        d_surge_confirmed = (d_surge and dv5d >= 1.15) or (z_pct >= 0.90 and dv5d >= 0.9)
        d_weak_confirmed  = d_weak and dv5d <= 0.90

        if d_surge_confirmed and pct_surge:
            if p_down:
                signal = "🔥 Secret Accumulation"
                action = "STRONG BUY — Institutions loading while retail panics"
            elif p_up:
                signal = "✅ Confirmed Accumulation"
                action = "BUY — Smart money buying on strength, momentum confirmed"
            else:
                signal = "👀 Early Accumulation"
                action = "WATCH — Institutional flow above norm, price not confirmed yet"
        elif d_surge and not pct_surge:
            # Delivery VALUE surged but PERCENTAGE fell >15% — speculative volume, not conviction.
            # Kept as single-day check by design: Volume Spike IS a one-session anomaly.
            pct_str  = f"{today_wtd_pct:.1f}%" if not pd.isna(today_wtd_pct) else "?"
            avg_str  = f"{avg_wtd_pct:.1f}%"   if not pd.isna(avg_wtd_pct)   else "?"
            drop_pct = (avg_wtd_pct - today_wtd_pct) / avg_wtd_pct * 100 if avg_wtd_pct > 0 else 0
            signal = "📊 Volume Spike"
            action = f"CAUTION — Delivery% ({pct_str}) is {drop_pct:.0f}% below 100D avg ({avg_str}); turnover surged but conviction fell; speculative, not institutional accumulation"
        elif p_up and d_weak_confirmed:
            signal = "⚠️ Distribution Trap"
            action = "EXIT / AVOID — Institutions selling into retail rally"
        elif p_down and d_weak_confirmed:
            signal = "❌ Active Selling"
            action = "AVOID — Broad institutional exit, no floor visible yet"
        elif d_weak and dv5d < 1.05:
            # Relatively weakest (bottom-quartile z-rank) AND recent flow not above
            # its own norm. The dv5d<1.05 guard prevents flagging a sector that
            # happens to rank low on z but still has genuinely strong sustained
            # delivery — mild caution only, not a reversal signal.
            signal = "📉 Weakening"
            action = "REDUCE — Relatively weakest flow, conviction fading"
        else:
            signal = "⚖️ Neutral"
            action = "HOLD — Flow within normal range, no clear directional bias"

        # ── Investment horizon ─────────────────────────────────────────────────
        # Short term: top-decile delivery abnormality (z-percentile) + breadth
        short_term = z_pct >= 0.90 and (not pd.isna(breadth)) and breadth >= 0.5
        # Long term: trend slope positive + DV Ratio above average + broad participation
        long_term  = (trend_slope > 0.05 and dv_ratio > 1.1
                      and (not pd.isna(breadth)) and breadth >= 0.4)

        if short_term and long_term:
            horizon = "Short + Long Term"
        elif long_term:
            horizon = "Long Term"
        elif short_term:
            horizon = "Short Term"
        else:
            horizon = "—"

        # ── Trading timeframe coverage ─────────────────────────────────────────
        _buy_signals = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation", "👀 Early Accumulation"}
        cov = []

        if signal in _buy_signals:
            # Swing (3–15 days): top-decile delivery abnormality + broad participation
            if z_pct >= 0.90 and (not pd.isna(breadth)) and breadth >= 0.5:
                cov.append("Swing (3–15 days)")
            # Positional (1–2 months): DV Ratio elevated + positive slope + breadth
            if (dv_ratio > 1.2 and trend_slope > 0.03
                    and (not pd.isna(breadth)) and breadth >= 0.4):
                cov.append("Positional (4–8 weeks)")
            # Mid Term (3–4 months): sustained high slope + strong DV Ratio + deep breadth
            if (trend_slope > 0.12 and dv_ratio > 1.3
                    and (not pd.isna(breadth)) and breadth >= 0.5):
                cov.append("Mid Term (3–4 months)")
        else:
            # Avoid signals — how long the weakness may persist
            if z_pct <= 0.10 and (not pd.isna(breadth)) and breadth <= 0.3:
                cov.append("Swing (3–15 days)")
            if dv_ratio < 0.8 and trend_slope < -0.03:
                cov.append("Positional (4–8 weeks)")
            if trend_slope < -0.12 and dv_ratio < 0.7:
                cov.append("Mid Term (3–4 months)")

        coverage = " + ".join(cov) if cov else "—"

        records.append({
            "sector":               sector,
            "signal":               signal,
            "action":               action,
            "horizon":              horizon,
            "coverage":             coverage,
            "dv_ratio":             round(dv_ratio,  3),
            "dv_ratio_5d":          round(dv_ratio_5d, 3) if not pd.isna(dv_ratio_5d) else None,
            "z_score":              round(z_score,   2),
            "breadth":              round(breadth,   3) if not pd.isna(breadth)  else None,
            "trend_slope":          round(trend_slope, 3),
            "price_1w":             round(p1w, 2)  if not pd.isna(p1w)  else None,
            "price_2w":             price_2w_map.get(sector, float("nan")),
            "price_1m":             round(p1m, 2)  if not pd.isna(p1m)  else None,
            "price_3m":             round(p3m, 2)  if not pd.isna(p3m)  else None,
            "today_dv_cr":          round(today_dv, 1) if not pd.isna(today_dv) else None,
            "deliv_val_1w_cr":      round(dv1w_cr,  1) if not pd.isna(dv1w_cr)  else None,
            "today_wtd_deliv_pct":  round(today_wtd_pct, 1) if not pd.isna(today_wtd_pct) else None,
            "avg_wtd_deliv_pct_100d": round(avg_wtd_pct, 1) if not pd.isna(avg_wtd_pct)  else None,
            "_dv5d":                dv5d,
            "_dv":                  dv_ratio,
            "_z":                   z_score,
            "_br":                  breadth  if not pd.isna(breadth)  else 0.5,
            "_pm":                  p1w      if not pd.isna(p1w)      else 0.0,
            "_slope":               trend_slope,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return result

    # ── Relative Strength vs Nifty50 ─────────────────────────────────────────
    # rs_Xp = sector_price_return_Xp − nifty50_price_return_Xp
    # Positive = outperforming benchmark; Negative = underperforming.
    # Uses same calendar-day windows as sector_aggregator (7 / 30 / 90 days).
    n50 = _get_nifty50_period_returns(as_of_date)
    for period, col_price, col_rs in [
        ("1w", "price_1w", "rs_1w"),
        ("2w", "price_2w", "rs_2w"),
        ("1m", "price_1m", "rs_1m"),
        ("3m", "price_3m", "rs_3m"),
    ]:
        n_ret = n50[period]
        if n_ret is not None and col_price in result.columns:
            result[col_rs] = (result[col_price] - n_ret).round(2)
        else:
            result[col_rs] = float("nan")
    result["nifty_1w"] = n50["1w"]
    result["nifty_2w"] = n50["2w"]
    result["nifty_1m"] = n50["1m"]
    result["nifty_3m"] = n50["3m"]

    # ── Score: cross-sectional rank blend, relative-strength-aware ──────────────
    # Rebuilt from a 348-day factor study (scripts/sector_score_compare.py):
    #   • RS vs Nifty is the single strongest predictor — it was computed here but
    #     previously DISCARDED from the score. Now the top weight (30%).
    #   • Rank (not min-max) normalization → outlier-robust + comparable across days.
    #   • Trend slope dropped — it had ~0 information coefficient at every horizon.
    # New IC beat the old min-max score at 5/10/20d (e.g. 20d: +0.165 vs +0.140).
    #
    # 30% RS vs Nifty (2W) + 25% 5-day avg DV + 15% today DV +
    # 15% Breadth + 15% Z-Score
    # RS falls back to raw 1W price momentum when index data is unavailable.
    rs_col = result["rs_2w"] if result["rs_2w"].notna().any() else result["_pm"]
    result["accum_score"] = (
        _rank01(rs_col)            * 30 +
        _rank01(result["_dv5d"])   * 25 +
        _rank01(result["_dv"])     * 15 +
        _rank01(result["_br"])     * 15 +
        _rank01(result["_z"])      * 15
    ).round(1)

    result = result.drop(columns=[c for c in result.columns if c.startswith("_")])
    return result.sort_values("accum_score", ascending=False).reset_index(drop=True)


def get_sector_stocks_rotation(
    sector: str,
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Per-stock rotation metrics for a sector over the last `lookback_days` calendar days.

    Returns avg_deliv_per_100d: each stock's own 100D average delivery %
    so conviction can compare against own history, not sector-relative percentiles.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    start = as_of_date - timedelta(days=lookback_days)

    # Pure-history 100D cutoff — same offset logic as sector_aggregator
    cutoff_row = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data "
        "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1 OFFSET 100",
        [as_of_date],
    )
    cutoff_100d = (
        pd.to_datetime(cutoff_row["trade_date"].iloc[0]).date()
        if not cutoff_row.empty
        else as_of_date - timedelta(days=200)
    )

    sql = """
        WITH recent AS (
            SELECT
                b.symbol,
                s.company_name,
                s.industry,
                ARGMAX(b.close_price, b.trade_date)                AS ltp,
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
        ),
        hist_avg AS (
            -- Turnover-WEIGHTED 100D delivery % so the baseline is in the same
            -- units as `recent.wtd_deliv_per` (also turnover-weighted). Previously
            -- a simple AVG(deliv_per), which made the conviction comparison
            -- apples-to-oranges and could flip a stock's buy/weak label.
            SELECT b.symbol,
                   SUM(b.deliv_per * b.turnover_lacs)
                       / NULLIF(SUM(b.turnover_lacs), 0) AS avg_deliv_per_100d
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND s.sector = ?
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date < ?
            GROUP BY b.symbol
        )
        SELECT r.*, h.avg_deliv_per_100d
        FROM recent r
        LEFT JOIN hist_avg h ON r.symbol = h.symbol
        ORDER BY r.deliv_value_cr DESC
    """
    return query_dataframe(
        sql,
        [sector, start, as_of_date, min_turnover_lacs,
         sector, min_turnover_lacs, cutoff_100d, as_of_date],
    )


def get_sector_rotation_history(
    sector: str,
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    lookback_days: int = _LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Daily delivery % + delivery value trend for a single sector over lookback period."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    start = as_of_date - timedelta(days=lookback_days)
    sql = """
        SELECT
            b.trade_date,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)             AS wtd_deliv_per,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100 AS deliv_value_cr,
            SUM(b.turnover_lacs) / 100                        AS turnover_cr,
            AVG(
                CASE WHEN b.prev_close > 0
                THEN (b.close_price - b.prev_close) / b.prev_close * 100
                END
            )                                                 AS avg_price_chg
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


def get_sector_rotation_timeframe(
    as_of_date: date,
    window_trading_days: int = 5,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Multi-period sector rotation clock — where is money flowing in vs out?

    Compares two consecutive N-trading-day windows:
      current  = last N trading days up to as_of_date
      prior    = N trading days before that

    Returns one row per sector with:
      phase             — Leading / Improving / Weakening / Lagging / Neutral
      flow_signal       — human-readable label
      delivery_slope    — linear slope of daily wtd delivery % (positive = rising conviction)
      slope_z           — cross-sectional z-score of delivery_slope across all sectors
      cum_price_ret_pct — cumulative turnover-weighted price return over current window (%)
      deliv_value_cr    — total delivery value ₹ Cr (current window)
      deliv_chg_pct     — delivery value change % vs prior equal-length window
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    lookback_cal = max(window_trading_days * 3 + 45, 200)
    start_date   = as_of_date - timedelta(days=lookback_cal)

    hist_sql = """
        SELECT
            s.sector,
            b.trade_date,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)              AS wtd_deliv_pct,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100  AS deliv_value_cr,
            SUM(b.turnover_lacs) / 100                         AS turnover_cr,
            SUM(b.turnover_lacs * (b.close_price - b.prev_close) / NULLIF(b.prev_close, 0) * 100)
                / NULLIF(SUM(CASE WHEN b.prev_close > 0 THEN b.turnover_lacs END), 0)
                                                               AS wtd_daily_ret_pct
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector NOT IN ('ETF', 'Others')
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date <= ?
        GROUP BY s.sector, b.trade_date
        ORDER BY s.sector, b.trade_date
    """
    hist = query_dataframe(hist_sql, [min_turnover_lacs, start_date, as_of_date])
    if hist.empty:
        return pd.DataFrame()

    hist["trade_date"] = pd.to_datetime(hist["trade_date"]).dt.date
    all_dates          = sorted(hist["trade_date"].unique())

    if len(all_dates) < window_trading_days + 3:
        return pd.DataFrame()

    curr_dates = set(all_dates[-window_trading_days:])
    prev_dates = (
        set(all_dates[-(window_trading_days * 2):-window_trading_days])
        if len(all_dates) >= window_trading_days * 2
        else set()
    )

    # Nifty50 return for this exact window — used as quadrant center so phase
    # classification is market-relative (a sector beating Nifty50 = Leading/Weakening,
    # a sector lagging Nifty50 = Improving/Lagging). Without this, during bull markets
    # all sectors have positive price returns and Improving/Lagging become empty.
    curr_dates_sorted = sorted(curr_dates)
    nifty_window_ret  = get_nifty50_custom_return(curr_dates_sorted[0], curr_dates_sorted[-1])
    nifty_threshold   = nifty_window_ret if nifty_window_ret is not None else 0.0

    records = []
    for sector, grp in hist.groupby("sector"):
        grp  = grp.sort_values("trade_date")
        curr = grp[grp["trade_date"].isin(curr_dates)].reset_index(drop=True)
        prev = grp[grp["trade_date"].isin(prev_dates)].reset_index(drop=True)

        if len(curr) < 3:
            continue

        # Delivery % linear slope — direction of institutional conviction
        y = curr["wtd_deliv_pct"].ffill().bfill().values
        x = np.arange(len(y), dtype=float)
        delivery_slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 3 else 0.0

        # Cumulative price return (compound daily turnover-weighted returns)
        daily_rets    = curr["wtd_daily_ret_pct"].fillna(0) / 100
        cum_price_pct = float((1 + daily_rets).prod() - 1) * 100

        # Price-delivery daily correlation — key Weakening filter.
        # Genuine distribution = price rising while delivery falls (corr < 0).
        # corr > 0 means both moving together → normal momentum, NOT distribution.
        corr_raw = curr["wtd_daily_ret_pct"].corr(curr["wtd_deliv_pct"])
        price_deliv_corr = float(corr_raw) if not pd.isna(corr_raw) else 0.0

        # Delivery value: current vs prior period
        curr_dv = float(curr["deliv_value_cr"].sum())
        if not prev.empty:
            prev_dv     = float(prev["deliv_value_cr"].sum())
            deliv_chg   = (curr_dv - prev_dv) / max(abs(prev_dv), 0.1) * 100 if prev_dv != 0 else None
        else:
            prev_dv   = None
            deliv_chg = None

        records.append({
            "sector":              sector,
            "delivery_slope":      round(delivery_slope, 4),
            "cum_price_ret_pct":   round(cum_price_pct, 2),
            "price_deliv_corr":    round(price_deliv_corr, 3),
            "deliv_value_cr":      round(curr_dv, 1),
            "deliv_value_prev_cr": round(prev_dv, 1) if prev_dv is not None else None,
            "deliv_chg_pct":       round(deliv_chg, 1) if deliv_chg is not None else None,
            "turnover_cr":         round(float(curr["turnover_cr"].sum()), 1),
            "avg_deliv_pct":       round(float(curr["wtd_deliv_pct"].mean()), 1),
            "num_days":            int(len(curr)),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Cross-sectional z-score of delivery slope
    s_mean = df["delivery_slope"].mean()
    s_std  = df["delivery_slope"].std()
    df["slope_z"] = ((df["delivery_slope"] - s_mean) / max(s_std, 1e-9)).round(2)

    def _phase_and_confidence(row) -> tuple:
        sz   = row["slope_z"]
        pr   = row["cum_price_ret_pct"] - nifty_threshold
        corr = row["price_deliv_corr"]

        if sz > 0.25 and pr > 0.5:
            # Leading: delivery rising + price beating market
            # corr > 0 (price and delivery co-moving) increases confidence
            conf = min(abs(sz) / 1.5, 1.0) * (1.0 + max(corr, 0.0) * 0.5)
            return "Leading", round(min(conf, 1.0), 2)

        elif sz > 0.25 and pr < -0.5:
            # Improving: delivery rising but price lagging market
            conf = min(abs(sz) / 1.5, 1.0)
            return "Improving", round(min(conf, 1.0), 2)

        elif sz < -0.25 and pr > 0.5:
            # Weakening candidate: delivery falling + price beating market.
            # ONLY confirmed if price and delivery are anti-correlated
            # (price rising as delivery falls = genuine sell-into-strength distribution).
            # If corr >= -0.15, delivery is just normalising from a peak — not distribution.
            if corr < -0.15:
                conf = min(abs(sz) / 1.5, 1.0) * (1.0 + abs(min(corr, 0.0)) * 0.5)
                return "Weakening", round(min(conf, 1.0), 2)
            else:
                return "Neutral", 0.3

        elif sz < -0.25 and pr < -0.5:
            # Lagging: delivery falling + price below market — confirmed exit
            conf = min(abs(sz) / 1.5, 1.0)
            return "Lagging", round(min(conf, 1.0), 2)

        else:
            return "Neutral", 0.5

    df[["phase", "signal_confidence"]] = df.apply(
        lambda r: pd.Series(_phase_and_confidence(r)), axis=1
    )
    df["flow_signal"] = df["phase"].map({
        "Leading":   "💰 MONEY ENTERING",
        "Improving": "🔍 CONTRARIAN INFLOW",
        "Weakening": "⚠️ TOPPING",
        "Lagging":   "📤 MONEY EXITING",
        "Neutral":   "⚖️ SIDEWAYS",
    })
    df["nifty_return"] = nifty_window_ret  # None if index_data unavailable

    return df.sort_values("slope_z", ascending=False).reset_index(drop=True)


def get_rotation_clock_backtest(
    as_of_date: date,
    window_trading_days: int = 22,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Signal validation: what did the rotation clock signal N trading days ago
    and how did those sectors actually perform since then?

    signal_date = the trading day exactly window_trading_days before as_of_date
    signals     = rotation phases computed AS OF signal_date (no look-ahead)
    forward_ret = cumulative turnover-weighted sector return from signal_date → as_of_date

    Returns signals enriched with:
      signal_date (date), forward_ret_pct (%), signal_correct (bool|None)

    signal_correct:
      Leading / Improving (inflow) → correct if forward_ret > 0%
      Weakening / Lagging (outflow) → correct if forward_ret < 0%
      Neutral → None (no prediction made)
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    # Find the Nth trading day before as_of_date
    dates_df = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data "
        "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT ?",
        [as_of_date, window_trading_days + 1],
    )
    if len(dates_df) < window_trading_days:
        return pd.DataFrame()

    signal_date = pd.to_datetime(dates_df["trade_date"].iloc[window_trading_days - 1]).date()

    # Rotation signals AS OF signal_date (pure past — no future data used)
    signals = get_sector_rotation_timeframe(signal_date, window_trading_days, min_turnover_lacs)
    if signals.empty:
        return pd.DataFrame()

    # Actual forward returns: signal_date → as_of_date
    fwd_sql = """
        SELECT
            s.sector,
            b.trade_date,
            SUM(b.turnover_lacs * (b.close_price - b.prev_close)
                    / NULLIF(b.prev_close, 0) * 100)
                / NULLIF(SUM(CASE WHEN b.prev_close > 0 THEN b.turnover_lacs END), 0)
                                                        AS wtd_daily_ret_pct
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector NOT IN ('ETF', 'Others')
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date >  ?
          AND b.trade_date <= ?
        GROUP BY s.sector, b.trade_date
        ORDER BY s.sector, b.trade_date
    """
    fwd = query_dataframe(fwd_sql, [min_turnover_lacs, signal_date, as_of_date])
    if fwd.empty:
        return pd.DataFrame()

    fwd["trade_date"] = pd.to_datetime(fwd["trade_date"]).dt.date

    forward_rets: dict[str, float] = {}
    for sector, grp in fwd.groupby("sector"):
        daily = grp.sort_values("trade_date")["wtd_daily_ret_pct"].fillna(0) / 100
        forward_rets[sector] = float((1 + daily).prod() - 1) * 100

    # Nifty50 forward return for the same period — market-relative validation.
    # In bull markets, even "distributing" sectors go up in absolute terms.
    # The correct question is: did the sector OUTPERFORM or UNDERPERFORM Nifty50?
    nifty_fwd = get_nifty50_custom_return(signal_date, as_of_date)

    signals = signals.copy()
    signals["signal_date"]          = signal_date
    signals["forward_ret_pct"]      = signals["sector"].map(forward_rets)
    signals["forward_nifty_ret"]    = nifty_fwd
    # Excess return vs Nifty50 over the forward period
    if nifty_fwd is not None:
        signals["forward_vs_nifty"] = signals["forward_ret_pct"] - nifty_fwd
    else:
        signals["forward_vs_nifty"] = signals["forward_ret_pct"]  # fallback to absolute

    def _correct(row) -> Optional[bool]:
        phase = row["phase"]
        # Use market-relative return: sector correct if it beat Nifty50 (inflow)
        # or underperformed Nifty50 (outflow). Falls back to absolute if no index data.
        fwd_r = row.get("forward_vs_nifty") if nifty_fwd is not None else row.get("forward_ret_pct")
        if fwd_r is None or pd.isna(fwd_r) or phase == "Neutral":
            return None
        if phase in ("Leading", "Improving"):
            return bool(fwd_r > 0)   # sector beat the market
        return bool(fwd_r < 0)       # Weakening/Lagging: sector underperformed market

    signals["signal_correct"] = signals.apply(_correct, axis=1)

    # Sort: phase rank first, then forward return descending within each phase
    _rank = {"Leading": 0, "Improving": 1, "Neutral": 2, "Weakening": 3, "Lagging": 4}
    signals["_pr"] = signals["phase"].map(_rank).fillna(5)
    return (
        signals.sort_values(["_pr", "forward_ret_pct"], ascending=[True, False])
        .drop(columns="_pr")
        .reset_index(drop=True)
    )


def get_sector_rotation_custom_range(
    from_date: date,
    to_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Rotation clock for an arbitrary date range [from_date, to_date].

    Same metrics as get_sector_rotation_timeframe but uses explicit calendar dates.
    Prior period for delivery change% = equal-length calendar window before from_date.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    period_days = max((to_date - from_date).days, 1)
    prior_from  = from_date - timedelta(days=period_days)

    hist_sql = """
        SELECT
            s.sector,
            b.trade_date,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)              AS wtd_deliv_pct,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100  AS deliv_value_cr,
            SUM(b.turnover_lacs) / 100                         AS turnover_cr,
            SUM(b.turnover_lacs * (b.close_price - b.prev_close)
                    / NULLIF(b.prev_close, 0) * 100)
                / NULLIF(SUM(CASE WHEN b.prev_close > 0
                             THEN b.turnover_lacs END), 0)     AS wtd_daily_ret_pct
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector NOT IN ('ETF', 'Others')
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date >= ?
          AND b.trade_date <= ?
        GROUP BY s.sector, b.trade_date
        ORDER BY s.sector, b.trade_date
    """
    hist = query_dataframe(hist_sql, [min_turnover_lacs, prior_from, to_date])
    if hist.empty:
        return pd.DataFrame()

    hist["trade_date"] = pd.to_datetime(hist["trade_date"]).dt.date
    curr_hist = hist[(hist["trade_date"] >= from_date) & (hist["trade_date"] <= to_date)]
    prev_hist = hist[(hist["trade_date"] >= prior_from) & (hist["trade_date"] < from_date)]

    records = []
    for sector, curr in curr_hist.groupby("sector"):
        curr = curr.sort_values("trade_date").reset_index(drop=True)
        prev = prev_hist[prev_hist["sector"] == sector].sort_values("trade_date").reset_index(drop=True)

        if len(curr) < 2:
            continue

        y = curr["wtd_deliv_pct"].ffill().bfill().values
        x = np.arange(len(y), dtype=float)
        delivery_slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 3 else 0.0

        daily_rets    = curr["wtd_daily_ret_pct"].fillna(0) / 100
        cum_price_pct = float((1 + daily_rets).prod() - 1) * 100

        corr_raw = curr["wtd_daily_ret_pct"].corr(curr["wtd_deliv_pct"])
        price_deliv_corr = float(corr_raw) if not pd.isna(corr_raw) else 0.0

        curr_dv = float(curr["deliv_value_cr"].sum())
        prev_dv = float(prev["deliv_value_cr"].sum()) if not prev.empty else None
        deliv_chg = (curr_dv - prev_dv) / max(abs(prev_dv), 0.1) * 100 if prev_dv is not None else None

        records.append({
            "sector":              sector,
            "delivery_slope":      round(delivery_slope, 4),
            "cum_price_ret_pct":   round(cum_price_pct, 2),
            "price_deliv_corr":    round(price_deliv_corr, 3),
            "deliv_value_cr":      round(curr_dv, 1),
            "deliv_value_prev_cr": round(prev_dv, 1) if prev_dv is not None else None,
            "deliv_chg_pct":       round(deliv_chg, 1) if deliv_chg is not None else None,
            "turnover_cr":         round(float(curr["turnover_cr"].sum()), 1),
            "avg_deliv_pct":       round(float(curr["wtd_deliv_pct"].mean()), 1),
            "num_days":            int(len(curr)),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    s_mean = df["delivery_slope"].mean()
    s_std  = df["delivery_slope"].std()
    df["slope_z"] = ((df["delivery_slope"] - s_mean) / max(s_std, 1e-9)).round(2)

    nifty_ret_custom  = get_nifty50_custom_return(from_date, to_date)
    nifty_thr_custom  = nifty_ret_custom if nifty_ret_custom is not None else 0.0

    def _phase_and_confidence_custom(row) -> tuple:
        sz   = row["slope_z"]
        pr   = row["cum_price_ret_pct"] - nifty_thr_custom
        corr = row["price_deliv_corr"]
        if sz > 0.25 and pr > 0.5:
            conf = min(abs(sz) / 1.5, 1.0) * (1.0 + max(corr, 0.0) * 0.5)
            return "Leading", round(min(conf, 1.0), 2)
        elif sz > 0.25 and pr < -0.5:
            return "Improving", round(min(abs(sz) / 1.5, 1.0), 2)
        elif sz < -0.25 and pr > 0.5:
            if corr < -0.15:
                conf = min(abs(sz) / 1.5, 1.0) * (1.0 + abs(min(corr, 0.0)) * 0.5)
                return "Weakening", round(min(conf, 1.0), 2)
            else:
                return "Neutral", 0.3
        elif sz < -0.25 and pr < -0.5:
            return "Lagging", round(min(abs(sz) / 1.5, 1.0), 2)
        else:
            return "Neutral", 0.5

    df[["phase", "signal_confidence"]] = df.apply(
        lambda r: pd.Series(_phase_and_confidence_custom(r)), axis=1
    )
    df["flow_signal"] = df["phase"].map({
        "Leading":   "💰 MONEY ENTERING",
        "Improving": "🔍 CONTRARIAN INFLOW",
        "Weakening": "⚠️ TOPPING",
        "Lagging":   "📤 MONEY EXITING",
        "Neutral":   "⚖️ SIDEWAYS",
    })
    df["nifty_return"] = nifty_ret_custom

    return df.sort_values("slope_z", ascending=False).reset_index(drop=True)


def get_nifty50_custom_return(from_date: date, to_date: date) -> Optional[float]:
    """Nifty50 return from from_date to to_date using index_data table."""
    df = query_dataframe(
        """
        SELECT trade_date, close_val
        FROM index_data
        WHERE index_name = 'Nifty 50'
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date
        """,
        [from_date - timedelta(days=7), to_date],
    )
    if df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    start_rows = df[df["trade_date"] <= from_date]
    end_rows   = df[df["trade_date"] <= to_date]
    if start_rows.empty or end_rows.empty:
        return None
    start = float(start_rows.iloc[-1]["close_val"])
    end   = float(end_rows.iloc[-1]["close_val"])
    return round((end - start) / start * 100, 2) if start > 0 else None


def get_sector_rs_custom_range(
    from_date: date,
    to_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Sector RS vs Nifty50 over an arbitrary date range.

    Returns: sector, cum_price_ret_pct (sector return), rs_custom (excess vs Nifty50), nifty_custom.
    Designed to be merged with get_sector_rotation() output on 'sector' for scatter charts.
    """
    rotation_df = get_sector_rotation_custom_range(from_date, to_date, min_turnover_lacs)
    if rotation_df.empty:
        return pd.DataFrame()
    nifty_ret = get_nifty50_custom_return(from_date, to_date)
    result = rotation_df[["sector", "cum_price_ret_pct"]].copy()
    if nifty_ret is not None:
        result["rs_custom"] = (result["cum_price_ret_pct"] - nifty_ret).round(2)
    else:
        result["rs_custom"] = float("nan")
    result["nifty_custom"] = nifty_ret
    return result


def get_sector_stocks_custom_range(
    sector: str,
    from_date: date,
    to_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Per-stock performance for a sector over a custom date range.

    Returns: symbol, company_name, industry, price_start, price_end,
             period_ret_pct, wtd_deliv_per, deliv_value_cr, turnover_cr
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    sql = """
        SELECT
            b.symbol,
            s.company_name,
            s.industry,
            ARGMIN(b.prev_close,   b.trade_date)                AS price_start,
            ARGMAX(b.close_price,  b.trade_date)                AS price_end,
            (ARGMAX(b.close_price, b.trade_date)
                - ARGMIN(b.prev_close, b.trade_date))
                / NULLIF(ARGMIN(b.prev_close, b.trade_date), 0) * 100
                                                                AS period_ret_pct,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)               AS wtd_deliv_per,
            SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100   AS deliv_value_cr,
            SUM(b.turnover_lacs) / 100                          AS turnover_cr,
            COUNT(DISTINCT b.trade_date)                        AS trading_days
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector = ?
          AND b.trade_date >= ?
          AND b.trade_date <= ?
          AND b.turnover_lacs >= ?
        GROUP BY b.symbol, s.company_name, s.industry
        HAVING COUNT(DISTINCT b.trade_date) >= 2
        ORDER BY deliv_value_cr DESC
    """
    return query_dataframe(sql, [sector, from_date, to_date, min_turnover_lacs])
