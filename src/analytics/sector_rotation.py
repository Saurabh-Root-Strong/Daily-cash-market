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

# ── Sector tradability gate ───────────────────────────────────────────────────
# A "sector" is only a meaningful rotation signal if it's a diversified, liquid
# basket. Many sector_master buckets are thin (e.g. "Oil & Gas" = 7 small-caps,
# the real majors sit in "Energy") or single-name dominated (one stock = 80% of
# turnover). Those produce single-stock momentum dressed up as sector rotation,
# so we flag them is_thin=True and exclude them from the ranked invest/avoid
# lists (still visible in the full reference). Tunable, evidence-based defaults:
_MIN_LIQUID_STOCKS      = 5      # need ≥5 liquid constituents to be a "sector"
_MAX_TOP_STOCK_PCT      = 50.0   # one name may not exceed this share of turnover
_MIN_SECTOR_TURNOVER_CR = 500.0  # total liquid turnover floor (₹ Cr)


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


def _sector_liquidity_stats(as_of_date: date, min_turnover_lacs: float) -> pd.DataFrame:
    """
    Per-sector basket-quality metrics for the tradability gate.

    Returns one row per sector with:
      n_liquid_stocks    — count of constituents with turnover ≥ min_turnover_lacs
      top_stock_pct      — single largest stock's share of the sector's liquid turnover
      sector_turnover_cr — total liquid turnover in ₹ Cr
      is_thin            — True if the basket is too thin / concentrated / illiquid
      thin_reason        — short human-readable reason ("" when tradable)
    """
    df = query_dataframe("""
        WITH liq AS (
            SELECT sm.sector, b.turnover_lacs
            FROM daily_data b
            JOIN v_sector_master sm ON b.symbol = sm.symbol
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND sm.sector NOT IN ('ETF', 'Others')
              AND b.turnover_lacs >= ?
              AND b.trade_date = (
                  SELECT MAX(trade_date) FROM daily_data WHERE trade_date <= ?
              )
        )
        SELECT sector,
               COUNT(*)              AS n_liquid_stocks,
               SUM(turnover_lacs)    AS tot_lacs,
               MAX(turnover_lacs)    AS top1_lacs
        FROM liq
        GROUP BY sector
    """, [min_turnover_lacs, as_of_date])

    if df.empty:
        return pd.DataFrame(columns=[
            "sector", "n_liquid_stocks", "top_stock_pct",
            "sector_turnover_cr", "is_thin", "thin_reason",
        ])

    df["top_stock_pct"]      = (df["top1_lacs"] / df["tot_lacs"] * 100).round(0)
    df["sector_turnover_cr"] = (df["tot_lacs"] / 100).round(0)

    def _reason(r) -> str:
        bits = []
        if r["n_liquid_stocks"] < _MIN_LIQUID_STOCKS:
            bits.append(f"only {int(r['n_liquid_stocks'])} liquid names")
        if r["top_stock_pct"] > _MAX_TOP_STOCK_PCT:
            bits.append(f"top stock {int(r['top_stock_pct'])}% of turnover")
        if r["sector_turnover_cr"] < _MIN_SECTOR_TURNOVER_CR:
            bits.append(f"₹{int(r['sector_turnover_cr'])} Cr total turnover")
        return " · ".join(bits)

    df["thin_reason"] = df.apply(_reason, axis=1)
    df["is_thin"]     = df["thin_reason"] != ""
    return df[[
        "sector", "n_liquid_stocks", "top_stock_pct",
        "sector_turnover_cr", "is_thin", "thin_reason",
    ]]


# ── Bear-market defense score + per-regime eligibility verdict ─────────────────

# Structure (falls less) + flow (smart money still holding). Weights sum to 1.0.
# Walk-forward validation: down-capture has a +0.41 persistence IC (it PREDICTS
# forward down-day behaviour, 94% hit) vs ~0.10 for momentum/RS — so down-capture
# carries the most weight and RS (less persistent) the least.
_DEFENSE_WEIGHTS = {
    "down_capture": 0.40,   # falls least on down days — validated, most persistent
    "rel_strength": 0.20,   # currently outperforming (momentum — less persistent)
    "beta":         0.10,   # low overall market sensitivity (secondary to down-capture)
    "breadth":      0.20,   # institutions still participating (not fleeing)
    "dv_ratio":     0.10,   # delivery vs own norm (accumulation, not a value trap)
}


def _add_defense_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add defense_score (0-100), a human defensive_verdict, and bear_eligible flag.

    A sector is bear-ELIGIBLE only if it genuinely cushions a downtrend AND smart
    money is not fleeing it. The verdict separates: Defensive Leader / Buy / Hold
    (eligible) vs Value-trap risk / Amplifier (NOT eligible — they fall as much or
    more than the market, or institutions are exiting). This is what powers
    "only sectors eligible for the current market appear" in a bear regime.
    """
    if df.empty or "down_capture" not in df.columns or df["down_capture"].isna().all():
        df["defense_score"] = None
        df["defensive_verdict"] = ""
        df["bear_eligible"] = False
        return df

    dcap = df["down_capture"].astype(float)
    dcap_f = dcap.fillna(dcap.median())
    beta_f = df["beta"].astype(float).fillna(df["beta"].astype(float).median())
    rels_f = df["rel_strength"].astype(float).fillna(0.0)
    brd_f  = (df["breadth"].astype(float).fillna(df["breadth"].astype(float).median())
              if "breadth" in df.columns else pd.Series(0.5, index=df.index))
    dvr_f  = (df["dv_ratio"].astype(float).fillna(1.0)
              if "dv_ratio" in df.columns else pd.Series(1.0, index=df.index))

    w = _DEFENSE_WEIGHTS
    df["defense_score"] = ((
        _rank01(-dcap_f)  * w["down_capture"] +
        _rank01(rels_f)   * w["rel_strength"] +
        _rank01(-beta_f)  * w["beta"] +
        _rank01(brd_f)    * w["breadth"] +
        _rank01(dvr_f)    * w["dv_ratio"]
    ) * 100).round(1)

    # Verdict is DOWN-CAPTURE-primary (the validated, persistent down-day metric).
    # Beta is NOT used to call something an "amplifier" — a high-beta sector with
    # low down-capture falls LESS on down days (and rises more on up days); beta is
    # shown only as a volatility caution. Thresholds: <=0.90 clearly cushions,
    # 0.90-1.05 ~tracks the market (no edge), >1.05 amplifies the fall.
    verdicts, eligible = [], []
    for _, r in df.iterrows():
        dc = r.get("down_capture")
        rs = r.get("rel_strength"); rs = float(rs) if rs is not None and not pd.isna(rs) else 0.0
        br = r.get("breadth"); br = float(br) if br is not None and not pd.isna(br) else 0.5
        if dc is None or pd.isna(dc):
            verdicts.append(""); eligible.append(False); continue
        dc = float(dc)
        if dc > 1.05:
            verdicts.append("🔴 Amplifier — falls MORE than the market"); eligible.append(False)
        elif dc > 0.95:
            verdicts.append("⚖️ Tracks the market — no defensive edge"); eligible.append(False)
        elif br < 0.30 and rs < 0:
            verdicts.append("⚠️ Value-trap risk — cushions the fall but smart money is exiting")
            eligible.append(False)
        elif rs > 0 and br >= 0.40:
            verdicts.append("🛡️ Defensive Leader — falls less AND outperforming, flow holding")
            eligible.append(True)
        elif rs > 0 or br >= 0.40:
            verdicts.append("🛡️ Defensive Buy — cushions the fall with stable flow")
            eligible.append(True)
        else:
            verdicts.append("🛡️ Defensive Hold — cushions the fall (weak catalyst)")
            eligible.append(True)
    df["defensive_verdict"] = verdicts
    df["bear_eligible"] = eligible
    return df


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
            MEDIAN(CASE WHEN b.prev_close > 0
                        THEN (b.close_price - b.prev_close) / b.prev_close * 100 END)
                                                              AS wtd_daily_ret_pct
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
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

    # ── Bottom-up ROBUST delivery signals (the coherent, mega-cap-independent,
    # %-based, multi-day replacement for the 1-day turnover-weighted ₹-value z).
    # Validated over 371d: coherence with own constituents ρ −0.05 → +0.49,
    # contradictions 26% → 10%, forward IC no worse. See sector_signal_v2.py.
    # Sectors absent here (thin / no delivery history) fall back to the legacy
    # z-percentile + dv5d gates below, so nothing silently drops out.
    robz_pct_map: dict = {}
    robz_raw_map: dict = {}
    breadth_accum_map: dict = {}
    ratio5_rob_map: dict = {}
    slope_rob_map: dict = {}
    rob_comp_pct_map: dict = {}   # cross-sectional rank of the robust delivery composite
    try:
        from src.analytics.sector_signal_v2 import get_robust_delivery_signals
        _rob = get_robust_delivery_signals(as_of_date, min_turnover_lacs)
        if not _rob.empty:
            _rob = _rob[~_rob["sector"].isin(("ETF", "Others"))]
            robz_pct_map      = dict(zip(_rob["sector"], _rob["robz"].rank(pct=True)))
            robz_raw_map      = dict(zip(_rob["sector"], _rob["robz"]))
            breadth_accum_map = dict(zip(_rob["sector"], _rob["breadth_accum"]))
            ratio5_rob_map    = dict(zip(_rob["sector"], _rob["deliv_ratio5"]))
            slope_rob_map     = dict(zip(_rob["sector"], _rob["deliv_slope"]))
            # Composite delivery strength = robust abnormality (½) + multi-day
            # breadth (3⁄10) + sustained-flow ratio (⅕), each cross-sectionally
            # ranked so the gate ADAPTS to the day's overall level (the same
            # regime-proofing rationale as the legacy z-percentile), then ranked
            # once more to a clean 0..1 discriminator. Absolute floors (ratio5,
            # breadth) are applied ALONGSIDE this in the gate so a top-ranked
            # sector in a broadly-shrinking week is not over-promoted.
            _comp = (0.5 * _rob["robz"].rank(pct=True)
                     + 0.3 * _rob["breadth_accum"].rank(pct=True)
                     + 0.2 * _rob["deliv_ratio5"].rank(pct=True))
            rob_comp_pct_map = dict(zip(_rob["sector"], _comp.rank(pct=True)))
    except Exception as exc:
        log.warning("Robust delivery signals failed (non-fatal, using legacy): %s", exc)

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
        deliv_1m  = float(row.get("1M_deliv_cr",        float("nan")))
        deliv_3m  = float(row.get("3M_deliv_cr",        float("nan")))
        deliv_100d = float(row.get("100D_deliv_cr",     float("nan")))

        if pd.isna(dv_ratio) or pd.isna(z_score):
            continue

        # ── Multi-timeframe delivery acceleration ─────────────────────────────
        # Daily delivery RATE in the last week vs the last month vs the last
        # quarter. accel_short = 1W pace / 1M pace (>1 = institutional interest
        # ramping recently); accel_med = 1M pace / 3M pace. Sustained acceleration
        # (both >1) is the leading tell of quiet accumulation before a breakout.
        _r1w = deliv_1w / max(n_1w_days, 1) if not pd.isna(deliv_1w) else float("nan")
        _r1m = deliv_1m / 21.0 if not pd.isna(deliv_1m) else float("nan")
        _r3m = deliv_3m / 63.0 if not pd.isna(deliv_3m) else float("nan")
        accel_short = (_r1w / _r1m) if (not pd.isna(_r1w) and not pd.isna(_r1m) and _r1m > 0) else float("nan")
        accel_med   = (_r1m / _r3m) if (not pd.isna(_r1m) and not pd.isna(_r3m) and _r3m > 0) else float("nan")
        # Composite: recent pace weighted more; neutral 1.0 when missing.
        delivery_accel = round(
            0.6 * (accel_short if not pd.isna(accel_short) else 1.0)
            + 0.4 * (accel_med if not pd.isna(accel_med) else 1.0), 3)

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

        # ── Robust bottom-up gate inputs (with legacy fallback) ───────────────
        # These replace the 1-day turnover-weighted ₹-value z with %-based,
        # median/MAD, multi-day, equal-weight signals aggregated FROM the stocks,
        # so the sector verdict agrees with its own constituents (validated).
        # A sector missing from the robust map (thin / no history) transparently
        # falls back to the legacy z-percentile + dv5d anchors.
        _has_rob   = sector in breadth_accum_map
        rz_pct     = float(robz_pct_map.get(sector, z_pct))          # robust delivery rank
        r_breadth  = float(breadth_accum_map.get(sector, breadth if not pd.isna(breadth) else 0.5))
        r_ratio5   = float(ratio5_rob_map.get(sector, dv5d))         # robust sustained-flow ratio
        eff_dv5d   = r_ratio5                                        # sustained-flow anchor (robust)

        # Effective gates: robust composite RANK (adaptive) + absolute floors when
        # available, legacy z-percentile + dv5d otherwise.
        #   comp_pct — cross-sectional rank of the robust delivery composite (0..1)
        #   floors   — sustained-flow ratio5 and multi-day breadth, so a top-ranked
        #              sector in a broadly-shrinking week is NOT promoted to a buy.
        comp_pct = float(rob_comp_pct_map.get(sector, 0.5))
        if _has_rob:
            d_surge = comp_pct >= 0.55                         # top ~45% delivery strength
            d_surge_mid = comp_pct >= 0.55 and r_ratio5 >= 1.00
            d_surge_confirmed = (comp_pct >= 0.70 and r_ratio5 >= 1.02 and r_breadth >= 0.40)
            d_weak = comp_pct <= 0.25
            d_weak_confirmed = (comp_pct <= 0.25 and r_ratio5 <= 0.98 and r_breadth <= 0.35)
        else:
            d_surge = z_pct >= 0.50
            d_surge_mid = d_surge
            d_weak  = z_pct <= 0.25
            d_surge_confirmed = (d_surge and dv5d >= 1.15) or (z_pct >= 0.90 and dv5d >= 0.9)
            d_weak_confirmed  = d_weak and dv5d <= 0.90

        p_up    = (not pd.isna(p1w)) and p1w > 1.0
        p_down  = (not pd.isna(p1w)) and p1w < -1.0

        # pct_surge (delivery % vs 100D avg) still separates speculative turnover
        # spikes from genuine conviction — unchanged, %-based already.
        if not pd.isna(today_wtd_pct) and not pd.isna(avg_wtd_pct) and avg_wtd_pct > 0:
            pct_surge = today_wtd_pct >= avg_wtd_pct * 0.85
        else:
            pct_surge = True

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
        elif d_surge_mid and pct_surge:
            # Mid-tier robust delivery strength (top-half but not top-decile) with
            # sustained flow at/above norm — genuine but early accumulation.
            if p_down:
                signal = "🔥 Secret Accumulation"
                action = "STRONG BUY — Institutions loading while retail panics"
            else:
                signal = "👀 Early Accumulation"
                action = "WATCH — Institutional flow building, not yet top-conviction"
        elif p_up and d_weak_confirmed:
            signal = "⚠️ Distribution Trap"
            action = "EXIT / AVOID — Institutions selling into retail rally"
        elif p_down and d_weak_confirmed:
            signal = "❌ Active Selling"
            action = "AVOID — Broad institutional exit, no floor visible yet"
        elif d_weak and eff_dv5d < 0.95:
            # Relatively weakest (bottom-tercile robust rank + low breadth) AND
            # recent 5-day flow genuinely BELOW its own norm. Tightened from <1.05
            # to <0.95 so a sector whose sustained flow is merely AT norm is not
            # mislabelled "fading" — mild caution only, not a reversal signal.
            signal = "📉 Weakening"
            action = "REDUCE — Relatively weakest flow, conviction fading"
        else:
            signal = "⚖️ Neutral"
            action = "HOLD — Flow within normal range, no clear directional bias"

        # ── Investment horizon (sustained-conviction strength ladder) ───────────
        # NOTE: these tags are a conviction-STRENGTH heuristic, not a per-horizon
        # forward forecast. They are gated on the SUSTAINED-FLOW anchor dv5d (the
        # 5-day avg delivery ratio — a validated score component) + today's dv_ratio
        # + breadth. The old gate used trend_slope, which was measured to have ~0
        # information coefficient at every horizon (see the score note below) and so
        # could VETO a label on a dead factor; it is no longer used here.
        # Short term: top-decile robust delivery abnormality + broad participation
        short_term = rz_pct >= 0.90 and r_breadth >= 0.5
        # Long term: robust sustained flow well above its own norm + broad breadth
        long_term  = (eff_dv5d >= 1.3 and r_breadth >= 0.4)

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
            # Swing (3–15 days): top-decile robust abnormality + broad participation
            if rz_pct >= 0.90 and r_breadth >= 0.5:
                cov.append("Swing (3–15 days)")
            # Positional (1–2 months): robust sustained flow elevated + breadth
            if eff_dv5d >= 1.5 and r_breadth >= 0.5:
                cov.append("Positional (4–8 weeks)")
            # Mid Term (3–4 months): strongly sustained robust flow + deep breadth
            if eff_dv5d >= 1.8 and r_breadth >= 0.6:
                cov.append("Mid Term (3–4 months)")
        else:
            # Avoid signals — how long the weakness may persist (sustained-WEAKNESS ladder)
            if rz_pct <= 0.10 and r_breadth <= 0.3:
                cov.append("Swing (3–15 days)")
            if eff_dv5d <= 0.90 and r_breadth <= 0.35:
                cov.append("Positional (4–8 weeks)")
            if eff_dv5d <= 0.80 and r_breadth <= 0.25:
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
            "z_pct":                round(z_pct, 3),   # cross-sectional rank of z (0..1)
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
            "delivery_accel":       delivery_accel,
            # ── Robust bottom-up delivery signals (analysis + display) ──────────
            "robz":                 round(float(robz_raw_map.get(sector, float("nan"))), 3)
                                        if sector in robz_raw_map else None,
            "breadth_accum":        round(r_breadth, 3),
            "deliv_ratio5":         round(r_ratio5, 3),
            "deliv_slope":          round(float(slope_rob_map.get(sector, float("nan"))), 4)
                                        if sector in slope_rob_map else None,
            "_dv5d":                dv5d,
            "_dv":                  dv_ratio,
            "_z":                   z_score,
            "_br":                  breadth  if not pd.isna(breadth)  else 0.5,
            "_pm":                  p1w      if not pd.isna(p1w)      else 0.0,
            "_slope":               trend_slope,
            "_accel":               delivery_accel,
            "_p1m":                 p1m      if not pd.isna(p1m)      else 0.0,
            # Robust scoring inputs (coalesced to legacy when robust absent).
            "_r5":                  float(r_ratio5),
            "_rbr":                 float(r_breadth),
            "_rz":                  float(robz_raw_map.get(sector, z_score)),
            "_rsl":                 float(slope_rob_map.get(sector, trend_slope)),
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
    # 45% RS vs Nifty (2W) + 20% 5-day avg DV + 10% today DV +
    # 15% Breadth + 10% Z-Score
    #
    # Re-tuned 2026-06 after a walk-forward IC study on the CANONICAL 22-sector
    # universe (scripts/factor_ic_diagnostic.py + sector_score_compare.py). RS/2W
    # momentum is by far the strongest predictor (IC ≈ +0.10, ICIR ≈ +0.46 @5-10d)
    # while delivery factors are weak (IC ≈ +0.03). IC rises MONOTONICALLY with RS
    # weight (live 0.30 → 0.45 → 0.60 → pure RS), so RS was raised 0.30 → 0.45.
    # We deliberately do NOT go to pure RS: (a) the ~250-day sample is essentially
    # one regime — momentum's edge is regime-sensitive and would over-fit; (b)
    # delivery is the differentiated smart-money thesis and drives the signal
    # labels. 0.45 captures most of the measured lift (~+40% IC) while keeping
    # delivery + breadth as the majority. RS falls back to 1W momentum if index
    # data is unavailable.
    # Delivery inputs are now the ROBUST BOTTOM-UP factors (_r5 sustained-flow
    # ratio, _rbr multi-day breadth, _rz robust %-based abnormality, _rsl slope),
    # which agree with each sector's own constituents (validated ρ +0.49 vs −0.05
    # for the legacy 1-day ₹-value z) at no cost to forward IC. RS/2W momentum
    # stays the top weight — the measured alpha.
    rs_col = result["rs_2w"] if result["rs_2w"].notna().any() else result["_pm"]
    result["accum_score"] = (
        _rank01(rs_col)            * 45 +
        _rank01(result["_r5"])     * 20 +   # robust sustained 5d/100d delivery ratio
        _rank01(result["_rbr"])    * 15 +   # multi-day accumulation breadth
        _rank01(result["_rz"])     * 10 +   # robust %-based delivery abnormality
        _rank01(result["_rsl"])    * 10     # delivery-% slope (direction)
    ).round(1)

    # ── ACCUMULATION score (SIDEWAYS/range) — quiet smart-money delivery, NOT
    # momentum. In a range, sustained + accelerating delivery with broad breadth
    # leads the eventual breakout; RS is only a minor tilt.
    result["accumulation_score"] = (
        _rank01(result["_r5"])     * 30 +   # robust sustained delivery flow
        _rank01(result["_rz"])     * 20 +   # robust delivery abnormality
        _rank01(result["_accel"])  * 20 +   # multi-timeframe delivery acceleration
        _rank01(result["_rbr"])    * 15 +   # multi-day accumulation breadth
        _rank01(rs_col)            * 15     # relative strength (minor)
    ).round(1)

    # ── REVERSAL score (DOWNTREND BOTTOMING) — delivery accumulation INTO price
    # weakness (the Secret-Accumulation thesis at an index bottom): institutions
    # taking delivery while price is still beaten down → highest risk/reward entry.
    result["reversal_score"] = (
        _rank01(result["_r5"])     * 30 +   # robust delivery strength despite weakness
        _rank01(result["_rz"])     * 20 +   # robust delivery abnormality
        _rank01(result["_accel"])  * 20 +   # institutions ramping
        _rank01(-result["_p1m"])   * 15 +   # oversold (more down = higher)
        _rank01(result["_rbr"])    * 15     # multi-day accumulation breadth
    ).round(1)

    result = result.drop(columns=[c for c in result.columns if c.startswith("_")])

    # ── Tradability gate: flag thin / single-name / illiquid baskets ──────────
    # These columns let the dashboard exclude non-tradable buckets from the
    # ranked invest/avoid lists and disclose basket quality on each card.
    try:
        liq = _sector_liquidity_stats(as_of_date, min_turnover_lacs)
        result = result.merge(liq, on="sector", how="left")
        # Sectors missing from the liquidity query (no liquid names) are thin.
        result["is_thin"] = result["is_thin"].fillna(True)
        result["thin_reason"] = result["thin_reason"].fillna("no liquid constituents")
        result["n_liquid_stocks"]    = result["n_liquid_stocks"].fillna(0).astype(int)
        result["top_stock_pct"]      = result["top_stock_pct"]
        result["sector_turnover_cr"] = result["sector_turnover_cr"]
    except Exception as exc:
        log.warning("Sector liquidity gate failed (non-fatal): %s", exc)
        result["is_thin"]            = False
        result["thin_reason"]        = ""
        result["n_liquid_stocks"]    = None
        result["top_stock_pct"]      = None
        result["sector_turnover_cr"] = None

    # ── Defensive metrics + bear-market defense score & eligibility ───────────
    # Point-in-time. defense_score combines STRUCTURE (falls less: low downside-
    # capture + low beta + positive relative strength) with FLOW (institutions
    # still holding: breadth + delivery). A low-beta sector that smart money is
    # FLEEING is a value trap, not a safe buy — so flow matters. This is what
    # should rank picks in a BEAR/CAUTION regime, not momentum (accum_score).
    try:
        from src.analytics.sector_defensive import get_sector_defensive_metrics
        dfn = get_sector_defensive_metrics(as_of_date, min_turnover_lacs=min_turnover_lacs)
        if not dfn.empty:
            result = result.merge(
                dfn[["sector", "beta", "down_capture", "up_capture", "rel_strength"]],
                on="sector", how="left",
            )
        else:
            for c in ("beta", "down_capture", "up_capture", "rel_strength"):
                result[c] = None
    except Exception as exc:
        log.warning("Sector defensive metrics failed (non-fatal): %s", exc)
        for c in ("beta", "down_capture", "up_capture", "rel_strength"):
            result[c] = None

    result = _add_defense_score(result)

    return result.sort_values("accum_score", ascending=False).reset_index(drop=True)


# ── Market Regime ─────────────────────────────────────────────────────────────

def get_market_regime(as_of_date: date) -> dict:
    """
    Compute the current market regime from 5 independent signals already in the DB.

    Design principle: sector rotation signals are RELATIVE (rank within universe).
    The market regime answers the ABSOLUTE question: is the environment suitable
    for deploying capital, or is the index beta drag larger than any sector alpha?

    Five inputs — no new data needed:
      1. Nifty50 vs 20D EMA  (short-term trend direction)
      2. Nifty50 vs 50D EMA  (medium-term trend confirmation)
      3. India VIX level + 5D change  (fear gauge)
      4. FII 5D cumulative futures net flow  (institutional stance)
      5. HMM statistical state from prediction_log  (regime memory)

    Score: 0.0 (max bear) → 10.0 (max bull), 5.0 = neutral.
    Regime labels:
      BULL          (≥ 7.0) — all systems go, sector signals have absolute return tailwind
      CAUTIOUS BULL (≥ 5.5) — market up but momentum soft, favour higher-conviction signals
      SIDEWAYS      (≥ 4.0) — no clear market direction, prefer relative/pairs trades
      CAUTION       (≥ 2.5) — deteriorating, reduce sector bets, tighten stops
      BEAR          (< 2.5) — index beta drag likely > sector alpha; pairs trade or wait

    Returns:
      regime        str   — one of the 5 labels above
      score         float — 0–10
      nifty_vs_ema20  str  — "ABOVE" / "BELOW" / "—"
      nifty_vs_ema50  str  — "ABOVE" / "BELOW" / "—"
      nifty_1m_pct  float | None  — Nifty50 1-month return %
      vix           float | None  — latest India VIX level
      vix_trend     str   — "RISING" / "FALLING" / "STABLE"
      fii_5d_cr     float | None  — FII 5D cumulative flow ₹Cr
      hmm_state     str   — "Bull" / "Bear" / "Sideways" / "—"
      signals       list  — human-readable signal descriptions (for the banner)
      invest_label  str   — adjusted heading for SECTORS TO INVEST
      invest_caption str  — adjusted caption explaining regime context
    """
    score   = 5.0   # neutral baseline
    signals = []

    # ── Signal 1: Nifty50 vs 20D EMA — short-term trend ─────────────────────
    # 200 calendar days ≈ 140 trading days.  EMA50 needs ~4×span to converge
    # (span=50 → need 200+ points).  Using 200 cal days ensures both EMA20 and
    # EMA50 are computed on a properly warmed-up series.
    # The 1M return is NOT scored separately — it is highly correlated with the
    # EMA position (both measure the same price direction) and would double-count.
    ema20_state = ema50_state = "—"
    nifty_1m    = None
    try:
        nifty_hist = query_dataframe("""
            SELECT trade_date, close_val
            FROM index_data
            WHERE index_name = 'Nifty 50'
              AND trade_date <= ?
              AND trade_date >= (? - INTERVAL 200 DAY)
            ORDER BY trade_date
        """, [as_of_date, as_of_date])

        if not nifty_hist.empty and len(nifty_hist) >= 22:
            nifty_hist = nifty_hist.sort_values("trade_date")
            close_s    = nifty_hist["close_val"].astype(float)
            latest     = float(close_s.iloc[-1])

            # ── EMA20 (short-term trend, primary directional signal) ──────────
            ema_20 = float(close_s.ewm(span=20, adjust=False).mean().iloc[-1])
            if latest > ema_20:
                score += 1.5
                ema20_state = "ABOVE"
                signals.append(("bull", f"Nifty50 {latest:,.0f} > 20D EMA {ema_20:,.0f} — short-term uptrend"))
            else:
                score -= 1.5
                ema20_state = "BELOW"
                signals.append(("bear", f"Nifty50 {latest:,.0f} < 20D EMA {ema_20:,.0f} — short-term downtrend"))

            # ── Golden / Death Cross: EMA20 vs EMA50 (independent from price vs EMA) ──
            # Using cross of EMAs rather than price vs EMA50 avoids double-counting
            # the same price direction information.  An EMA cross is a structurally
            # different signal — it fires only at TREND INFLECTIONS, not every day.
            if len(close_s) >= 80:   # need warmup: span=50 needs ~80 pts for reliable cross
                ema_50 = float(close_s.ewm(span=50, adjust=False).mean().iloc[-1])
                if ema_20 > ema_50:
                    score += 1.0
                    ema50_state = "ABOVE"
                    signals.append(("bull", f"20D EMA ({ema_20:,.0f}) > 50D EMA ({ema_50:,.0f}) — golden cross, medium-term structure bullish"))
                else:
                    score -= 1.0
                    ema50_state = "BELOW"
                    signals.append(("bear", f"20D EMA ({ema_20:,.0f}) < 50D EMA ({ema_50:,.0f}) — death cross, medium-term structure broken"))

            # 1M return — stored for display only, NOT scored (correlated with EMA signals)
            if len(close_s) >= 22:
                start22 = float(close_s.iloc[-22])
                nifty_1m = round((latest - start22) / start22 * 100, 2) if start22 else None
    except Exception:
        pass

    # ── Signal 2: India VIX — fear gauge (level only, symmetric scoring) ─────
    # Scoring calibrated so that NORMAL VIX range for India (14–18%) = 0 points.
    # Only EXTREMES shift the score. Avoids the persistent bull bias caused by
    # treating normal VIX (12–16) as inherently bullish.
    vix       = None
    vix_trend = "STABLE"
    try:
        vix_hist = query_dataframe("""
            SELECT close_val FROM index_data
            WHERE index_name = 'India VIX'
              AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 7
        """, [as_of_date])

        if not vix_hist.empty:
            vix_vals  = vix_hist["close_val"].astype(float).tolist()
            vix       = vix_vals[0]
            # True 5-trading-day change: index 0 = today, index 5 = 5 trading days ago
            vix_5d_ch = vix_vals[0] - vix_vals[min(5, len(vix_vals) - 1)]

            # Level scoring — symmetric neutral zone (14–18%) scores zero
            if vix < 12:
                score += 1.0
                signals.append(("bull", f"VIX {vix:.1f} — very low fear, stable environment"))
            elif vix > 25:
                score -= 1.5
                signals.append(("bear", f"VIX {vix:.1f} — ELEVATED fear, high downside risk"))
            elif vix > 20:
                score -= 0.75
                signals.append(("bear", f"VIX {vix:.1f} — rising fear, caution warranted"))
            # VIX 12–20 = normal range = 0 contribution (no bull bias on flat days)

            # Trend scoring — direction of fear change is independent of level
            if vix_5d_ch > 3:
                score -= 0.5
                vix_trend = "RISING"
                signals.append(("bear", f"VIX +{vix_5d_ch:.1f} pts in 5D — fear accelerating"))
            elif vix_5d_ch < -3:
                score += 0.5
                vix_trend = "FALLING"
                signals.append(("bull", f"VIX {vix_5d_ch:.1f} pts in 5D — fear dissipating"))
    except Exception:
        pass

    # ── Signal 3: FII 5D cumulative flow — all index futures combined ─────────
    # FIX 1: Use ALL index futures (NIFTY + BANKNIFTY + FINNIFTY + MIDCAP) to
    #   capture the total institutional directional stance, not just NIFTY.
    # FIX 2: Subquery limits to EXACTLY 5 trading days before aggregating.
    #   The original LIMIT 5 on a SUM() aggregate was dead code — it limited the
    #   output rows (already 1 from SUM), not the input rows entering the sum.
    fii_5d = None
    try:
        fii_df = query_dataframe("""
            SELECT SUM(net_cr) AS net_cr
            FROM (
                SELECT trade_date, SUM(buy_value_cr - sell_value_cr) AS net_cr
                FROM fii_derivatives_stats
                WHERE category IN (
                    'NIFTY FUTURES', 'BANKNIFTY FUTURES',
                    'FINNIFTY FUTURES', 'MIDCPNIFTY FUTURES'
                )
                  AND trade_date <= ?
                GROUP BY trade_date
                ORDER BY trade_date DESC
                LIMIT 5
            ) last5
        """, [as_of_date])

        if not fii_df.empty and fii_df["net_cr"].notna().any():
            fii_5d = round(float(fii_df["net_cr"].iloc[0]), 0)
            if fii_5d > 3000:
                score += 1.0
                signals.append(("bull", f"FII all-index 5D net +₹{fii_5d:,.0f} Cr — broad institutional buying"))
            elif fii_5d > 800:
                score += 0.5
                signals.append(("bull", f"FII all-index 5D net +₹{fii_5d:,.0f} Cr — mild institutional buying"))
            elif fii_5d < -3000:
                score -= 1.0
                signals.append(("bear", f"FII all-index 5D net ₹{fii_5d:,.0f} Cr — broad institutional selling"))
            elif fii_5d < -800:
                score -= 0.5
                signals.append(("bear", f"FII all-index 5D net ₹{fii_5d:,.0f} Cr — mild institutional selling"))
            else:
                signals.append(("neut", f"FII all-index 5D net ₹{fii_5d:+,.0f} Cr — neutral institutional stance"))
    except Exception:
        pass

    # ── Signal 4: HMM statistical regime — staleness-capped ──────────────────
    # Cap: only use HMM state if it was published within the last 7 calendar days.
    # Without this, a 30-day-old prediction_log entry would contaminate the score.
    hmm_state = "—"
    try:
        hmm_df = query_dataframe("""
            SELECT hmm_state FROM prediction_log
            WHERE fno_symbol = 'NIFTY'
              AND trade_date <= ?
              AND trade_date >= (? - INTERVAL 7 DAY)
            ORDER BY trade_date DESC
            LIMIT 1
        """, [as_of_date, as_of_date])

        if not hmm_df.empty:
            hmm_state = str(hmm_df["hmm_state"].iloc[0])
            if hmm_state == "Bull":
                score += 1.0
                signals.append(("bull", "HMM: BULL — statistical price memory supports upside"))
            elif hmm_state == "Bear":
                score -= 1.0
                signals.append(("bear", "HMM: BEAR — statistical price memory supports downside"))
            else:
                signals.append(("neut", f"HMM: {hmm_state} — no statistical directional bias"))
    except Exception:
        pass

    # ── Signal 5: Market breadth — % of stocks advancing ─────────────────────
    # Independent of EMA/FII — measures PARTICIPATION quality.
    # Low breadth in a rising market = narrow leadership = fragile rally.
    # High breadth in a falling market = broad-based genuine selling.
    try:
        breadth_df = query_dataframe("""
            SELECT
                SUM(CASE WHEN close_price > prev_close THEN 1 ELSE 0 END) * 1.0
                / NULLIF(COUNT(*), 0) AS breadth_frac
            FROM daily_data
            WHERE trade_date = (
                SELECT MAX(trade_date) FROM daily_data WHERE trade_date <= ?
            )
              AND series = 'EQ'
              AND prev_close > 0
              AND turnover_lacs >= 5
        """, [as_of_date])

        if not breadth_df.empty and breadth_df["breadth_frac"].notna().any():
            breadth = float(breadth_df["breadth_frac"].iloc[0])
            if breadth > 0.65:
                score += 0.75
                signals.append(("bull", f"Breadth {breadth*100:.0f}% advancing — broad market participation"))
            elif breadth > 0.55:
                score += 0.25
                signals.append(("bull", f"Breadth {breadth*100:.0f}% advancing — mild positive participation"))
            elif breadth < 0.35:
                score -= 0.75
                signals.append(("bear", f"Breadth {breadth*100:.0f}% advancing — broad-based selling confirmed"))
            elif breadth < 0.45:
                score -= 0.25
                signals.append(("bear", f"Breadth {breadth*100:.0f}% advancing — below-average participation"))
            # 45–55% = neutral zone = 0 contribution
    except Exception:
        pass

    # ── Signal 6: EOD Put-Call Ratio — forward-looking options positioning ────
    # PCR is the ONLY forward-looking signal in the regime.  Unlike EMA/FII/HMM
    # (all backward-looking), PCR reflects where option WRITERS (dealers/smart
    # money) are positioned for the NEXT session.
    #
    # PCR interpretation (from the option writer's perspective, not buyer's):
    #   PCR > 1.3 → dealers sold MORE puts than calls → they expect support below
    #               (put writers only sell if they believe the floor holds)
    #   PCR < 0.75 → dealers sold MORE calls than puts → they see a ceiling above
    #               (call writers only sell if they believe upside is capped)
    #
    # Uses feat_pcr from prediction_log (same as DCM prediction engine) — computed
    # from the most recent fno_bhavcopy near-expiry OI.  Staleness cap: 7 days.
    try:
        pcr_df = query_dataframe("""
            SELECT feat_pcr FROM prediction_log
            WHERE fno_symbol = 'NIFTY'
              AND trade_date <= ?
              AND trade_date >= (? - INTERVAL 7 DAY)
              AND feat_pcr IS NOT NULL
            ORDER BY trade_date DESC
            LIMIT 1
        """, [as_of_date, as_of_date])

        if not pcr_df.empty and pcr_df["feat_pcr"].notna().any():
            pcr = float(pcr_df["feat_pcr"].iloc[0])
            if pcr > 1.3:
                score += 0.75
                signals.append(("bull", f"PCR {pcr:.2f} — heavy put writing, dealers expecting market support"))
            elif pcr > 1.05:
                score += 0.25
                signals.append(("bull", f"PCR {pcr:.2f} — mild put writing bias, slight options support"))
            elif pcr < 0.75:
                score -= 0.75
                signals.append(("bear", f"PCR {pcr:.2f} — heavy call writing, dealers capping upside"))
            elif pcr < 0.90:
                score -= 0.25
                signals.append(("bear", f"PCR {pcr:.2f} — mild call writing dominance, options headwind"))
            # PCR 0.90–1.05 = balanced = 0 contribution
    except Exception:
        pass

    score = round(max(0.0, min(10.0, score)), 2)

    # ── Regime classification ─────────────────────────────────────────────────
    # Bands designed to minimise whipsaw.  With 6 signals (max swing ≈ ±6.25 pts):
    #  BULL (≥ 7.5)     — majority of signals strongly bullish; clear uptrend
    #  CAUTIOUS BULL    — trend intact, some mixed signals; reduce size
    #  SIDEWAYS (4.0–6.0) — widest band (2 pts) to absorb normal daily noise
    #  CAUTION          — deteriorating but not confirmed bear
    #  BEAR (< 2.5)     — majority bearish; defer new capital
    if score >= 7.5:
        regime = "BULL"
    elif score >= 6.0:
        regime = "CAUTIOUS BULL"
    elif score >= 4.0:
        regime = "SIDEWAYS"
    elif score >= 2.5:
        regime = "CAUTION"
    else:
        regime = "BEAR"

    # ── Dynamic headings + guidance based on regime ───────────────────────────
    _REGIME_CONTEXT = {
        "BULL": {
            "invest_label":   "🟢 SECTORS TO INVEST",
            "invest_caption": (
                "All accumulation signals — highest score first. "
                "Market uptrend confirmed: sector alpha + market beta both working for you."
            ),
            "avoid_label":    "🔴 SECTORS TO AVOID / EXIT",
            "avoid_caption":  "Active distribution/selling signals first; then relative laggards.",
        },
        "CAUTIOUS BULL": {
            "invest_label":   "🟡 SECTORS TO OVERWEIGHT",
            "invest_caption": (
                "Market up but momentum soft. "
                "Favour the highest-conviction signals (Secret + Confirmed Accumulation) only. "
                "Reduce size on early/watch signals."
            ),
            "avoid_label":    "🟠 SECTORS TO REDUCE / AVOID",
            "avoid_caption":  "Distribution and weakness signals — exit before market confirms turn.",
        },
        "SIDEWAYS": {
            "invest_label":   "🔵 RELATIVE ALPHA — SELECTIVE ACCUMULATION",
            "invest_caption": (
                "Market in range — no broad tailwind. "
                "These sectors outperform on a RELATIVE basis. "
                "Prefer stock-level entry with sector as thesis, not sector ETF/basket. "
                "Or: long accumulation sectors / short laggard sectors (pairs trade)."
            ),
            "avoid_label":    "🔵 SECTORS TO UNDERWEIGHT / SHORT",
            "avoid_caption":  (
                "Weakest relative performers in a range-bound market. "
                "Good short leg for pairs trades against the invest column."
            ),
        },
        "CAUTION": {
            "invest_label":   "⚠️ RELATIVE ALPHA — HEADWIND FROM MARKET",
            "invest_caption": (
                "Market deteriorating. These sectors FALL LESS than the market. "
                "Absolute returns likely negative even for accumulation signals. "
                "Use only for pairs (long accumulation / short index) or wait for regime flip."
            ),
            "avoid_label":    "🔴 DOUBLE SELL — SECTOR + MARKET BOTH BEARISH",
            "avoid_caption":  (
                "Both sector signal AND market regime are against these sectors. "
                "Exit immediately. Do not average down."
            ),
        },
        "BEAR": {
            "invest_label":   "🛡️ DEFENSIVE HOLD — RELATIVE ALPHA ONLY",
            "invest_caption": (
                "BEAR market confirmed. Even strong accumulation sectors will LOSE money "
                "in absolute terms as index beta drag exceeds sector alpha. "
                "These sectors fall LEAST — suitable for: (a) pairs trade vs Nifty short, "
                "(b) defensive core positioning, or (c) staged accumulation for the regime flip. "
                "Do NOT deploy fresh capital expecting positive returns until regime improves."
            ),
            "avoid_label":    "🔴 REDUCE POSITIONS — SECTOR + MARKET BOTH BEARISH",
            "avoid_caption":  (
                "Bear market + sector weakness = double risk. "
                "Reduce or exit existing positions. Priority: weakest sectors first."
            ),
        },
    }

    ctx = _REGIME_CONTEXT.get(regime, _REGIME_CONTEXT["SIDEWAYS"])

    # ── Bear sub-type: distinguish early-distribution from full-panic bear ─────
    # VIX < 18 in a BEAR regime = "quiet bear" — FII-driven selling without retail
    # panic. Less dangerous for holders; regime may reverse faster than a VIX-spike
    # bear.  VIX ≥ 20 = fear rising = "confirmed bear" — reduce all positions.
    if regime == "BEAR" and vix is not None:
        if vix < 18:
            ctx = dict(ctx)   # copy to avoid mutating the template
            ctx["invest_caption"] = (
                "EARLY BEAR / DISTRIBUTION PHASE — VIX still calm ({:.1f}): institutions "
                "are quietly selling but retail has not panicked yet. "
                "These sectors fall LESS than the index. "
                "Strategy: (a) hold existing positions with hard stops, "
                "(b) do NOT add fresh capital outright, "
                "(c) if FII selling persists another week, move to pairs or exit. "
                "Watch for VIX breaking above 18 — that confirms acceleration."
            ).format(vix)
            ctx["avoid_caption"] = (
                "QUIET DISTRIBUTION — Sector weakness in a low-VIX bear. "
                "Reduce exposure gradually (not panic-exit). "
                "Priority: exit positions already in loss; hold profitable ones with trailing stop."
            )
        elif vix >= 20:
            ctx = dict(ctx)
            ctx["invest_caption"] = (
                "CONFIRMED BEAR + ELEVATED FEAR — VIX {:.1f}: retail is beginning to panic. "
                "Index beta drag is accelerating. Even the strongest delivery sectors "
                "will face forced selling from margin calls. "
                "Exit all but highest-conviction long-term holdings. "
                "Do not buy ANY new positions. Wait for VIX to peak and turn down."
            ).format(vix)
            ctx["avoid_caption"] = (
                "MAXIMUM RISK — Bear market + elevated fear + sector distribution. "
                "Exit immediately. Do not average down under any circumstances."
            )

    # ── Banner guidance: short 1-2 sentence text for the regime badge ───────────
    # This is VIX-differentiated — a different message for each bear sub-type.
    # Sourced from ctx["invest_caption"] which was already overridden by VIX check.
    # The view _REGIME_GUIDANCE dict was a parallel copy that was NOT VIX-aware;
    # routing through here ensures both the banner AND the section caption are
    # consistent and show the same calibrated message.
    _BANNER_GUIDANCE = {
        "BULL":          "Market uptrend confirmed — sector alpha + market beta both working FOR you. Deploy full size on high-conviction signals.",
        "CAUTIOUS BULL": "Market rising but momentum is soft. Favour Secret + Confirmed Accumulation only. Reduce size on Watch/Early signals.",
        "SIDEWAYS":      "No clear market direction. Sector signals are RELATIVE — best as long/short pairs, not outright directional bets.",
        "CAUTION":       "Market deteriorating. Accumulation sectors likely lose money in absolute terms. Use as pairs (long sector / short Nifty) or wait for regime flip.",
        "BEAR":          ctx["invest_caption"],   # VIX-differentiated bear guidance
    }
    banner_guidance = _BANNER_GUIDANCE.get(regime, ctx["invest_caption"])

    return {
        "regime":           regime,
        "score":            score,
        "nifty_vs_ema20":   ema20_state,
        "nifty_vs_ema50":   ema50_state,
        "nifty_1m_pct":     nifty_1m,
        "vix":              vix,
        "vix_trend":        vix_trend,
        "fii_5d_cr":        fii_5d,
        "hmm_state":        hmm_state,
        "signals":          signals,
        "invest_label":     ctx["invest_label"],
        "invest_caption":   ctx["invest_caption"],
        "avoid_label":      ctx["avoid_label"],
        "avoid_caption":    ctx["avoid_caption"],
        "banner_guidance":  banner_guidance,        # VIX-differentiated, for the banner
    }


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
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
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
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
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


def get_clock_stock_detail(
    sector: str,
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    lookback_days: int = 14,
) -> pd.DataFrame:
    """
    Which stocks inside a Rotation-Clock sector are worth looking at, and which
    merely DROVE the move. Two different questions — the drill-down shows both,
    because ranking by the second one is measurably backwards.

    MEASURED (scripts/audit_stock_pick_in_clock.py, 1.19M stock-days 2018-2026,
    forward 10d EXCESS OVER THE STOCK'S OWN SECTOR, so this is pure stock
    selection with the sector call stripped out; Newey-West t at lag=10):

        feature                              IC        t
        dacc  (delivery vs own 100d normal)  +0.0323  +9.56   <- the one that works
        rel_mom (momentum vs sector)         -0.0226  -4.79
        dvshare (share of sector delivery)   -0.0351  -4.79
        turnsurge (turnover vs own normal)   -0.0164  -3.52
        contrib (drove the sector move)      -0.0212  -4.85

    `dacc` quintiles are MONOTONIC (-0.245 / -0.082 / -0.012 / +0.100 / +0.247),
    Q5-Q1 = +0.562%/10d (t +7.36, 65% hit), stable in all three eras
    (t +4.85 / +6.66 / +5.13) and it SURVIVES residualising on the stock's own
    momentum (t +8.42). It is the cleanest stock-level signal in this codebase.

    THE COST CAVEAT THAT DECIDES HOW TO USE IT: as a STANDALONE trade, top-3 by
    dacc nets -4.9%/yr (t -2.32) — a 0.5% round trip against +0.31%/10d gross.
    It only pays as a SELECTION rule inside a sector you have ALREADY decided to
    buy, where you must hold something anyway and the incremental cost is zero.

    Returns one row per stock with `dacc`, `contrib_pct`, `rel_ret_pct` and a
    plain-English `read`.
    """
    base = get_sector_stocks_rotation(sector, as_of_date, min_turnover_lacs,
                                      lookback_days=lookback_days)
    if base is None or base.empty:
        return pd.DataFrame()
    df = base.copy()

    # delivery conviction vs the stock's OWN normal (NOT a sector-relative percentile —
    # that distinction is exactly why the earlier stock-level read came out inverted)
    if {"wtd_deliv_per", "avg_deliv_per_100d"}.issubset(df.columns):
        df["dacc"] = df["wtd_deliv_per"] / df["avg_deliv_per_100d"].replace(0, np.nan)
    else:
        df["dacc"] = np.nan

    # REAL prices. `price_chg_pct` from get_sector_stocks_rotation is a
    # turnover-weighted MEAN OF DAILY returns, not a period return — over a 5-day
    # window it reads roughly a daily average, ~5x understated. For a "how did this
    # stock do" column we need actual close-to-close moves, so fetch them directly:
    #   ltp        last close on/before as_of
    #   chg_1d_pct last close vs the session before it
    #   ret_win_pct last close vs the close just before the window opened
    px = query_dataframe(
        """
        WITH px AS (
            SELECT b.symbol, b.trade_date, b.close_price,
                   ROW_NUMBER() OVER (PARTITION BY b.symbol
                                      ORDER BY b.trade_date DESC) AS rn_desc
            FROM daily_data b
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND s.sector = ?
              AND b.trade_date <= ?
        ),
        last2 AS (
            SELECT symbol,
                   MAX(CASE WHEN rn_desc = 1 THEN close_price END) AS ltp,
                   MAX(CASE WHEN rn_desc = 2 THEN close_price END) AS prev_close
            FROM px WHERE rn_desc <= 2 GROUP BY symbol
        ),
        start_px AS (
            SELECT symbol, ARGMAX(close_price, trade_date) AS start_close
            FROM px WHERE trade_date <= ? GROUP BY symbol
        )
        SELECT l.symbol, l.ltp,
               CASE WHEN l.prev_close > 0
                    THEN (l.ltp - l.prev_close) / l.prev_close * 100 END AS chg_1d_pct,
               CASE WHEN s.start_close > 0
                    THEN (l.ltp - s.start_close) / s.start_close * 100 END AS ret_win_pct
        FROM last2 l LEFT JOIN start_px s ON l.symbol = s.symbol
        """,
        [sector, as_of_date, as_of_date - timedelta(days=lookback_days)],
    )
    if not px.empty:
        df = df.drop(columns=[c for c in ("ltp",) if c in df.columns]).merge(
            px, on="symbol", how="left")
    else:
        df["ltp"] = np.nan; df["chg_1d_pct"] = np.nan; df["ret_win_pct"] = np.nan

    # relative move = this stock's WINDOW return minus the sector's typical one
    df["rel_ret_pct"] = df["ret_win_pct"] - df["ret_win_pct"].median()

    dv_col = next((c for c in ("deliv_value_cr", "deliv_val_cr") if c in df.columns), None)
    if dv_col:
        tot = float(df[dv_col].sum())
        df["dv_share_pct"] = 100.0 * df[dv_col] / tot if tot else np.nan
        # attribution: how much of the sector's move this name accounts for
        df["contrib_pct"] = df["rel_ret_pct"] * df["dv_share_pct"] / 100.0
    else:
        df["dv_share_pct"] = np.nan
        df["contrib_pct"] = np.nan

    def _read(r) -> str:
        d, rr = r.get("dacc"), r.get("rel_ret_pct")
        if pd.isna(d):
            return "no delivery baseline yet"
        if d >= 1.20 and (pd.isna(rr) or rr <= 0):
            return "heavy delivery, price still lagging - the strongest combination"
        if d >= 1.20:
            return "heavy delivery vs its own normal"
        if d >= 1.05:
            return "delivery a little above its own normal"
        if d <= 0.85:
            return "delivery BELOW its own normal - weakest quintile historically"
        return "delivery near its own normal"

    df["read"] = df.apply(_read, axis=1)
    # Thin-name guard. Extreme dacc skews to SMALL names: on 2026-08-07 the only
    # dacc>3 name in Metals & Mining carried Rs48 Cr of delivery vs a sector median
    # of Rs167 Cr. A 5x delivery jump on a small base is real but not necessarily
    # tradable, so it is flagged rather than silently ranked first. (Separately,
    # 855 of 2,712 symbols have <20 liquid days in the trailing 150, which makes
    # their 100-day baseline itself noisy.)
    if "deliv_value_cr" in df.columns and len(df) >= 4:
        med = float(df["deliv_value_cr"].median())
        df["thin"] = df["deliv_value_cr"] < (0.33 * med)
    else:
        df["thin"] = False
    df.loc[df["thin"], "read"] = df.loc[df["thin"], "read"] + " (small base - check liquidity)"
    return df.sort_values("dacc", ascending=False).reset_index(drop=True)


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
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
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
            -- ROBUST daily sector return = MEDIAN of constituent returns. Turnover-
            -- weighting uses contemporaneous volume (big moves trade big) → a
            -- systematic upward bias that inflated the clock's price axis and made
            -- almost every sector look "above market" (Improving/Lagging empty).
            -- Median is unbiased and immune to penny-stock / corporate-action spikes.
            MEDIAN(CASE WHEN b.prev_close > 0
                        THEN (b.close_price - b.prev_close) / b.prev_close * 100 END)
                                                               AS wtd_daily_ret_pct
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
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

    # Quadrant CENTER = cross-sectional MEDIAN sector return (the typical sector),
    # NOT Nifty50. Both this and the Y-axis (slope_z) are then relative to the SECTOR
    # universe — apples-to-apples for rotation. Benchmarking equal-weight median
    # sector returns against the cap-weighted Nifty50 created a systematic gap (every
    # sector looked like it lagged the index when mega-caps led). Nifty is kept only
    # as a reference annotation.
    center = float(df["cum_price_ret_pct"].median())
    df["sector_median_ret"] = round(center, 2)

    def _phase_and_confidence(row) -> tuple:
        sz   = row["slope_z"]
        pr   = row["cum_price_ret_pct"] - center
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
            MEDIAN(CASE WHEN b.prev_close > 0
                        THEN (b.close_price - b.prev_close) / b.prev_close * 100 END)
                                                        AS wtd_daily_ret_pct
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
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
    # Correctness is judged vs PEER SECTORS (cross-sectional median forward return),
    # matching the clock's cross-sectional classification — apples-to-apples. (Judging
    # equal-weight median sector returns against the cap-weighted Nifty50 introduced a
    # systematic gap.) forward_vs_nifty is kept only as a reference column.
    peer_med = float(pd.Series(forward_rets).median()) if forward_rets else 0.0
    signals["forward_vs_peers"] = signals["forward_ret_pct"] - peer_med
    signals["forward_vs_nifty"] = (signals["forward_ret_pct"] - nifty_fwd
                                   if nifty_fwd is not None else signals["forward_ret_pct"])

    def _correct(row) -> Optional[bool]:
        phase = row["phase"]
        fwd_r = row.get("forward_vs_peers")
        if fwd_r is None or pd.isna(fwd_r) or phase == "Neutral":
            return None
        if phase in ("Leading", "Improving"):
            return bool(fwd_r > 0)   # sector beat its peers
        return bool(fwd_r < 0)       # Weakening/Lagging: underperformed peers

    signals["signal_correct"] = signals.apply(_correct, axis=1)

    # Sort: phase rank first, then forward return descending within each phase
    _rank = {"Leading": 0, "Improving": 1, "Neutral": 2, "Weakening": 3, "Lagging": 4}
    signals["_pr"] = signals["phase"].map(_rank).fillna(5)
    return (
        signals.sort_values(["_pr", "forward_ret_pct"], ascending=[True, False])
        .drop(columns="_pr")
        .reset_index(drop=True)
    )


def get_rotation_clock_accuracy(
    as_of_date: date,
    window_trading_days: int = 22,
    n_signals: int = 40,
    min_turnover_lacs: Optional[float] = None,
) -> dict:
    """
    MULTI-PERIOD validation of the rotation clock — a statistically meaningful
    hit-rate, NOT the single noisy 5-day snapshot.

    Walks back over up to n_signals NON-OVERLAPPING past signal dates (stepped by
    window_trading_days). At each, classifies phases AS OF that date (point-in-time)
    and measures the forward median sector return over the next window vs Nifty50.
    Aggregates per phase: sample count, hit-rate (inflow beat / outflow lagged the
    index), and average forward excess return.

    Returns {n_signals, n_predictions, overall_hit, inflow_outflow_spread,
             by_phase: {phase: {n, hit_rate, avg_excess}}}.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    dts = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data WHERE trade_date <= ? ORDER BY trade_date",
        [as_of_date],
    )
    if dts.empty:
        return {}
    all_dates = [pd.to_datetime(d).date() for d in dts["trade_date"]]
    n = len(all_dates)
    w = window_trading_days
    if n < w * 3:
        return {}

    # Non-overlapping signal indices: each needs `w` forward days available.
    sig_idx = list(range(n - 1 - w, w, -w))[:n_signals]
    if not sig_idx:
        return {}

    # Forward-return panel (robust median) spanning the whole backtest, one query.
    span_start = all_dates[min(sig_idx)]
    panel = query_dataframe("""
        SELECT s.sector, b.trade_date,
               MEDIAN(CASE WHEN b.prev_close > 0
                           THEN (b.close_price - b.prev_close) / b.prev_close * 100 END) AS ret
        FROM daily_data b JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE s.sector NOT IN ('ETF', 'Others') AND b.series IN ('EQ','SM','ST')
          AND b.turnover_lacs >= ? AND b.trade_date >= ? AND b.trade_date <= ?
        GROUP BY s.sector, b.trade_date
    """, [min_turnover_lacs, span_start, as_of_date])
    nif = query_dataframe("""
        SELECT trade_date, pct_chg AS nret FROM index_data
        WHERE index_name = 'Nifty 50' AND trade_date >= ? AND trade_date <= ?
    """, [span_start, as_of_date])
    if panel.empty:
        return {}
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.date
    piv = panel.pivot_table("ret", "trade_date", "sector")
    nser = (nif.assign(trade_date=pd.to_datetime(nif["trade_date"]).dt.date)
               .set_index("trade_date")["nret"]) if not nif.empty else None

    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "hit": 0, "exc": 0.0})
    n_pred = n_hit = 0
    infl_exc, outf_exc = [], []

    for i in sig_idx:
        sig_date = all_dates[i]
        fwd_dates = all_dates[i + 1: i + 1 + w]
        if len(fwd_dates) < max(3, w // 2):
            continue
        try:
            ph = get_sector_rotation_timeframe(sig_date, w, min_turnover_lacs)
        except Exception:
            continue
        if ph.empty:
            continue
        # Peer benchmark = cross-sectional median forward return (apples-to-apples
        # with the clock's cross-sectional classification).
        fwd_all = {s: float((1 + piv[s].reindex(fwd_dates) / 100).prod() - 1) * 100
                   for s in piv.columns}
        peer_med = float(np.nanmedian([v for v in fwd_all.values() if not pd.isna(v)]))
        for _, r in ph.iterrows():
            sec, phase = r["sector"], r["phase"]
            if phase == "Neutral" or sec not in fwd_all:
                continue
            sfwd = fwd_all[sec]
            if pd.isna(sfwd):
                continue
            exc = sfwd - peer_med
            correct = (exc > 0) if phase in ("Leading", "Improving") else (exc < 0)
            a = agg[phase]; a["n"] += 1; a["hit"] += int(correct); a["exc"] += exc
            n_pred += 1; n_hit += int(correct)
            (infl_exc if phase in ("Leading", "Improving") else outf_exc).append(exc)

    if n_pred == 0:
        return {}
    by_phase = {p: {"n": v["n"], "hit_rate": round(v["hit"] / v["n"] * 100, 0),
                    "avg_excess": round(v["exc"] / v["n"], 2)}
                for p, v in agg.items()}
    spread = ((sum(infl_exc) / len(infl_exc) if infl_exc else 0)
              - (sum(outf_exc) / len(outf_exc) if outf_exc else 0))
    return {
        "n_signals": len(sig_idx), "n_predictions": n_pred,
        "overall_hit": round(n_hit / n_pred * 100, 0),
        "inflow_outflow_spread": round(spread, 2),
        "by_phase": by_phase,
    }


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
            MEDIAN(CASE WHEN b.prev_close > 0
                        THEN (b.close_price - b.prev_close) / b.prev_close * 100 END)
                                                               AS wtd_daily_ret_pct
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
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
    # Cross-sectional center (typical sector), consistent with the timeframe clock.
    center_custom = float(df["cum_price_ret_pct"].median())
    df["sector_median_ret"] = round(center_custom, 2)

    def _phase_and_confidence_custom(row) -> tuple:
        sz   = row["slope_z"]
        pr   = row["cum_price_ret_pct"] - center_custom
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
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
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


# ── replay: "what did the Rotation Clock say on date X, and what happened?" ───
_CLOCK_ACTION = {
    "Leading":   ("BUY-LIKE",  True,  "delivery rising AND price leading peers"),
    "Improving": ("WATCH",     None,  "delivery rising but price still lagging"),
    "Neutral":   ("NO CALL",   None,  "no clear delivery direction"),
    "Weakening": ("AVOID",     False, "delivery falling while price still leads"),
    "Lagging":   ("AVOID",     False, "delivery falling and price lagging"),
}


def get_clock_replay(as_of_date: date, window_trading_days: int = 5,
                     min_turnover_lacs: Optional[float] = None,
                     today: Optional[date] = None) -> dict:
    """
    Re-run the Rotation Clock as it stood on `as_of_date` and score what followed.

    SCORED SYMMETRICALLY: a clock built on an N-session lookback is judged over
    the NEXT N sessions. "The 2-week clock, over the next 2 weeks." Any other
    pairing invites the category error the Forward-Tilt panel already guards
    against — scoring a short read over a long window.

    PHASES ARE NOT ALL CALLS, and treating them as one buy list would be wrong:
      Leading    the only buy-like phase (delivery rising, price leading)
      Improving  explicitly WATCH on this tab, and measured NEGATIVE at both ends
                 (-10.6%/yr at 1-2wk, t -2.64; -11.7%/yr at 11-12wk, t -2.54 in
                 scripts/backtest_tilt_vs_clock.py) — it is scored, and shown,
                 but never counted as a buy
      Weakening  } avoid-calls: LAGGING the benchmark is the win, same inversion
      Lagging    } the Forward-Tilt replay uses
      Neutral    no call — reported for completeness only

    Returns `horizon` (the N-session window) and `to_today` blocks, each carrying
    per-phase and per-sector returns plus the basket and Nifty benchmarks.
    Causality, snapping, open windows and missing sectors follow the same rules
    as get_tilt_replay; see that docstring for the constituent-look-ahead caveat,
    which applies here identically.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()
    W = max(2, int(window_trading_days))
    out: dict = {"ok": False, "as_of": as_of_date, "window": W}

    sess = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data ORDER BY trade_date", [])
    if sess.empty:
        out["error"] = "no sessions in the archive"; return out
    days = pd.to_datetime(sess["trade_date"]).dt.date.tolist()
    today = today or days[-1]
    prior = [d for d in days if d <= as_of_date]
    if not prior:
        out["error"] = f"no session on or before {as_of_date} (archive starts {days[0]})"
        return out
    anchor = prior[-1]
    out["anchor"] = anchor
    out["snapped"] = anchor != as_of_date
    if anchor >= today:
        out["error"] = "pick a date before the latest session — nothing has happened yet"
        return out

    try:
        clock = get_sector_rotation_timeframe(anchor, W, min_turnover_lacs)
    except Exception as exc:                                    # noqa: BLE001
        out["error"] = f"clock unavailable for {anchor}: {exc}"; return out
    if clock is None or clock.empty:
        out["error"] = (f"not enough history on {anchor} for a {W}-session clock")
        return out

    i = days.index(anchor)
    h_end = days[i + W] if i + W < len(days) else None
    out["sessions_elapsed"] = max(0, min(W, len(days) - 1 - i))
    out["sessions_remaining"] = max(0, W - out["sessions_elapsed"])
    out["horizon_end"] = h_end
    out["today"] = today

    # Forward sector returns MUST use the same definition the clock CLASSIFIES on,
    # or the panel scores a different quantity than it sorted. get_sector_rotation_
    # timeframe builds its price axis from the MEDIAN constituent return -- a
    # deliberate choice (see its SQL comment: turnover weighting uses
    # contemporaneous volume, which inflated the axis until almost every sector
    # read "above market"). Scoring with a turnover-weighted mean instead was
    # measurably inconsistent: rank correlation between the two definitions is
    # only 0.65-0.77, 6-7 of 24 sectors flip their above/below-basket verdict, and
    # over 65 sessions the two baskets differ by 11pp (+16.9% vs +5.9%).
    # So: `ret` is the MEDIAN (matches the clock, drives the tick marks), and
    # `ret_eq` is the equal-weight mean of the same stocks -- what an actual
    # basket would have earned -- reported alongside for investability.
    panel = query_dataframe(
        """
        WITH base AS (
            SELECT b.trade_date, s.sector, b.turnover_lacs,
                   CASE WHEN b.prev_close > 0
                        THEN (b.close_price - b.prev_close) / b.prev_close * 100 END AS r_raw,
                   LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol
                                              ORDER BY b.trade_date) AS w_lag
            FROM daily_data b
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ','SM','ST')
              AND s.sector IS NOT NULL AND s.sector NOT IN ('ETF','Others')
              AND b.trade_date > (?::date - 30) AND b.trade_date <= ?
        )
        SELECT sector, trade_date,
               MEDIAN(r_raw) AS ret,
               AVG(GREATEST(LEAST(r_raw, 25), -25)) AS ret_eq
        FROM base
        WHERE turnover_lacs >= ? AND w_lag IS NOT NULL AND trade_date > ?
        GROUP BY sector, trade_date
        """, [anchor, today, min_turnover_lacs, anchor])
    if panel.empty:
        out["error"] = "no forward sector data"; return out
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    wide = panel.pivot(index="trade_date", columns="sector", values="ret").sort_index()
    wide_eq = panel.pivot(index="trade_date", columns="sector", values="ret_eq").sort_index()

    nf = query_dataframe(
        "SELECT trade_date, close_val FROM index_data WHERE index_name='Nifty 50' "
        "AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date", [anchor, today])
    nf["trade_date"] = pd.to_datetime(nf["trade_date"])
    nser = nf.set_index("trade_date")["close_val"].astype(float)

    phases = {p: clock.loc[clock["phase"] == p, "sector"].tolist()
              for p in _CLOCK_ACTION}

    def block(upto: Optional[date], label: str) -> dict:
        if upto is None:
            return {"status": "OPEN", "label": label}
        seg = wide[wide.index <= pd.Timestamp(upto)]
        if seg.empty:
            return {"status": "NO_DATA", "label": label}
        rets = np.expm1(np.log1p(seg / 100.0).sum(axis=0)) * 100.0
        seg_eq = wide_eq[wide_eq.index <= pd.Timestamp(upto)]
        rets_eq = (np.expm1(np.log1p(seg_eq / 100.0).sum(axis=0)) * 100.0
                   if not seg_eq.empty else pd.Series(dtype=float))
        nseg = nser[nser.index <= pd.Timestamp(upto)]
        nif = float(nseg.iloc[-1] / nseg.iloc[0] - 1.0) * 100.0 if len(nseg) >= 2 else float("nan")
        basket = float(rets.mean())
        per_phase = {}
        for p, names in phases.items():
            have = [s for s in names if s in rets.index]
            per_phase[p] = {
                "n": len(names), "n_scored": len(have),
                "ret": float(rets[have].mean()) if have else float("nan"),
                "missing": [s for s in names if s not in rets.index],
            }
        return {"status": "DONE", "label": label, "end": upto,
                "sessions": int(len(seg)), "basket_abs": basket, "nifty_abs": nif,
                "basket_eq_abs": float(rets_eq.mean()) if len(rets_eq) else float("nan"),
                "per_phase": per_phase, "per_sector": rets, "per_sector_eq": rets_eq}

    out.update(ok=True, clock=clock, phases=phases,
               nifty_window=float(clock["nifty_return"].iloc[0])
               if "nifty_return" in clock.columns and len(clock) else float("nan"),
               horizon=block(h_end, f"over the next {W} sessions"),
               to_today=block(today, "from then until now"))
    return out
