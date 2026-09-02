"""
1–2 Week Forward Sector Tilt — the only sector call that survived deep validation.

WHAT IS VALIDATED — NOTHING, AT ANY HORIZON. Read _HORIZON_EVIDENCE below.
  The old headline here ("daily-IC t ≈ 9", "Monte-Carlo p < 0.002", "cost-robust
  0→40bps", "sub-period stable") came from a 371-day bull sample and a NAIVE
  t-statistic that ignored the overlap between forward windows. Re-measured on
  2,129 sessions (2018-2026), averaged over ALL h rebalance calendars rather than
  one arbitrary offset, and net of 25bps/side:
    • 1-2 wk is NEGATIVE (-1.4%/yr excess vs the equal-weight sector basket, 30%
      of calendars positive) and its breakeven cost is 21bps/side — below realistic
      retail cost. It is not tradeable. 3-4 wk is likewise ~0 with a 17bps breakeven.
    • 7-12 wk is weakly positive (+4.1 to +6.7%/yr, 93-98% of calendars positive,
      same sign in every era at 9-12wk) but the median per-phase t is 0.9-1.2, so it
      does NOT clear significance. It is a lean, not an edge.
    • "cost-robust 0→40bps" was false: cost sensitivity is the single largest driver
      of the result at every horizon.
  Delivery flow (dv5d) carries 15% of the composite weight and contributes NOTHING —
  removing it IMPROVES five of six horizons. It is retained only so the displayed
  column keeps its meaning; it should be dropped once the UI stops showing it.
  The one component that does pay is the sector-persistence gate (+3.9 to +6.2%/yr
  at every horizon — larger than the tilt's own edge). That, not the momentum
  ranking, is the real content of this module.
  F&O positioning is NOT usable at sector granularity (only ~4 sectors carry enough
  F&O names to aggregate) and is deliberately excluded here.

KNOWN, UNFIXED BIAS: v_sector_master has no as-of column and carries 8 dead names
  out of 1,045 over 8.6 years — it is a present-day snapshot applied to history.
  Every LEVEL figure computed from this panel is inflated (the mapped basket returns
  +15.2%/yr vs +10.6%/yr for the full liquid universe including delisted names; the
  gap decays 8.3 → 1.3 → 1.1 across the three eras, the signature of survivorship).
  The RELATIVE tilt is largely immune — the bias is common-mode and the edge does not
  correlate with deletion exposure — but nothing here may be quoted as an absolute
  return or benchmarked to Nifty until the master gains an as-of dimension.

REGIME BEHAVIOUR (real-bear calibrated on DCM's OWN sector data 2018–2026, incl the 2018
midcap crisis / 2020 COVID / 2022 bear — scripts/audit_tilt_realbears.py +
audit_suppression_adjudicate.py; supersedes the earlier stock-level read):
  • TRENDING_DOWN (Nifty px<EMA20<EMA50): the SECTOR tilt does NOT reliably invert. OW−UW is
    +1.7%/10d pooled (t+2.3) and +1.6% even when the market KEPT falling (t+0.5, insignificant).
    The earlier "inverts −0.8%/10d" was a STOCK-level (Tradebot F&O) result that does NOT
    transfer to the sector product. → overweights are KEPT, but size is REDUCED (×0.50): the
    relative signal is unreliable in downtrends AND a long-only book carries market beta.
  • REVERSAL (sharp pullback in an up-run): momentum-crash TAIL risk (rare) → small size ×0.40.
  • CHOPPY: overweight ≈ underweight (t−1.5) → conviction muted ×0.70.
  • TRENDING_UP / HIGH_VOL: edge intact → ×1.0 (high-vol here was up-vol; flagged for beta).
  • DIVERGENCE axis: a 1-2wk pop inside a 1-2mo downtrend (BULLTRAP) is the weakest forward
    state → extra ×0.70; a 1-2wk dip inside a 1-2mo uptrend (DIP_IN_UP) is the best entry.
  • The "smart money buys the crash" long (2-3mo) was tested and FAILED — beaten-down names
    mean-revert HARDER than resilient ones — so NO buy-the-crash signal is wired.

OUTPUT (decision-support, not a signal generator):
  get_forward_tilt(as_of_date, ...) -> (DataFrame per sector, regime meta dict)
  columns: sector, score, rank, tilt, rs_2w, rs_1w, dv5d, accum_breadth, deliv_slope,
           n_liq, thin, divergence, est_rel_bps, confidence
  tilt ∈ {OVERWEIGHT, NEUTRAL, UNDERWEIGHT, WATCH}. UNDERWEIGHT = reduce/avoid, NOT a
  short recommendation (a sector basket cannot be shorted cheaply). WATCH = heavy
  accumulation but momentum not yet turned (contrarian; momentum is the validated timer,
  so these are held out of the active tilt).

All factors use data <= as_of_date only; forward returns are never read live.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.base import get_min_turnover_filter
from src.analytics.sector_signal_v2 import (get_accum_breadth_history,
                                            get_robust_delivery_signals)
from src.data.repository import query_dataframe
from src.logging_setup import get_logger

log = get_logger(__name__)

# ── factor construction (mirrors the validated build_factors) ────────────────
_MOM_2W        = 10       # trading days for 2-week momentum / relative strength
_MOM_1W        = 5        # trading days for 1-week momentum
_DV_BASE       = 100      # delivery-flow baseline AT THE 1-2wk HORIZON (see _dv_windows)
_DV_FLOW       = 5        # short delivery-flow window AT THE 1-2wk HORIZON
_MIN_HIST      = 12       # min sector daily rows to compute momentum
_MIN_SECTORS   = 8        # need a real cross-section to rank
_MIN_LIQ_NAMES = 5        # below this a sector is "thin" (noisy rs/breadth)

# ── composite weights (IC-proportional; momentum dominates) ──────────────────
_W_RS2, _W_RS1, _W_DV5 = 0.60, 0.25, 0.15

# ── tilt thresholds with hysteresis headroom ─────────────────────────────────
_OW_RANK       = 0.75     # rank >= this  → OVERWEIGHT
_UW_RANK       = 0.25     # rank <= this  → UNDERWEIGHT
# WATCH: contrarian accumulation — strong breadth but momentum not yet turned
_WATCH_BREADTH = 0.55
_WATCH_RS_MAX  = 0.35     # ... while momentum rank is still weak

# ── expected relative return map (from the validated ~1.9%/10d tercile spread) ─
_REL_SLOPE_BPS = 290.0    # bps of 10d relative return per unit of (rank-0.5)

# ── regime read (Nifty) — now a DATA-BACKED reliability lever, not just context ──
# The old code treated the market backdrop as pure context (mult ~1.0 everywhere) on the
# grounds that the relative tilt was "regime-independent". That was a BULL-ONLY-SAMPLE
# artifact. A 4-year multi-regime OOS stress test (Tradebot 211-name F&O panel 2022–26,
# scripts/audit_forward_tilt_regimes.py) OVERTURNS it for genuine downtrends:
#   • TRENDING_DOWN (Nifty px<EMA20<EMA50): the OVERWEIGHT (high relative-momentum) basket
#     UNDERPERFORMS the underweight by −0.81%/10d (t−2.1), −1.77%/40d, −2.80%/60d — the
#     momentum tilt INVERTS. Quintile Q5−Q1 = −0.84% (monotone down). Negative in BOTH
#     halves and ALL FOUR years (2023 −0.58 / 2024 −0.83 / 2025 −0.60 / 2026 −1.26) — never
#     positive. So chasing leaders in a downtrend is measurably the wrong side.
#   • CHOPPY: OW ≈ UW (OW−UW −0.10%/10d, t−1.5) — the tilt adds ~nothing.
#   • TRENDING_UP / HIGH_VOL: edge intact (OW−UW +0.47 / +0.59%). (High-vol here was up-vol.)
# So downtrend/reversal now HARD-suppress the overweight call and flip posture to defensive;
# chop is muted. The "smart-money accumulation buys the crash" long (2-3mo) was tested with a
# price proxy and FAILED — beaten-down names mean-revert HARDER than resilient ones, so no
# buy-the-crash signal is wired (delivery-breadth accumulation is DCM-bull-only, untestable OOS).
_VOL_HI_PCT    = 0.80     # realized-vol percentile above this = high-vol regime
_PULLBACK_5D   = -3.0     # Nifty 5d return below this % after an up-run = reversal caution
_MED_TREND_WIN = 40       # ~2-month (trading-day) window for the medium-term trend axis
_EMA_SLOPE_WIN = 10       # bars over which the 20-EMA slope is read (medium direction)
_REGIME_CONFIRM = 3       # a persistent regime must hold this many days before it switches
                          # (8yr audit: raw EMA-stack whipsaws — median run 3d, 42% flip back;
                          #  confirmation-gating debounces it so the tilt size stops flip-flopping)
# trend-quality (Kaufman efficiency ratio, 20d) — 8yr audit: within an uptrend, strong-ER
# days hit 77% (fwd10) vs choppy-ER 55% (monotone 77/62/55, +0.63%/yr, 5/7 yrs +). A real
# trend persists; a choppy 'uptrend' (price>EMAs but grinding) doesn't. Gentle size nudge only.
_ER_STRONG     = 0.50     # ER20 >= this = strong clean trend → full size
_ER_CHOPPY     = 0.30     # ER20 <  this = choppy/grinding → trim size
_TQ_MULT       = {"strong": 1.00, "moderate": 0.90, "choppy": 0.80}  # applied in ACT regimes only
# data-backed confidence multipliers per regime (OOS 4yr, above)
_MULT_UP       = 1.00     # edge intact
_MULT_HIVOL    = 1.00     # edge intact in-sample (up-vol) — flagged, not penalised
_MULT_CHOP     = 0.70     # tilt adds ~nothing (OW≈UW)
_MULT_DOWN     = 0.50     # sector tilt UNRELIABLE (not inverting) + long-only beta → reduce, keep OW
_MULT_REVERSAL = 0.40     # momentum-crash tail risk (rare) → small size
_MULT_BULLTRAP = 0.70     # 1-2wk pop inside a 1-2mo downtrend — reduce size (overlay)

# ── sector momentum-persistence gate (the real reliability lever) ────────────
# Validated (walk-forward, causal): sectors differ structurally — industrials/materials
# TREND (a high rank keeps outperforming) while consumer/rate-sensitive/financials REVERT
# (a high rank fades). OOS-persistent (H1→H2 Spearman +0.60). Demoting OVERWEIGHT calls in
# historically-reverting sectors lifts OW accuracy ~56% → ~60%. Signal = each sector's
# trailing mean forward RELATIVE edge (expanding, realized ≤ as_of − _PERS_FWD days).
_PERS_LOOKBACK_CAL = 620  # ~2 trading years of history for the persistence estimate
_PERS_FWD          = 10   # forward horizon the persistence is measured over (matches tilt)
_PERS_MIN_OBS      = 30   # min realized forward windows before a sector's persistence is used


def _compound(pct_series: pd.Series, n: int) -> pd.Series:
    """Compounded return over the trailing n rows (log1p sum), aligned to the last row."""
    lr = np.log1p(pct_series / 100.0)
    cr = lr.cumsum()
    return np.expm1(cr - cr.shift(n)) * 100.0


def _dv_windows(h: int) -> tuple[int, int]:
    """Delivery-flow (flow, baseline) windows for horizon `h`, in trading sessions.

    The RS legs scale with the horizon (long=h, short=h/2) but this factor did NOT
    — it stayed pinned at 5/100 at every horizon, so an 11-12wk call was ranked
    partly on the last FIVE sessions of delivery. Measured (scripts/tilt_dv_scaling.py,
    tilt_dv_design.py; 2018-2026, 24 sectors): a sector in the top quartile of the 5d
    flow is still there only 41.9% of the time 10 sessions later and ~29% after 60
    (random 25%), i.e. the state has a ~2-3 week half-life and had fully decayed over
    most of a long hold.

    BOTH windows scale, and that is the point. The baseline CONTAINS the flow window,
    so a surge inflates its own denominator; the shipped overlap is 5/100 = 5%.
    Scaling the flow alone drives it to 30/100 = 30% at 11-12wk and measurably HURTS
    (5-6wk OW excess +0.507pp vs +0.537 shipped). Scaling both holds the overlap at a
    constant 5% and improves the OVERWEIGHT bucket at every longer horizon:
        horizon    shipped 5/100      flow=h/2, base=10h
        1-2 wk     +0.221pp t+2.96    +0.221pp t+2.96   (identical by construction)
        3-4 wk     +0.312pp t+1.95    +0.402pp t+2.40
        5-6 wk     +0.537pp t+2.20    +0.696pp t+2.69
        7-8 wk     +0.886pp t+2.64    +1.120pp t+2.84
        9-10 wk    +1.412pp t+3.23    +1.691pp t+3.12
        11-12 wk   +1.555pp t+2.89    +2.016pp t+2.74

    THIS IS A CONSISTENCY FIX, NOT AN EDGE. The gaps above are ~1 SE and were found
    inside a search over ~54 design x horizon cells; they would not survive a reality
    check as a discovered edge. The case for scaling is a priori (constant overlap,
    horizon-matched state), and the numbers merely fail to contradict it. At 3-4wk it
    moves ~5%/yr gross against a measured 5.1pp/yr cost drag: negative -> roughly
    breakeven, still not tradeable. See _HORIZON_EVIDENCE.

    h=10 MUST return (5, 100) exactly so the validated 1-2wk build is untouched;
    tests/test_forward_tilt_dv.py pins that.
    """
    return max(2, h // 2), 10 * h


def _panel_lookback_cal(h: int) -> int:
    """Calendar days `_load_sector_panel` must cover to satisfy `_dv_windows(h)`.

    The baseline needs 10*h TRADING sessions; ~1.45 calendar days per session, plus a
    60-day buffer for holidays. Floored at the historical 275 so the 1-2wk default
    scans exactly what it always did — no perf regression on the shipped horizon.
    """
    return max(275, int(_dv_windows(h)[1] * 1.45) + 60)


def _load_sector_panel(as_of_date: date, min_turnover_lacs: float,
                       lookback_cal: int = 275) -> pd.DataFrame:
    return query_dataframe(
        """
        WITH base AS (
            SELECT s.sector, b.trade_date, b.turnover_lacs, b.deliv_per,
                   -- raw move, kept so a corporate action can be DROPPED rather than
                   -- clipped. Winsorizing a 1:1 bonus (raw ~ -50%) turns it into a
                   -- -25% print that is indistinguishable from a real crash and feeds
                   -- straight into the sector's momentum. Measured 2018-2026: 741
                   -- stock-days at |raw| >= 40%, of which 444 are split/bonus-shaped.
                   -- NSE price bands make a genuine >40% cash move impossible, so the
                   -- cut is a corporate-action filter, not a return filter.
                   (b.close_price - b.prev_close)
                       / NULLIF(b.prev_close, 0) * 100                        AS raw_r,
                   -- winsorize what remains: one uncapped print (illiquid spike)
                   -- otherwise distorts the whole sector's momentum
                   LEAST(GREATEST((b.close_price - b.prev_close)
                                  / NULLIF(b.prev_close, 0) * 100, -25), 25)  AS r,
                   -- LAGGED weight: a stock's SAME-DAY turnover explodes on the day it jumps,
                   -- so weighting by it correlates the weight with the return being weighted
                   -- (+0.717%/day of fake drift vs +0.025% lagged, measured 2026-07-31). The
                   -- prior session's turnover is knowable at entry and is what a real basket
                   -- would hold. Data-correctness fix only — the tilt LOGIC is untouched.
                   LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol
                                              ORDER BY b.trade_date)          AS w_lag
            FROM daily_data b
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND s.sector NOT IN ('ETF', 'Others')
              AND b.trade_date > (?::date - ?)
              AND b.trade_date <= ?
        )
        SELECT sector, trade_date,
               SUM(turnover_lacs * deliv_per / 100.0) / 100.0                 AS daily_dv_cr,
               SUM(w_lag * r) / NULLIF(SUM(CASE WHEN r IS NOT NULL
                                           THEN w_lag END), 0)                AS wtd_ret_pct
        FROM base
        -- LAGGED filter, not same-day. The weight was already lagged for exactly this
        -- reason but the FILTER was left on same-day turnover, which admits a stock
        -- BECAUSE it moved that day. Measured 2018-2026: the mean daily return of the
        -- same-day-filtered universe is +0.223%/day at the 1 Cr floor and +0.507%/day
        -- at 25 Cr, versus +0.072%/day and +0.076%/day when the same floor is applied
        -- to the PRIOR session. That gap is pure outcome-conditioning. The lagged
        -- weight was masking it here; it is not masked in any equal-weighted variant.
        WHERE w_lag >= ? AND ABS(raw_r) < 40
          AND trade_date > (?::date - ?)
        GROUP BY sector, trade_date
        ORDER BY sector, trade_date
        """,
        [as_of_date, lookback_cal + 15, as_of_date,
         min_turnover_lacs, as_of_date, lookback_cal],
    )


def _load_nifty(as_of_date: date) -> pd.DataFrame:
    try:
        df = query_dataframe(
            "SELECT trade_date, close_val, pct_chg FROM index_data "
            "WHERE index_name = 'Nifty 50' AND trade_date <= ? ORDER BY trade_date",
            [as_of_date],
        )
    except Exception as exc:                                  # noqa: BLE001
        log.warning("nifty load failed (%s); relative strength falls back to absolute", exc)
        return pd.DataFrame()
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # `pct_chg` is preferred but must be COMPLETE. The old guard was `.notna().any()`,
    # so a single null anywhere still selected the column wholesale — and a NaN inside
    # the trailing window makes _compound() return NaN, which makes n_2w NaN, which
    # makes EVERY sector's rs_2w NaN, which makes every rank NaN, which silently
    # renders the entire board NEUTRAL with no error shown. Fall back per-row instead.
    px_chg = df["close_val"].astype(float).pct_change() * 100
    nret = df["pct_chg"].astype(float) if "pct_chg" in df.columns else px_chg
    df["nret"] = nret.fillna(px_chg)
    if df["nret"].iloc[1:].isna().any():
        log.warning("Nifty return series still has %d gaps after fallback; "
                    "relative strength will be degraded on those windows",
                    int(df["nret"].iloc[1:].isna().sum()))
    return df


def _liquid_name_counts(as_of_date: date, min_turnover_lacs: float) -> pd.Series:
    # Liquidity is judged on the PRIOR session, like every other universe rule here.
    # A same-day floor counts a stock as "liquid" on the day it spiked, which both
    # inflates n_liq on volatile days and lets the thin-flag flicker with the news.
    df = query_dataframe(
        """
        WITH liq AS (
            SELECT b.symbol, b.trade_date,
                   LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol
                                              ORDER BY b.trade_date) AS w_lag
            FROM daily_data b
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND b.trade_date > (?::date - 15) AND b.trade_date <= ?
        )
        SELECT s.sector, COUNT(*) AS n_liq
        FROM liq
        INNER JOIN v_sector_master s ON liq.symbol = s.symbol
        WHERE liq.trade_date = ? AND s.sector NOT IN ('ETF', 'Others')
          AND liq.w_lag >= ?
        GROUP BY s.sector
        """,
        [as_of_date, as_of_date, as_of_date, min_turnover_lacs],
    )
    return df.set_index("sector")["n_liq"] if not df.empty else pd.Series(dtype=int)


def _confirmed_base_state(close: pd.Series, ema20_s: pd.Series, ema50_s: pd.Series,
                          vol20: pd.Series, k: int = _REGIME_CONFIRM, look: int = 30) -> str:
    """Debounced persistent regime (UP/DOWN/HIGH_VOL/CHOP) for the last day — kills whipsaw.

    Classifies each of the last `look` days by the raw EMA-stack + vol rule (causal), then
    only accepts a regime SWITCH after the new label has held `k` consecutive days; transient
    <k-day flips keep the prior confirmed regime. REVERSAL is handled separately (immediate).
    """
    n = len(close)
    lo = max(60, n - look)
    labels = []
    for i in range(lo, n):
        px, e20, e50 = close.iloc[i], ema20_s.iloc[i], ema50_s.iloc[i]
        vp = float((vol20.iloc[:i + 1] <= vol20.iloc[i]).mean()) if np.isfinite(vol20.iloc[i]) else np.nan
        if px < e20 < e50:                                   labels.append("DOWN")
        elif np.isfinite(vp) and vp >= _VOL_HI_PCT:          labels.append("HIGH_VOL")
        elif px > e20 > e50:                                 labels.append("UP")
        else:                                                labels.append("CHOP")
    if not labels:
        return "CHOP"
    last = pending = labels[0]; cnt = 0
    for v in labels:
        if v == pending: cnt += 1
        else:            pending, cnt = v, 1
        if cnt >= k:     last = pending
    return last


def _market_regime(nifty: pd.DataFrame) -> dict:
    """Nifty trend/vol read → DATA-BACKED reliability lever + posture (OOS 4yr calibrated).

    Returns, besides the label: a confidence multiplier that HARD-suppresses the overweight
    call where the tilt was measured to invert (downtrend/reversal) or add nothing (chop);
    a medium-term (1-2mo) trend axis and a short-vs-medium DIVERGENCE state (BULLTRAP =
    1-2wk up inside a 1-2mo downtrend → the weakest forward state, size down); a `momentum_inverts`
    flag the engine uses to demote overweights; and a plain-language `posture`.
    """
    default = dict(state="UNKNOWN", vol_pct=float("nan"), ret_5d=float("nan"),
                   confidence_mult=1.0, momentum_inverts=False, med_trend="UNKNOWN",
                   divergence="n/a", verdict="SELECTIVE", size_hint=0.5,
                   trend_strength="moderate", er20=float("nan"),
                   action="Half size — market context unavailable, treat as low conviction.",
                   posture="Market context unavailable.",
                   banner="Regime unknown — market context unavailable.")
    if nifty.empty or len(nifty) < 60:
        return default
    nf = nifty.sort_values("trade_date").reset_index(drop=True)
    close = nf["close_val"].astype(float)
    ret = nf["nret"].astype(float) / 100.0
    ema20_s = close.ewm(span=20, adjust=False).mean()
    ema50_s = close.ewm(span=50, adjust=False).mean()
    ema20 = ema20_s.iloc[-1]; ema50 = ema50_s.iloc[-1]
    px = close.iloc[-1]
    sma200_s = close.rolling(200).mean()
    # Long-term structure. The regime label itself is an EMA20/EMA50 read and is
    # deliberately NOT gated on this — measured 2018-2026, a short-term uptrend
    # BELOW the 200-DMA is the tilt's STRONGEST state (OW-UW +3.35%/10d, t 2.87,
    # vs +0.57% t 2.16 above it), because dispersion is high after a drawdown and
    # leadership rotates hard. Carried here only so the UI can explain why the
    # mood can read risk-on while the long-term tape is still broken.
    vol20 = ret.rolling(20).std()
    vol_pct = float((vol20 <= vol20.iloc[-1]).mean()) if vol20.notna().sum() > 20 else float("nan")
    ret_5d = float(_compound(nf["nret"], 5).iloc[-1])
    ret_20d = float(_compound(nf["nret"], 20).iloc[-1])
    ret_med = float(_compound(nf["nret"], _MED_TREND_WIN).iloc[-1])   # ~2-month trend
    ema_slope = float(ema20_s.iloc[-1] - ema20_s.iloc[-1 - _EMA_SLOPE_WIN]) if len(ema20_s) > _EMA_SLOPE_WIN else 0.0

    # trend quality (Kaufman efficiency ratio 20d): |net move| / path length. High = clean trend.
    diff20 = float(close.diff().abs().iloc[-20:].sum())
    er20 = float(abs(close.iloc[-1] - close.iloc[-21]) / diff20) if len(close) > 21 and diff20 > 0 else float("nan")
    trend_strength = ("strong" if np.isfinite(er20) and er20 >= _ER_STRONG else
                      "choppy" if np.isfinite(er20) and er20 < _ER_CHOPPY else "moderate")

    # confirmation-gated persistent regime (debounced — 8yr audit: raw stack whipsaws 42%)
    base = _confirmed_base_state(close, ema20_s, ema50_s, vol20)
    up_trend = base == "UP"
    dn_trend = base == "DOWN"
    high_vol = base == "HIGH_VOL"
    reversal = ret_5d <= _PULLBACK_5D and ret_20d > 0     # sharp pullback inside an up-run (immediate)

    # ── medium-term (1-2 month) trend axis + short-vs-medium divergence ──────────
    med_up = ret_med > 0 and ema_slope > 0
    med_dn = ret_med < 0 and ema_slope < 0
    short_up = ret_5d > 0
    med_trend = "UP" if med_up else "DOWN" if med_dn else "FLAT"
    if short_up and med_dn:      divergence = "BULLTRAP"    # 1-2wk up, 1-2mo down (weakest)
    elif (not short_up) and med_up: divergence = "DIP_IN_UP"  # 1-2wk down, 1-2mo up (best dip)
    elif short_up and med_up:    divergence = "ALIGNED_UP"
    elif (not short_up) and med_dn: divergence = "ALIGNED_DN"
    else:                        divergence = "MIXED"

    # ── primary regime → the POSTURE MATRIX (verdict · size · mult · action) ──────
    # Sector-level, real-bear-calibrated (scripts/audit_suppression_adjudicate.py, DCM
    # 2018-2026 incl COVID/2022). Correction to the earlier stock-level read: at the SECTOR
    # level the tilt does NOT significantly invert in downtrends — even when the market kept
    # falling, OW−UW was +1.6% (t+0.5, insignificant); pooled +1.7%. So downtrend/reversal
    # now REDUCE SIZE (unreliable relative signal + long-only market beta) rather than hard-
    # suppress overweights. verdict ∈ {ACT, SELECTIVE, STAND-ASIDE}; size_hint ∈ [0,1].
    if reversal:
        state, mult, inv = "REVERSAL", _MULT_REVERSAL, False
        verdict, size = "SELECTIVE", 0.3
        posture = "Sharp pullback — momentum-crash tail risk + market beta. Small size, top names only."
        banner = (f"⚠ Sharp pullback (Nifty {ret_5d:+.1f}% / 5d). Momentum-crash tail risk and a "
                  f"long-only book carries beta here — small size, top-ranked names only.")
    elif dn_trend:
        state, mult, inv = "TRENDING_DOWN", _MULT_DOWN, False
        verdict, size = "SELECTIVE", 0.4
        posture = "Downtrend — the sector tilt is unreliable here + long-only carries market beta. Reduced size, top names."
        banner = ("Nifty below 20/50 EMA (downtrend). Real-bear-measured (2018-2026 incl COVID/2022): "
                  "the sector overweight does NOT reliably underperform here (OW−UW ~+1.6%, not "
                  "significant) — earlier 'inverts' was a stock-level read that doesn't transfer. "
                  "But the relative signal is unreliable and a long-only book carries market beta, "
                  "so size DOWN and stick to top-ranked names — don't deploy full.")
    elif high_vol:
        state, mult, inv = "HIGH_VOL", _MULT_HIVOL, False
        verdict, size = "ACT", 0.75           # edge intact but big beta swings → size down for vol
        posture = "Tilt live, but a long-only book carries elevated market beta — size down for vol."
        banner = (f"Realized vol in the top {(1-vol_pct)*100:.0f}%. Relative tilt held up in-sample; "
                  f"a long-only book still carries market beta — size for the swing.")
    elif up_trend:
        state, mult, inv = "TRENDING_UP", _MULT_UP, False
        verdict, size = "ACT", 1.0
        posture = "Tilt active — favourable backdrop. Overweights are live; rotate into leaders."
        banner = "Nifty in a clean uptrend — favourable backdrop for a long-only sector tilt."
    else:
        state, mult, inv = "CHOPPY", _MULT_CHOP, False
        verdict, size = "SELECTIVE", 0.5
        posture = "Tilt muted — leaders ≈ laggards in chop. Half size, top names only; wait for a trend."
        banner = ("Nifty rangebound (mixed EMAs). OOS-measured: overweight ≈ underweight in chop — "
                  "the tilt has little to add; conviction is muted.")

    # ── trend-quality size nudge (ACT regimes only): a choppy 'uptrend' persists worse ──
    if verdict == "ACT":
        size *= _TQ_MULT[trend_strength]
        if trend_strength == "choppy":
            banner += "  (Choppy uptrend — grinding, not a clean trend; size trimmed.)"
        elif trend_strength == "strong":
            banner += "  (Strong clean trend — full conviction.)"

    # ── divergence overlays: downgrade size only; never upgrade a STAND-ASIDE ─────
    if divergence == "BULLTRAP" and verdict != "STAND-ASIDE":
        mult *= _MULT_BULLTRAP
        size = min(size, 0.5)
        if verdict == "ACT":
            verdict = "SELECTIVE"
        banner += ("  ⚠ Bull-trap: 1-2wk bounce inside a 1-2mo DOWNTREND (weakest measured forward "
                   "state) — reduce size, don't add on this pop.")
        posture = "Bull-trap (1-2wk up, 1-2mo down) — " + posture
    elif divergence == "ALIGNED_DN" and verdict != "ACT":
        # both short- and medium-term down → trim size further (not a hard sideline: sector
        # momentum does not significantly invert even in sustained falls), keep top names.
        size = min(size, 0.3); mult = min(mult, _MULT_DOWN)
        posture = "Short- and medium-term both down — extra caution, smaller size. " + posture
    elif divergence == "DIP_IN_UP" and verdict == "ACT":
        banner += "  ✓ 1-2wk dip inside a 1-2mo uptrend — historically the best entry timing."

    _ACTION = {
        "ACT":         "Trade the tilt — overweights are live. Rotate into the leaders below.",
        "SELECTIVE":   "Reduced size, top-ranked names only — the edge is thin/unreliable here.",
        "STAND-ASIDE": "No long rotation — preserve capital.",
    }
    return dict(state=state, vol_pct=vol_pct, ret_5d=ret_5d, ret_med=ret_med,
                med_trend=med_trend, divergence=divergence, momentum_inverts=bool(inv),
                confidence_mult=float(mult), verdict=verdict, size_hint=round(float(size), 2),
                above_200=(bool(px > sma200_s.iloc[-1])
                           if len(close) >= 200 and np.isfinite(sma200_s.iloc[-1]) else None),
                pct_vs_200=(float(px / sma200_s.iloc[-1] - 1.0) * 100.0
                            if len(close) >= 200 and np.isfinite(sma200_s.iloc[-1]) else float('nan')),
                trend_strength=trend_strength, er20=er20,
                action=_ACTION[verdict], posture=posture, banner=banner)


def _sector_persistence(as_of_date: date, min_turnover_lacs: float,
                        fwd_days: int = _PERS_FWD, with_history: bool = False):
    """Per-sector momentum-persistence: trailing mean forward RELATIVE edge (causal).

    For every past day with a realized 10-day forward return (date ≤ as_of − 10 trading
    days), edge = sector's fwd-10 return minus the cross-sectional median sector fwd-10.
    A sector's persistence = the expanding mean of that edge. >0 ⇒ high ranks historically
    kept outperforming (trend-follow, trust the overweight); <0 ⇒ they faded (mean-revert,
    demote the overweight). No lookahead — forward returns beyond as_of are NaN by
    construction.
    """
    empty = pd.DataFrame(columns=["sector", "persistence", "pers_n"])
    _none = (empty, pd.DataFrame()) if with_history else empty
    panel = query_dataframe(
        """
        WITH base AS (
            SELECT s.sector, b.trade_date, b.turnover_lacs,
                   (b.close_price - b.prev_close)
                       / NULLIF(b.prev_close, 0) * 100                        AS raw_r,
                   LEAST(GREATEST((b.close_price - b.prev_close)
                                  / NULLIF(b.prev_close, 0) * 100, -25), 25)  AS r,
                   LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol
                                              ORDER BY b.trade_date)          AS w_lag
            FROM daily_data b
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND s.sector NOT IN ('ETF', 'Others')
              AND b.trade_date > (?::date - ? - 15)
              AND b.trade_date <= ?
        )
        SELECT sector, trade_date,
               SUM(w_lag * r) / NULLIF(SUM(CASE WHEN r IS NOT NULL
                                           THEN w_lag END), 0)                AS wtd_ret_pct
        FROM base
        -- lagged filter + corporate-action drop, matching _load_sector_panel. The
        -- gate is measured on this panel, so it must use the same universe rule or
        -- it demotes sectors on a different construction than the one displayed.
        WHERE w_lag >= ? AND ABS(raw_r) < 40
          AND trade_date > (?::date - ?)
        GROUP BY sector, trade_date
        ORDER BY trade_date
        """,
        [as_of_date, _PERS_LOOKBACK_CAL, as_of_date,
         min_turnover_lacs, as_of_date, _PERS_LOOKBACK_CAL],
    )
    if panel.empty:
        return _none
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    ret = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
    if len(ret) < fwd_days + _PERS_MIN_OBS:
        return _none
    # forward compounded return per sector over the SELECTED horizon (NaN for the
    # last fwd_days rows → causal). The persistence gate must be measured over the
    # same window the user is being shown, or it demotes sectors on the wrong test.
    cr = np.log1p(ret / 100.0).cumsum()
    fwd = (np.expm1(cr.shift(-fwd_days) - cr) * 100.0)
    edge = fwd.sub(fwd.median(axis=1), axis=0)            # vs cross-sectional median sector
    out = pd.DataFrame({
        "persistence": edge.mean(axis=0, skipna=True),
        "pers_n": edge.notna().sum(axis=0),
    })
    out = out[out["pers_n"] >= _PERS_MIN_OBS].reset_index().rename(columns={"index": "sector"})
    if "sector" not in out.columns:
        out = out.rename(columns={out.columns[0]: "sector"})
    # PER-DATE history of the same statistic, for the tenure badge. At row t only
    # forward windows that CLOSED on or before t may be used, which is exactly
    # edge.shift(fwd_days) expanded to date. At the final row this reduces to the
    # identical set of observations as `out` above, so the history's last value and
    # the live gate agree by construction (verified in tests).
    hist = edge.shift(fwd_days).expanding(min_periods=_PERS_MIN_OBS).mean()
    return (out, hist) if with_history else out


# ── tenure: "how many sessions has this sector held this call?" ──────────────
# COSTS NOTHING EXTRA. _load_sector_panel already pulls 186-628 sessions (it is
# horizon-scaled — see _panel_lookback_cal) and
# _sector_persistence ~620 calendar days; both were being reduced to a single row
# and the rest discarded. Re-ranking those dates is pure pandas on data already in
# memory, so tenure adds no query and no round-trip.
#
# WHAT IT REPRODUCES, AND WHAT IT DOES NOT. The rank cut and the persistence gate
# are rebuilt exactly (same panel, same weights, same thresholds, all causal — a
# date's rank uses only that date's trailing windows). The WATCH and `thin`
# overlays are NOT rebuilt: WATCH needs the per-stock delivery panel and `thin`
# needs per-date liquidity counts, both of which WOULD need extra queries.
#
# SCOPE — OVERWEIGHT ONLY, and that limit is load-bearing. WATCH is defined on
# rank <= 0.35, so it can NEVER mask an OVERWEIGHT (cut 0.75) — an OW streak is
# therefore exactly reproducible. It CAN mask a NEUTRAL or an UNDERWEIGHT
# (rank <= 0.25 sits inside the WATCH band), and validation against 16 real
# per-date engine runs found precisely that: all 5 OVERWEIGHT streaks matched, while
# 4 of the NEUTRAL/UNDERWEIGHT rows were wrong because WATCH intervened mid-window
# WITHOUT changing today's label — so an "agrees on as_of" guard does not catch it.
# Rather than print a number that is wrong a third of the time on those buckets, the
# badge is restricted to the buy list, which is what it is for.
#
# `thin` IS modelled (it can demote an OVERWEIGHT), via one cheap grouped count over
# the lookback window — see _liquid_name_counts_hist.
#
# HONEST READING: this answers "how long would TODAY's engine say this sector has
# held this call", not "what did the screen show you each day". The two differ only
# if the data was later revised or the logic changed. A logged daily snapshot would
# answer the second question, needs a nightly job, and cannot survive a missed run —
# this cannot go stale and cannot develop holes.
_TENURE_LOOKBACK = 90     # sessions walked back; longer streaks render as "90+"


def _liquid_name_counts_hist(as_of_date: date, min_turnover_lacs: float,
                             lookback: int = _TENURE_LOOKBACK) -> pd.DataFrame:
    """date × sector liquid-name counts over the lookback, so `thin` is exact."""
    df = query_dataframe(
        """
        WITH liq AS (
            SELECT b.symbol, b.trade_date,
                   LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol
                                              ORDER BY b.trade_date) AS w_lag
            FROM daily_data b
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND b.trade_date > (?::date - ? - 20) AND b.trade_date <= ?
        )
        SELECT s.sector, liq.trade_date, COUNT(*) AS n_liq
        FROM liq
        INNER JOIN v_sector_master s ON liq.symbol = s.symbol
        WHERE s.sector NOT IN ('ETF', 'Others') AND liq.w_lag >= ?
        GROUP BY s.sector, liq.trade_date
        """,
        [as_of_date, int(lookback * 1.6), as_of_date, min_turnover_lacs],
    )
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot_table("n_liq", "trade_date", "sector").sort_index()


def _tilt_history(panel: pd.DataFrame, nifty: pd.DataFrame, L: int, S: int,
                  pers_hist: Optional[pd.DataFrame],
                  nliq_hist: Optional[pd.DataFrame] = None,
                  lookback: int = _TENURE_LOOKBACK,
                  breadth_hist: Optional[pd.DataFrame] = None,
                  dv_flow: int = _DV_FLOW,
                  dv_base: int = _DV_BASE) -> pd.DataFrame:
    """date × sector tilt label over the trailing `lookback` sessions.

    With `breadth_hist` supplied the WATCH overlay is applied too, which is what
    makes the UNDERWEIGHT band reproducible (WATCH sits inside it).
    """
    ret = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
    dvc = panel.pivot_table("daily_dv_cr", "trade_date", "sector").sort_index()
    if ret.empty or len(ret) < L + 5:
        return pd.DataFrame()

    lg = np.log1p(ret / 100.0)
    def _tr(n): return np.expm1(lg.rolling(n).sum()) * 100.0

    rsL, rsS = _tr(L), _tr(S)
    if not nifty.empty:
        nser = (nifty.drop_duplicates("trade_date").set_index("trade_date")["nret"]
                .astype(float).reindex(ret.index))
        nlg = np.log1p(nser.fillna(0.0) / 100.0)
        rsL = rsL.sub(np.expm1(nlg.rolling(L).sum()) * 100.0, axis=0)
        rsS = rsS.sub(np.expm1(nlg.rolling(S).sum()) * 100.0, axis=0)

    # dv windows are passed in, NOT read from the module constants: they scale with
    # the horizon (see _dv_windows) and this history must stay in lockstep with the
    # live ranking in get_forward_tilt, or the tenure badge counts a streak of a
    # different signal than the one on the card.
    dv5 = dvc.rolling(dv_flow).mean() / dvc.shift(1).rolling(dv_base).mean()
    def _rk(d): return d.rank(axis=1, pct=True)
    rank = _rk(_W_RS2 * _rk(rsL) + _W_RS1 * _rk(rsS) + _W_DV5 * _rk(dv5))

    lab = pd.DataFrame(
        np.where(rank.isna(), None,
                 np.where(rank >= _OW_RANK, "OVERWEIGHT",
                          np.where(rank <= _UW_RANK, "UNDERWEIGHT", "NEUTRAL"))),
        index=rank.index, columns=rank.columns)

    # thin gate: too few liquid names that day → cannot be a confident overweight.
    # Applied BEFORE persistence, matching the order in get_forward_tilt.
    if nliq_hist is not None and not nliq_hist.empty:
        thin = (nliq_hist.reindex(index=rank.index, columns=rank.columns)
                < _MIN_LIQ_NAMES).fillna(False)
        lab = lab.mask((lab == "OVERWEIGHT") & thin, "NEUTRAL")

    # persistence gate: demote OVERWEIGHT where the sector was historically reverting
    # AS OF THAT DATE (NaN ⇒ unknown ⇒ keep, matching the live `revert` semantics)
    if pers_hist is not None and not pers_hist.empty:
        rev = (pers_hist.reindex(index=rank.index, columns=rank.columns) < 0).fillna(False)
        lab = lab.mask((lab == "OVERWEIGHT") & rev, "NEUTRAL")

    # WATCH overlay, applied LAST to match get_forward_tilt's ordering. Without it
    # the UNDERWEIGHT band is not reproducible: WATCH needs rank <= 0.35 and the
    # underweight cut is 0.25, so it masks underweights mid-window without changing
    # today's label. It can never touch an OVERWEIGHT (cut 0.75).
    if breadth_hist is not None and not breadth_hist.empty:
        brd = breadth_hist.reindex(index=rank.index, columns=rank.columns)
        watch = ((brd >= _WATCH_BREADTH) & (rank <= _WATCH_RS_MAX)).fillna(False)
        lab = lab.mask(watch, "WATCH")
    return lab.tail(lookback)


def _tenure_days(lab: pd.DataFrame, current: pd.Series) -> pd.Series:
    """Consecutive trailing sessions each sector has held its CURRENT label.

    OVERWEIGHT and UNDERWEIGHT only. NEUTRAL is excluded because it is the residual
    bucket — every overlay drops into it, so a NEUTRAL streak measures "nothing else
    fired", not a call being held. UNDERWEIGHT became reproducible once _tilt_history
    started modelling WATCH (see its docstring).

    0 means "not established": not in a directional bucket, the rebuilt label
    disagrees with the live one, or no history in the window. Callers must render 0
    as blank, never as "0 days".
    """
    out = {}
    for sec, cur in current.items():
        if sec not in lab.columns or cur not in ("OVERWEIGHT", "UNDERWEIGHT"):
            out[sec] = 0
            continue
        n = 0
        for v in lab[sec].values[::-1]:
            if v != cur:                    # None (no data that day) also breaks it
                break
            n += 1
        out[sec] = n
    return pd.Series(out, dtype=int)


# ── measured evidence per horizon — REGENERATED, see scripts/gen_tilt_evidence.py ──
# Long-only top-4, EXCESS over the equal-weight sector basket, NON-overlapping
# rebalance, net of 25bps/side, on the CORRECTED panel (lagged liquidity filter,
# corporate actions dropped).
#
# WHY THESE NUMBERS CHANGED SO MUCH. The previous table claimed +19.6%/yr (t 2.03)
# at 1-2wk and cited scripts/study_tilt_horizons.py — a script that computes only IC
# and long/short spread and never produces a %/yr figure at all. The repo's own
# net-%/yr backtest (scripts/backtest_tilt_vs_clock.py) prints +0.4%/yr for the same
# spec, and an independent rebuild agrees to the decimal. The old table had no
# reproducible provenance; every horizon was overstated by 12-49pp/yr and every t
# was inflated. gen_tilt_evidence.py is now that provenance — the numbers below can
# be reproduced with one command.
#
# TWO METHOD FIXES, both of which move the answer more than any parameter here:
#   • PHASE. `index[::h]` is one of h possible rebalance calendars. Offset alone
#     swings a horizon by 10-16pp/yr IN BOTH DIRECTIONS (it made 1-2wk look
#     positive when it is negative, and 9-10wk look dead when it is the strongest).
#     net_yr is now averaged over ALL h calendars.
#   • net_t is the MEDIAN per-phase Newey-West t, not the best one.
#
# `pct_pos` = share of rebalance calendars that came out positive. Descriptive only:
# the phases overlap almost completely, so it is NOT h independent tests.
#
# `validated` requires ALL of: >=90% of phases positive, the same sign in all three
# eras, and median per-phase |t| >= 2. NO HORIZON CURRENTLY QUALIFIES — the long
# horizons fail only on the t, which is the honest state of the evidence. The UI
# must present every horizon as a lean, never as a validated edge.
_HORIZON_EVIDENCE: dict[int, dict] = {
    10: dict(label="1-2 wk",   reb_yr=25.2, net_yr=-1.4, net_t=-0.30, ls_ic_t=2.38,
             pct_pos=0.30, era={"2018-21": 1.0, "2022-24": -3.3, "2025-26": -3.5},
             validated=False),
    20: dict(label="3-4 wk",   reb_yr=12.6, net_yr=-0.3, net_t=-0.01, ls_ic_t=1.09,
             pct_pos=0.50, era={"2018-21": -2.2, "2022-24": 3.2, "2025-26": -2.6},
             validated=False),
    30: dict(label="5-6 wk",   reb_yr=8.4,  net_yr=1.5,  net_t=0.39,  ls_ic_t=1.27,
             pct_pos=0.73, era={"2018-21": -3.2, "2022-24": 8.8, "2025-26": -1.9},
             validated=False),
    40: dict(label="7-8 wk",   reb_yr=6.3,  net_yr=4.1,  net_t=0.89,  ls_ic_t=2.13,
             pct_pos=0.93, era={"2018-21": -0.3, "2022-24": 10.8, "2025-26": 1.3},
             validated=False),
    50: dict(label="9-10 wk",  reb_yr=5.0,  net_yr=6.7,  net_t=1.21,  ls_ic_t=2.75,
             pct_pos=0.96, era={"2018-21": 2.0, "2022-24": 14.4, "2025-26": 1.9},
             validated=False),
    60: dict(label="11-12 wk", reb_yr=4.2,  net_yr=6.5,  net_t=1.15,  ls_ic_t=2.69,
             pct_pos=0.98, era={"2018-21": 2.7, "2022-24": 12.9, "2025-26": 2.3},
             validated=False),
}
# Breakeven cost, phase-averaged (scripts/audit_tilt_robustness.py test 4): the
# gross edge divided by its sensitivity to cost. Realistic all-in Indian retail cost
# on a 4-sector stock basket is ~20-40bps/side, so the two shortest horizons sit at
# or below their own breakeven BEFORE a trade is placed.
_HORIZON_BREAKEVEN_BPS = {10: 21, 20: 17, 30: 30, 40: 65, 50: 98, 60: 102}

# ── bucket-level evidence — what a sector ON the list actually did ───────────
# THIS REPLACES THE PER-SECTOR `est_rel_bps`, and the replacement is a change of
# granularity, not of scale. `est_rel_bps` was a straight line through rank (corr
# 1.0000 with rank — it restated the list order and nothing else), calibrated on a
# "~1.9%/10d tercile spread" from the same unreproducible lineage as the old
# evidence box. It could not be repaired by retuning `_REL_SLOPE_BPS`, because the
# measured rank ladder is FLAT and then STEPS at the top quintile (Q1 +0.27 /
# Q2 +0.26 / Q3 +0.29 / Q4 +0.28 / Q5 +0.67) — a linear map is the wrong SHAPE.
# (The constant's own history says the same thing: 290 -> 75 -> 110 -> 190 -> 290,
# four values in one session, each from a different method.)
#
# What IS measurable is the bucket. Mean forward excess over the equal-weight
# sector basket across every sector-day in the bucket, GROSS of cost; Newey-West t
# at lag=h on the per-date cross-sectional mean, so overlap and within-date
# correlation are both handled. Source: scripts/gen_tilt_evidence.py.
#
# TWO THINGS TO READ OFF IT:
#  • The buy list WORKS gross at every horizon (t 2.9-4.0). It is turnover that
#    kills the short ones: +0.409pp x 25.2 rebalances/yr is ~+10%/yr gross against
#    a ~10.0pp/yr cost drag, which is why 1-2wk nets negative while the ranking
#    itself is fine. `cost_drag` is carried here so the UI can say exactly that.
#  • UNDERWEIGHT is only an "avoid" list at LONG horizons (t -2.14/-2.87/-2.79 at
#    7-8/9-10/11-12wk). At 1-6wk it is indistinguishable from neutral (t -1.36 to
#    -0.46), so the "AVOID / trim" copy is unsupported there.
# OVERWEIGHT figures are exact; UNDERWEIGHT is the rank band only (WATCH can mask
# it and is not modelled).
_BUCKET_EVIDENCE: dict[int, dict] = {
    10: dict(ow_pp=0.409, ow_t=3.69, ow_n=9581,
             uw_pp=-0.107, uw_t=-1.36, uw_n=12084, gross_yr=8.6, cost_drag=10.0),
    20: dict(ow_pp=0.658, ow_t=2.87, ow_n=9321,
             uw_pp=-0.074, uw_t=-0.46, uw_n=12030, gross_yr=4.8, cost_drag=5.1),
    30: dict(ow_pp=1.105, ow_t=3.11, ow_n=9169,
             uw_pp=-0.184, uw_t=-0.85, uw_n=11970, gross_yr=4.9, cost_drag=3.4),
    40: dict(ow_pp=1.680, ow_t=3.49, ow_n=9193,
             uw_pp=-0.621, uw_t=-2.14, uw_n=11903, gross_yr=6.7, cost_drag=2.6),
    50: dict(ow_pp=2.499, ow_t=4.04, ow_n=9168,
             uw_pp=-1.167, uw_t=-2.87, uw_n=11855, gross_yr=8.7, cost_drag=2.0),
    60: dict(ow_pp=3.037, ow_t=4.02, ow_n=9370,
             uw_pp=-1.456, uw_t=-2.79, uw_n=11778, gross_yr=8.2, cost_drag=1.7),
}
TILT_HORIZONS = [(v["label"], k) for k, v in sorted(_HORIZON_EVIDENCE.items())]


def get_forward_tilt(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    horizon_days: int = _MOM_2W,
) -> tuple[pd.DataFrame, dict]:
    """
    Per-sector forward tilt + regime meta. Causal (data <= as_of_date).

    `horizon_days` selects the forward window the tilt is aimed at, and the RS
    lookbacks SCALE with it: long = horizon_days, short = horizon_days // 2.
    At the default 10 that is exactly 10d/5d — bit-identical to the validated
    1-2wk tilt, so the shipped product is unchanged.

    WHY THE LOOKBACK SCALES (scripts/study_tilt_horizons.py, 24 sectors x 2,124
    sessions 2018-2026). Holding the 10d/5d lookback and simply lengthening the
    forward window is NOT the right build — a horizon-matched lookback is better
    at every long horizon (e.g. 9-10wk: matched IC +0.0553 vs fixed +0.0310).

    MEASURED PER HORIZON — long-only top-4, excess over the equal-weight sector
    basket, NON-overlapping rebalance, net of 25bps/side, AVERAGED OVER ALL h
    rebalance calendars (scripts/gen_tilt_evidence.py, 2,129 sessions 2018-2026):
        horizon   reb/yr  net %/yr  med t  %cal+  breakeven  2018-21 2022-24 2025-26
        1-2 wk     25.2     -1.4    -0.30   30%     21bps      +1.0    -3.3    -3.5
        3-4 wk     12.6     -0.3    -0.01   50%     17bps      -2.2    +3.2    -2.6
        5-6 wk      8.4     +1.5    +0.39   73%     30bps      -3.2    +8.8    -1.9
        7-8 wk      6.3     +4.1    +0.89   93%     65bps      -0.3   +10.8    +1.3
        9-10 wk     5.0     +6.7    +1.21   96%     98bps      +2.0   +14.4    +1.9
        11-12 wk    4.2     +6.5    +1.15   98%    102bps      +2.7   +12.9    +2.3
    Three things to read off that table. (1) The two shortest horizons are NEGATIVE
    and their breakeven cost is below realistic retail cost — they are not tradeable
    and must not be offered as if they were. (2) Longer horizons rebalance 4-6x less
    often, which is most of why they survive at all. (3) Not one horizon clears
    |t| >= 2 on the median calendar, so every one of them is a lean.

    CAUTION — the "daily-IC t ~ 9" that used to headline this module was a NAIVE t.
    Forward windows overlap, so adjacent dates are dependent. Long/short IC t under a
    Newey-West correction at lag=h is the more robust read: +2.38 (1-2wk), +2.13
    (7-8wk), +2.75 (9-10wk), +2.69 (11-12wk); not significant at 3-6wk. Note the
    1-2wk long/short IC clears |t|>=2 while its long-only net return is NEGATIVE:
    the ranking carries some information, but not enough to pay the turnover the
    short horizon demands. That gap is the whole story of this tab.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    _H = max(2, int(horizon_days))
    _L = _H                      # long RS lookback  (10 at the default → shipped)
    _S = max(2, _H // 2)         # short RS lookback (5  at the default → shipped)
    # Delivery-flow windows scale too (5/100 at the default → shipped). Before this,
    # they were pinned at 5/100 at EVERY horizon while the RS legs scaled — so a
    # 11-12wk call was ranked partly on the last five sessions of delivery. See
    # _dv_windows for the measurement and for why BOTH windows must scale.
    _DVF, _DVB = _dv_windows(_H)

    cols = ["sector", "score", "rank", "tilt", "rs_2w", "rs_1w", "dv5d",
            "accum_breadth", "deliv_slope", "n_liq", "thin", "divergence",
            "persistence", "revert", "est_rel_bps", "confidence", "days_in_tilt",
            "ret_since_tilt_pct", "ret_since_tilt_rel_pp"]
    empty = pd.DataFrame(columns=cols)

    panel = _load_sector_panel(as_of_date, min_turnover_lacs, _panel_lookback_cal(_H))
    if panel.empty:
        return empty, _market_regime(pd.DataFrame())
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    nifty = _load_nifty(as_of_date)
    regime = _market_regime(nifty)

    # per-sector momentum + delivery flow, all trailing (causal)
    recs = []
    for sector, g in panel.groupby("sector", sort=False):
        g = g.sort_values("trade_date")
        if len(g) < max(_MIN_HIST, _L + 2):
            continue
        mom_2w = float(_compound(g["wtd_ret_pct"], _L).iloc[-1])
        mom_1w = float(_compound(g["wtd_ret_pct"], _S).iloc[-1])
        dv = g["daily_dv_cr"].astype(float)
        base = dv.iloc[:-1].tail(_DVB).mean()
        dv5d = float(dv.tail(_DVF).mean() / base) if base and base > 0 else np.nan
        recs.append(dict(sector=sector, mom_2w=mom_2w, mom_1w=mom_1w, dv5d=dv5d))
    fac = pd.DataFrame(recs)
    if len(fac) < _MIN_SECTORS:
        return empty, regime

    # relative strength vs Nifty (fallback to absolute momentum if Nifty missing)
    if not nifty.empty:
        n_1w = float(_compound(nifty["nret"], _S).iloc[-1])
        n_2w = float(_compound(nifty["nret"], _L).iloc[-1])
    else:
        n_1w = n_2w = 0.0
    fac["rs_2w"] = fac["mom_2w"] - n_2w
    fac["rs_1w"] = fac["mom_1w"] - n_1w

    # robust bottom-up accumulation breadth (quality overlay, not a return driver)
    try:
        rob = get_robust_delivery_signals(as_of_date, min_turnover_lacs)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("robust delivery signal failed (%s); breadth overlay disabled", exc)
        rob = pd.DataFrame(columns=["sector", "breadth_accum", "deliv_slope"])
    fac = fac.merge(rob[["sector", "breadth_accum", "deliv_slope"]]
                    if not rob.empty else rob, on="sector", how="left")
    for c in ("breadth_accum", "deliv_slope"):
        if c not in fac.columns:
            fac[c] = np.nan

    # liquid-name counts → thin flag
    n_liq = _liquid_name_counts(as_of_date, min_turnover_lacs)
    fac["n_liq"] = fac["sector"].map(n_liq).fillna(0).astype(int)
    fac["thin"] = fac["n_liq"] < _MIN_LIQ_NAMES

    # cross-sectional pct-rank composite (momentum-led)
    def _r(col):
        return fac[col].rank(pct=True)
    r_rs2 = _r("rs_2w")
    fac["score"] = (_W_RS2 * r_rs2 + _W_RS1 * _r("rs_1w") + _W_DV5 * _r("dv5d")).astype(float)
    fac["rank"] = fac["score"].rank(pct=True)

    # divergence: delivery rank minus momentum rank (+ = accumulating ahead of price)
    fac["divergence"] = (_r("dv5d") - r_rs2).astype(float)

    # dispersion of momentum → is there anything to rotate on today?
    disp = float(fac["rs_2w"].std())
    regime["dispersion"] = disp
    # Horizon provenance + the MEASURED evidence for THAT horizon, so the UI can
    # never imply the 1-2wk validation covers a 12-week call.
    regime["horizon_days"] = _H
    regime["rs_long_days"], regime["rs_short_days"] = _L, _S
    # Published so the UI can label the column for the horizon in force ("dv5d" only
    # at 1-2wk). A fixed "dv5d" caption on a scaled factor is the same copy bug the
    # rs2w/rs4w labels already had to be fixed for.
    regime["dv_flow_days"], regime["dv_base_days"] = _DVF, _DVB
    regime["horizon_stats"] = _HORIZON_EVIDENCE.get(_H)
    if np.isfinite(disp) and disp < 1.5:
        regime["banner"] += "  (Low sector dispersion — tilt has little to add today.)"
        regime["size_hint"] = round(float(regime.get("size_hint", 0.5)) * 0.5, 2)
        if regime.get("verdict") == "ACT":
            regime["verdict"] = "SELECTIVE"
            regime["action"] = ("Half size — sectors are bunched (low dispersion); little to "
                                "rotate on even in a good backdrop.")

    conf_mult = regime["confidence_mult"]
    brd = fac["breadth_accum"].fillna(0.0)     # a FRACTION in [0,1], not a rank

    def _tilt(rr: float, br: float) -> str:
        # WATCH: heavy accumulation but momentum has not turned (contrarian, held out)
        if br >= _WATCH_BREADTH and rr <= _WATCH_RS_MAX:
            return "WATCH"
        if rr >= _OW_RANK:
            return "OVERWEIGHT"
        if rr <= _UW_RANK:
            return "UNDERWEIGHT"
        return "NEUTRAL"

    fac["tilt"] = [_tilt(rr, br) for rr, br in zip(fac["rank"].values, brd.values)]
    # thin sectors cannot be a confident overweight — demote to NEUTRAL/WATCH
    fac.loc[fac["thin"] & (fac["tilt"] == "OVERWEIGHT"), "tilt"] = "NEUTRAL"

    # ── momentum-persistence gate (the validated reliability lever) ───────────
    # Demote OVERWEIGHT in structurally mean-reverting sectors (trailing edge < 0):
    # those OW calls historically fade (~44% accuracy). Lifts OW accuracy ~56% → ~60%.
    try:
        pers, pers_hist = _sector_persistence(as_of_date, min_turnover_lacs,
                                              fwd_days=_H, with_history=True)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("sector persistence failed (%s); gate disabled", exc)
        pers = pd.DataFrame(columns=["sector", "persistence", "pers_n"])
        pers_hist = pd.DataFrame()
    pmap = pers.set_index("sector")["persistence"] if not pers.empty else pd.Series(dtype=float)
    fac["persistence"] = fac["sector"].map(pmap)
    fac["revert"] = fac["persistence"] < 0                     # NaN → False (unknown ⇒ keep)
    fac.loc[(fac["tilt"] == "OVERWEIGHT") & fac["revert"], "tilt"] = "NEUTRAL"

    # ── regime-inversion gate: REMOVED (was dead code) ────────────────────────
    # `momentum_inverts` was set False in every branch of _market_regime after the
    # real-bear recalibration, so this suppression, `ow_suppressed`, and the bps
    # zero-out below could never fire. The comment still claimed overweights were
    # being suppressed in downtrends, which was untrue of the shipped behaviour.
    #
    # ── est_rel_bps: SUPPRESSED, not computed ────────────────────────────────
    # It was a linear map off _REL_SLOPE_BPS = 290, calibrated on a ~1.9%/10d
    # tercile spread. That spread does not reproduce: phase-averaged over all h
    # rebalance calendars (scripts/audit_tilt_robustness.py) the 1-2wk top-4 basket
    # is NEGATIVE vs the equal-weight sector basket, and no horizon clears |t|>=2.
    # Printing "est +145bps" next to a sector asserted a per-sector expected return
    # the data does not support. The column is kept (callers index it) but carries
    # NaN, and the UI omits it rather than quoting a number that is not measured.
    fac["est_rel_bps"] = np.nan
    # per-sector confidence: regime × thin × historically-reverting down-weights
    fac["confidence"] = (conf_mult
                         * np.where(fac["thin"], 0.5, 1.0)
                         * np.where(fac["revert"], 0.6, 1.0)).round(2)

    # ── tenure: consecutive sessions this sector has held its current call ────
    # Rebuilt from the panel already loaded above — no extra query. 0 = not
    # established (an overlay intervened, or no history) and must render blank.
    try:
        _nliq_h = _liquid_name_counts_hist(as_of_date, min_turnover_lacs)
        try:
            _brd_h = get_accum_breadth_history(as_of_date, _TENURE_LOOKBACK,
                                               min_turnover_lacs)
        except Exception as exc:                              # noqa: BLE001
            # Without it the UNDERWEIGHT band is not reproducible, so tenure falls
            # back to OVERWEIGHT only rather than printing an overstated streak.
            log.warning("accum-breadth history failed (%s); UW tenure disabled", exc)
            _brd_h = None
        _lab = _tilt_history(panel, nifty, _L, _S, pers_hist, _nliq_h,
                             breadth_hist=_brd_h, dv_flow=_DVF, dv_base=_DVB)
        fac["days_in_tilt"] = (_tenure_days(_lab, fac.set_index("sector")["tilt"])
                               .reindex(fac["sector"]).fillna(0).astype(int).values)
        if _brd_h is None or _brd_h.empty:
            # WATCH could not be reproduced, so an UNDERWEIGHT streak may be
            # overstated (measured: 5.7% wrong, median 6 sessions too long). Suppress
            # rather than print it.
            fac.loc[fac["tilt"] == "UNDERWEIGHT", "days_in_tilt"] = 0
    except Exception as exc:                                  # noqa: BLE001
        log.warning("tilt tenure failed (%s); badge disabled", exc)
        fac["days_in_tilt"] = 0
    regime["tenure_lookback"] = _TENURE_LOOKBACK

    # ── move since the call appeared ─────────────────────────────────────────
    # `days_in_tilt` = N counts trailing sessions INCLUDING today, so the call first
    # showed on session t-(N-1). That list is published after the close, so the
    # earliest you could act is the next session — the window is therefore the LAST
    # N-1 daily returns, i.e. close(t-(N-1)) to close(t). N=1 ("NEW") has nothing
    # elapsed and stays blank rather than printing 0.0%.
    #
    # BOTH numbers are reported because the absolute one alone misleads: a sector up
    # 3% while every sector rose 4% has LOST ground, and this engine's entire claim is
    # about EXCESS over the equal-weight sector basket, never absolute return. The
    # relative figure is the one that corresponds to what the tilt is trying to do.
    fac["ret_since_tilt_pct"] = np.nan
    fac["ret_since_tilt_rel_pp"] = np.nan
    try:
        _rw = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
        _bench = _rw.mean(axis=1)              # equal-weight sector basket
        _abs, _rel = {}, {}
        for _sec, _n in zip(fac["sector"], fac["days_in_tilt"]):
            _k = int(_n) - 1
            if _k < 1 or _sec not in _rw.columns:
                continue
            _w = _rw[_sec].iloc[-_k:]
            if _w.isna().any() or len(_w) < _k:
                continue                        # a hole in the window ⇒ no claim
            _b = _bench.reindex(_w.index)
            _cs = float((1.0 + _w / 100.0).prod() - 1.0) * 100.0
            _cb = float((1.0 + _b / 100.0).prod() - 1.0) * 100.0
            _abs[_sec], _rel[_sec] = _cs, _cs - _cb
        fac["ret_since_tilt_pct"] = fac["sector"].map(_abs)
        fac["ret_since_tilt_rel_pp"] = fac["sector"].map(_rel)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("since-entry return failed (%s); omitted", exc)

    fac["accum_breadth"] = fac["breadth_accum"]
    fac = fac.sort_values("score", ascending=False).reset_index(drop=True)
    return fac[cols], regime


# ── market-breadth regime NOWCAST (situational awareness, NOT a forecast) ─────────
# Backtested on the 4yr F&O panel (scripts/audit_regime_detection.py). HONEST verdict:
# structural bear signals (below-200-DMA, EMA50<EMA200 death cross) are LAGGING and
# CONTRARIAN in this regime — a confirmed death cross was followed by +5.3%/40d (91%
# false-alarm), so NONE of these forecast a 1-2mo decline. Breadth also does NOT sharpen
# the tilt inversion (OW−UW ≈ −0.8% with or without a breadth split). So this nowcast is
# DISPLAY-ONLY context — it describes the CONCURRENT risk state ("are we broadly weak
# right now"), it is not wired into the tilt and it is not a market-direction call. The one
# mildly forward-tilted signal is breadth DIVERGENCE (index up while breadth quietly falls =
# leadership narrowing) — surfaced as an explicit LOW-CONFIDENCE early-caution only.
_BREADTH_MIN_TO_LACS = 5000.0     # ~50 Cr turnover floor → stable large/mid-cap breadth set
_BREADTH_BULL        = 55.0       # % of members above 50-DMA for a broad-bull read
_BREADTH_BEAR        = 40.0       # ... below this = broad weakness


def get_market_breadth(as_of_date: date, min_turnover_lacs: Optional[float] = None) -> dict:
    """Large-cap breadth + index-structure NOWCAST (concurrent risk state, causal).

    Returns state ∈ {BULL, NEUTRAL, BEAR, UNKNOWN}, breadth %s, structural flags, how many
    days the state has held, and a low-confidence 'leadership narrowing' early-caution. All
    inputs use data ≤ as_of_date. This is context, not a forecast (see module note above).
    """
    out = dict(ok=False, state="UNKNOWN", b50=float("nan"), b200=float("nan"),
               px_vs_200=float("nan"), death_cross=False, dur_days=0,
               narrowing=False, n=0, caption="Breadth unavailable (insufficient history).")
    try:
        df = query_dataframe(
            """
            SELECT b.symbol, b.trade_date, b.close_price
            FROM daily_data b
            WHERE b.series IN ('EQ','SM','ST')
              AND b.trade_date > (?::date - 400) AND b.trade_date <= ?
              -- membership on a 20-session MEDIAN turnover, not the as-of day's.
              -- A same-day floor rebuilds the breadth universe out of whichever
              -- names happened to be active today, which ties the universe to the
              -- day's move and makes breadth drift with the news rather than the
              -- trend it is meant to describe.
              AND b.symbol IN (
                  SELECT symbol FROM daily_data
                  WHERE series IN ('EQ','SM','ST')
                    AND trade_date > (?::date - 35) AND trade_date <= ?
                  GROUP BY symbol
                  HAVING MEDIAN(turnover_lacs) >= ?)
            ORDER BY b.symbol, b.trade_date
            """,
            [as_of_date, as_of_date, as_of_date, as_of_date, _BREADTH_MIN_TO_LACS],
        )
    except Exception as exc:                                      # noqa: BLE001
        log.warning("breadth query failed (%s)", exc)
        return out
    if df.empty:
        return out
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    p = df.pivot_table("close_price", "trade_date", "symbol").sort_index()
    if len(p) < 55:
        return out
    s50 = p.rolling(50).mean()
    s200 = p.rolling(200).mean()
    b50 = ((p > s50).sum(axis=1) / s50.notna().sum(axis=1).replace(0, np.nan) * 100)
    has200 = s200.notna().sum(axis=1)
    b200 = ((p > s200).sum(axis=1) / has200.replace(0, np.nan) * 100)

    # index structure (Nifty 50)
    nf = _load_nifty(as_of_date)
    px_vs_200 = float("nan"); death = False
    if not nf.empty and len(nf) >= 200:
        c = nf.sort_values("trade_date")["close_val"].astype(float)
        sma200 = c.rolling(200).mean().iloc[-1]
        px_vs_200 = float(c.iloc[-1] / sma200 - 1) * 100 if sma200 and sma200 > 0 else float("nan")
        death = bool(c.ewm(span=50, adjust=False).mean().iloc[-1]
                     < c.ewm(span=200, adjust=False).mean().iloc[-1])

    # 5-state concurrent nowcast: combine breadth (short-term participation) with the
    # 200-DMA (long-term trend). RECOVERING/WEAKENING name the TRANSITIONS the user asked
    # about — short-term and long-term disagreeing = regime turning but not yet confirmed.
    px200_ok = np.isfinite(px_vs_200)
    def _state_at(bp):
        long_up = (not px200_ok) or px_vs_200 > 0
        if bp >= _BREADTH_BULL:
            return "BULL" if long_up else "RECOVERING"       # strong breadth; long-line yes/no
        if bp < _BREADTH_BEAR:
            return "BEAR" if not long_up else "WEAKENING"    # weak breadth; long-line no/yes
        return "NEUTRAL"
    state = _state_at(float(b50.iloc[-1]))
    # Duration of the BREADTH BAND only — not of `state`. The two differ: `state` also
    # needs the 200-DMA overlay, which is a today-only read here, so the state's own
    # age is not computable from this series. It was being rendered as "STATE ... held
    # N days", which reads as the state's age and can be badly wrong (a 2-day-old
    # RECOVERING shown as "held 15 days"). Returned under an honest name and labelled
    # as the breadth band in the UI.
    def _band(x): return "hi" if x >= _BREADTH_BULL else ("lo" if x < _BREADTH_BEAR else "mid")
    daily_band = b50.apply(_band); today_band = daily_band.iloc[-1]
    dur = 0
    for v in daily_band.iloc[::-1]:
        if v == today_band: dur += 1
        else: break

    # early-caution: index 20d up but breadth rolling over (narrowing leadership)
    idx_r20 = (float(nf.sort_values("trade_date")["close_val"].astype(float).iloc[-1]
                     / nf.sort_values("trade_date")["close_val"].astype(float).iloc[-21] - 1) * 100
               if not nf.empty and len(nf) > 21 else float("nan"))
    b50_chg10 = float(b50.iloc[-1] - b50.iloc[-11]) if len(b50) > 11 else float("nan")
    narrowing = bool(np.isfinite(idx_r20) and idx_r20 > 0
                     and np.isfinite(b50_chg10) and b50_chg10 < 0 and b50.iloc[-1] < 60)

    b50v = float(b50.iloc[-1]); b200v = float(b200.iloc[-1]) if b200.notna().any() else float("nan")
    _pct = lambda x: f"{x:.0f}%" if np.isfinite(x) else "n/a"
    cap = (f"{_pct(b50v)} of large-caps above their 50-day line, {_pct(b200v)} above the "
           f"200-day line. Nifty is {abs(px_vs_200):.1f}% {'above' if px_vs_200>=0 else 'below'} "
           f"its 200-day line" if np.isfinite(px_vs_200) else
           f"{_pct(b50v)} of large-caps above their 50-day line")
    out.update(ok=True, state=state, b50=b50v, b200=b200v, px_vs_200=px_vs_200,
               death_cross=death, band_days=int(dur), dur_days=int(dur),
               narrowing=narrowing, n=int(p.shape[1]), caption=cap)
    return out


# ── Nifty breakout state (held vs fake) — the sharpest 8yr signal (context flag) ──────
# audit_trend_4band_8yr.py: a 20d-high close-break that HELD 3 days → +1.0%/10d (t+3.5),
# +8.7%/6mo (t+4.0, 92% hit); a FAKE break (reversed ≤3d) → −0.5%/10d (t−1.7 short) then
# recovers. Display context only — the single most significant state in the 8-year audit.
def get_nifty_breakout(as_of_date: date) -> dict:
    """Detect a recent Nifty 20d-high breakout and whether it HELD or FAILED (causal)."""
    out = dict(ok=False, state="none", days_since=0, level=float("nan"))
    try:
        df = query_dataframe(
            "SELECT trade_date, high_val, close_val FROM index_data "
            "WHERE index_name = 'Nifty 50' AND trade_date <= ? ORDER BY trade_date",
            [as_of_date])
    except Exception:                                            # noqa: BLE001
        return out
    if df.empty or len(df) < 30:
        return out
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    c = df["close_val"].astype(float).reset_index(drop=True)
    h = df["high_val"].astype(float).reset_index(drop=True)
    hi = h.shift(1).rolling(20).max()                            # prior 20d high (causal)
    brk = c > hi                                                 # 20d-high close-break
    look = 7                                                     # was there a fresh break in the last week?
    recent = brk.iloc[-look:]
    if not recent.any():
        return dict(ok=True, state="none", days_since=0, level=float("nan"))
    pos = int(np.where(recent.values)[0][-1])                    # most recent break within window
    abs_i = len(c) - look + pos
    level = float(hi.iloc[abs_i])
    days_since = len(c) - 1 - abs_i
    held = bool((c.iloc[abs_i:] >= level).all())                 # stayed above the broken level
    return dict(ok=True, state="HELD" if held else "FAILED",
                days_since=days_since, level=level)


# ── multi-timeframe trend read (swing/short/long/vlong) — "is the trend changing?" ────
# RE-AUDITED 2026-08-09 on 13.6yr (scripts/audit_mtf_entry_claims.py, 3,165 sessions
# 2013-2026). The previous "3/4-up is the BEST entry (t+3.4 swing)" claim DOES NOT HOLD:
#
#   • BASE RATE. Nifty is up in 59.9% of 10d windows and 67.5% of 65d windows
#     unconditionally (mean +0.53% / +3.33%). Any state must be scored as EXCESS
#     over that, not as a raw hit rate.
#   • OVERLAP. The state is read daily but the forward window is 10-65 days, so
#     consecutive rows share nearly all their return. n_up==3 at 10d: excess
#     +0.29pp, t_naive 6.91 -> t_NEWEY-WEST 1.17. The "t+3.4" was a naive number.
#   • MULTIPLICITY. "BEST" is the max over 5 states. Permutation test on max-|t|:
#     p=0.59 (10d), p=0.26 (65d). NO alignment state carries demonstrable forward
#     information.
#   • NON-OVERLAPPING sample REVERSES the ordering: 3/4-up excess -0.20pp vs
#     4/4-up -0.03pp, i.e. 4/4 looks better, the opposite of the old claim.
#   • The "all-4-down -> ~76% up over 3mo" line was not reproducible: measured
#     71.2% against a 67.5% base rate = +3.7pp, t_NW 1.64. Still the best cell in
#     the table, but not significant.
#   • n_up==3 POOLS "3 up + 1 FLAT" (n=463, +0.69%/10d) with "3 up + 1 DOWN"
#     (n=105, +1.40%/10d) — materially different states scored identically.
#
# The bands are therefore a DESCRIPTIVE alignment read, not an entry timer. Labels
# below describe what the market IS doing; they no longer promise forward edge.
_MTF_EVIDENCE = {   # excess pp over the unconditional base, 13.6yr, NW-t at lag=h
    0: dict(ex10=+0.10, ex65=+2.15, t10=+0.21, t65=+1.64, freq=19.1),
    1: dict(ex10=-0.43, ex65=-0.82, t10=-1.34, t65=-1.17, freq=10.8),
    2: dict(ex10=+0.21, ex65=-0.46, t10=+0.83, t65=-0.48, freq=16.5),
    3: dict(ex10=+0.29, ex65=-0.48, t10=+1.17, t65=-0.39, freq=18.2),
    4: dict(ex10=-0.17, ex65=-0.42, t10=-1.10, t65=-0.37, freq=35.4),
}
_MTF_BANDS = {"swing": (10, "1-3 wk"), "short": (30, "1-2 mo"),
              "long": (60, "2.5-4 mo"), "vlong": (120, "4-6 mo")}

def get_mtf_trend(as_of_date: date) -> dict:
    """Per-band Nifty trend (UP/DOWN/FLAT) + alignment + validated entry posture (causal)."""
    out = dict(ok=False, bands={}, n_up=0, n_dn=0, posture="", entry="", flips=[])
    try:
        df = query_dataframe(
            "SELECT trade_date, close_val FROM index_data WHERE index_name = 'Nifty 50' "
            "AND trade_date <= ? ORDER BY trade_date", [as_of_date])
    except Exception:                                            # noqa: BLE001
        return out
    if df.empty or len(df) < 130:
        return out
    c = df["close_val"].astype(float).reset_index(drop=True)
    bands = {}; n_up = n_dn = 0; flips = []
    for name, (span, horizon) in _MTF_BANDS.items():
        e = c.ewm(span=span, adjust=False).mean()
        sl_win = max(5, span // 4)
        slope = e - e.shift(sl_win)
        px, ev, sv = c.iloc[-1], e.iloc[-1], slope.iloc[-1]
        state = "UP" if (px > ev and sv > 0) else "DOWN" if (px < ev and sv < 0) else "FLAT"
        # recent flip (state 3 days ago differed and was the opposite trend)
        px3, ev3, sv3 = c.iloc[-4], e.iloc[-4], (e.iloc[-4] - e.iloc[-4 - sl_win]) if len(e) > 4 + sl_win else 0
        prev = "UP" if (px3 > ev3 and sv3 > 0) else "DOWN" if (px3 < ev3 and sv3 < 0) else "FLAT"
        flipped = state != prev and state in ("UP", "DOWN") and prev in ("UP", "DOWN")
        bands[name] = dict(state=state, horizon=horizon, flipped=bool(flipped))
        if flipped: flips.append((name, state))
        n_up += state == "UP"; n_dn += state == "DOWN"

    # Descriptive labels. No state showed forward edge that survived base-rate,
    # overlap and multiplicity correction (see _MTF_EVIDENCE above), so none of
    # these promise an entry — they say what the market IS doing.
    if   n_up == 4: entry, posture = ("ALIGNED_UP", "All four timeframes up — trend intact and mature. Descriptive only: this is the most common state (35% of sessions) and its 10-day forward return is slightly BELOW the market's own drift.")
    elif n_up == 3: entry, posture = ("MOSTLY_UP",  "Three of four timeframes up. Forward return is +0.3pp above drift at 10 days (t 1.2) — inside noise, and it fails a multiplicity test against the other states. Treat as context, not a buy trigger.")
    elif n_up == 2: entry, posture = ("MIXED",      "Mixed / transition — timeframes disagree. Nothing measurable either way.")
    elif n_up == 1: entry, posture = ("MOSTLY_DOWN","Mostly down — the weakest state measured (-0.4pp vs drift at 10d), though still not significant.")
    else:           entry, posture = ("ALL_DOWN",   "All four timeframes down. Historically the BEST of the five states over ~3 months (+2.2pp above drift, 71% up vs a 67% base rate) — a bottom-ish read, NOT a short. Still inside noise (t 1.6).")
    ev = _MTF_EVIDENCE.get(int(n_up), {})
    # 3-up-plus-FLAT and 3-up-plus-DOWN are different states; the count alone hides that.
    detail = ""
    if n_up == 3:
        detail = ("3 up + 1 flat" if n_dn == 0 else "3 up + 1 down")
        if n_dn == 0:
            detail += " (the weaker of the two: +0.7%/10d vs +1.4% when the 4th band is DOWN)"
    out.update(ok=True, bands=bands, n_up=int(n_up), n_dn=int(n_dn),
               posture=posture, entry=entry, flips=flips,
               evidence=ev, detail=detail)
    return out


# ── replay: "what did this tab say on date X, and what happened?" ─────────────
def get_tilt_replay(as_of_date: date, horizon_days: int = 10,
                    min_turnover_lacs: Optional[float] = None,
                    today: Optional[date] = None) -> dict:
    """
    Re-run the tilt as it stood on `as_of_date` and score what followed.

    TWO DIFFERENT QUESTIONS, BOTH RETURNED, DELIBERATELY NOT MERGED:

      `horizon`  as_of -> as_of + horizon_days. This is the ONLY fair scorecard,
                 because it is the window the call was actually making. If that
                 window has not elapsed yet the block is returned with
                 status="OPEN" and no return figures — a partial window is not a
                 result.
      `to_today` as_of -> today. This answers "what would I have if I bought
                 then and still held", which is a legitimate P&L question but is
                 NOT the signal's claim. Scoring a 2-week call over 6 months is a
                 category error, so the two are kept apart and labelled.

    Every return is reported three ways, because a raw number is meaningless:
      abs        the basket's own compound return
      vs_basket  minus the equal-weight ALL-sector basket (what the tilt claims)
      vs_nifty   minus Nifty 50 (what you could have held instead)

    CAUSALITY: the tilt inputs are as-of by construction (verified: sector panel
    and Nifty both stop at as_of_date; six repeated runs give identical tilt
    labels, ranks and OVERWEIGHT sets — residual float noise is ~1e-14 from
    DuckDB's parallel SUM and cannot move a decision).

    KNOWN, UNFIXABLE CAVEAT: `v_sector_master` has no as-of column, so a sector's
    history is measured on the stocks in it TODAY. A name that later joined the
    sector is treated as having always been there. That is constituent
    look-ahead and it flatters every historical sector return here.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()
    out: dict = {"ok": False, "as_of": as_of_date, "horizon_days": int(horizon_days)}

    sess = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data ORDER BY trade_date", [])
    if sess.empty:
        out["error"] = "no sessions in the archive"
        return out
    days = pd.to_datetime(sess["trade_date"]).dt.date.tolist()
    last = days[-1]
    today = today or last

    # snap a weekend/holiday pick back to the previous real session
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
        tilt, regime = get_forward_tilt(anchor, min_turnover_lacs,
                                        horizon_days=int(horizon_days))
    except Exception as exc:                                    # noqa: BLE001
        out["error"] = f"tilt unavailable for {anchor}: {exc}"
        return out
    if tilt is None or tilt.empty:
        out["error"] = (f"not enough history on {anchor} to build a "
                        f"{horizon_days}-day tilt (longer horizons need more)")
        return out

    i = days.index(anchor)
    h_idx = i + int(horizon_days)
    h_end = days[h_idx] if h_idx < len(days) else None
    # When the window has not finished, say HOW FAR IN it is. "Still open" alone
    # leaves the user unable to tell whether it ends tomorrow or in three months.
    _elapsed = max(0, min(int(horizon_days), len(days) - 1 - i))
    out["sessions_elapsed"] = _elapsed
    out["sessions_remaining"] = max(0, int(horizon_days) - _elapsed)

    # daily sector returns, then compound between two dates
    # The LAG must see sessions BEFORE the window opens. Computing it inside a
    # `trade_date > anchor` filter leaves the first session of the window with a
    # NULL w_lag, and `w_lag IS NOT NULL` then deletes that whole session — so a
    # 10-session window silently measured 9, dropping the first day's move (and a
    # 1-session window returned nothing at all). Pull a 30-calendar-day buffer so
    # the lag is defined even across a long holiday cluster, then filter to the
    # real window in the OUTER query.
    panel = query_dataframe(
        f"""
        WITH base AS (
            SELECT b.trade_date, s.sector,
                   (b.close_price - b.prev_close)
                       / NULLIF(b.prev_close,0) * 100 AS raw_r,
                   GREATEST(LEAST((b.close_price - b.prev_close)
                            / NULLIF(b.prev_close,0) * 100, 25), -25) AS r,
                   LAG(b.turnover_lacs) OVER (PARTITION BY b.symbol
                                              ORDER BY b.trade_date) AS w_lag,
                   b.turnover_lacs
            FROM daily_data b
            INNER JOIN v_sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ','SM','ST')
              AND s.sector IS NOT NULL AND s.sector NOT IN ('ETF','Others')
              AND b.trade_date > (?::date - 30) AND b.trade_date <= ?
        )
        SELECT sector, trade_date,
               SUM(w_lag * r) / NULLIF(SUM(CASE WHEN r IS NOT NULL THEN w_lag END),0) AS ret
        FROM base
        -- lagged filter + corporate-action drop, identical to _load_sector_panel.
        -- The replay SCORES the tilt, so scoring it on a different universe rule than
        -- the one that generated the call would make the scorecard measure the
        -- construction difference rather than the call.
        WHERE w_lag >= ? AND ABS(raw_r) < 40
          AND trade_date > ?
        GROUP BY sector, trade_date
        """, [anchor, today, min_turnover_lacs, anchor])
    if panel.empty:
        out["error"] = "no forward sector data"
        return out
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    wide = panel.pivot(index="trade_date", columns="sector", values="ret").sort_index()

    nf = query_dataframe(
        "SELECT trade_date, close_val FROM index_data WHERE index_name='Nifty 50' "
        "AND trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
        [anchor, today])
    nf["trade_date"] = pd.to_datetime(nf["trade_date"])
    nser = nf.set_index("trade_date")["close_val"].astype(float)

    def compound(upto: date) -> pd.Series:
        seg = wide[wide.index <= pd.Timestamp(upto)]
        return (np.expm1(np.log1p(seg / 100.0).sum(axis=0)) * 100.0) if len(seg) else pd.Series(dtype=float)

    def nifty_ret(upto: date) -> float:
        seg = nser[nser.index <= pd.Timestamp(upto)]
        return float(seg.iloc[-1] / seg.iloc[0] - 1.0) * 100.0 if len(seg) >= 2 else float("nan")

    ow = tilt.loc[tilt["tilt"] == "OVERWEIGHT", "sector"].tolist()
    uw = tilt.loc[tilt["tilt"] == "UNDERWEIGHT", "sector"].tolist()

    def block(upto: Optional[date], label: str) -> dict:
        if upto is None:
            return {"status": "OPEN", "label": label}
        rets = compound(upto)
        if rets.empty:
            return {"status": "NO_DATA", "label": label}
        basket = float(rets.mean())
        nif = nifty_ret(upto)
        # a suggested sector that has no forward series is reported, never dropped:
        # silently dropping would bias the result upward if the missing one fell.
        ow_have = [s for s in ow if s in rets.index]
        uw_have = [s for s in uw if s in rets.index]
        ow_r = float(rets[ow_have].mean()) if ow_have else float("nan")
        uw_r = float(rets[uw_have].mean()) if uw_have else float("nan")
        return {
            "status": "DONE", "label": label, "end": upto,
            "sessions": int((wide.index <= pd.Timestamp(upto)).sum()),
            "ow_abs": ow_r, "uw_abs": uw_r, "basket_abs": basket, "nifty_abs": nif,
            "ow_vs_basket": ow_r - basket, "ow_vs_nifty": ow_r - nif,
            "uw_vs_basket": uw_r - basket,
            "ow_minus_uw": ow_r - uw_r,
            "per_sector": rets.reindex(ow + uw),
            "missing": [s for s in (ow + uw) if s not in rets.index],
        }

    out.update(
        ok=True,
        tilt=tilt, regime=regime,
        ow=ow, uw=uw,
        verdict=regime.get("verdict"), size_hint=regime.get("size_hint"),
        state=regime.get("state"),
        horizon=block(h_end, f"over its own {horizon_days}-session horizon"),
        to_today=block(today, "from then until now"),
        horizon_end=h_end, today=today,
        evidence=_HORIZON_EVIDENCE.get(int(horizon_days)),
    )
    return out
