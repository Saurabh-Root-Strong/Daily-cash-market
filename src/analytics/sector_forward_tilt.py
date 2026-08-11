"""
1–2 Week Forward Sector Tilt — the only sector call that survived deep validation.

WHAT IS VALIDATED (scripts/factor_ic_diagnostic.py, backtest_rotation.py, mc_null.py,
371-day panel 2024-12 → 2026-07):
  Cross-sectional sector MOMENTUM predicts 1–2wk forward returns. Relative strength
  vs Nifty (rs_2w, 10d) has daily-IC t ≈ 9 and a Monte-Carlo long/short p < 0.002 vs
  600 random portfolios; edge is cost-robust (0→40bps), sub-period stable (both halves
  Sharpe ~1.7) and top-K insensitive. Delivery flow (dv5d) is a WEAK confirm (t ≈ 3.8).
  F&O positioning is NOT usable at sector granularity (only ~4 sectors carry enough
  F&O names to aggregate) and is deliberately excluded here.

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
from src.analytics.sector_signal_v2 import get_robust_delivery_signals
from src.data.repository import query_dataframe
from src.logging_setup import get_logger

log = get_logger(__name__)

# ── factor construction (mirrors the validated build_factors) ────────────────
_MOM_2W        = 10       # trading days for 2-week momentum / relative strength
_MOM_1W        = 5        # trading days for 1-week momentum
_DV_BASE       = 100      # trailing window for the delivery-flow baseline
_DV_FLOW       = 5        # short delivery-flow window
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


def _load_sector_panel(as_of_date: date, min_turnover_lacs: float) -> pd.DataFrame:
    return query_dataframe(
        """
        WITH base AS (
            SELECT s.sector, b.trade_date, b.turnover_lacs, b.deliv_per,
                   -- winsorize the per-stock daily move: one uncapped print (bonus / illiquid
                   -- spike) otherwise distorts the whole sector's momentum
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
              AND b.trade_date > (?::date - 275)
              AND b.trade_date <= ?
        )
        SELECT sector, trade_date,
               SUM(turnover_lacs * deliv_per / 100.0) / 100.0                 AS daily_dv_cr,
               SUM(w_lag * r) / NULLIF(SUM(CASE WHEN r IS NOT NULL
                                           THEN w_lag END), 0)                AS wtd_ret_pct
        FROM base
        WHERE turnover_lacs >= ? AND w_lag IS NOT NULL
          AND trade_date > (?::date - 260)
        GROUP BY sector, trade_date
        ORDER BY sector, trade_date
        """,
        [as_of_date, as_of_date, min_turnover_lacs, as_of_date],
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
    df["nret"] = (df["pct_chg"].astype(float) if df["pct_chg"].notna().any()
                  else df["close_val"].astype(float).pct_change() * 100)
    return df


def _liquid_name_counts(as_of_date: date, min_turnover_lacs: float) -> pd.Series:
    df = query_dataframe(
        """
        SELECT s.sector, COUNT(*) AS n_liq
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE b.trade_date = ? AND b.series IN ('EQ', 'SM', 'ST')
          AND s.sector NOT IN ('ETF', 'Others')
          AND b.turnover_lacs >= ?
        GROUP BY s.sector
        """,
        [as_of_date, min_turnover_lacs],
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
                        fwd_days: int = _PERS_FWD) -> pd.DataFrame:
    """Per-sector momentum-persistence: trailing mean forward RELATIVE edge (causal).

    For every past day with a realized 10-day forward return (date ≤ as_of − 10 trading
    days), edge = sector's fwd-10 return minus the cross-sectional median sector fwd-10.
    A sector's persistence = the expanding mean of that edge. >0 ⇒ high ranks historically
    kept outperforming (trend-follow, trust the overweight); <0 ⇒ they faded (mean-revert,
    demote the overweight). No lookahead — forward returns beyond as_of are NaN by
    construction.
    """
    empty = pd.DataFrame(columns=["sector", "persistence", "pers_n"])
    panel = query_dataframe(
        """
        WITH base AS (
            SELECT s.sector, b.trade_date, b.turnover_lacs,
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
        WHERE turnover_lacs >= ? AND w_lag IS NOT NULL
          AND trade_date > (?::date - ?)
        GROUP BY sector, trade_date
        ORDER BY trade_date
        """,
        [as_of_date, _PERS_LOOKBACK_CAL, as_of_date,
         min_turnover_lacs, as_of_date, _PERS_LOOKBACK_CAL],
    )
    if panel.empty:
        return empty
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    ret = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
    if len(ret) < fwd_days + _PERS_MIN_OBS:
        return empty
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
    return out


# ── measured evidence per horizon (scripts/study_tilt_horizons.py, 2018-2026) ──
# Long-only top-4, EXCESS over the equal-weight sector basket, NON-overlapping
# rebalance, net of 25bps/side. `ls_ic_t` is the Newey-West t of the long/short IC
# at lag=h — the overlap-corrected read. The UI must show these, per horizon,
# instead of repeating the module's 1-2wk headline everywhere.
_HORIZON_EVIDENCE: dict[int, dict] = {
    10: dict(label="1-2 wk",   reb_yr=25.2, net_yr=19.6, net_t=2.03, ls_ic_t=2.15,
             era={"2018-21": 24.1, "2022-24": 28.4, "2025-26": -7.5}, validated=True),
    20: dict(label="3-4 wk",   reb_yr=12.6, net_yr=19.5, net_t=2.14, ls_ic_t=0.62,
             era={"2018-21": 25.7, "2022-24": 23.7, "2025-26": -3.7}, validated=False),
    30: dict(label="5-6 wk",   reb_yr=8.4,  net_yr=14.3, net_t=1.57, ls_ic_t=1.02,
             era={"2018-21": 11.6, "2022-24": 25.5, "2025-26": -2.7}, validated=False),
    40: dict(label="7-8 wk",   reb_yr=6.3,  net_yr=28.1, net_t=2.77, ls_ic_t=1.93,
             era={"2018-21": 35.1, "2022-24": 30.8, "2025-26": 5.5},  validated=False),
    50: dict(label="9-10 wk",  reb_yr=5.0,  net_yr=20.8, net_t=1.94, ls_ic_t=2.36,
             era={"2018-21": 13.9, "2022-24": 35.1, "2025-26": 7.7},  validated=True),
    60: dict(label="11-12 wk", reb_yr=4.2,  net_yr=29.5, net_t=2.60, ls_ic_t=2.26,
             era={"2018-21": 32.1, "2022-24": 37.8, "2025-26": 6.2},  validated=True),
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

    WHAT IS ACTUALLY VALIDATED PER HORIZON — long-only top-4, excess over the
    equal-weight sector basket, NON-overlapping rebalance, net of 25bps/side:
        horizon   reb/yr  net %/yr   t     2018-21  2022-24  2025-26
        1-2 wk     25.2     19.6    2.03    +24.1    +28.4     -7.5
        3-4 wk     12.6     19.5    2.14    +25.7    +23.7     -3.7
        5-6 wk      8.4     14.3    1.57    +11.6    +25.5     -2.7
        7-8 wk      6.3     28.1    2.77    +35.1    +30.8     +5.5
        9-10 wk     5.0     20.8    1.94    +13.9    +35.1     +7.7
        11-12 wk    4.2     29.5    2.60    +32.1    +37.8     +6.2
    Two things to read off that table. (1) Longer horizons rebalance 4-6x less
    often, so the same gross edge survives cost far better. (2) In the CURRENT
    era the short horizons have stopped working (1-2wk is -7.5%/yr in 2025-26)
    while 7-12wk still pay — the opposite of the tab's original premise.

    CAUTION — the headline claim in this module's docstring ("daily-IC t ~ 9") is
    a NAIVE t. Forward windows overlap, so adjacent dates are dependent. Under a
    Newey-West correction at lag=h the same reproduction gives IC t=1.64 at 1-2wk
    and t=2.02 at 9-10wk. Long/short IC t is the more robust read: +2.15 (1-2wk),
    +2.36 (9-10wk), +2.26 (11-12wk), and NOT significant at 3-8wk. Treat every
    horizon as a lean, not an oracle.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    _H = max(2, int(horizon_days))
    _L = _H                      # long RS lookback  (10 at the default → shipped)
    _S = max(2, _H // 2)         # short RS lookback (5  at the default → shipped)

    cols = ["sector", "score", "rank", "tilt", "rs_2w", "rs_1w", "dv5d",
            "accum_breadth", "deliv_slope", "n_liq", "thin", "divergence",
            "persistence", "revert", "est_rel_bps", "confidence"]
    empty = pd.DataFrame(columns=cols)

    panel = _load_sector_panel(as_of_date, min_turnover_lacs)
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
        base = dv.iloc[:-1].tail(_DV_BASE).mean()
        dv5d = float(dv.tail(_DV_FLOW).mean() / base) if base and base > 0 else np.nan
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
    regime["horizon_stats"] = _HORIZON_EVIDENCE.get(_H)
    if np.isfinite(disp) and disp < 1.5:
        regime["banner"] += "  (Low sector dispersion — tilt has little to add today.)"
        regime["size_hint"] = round(float(regime.get("size_hint", 0.5)) * 0.5, 2)
        if regime.get("verdict") == "ACT":
            regime["verdict"] = "SELECTIVE"
            regime["action"] = ("Half size — sectors are bunched (low dispersion); little to "
                                "rotate on even in a good backdrop.")

    conf_mult = regime["confidence_mult"]
    brd = fac["breadth_accum"].fillna(0.0)
    brd_rank = brd.rank(pct=True)

    def _tilt(row, rr, br) -> str:
        # WATCH: heavy accumulation but momentum has not turned (contrarian, held out)
        if br >= _WATCH_BREADTH and rr <= _WATCH_RS_MAX:
            return "WATCH"
        if rr >= _OW_RANK:
            return "OVERWEIGHT"
        if rr <= _UW_RANK:
            return "UNDERWEIGHT"
        return "NEUTRAL"

    fac["tilt"] = [
        _tilt(row, rr, br) for row, rr, br in
        zip(fac.to_dict("records"), fac["rank"].values, brd.values)
    ]
    # thin sectors cannot be a confident overweight — demote to NEUTRAL/WATCH
    fac.loc[fac["thin"] & (fac["tilt"] == "OVERWEIGHT"), "tilt"] = "NEUTRAL"

    # ── momentum-persistence gate (the validated reliability lever) ───────────
    # Demote OVERWEIGHT in structurally mean-reverting sectors (trailing edge < 0):
    # those OW calls historically fade (~44% accuracy). Lifts OW accuracy ~56% → ~60%.
    try:
        pers = _sector_persistence(as_of_date, min_turnover_lacs, fwd_days=_H)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("sector persistence failed (%s); gate disabled", exc)
        pers = pd.DataFrame(columns=["sector", "persistence", "pers_n"])
    pmap = pers.set_index("sector")["persistence"] if not pers.empty else pd.Series(dtype=float)
    fac["persistence"] = fac["sector"].map(pmap)
    fac["revert"] = fac["persistence"] < 0                     # NaN → False (unknown ⇒ keep)
    fac.loc[(fac["tilt"] == "OVERWEIGHT") & fac["revert"], "tilt"] = "NEUTRAL"

    # ── regime-inversion gate (OOS 4yr): in downtrends/reversals the overweight call
    # is anti-predictive (OW−UW −0.8%/10d, monotone, every year) — suppress ALL overweights.
    regime["ow_suppressed"] = 0
    if regime.get("momentum_inverts"):
        n_sup = int((fac["tilt"] == "OVERWEIGHT").sum())
        fac.loc[fac["tilt"] == "OVERWEIGHT", "tilt"] = "NEUTRAL"
        regime["ow_suppressed"] = n_sup

    # expected relative return (bps, 10d), scaled by the data-backed regime mult (wide error
    # bars). Under momentum inversion (downtrend/reversal) the high-rank edge is measured to
    # be NEGATIVE, not positive — so we do NOT advertise a positive bps there: zero it out to
    # stay consistent with the suppressed overweights and the STAND-ASIDE verdict.
    if regime.get("momentum_inverts"):
        fac["est_rel_bps"] = 0.0
    else:
        # _REL_SLOPE_BPS is calibrated on the 10-day tercile spread, so it must be
        # scaled to the selected window or a 12-week tilt would quote a 2-week number.
        # Relative drift accumulates roughly linearly in time, so scale by _H/10.
        fac["est_rel_bps"] = ((fac["rank"] - 0.5) * _REL_SLOPE_BPS
                              * (_H / float(_MOM_2W)) * conf_mult).round(0)
    # per-sector confidence: regime × thin × historically-reverting down-weights
    fac["confidence"] = (conf_mult
                         * np.where(fac["thin"], 0.5, 1.0)
                         * np.where(fac["revert"], 0.6, 1.0)).round(2)

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
              AND b.symbol IN (SELECT symbol FROM daily_data
                               WHERE trade_date = ? AND turnover_lacs >= ?)
            ORDER BY b.symbol, b.trade_date
            """,
            [as_of_date, as_of_date, as_of_date, _BREADTH_MIN_TO_LACS],
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
    # duration: consecutive trailing days breadth has held today's strength band (breadth-only,
    # so it is a true time series — the long-line overlay is a today-only read).
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
               death_cross=death, dur_days=int(dur), narrowing=narrowing,
               n=int(p.shape[1]), caption=cap)
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
        WHERE turnover_lacs >= ? AND w_lag IS NOT NULL
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
