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
        SELECT s.sector, b.trade_date,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0            AS daily_dv_cr,
               SUM(b.turnover_lacs * (b.close_price - b.prev_close)
                     / NULLIF(b.prev_close, 0) * 100)
                 / NULLIF(SUM(CASE WHEN b.prev_close > 0
                              THEN b.turnover_lacs END), 0)                  AS wtd_ret_pct
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector NOT IN ('ETF', 'Others')
          AND b.turnover_lacs >= ?
          AND b.trade_date > (?::date - 260)
          AND b.trade_date <= ?
        GROUP BY s.sector, b.trade_date
        ORDER BY s.sector, b.trade_date
        """,
        [min_turnover_lacs, as_of_date, as_of_date],
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
    vol20 = ret.rolling(20).std()
    vol_pct = float((vol20 <= vol20.iloc[-1]).mean()) if vol20.notna().sum() > 20 else float("nan")
    ret_5d = float(_compound(nf["nret"], 5).iloc[-1])
    ret_20d = float(_compound(nf["nret"], 20).iloc[-1])
    ret_med = float(_compound(nf["nret"], _MED_TREND_WIN).iloc[-1])   # ~2-month trend
    ema_slope = float(ema20_s.iloc[-1] - ema20_s.iloc[-1 - _EMA_SLOPE_WIN]) if len(ema20_s) > _EMA_SLOPE_WIN else 0.0

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
                confidence_mult=float(mult), verdict=verdict, size_hint=float(size),
                action=_ACTION[verdict], posture=posture, banner=banner)


def _sector_persistence(as_of_date: date, min_turnover_lacs: float) -> pd.DataFrame:
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
        SELECT s.sector, b.trade_date,
               SUM(b.turnover_lacs * (b.close_price - b.prev_close)
                     / NULLIF(b.prev_close, 0) * 100)
                 / NULLIF(SUM(CASE WHEN b.prev_close > 0
                              THEN b.turnover_lacs END), 0)                  AS wtd_ret_pct
        FROM daily_data b
        INNER JOIN v_sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector NOT IN ('ETF', 'Others')
          AND b.turnover_lacs >= ?
          AND b.trade_date > (?::date - ?)
          AND b.trade_date <= ?
        GROUP BY s.sector, b.trade_date
        ORDER BY b.trade_date
        """,
        [min_turnover_lacs, as_of_date, _PERS_LOOKBACK_CAL, as_of_date],
    )
    if panel.empty:
        return empty
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    ret = panel.pivot_table("wtd_ret_pct", "trade_date", "sector").sort_index()
    if len(ret) < _PERS_FWD + _PERS_MIN_OBS:
        return empty
    # forward 10d compounded return per sector (NaN for the last _PERS_FWD rows → causal)
    cr = np.log1p(ret / 100.0).cumsum()
    fwd = (np.expm1(cr.shift(-_PERS_FWD) - cr) * 100.0)
    edge = fwd.sub(fwd.median(axis=1), axis=0)            # vs cross-sectional median sector
    out = pd.DataFrame({
        "persistence": edge.mean(axis=0, skipna=True),
        "pers_n": edge.notna().sum(axis=0),
    })
    out = out[out["pers_n"] >= _PERS_MIN_OBS].reset_index().rename(columns={"index": "sector"})
    if "sector" not in out.columns:
        out = out.rename(columns={out.columns[0]: "sector"})
    return out


def get_forward_tilt(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
    horizon_days: int = _MOM_2W,
) -> tuple[pd.DataFrame, dict]:
    """Per-sector 1–2wk forward tilt + regime meta. Causal (data <= as_of_date)."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

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
        if len(g) < _MIN_HIST:
            continue
        mom_2w = float(_compound(g["wtd_ret_pct"], _MOM_2W).iloc[-1])
        mom_1w = float(_compound(g["wtd_ret_pct"], _MOM_1W).iloc[-1])
        dv = g["daily_dv_cr"].astype(float)
        base = dv.iloc[:-1].tail(_DV_BASE).mean()
        dv5d = float(dv.tail(_DV_FLOW).mean() / base) if base and base > 0 else np.nan
        recs.append(dict(sector=sector, mom_2w=mom_2w, mom_1w=mom_1w, dv5d=dv5d))
    fac = pd.DataFrame(recs)
    if len(fac) < _MIN_SECTORS:
        return empty, regime

    # relative strength vs Nifty (fallback to absolute momentum if Nifty missing)
    if not nifty.empty:
        n_1w = float(_compound(nifty["nret"], _MOM_1W).iloc[-1])
        n_2w = float(_compound(nifty["nret"], _MOM_2W).iloc[-1])
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
        pers = _sector_persistence(as_of_date, min_turnover_lacs)
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
        fac["est_rel_bps"] = ((fac["rank"] - 0.5) * _REL_SLOPE_BPS * conf_mult).round(0)
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
