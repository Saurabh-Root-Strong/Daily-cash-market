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

MULTI-TIMEFRAME overlay (Daily × Weekly) — VALIDATED walk-forward on the full 371-day
panel (scripts/backtest_mtf_price_action.py; relative-to-universe forward returns, causal):

  A. Daily direction ALONE is mildly CONTRARIAN at the 1–2wk stock horizon (IC −0.03,
     t≈−2.6) — a raw daily pop is a weak fade, not a buy. What persists is trend
     EFFICIENCY (Kaufman ER, IC +0.023 t≈3.0): clean movers continue, direction doesn't.
  B. WEEKLY AGREEMENT is a momentum-quality gate. Among daily-up names, weekly-up win
     52.8% fwd-10d vs weekly-DOWN 46.6% — the daily pop is "false" when the weekly
     structure disagrees. mtf_align encodes the 4 quadrants (D-dir × weekly EMA20/50).
  C. BREAKOUT-from-consolidation is the standout edge: close breaks the 20d Donchian
     high by >1% out of a prior tight (cross-sectional bottom-third ATR) base. On the
     BROAD DCM universe (small+mid+large) this is strong and NOT an overlap artifact —
     NON-overlapping (step=horizon) it is +3.99%/10d relative, t=+5.85, win 62% (n=28
     clean periods); the consolidation filter is ESSENTIAL (plain breakout t≈1.9,
     win<50%). Weekly-up alignment sharpens it — and is now BUILT INTO the label:
     a tight-base up-break is 🚀 Breakout only when the weekly trend agrees, else it
     is split off as ⚠️ False break. The "daily break vs weekly downtrend" case
     (textbook false breakout) is measured near-zero to negative — DCM broad
     +0.5%/10d net win 48%, 4yr F&O OOS −0.6%/10d net win 41% (win 32% in bull
     large-caps) — vs weekly-up break +1.5%/10d t3.0 net; SPREAD +1.0..+2.4pp/DCM
     and +0.3..+0.8pp/F&O, positive every horizon both panels. TRUE weekly-candle
     reads (close>10wSMA, HH-HL) agree on the sign but separate noisier than the
     EMA proxy — do NOT switch. scripts/audit_breakout_weekly_gate.py.
     CAVEATS (4yr OOS audit, 211-name F&O daily 2022–2026, scripts/audit_mtf_price_action_oos.py):
       • UNIVERSE-DEPENDENT — a broad / small-mid-cap effect. On LARGE-CAPS ONLY it is
         weak (t≈1.4); big-cap breakouts are more efficiently priced.
       • BULL-CONCENTRATED — BULL win ~58%, CHOP ~43%, BEAR too thin to fire. Breakouts
         fail outside an uptrend; the DCM window is all-bull so in-sample flatters it.
         Expect decay in chop/bear (same regime risk as the sector-momentum edge).
       • Cost-sensitive on large-caps (~0.3% round-trip eats ~40% there); the broad
         universe (+3.99%/10d) has ample cushion.
  D. DOWN-breakdowns do NOT follow through — they bounce (OOS fwd −0.4%/10d, t−0.7,
     win 54% UP); flagged as bounce-watch, never a short (validated out-of-window).
     Support/resistance PROXIMITY = no edge (IC≈0) — context only, never a call.
     The ALIGNMENT gate is the most robust piece — the weekly-up vs weekly-down win
     spread holds OUT-OF-WINDOW on the independent 4yr F&O panel (+4.0pp OOS).

GUARDRAILS — traps this design DELIBERATELY avoids (scripts/audit_price_action_v3.py,
dual-panel: DCM broad 372d/940k bars + Tradebot 4yr F&O 1018d/210k bars; every masked
event forward-relative, honest non-overlapping t). Do NOT "improve" the engine by adding
any of these — each was measured and FAILS out-of-sample:
  • CANDLESTICK PATTERNS (engulfing / hammer / shooting-star / doji / marubozu / 3-soldiers)
    carry NO reliable swing edge: win-rates 43–52%, non-overlapping t≈0–3, and the SIGN
    flips across panels (hammer +0.80%/10d on DCM but dead and −0.5% in F&O-bull; marubozu-
    DOWN mean-reverts on DCM, weak on F&O). The big t_ov values are n-driven, not edges.
    This RE-CONFIRMS the founding finding (candle anatomy ⟂ trend) on 4 fresh years — hence
    body/wick stay CONTEXT, and only ER + breakout-from-base + alignment are signals.
  • "BIG breakout candle" is a FADE on the live broad universe. Range-expansion (≥1.5×ATR)
    breakouts −0.14%/10d (win 42); volume-surge (≥2×) +0.02% (dead). A 5-factor breakout
    QUALITY score is WORSE than tight-base alone (+0.49 vs +1.92%/10d) because it dilutes
    the one real axis. The edge is the CONSOLIDATION before the break (bottom-third ATR20,
    already `_CONSOL_PCTL`), NOT the drama of the break bar. (≥3 prior level-touches is the
    only add positive on BOTH panels, +1.09/+0.79%/10d, but weaker than tight-base on DCM —
    not worth the complexity.) NOTE the universe flip: on large-cap F&O tight-base is dead
    and volume/range-expansion work — but the live product runs the broad universe, so the
    shipped tight-base definition is correct here.
  • TRUE WEEKLY CANDLES LOSE to the EMA20/50-of-daily proxy for the alignment gate. Horse
    race (ConfirmUp-minus-FalsePop f10 spread), both panels: EMA proxy +0.69pp t6.5 / +0.45pp
    t5.9  >  weekly close>SMA10w +0.35/+0.42  >  weekly HH+HL/LL+LH +0.51/+0.23. The daily
    EMA is a smoother, more responsive weekly-trend read than a resampled weekly bar — do
    NOT switch `_mtf_fields` to W-FRI resampling.
  • PULLBACK (weekly-up + daily-down) cannot be sharpened: shallow-vs-deep doesn't separate
    robustly (DCM flat, F&O deep-better — opposite), and a reversal-candle trigger (hammer/
    engulf) HURTS (+0.17 vs +0.49%/10d). Keep the single undifferentiated Pullback state.
  • WEEKLY candlestick patterns FAIL OOS too (v4, scripts/audit_candle_anatomy_v4.py). On
    DCM every weekly pattern looks positive (marubozu-up +2.0%/20d t2.9, hammer +1.5 t4.4)
    — but marubozu-DOWN is ALSO +1.1% t2.0 and shooting-star +0.65: bearish patterns "work"
    as well as bullish = it's bull-BETA (big weekly bar ⇒ high-beta name), not a pattern edge.
    The 4yr F&O OOS panel breaks it: weekly marubozu-UP +0.31 t1.1 → −0.04 t−0.1 (dead,
    sign-incoherent). Slower bars do NOT rescue the daily-pattern null.
  • The BREAKOUT BAR's OWN anatomy is ANTI-predictive — do NOT add a "clean/marubozu
    breakout candle" filter, it would INVERT the edge. Splitting 🚀 Breakout by break-bar
    body%: the STRONG-body (marubozu-like) break UNDER-performs the weak/wick-y break on
    BOTH panels (DCM +3.1 vs +5.1%/20d; F&O −0.7 vs +1.3 — the marubozu break is NEGATIVE
    on large-caps), and "close in top-25%, no rejection" is −0.6% F&O while "close mid/low
    with upper rejection" is +2.8%. Textbook is backwards here: a giant break bar has spent
    its move intraday and gives back; a modest break has room. Re-confirms "big breakout
    candle is a fade" via body% (v3 had it via range/volume). The edge is the base BEFORE
    the break, never the break candle's shape.
  • DAILY×WEEKLY candle CONFLUENCE (breakout + weekly strong-body / weekly close-near-high)
    is DCM/bull/broad-only: +6.3%/20d t4.4 win 61 on DCM but +0.46 t0.6 (dead) on 4yr F&O —
    collapses OOS exactly like the base breakout's universe-dependence. Not a robust add.
  DATA INTEGRITY verified across runs: 0 OHLC violations, body+wick≡range exactly, splits
  <0.02% of bars — the candle math is sound; edges are absent, not mismeasured.

Public entry point:
  get_price_action(as_of_date, lookback_days=60, min_turnover_lacs) -> DataFrame,
  one row per liquid stock with the raw metrics, the class, the risk flags, and the
  multi-timeframe columns (mtf_align, breakout, w_trend, range_pos).
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

# ── Multi-timeframe (Daily × Weekly) parameters (validated — see module docstring) ──
_D_MOM_WIN   = 10       # daily direction window (trading days)
_W_MOM_WIN   = 50       # ~10-week trend window; also the Donchian range-position window
_DON_WIN     = 20       # breakout Donchian window
_BRK_MARGIN  = 0.01     # close must clear the prior 20d high/low by ≥1%
_CONSOL_PCTL = 0.33     # ATR bottom-third (cross-sectional) = a tight base
_NEAR_EDGE   = 0.20     # within 20% of the range extreme = near support/resistance

# MTF alignment labels (Daily direction × Weekly EMA20/50 structure) — stable strings.
MTF_CONFIRM_UP = "✅ Confirmed Up"     # daily-up + weekly-up  (aligned, best win-rate)
MTF_FALSE_POP  = "⚠️ False Pop"        # daily-up + weekly-DOWN (fade — daily lies here)
MTF_PULLBACK   = "🔵 Pullback"         # daily-down + weekly-up (dip inside an uptrend)
MTF_DOWN_ALGN  = "🔻 Down-trend"       # daily-down + weekly-down (aligned lower)
MTF_NEUTRAL    = "• Neutral"           # no clear direction on one leg
MTF_ALIGN      = (MTF_CONFIRM_UP, MTF_FALSE_POP, MTF_PULLBACK, MTF_DOWN_ALGN, MTF_NEUTRAL)

# Breakout state labels — stable strings.
BRK_BREAKOUT   = "🚀 Breakout"         # tight-base 20d-high break WITH weekly uptrend (the edge)
BRK_FALSEBRK   = "⚠️ False break"      # tight-base 20d-high break AGAINST a weekly downtrend (trap)
BRK_EXTENDED   = "↗ Break (extended)"  # broke out but no prior consolidation (weaker)
BRK_BOUNCE     = "💥 Breakdown-bounce"  # broke 20d low — tends to bounce, NOT a short
BRK_COILING    = "🧊 Coiling"          # tight base, no break yet (watch for the trigger)
BRK_NONE       = "—"
BRK_STATES     = (BRK_BREAKOUT, BRK_FALSEBRK, BRK_EXTENDED, BRK_BOUNCE, BRK_COILING, BRK_NONE)


def get_price_action(
    as_of_date: date,
    lookback_days: int = _LOOKBACK_DAYS,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """Per-stock price-action character as of `as_of_date` (stocks liquid that day)."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    # ~2× the window in calendar days to guarantee the trading bars we need. The MTF
    # overlay needs ~_W_MOM_WIN+_DON_WIN bars for weekly trend + a prior Donchian base,
    # so fetch for whichever window is longer.
    mtf_bars = _W_MOM_WIN + _DON_WIN + 5
    cal_span = int(max(lookback_days, mtf_bars) * 2 + 20)
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
            "mtf_align", "breakout", "w_trend", "range_pos",
            "atr20_pct", "_brk_up", "_brk_dn",
        ])

    P["trade_date"] = pd.to_datetime(P["trade_date"]).dt.date

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for sym, g_full in P.groupby("symbol", sort=False):
            g_full = g_full.sort_values("trade_date")
            g = g_full.tail(lookback_days)             # character = recent window
            if len(g) < _MIN_DAYS:
                continue
            o = g["o"].to_numpy(float); h = g["h"].to_numpy(float)
            l = g["l"].to_numpy(float); c = g["c"].to_numpy(float)
            pc = g["pc"].to_numpy(float)
            rng = h - l
            good = rng > 0
            if good.sum() < _MIN_GOOD_RANGE:
                continue

            # ── Multi-timeframe overlay (uses the fuller series g_full) ──────────
            mtf = _mtf_fields(g_full)

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
                **mtf,
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

    # ── Breakout classification — consolidation tightness is CROSS-SECTIONAL ──────
    # A base is "tight" only relative to the day's universe (matches the validation:
    # bottom-third ATR20). A breakout FROM a tight base is the validated edge; a
    # break with no base is weaker; a broken low tends to bounce (never a short).
    tight = df["atr20_pct"] <= df["atr20_pct"].quantile(_CONSOL_PCTL)

    # A tight-base up-break is the validated edge ONLY when the weekly trend agrees.
    # The same break AGAINST a weekly downtrend is a "false break": measured near-zero
    # to negative on both panels (DCM broad +0.5%/10d net win 48%; 4yr F&O OOS −0.6%
    # net win 41%, worst in BULL large-caps win 32%), vs weekly-up break +1.5%/10d
    # t3.0 net. scripts/audit_breakout_weekly_gate.py. Splitting the label keeps the
    # 🚀 filter honest instead of mixing the two halves. (Extended/bounce/coiling are
    # left un-gated — the edge/false split only carries where the tight-base edge lives.)
    def _breakout(r, is_tight: bool) -> str:
        if r["_brk_up"]:
            if not is_tight:
                return BRK_EXTENDED
            return BRK_BREAKOUT if r["w_trend"] == "↑ up" else BRK_FALSEBRK
        if r["_brk_dn"]:
            return BRK_BOUNCE
        return BRK_COILING if is_tight else BRK_NONE

    df["breakout"] = [
        _breakout(r, bool(t)) for r, (_, t) in zip(
            df.to_dict("records"), tight.items())
    ]
    df = df.drop(columns=["_brk_up", "_brk_dn"])
    return df.reset_index(drop=True)


def _mtf_fields(g_full: pd.DataFrame) -> dict:
    """Daily × Weekly overlay for one stock from its fuller OHLC series.

    Returns the alignment class, weekly-trend label, breakout RAW flags (final
    breakout label is set cross-sectionally by the caller), range position, and the
    20d ATR% used for the cross-sectional consolidation cut. All info is ≤ as_of
    (the last row is the as-of bar) — no lookahead.
    """
    c = g_full["c"].to_numpy(float)
    h = g_full["h"].to_numpy(float)
    l = g_full["l"].to_numpy(float)
    n = len(c)

    # Daily direction (mildly contrarian ALONE — used only as the alignment leg).
    d_dir = 0
    if n > _D_MOM_WIN and c[-1 - _D_MOM_WIN] > 0:
        d_ret = c[-1] / c[-1 - _D_MOM_WIN] - 1.0
        d_dir = 1 if d_ret > 0 else (-1 if d_ret < 0 else 0)

    # Weekly structure via EMA20/EMA50 of daily close (the "real picture" leg).
    s = pd.Series(c)
    ema_s = s.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_l = s.ewm(span=50, adjust=False).mean().iloc[-1]
    w_dir = 1 if ema_s > ema_l else (-1 if ema_s < ema_l else 0)
    w_trend = "↑ up" if w_dir > 0 else ("↓ down" if w_dir < 0 else "→ flat")

    if d_dir > 0 and w_dir > 0:
        align = MTF_CONFIRM_UP
    elif d_dir > 0 and w_dir < 0:
        align = MTF_FALSE_POP
    elif d_dir < 0 and w_dir > 0:
        align = MTF_PULLBACK
    elif d_dir < 0 and w_dir < 0:
        align = MTF_DOWN_ALGN
    else:
        align = MTF_NEUTRAL

    # Breakout raw flags — close clears the PRIOR 20d Donchian (excl. today) by ≥1%.
    brk_up = brk_dn = False
    if n > _DON_WIN:
        prior_hi = float(np.max(h[-1 - _DON_WIN:-1]))
        prior_lo = float(np.min(l[-1 - _DON_WIN:-1]))
        brk_up = c[-1] > prior_hi * (1.0 + _BRK_MARGIN)
        brk_dn = c[-1] < prior_lo * (1.0 - _BRK_MARGIN)

    # Range position over the weekly window (0 = at support, 1 = at resistance) —
    # CONTEXT only (no forward edge), shown to locate the bar, not to trade it.
    win = min(_W_MOM_WIN, n)
    rp_hi = float(np.max(h[-win:])); rp_lo = float(np.min(l[-win:]))
    range_pos = float((c[-1] - rp_lo) / (rp_hi - rp_lo)) if rp_hi > rp_lo else 0.5

    # 20d ATR% for the cross-sectional consolidation cut.
    w20 = min(_DON_WIN, n)
    rng = h[-w20:] - l[-w20:]
    cc = c[-w20:]
    atr20_pct = float(np.nanmean(np.where(cc > 0, rng / cc * 100, np.nan)))

    return {
        "mtf_align": align,
        "w_trend":   w_trend,
        "range_pos": range_pos,
        "atr20_pct": atr20_pct,
        "_brk_up":   bool(brk_up),
        "_brk_dn":   bool(brk_dn),
    }
