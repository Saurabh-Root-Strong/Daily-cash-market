"""
Index Prediction Engine — per-index next-day directional forecast.

24-signal quant framework — institutional-grade + statistical regime detection:
Core 18 signals apply to all indices (Signal 5 = OI-Premium Matrix, new).
Signals 19–21 are Nifty 50 exclusive (weekly + monthly OI bifurcation).
Signals 22–24 are Statistical Regime signals applied to all indices.

  ── Statistical Regime Detection (Signals 21–23, all indices) ────────────────
  21. Market Memory       Hurst R/S + DFA — autocorrelation structure of returns.
                          H>0.58 = trending (follow signals); H<0.42 = mean-reverting
                          (fade signals). Quantifies HOW the market moves, not just where.
  22. HMM Regime          3-state Gaussian Hidden Markov Model on 90D returns.
                          Detects the latent regime: Bull / Sideways / Bear.
                          Score is probability-weighted: p_bull×3 − p_bear×3.
                          Fitted with Baum-Welch; decoded with scaled forward-backward.
  23. Entropy             Permutation Entropy (Bandt & Pompe) on 60D returns.
                          Low PE (<0.50) = ordered/predictable → confidence boost.
                          High PE (>0.72) = chaotic/noisy → confidence reduction.
                          Shannon complexity of ordinal return patterns.

  ── Price / Futures / Options (index-specific) ──────────────────────────────
   1. OI-Price Matrix       Murphy (Technical Analysis of Financial Markets)
   2. Cost of Carry         Hull (Options, Futures & Other Derivatives)
   3. Max Pain              McMillan (Options as a Strategic Investment)
   4. PCR Contrarian        Put-Call Ratio as fear/greed indicator
   5. OI-Premium Matrix     Disambiguates PCR — identifies buyers vs writers
                            OI↑+Prem↑=Buying | OI↑+Prem↓=Writing | OI↓+Prem↑=SC
                            C.Buy+P.Write=Net Long | C.Write+P.Buy=Net Short
   6. Price Mean-Reversion  90-day NSE backtest — 67% oversold bounce rate
   6. Wyckoff Range Position  Close vs day's H-L range

  ── Institutional / FII Analytics (all sourced from NSE publications) ────────
   7. FII Institutional     FAO participant net OI — FII vs Client divergence
                            (Schwager: follow smart money, fade the crowd)
   8. FII Options Delta     FAO net call-OI minus net put-OI (directional bias)
   9. FII Flow Today        fii_derivatives_stats: today's ₹Cr buy vs sell
  10. FII 5-Day Cumulative  5D rolling ₹Cr flow — sustained pressure detection
  11. FII OI Buildup        FII OI ₹Cr trend: conviction build vs position unwind
  12. FII Position Change   Day-over-day FAO net change: covering vs adding
  13. Short Squeeze Setup   Livermore's trap: FII trapped short + market rising
                            + extreme PCR = forced covering fuel (explosive rally)

  ── Market-Wide Context (computed once, shared across all 4 indices) ─────────
  14. India VIX Regime      Whaley: VIX regime + trend signal
  15. Sector Breadth        Zweig (Winning on Wall Street): participation quality
  16. Defensive/Cyclical    O'Neil (CANSLIM): sector rotation leads index
  17. PE Valuation          Nifty PE ratio as background context
  17b.Index Relative Strength  BankNifty / FinNifty / MidcapNifty ONLY
                               Today's index return vs Nifty 50 → sector rotation direction
                               RS ≥ +0.75% = capital rotating IN (bullish momentum)
                               RS ≤ −0.75% = capital rotating OUT (distribution)
                               Mansfield RS / O'Neil CANSLIM principle.
  17c.Constituent Breadth      NIFTY · BANKNIFTY · FINNIFTY — stock-level breadth + delivery quality
                               Turnover-weighted advance ratio across all constituent stocks
                               Score ≥ 75% turnover advancing = +2.5 exceptional breadth
                               Delivery ≥ 55% on advancers = +0.5 institutional accumulation
                               Reveals WHO is buying (institutions vs day traders)
  17d.Constituent Fut OI       NIFTY · BANKNIFTY · FINNIFTY — stock futures OI-Price matrix consensus
                               Fresh long/short buildup across constituent stocks
                               Stock futures = always directional (no hedge use case)
                               net bull ≥ 30% above bear = +2.0 basket long conviction
  17e.Constituent Opt OI-Prem  NIFTY · BANKNIFTY · FINNIFTY — stock options OI-Premium matrix basket
                               C.Buy+P.Write = Net Long per stock; aggregated across basket
                               Disambiguates PCR: OI↑+Prem↑=Buying vs OI↑+Prem↓=Writing
                               net_long_pct - net_short_pct ≥ 0.25 = +2.0 basket bull

  ── Multi-Expiry (Nifty 50 only — weekly + monthly expiry both available) ─────
  18. Cross-Expiry PCR       Weekly PCR vs Monthly PCR divergence
                             Near-term fear ≠ medium-term fear → timeframe edge
  19. Dual Max Pain          Weekly max pain vs monthly max pain convergence
                             Both GPs point same direction = double pin strength
  20. Gamma Wall Structure   Weekly OI / total OI ratio
                             >60% weekly = near-term pinning; <30% = directional scope

THEORY BEHIND KEY SIGNALS:
  OI-Price Matrix: Price ↑ + OI ↑ = fresh longs (conviction); Price ↑ + OI ↓ = short covering
  Carry: F = S·e^(r-d)·T; India repo ~6.5%; carry above fair value = bullish demand
  Short Squeeze: Livermore — "The market will hurt the most people." FII short + PCR extreme
                 + market rising = shorts trapped and must cover into upward price discovery
  FII Cumulative Flow: Institutional trend confirmation — single-day flows can be noise;
                       5D cumulative reveals systematic accumulation vs distribution

Indices: NIFTY  ·  BANKNIFTY  ·  FINNIFTY  ·  MIDCPNIFTY
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_index_predictions",
    "IndexPrediction", "IndexSignal", "IndexKeyLevels", "DaySnapshot", "MarketContext",
]

# ── Constants ─────────────────────────────────────────────────────────────────

_INDEX_MAP: dict[str, tuple[str, str]] = {
    "NIFTY":      ("Nifty 50",     "Nifty 50"),
    "BANKNIFTY":  ("Bank Nifty",   "Nifty Bank"),
    "FINNIFTY":   ("Fin Nifty",    "Nifty Financial Services"),
    "MIDCPNIFTY": ("Midcap Nifty", "Nifty Midcap Select"),
}

# Maps FNO symbol → fii_derivatives_stats category (futures and options)
_SYMBOL_TO_FII_FUT: dict[str, str] = {
    "NIFTY":      "NIFTY FUTURES",
    "BANKNIFTY":  "BANKNIFTY FUTURES",
    "FINNIFTY":   "FINNIFTY FUTURES",
    "MIDCPNIFTY": "MIDCPNIFTY FUTURES",
}
_SYMBOL_TO_FII_OPT: dict[str, str] = {
    "NIFTY":      "NIFTY OPTIONS",
    "BANKNIFTY":  "BANKNIFTY OPTIONS",
    "FINNIFTY":   "FINNIFTY OPTIONS",
    "MIDCPNIFTY": "MIDCPNIFTY OPTIONS",
}

_CYCLICAL_SECTORS = [
    "Nifty Auto", "Nifty Realty", "Nifty Metal",
    "Nifty Infrastructure", "Nifty PSU Bank", "Nifty Private Bank",
]
_DEFENSIVE_SECTORS = [
    "Nifty FMCG", "Nifty Pharma",
    "Nifty IT", "Nifty Consumer Durables",
]

# Canonical NSE sectoral indices for the breadth signal (Zweig) — one entry per
# distinct economic sector. index_data also carries 100+ broad-market clones
# (Nifty 100/500, Total Market), factor/strategy indices (Alpha, Quality, Low-Vol),
# corporate-group baskets and even a G-sec bond index; counting those as "sectors"
# double-counts the same large caps and dilutes breadth toward the index itself.
_SECTOR_BREADTH_INDICES = [
    "Nifty Auto", "Nifty Private Bank", "Nifty PSU Bank", "Nifty FMCG",
    "Nifty IT", "Nifty Media", "Nifty Metal", "Nifty Pharma", "Nifty Realty",
    "Nifty Consumer Durables", "Nifty Oil & Gas", "Nifty Energy",
    "Nifty Infrastructure", "Nifty Chemicals", "Nifty Capital Markets",
]

_INDIA_REPO = 6.5   # annualised % — RBI repo for fair-value carry

# Minimum share of total FII index-futures OI an index must hold before the AGGREGATE
# FII positioning signals (which are pooled across all index futures) may be attributed
# to it. On current data: NIFTY ~76%, BankNifty ~17% (pass); Midcap ~6%, FinNifty ~0.1%
# (suppressed — those indices use their per-index FII flow signals instead).
_FII_FUT_OI_SHARE_MIN = 0.10

# Options-chain liquidity gate. PCR / max-pain / OI-premium only reflect real positioning
# when the near-expiry chain is liquid AND reasonably distributed. Below the OI floor or
# above the single-strike concentration ceiling the chain is too thin/lumpy to read
# (FinNifty: ~90k OI with ~half the call OI in one far-OTM strike → PCR 0.30 and walls are
# noise). On current data this isolates FinNifty while sparing Midcap (~722k), BankNifty
# (~1.6M) and NIFTY (~70M).
_OPT_OI_MIN   = 250_000   # near-expiry total option OI (contracts)
_OPT_CONC_MAX = 0.40      # max fraction of total OI allowed in a single strike

# Index-futures carry is pinned near cost-of-carry (~repo 6.5% − div yield, ≈5% ann) by
# cash-futures arbitrage. A reading far outside this band is a thin/stale settlement-price
# artifact (e.g. Midcap 38% ann off a thin monthly contract), not tradeable carry — the
# directional carry signal is suppressed outside the band. Legitimate stress backwardation
# (down to ~−12%) is preserved.
_CARRY_MIN_ANN = -12.0
_CARRY_MAX_ANN = 18.0

# ── Index constituent symbol tuples ──────────────────────────────────────────
# Used for Signals 17c/d/e: breadth + delivery quality, stock futures OI-Price
# matrix, and stock options OI-Premium matrix applied per index basket.
# Source: NSE CSV files  |  Last reviewed: Jun 2026
_NIFTY50_SYMBOLS: tuple = (
    "ADANIENT",   "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",  "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL",         "BHARTIARTL",
    "CIPLA",      "COALINDIA",  "DRREDDY",    "EICHERMOT",   "ETERNAL",
    "GRASIM",     "HCLTECH",    "HDFCBANK",   "HDFCLIFE",    "HINDALCO",
    "HINDUNILVR", "ICICIBANK",  "ITC",        "INFY",        "INDIGO",
    "JSWSTEEL",   "JIOFIN",     "KOTAKBANK",  "LT",          "M&M",
    "MARUTI",     "MAXHEALTH",  "NTPC",       "NESTLEIND",   "ONGC",
    "POWERGRID",  "RELIANCE",   "SBILIFE",    "SHRIRAMFIN",  "SBIN",
    "SUNPHARMA",  "TCS",        "TATACONSUM", "TMPV",        "TATASTEEL",
    "TECHM",      "TITAN",      "TRENT",      "ULTRACEMCO",  "WIPRO",
)

_BANKNIFTY_SYMBOLS: tuple = (
    "AUBANK",     "AXISBANK",   "BANKBARODA", "CANBK",       "FEDERALBNK",
    "HDFCBANK",   "ICICIBANK",  "IDFCFIRSTB", "INDUSINDBK",  "KOTAKBANK",
    "PNB",        "SBIN",       "UNIONBANK",  "YESBANK",
)

_FINNIFTY_SYMBOLS: tuple = (
    "AXISBANK",   "BSE",        "BAJFINANCE", "BAJAJFINSV",  "CHOLAFIN",
    "HDFCBANK",   "HDFCLIFE",   "ICICIBANK",  "ICICIGI",     "JIOFIN",
    "KOTAKBANK",  "LICHSGFIN",  "MFSL",       "MUTHOOTFIN",  "PFC",
    "RECLTD",     "SBICARD",    "SBILIFE",    "SHRIRAMFIN",  "SBIN",
)

# Maps fno_symbol → (symbol universe, short label, full label) for signals 17c/d/e
_CONSTITUENT_SYMBOLS: dict[str, tuple] = {
    "NIFTY":     _NIFTY50_SYMBOLS,
    "BANKNIFTY": _BANKNIFTY_SYMBOLS,
    "FINNIFTY":  _FINNIFTY_SYMBOLS,
}
_CONSTITUENT_INFO: dict[str, tuple[str, str]] = {
    "NIFTY":     ("N50", "Nifty 50"),
    "BANKNIFTY": ("BN",  "Nifty Bank"),
    "FINNIFTY":  ("FN",  "Fin Nifty"),
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class IndexSignal:
    name: str
    category: str   # Price Action | Futures OI | Options OI | Carry |
                    # Institutional | Market Context | Sector
    direction: int  # +1 bullish | −1 bearish | 0 neutral
    score: float
    headline: str
    description: str
    emoji: str = ""


@dataclass
class IndexKeyLevels:
    max_pain: Optional[float] = None
    top_call_strike: Optional[float] = None
    top_put_strike: Optional[float] = None
    second_call_strike: Optional[float] = None
    second_put_strike: Optional[float] = None
    call_oi_at_top: int = 0
    put_oi_at_top: int = 0


@dataclass
class DaySnapshot:
    """One day's key metrics — used for today vs yesterday comparison."""
    trade_date: date
    spot_close: Optional[float]
    pct_chg: Optional[float]
    high: Optional[float]
    low: Optional[float]
    fut_settle: Optional[float]
    fut_oi: int
    call_oi: int
    put_oi: int
    pcr: Optional[float]
    total_volume: int
    carry_pts: Optional[float]
    carry_pct_ann: Optional[float]


@dataclass
class _ConstituentStats:
    """Constituent-level data for Signals 17c/d/e — one instance per index basket."""
    total_stocks:  int   = 0   # size of the symbol universe (used for threshold scaling)
    # 17c — cash breadth + delivery quality
    coverage:      int   = 0
    advancing:     int   = 0
    declining:     int   = 0
    adv_turn_pct:  Optional[float] = None
    adv_deliv_avg: Optional[float] = None
    dec_deliv_avg: Optional[float] = None
    # 17d — stock futures OI-Price matrix
    fut_valid:       int = 0
    fut_fresh_long:  int = 0
    fut_fresh_short: int = 0
    fut_long_unwind: int = 0
    fut_short_cov:   int = 0
    # 17e — stock options OI-Premium matrix
    opt_valid:     int = 0
    opt_net_long:  int = 0
    opt_net_short: int = 0
    opt_straddle:  int = 0
    opt_range:     int = 0


@dataclass
class MarketContext:
    """
    Market-wide context computed ONCE per trade_date, shared across all 4 index predictions.
    Sources: fao_participant, fii_derivatives_stats, index_data (VIX + PE + sectors).
    """
    trade_date: date

    # ── India VIX ─────────────────────────────────────────────────────────────
    vix_close: Optional[float] = None
    vix_pct_chg: Optional[float] = None
    vix_5d_chg_pct: Optional[float] = None
    vix_regime: str = "Unknown"

    # ── FAO participant OI (most recent date ≤ trade_date; may lag 1-3 days) ──
    fao_date: Optional[date] = None
    fii_fut_idx_net: int = 0
    dii_fut_idx_net: int = 0
    client_fut_idx_net: int = 0
    pro_fut_idx_net: int = 0
    fii_opt_call_net: int = 0       # FII: call_long − call_short
    fii_opt_put_net: int = 0        # FII: put_long  − put_short
    fii_opt_delta: int = 0          # call_net − put_net
    # Per-participant options delta (call_net − put_net) for the role-aware options read:
    # Client = retail (contrarian), Pro = informed desks, DII = minor.
    client_opt_delta: int = 0
    pro_opt_delta: int = 0
    dii_opt_delta: int = 0

    # ── FAO day-over-day position change (adds conviction to direction) ────────
    fii_prev_fao_date: Optional[date] = None
    fii_fut_idx_net_prev: int = 0           # FII net on the PREVIOUS FAO date
    fii_net_change_1d: int = 0              # positive = covering shorts; negative = adding

    # ── FII derivatives stats — today's ₹Cr flow ─────────────────────────────
    fii_stats_date: Optional[date] = None
    fii_symbol_flows: dict = field(default_factory=dict)    # fno_symbol → net ₹Cr (futures)
    fii_opt_net_flows: dict = field(default_factory=dict)   # fno_symbol → net ₹Cr (options)
    fii_stock_fut_net_cr: Optional[float] = None            # FII stock-futures net (market-neutral clue)

    # ── FII 5-day cumulative futures flow ₹Cr ────────────────────────────────
    fii_cumul_flows_5d: dict = field(default_factory=dict)  # fno_symbol → 5D sum ₹Cr

    # ── FII OI ₹Cr buildup trend (conviction vs unwinding) ───────────────────
    fii_oi_cr_latest: dict = field(default_factory=dict)    # fno_symbol → latest OI ₹Cr
    fii_oi_cr_5d_ago: dict = field(default_factory=dict)    # fno_symbol → OI ₹Cr 5D ago

    # ── Sector breadth ────────────────────────────────────────────────────────
    breadth_pct_advancing: Optional[float] = None
    advancing_sectors: int = 0
    declining_sectors: int = 0
    total_sectors: int = 0

    # ── Cyclical vs Defensive rotation ───────────────────────────────────────
    cyclical_avg_chg: Optional[float] = None
    defensive_avg_chg: Optional[float] = None
    risk_mode: str = "MIXED"

    # ── PE valuation ──────────────────────────────────────────────────────────
    nifty_pe: Optional[float] = None

    # ── Nifty 50 daily return — used for non-Nifty relative-strength signal ──
    nifty_pct_chg: Optional[float] = None

    # ── Constituent-level data (Signals 17c/d/e): NIFTY, BANKNIFTY, FINNIFTY ─
    # Maps fno_symbol → _ConstituentStats. Populated for all three in
    # _build_market_context() via _compute_constituent_stats().
    constituent_stats: dict = field(default_factory=dict)


@dataclass
class IndexPrediction:
    fno_symbol: str
    display_name: str
    as_of_date: date

    spot_close: Optional[float]
    prev_close: Optional[float]
    day_change_pct: Optional[float]
    high: Optional[float]
    low: Optional[float]

    near_expiry: Optional[date]
    days_to_expiry: int
    futures_price: Optional[float]
    carry_pts: Optional[float]
    carry_pct_ann: Optional[float]
    carry_label: str

    fut_oi: int
    fut_oi_chg: int
    call_oi: int
    put_oi: int
    pcr: Optional[float]

    today: Optional[DaySnapshot]
    yesterday: Optional[DaySnapshot]
    levels: IndexKeyLevels

    direction: str
    confidence: str
    direction_color: str
    composite_score: float
    headline: str
    key_driver: str
    key_risk: str
    signals: list[IndexSignal] = field(default_factory=list)
    data_available: bool = True
    note: str = ""
    market_context: Optional[MarketContext] = None

    # ── Statistical Regime Detection (signals 21-23) ──────────────────────────
    regime: Optional[object] = None    # RegimeResult

    # ── Prediction Memory Engine (signal 24) ──────────────────────────────────
    mem_signal: Optional[object] = None   # MemorySignal

    # ── Multi-expiry (Nifty 50 only: weekly + monthly bifurcation) ────────────
    # After Sep 2025 NSE change: NIFTY weekly = every Tuesday; monthly = last Tuesday.
    # BANKNIFTY / FINNIFTY / MIDCPNIFTY = monthly-only (last Tuesday); no weekly contracts.
    weekly_expiry: Optional[date] = None    # near options expiry (weekly Tuesday for NIFTY)
    monthly_expiry: Optional[date] = None   # last Tuesday of calendar month
    weekly_pcr: Optional[float] = None      # PCR for weekly expiry
    monthly_pcr: Optional[float] = None     # PCR for monthly expiry
    weekly_call_oi: int = 0
    weekly_put_oi: int = 0
    monthly_call_oi: int = 0
    monthly_put_oi: int = 0
    monthly_max_pain: Optional[float] = None
    gamma_ratio: Optional[float] = None     # weekly_oi / (weekly_oi + monthly_oi)

    # ── Expected-Move Forecast (magnitude — "how many points") ────────────────
    expected_move_pts: Optional[float] = None    # 1σ daily expected move (points, ~68% band)
    expected_move_pct: Optional[float] = None    # same as % of spot
    range_low: Optional[float] = None            # spot − expected_move (68% lower bound)
    range_high: Optional[float] = None           # spot + expected_move (68% upper bound)
    target_move_pts: Optional[float] = None      # directional target move (signed, conviction-scaled)
    target_close: Optional[float] = None         # spot + target_move (predicted close)
    sideways_band_pts: Optional[float] = None    # ± threshold below which = sideways
    move_basis: str = ""                         # one-line explanation of the calc

    # ── Breakout Extension Ranges (second range if first range is broken) ─────
    # Triggered when price moves >breakout_threshold_pts beyond range_low/range_high.
    # Extension target = nearest OI wall inside the 2σ zone, or 2σ if no wall found.
    breakout_threshold_pts: Optional[float] = None  # pts beyond range to confirm breakout
    breakout_up_start: Optional[float] = None       # upside break confirmation level
    breakout_up_end: Optional[float] = None         # second range upper target
    breakout_dn_start: Optional[float] = None       # second range lower target
    breakout_dn_end: Optional[float] = None         # downside break confirmation level
    breakout_up_src: str = ""                       # driver: "call wall 23900" / "2σ statistical"
    breakout_dn_src: str = ""                       # driver: "put wall 23100" / "2σ statistical"

    # ── Weekly (to-expiry) range — where price can travel until the NEXT expiry ──
    # Forward-looking, recomputed each EOD: rolls to the next expiry on expiry day,
    # shrinks as expiry approaches. Built from the ATM straddle (market-implied move),
    # σ√T statistical move, and the OI walls / max-pain that writers defend.
    wk_expiry: Optional[date] = None                # the target expiry this range runs to
    wk_dte_cal: Optional[int] = None                # calendar days to that expiry
    wk_kind: str = ""                               # "weekly" | "monthly"
    wk_move_pts: Optional[float] = None             # expected move to expiry (points)
    wk_range_low: Optional[float] = None            # spot − weekly move
    wk_range_high: Optional[float] = None           # spot + weekly move
    wk_straddle_pts: Optional[float] = None         # ATM straddle premium (implied move)
    wk_put_floor: Optional[float] = None            # put-wall support (OI-defended)
    wk_call_cap: Optional[float] = None             # call-wall resistance (OI-defended)
    wk_max_pain: Optional[float] = None             # gravity / pin into expiry
    wk_basis: str = ""                              # one-line explanation

    # ── Monthly (to monthly-expiry) range — NIFTY only, built from MONTHLY-expiry OI ──
    # Distinct from wk_* (the weekly): a wider, slower scenario band to the monthly
    # expiry, off the monthly chain's ATM straddle + monthly OI walls + monthly max pain.
    # Empty for Bank/Fin/Midcap (monthly-only → their wk_* range already IS the monthly).
    mn_expiry: Optional[date] = None                # the monthly expiry this range runs to
    mn_dte_cal: Optional[int] = None                # calendar days to the monthly expiry
    mn_kind: str = ""                               # "monthly"
    mn_move_pts: Optional[float] = None             # expected move to monthly expiry (points)
    mn_range_low: Optional[float] = None            # spot − monthly move
    mn_range_high: Optional[float] = None           # spot + monthly move
    mn_straddle_pts: Optional[float] = None         # monthly ATM straddle premium
    mn_put_floor: Optional[float] = None            # monthly put-wall support
    mn_call_cap: Optional[float] = None             # monthly call-wall resistance
    mn_max_pain: Optional[float] = None             # monthly max pain (pin into monthly expiry)
    mn_basis: str = ""                              # one-line explanation

    # ── Price-structure S/R (swing-pivot zones from daily OHLC — Signal 6b) ──
    sr_support: Optional[float] = None              # nearest multi-touch support below spot
    sr_resistance: Optional[float] = None           # nearest multi-touch resistance above spot
    sr_support_touches: int = 0                     # pivot touches at that support
    sr_resistance_touches: int = 0                  # pivot touches at that resistance


# ── Public entry point ────────────────────────────────────────────────────────

def get_index_predictions(
    trade_date: date, persist: bool = True,
    market_ctx: Optional[MarketContext] = None,
) -> list[IndexPrediction]:
    """
    Compute next-day predictions for all major F&O indices.

    persist=True (default, used by cmd_daily) logs each prediction to
    prediction_log. The read-only dashboard passes persist=False to avoid a write
    on every render. market_ctx lets a caller pass a pre-built (cached) context so
    the ~7s-cold MarketContext isn't rebuilt per call.
    """
    ctx = market_ctx if market_ctx is not None else _build_market_context(trade_date)
    return [
        _compute_prediction(trade_date, sym, name, idx_name, ctx, persist=persist)
        for sym, (name, idx_name) in _INDEX_MAP.items()
    ]


def get_engine_accuracy(min_n: int = 20, fno_symbol: Optional[str] = None) -> dict:
    """
    HONEST self-validation — the engine's REALIZED next-day track record from
    prediction_log (logged daily by cmd_daily, outcomes filled after the fact).

    fno_symbol=None pools all indices; pass a symbol (e.g. "NIFTY") for the
    PER-INDEX track record — pooling hides the worst case (NIFTY's hit-rate is far
    below the blended average), so the per-index number is what the verdict card
    must show next to its own call.

    Returns {} when too few outcomes. Otherwise: n, directional sign hit-rate,
    composite→return rank IC, and the date range. This is published on the page so
    the verdict is never trusted more than its measured accuracy warrants.
    """
    try:
        sql = ("SELECT direction_pred, composite_score, actual_return, was_correct, trade_date "
               "FROM prediction_log WHERE outcome_filled AND actual_return IS NOT NULL")
        params: list = []
        if fno_symbol:
            sql += " AND fno_symbol = ?"
            params.append(fno_symbol)
        df = query_dataframe(sql, params)
    except Exception:
        return {}
    if df is None or df.empty or len(df) < min_n:
        return {}
    df["ar"] = df["actual_return"].astype(float)
    d = df[df["direction_pred"].isin(["UP", "DOWN"])].copy()
    if d.empty:
        return {}
    pred_up = d["direction_pred"].eq("UP")
    act_up  = d["ar"] > 0
    sign_hit = float((pred_up == act_up).mean() * 100)
    try:
        ic = float(df["composite_score"].astype(float).rank().corr(df["ar"].rank()))
    except Exception:
        ic = float("nan")
    return {
        "symbol": fno_symbol or "ALL",
        "n": int(len(df)), "n_dir": int(len(d)),
        "sign_hit": round(sign_hit, 0), "ic": round(ic, 3),
        "since": str(df["trade_date"].min())[:10], "until": str(df["trade_date"].max())[:10],
    }


def get_index_prediction_for(
    trade_date: date, fno_symbol: str, persist: bool = True,
    market_ctx: Optional[MarketContext] = None,
) -> Optional[IndexPrediction]:
    """
    Compute the next-day prediction for a SINGLE index.

    Views that show one index at a time (e.g. Prediction Memory) should call this
    rather than get_index_predictions(), which computes all four. market_ctx lets
    the caller supply a cached MarketContext so switching index doesn't rebuild it.
    persist defaults True; the dashboard passes persist=False.
    """
    entry = _INDEX_MAP.get(fno_symbol)
    if entry is None:
        return None
    name, idx_name = entry
    ctx = market_ctx if market_ctx is not None else _build_market_context(trade_date)
    return _compute_prediction(trade_date, fno_symbol, name, idx_name, ctx, persist=persist)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore(value: float, series: pd.Series) -> float:
    std = float(series.std())
    return 0.0 if std < 1e-9 else (value - float(series.mean())) / std


def _carry_label(carry_pct_ann: float) -> str:
    if carry_pct_ann >= 9:   return "Premium — Bullish Demand"
    if carry_pct_ann >= 4:   return "Near Fair Value — Neutral"
    if carry_pct_ann >= 0:   return "Slight Discount — Mild Bearish"
    return "Backwardation — Bearish / Stress"


def _lag_mult(lag_days: int) -> float:
    """Score decay for stale institutional data (FAO participant / FII derivatives stats).

    0d→1.00 | 1d→0.80 | 2d→0.60 | 3d→0.40 | ≥4d→0.30 (floor)

    A 3-day-old FAO snapshot fires at 40% weight — still informative as a structural
    context signal, but does not dominate the composite as if it were fresh.
    Applied multiplicatively to every institutional signal score except the zero-score
    FII options put-hedge (which is context-only and already carries no vote weight).
    """
    return max(0.30, 1.0 - lag_days * 0.20)


def _to_date(v) -> Optional[date]:
    if v is None: return None
    return v.date() if hasattr(v, "date") else v


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_index_history(index_name: str, trade_date: date, lookback: int = 60) -> pd.DataFrame:
    # ~60 trading-day window (×2 calendar buffer) — enough for RSI(14), EMA50 trend
    # regime and a 20-day anchored VWAP. volume/turnover_cr drive the VWAP weight.
    start = trade_date - timedelta(days=lookback * 2)
    df = query_dataframe("""
        SELECT trade_date, open_val, high_val, low_val, close_val, prev_close, pct_chg,
               volume, turnover_cr
        FROM index_data
        WHERE index_name = ? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """, [index_name, start, trade_date])
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def _load_fno_for_date(symbol: str, trade_date: date) -> pd.DataFrame:
    df = query_dataframe("""
        SELECT instrument, expiry_date, option_type, strike_price,
               open_price, high_price, low_price, close_price, settle_price,
               contracts, value_lacs, open_interest, chg_in_oi
        FROM fno_bhavcopy
        WHERE symbol = ? AND trade_date = ?
          AND instrument IN ('FUTIDX', 'OPTIDX')
        ORDER BY expiry_date, strike_price
    """, [symbol, trade_date])
    if not df.empty:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    return df


def _get_two_fno_dates(symbol: str, trade_date: date) -> tuple[Optional[date], Optional[date]]:
    df = query_dataframe("""
        SELECT DISTINCT trade_date FROM fno_bhavcopy
        WHERE symbol = ? AND trade_date <= ?
          AND instrument IN ('FUTIDX', 'OPTIDX')
        ORDER BY trade_date DESC LIMIT 2
    """, [symbol, trade_date])
    if df.empty: return None, None
    dates = [_to_date(d) for d in df["trade_date"].tolist()]
    return dates[0], (dates[1] if len(dates) > 1 else None)


def _load_vix_history(trade_date: date, lookback: int = 12) -> pd.DataFrame:
    start = trade_date - timedelta(days=lookback * 2)
    df = query_dataframe("""
        SELECT trade_date, close_val, prev_close, pct_chg FROM index_data
        WHERE index_name = 'India VIX' AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """, [start, trade_date])
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def _load_fao_participant_two_dates(trade_date: date) -> pd.DataFrame:
    """Last 2 available FAO OI dates (all participants) for day-over-day position change."""
    df = query_dataframe("""
        WITH latest2 AS (
            SELECT DISTINCT trade_date FROM fao_participant
            WHERE data_type = 'OI' AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 2
        )
        SELECT fp.*
        FROM fao_participant fp
        WHERE fp.trade_date IN (SELECT trade_date FROM latest2)
          AND fp.data_type = 'OI'
        ORDER BY fp.trade_date DESC
    """, [trade_date])
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def _load_fii_stats_history(trade_date: date, lookback: int = 8) -> pd.DataFrame:
    """FII derivatives stats for the last N available trading dates."""
    df = query_dataframe("""
        WITH latest_dates AS (
            SELECT DISTINCT trade_date FROM fii_derivatives_stats
            WHERE trade_date <= ?
            ORDER BY trade_date DESC LIMIT ?
        )
        SELECT fds.*
        FROM fii_derivatives_stats fds
        WHERE fds.trade_date IN (SELECT trade_date FROM latest_dates)
        ORDER BY fds.trade_date DESC, fds.category
    """, [trade_date, lookback])
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def _load_nifty_pe(trade_date: date) -> Optional[float]:
    """Nifty 50 PE ratio on or before trade_date."""
    df = query_dataframe("""
        SELECT pe_ratio FROM index_data
        WHERE index_name = 'Nifty 50' AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, [trade_date])
    if df.empty or df["pe_ratio"].isna().all():
        return None
    return float(df["pe_ratio"].iloc[0])


def _load_sector_indices(trade_date: date) -> pd.DataFrame:
    _ph = ", ".join("?" * len(_SECTOR_BREADTH_INDICES))
    return query_dataframe(f"""
        SELECT index_name, close_val, pct_chg FROM index_data
        WHERE trade_date = ?
          AND index_name IN ({_ph})
    """, [trade_date, *_SECTOR_BREADTH_INDICES])


def _compute_constituent_stats(symbols: tuple, trade_date: date) -> _ConstituentStats:
    """
    Compute Signals 17c/d/e constituent data for any index symbol universe.

    17c — turnover-weighted breadth + delivery quality (daily_data)
    17d — stock futures OI-Price matrix (fno_bhavcopy FUTSTK)
    17e — stock options OI-Premium matrix (fno_bhavcopy OPTSTK)
    """
    cs = _ConstituentStats(total_stocks=len(symbols))
    if not symbols:
        return cs

    _ph = ", ".join("?" * len(symbols))

    # ── 17c: cash breadth + delivery quality ─────────────────────────────────
    _stocks = query_dataframe(f"""
        SELECT
            symbol,
            CASE WHEN prev_close > 0
                 THEN (close_price - prev_close) / prev_close * 100
                 ELSE NULL END AS pct_chg,
            deliv_per,
            turnover_lacs
        FROM daily_data
        WHERE symbol IN ({_ph})
          AND trade_date = ?
          AND series IN ('EQ', 'SM', 'ST')
    """, [*symbols, trade_date])
    if not _stocks.empty:
        _stocks = _stocks.dropna(subset=["pct_chg", "turnover_lacs"]).copy()
        _stocks = _stocks[_stocks["turnover_lacs"] > 0]
        cs.coverage  = len(_stocks)
        _adv = _stocks[_stocks["pct_chg"] > 0]
        _dec = _stocks[_stocks["pct_chg"] < 0]
        cs.advancing = len(_adv)
        cs.declining = len(_dec)
        _total_turn = float(_stocks["turnover_lacs"].sum())
        if _total_turn > 0:
            cs.adv_turn_pct = float(_adv["turnover_lacs"].sum()) / _total_turn * 100
        if len(_adv) >= 3:
            cs.adv_deliv_avg = float(_adv["deliv_per"].dropna().mean()) if "deliv_per" in _adv else None
        if len(_dec) >= 3:
            cs.dec_deliv_avg = float(_dec["deliv_per"].dropna().mean()) if "deliv_per" in _dec else None

    # ── 17d: stock futures OI-Price matrix ───────────────────────────────────
    _fut_df = query_dataframe(f"""
        WITH last2 AS (
            SELECT DISTINCT trade_date
            FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK'
              AND symbol IN ({_ph})
              AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 2
        )
        SELECT f.symbol, f.trade_date, f.expiry_date,
               f.open_interest, f.settle_price
        FROM fno_bhavcopy f
        WHERE f.instrument = 'FUTSTK'
          AND f.symbol IN ({_ph})
          AND f.trade_date IN (SELECT trade_date FROM last2)
          AND f.open_interest > 0
        ORDER BY f.symbol, f.trade_date DESC, f.expiry_date
    """, [*symbols, trade_date, *symbols])

    if not _fut_df.empty:
        _fut_df["trade_date"]  = pd.to_datetime(_fut_df["trade_date"]).dt.date
        _fut_df["expiry_date"] = pd.to_datetime(_fut_df["expiry_date"]).dt.date
        _fdates = sorted(_fut_df["trade_date"].unique(), reverse=True)
        _ftd = _fdates[0] if _fdates else None
        _fpd = _fdates[1] if len(_fdates) >= 2 else None

        # Settlement-day detection is DATA-DRIVEN (any contract expiring today),
        # not calendar-based: a "last Tuesday" rule misses expiries shifted a day
        # earlier by a trading holiday. On settlement day every constituent's OI
        # collapses, creating a false basket read regardless of direction.
        _is_settlement_day = bool(
            (_fut_df.loc[_fut_df["trade_date"] == _ftd, "expiry_date"] == _ftd).any()
        ) if _ftd else False
        if _ftd and _fpd and not _is_settlement_day:
            _today_f = (
                _fut_df[_fut_df["trade_date"] == _ftd]
                .groupby("symbol", as_index=False).first()
            )
            _prev_f  = (
                _fut_df[_fut_df["trade_date"] == _fpd]
                .groupby("symbol", as_index=False).first()
            )
            _prev_idx = _prev_f.set_index("symbol")
            _fl = _sc = _lu = _fs = _valid = 0
            for _, tr in _today_f.iterrows():
                sym = tr["symbol"]
                if sym not in _prev_idx.index:
                    continue
                pr = _prev_idx.loc[sym]
                if tr["expiry_date"] != pr["expiry_date"]:
                    continue
                oi_t = float(tr["open_interest"]); oi_p = float(pr["open_interest"])
                sp_t = float(tr["settle_price"]);  sp_p = float(pr["settle_price"])
                if oi_p <= 0 or sp_p <= 0:
                    continue
                oi_chg = (oi_t - oi_p) / oi_p * 100
                pr_chg = (sp_t - sp_p) / sp_p * 100
                oi_up = oi_chg >  0.5;  oi_dn = oi_chg < -0.5
                pr_up = pr_chg >  0.1;  pr_dn = pr_chg < -0.1
                _valid += 1
                if   pr_up and oi_up: _fl += 1
                elif pr_up and oi_dn: _sc += 1
                elif pr_dn and oi_dn: _lu += 1
                elif pr_dn and oi_up: _fs += 1

            cs.fut_valid       = _valid
            cs.fut_fresh_long  = _fl
            cs.fut_fresh_short = _fs
            cs.fut_long_unwind = _lu
            cs.fut_short_cov   = _sc

    # ── 17e: stock options OI-Premium matrix ─────────────────────────────────
    _opt_df = query_dataframe(f"""
        WITH last2 AS (
            SELECT DISTINCT trade_date
            FROM fno_bhavcopy
            WHERE instrument = 'OPTSTK'
              AND symbol IN ({_ph})
              AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 2
        ),
        near_exp_date AS (
            SELECT symbol, trade_date, MIN(expiry_date) AS near_expiry
            FROM fno_bhavcopy
            WHERE instrument = 'OPTSTK'
              AND symbol IN ({_ph})
              AND trade_date IN (SELECT trade_date FROM last2)
              AND open_interest > 0
            GROUP BY symbol, trade_date
        )
        SELECT
            f.symbol,
            f.trade_date,
            ne.near_expiry,
            f.option_type,
            SUM(f.open_interest) AS total_oi,
            SUM(CASE WHEN f.contracts > 0 AND f.settle_price > 0
                     THEN (f.close_price - f.settle_price) * f.contracts
                     ELSE 0 END
            ) / NULLIF(SUM(CASE WHEN f.contracts > 0 AND f.settle_price > 0
                                THEN f.contracts ELSE 0 END), 0) AS wt_prem_chg,
            SUM(f.contracts) AS total_contracts
        FROM fno_bhavcopy f
        INNER JOIN near_exp_date ne
            ON  f.symbol     = ne.symbol
            AND f.trade_date = ne.trade_date
            AND f.expiry_date= ne.near_expiry
        WHERE f.instrument = 'OPTSTK'
          AND f.open_interest > 0
        GROUP BY f.symbol, f.trade_date, ne.near_expiry, f.option_type
        ORDER BY f.symbol, f.trade_date DESC, f.option_type
    """, [*symbols, trade_date, *symbols])

    if not _opt_df.empty:
        _opt_df["trade_date"]  = pd.to_datetime(_opt_df["trade_date"]).dt.date
        _opt_df["near_expiry"] = pd.to_datetime(_opt_df["near_expiry"]).dt.date
        _odates = sorted(_opt_df["trade_date"].unique(), reverse=True)
        _otd = _odates[0] if _odates else None
        _opd = _odates[1] if len(_odates) >= 2 else None

        if _otd and _opd:
            _MIN_CONTR = 100

            def _make_opt_idx(df, td):
                return (
                    df[df["trade_date"] == td]
                    .drop_duplicates(subset=["symbol", "option_type"], keep="last")
                    .set_index(["symbol", "option_type"])
                )
            _opt_today = _make_opt_idx(_opt_df, _otd)
            _opt_prev  = _make_opt_idx(_opt_df, _opd)

            _on = _os = _ostr = _or = _ov = 0
            for _sym in {s for s, _ in _opt_today.index}:
                try:
                    t_ce = _opt_today.loc[(_sym, "CE")]
                    t_pe = _opt_today.loc[(_sym, "PE")]
                    p_ce = _opt_prev.loc[(_sym, "CE")]
                    p_pe = _opt_prev.loc[(_sym, "PE")]
                except KeyError:
                    continue

                if t_ce["near_expiry"] != p_ce["near_expiry"]:
                    continue

                _c_oi_t = float(t_ce["total_oi"]); _c_oi_p = float(p_ce["total_oi"])
                _p_oi_t = float(t_pe["total_oi"]); _p_oi_p = float(p_pe["total_oi"])
                if _c_oi_p <= 0 or _p_oi_p <= 0:
                    continue

                c_oi_chg = (_c_oi_t - _c_oi_p) / _c_oi_p * 100
                p_oi_chg = (_p_oi_t - _p_oi_p) / _p_oi_p * 100
                c_prem   = t_ce.get("wt_prem_chg")
                p_prem   = t_pe.get("wt_prem_chg")
                c_contr  = float(t_ce.get("total_contracts", 0) or 0)
                p_contr  = float(t_pe.get("total_contracts", 0) or 0)

                c_oi_up = c_oi_chg > 0.5
                p_oi_up = p_oi_chg > 0.5
                has_call_vol = c_contr >= _MIN_CONTR
                has_put_vol  = p_contr >= _MIN_CONTR
                c_prem_up = (has_call_vol and c_prem is not None and
                             not pd.isna(c_prem) and float(c_prem) > 0)
                p_prem_up = (has_put_vol  and p_prem is not None and
                             not pd.isna(p_prem) and float(p_prem) > 0)
                c_buy = c_oi_up and c_prem_up
                c_wrt = c_oi_up and has_call_vol and not c_prem_up
                p_buy = p_oi_up and p_prem_up
                p_wrt = p_oi_up and has_put_vol  and not p_prem_up

                if not (c_oi_up or p_oi_up):
                    continue
                _ov += 1
                if   c_buy and p_wrt:  _on   += 1
                elif c_wrt and p_buy:  _os   += 1
                elif c_buy and p_buy:  _ostr += 1
                elif c_wrt and p_wrt:  _or   += 1
                elif c_buy or p_wrt:   _on   += 1
                elif p_buy or c_wrt:   _os   += 1

            cs.opt_valid     = _ov
            cs.opt_net_long  = _on
            cs.opt_net_short = _os
            cs.opt_straddle  = _ostr
            cs.opt_range     = _or

    return cs


# ── Market context builder ────────────────────────────────────────────────────

def _build_market_context(trade_date: date) -> MarketContext:
    ctx = MarketContext(trade_date=trade_date)

    # ── India VIX ─────────────────────────────────────────────────────────────
    vix_hist = _load_vix_history(trade_date)
    if not vix_hist.empty:
        today_row = vix_hist[vix_hist["trade_date"] == trade_date]
        if today_row.empty: today_row = vix_hist.tail(1)
        if not today_row.empty:
            r = today_row.iloc[0]
            ctx.vix_close   = float(r["close_val"]) if pd.notna(r.get("close_val")) else None
            ctx.vix_pct_chg = float(r["pct_chg"])   if pd.notna(r.get("pct_chg"))   else None
        if len(vix_hist) >= 6 and ctx.vix_close:
            old = float(vix_hist.iloc[-6]["close_val"])
            if old > 0: ctx.vix_5d_chg_pct = (ctx.vix_close / old - 1) * 100
        v = ctx.vix_close
        if v is not None:
            ctx.vix_regime = (
                "Complacency (<12)" if v < 12 else
                f"Normal ({v:.1f})"  if v < 18 else
                f"Elevated ({v:.1f})" if v < 22 else
                f"Fear ({v:.1f})"
            )

    # ── FAO participant OI — current + previous date ───────────────────────────
    fao_df = _load_fao_participant_two_dates(trade_date)
    if not fao_df.empty:
        dates_avail = sorted(fao_df["trade_date"].unique(), reverse=True)
        ctx.fao_date = dates_avail[0] if dates_avail else None
        if len(dates_avail) >= 2:
            ctx.fii_prev_fao_date = dates_avail[1]

        def _parse_fao_row(row):
            ct = str(row.get("client_type", "")).strip().upper()
            fl = int(row.get("fut_idx_long",  0) or 0)
            fs = int(row.get("fut_idx_short", 0) or 0)
            cl = int(row.get("opt_idx_call_long",  0) or 0)
            cs = int(row.get("opt_idx_call_short", 0) or 0)
            pl = int(row.get("opt_idx_put_long",   0) or 0)
            ps = int(row.get("opt_idx_put_short",  0) or 0)
            return ct, fl - fs, cl - cs, pl - ps

        # Latest date
        for _, row in fao_df[fao_df["trade_date"] == ctx.fao_date].iterrows():
            ct, fut_net, call_net, put_net = _parse_fao_row(row)
            if ct == "FII":
                ctx.fii_fut_idx_net  = fut_net
                ctx.fii_opt_call_net = call_net
                ctx.fii_opt_put_net  = put_net
                ctx.fii_opt_delta    = call_net - put_net
            elif ct == "DII":
                ctx.dii_fut_idx_net = fut_net
                ctx.dii_opt_delta   = call_net - put_net
            elif ct == "CLIENT":
                ctx.client_fut_idx_net = fut_net
                ctx.client_opt_delta   = call_net - put_net
            elif ct in ("PRO", "PROPRIETORY", "PROP"):
                ctx.pro_fut_idx_net = fut_net
                ctx.pro_opt_delta   = call_net - put_net

        # Previous date (for day-over-day change)
        if ctx.fii_prev_fao_date:
            for _, row in fao_df[fao_df["trade_date"] == ctx.fii_prev_fao_date].iterrows():
                ct, fut_net, _, _ = _parse_fao_row(row)
                if ct == "FII":
                    ctx.fii_fut_idx_net_prev = fut_net
                    ctx.fii_net_change_1d    = ctx.fii_fut_idx_net - fut_net
                    break

    # ── FII derivatives stats — current + 5D history ─────────────────────────
    fii_hist = _load_fii_stats_history(trade_date, lookback=8)
    if not fii_hist.empty:
        fii_dates = sorted(fii_hist["trade_date"].unique(), reverse=True)
        ctx.fii_stats_date = fii_dates[0] if fii_dates else None

        # Today's flows (most recent date)
        today_stats = fii_hist[fii_hist["trade_date"] == ctx.fii_stats_date] if ctx.fii_stats_date else pd.DataFrame()
        for _, row in today_stats.iterrows():
            cat = str(row.get("category", "")).strip().upper()
            bv  = float(row["buy_value_cr"])  if pd.notna(row.get("buy_value_cr"))  else 0.0
            sv  = float(row["sell_value_cr"]) if pd.notna(row.get("sell_value_cr")) else 0.0
            net = bv - sv
            oi_cr = float(row["oi_value_cr"]) if pd.notna(row.get("oi_value_cr")) else 0.0

            for sym, cat_name in _SYMBOL_TO_FII_FUT.items():
                if cat == cat_name:
                    ctx.fii_symbol_flows[sym] = net
                    ctx.fii_oi_cr_latest[sym]  = oi_cr
            for sym, cat_name in _SYMBOL_TO_FII_OPT.items():
                if cat == cat_name:
                    ctx.fii_opt_net_flows[sym] = net
            if cat == "STOCK FUTURES":
                ctx.fii_stock_fut_net_cr = net

        # 5-day cumulative futures flow (up to 5 most recent dates available)
        recent5_dates = fii_dates[:5]
        hist5 = fii_hist[fii_hist["trade_date"].isin(recent5_dates)]
        for sym, cat_name in _SYMBOL_TO_FII_FUT.items():
            rows = hist5[hist5["category"].str.upper() == cat_name]
            if not rows.empty:
                nets = (rows["buy_value_cr"].fillna(0) - rows["sell_value_cr"].fillna(0))
                ctx.fii_cumul_flows_5d[sym] = float(nets.sum())

        # OI ₹Cr 5D ago (oldest available in our history)
        if len(fii_dates) >= 5:
            oldest = fii_dates[4]   # 5th most recent date
            old_stats = fii_hist[fii_hist["trade_date"] == oldest]
            for _, row in old_stats.iterrows():
                cat   = str(row.get("category", "")).strip().upper()
                oi_cr = float(row["oi_value_cr"]) if pd.notna(row.get("oi_value_cr")) else 0.0
                for sym, cat_name in _SYMBOL_TO_FII_FUT.items():
                    if cat == cat_name:
                        ctx.fii_oi_cr_5d_ago[sym] = oi_cr

    # ── Nifty PE ──────────────────────────────────────────────────────────────
    ctx.nifty_pe = _load_nifty_pe(trade_date)

    # ── Nifty 50 daily return (for non-Nifty relative-strength signal) ───────
    _n50idx = query_dataframe("""
        SELECT pct_chg FROM index_data
        WHERE index_name = 'Nifty 50' AND trade_date = ?
    """, [trade_date])
    if not _n50idx.empty and pd.notna(_n50idx["pct_chg"].iloc[0]):
        ctx.nifty_pct_chg = float(_n50idx["pct_chg"].iloc[0])

    # ── Constituent-level data (Signals 17c/d/e) ──────────────────────────────
    for _csym, _csyms in _CONSTITUENT_SYMBOLS.items():
        ctx.constituent_stats[_csym] = _compute_constituent_stats(_csyms, trade_date)

    # ── Sector breadth + rotation ─────────────────────────────────────────────
    sec_df = _load_sector_indices(trade_date)
    if not sec_df.empty and "pct_chg" in sec_df.columns:
        valid = sec_df["pct_chg"].dropna()
        ctx.total_sectors     = len(valid)
        ctx.advancing_sectors = int((valid > 0).sum())
        ctx.declining_sectors = int((valid < 0).sum())
        if ctx.total_sectors > 0:
            ctx.breadth_pct_advancing = ctx.advancing_sectors / ctx.total_sectors * 100

        cyc = sec_df[sec_df["index_name"].isin(_CYCLICAL_SECTORS)]["pct_chg"].dropna()
        dfe = sec_df[sec_df["index_name"].isin(_DEFENSIVE_SECTORS)]["pct_chg"].dropna()
        ctx.cyclical_avg_chg  = float(cyc.mean()) if len(cyc) >= 2 else None
        ctx.defensive_avg_chg = float(dfe.mean()) if len(dfe) >= 2 else None
        if ctx.cyclical_avg_chg is not None and ctx.defensive_avg_chg is not None:
            diff = ctx.cyclical_avg_chg - ctx.defensive_avg_chg
            ctx.risk_mode = "RISK-ON" if diff > 0.50 else ("RISK-OFF" if diff < -0.50 else "MIXED")

    return ctx


# ── Key-level computation ─────────────────────────────────────────────────────

def _compute_key_levels(opt_near: pd.DataFrame, spot_close: float) -> IndexKeyLevels:
    if opt_near.empty: return IndexKeyLevels()
    calls = opt_near[opt_near["option_type"] == "CE"].set_index("strike_price")["open_interest"].astype(float)
    puts  = opt_near[opt_near["option_type"] == "PE"].set_index("strike_price")["open_interest"].astype(float)
    all_s = sorted(set(calls.index.tolist()) | set(puts.index.tolist()))
    if not all_s: return IndexKeyLevels()

    min_loss, max_pain = float("inf"), None
    for P in all_s:
        P = float(P)
        pain = (sum(max(0.0, P - float(K)) * float(v) for K, v in calls.items())
              + sum(max(0.0, float(K) - P) * float(v) for K, v in puts.items()))
        if pain < min_loss: min_loss, max_pain = pain, P

    band = spot_close * 0.05
    nc = calls[(calls.index >= spot_close - band)     & (calls.index <= spot_close + band * 3)]
    np_ = puts[(puts.index  >= spot_close - band * 3) & (puts.index  <= spot_close + band)]
    if nc.empty:  nc  = calls
    if np_.empty: np_ = puts
    tc = nc.nlargest(2); tp = np_.nlargest(2)

    return IndexKeyLevels(
        max_pain=round(max_pain, 0) if max_pain else None,
        top_call_strike   =float(tc.index[0]) if len(tc) >= 1 else None,
        top_put_strike    =float(tp.index[0]) if len(tp) >= 1 else None,
        second_call_strike=float(tc.index[1]) if len(tc) >= 2 else None,
        second_put_strike =float(tp.index[1]) if len(tp) >= 2 else None,
        call_oi_at_top=int(tc.iloc[0]) if len(tc) >= 1 else 0,
        put_oi_at_top =int(tp.iloc[0]) if len(tp) >= 1 else 0,
    )


# ── Snapshot builder ──────────────────────────────────────────────────────────

def _build_snapshot(
    snap_date: date, idx_hist: pd.DataFrame, fno_df: pd.DataFrame,
    near_expiry: Optional[date], spot_close: Optional[float],
    futures_price: Optional[float], carry_pts: Optional[float],
    carry_pct_ann: Optional[float],
) -> DaySnapshot:
    pct_chg = high = low = None
    if not idx_hist.empty:
        row = idx_hist[idx_hist["trade_date"] == snap_date]
        if not row.empty:
            r = row.iloc[0]
            pct_chg = float(r["pct_chg"])  if pd.notna(r.get("pct_chg"))  else None
            high    = float(r["high_val"]) if pd.notna(r.get("high_val")) else None
            low     = float(r["low_val"])  if pd.notna(r.get("low_val"))  else None

    fut_oi = call_oi = put_oi = volume = 0; pcr = None
    if not fno_df.empty:
        fr = fno_df[fno_df["instrument"] == "FUTIDX"]
        if not fr.empty and near_expiry:
            nr = fr[fr["expiry_date"] == near_expiry]
            fut_oi = int(nr.iloc[0]["open_interest"]) if not nr.empty else 0
        if near_expiry:
            opt = fno_df[(fno_df["instrument"] == "OPTIDX") & (fno_df["expiry_date"] == near_expiry)]
            call_oi = int(opt[opt["option_type"] == "CE"]["open_interest"].sum())
            put_oi  = int(opt[opt["option_type"] == "PE"]["open_interest"].sum())
            pcr     = round(put_oi / call_oi, 2) if call_oi > 0 else None
        volume = int(fno_df[fno_df["instrument"].isin(["FUTIDX", "OPTIDX"])]["contracts"].sum())

    return DaySnapshot(
        trade_date=snap_date, spot_close=spot_close, pct_chg=pct_chg,
        high=high, low=low, fut_settle=futures_price, fut_oi=fut_oi,
        call_oi=call_oi, put_oi=put_oi, pcr=pcr, total_volume=volume,
        carry_pts=carry_pts, carry_pct_ann=carry_pct_ann,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTIONS — index-specific (signals 1-6)
# ═══════════════════════════════════════════════════════════════════════════════

def _sig_price_action(idx_hist: pd.DataFrame) -> Optional[IndexSignal]:
    """3D return z-score. z ≤ -1.8 = a statistically stretched decline → mean-reversion bias (directional only, not a validated win-rate)."""
    if idx_hist.empty or len(idx_hist) < 8: return None
    df     = idx_hist.sort_values("trade_date").reset_index(drop=True)
    closes = df["close_val"].dropna()
    if len(closes) < 5 or float(closes.iloc[-4]) <= 0: return None

    cum_3d   = (float(closes.iloc[-1]) / float(closes.iloc[-4]) - 1) * 100
    baseline = (closes.pct_change(3) * 100).iloc[-21:-1].dropna()
    if len(baseline) < 5: return None
    z = _zscore(cum_3d, baseline)

    pcts     = df["pct_chg"].tail(3).values if "pct_chg" in df.columns else []
    all_down = len(pcts) == 3 and all(v < -0.15 for v in pcts)

    if z <= -1.8 or (z <= -1.2 and all_down):
        return IndexSignal(
            name="Oversold Bounce Setup", category="Price Action",
            direction=1, score=2.0 if z <= -2.2 else 1.0,
            headline=f"Oversold {cum_3d:.1f}% / 3D (z={z:.1f})",
            description=(
                f"3-day decline {abs(cum_3d):.1f}% statistically extreme (z={z:.1f}). "
                "Multi-day declines at this intensity tend to mean-revert (directional bias only — "
                "not a validated win-rate; see the engine's live hit-rate at the top of the page). "
                "Selling looks stretched."
            ), emoji="🔄",
        )
    if z >= 3.0:
        return IndexSignal(
            name="Extreme Overbought (Caution)", category="Price Action",
            direction=-1, score=-1.0,
            headline=f"Extreme Run +{cum_3d:.1f}% / 3D (z=+{z:.1f})",
            description=(
                f"3-day gain {cum_3d:.1f}% statistically extreme. "
                "Strong runs tend to continue more often than they reverse — mild caution only."
            ), emoji="⚠️",
        )
    return None


def _sig_vwap(idx_hist: pd.DataFrame) -> Optional[IndexSignal]:
    """
    Anchored VWAP — the institutional cost-basis / control line.

    Institutions benchmark execution to VWAP, so a multi-day anchored VWAP acts as
    dynamic support/resistance: trading ABOVE it = buyers in control (price above the
    average institutional cost), BELOW = sellers in control. A fresh RECLAIM or LOSS
    of VWAP is a directional trigger; price stretched far from VWAP tends to revert
    (execution algos pull price back toward VWAP). Volume-weighted by the index's own
    traded value (turnover_cr), typical price = (H+L+C)/3. Point-in-time: ≤ today.
    """
    if idx_hist is None or len(idx_hist) < 22:
        return None
    df = idx_hist.copy()
    w = df["turnover_cr"].fillna(0.0)
    if float(w.sum()) <= 0:
        w = df["volume"].fillna(0.0)
    if float(w.sum()) <= 0:
        return None
    typ = (df["high_val"] + df["low_val"] + df["close_val"]) / 3.0
    N = 20  # rolling monthly institutional cost basis

    def _vwap(tp: pd.Series, wt: pd.Series) -> Optional[float]:
        wt = wt.to_numpy(); tp = tp.to_numpy()
        return float((tp * wt).sum() / wt.sum()) if wt.sum() > 0 else None

    vwap     = _vwap(typ.iloc[-N:],      w.iloc[-N:])
    vwap_prev = _vwap(typ.iloc[-(N + 1):-1], w.iloc[-(N + 1):-1])
    if vwap is None or vwap_prev is None:
        return None
    spot = float(df["close_val"].iloc[-1])
    prev = float(df["close_val"].iloc[-2])
    dist = (spot - vwap) / vwap * 100.0
    stretched = abs(dist) > 3.0   # ~3% from 20D VWAP is extended for a broad index

    reclaim = prev <= vwap_prev and spot > vwap
    loss    = prev >= vwap_prev and spot < vwap

    if reclaim:
        return IndexSignal(
            name="VWAP Reclaim", category="Price Action", direction=1,
            score=1.5, emoji="🟢",
            headline=f"Reclaimed 20D VWAP ({vwap:,.0f}) — buyers seize control",
            description=(
                f"Spot {spot:,.0f} crossed back ABOVE the 20-day anchored VWAP {vwap:,.0f} "
                f"(+{dist:.2f}%). Institutions are now in profit on the average position — "
                "a fresh bullish control shift; VWAP becomes support."),
        )
    if loss:
        return IndexSignal(
            name="VWAP Loss", category="Price Action", direction=-1,
            score=-1.5, emoji="🔴",
            headline=f"Lost 20D VWAP ({vwap:,.0f}) — sellers seize control",
            description=(
                f"Spot {spot:,.0f} broke BELOW the 20-day anchored VWAP {vwap:,.0f} "
                f"({dist:.2f}%). Average institutional position now underwater — bearish "
                "control shift; VWAP becomes resistance."),
        )
    if spot > vwap:
        sc = 0.4 if stretched else 1.0
        note = (" — but extended +%.1f%% above VWAP (mean-reversion risk)" % dist) if stretched else ""
        return IndexSignal(
            name="Above VWAP", category="Price Action", direction=1,
            score=sc, emoji="📈",
            headline=f"Holding above 20D VWAP (+{dist:.2f}%){note}",
            description=(
                f"Spot {spot:,.0f} is above the 20-day anchored VWAP {vwap:,.0f} — buyers "
                "in control, trend bias up." + (" Stretched: algos may pull price back toward "
                "VWAP, fade chasing." if stretched else " VWAP is the support to defend.")),
        )
    sc = -0.4 if stretched else -1.0
    note = (" — but %.1f%% below VWAP (oversold, bounce risk)" % dist) if stretched else ""
    return IndexSignal(
        name="Below VWAP", category="Price Action", direction=-1,
        score=sc, emoji="📉",
        headline=f"Trading below 20D VWAP ({dist:.2f}%){note}",
        description=(
            f"Spot {spot:,.0f} is below the 20-day anchored VWAP {vwap:,.0f} — sellers in "
            "control, trend bias down." + (" Stretched below: prone to a snap-back toward VWAP."
            if stretched else " VWAP is the resistance capping rallies.")),
    )


def _sig_rsi(idx_hist: pd.DataFrame) -> Optional[IndexSignal]:
    """
    Institution-grade RSI(14) — Cardwell/Brown range rules + divergence, NOT the
    retail 70/30. Pros read RSI relative to the TREND: in a bull market RSI oscillates
    ~40–80 (40–50 is support), in a bear market ~20–60 (50–60 is resistance). The
    highest-value reads are DIVERGENCES (price/momentum disagree) and range
    breaks/holds. Wilder smoothing on the index close. Point-in-time: ≤ today.
    """
    if idx_hist is None or len(idx_hist) < 30:
        return None
    close = idx_hist["close_val"].astype(float).reset_index(drop=True)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1 / 14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + ag / al.replace(0, np.nan))
    rsi = rsi.fillna(50.0)
    cur = float(rsi.iloc[-1]); prev = float(rsi.iloc[-2])

    # Trend regime via EMA structure
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(close) >= 50 else ema20
    last = float(close.iloc[-1])
    uptrend   = last > ema50 and ema20 >= ema50
    downtrend = last < ema50 and ema20 <= ema50

    # Divergence over a ~20-bar window (exclude the last 2 bars for the prior pivot)
    look = min(20, len(close) - 1)
    p = close.tail(look).reset_index(drop=True)
    r = rsi.tail(look).reset_index(drop=True)
    n = len(p)
    bear_div = bull_div = False
    if n >= 8:
        prior = p.iloc[:n - 2]
        hi_i = int(prior.idxmax()); lo_i = int(prior.idxmin())
        bear_div = (p.iloc[-1] >= prior.iloc[hi_i] * 0.999) and (r.iloc[-1] < r.iloc[hi_i] - 2.0)
        bull_div = (p.iloc[-1] <= prior.iloc[lo_i] * 1.001) and (r.iloc[-1] > r.iloc[lo_i] + 2.0)

    _band = ("bull-market 40–80 band" if uptrend else
             "bear-market 20–60 band" if downtrend else "neutral 30–70 band")

    # Priority: divergence > true extreme > range continuation > centerline
    if bear_div:
        return IndexSignal(
            name="RSI Bearish Divergence", category="Price Action", direction=-1,
            score=-2.0, emoji="🔴",
            headline=f"Bearish RSI divergence (RSI {cur:.0f}) — momentum fading at the high",
            description=("Price made a fresh high but RSI did NOT — institutional momentum is "
                         "waning into strength. A classic distribution / top-warning read."))
    if bull_div:
        return IndexSignal(
            name="RSI Bullish Divergence", category="Price Action", direction=1,
            score=2.0, emoji="🟢",
            headline=f"Bullish RSI divergence (RSI {cur:.0f}) — momentum building at the low",
            description=("Price made a fresh low but RSI did NOT — selling pressure is exhausting. "
                         "A classic accumulation / bottoming read."))
    if cur >= 80:
        return IndexSignal(
            name="RSI Extreme Overbought", category="Price Action", direction=-1,
            score=-1.0, emoji="⚠️",
            headline=f"RSI {cur:.0f} — true overbought (>80)",
            description=("Even by institutional bull-market rules (overbought only >80), momentum "
                         "is stretched — elevated reversion risk into the next session."))
    if cur <= 20:
        return IndexSignal(
            name="RSI Extreme Oversold", category="Price Action", direction=1,
            score=1.0, emoji="🟢",
            headline=f"RSI {cur:.0f} — true oversold (<20)",
            description=("Deep oversold even by bear-market rules (<20) — bounce setup as selling "
                         "exhausts."))
    if uptrend and 40 <= cur <= 55 and cur > prev:
        return IndexSignal(
            name="RSI Bull-Support Hold", category="Price Action", direction=1,
            score=1.0, emoji="📈",
            headline=f"RSI {cur:.0f} held bull-market support (40–50) & turning up",
            description=("In the uptrend, RSI pulled back to the 40–50 institutional support zone "
                         "and is turning up — continuation read, not weakness."))
    if downtrend and 45 <= cur <= 60 and cur < prev:
        return IndexSignal(
            name="RSI Bear-Resistance Reject", category="Price Action", direction=-1,
            score=-1.0, emoji="📉",
            headline=f"RSI {cur:.0f} rejected at bear-market resistance (50–60)",
            description=("In the downtrend, RSI rallied into the 50–60 institutional resistance "
                         "zone and is rolling over — continuation lower."))
    if cur > 55:
        return IndexSignal(
            name="RSI Momentum Up", category="Price Action", direction=1,
            score=0.5, emoji="📈",
            headline=f"RSI {cur:.0f} above centerline ({_band})",
            description="RSI above 50 with positive momentum — mild bullish tilt.")
    if cur < 45:
        return IndexSignal(
            name="RSI Momentum Down", category="Price Action", direction=-1,
            score=-0.5, emoji="📉",
            headline=f"RSI {cur:.0f} below centerline ({_band})",
            description="RSI below 50 with negative momentum — mild bearish tilt.")
    return None


# ── Price-structure Support / Resistance (Signal 6b — S/R Confluence) ─────────
# Swing-pivot S/R zones from the daily index OHLC, then a confluence read: did the
# last few sessions TEST a level and hold/reject it, and do the option writers
# (put/call OI walls), volume expansion and FII flow agree with that defense?
# Edwards & Magee level logic + the OI-wall confirmation this engine already trusts.

_SR_PIVOT_WING  = 2      # fractal pivot = extreme of ±2 bars (5-bar fractal)
_SR_CLUSTER_PCT = 0.45   # pivots within this % of each other merge into one zone
_SR_BAND_PCT    = 0.30   # zone half-width used for test / hold / break detection
_SR_NEAR_PCT    = 3.5    # only read zones within this % of spot
_SR_MIN_TOUCH   = 2      # a level needs ≥2 pivot touches to count as a line
_SR_TEST_WIN    = 4      # sessions scanned for hold / reject / break events
_SR_WALL_PCT    = 0.60   # OI wall within this % of a price level = confluence
_SR_VOL_CONFIRM = 1.15   # volume ≥1.15× 20D avg on the event day (validated threshold)


def _detect_sr_zones(idx_hist: pd.DataFrame) -> list[dict]:
    """Swing-pivot S/R zones from daily OHLC.

    Confirmed 5-bar fractal pivots (the last _SR_PIVOT_WING bars can never confirm
    a pivot, so the detection is point-in-time safe), clustered into zones when
    pivots sit within _SR_CLUSTER_PCT of each other. Returns zones with ≥2 touches:
    [{level, touches, last_bar}] sorted by level ascending.
    """
    if idx_hist is None or len(idx_hist) < 25:
        return []
    df = idx_hist.sort_values("trade_date").reset_index(drop=True)
    hi = df["high_val"].astype(float)
    lo = df["low_val"].astype(float)
    n, w = len(df), _SR_PIVOT_WING

    pivots: list[tuple[float, int]] = []   # (price, bar index)
    for i in range(w, n - w):
        h, l = hi.iloc[i], lo.iloc[i]
        if pd.notna(h) and h >= hi.iloc[i - w:i + w + 1].max():
            pivots.append((float(h), i))
        if pd.notna(l) and l <= lo.iloc[i - w:i + w + 1].min():
            pivots.append((float(l), i))
    if not pivots:
        return []

    pivots.sort(key=lambda p: p[0])
    clusters: list[dict] = []
    for price, bar in pivots:
        if clusters and abs(price - clusters[-1]["level"]) / clusters[-1]["level"] * 100 <= _SR_CLUSTER_PCT:
            c = clusters[-1]
            c["prices"].append(price); c["bars"].append(bar)
            c["level"] = float(np.mean(c["prices"]))
        else:
            clusters.append({"level": price, "prices": [price], "bars": [bar]})

    return [
        {"level": c["level"], "touches": len(c["prices"]), "last_bar": max(c["bars"])}
        for c in clusters if len(c["prices"]) >= _SR_MIN_TOUCH
    ]


def _sr_level_read(
    idx_hist: pd.DataFrame,
    spot_close: Optional[float],
    levels: Optional[IndexKeyLevels] = None,
    fii_5d: Optional[float] = None,
) -> dict:
    """Read the last _SR_TEST_WIN sessions against the nearby S/R zones.

    Pure-price core (works on OHLC alone — validated over the full index history)
    plus optional confluence inputs: OI walls (levels) and FII 5D flow (fii_5d).
    Returns a dict consumed by _sig_sr_confluence and the IndexPrediction sr_* fields.
    """
    out: dict = {
        "support": None, "resistance": None,
        "sup_touches": 0, "res_touches": 0,
        "bull": 0.0, "bear": 0.0, "score": 0.0, "direction": 0,
        "bull_notes": [], "bear_notes": [], "range_read": False,
    }
    if not spot_close or idx_hist is None or len(idx_hist) < 25:
        return out
    zones = _detect_sr_zones(idx_hist)
    if not zones:
        return out

    df = idx_hist.sort_values("trade_date").reset_index(drop=True)
    closes = df["close_val"].astype(float)
    highs  = df["high_val"].astype(float)
    lows   = df["low_val"].astype(float)
    vols   = df["turnover_cr"].fillna(0.0)
    if float(vols.sum()) <= 0:
        vols = df["volume"].fillna(0.0)
    n = len(df)
    win = min(_SR_TEST_WIN, n - 21)   # need 20 prior bars for the volume baseline
    if win < 1:
        return out

    def _vol_ratio(i: int) -> Optional[float]:
        base = float(vols.iloc[max(0, i - 20):i].mean())
        return (float(vols.iloc[i]) / base) if base > 0 else None

    # Most-recent event per zone over the last `win` sessions.
    # hold   = dipped into the zone intraday but closed back above (defense)
    # reject = poked into the zone intraday but closed back below (supply)
    # break_dn / break_up = close crossed through the whole band in one session
    near_band = spot_close * _SR_NEAR_PCT / 100
    events: list[dict] = []
    for z in zones:
        if abs(z["level"] - spot_close) > near_band:
            continue
        b_lo = z["level"] * (1 - _SR_BAND_PCT / 100)
        b_hi = z["level"] * (1 + _SR_BAND_PCT / 100)
        for i in range(n - win, n):
            c, c_prev = float(closes.iloc[i]), float(closes.iloc[i - 1])
            kind = None
            if c_prev > b_hi and c < b_lo:
                kind = "break_dn"
            elif c_prev < b_lo and c > b_hi:
                kind = "break_up"
            elif float(lows.iloc[i]) <= b_hi and c > b_hi:
                kind = "hold"
            elif float(highs.iloc[i]) >= b_lo and c < b_lo:
                kind = "reject"
            if kind:
                events.append({
                    "zone": z, "kind": kind, "bar": i,
                    "ago": n - 1 - i, "vol_ratio": _vol_ratio(i),
                    "date": df["trade_date"].iloc[i],
                })

    # Nearest defended support below spot / nearest supply resistance above spot
    sup = max((z for z in zones if z["level"] < spot_close
               and spot_close - z["level"] <= near_band),
              key=lambda z: z["level"], default=None)
    res = min((z for z in zones if z["level"] > spot_close
               and z["level"] - spot_close <= near_band),
              key=lambda z: z["level"], default=None)
    out["support"]     = round(sup["level"], 0) if sup else None
    out["resistance"]  = round(res["level"], 0) if res else None
    out["sup_touches"] = sup["touches"] if sup else 0
    out["res_touches"] = res["touches"] if res else 0

    def _wall_near(level: float, *walls) -> Optional[float]:
        cand = [w for w in walls if w]
        if not cand:
            return None
        best = min(cand, key=lambda w: abs(w - level))
        return best if abs(best - level) / level * 100 <= _SR_WALL_PCT else None

    bull = bear = 0.0

    # ── Support defended ─────────────────────────────────────────────────────
    if sup is not None:
        holds = [e for e in events if e["zone"] is sup and e["kind"] == "hold"]
        if holds:
            latest = min(holds, key=lambda e: e["ago"])
            bull += 1.0 + 0.4 * min(len(holds) - 1, 2)
            out["bull_notes"].append(
                f"support {sup['level']:,.0f} ({sup['touches']}-touch) tested "
                f"{len(holds)}× in last {win} sessions and HELD (latest {latest['date']})")
            if sup["touches"] >= 3:
                bull += 0.25
            vr = latest["vol_ratio"]
            if vr is not None and vr >= _SR_VOL_CONFIRM:
                bull += 0.5
                out["bull_notes"].append(f"defense on {vr:.2f}× volume (real buying)")
            pw = _wall_near(sup["level"],
                            levels.top_put_strike if levels else None,
                            levels.second_put_strike if levels else None)
            if pw is not None:
                bull += 0.75
                out["bull_notes"].append(
                    f"put wall {pw:,.0f} sits ON the price support — writers defend the same line")

    # ── Resistance rejecting ─────────────────────────────────────────────────
    if res is not None:
        rejects = [e for e in events if e["zone"] is res and e["kind"] == "reject"]
        if rejects:
            latest = min(rejects, key=lambda e: e["ago"])
            bear += 1.0 + 0.4 * min(len(rejects) - 1, 2)
            out["bear_notes"].append(
                f"resistance {res['level']:,.0f} ({res['touches']}-touch) tested "
                f"{len(rejects)}× in last {win} sessions and REJECTED (latest {latest['date']})")
            if res["touches"] >= 3:
                bear += 0.25
            vr = latest["vol_ratio"]
            if vr is not None and vr >= _SR_VOL_CONFIRM:
                bear += 0.5
                out["bear_notes"].append(f"rejection on {vr:.2f}× volume (real supply)")
            cw = _wall_near(res["level"],
                            levels.top_call_strike if levels else None,
                            levels.second_call_strike if levels else None)
            if cw is not None:
                bear += 0.75
                out["bear_notes"].append(
                    f"call wall {cw:,.0f} sits ON the price resistance — writers cap the same line")

    # ── Fresh breaks (last 2 sessions, price still beyond the zone) ──────────
    for e in events:
        if e["ago"] > 1:
            continue
        z = e["zone"]
        if e["kind"] == "break_dn" and spot_close < z["level"] * (1 - _SR_BAND_PCT / 100):
            bear += 1.25
            out["bear_notes"].append(
                f"BROKE {z['touches']}-touch support {z['level']:,.0f} ({e['date']}) — level flips to resistance")
            vr = e["vol_ratio"]
            if vr is not None and vr >= _SR_VOL_CONFIRM:
                bear += 0.5
                out["bear_notes"].append(f"breakdown on {vr:.2f}× volume")
        elif e["kind"] == "break_up" and spot_close > z["level"] * (1 + _SR_BAND_PCT / 100):
            bull += 1.25
            out["bull_notes"].append(
                f"BROKE ABOVE {z['touches']}-touch resistance {z['level']:,.0f} ({e['date']}) — level flips to support")
            vr = e["vol_ratio"]
            if vr is not None and vr >= _SR_VOL_CONFIRM:
                bull += 0.5
                out["bull_notes"].append(f"breakout on {vr:.2f}× volume")

    # ── FII flow agreement (5D cumulative ₹Cr in THIS index's futures) ───────
    if fii_5d is not None and abs(fii_5d) >= 500 and bull != bear:
        if fii_5d > 0 and bull > bear:
            bull += 0.25
            out["bull_notes"].append(f"FII 5D futures flow +₹{fii_5d:,.0f}Cr agrees")
        elif fii_5d < 0 and bear > bull:
            bear += 0.25
            out["bear_notes"].append(f"FII 5D futures flow −₹{abs(fii_5d):,.0f}Cr agrees")

    # Defended floor AND capped ceiling at once = range market, not a direction.
    if bull >= 1.0 and bear >= 1.0:
        out["range_read"] = True
        out["bull"], out["bear"] = bull, bear
        return out

    net = max(-2.5, min(2.5, bull - bear))
    out["bull"], out["bear"] = bull, bear
    out["score"] = net
    out["direction"] = 1 if net >= 0.75 else (-1 if net <= -0.75 else 0)
    return out


def _sig_sr_confluence(sr: dict) -> Optional[IndexSignal]:
    """Signal 6b — S/R Confluence: price level memory + writer/volume/FII confirmation."""
    if sr.get("range_read"):
        s_txt = f"{sr['support']:,.0f}" if sr.get("support") else "—"
        r_txt = f"{sr['resistance']:,.0f}" if sr.get("resistance") else "—"
        return IndexSignal(
            name="S/R Range — Floor & Ceiling Both Defended", category="Price Action",
            direction=0, score=0.0, emoji="↔️",
            headline=f"Range market: support {s_txt} held AND resistance {r_txt} rejected",
            description=("Recent sessions defended the floor and rejected the ceiling — "
                         "two-sided conviction. Bullish case: " + "; ".join(sr["bull_notes"]) +
                         ". Bearish case: " + "; ".join(sr["bear_notes"]) +
                         ". Expect rotation between the levels until one side breaks."),
        )
    if sr.get("direction", 0) == 0 or not (sr.get("bull_notes") or sr.get("bear_notes")):
        return None
    if sr["direction"] > 0:
        return IndexSignal(
            name="S/R Confluence — Support Defended", category="Price Action",
            direction=1, score=sr["score"], emoji="🛡️",
            headline=sr["bull_notes"][0],
            description=("Price-structure read (Edwards & Magee): " + "; ".join(sr["bull_notes"]) +
                         ". A defended multi-touch level with writer/volume agreement is the "
                         "market's revealed demand line — dips into it are being bought."),
        )
    return IndexSignal(
        name="S/R Confluence — Resistance / Breakdown", category="Price Action",
        direction=-1, score=sr["score"], emoji="🧱",
        headline=sr["bear_notes"][0],
        description=("Price-structure read (Edwards & Magee): " + "; ".join(sr["bear_notes"]) +
                     ". Supply is active at the level — rallies into it are being sold "
                     "(or the demand line just failed)."),
    )


def _context_only(sig: Optional[IndexSignal]) -> Optional[IndexSignal]:
    """Keep a signal's institutional read for display but zero its composite vote.

    Used for signals that are informative context yet have no validated standalone
    forward edge (VWAP, RSI) — they should not move the next-day directional score.
    """
    if sig is not None:
        sig.score = 0.0
        if "Context only" not in sig.description:
            sig.description += (
                " · 📊 Context only — institutional reference; walk-forward showed no "
                "standalone next-day edge, so it informs but does not vote.")
    return sig


def _sig_oi_price_matrix(
    pct_chg: Optional[float],
    fut_oi_chg: int,
    fut_oi_base: int = 0,
    oi_uncomparable: bool = False,
    roll_note: str = "",
) -> IndexSignal:
    """OI-Price matrix — Murphy. Most reliable institutional directional signal.

    Runs on TOTAL futures OI summed across all active expiries, not the near month
    alone. Near-month-only OI bleeds to the next contract during rollover week and
    fakes "Long Unwinding / Short Covering" (near OI can fall ~20%/day while total OI
    is flat); total OI is continuous through the roll, so the verdict reflects real
    net positioning. `roll_note` annotates how far the roll has progressed. The
    percentage OI threshold (>0.5%) keeps the read scale-invariant across data formats.

    oi_uncomparable=True: no prior-day total to diff against (e.g. first day of data)
    — fall back to pure price direction at half score, labeled so the user isn't misled.
    """
    pct = pct_chg or 0.0
    up  = pct > 0.10; dn = pct < -0.10

    # ── No comparable prior OI: can't read the matrix — price direction only ──
    if oi_uncomparable:
        if up:
            return IndexSignal(
                "Price Up — Futures OI N/A", "Futures OI", 1, 1.0,
                f"Price +{pct:.2f}%, OI change not comparable",
                f"No prior-day futures OI to compare against. Price direction only: "
                f"+{pct:.2f}% (mild bullish lean).{roll_note}", "🔄",
            )
        if dn:
            return IndexSignal(
                "Price Down — Futures OI N/A", "Futures OI", -1, -1.0,
                f"Price {pct:.2f}%, OI change not comparable",
                f"No prior-day futures OI to compare against. Price direction only: "
                f"{pct:.2f}% (mild bearish lean).{roll_note}", "🔄",
            )
        return IndexSignal(
            "Flat — Futures OI N/A", "Futures OI", 0, 0.0,
            "OI not comparable; price flat",
            f"No prior-day futures OI to compare against. No directional edge today.{roll_note}", "🔄",
        )

    oi_chg_pct = (fut_oi_chg / max(fut_oi_base, 1)) * 100 if fut_oi_base > 0 else 0.0
    oi_up = oi_chg_pct > 0.5
    oi_dn = oi_chg_pct < -0.5
    # Cap display value: extreme % from a tiny base is misleading; logic uses the raw value.
    oi_disp = max(-999.9, min(999.9, oi_chg_pct))
    if up and oi_up:
        sig = IndexSignal("Fresh Long Buildup", "Futures OI", 1, 3.0, "Fresh Long Build",
            f"Price +{pct:.2f}% AND total OI +{oi_disp:.1f}%. New money entering longs — "
            "strongest bullish confirmation. Institutions building directional long.", "🟢")
    elif up and oi_dn:
        sig = IndexSignal("Short Covering Rally", "Futures OI", 1, 1.0, "Short Covering",
            f"Price +{pct:.2f}% but total OI {oi_disp:+.1f}% (falling). "
            "Short covering rally — real but lacks fresh conviction.", "🟡")
    elif dn and oi_up:
        sig = IndexSignal("Fresh Short Buildup", "Futures OI", -1, -3.0, "Fresh Short Build",
            f"Price {pct:.2f}% AND total OI +{oi_disp:.1f}%. "
            "New shorts added into decline — strongest bearish confirmation.", "🔴")
    elif dn and oi_dn:
        sig = IndexSignal("Long Unwinding", "Futures OI", -1, -1.0, "Long Unwinding",
            f"Price {pct:.2f}% and total OI {oi_disp:+.1f}% (falling). "
            "Longs closing — bearish but self-limiting.", "🟡")
    else:
        sig = IndexSignal("Sideways / Indecisive", "Futures OI", 0, 0.0, "Indecisive OI",
            f"Price {pct:.2f}% with small total OI change ({oi_disp:+.1f}%). No directional conviction.", "⚪")
    if roll_note:
        sig.description += roll_note
    return sig


def _sig_carry(carry_pct_ann: Optional[float], carry_pts: Optional[float]) -> Optional[IndexSignal]:
    """Cost of carry vs India fair value (Hull). Backwardation = stress."""
    if carry_pct_ann is None or carry_pts is None: return None
    if carry_pct_ann >= 9.0:
        return IndexSignal("Futures Premium — Bullish Demand", "Carry", 1, 2.0,
            f"Carry +{carry_pct_ann:.1f}% ann (+{carry_pts:.0f} pts) — Bullish",
            f"Futures at +{carry_pts:.0f} pts above spot ({carry_pct_ann:.1f}% ann). "
            "Above fair value — market paying premium for long futures exposure.", "🚀")
    if 4.0 <= carry_pct_ann < 9.0:
        return IndexSignal("Futures Near Fair Value", "Carry", 0, 0.0,
            f"Carry +{carry_pct_ann:.1f}% ann — Fair Value",
            f"Futures at +{carry_pts:.0f} pts ({carry_pct_ann:.1f}% ann). Normal range (4-9%).", "⚪")
    if 0.0 <= carry_pct_ann < 4.0:
        return IndexSignal("Futures Below Fair Value", "Carry", -1, -1.0,
            f"Carry {carry_pct_ann:.1f}% ann — Below Fair Value",
            f"Futures at +{carry_pts:.0f} pts ({carry_pct_ann:.1f}% ann) — still in contango "
            "but below the 4-9% fair-value band. Soft demand for long futures exposure.", "🟡")
    return IndexSignal("Backwardation — Stress Signal", "Carry", -1, -2.0,
        f"Backwardation {carry_pts:.0f} pts — Bearish / Stress",
        f"Futures BELOW spot by {abs(carry_pts):.0f} pts ({carry_pct_ann:.1f}% ann). "
        "Backwardation = stress signal. Market pricing in decline.", "🔴")


def _sig_pcr(pcr: Optional[float], prev_pcr: Optional[float]) -> Optional[IndexSignal]:
    """PCR contrarian — trend-aware fear/complacency indicator.

    Threshold CROSSINGS carry higher conviction than sustained extremes:
      PCR crossing 1.30 today (was 1.15 yesterday) = fresh capitulation → score +3.0
      PCR already at 1.40 for 5 days = fear already priced in              → score +2.0
    Rising PCR into moderate zone also earns a partial boost vs static reading.
    This eliminates the "stale fear" mis-signal where day-5 of an extreme PCR
    fires at the same score as the initial spike that first predicted the bounce.
    """
    if pcr is None: return None

    rising      = prev_pcr is not None and pcr > prev_pcr
    falling     = prev_pcr is not None and pcr < prev_pcr
    fresh_spike = rising  and prev_pcr is not None and prev_pcr < 1.30
    fresh_drop  = falling and prev_pcr is not None and prev_pcr > 0.70
    trend_tag   = (f" (↑ from {prev_pcr:.2f})" if rising else
                   f" (↓ from {prev_pcr:.2f})" if falling else "")

    if pcr > 1.3:
        score = 3.0 if fresh_spike else 2.0
        extra = (f"Fresh spike from {prev_pcr:.2f} — first crossover = strongest contrarian signal. "
                 if fresh_spike else
                 f"{'Rising' if rising else 'Sustained'} extreme — fear partially priced in. ")
        return IndexSignal("PCR Extreme High — Contrarian Bullish", "Options OI", 1, score,
            f"PCR {pcr:.2f}{trend_tag} — Peak Fear (Contrarian BUY)",
            f"PCR {pcr:.2f} = extreme put buying = peak fear. {extra}"
            "When everyone is hedged, downside is already priced in. PCR >1.3 historically marks bottoms.", "🔄")
    if 1.1 <= pcr <= 1.3:
        s = 1.5 if rising else 1.0
        return IndexSignal("PCR Elevated — Mildly Bullish", "Options OI", 1, s,
            f"PCR {pcr:.2f}{trend_tag} — Defensive Hedging",
            f"PCR {pcr:.2f} above neutral. "
            f"{'Rising fear — fresh hedge demand building.' if rising else 'Participants buying protection.'} Bullish lean.", "🟡")
    if pcr < 0.70:
        score = -3.0 if fresh_drop else -2.0
        extra = (f"Fresh collapse from {prev_pcr:.2f} — first crossover = strongest contrarian signal. "
                 if fresh_drop else
                 f"{'Falling further' if falling else 'Sustained'} complacency — correction risk elevated. ")
        return IndexSignal("PCR Extreme Low — Contrarian Bearish", "Options OI", -1, score,
            f"PCR {pcr:.2f}{trend_tag} — Complacency (Contrarian SELL)",
            f"PCR {pcr:.2f} = extreme call buying = complacency. {extra}"
            "PCR <0.70 precedes corrections.", "🔄")
    if 0.70 <= pcr < 0.85:
        s = -1.5 if falling else -1.0
        return IndexSignal("PCR Low — Under-Hedged", "Options OI", -1, s,
            f"PCR {pcr:.2f}{trend_tag} — Mild Caution",
            f"PCR {pcr:.2f} below neutral. "
            f"{'Falling — participants unhedging into complacency.' if falling else 'Under-hedged.'} Vulnerable if sentiment shifts.", "🟡")
    return None


def _sig_opt_oi_premium(
    opt_near_today: pd.DataFrame,
    opt_near_prev: pd.DataFrame,
) -> Optional[IndexSignal]:
    """
    Options OI-Premium Matrix — disambiguates PCR by identifying BUYERS vs WRITERS.

    PCR tells you the ratio of put/call OI but NOT whether that OI was built
    by buyers (directional bet) or writers (premium collectors with opposite view).

    OI-Premium Matrix (applied per option type):
      OI ↑ + Premium ↑  →  Buying    (demand drives both up — direct directional bet)
      OI ↑ + Premium ↓  →  Writing   (supply: writers push premium down)
      OI ↓ + Premium ↑  →  Short Covering (writers buying back their shorts)
      OI ↓ + Premium ↓  →  Long Exiting  (buyers selling out their longs)

    Combined call + put signal:
      C.Buy  + P.Write  →  Net Long  (+2.5)  — both sides confirm bulls
      C.Write + P.Buy   →  Net Short (-2.5)  — both sides confirm bears
      C.Buy  + P.Buy    →  Straddle  (0)     — big move expected, no direction
      C.Write + P.Write →  Range Play (0)    — low vol / theta decay expected
    """
    if opt_near_today.empty:
        return None

    calls_t = opt_near_today[opt_near_today["option_type"] == "CE"]
    puts_t  = opt_near_today[opt_near_today["option_type"] == "PE"]
    if calls_t.empty and puts_t.empty:
        return None

    # ── Near-expiry noise guard ───────────────────────────────────────────────
    # On expiry day / day before (DTE ≤ 2), only a handful of ATM strikes trade.
    # Volume-weighted premium becomes a single-trade data point — pure noise.
    # Minimum 10,000 contracts needed across calls+puts for a statistically valid read.
    _MIN_CONTRACTS = 10_000
    total_contracts = float(
        calls_t[calls_t["contracts"] > 0]["contracts"].sum() +
        puts_t[puts_t["contracts"] > 0]["contracts"].sum()
    )
    if total_contracts < _MIN_CONTRACTS:
        return None   # too thin — signal would be noise, not information

    def _wt_prem_chg(df: pd.DataFrame) -> Optional[float]:
        """Volume-weighted (close_price − settle_price) across all strikes.
        settle_price = NSE previous-day settlement embedded in today's row.
        Thins strikes (contracts=0) are excluded to avoid zero-weight pollution."""
        df = df[df["contracts"] > 0]
        if df.empty:
            return None
        vol = float(df["contracts"].sum())
        if vol < 100:   # still too thin even after filter
            return None
        return float(((df["close_price"] - df["settle_price"]) * df["contracts"]).sum() / vol)

    def _oi_diff(opt_type: str) -> Optional[int]:
        t_oi = int(opt_near_today[opt_near_today["option_type"] == opt_type]["open_interest"].sum())
        if opt_near_prev.empty:
            return None
        p_oi = int(opt_near_prev[opt_near_prev["option_type"] == opt_type]["open_interest"].sum())
        return t_oi - p_oi

    call_prem = _wt_prem_chg(calls_t)
    put_prem  = _wt_prem_chg(puts_t)
    call_doi  = _oi_diff("CE")
    put_doi   = _oi_diff("PE")

    if call_prem is None and put_prem is None:
        return None

    def _classify(doi: Optional[int], prem: Optional[float], leg: str) -> str:
        if prem is None:
            return f"{leg}.—"
        oi_up = (doi is not None and doi > 0)
        oi_dn = (doi is not None and doi < 0)
        pr_up = prem > 0
        if oi_up and pr_up:   return f"{leg}.Buying"
        if oi_up and not pr_up: return f"{leg}.Writing"
        if oi_dn and pr_up:   return f"{leg}.SC"
        if oi_dn and not pr_up: return f"{leg}.LE"
        return f"{leg}.Neutral"

    call_sig = _classify(call_doi, call_prem, "C")
    put_sig  = _classify(put_doi,  put_prem,  "P")

    c_buy = call_sig == "C.Buying"
    c_wrt = call_sig == "C.Writing"
    c_sc  = call_sig == "C.SC"
    p_buy = put_sig  == "P.Buying"
    p_wrt = put_sig  == "P.Writing"
    p_sc  = put_sig  == "P.SC"

    def _fmt(v: Optional[float]) -> str:
        return f"{v:+.1f}pts" if v is not None else "—"
    def _doi_s(v: Optional[int]) -> str:
        return f"{v:+,}" if v is not None else "—"

    detail = (
        f"Call OI chg {_doi_s(call_doi)} | Call Prem {_fmt(call_prem)} → {call_sig}  "
        f"| Put OI chg {_doi_s(put_doi)} | Put Prem {_fmt(put_prem)} → {put_sig}"
    )
    base_desc = (
        "OI-Premium matrix: when OI and premium move in the SAME direction → Buying; "
        "OI up + premium DOWN → Writing (writers collected premium, bearish/bullish lean). "
        "This disambiguates PCR — a low PCR could mean call-buying (bullish) OR call-writing (bearish). "
        f"\n{detail}"
    )

    # ── Combined signal ────────────────────────────────────────────────────────
    if (c_buy or c_sc) and (p_wrt or p_sc):
        return IndexSignal(
            "Opt OI-Premium: Net Long (C.Buy + P.Write)", "Options OI", 1, 2.5,
            f"Call Buying + Put Writing — net institutional LONG (PCR ambiguity resolved)",
            f"Call OI rising + premium rising = direct call buyers (bullish). "
            f"Put OI rising + premium falling = put writers selling protection (also bullish). "
            f"Both sides confirm long bias — stronger signal than PCR alone.\n{detail}", "🔥",
        )
    if (c_wrt or c_sc) and (p_buy or p_sc):
        return IndexSignal(
            "Opt OI-Premium: Net Short (C.Write + P.Buy)", "Options OI", -1, -2.5,
            f"Call Writing + Put Buying — net institutional SHORT (PCR ambiguity resolved)",
            f"Call OI rising + premium falling = call writers capping upside (bearish). "
            f"Put OI rising + premium rising = direct put buyers (bearish hedge). "
            f"Both sides confirm distribution.\n{detail}", "❄️",
        )
    if c_buy and p_buy:
        return IndexSignal(
            "Opt OI-Premium: Straddle (C.Buy + P.Buy)", "Options OI", 0, 0.0,
            f"Call + Put Buying — volatility bet, no directional edge",
            f"Both calls AND puts OI rising with rising premiums = straddle/strangle buyers. "
            f"Big move expected — direction unknown. Watch max pain / key S-R for breakout side.\n{detail}", "⚡",
        )
    if c_wrt and p_wrt:
        return IndexSignal(
            "Opt OI-Premium: Range Play (C.Write + P.Write)", "Options OI", 0, 0.0,
            f"Call + Put Writing — iron condor / range-bound (low vol expected)",
            f"Both calls AND puts OI rising with falling premiums = premium writers on both sides. "
            f"Option writers expect the index to stay in a range. Pinning near max pain likely.\n{detail}", "📊",
        )
    # ── Single-sided signals ──────────────────────────────────────────────────
    if c_buy:
        return IndexSignal(
            "Opt OI-Premium: Call Buying (Direct Bullish)", "Options OI", 1, 1.5,
            f"Call Buying — participants PAYING UP for upside exposure",
            f"Call OI rising + premium rising = direct bullish bet, NOT contrarian. "
            f"Participants are buying calls at higher prices — genuine demand.\n{detail}", "🟢",
        )
    if p_wrt:
        return IndexSignal(
            "Opt OI-Premium: Put Writing (Bullish Lean)", "Options OI", 1, 1.0,
            f"Put Writing — writers selling downside protection (bullish lean)",
            f"Put OI rising + premium falling = put writers. "
            f"They accept downside risk for premium — expect market to hold or rise.\n{detail}", "🟡",
        )
    if p_buy:
        return IndexSignal(
            "Opt OI-Premium: Put Buying (Direct Bearish)", "Options OI", -1, -1.5,
            f"Put Buying — participants PAYING UP for downside protection",
            f"Put OI rising + premium rising = direct bearish hedge, NOT contrarian. "
            f"Participants are buying puts at higher prices — genuine fear.\n{detail}", "🔴",
        )
    if c_wrt:
        return IndexSignal(
            "Opt OI-Premium: Call Writing (Bearish Lean)", "Options OI", -1, -1.0,
            f"Call Writing — writers capping upside (bearish lean)",
            f"Call OI rising + premium falling = call writers. "
            f"They cap the upside — expect the market to stay below the strike.\n{detail}", "🟠",
        )
    return None


def _sig_max_pain(max_pain: Optional[float], spot_close: float, dte: int) -> Optional[IndexSignal]:
    """Max pain gravity — McMillan. Meaningful within 5 DTE."""
    if max_pain is None or dte > 5: return None
    gap = max_pain - spot_close; gap_pct = gap / spot_close * 100
    if abs(gap_pct) < 0.25: return None
    w = 1.5 if dte <= 2 else 1.0
    if gap > 0:
        return IndexSignal(f"Max Pain {max_pain:.0f} — Above Spot", "Options OI", 1, w,
            f"Max Pain {max_pain:.0f} ({gap:.0f} pts up) — Bullish Pin",
            f"Max pain {max_pain:.0f} vs spot {spot_close:.0f} — expiry in {dte}d. "
            "Market gravitates toward max pain as DTE→0.", "📌")
    return IndexSignal(f"Max Pain {max_pain:.0f} — Below Spot", "Options OI", -1, -w,
        f"Max Pain {max_pain:.0f} ({abs(gap):.0f} pts down) — Bearish Pin",
        f"Max pain {max_pain:.0f} vs spot {spot_close:.0f} — expiry in {dte}d. "
        "Option writers resist moves above spot. Capping pressure.", "📌")


def _sig_range_position(spot_close, high, low, pct_chg) -> Optional[IndexSignal]:
    """Wyckoff: close vs day's range reveals buying/selling conviction."""
    if None in (spot_close, high, low, pct_chg): return None
    r = high - low
    if r < 5.0: return None
    pos = (spot_close - low) / r
    if pos >= 0.75 and pct_chg > 0.30:
        return IndexSignal("Closed Near High — Accumulation", "Price Action", 1, 1.0,
            f"Closed Near HOD ({pos*100:.0f}% of range)",
            f"Closed at {spot_close:.0f} — {pos*100:.0f}% of range (H:{high:.0f}/L:{low:.0f}). "
            "Buyers dominated and defended gains into close — strength.", "💪")
    if pos <= 0.25 and pct_chg < -0.30:
        return IndexSignal("Closed Near Low — Distribution", "Price Action", -1, -1.0,
            f"Closed Near LOD ({pos*100:.0f}% of range)",
            f"Closed at {spot_close:.0f} — {pos*100:.0f}% of range (H:{high:.0f}/L:{low:.0f}). "
            "Sellers dominated with no recovery — weakness.", "📉")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTIONS — FII / Institutional (signals 7-13)
# ═══════════════════════════════════════════════════════════════════════════════

def _sig_fii_institutional(ctx: MarketContext) -> Optional[IndexSignal]:
    """
    Informed-desk positioning in index futures (FAO participant-wise OI).

    Index futures are zero-sum: FII + DII + Pro + Client = 0. So "FII short vs
    Client long" is an accounting IDENTITY — Client (retail) is the residual mirror
    of the institutions, not independent evidence. And FIIs are structurally net-SHORT
    index futures as a standing hedge on their cash longs, so that read fired on ~100%
    of days with no edge (51% hit in the walk-forward).

    We instead read the UNHEDGED informed cohorts whose positioning is genuinely
    directional:
      • DII — domestic institutions (mutual funds / insurers), counter-cyclical.
      • Pro — proprietary / prop desks: sharp, unhedged, short-horizon money
              (previously loaded but never used).
    Their combined net is the institutional lean. FII's hedge-heavy level is shown for
    context only (its informative part — covering vs adding — is the separate FII
    Position-Change signal); Client is the retail counterparty. A genuine FII NET-LONG
    (rare — the hedge dropped) is kept as a distinct strong-bullish tell.
    Score decays with data age via _lag_mult.
    """
    if ctx.fao_date is None:
        return None
    fii = ctx.fii_fut_idx_net
    dii = ctx.dii_fut_idx_net
    pro = ctx.pro_fut_idx_net
    cli = ctx.client_fut_idx_net
    lag = (ctx.trade_date - ctx.fao_date).days
    lm  = _lag_mult(lag)
    tag = f" [FAO {ctx.fao_date.strftime('%d %b')}{', ' + str(lag) + 'd lag' if lag else ''}]"

    roles = f"FII {fii:+,} (mm/hedge) · DII {dii:+,} (support) · Pro {pro:+,} (directional) · Client {cli:+,} (retail)"

    # ── 1) Rare ROLE-BREAKS — the strongest tells ────────────────────────────────
    # FII net LONG: market makers dropped the structural hedge and went long — rare,
    # strongly bullish.
    if fii > 80_000:
        return IndexSignal("FII Net LONG Index Futures — Hedge Dropped (Strong Bull)",
            "Institutional", 1, round(3.0 * lm, 2),
            f"FII {fii:+,} (net LONG){tag}",
            f"FIIs are net LONG {fii:,} index futures — they rarely abandon the hedge. "
            f"{roles}. A genuine institutional upside bet.", "🐂")
    # DII net SHORT: domestic institutions normally net long to SUPPORT the market —
    # a net short means support is withdrawn, a real bearish warning.
    if dii < -40_000:
        return IndexSignal("DII Net SHORT — Support Withdrawn (Bearish Warning)",
            "Institutional", -1, round(-2.5 * lm, 2),
            f"DII {dii:+,} (net SHORT){tag}",
            f"DIIs are net SHORT {abs(dii):,} index futures — they normally buy to cushion "
            f"the downside, so a short stance removes the floor. {roles}.", "📉")

    # ── 2) Pro = primary DIRECTIONAL read (cautious, condition-dependent desks) ──
    # FII (hedge) and DII (support) carry structural defaults, so Pro — which has none —
    # is the cleanest informed directional tell. DII supporting in the same direction
    # adds a mild confirmation note; it is not itself the driver.
    PRO_T = 10_000
    if pro > PRO_T:
        dii_note = " DII also supporting (net long) — downside cushioned." if dii > 0 else ""
        return IndexSignal("Pro Desks Net Long — Informed Directional Bullish",
            "Institutional", 1, round(1.5 * lm, 2),
            f"Pro {pro:+,} (net LONG){tag}",
            f"Proprietary desks — the cautious, condition-driven money — are net LONG "
            f"index futures, the cleanest informed directional lean.{dii_note} {roles}.", "🏦")
    if pro < -PRO_T:
        return IndexSignal("Pro Desks Net Short — Informed Directional Bearish",
            "Institutional", -1, round(-1.5 * lm, 2),
            f"Pro {pro:+,} (net SHORT){tag}",
            f"Proprietary desks — the cautious, condition-driven money — have turned net "
            f"SHORT index futures, an informed bearish lean. {roles}.", "📉")
    return None


def _sig_fii_options_delta(ctx: MarketContext) -> Optional[IndexSignal]:
    """FII net call minus net put OI — directional bias via options book."""
    if ctx.fao_date is None: return None
    delta = ctx.fii_opt_delta; cn = ctx.fii_opt_call_net; pn = ctx.fii_opt_put_net
    lag = (ctx.trade_date - ctx.fao_date).days
    lm  = _lag_mult(lag)
    tag = f" [FAO {ctx.fao_date.strftime('%d %b')}{', ' + str(lag) + 'd lag' if lag else ''}]"
    if delta > 150_000:
        return IndexSignal("FII Options Delta Bullish — Net Long Calls",
            "Institutional", 1, round(1.5 * lm, 2), f"FII Options Delta +{delta:,}{tag}",
            f"FII net LONG {cn:,} calls vs LONG {pn:,} puts. "
            f"Options delta +{delta:,} = institutional upside bias via options.", "📊")
    if delta < -150_000:
        # Score 0.0 (context-only): FII run a permanent net-long-put book as
        # downside insurance on cash holdings, so this fires ~every day (148/152
        # in the diagnostic) with no directional edge. Display for context only.
        return IndexSignal("FII Options Bearish — Heavy Put Hedge",
            "Institutional", -1, 0.0, f"FII Options Delta {delta:,}{tag}",
            f"FII net LONG {pn:,} puts vs SHORT {abs(cn):,} calls. "
            f"Options delta {delta:,} = FII structurally hedged against downside "
            "(standing put book, not a fresh directional bet).", "🛡️")
    return None


def _sig_options_participant(ctx: MarketContext) -> Optional[IndexSignal]:
    """
    Role-aware index-OPTIONS positioning (FAO participant-wise OI).

    Options market structure (per participant delta = call_net − put_net):
      • FII — the real MARKET MAKER / smart money in index options (runs the dominant
        book). Its footprint is the informed side.
      • Pro — proprietary desks that largely SHADOW the FII footprint; alignment with FII
        CONFIRMS the read (Pro on FII's side = smart-money consensus).
      • Client (retail) & DII — the WEAK / uninformed side; Client is the dumb-money crowd
        you FADE, DII is negligible in options and ignored.

    Signal: when retail (Client) is at an EXTREME OPPOSITE the FII footprint — the normal
    smart-vs-dumb divergence — fade the crowd (i.e. side with FII). Pro confirming FII
    strengthens it; Pro diverging from FII weakens it.
    """
    if ctx.fao_date is None:
        return None
    cli = ctx.client_opt_delta
    fii = ctx.fii_opt_delta
    pro = ctx.pro_opt_delta
    lag = (ctx.trade_date - ctx.fao_date).days
    lm  = _lag_mult(lag)
    tag = f" [FAO {ctx.fao_date.strftime('%d %b')}{', ' + str(lag) + 'd lag' if lag else ''}]"

    RETAIL_T = 500_000   # extreme retail options skew (net contracts) — heuristic
    if abs(cli) < RETAIL_T or fii == 0:
        return None
    if (cli * fii) > 0:
        return None      # retail aligned with the FII footprint — no smart-vs-dumb edge

    pro_follows_fii = (pro * fii) > 0    # Pro shadowing the market maker = confirmation
    base   = 1.5 if pro_follows_fii else 0.8
    score  = round(base * lm, 2)
    pro_txt = (f" Pro {pro:+,} {'shadows FII (confirms)' if pro_follows_fii else 'diverges from FII (weakens)'}.")
    smart_txt = f"Smart money: FII (market maker) delta {fii:+,}.{pro_txt}"

    if cli > 0:   # retail bullish, opposite the FII footprint → fade → BEARISH
        return IndexSignal("Retail Options Euphoria vs FII Footprint — Contrarian Bearish",
            "Institutional", -1, -score, f"Client {cli:+,} vs FII {fii:+,}{tag}",
            f"Retail (Client) heavily bullish via options (delta +{cli:,}: long calls / "
            f"put writing) against the FII market-maker book. {smart_txt} Crowded retail "
            "opposite smart money tends to resolve in the institutions' favour.", "🎈")
    return IndexSignal("Retail Options Fear vs FII Footprint — Contrarian Bullish",
        "Institutional", 1, score, f"Client {cli:+,} vs FII {fii:+,}{tag}",
        f"Retail (Client) heavily bearish via options (delta {cli:,}: heavy put buying) "
        f"against the FII market-maker book. {smart_txt} Capitulation opposite smart money "
        "tends to mean-revert up.", "😱")


def _sig_fii_flow(ctx: MarketContext, fno_symbol: str) -> Optional[IndexSignal]:
    """FII today's ₹Cr buy vs sell for this specific index futures."""
    if ctx.fii_stats_date is None: return None
    net = ctx.fii_symbol_flows.get(fno_symbol)
    if net is None: return None
    lag = (ctx.trade_date - ctx.fii_stats_date).days
    lm  = _lag_mult(lag)
    tag = f" [FII Stats {ctx.fii_stats_date.strftime('%d %b')}{', ' + str(lag) + 'd lag' if lag else ''}]"
    if net >= 500:
        s = round((2.0 if net >= 2_000 else 1.0) * lm, 2)
        return IndexSignal(f"FII Net Buyer — {fno_symbol} Futures",
            "Institutional", 1, s,
            f"FII Net Bought Rs{net:,.0f}Cr in {fno_symbol} Futures{tag}",
            f"FII net bought Rs{net:,.0f}Cr of {fno_symbol} index futures. "
            "Active institutional accumulation — directional conviction.", "💰")
    if net <= -500:
        s = round((-2.0 if net <= -2_000 else -1.0) * lm, 2)
        return IndexSignal(f"FII Net Seller — {fno_symbol} Futures",
            "Institutional", -1, s,
            f"FII Net Sold Rs{abs(net):,.0f}Cr in {fno_symbol} Futures{tag}",
            f"FII net sold Rs{abs(net):,.0f}Cr of {fno_symbol} index futures. "
            "Institutional distribution — selling pressure.", "💸")
    return None


def _sig_fii_cumulative_flow(ctx: MarketContext, fno_symbol: str) -> Optional[IndexSignal]:
    """
    5-day cumulative FII futures flow in ₹Cr.
    Single-day flows can be noise; 5D trend reveals systematic accumulation vs distribution.
    Threshold: Rs2,000 Cr sustained = institutional trend; Rs5,000 Cr = extreme.
    """
    cumul = ctx.fii_cumul_flows_5d.get(fno_symbol)
    if cumul is None: return None
    lag = (ctx.trade_date - ctx.fii_stats_date).days if ctx.fii_stats_date else 0
    lm  = _lag_mult(lag)
    tag = f" [5D ending {ctx.fii_stats_date.strftime('%d %b') if ctx.fii_stats_date else 'N/A'}{', ' + str(lag) + 'd lag' if lag else ''}]"

    if cumul >= 2_000:
        s = round((2.5 if cumul >= 5_000 else 1.5) * lm, 2)
        return IndexSignal(f"FII 5D Accumulation — {fno_symbol}",
            "Institutional", 1, s,
            f"FII 5D Cumulative +Rs{cumul:,.0f}Cr in {fno_symbol} Futures{tag}",
            f"FII net bought Rs{cumul:,.0f}Cr over 5 trading days in {fno_symbol} futures. "
            "Sustained institutional accumulation — systematic buying, not a one-day event. "
            "This is the strongest institutional trend signal available.", "🏦")
    if cumul <= -2_000:
        s = round((-2.5 if cumul <= -5_000 else -1.5) * lm, 2)
        return IndexSignal(f"FII 5D Distribution — {fno_symbol}",
            "Institutional", -1, s,
            f"FII 5D Cumulative Rs{cumul:,.0f}Cr in {fno_symbol} Futures{tag}",
            f"FII net sold Rs{abs(cumul):,.0f}Cr over 5 trading days in {fno_symbol} futures. "
            "Systematic institutional distribution — not noise. "
            "Sustained FII selling = persistent overhead pressure on every rally.", "🏦")
    return None


def _sig_fii_oi_buildup(ctx: MarketContext, fno_symbol: str) -> Optional[IndexSignal]:
    """
    FII OI ₹Cr trend: is FII building or unwinding their position?
    Growing OI + net short = adding conviction to their short = more bearish.
    Shrinking OI + net short = covering = potential reversal catalyst.
    """
    # NIFTY only: the "while NET SHORT/LONG" qualifier below comes from the POOLED
    # FAO net (fii_fut_idx_net, ~76% NIFTY). Pairing BankNifty's per-symbol OI ₹Cr
    # trend with a NIFTY-dominated position sign misattributes the direction.
    if fno_symbol != "NIFTY": return None
    oi_now = ctx.fii_oi_cr_latest.get(fno_symbol)
    oi_ago = ctx.fii_oi_cr_5d_ago.get(fno_symbol)
    if oi_now is None or oi_ago is None or abs(oi_ago) < 1.0: return None

    chg_pct = (oi_now - oi_ago) / oi_ago * 100
    fii_net = ctx.fii_fut_idx_net
    lag = (ctx.trade_date - ctx.fii_stats_date).days if ctx.fii_stats_date else 0
    lm  = _lag_mult(lag)
    tag = f" [OI: Rs{oi_now:,.0f}Cr, 5D chg: {chg_pct:+.1f}%{', ' + str(lag) + 'd lag' if lag else ''}]"

    if chg_pct >= 10 and fii_net < -50_000:
        return IndexSignal("FII Building Short Position — High Conviction",
            "Institutional", -1, round(-2.0 * lm, 2),
            f"FII OI grew {chg_pct:+.1f}% in 5D while NET SHORT{tag}",
            f"FII's {fno_symbol} futures OI increased from Rs{oi_ago:,.0f}Cr to Rs{oi_now:,.0f}Cr "
            f"({chg_pct:+.1f}% in 5D). Growing OI with net short position = FII adding conviction "
            "to their bearish bet. Not a hedge — a directional call.", "🔨")
    if chg_pct >= 10 and fii_net > 50_000:
        return IndexSignal("FII Building Long Position — High Conviction",
            "Institutional", 1, round(2.0 * lm, 2),
            f"FII OI grew {chg_pct:+.1f}% in 5D while NET LONG{tag}",
            f"FII OI increased {chg_pct:+.1f}% in 5D. Growing OI with net long = "
            "FII adding to bullish bet. Strong institutional conviction for upside.", "🏗️")
    if chg_pct <= -10 and fii_net < -50_000:
        return IndexSignal("FII Unwinding Short — Potential Reversal",
            "Institutional", 1, round(1.5 * lm, 2),
            f"FII OI SHRINKING {chg_pct:+.1f}% in 5D while short{tag}",
            f"FII OI dropped from Rs{oi_ago:,.0f}Cr to Rs{oi_now:,.0f}Cr ({chg_pct:+.1f}% in 5D). "
            "Short position shrinking = FII covering. "
            "Position unwinds create buying pressure and short squeeze fuel.", "🔄")
    if chg_pct <= -10 and fii_net > 50_000:
        return IndexSignal("FII Reducing Long — Distribution",
            "Institutional", -1, round(-1.5 * lm, 2),
            f"FII OI shrinking {chg_pct:+.1f}% in 5D while long{tag}",
            f"FII reducing long exposure — potential distribution phase.", "⚠️")
    return None


def _sig_fii_position_change(ctx: MarketContext) -> Optional[IndexSignal]:
    """
    Day-over-day change in FII net index futures position (from FAO participant data).
    Adding shorts = increasing conviction; covering shorts = potential reversal trigger.
    """
    if ctx.fii_prev_fao_date is None: return None
    chg     = ctx.fii_net_change_1d
    cur_net = ctx.fii_fut_idx_net
    lag     = (ctx.trade_date - ctx.fao_date).days if ctx.fao_date else 0
    lm      = _lag_mult(lag)
    tag     = f" [{ctx.fii_prev_fao_date.strftime('%d %b')} → {ctx.fao_date.strftime('%d %b')}{', ' + str(lag) + 'd lag' if lag else ''}]" if ctx.fao_date else ""

    # Only signal on meaningful moves (>5,000 contracts; ±10,000 = strong conviction)
    if chg < -5_000:
        s = round((-2.0 if chg < -10_000 else -1.0) * lm, 2)
        return IndexSignal("FII Adding to Short — Increasing Conviction",
            "Institutional", -1, s,
            f"FII added {abs(chg):,} SHORT contracts in 1 day{tag}",
            f"FII increased short by {abs(chg):,} contracts (now {cur_net:+,}). "
            f"Aggressively building bearish position — not covering despite recent market moves. "
            "This is real conviction: institutions are doubling down on their bearish thesis.", "🐻")
    if chg > 5_000:
        s = round((2.0 if chg > 10_000 else 1.0) * lm, 2)
        return IndexSignal("FII Covering Shorts — Reversal Trigger",
            "Institutional", 1, s,
            f"FII covered {chg:,} SHORT contracts in 1 day{tag}",
            f"FII reduced short by {chg:,} contracts (now {cur_net:+,}). "
            "Short covering = buying pressure. "
            "When FII covers at this scale, market typically moves up sharply.", "🔄")
    return None


def _sig_short_squeeze_setup(
    ctx: MarketContext,
    fno_symbol: str,
    pct_chg: Optional[float],
    range_pos: Optional[float],
    pcr: Optional[float],
) -> Optional[IndexSignal]:
    """
    Livermore's Trap — the market forces the most pain.
    Setup: FII is very short + large OI + market closing UP + PCR extreme.
    When shorts are trapped with no escape, forced covering tends to spark sharp
    rallies (directional bias — not a validated win-rate).
    """
    if ctx.fao_date is None: return None
    fii_net = ctx.fii_fut_idx_net
    if fii_net >= -100_000: return None            # need significant short base
    if (pct_chg or 0) <= 0: return None            # need market up today (shorts trapped)

    near_hod    = range_pos is not None and range_pos > 0.65
    extreme_pcr = pcr is not None and pcr > 1.2
    big_up      = (pct_chg or 0) > 0.5

    hits = sum([near_hod, extreme_pcr, big_up])
    if hits < 2: return None

    lag   = (ctx.trade_date - ctx.fao_date).days
    lm    = _lag_mult(lag)
    tag   = f" [FAO {ctx.fao_date.strftime('%d %b')}{', ' + str(lag) + 'd lag' if lag else ''}]"
    score = round((2.5 if hits == 3 else 2.0) * lm, 2)

    criteria = []
    if near_hod:    criteria.append(f"HOD close ({range_pos*100:.0f}% of range)")
    if extreme_pcr: criteria.append(f"extreme PCR {pcr:.2f}")
    if big_up:      criteria.append(f"strong rally +{pct_chg:.2f}%")

    return IndexSignal("Short Squeeze Setup — FII Trapped",
        "Institutional", 1, score,
        f"FII Short Squeeze: {abs(fii_net):,} contracts trapped + {', '.join(criteria)}{tag}",
        f"FII holding {abs(fii_net):,} net short contracts while market closed at "
        f"{(range_pos or 0)*100:.0f}% of range with PCR {pcr if pcr is not None else 0.0:.2f}. "
        "Livermore's Trap: when every short is already in, the next move is UP. "
        f"Confirmations: {', '.join(criteria)}. Forced covering tends to spark sharp rallies.", "💥")


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTIONS — Market Context (signals 14-17)
# ═══════════════════════════════════════════════════════════════════════════════

def _sig_vix_regime(ctx: MarketContext) -> Optional[IndexSignal]:
    """India VIX regime + trend. Whaley: VIX inversely correlated with markets."""
    vix = ctx.vix_close; chg = ctx.vix_pct_chg
    if vix is None: return None

    if vix < 12.0:
        return IndexSignal("VIX Complacency Zone", "Market Context", -1, -0.5,
            f"India VIX {vix:.1f} — Extreme Calm",
            f"VIX {vix:.1f} = extreme complacency. Sub-12 VIX historically precedes spikes.", "😴")
    if vix < 18.0:
        if chg and chg <= -5.0:
            return IndexSignal("VIX Falling — Fear Clearing", "Market Context", 1, 1.5,
                f"India VIX {vix:.1f} ({chg:+.1f}%) — Fear Reducing",
                f"VIX dropped {abs(chg):.1f}% to {vix:.1f}. Falling VIX = institutions "
                "selling protection = confidence building = bullish.", "🎯")
        if chg and chg >= 5.0:
            return IndexSignal("VIX Rising — Anxiety Building", "Market Context", -1, -1.5,
                f"India VIX {vix:.1f} ({chg:+.1f}%) — Volatility Uptick",
                f"VIX rose {chg:.1f}% to {vix:.1f}. Rising VIX = increasing hedge demand.", "⚡")
        return None
    if vix < 22.0:
        if chg and chg <= -5.0:
            return IndexSignal("VIX Falling from Elevated — Relief Rally", "Market Context", 1, 2.0,
                f"India VIX {vix:.1f} ({chg:+.1f}%) — Elevated but Falling Fast",
                f"VIX elevated at {vix:.1f} but falling {abs(chg):.1f}% — peak fear receding. "
                "Historical: sharp VIX drop from elevated = next 3-5 sessions bullish.", "📉")
        return IndexSignal("VIX Elevated — Heightened Uncertainty", "Market Context", -1, -1.0,
            f"India VIX {vix:.1f} — Above Normal Range",
            f"VIX {vix:.1f} = elevated risk environment. Expect wider swings; await catalyst.", "⚠️")
    s = 2.0 if vix > 25 else 1.0
    return IndexSignal("VIX Fear Zone — Potential Capitulation", "Market Context", 1, s,
        f"India VIX {vix:.1f} — Peak Fear (Contrarian BUY)",
        f"VIX {vix:.1f} = extreme fear zone (>22). Peak fear marks bottoms — "
        "when everyone has sold, few sellers remain — peak fear often precedes a bounce.", "🔥")


def _sig_sector_breadth(ctx: MarketContext) -> Optional[IndexSignal]:
    """Sector breadth — Zweig. >80% advancing = durable rally; <25% = broad selloff."""
    breadth = ctx.breadth_pct_advancing
    if breadth is None or ctx.total_sectors < 10: return None
    adv = ctx.advancing_sectors; tot = ctx.total_sectors
    if breadth >= 80:
        return IndexSignal("Broad-Based Rally — High Breadth", "Sector", 1, 2.0,
            f"Breadth {breadth:.0f}% — {adv}/{tot} Sectors Advancing",
            f"{adv}/{tot} sector indices advancing ({breadth:.0f}%). "
            "Broad participation = institutional capital across all sectors. "
            "Zweig: breadth >80% = durable rally, low reversal probability.", "🌐")
    if breadth >= 65:
        return IndexSignal("Good Market Breadth", "Sector", 1, 1.0,
            f"Breadth {breadth:.0f}% — {adv}/{tot} Sectors Up",
            f"{adv}/{tot} sectors advancing. Solid breadth supports index.", "📊")
    if breadth <= 25:
        return IndexSignal("Broad-Based Selloff — Weak Breadth", "Sector", -1, -2.0,
            f"Breadth {breadth:.0f}% — Only {adv}/{tot} Advancing",
            f"Only {adv}/{tot} sectors advancing ({breadth:.0f}%). "
            "Broad de-risking — institutions selling across all sectors.", "🌐")
    if breadth <= 40:
        return IndexSignal("Weak Breadth — Concentrated Move", "Sector", -1, -1.0,
            f"Breadth {breadth:.0f}% — Narrow Market",
            f"Only {adv}/{tot} sectors advancing. Narrow = concentration risk.", "📉")
    return None


def _sig_defensive_cyclical(ctx: MarketContext) -> Optional[IndexSignal]:
    """Risk-ON/OFF rotation — O'Neil (CANSLIM). Leading sectors precede index direction."""
    cyc = ctx.cyclical_avg_chg; dfe = ctx.defensive_avg_chg
    if cyc is None or dfe is None: return None
    diff = cyc - dfe
    if ctx.risk_mode == "RISK-ON" and diff > 0.5:
        return IndexSignal("Risk-ON — Cyclicals Leading", "Sector", 1, 1.5,
            f"Cyclicals +{cyc:.1f}% vs Defensives +{dfe:.1f}% — RISK-ON",
            f"Cyclicals (Bank/Auto/Metal) avg {cyc:.1f}% vs Defensives (FMCG/Pharma) "
            f"avg {dfe:.1f}% — gap {diff:.1f}%. Institutions rotating into growth sectors. "
            "O'Neil: cyclical leadership precedes sustained index breakouts.", "⚡")
    if ctx.risk_mode == "RISK-OFF" and diff < -0.5:
        return IndexSignal("Risk-OFF — Defensives Dominating", "Sector", -1, -1.5,
            f"Defensives +{dfe:.1f}% vs Cyclicals +{cyc:.1f}% — RISK-OFF",
            f"Defensives avg {dfe:.1f}% vs Cyclicals avg {cyc:.1f}% — gap {abs(diff):.1f}%. "
            "Flight to safety = institutions reducing risk exposure.", "🛡️")
    return None


def _sig_index_relative_strength(
    fno_symbol: str,
    index_pct_chg: Optional[float],
    ctx: MarketContext,
) -> Optional[IndexSignal]:
    """
    Sector relative strength vs Nifty 50 — non-Nifty exclusive signal.

    For BankNifty / FinNifty / MidcapNifty: today's performance relative to
    the broad market reveals which way institutional capital is rotating.

    WHY THIS MATTERS:
      Nifty 50 is the broad market. When BankNifty outperforms Nifty today,
      money is specifically rotating INTO banking — this rotation momentum
      tends to persist short-term (O'Neil CANSLIM RS / Mansfield RS principle).
      When BankNifty underperforms, institutions are moving OUT of banking even
      on a market-up day — relative weakness signals continued distribution.

    This signal fills the gap left by market-wide signals: sector breadth (5% or
    90% advancing) and defensive/cyclical rotation are both Nifty-centric. None
    of them answer "is BankNifty specifically attracting capital right now?"

    Thresholds: ±0.75% RS = actionable edge; ±1.50% = strong rotation conviction.
    Not fired for NIFTY since relative-to-itself is undefined.
    """
    if fno_symbol == "NIFTY": return None
    if index_pct_chg is None or ctx.nifty_pct_chg is None: return None

    relative = index_pct_chg - ctx.nifty_pct_chg

    if relative >= 0.75:
        s = 2.0 if relative >= 1.50 else 1.0
        return IndexSignal(
            f"Sector RS: {fno_symbol} Outperforming Nifty — Rotation IN",
            "Sector", 1, s,
            f"{fno_symbol} {index_pct_chg:+.2f}% vs Nifty {ctx.nifty_pct_chg:+.2f}% (RS {relative:+.2f}%)",
            f"Outperformed Nifty 50 by {relative:.2f}% today — institutional capital is "
            f"rotating specifically INTO this sector. "
            f"O'Neil RS / Mansfield principle: sectors leading the market today tend to "
            f"attract continued inflows tomorrow as momentum players follow the signal. "
            f"{'Strong rotation — multiple standard deviations above market.' if relative >= 1.50 else 'Moderate rotation — clear preference visible.'}",
            "⚡"
        )
    if relative <= -0.75:
        s = -2.0 if relative <= -1.50 else -1.0
        return IndexSignal(
            f"Sector RS: {fno_symbol} Underperforming Nifty — Rotation OUT",
            "Sector", -1, s,
            f"{fno_symbol} {index_pct_chg:+.2f}% vs Nifty {ctx.nifty_pct_chg:+.2f}% (RS {relative:+.2f}%)",
            f"Underperformed Nifty 50 by {abs(relative):.2f}% today — capital rotating OUT of "
            f"this sector even as the broader market {'rises' if ctx.nifty_pct_chg and ctx.nifty_pct_chg > 0 else 'falls'}. "
            f"Mansfield RS: persistent underperformers continue to see distribution. "
            f"{'Severe underperformance — institutional sector preference has clearly shifted.' if relative <= -1.50 else 'Relative weakness persisting — watch for recovery above Nifty level to confirm reversal.'}",
            "📉"
        )
    return None


def _sig_constituent_breadth(
    stats: _ConstituentStats, fno_symbol: str,
) -> Optional[IndexSignal]:
    """
    Signal 17c: Constituent breadth + delivery quality — NIFTY, BANKNIFTY, FINNIFTY.

    Turnover-weighted advance ratio distinguishes index-weight-relevant moves from
    noise. Delivery % further separates institutional accumulation from speculation.

    SCORING (turnover-weighted advance ratio, 0–100%):
      ≥ 75%  → +2.5  Exceptional breadth: almost the entire basket rising
      ≥ 60%  → +1.5  Strong breadth: majority rising with broad participation
      40–60% → None  Mixed: no edge, stocks split roughly evenly
      ≤ 40%  → -1.5  Weak breadth: majority declining
      ≤ 25%  → -2.5  Exceptional weakness: near-universal decline

    DELIVERY MODIFIER:
      +0.5 advancing avg delivery ≥ 55% → institutional accumulation
      -0.3 advancing avg delivery ≤ 25% → speculative rally, low conviction
      -0.5 declining avg delivery ≥ 55% → institutional distribution
    """
    short_lbl, full_lbl = _CONSTITUENT_INFO.get(fno_symbol, ("IDX", "Index"))
    min_cov = max(5, stats.total_stocks // 2)
    if stats.coverage < min_cov or stats.adv_turn_pct is None:
        return None

    twar    = stats.adv_turn_pct
    adv     = stats.advancing
    dec     = stats.declining
    cov     = stats.coverage
    adv_del = stats.adv_deliv_avg
    dec_del = stats.dec_deliv_avg

    if twar >= 75:
        base  = 2.5
        name  = f"{short_lbl} Exceptional Breadth — Broad-Based Basket Rally"
        head  = f"{adv}/{cov} {full_lbl} stocks up ({twar:.0f}% of turnover) — Exceptional"
        emoji = "🟢"
    elif twar >= 60:
        base  = 1.5
        name  = f"{short_lbl} Strong Breadth — Majority of Basket Rising"
        head  = f"{adv}/{cov} {full_lbl} stocks up ({twar:.0f}% of turnover) — Broad Participation"
        emoji = "🟢"
    elif twar <= 25:
        base  = -2.5
        name  = f"{short_lbl} Exceptional Weakness — Near-Universal Basket Decline"
        head  = f"{dec}/{cov} {full_lbl} stocks down ({100-twar:.0f}% of turnover) — Broad Sell-Off"
        emoji = "🔴"
    elif twar <= 40:
        base  = -1.5
        name  = f"{short_lbl} Weak Breadth — Majority of Basket Declining"
        head  = f"{dec}/{cov} {full_lbl} stocks down ({100-twar:.0f}% of turnover) — Weak Participation"
        emoji = "🔴"
    else:
        return None   # 40–60%: mixed, no directional edge

    deliv_note = ""
    if base > 0:
        if adv_del is not None and adv_del >= 55:
            base += 0.5
            deliv_note = (
                f" Advancing stocks: {adv_del:.0f}% avg delivery = institutional quality — "
                "real money buying the basket, not just intraday momentum."
            )
            emoji = "💎"
        elif adv_del is not None and adv_del <= 25:
            base -= 0.3
            deliv_note = (
                f" Advancing stocks: only {adv_del:.0f}% avg delivery = speculative rally. "
                "Low conviction — intraday/leveraged buying, not institutional accumulation."
            )
            emoji = "🟡"
        elif adv_del is not None:
            deliv_note = f" Advancing stocks: {adv_del:.0f}% avg delivery."
    else:
        if dec_del is not None and dec_del >= 55:
            base -= 0.5
            deliv_note = (
                f" Declining stocks: {dec_del:.0f}% avg delivery = institutional distribution — "
                "real money selling the basket."
            )
            emoji = "💀"
        elif dec_del is not None:
            deliv_note = f" Declining stocks: {dec_del:.0f}% avg delivery."

    direction = 1 if base > 0 else -1
    flat = cov - adv - dec
    desc = (
        f"{full_lbl} constituent breadth: {adv} advancing / {dec} declining / "
        f"{flat} flat out of {cov} stocks tracked. "
        f"Turnover-weighted advance ratio: {twar:.1f}% — large-cap members contribute "
        f"proportionally to their index weight."
        f"{deliv_note}"
        f"\nBreadth edge: when >60% turnover-weight advances + institutional delivery, "
        f"the rally is broad-based — historically more durable than a 2-3 heavyweight lift."
    )

    return IndexSignal(name, "Sector", direction, round(base, 2), head, desc, emoji)


def _sig_constituent_fut_oi(
    stats: _ConstituentStats, fno_symbol: str,
) -> Optional[IndexSignal]:
    """
    Signal 17d: Constituent stock futures OI-Price matrix — NIFTY, BANKNIFTY, FINNIFTY.

    Stock futures (FUTSTK) carry no hedge ambiguity — every OI change is a directional bet.
    When a large fraction of constituent stocks simultaneously show Fresh Long Buildup,
    institutions are constructing BASKET LONGS via individual names.

    SCORING (net_balance = bull_pct − bear_pct):
      net ≥ 0.45 + fresh_long ≥ 30%  → +3.0  Exceptional basket conviction long
      net ≥ 0.30                      → +2.0  Broadly bullish
      net ≥ 0.15                      → +1.0  Mild bullish lean
      net ≤ −0.45 + fresh_short ≥ 30% → −3.0  Exceptional basket conviction short
      net ≤ −0.30                      → −2.0  Broadly bearish
      net ≤ −0.15                      → −1.0  Mild bearish lean
    """
    short_lbl, full_lbl = _CONSTITUENT_INFO.get(fno_symbol, ("IDX", "Index"))
    min_valid = max(5, stats.total_stocks * 4 // 10)
    valid = stats.fut_valid
    if valid < min_valid:
        return None

    fl = stats.fut_fresh_long
    sc = stats.fut_short_cov
    lu = stats.fut_long_unwind
    fs = stats.fut_fresh_short

    bull_pct = (fl + sc) / valid
    bear_pct = (fs + lu) / valid
    fl_pct   = fl / valid
    fs_pct   = fs / valid
    net      = bull_pct - bear_pct

    if net >= 0.45 and fl_pct >= 0.30:
        score = 3.0
        name  = f"{short_lbl} Basket Fresh Long Build — Institutional Conviction BUY"
        head  = (f"{fl}/{valid} {short_lbl} stocks Fresh Long + {sc} Short Cov "
                 f"= {bull_pct:.0%} bulls (net {net:+.0%})")
        emoji = "🔥"
    elif net >= 0.30:
        score = 2.0
        name  = f"{short_lbl} Basket Broadly Bullish — Stock Futures Net Long"
        head  = (f"{fl} fresh long + {sc} short cov vs {fs} fresh short + {lu} unwind "
                 f"({bull_pct:.0%} bulls, net {net:+.0%})")
        emoji = "🟢"
    elif net >= 0.15:
        score = 1.0
        name  = f"{short_lbl} Basket Mild Bullish — More Stock Longs than Shorts"
        head  = f"{short_lbl} stock futures: {bull_pct:.0%} bullish vs {bear_pct:.0%} bearish"
        emoji = "🟡"
    elif net <= -0.45 and fs_pct >= 0.30:
        score = -3.0
        name  = f"{short_lbl} Basket Fresh Short Build — Institutional Conviction SELL"
        head  = (f"{fs}/{valid} {short_lbl} stocks Fresh Short + {lu} Long Unwind "
                 f"= {bear_pct:.0%} bears (net {net:+.0%})")
        emoji = "💀"
    elif net <= -0.30:
        score = -2.0
        name  = f"{short_lbl} Basket Broadly Bearish — Stock Futures Net Short"
        head  = (f"{fs} fresh short + {lu} unwind vs {fl} fresh long + {sc} cov "
                 f"({bear_pct:.0%} bears, net {net:+.0%})")
        emoji = "🔴"
    elif net <= -0.15:
        score = -1.0
        name  = f"{short_lbl} Basket Mild Bearish — More Stock Shorts than Longs"
        head  = f"{short_lbl} stock futures: {bear_pct:.0%} bearish vs {bull_pct:.0%} bullish"
        emoji = "🟠"
    else:
        return None

    direction = 1 if score > 0 else -1
    desc = (
        f"OI-Price matrix applied to {valid} {full_lbl} constituent stocks with active futures:\n"
        f"  Fresh Long Build  : {fl:2d} ({fl_pct:.0%}) — price ↑ + OI ↑ → new money entering longs\n"
        f"  Short Covering    : {sc:2d} — price ↑ + OI ↓ → shorts forced to buy back\n"
        f"  Long Unwinding    : {lu:2d} — price ↓ + OI ↓ → longs choosing to exit\n"
        f"  Fresh Short Build : {fs:2d} ({fs_pct:.0%}) — price ↓ + OI ↑ → new money entering shorts\n"
        f"\n"
        f"STOCK futures have no hedge role (unlike FUTIDX). When {fl} {full_lbl} components "
        f"simultaneously show fresh long buildup, institutions are constructing basket longs "
        f"via individual names — directional conviction independent of index-level OI."
    )

    return IndexSignal(name, "Futures OI", direction, score, head, desc, emoji)


def _sig_constituent_opt_oi_prem(
    stats: _ConstituentStats, fno_symbol: str,
) -> Optional[IndexSignal]:
    """
    Signal 17e: Constituent stock options OI-Premium matrix — NIFTY, BANKNIFTY, FINNIFTY.

    OI-Premium matrix applied per constituent stock and aggregated to a basket view.
    Disambiguates PCR — identifies whether OI is driven by buyers or writers.

    SCORING (net_long_pct − net_short_pct):
      net ≥ 0.45  → +3.0  Exceptional bullish basket options config
      net ≥ 0.25  → +2.0  Broadly bullish
      net ≥ 0.12  → +1.0  Mild bullish lean
      net ≤ −0.45 → −3.0  Exceptional bearish
      net ≤ −0.25 → −2.0  Broadly bearish
      net ≤ −0.12 → −1.0  Mild bearish lean
    """
    short_lbl, full_lbl = _CONSTITUENT_INFO.get(fno_symbol, ("IDX", "Index"))
    min_valid = max(5, stats.total_stocks * 3 // 10)
    valid = stats.opt_valid
    if valid < min_valid:
        return None

    nl   = stats.opt_net_long
    ns   = stats.opt_net_short
    ostr = stats.opt_straddle
    orng = stats.opt_range

    nl_pct = nl / valid
    ns_pct = ns / valid
    net    = nl_pct - ns_pct
    straddle_dominant = ostr / valid >= 0.30

    if net >= 0.45:
        score = 3.0
        name  = f"{short_lbl} Basket Options: Exceptional Call Buying / Put Writing"
        head  = (f"{nl}/{valid} {short_lbl} stocks Net Long options config "
                 f"({nl_pct:.0%} bull vs {ns_pct:.0%} bear, net {net:+.0%})")
        emoji = "🔥"
    elif net >= 0.25:
        score = 2.0
        name  = f"{short_lbl} Basket Options: Broadly Bullish Configuration"
        head  = (f"{nl} net-long vs {ns} net-short {short_lbl} stocks in options "
                 f"({nl_pct:.0%} vs {ns_pct:.0%}, net {net:+.0%})")
        emoji = "🟢"
    elif net >= 0.12:
        score = 1.0
        name  = f"{short_lbl} Basket Options: Mild Bullish Lean"
        head  = f"{short_lbl} options basket: {nl_pct:.0%} bullish vs {ns_pct:.0%} bearish"
        emoji = "🟡"
    elif net <= -0.45:
        score = -3.0
        name  = f"{short_lbl} Basket Options: Exceptional Put Buying / Call Writing"
        head  = (f"{ns}/{valid} {short_lbl} stocks Net Short options config "
                 f"({ns_pct:.0%} bear vs {nl_pct:.0%} bull, net {net:+.0%})")
        emoji = "💀"
    elif net <= -0.25:
        score = -2.0
        name  = f"{short_lbl} Basket Options: Broadly Bearish Configuration"
        head  = (f"{ns} net-short vs {nl} net-long {short_lbl} stocks in options "
                 f"({ns_pct:.0%} vs {nl_pct:.0%}, net {net:+.0%})")
        emoji = "🔴"
    elif net <= -0.12:
        score = -1.0
        name  = f"{short_lbl} Basket Options: Mild Bearish Lean"
        head  = f"{short_lbl} options basket: {ns_pct:.0%} bearish vs {nl_pct:.0%} bullish"
        emoji = "🟠"
    else:
        return None

    straddle_note = ""
    if straddle_dominant:
        straddle_note = (
            f" Note: {ostr}/{valid} stocks show straddle config (both call + put buying) — "
            "high volatility expected regardless of direction."
        )

    direction = 1 if score > 0 else -1
    desc = (
        f"OI-Premium matrix applied to near-expiry options chains of {valid} {full_lbl} stocks:\n"
        f"  Net Long  (C.Buy/P.Write) : {nl:2d} ({nl_pct:.0%}) — both sides confirm bullish\n"
        f"  Net Short (C.Write/P.Buy) : {ns:2d} ({ns_pct:.0%}) — both sides confirm bearish\n"
        f"  Straddle  (C.Buy+P.Buy)   : {ostr:2d} — big move expected, no direction\n"
        f"  Range     (C.Write+P.Write): {orng:2d} — low vol / theta play\n"
        f"\n"
        f"OI-Premium matrix disambiguates PCR: put OI rising could mean put buying (bearish) "
        f"OR put writing (bullish) — opposite conclusions from the same raw number. "
        f"Checking whether premium moves WITH OI determines who is driving it."
        f"{straddle_note}"
    )

    return IndexSignal(name, "Options OI", direction, score, head, desc, emoji)


def _sig_valuation_pe(ctx: MarketContext) -> Optional[IndexSignal]:
    """
    Nifty PE ratio as background valuation context.
    Historical Nifty PE range: ~14-33; mean ~22; post-COVID normal ~18-25.
    Only extreme readings matter for short-term prediction.
    """
    pe = ctx.nifty_pe
    if pe is None: return None
    if pe > 28:
        return IndexSignal("Nifty Overvalued — High PE", "Market Context", -1, -0.5,
            f"Nifty PE {pe:.1f}x — Expensive",
            f"Nifty 50 PE at {pe:.1f}x = elevated (historical mean ~22). "
            "Expensive markets are more vulnerable to corrections on negative news. "
            "Background caution — does not override near-term signals.", "📊")
    if pe < 16:
        return IndexSignal("Nifty Undervalued — Low PE", "Market Context", 1, 0.5,
            f"Nifty PE {pe:.1f}x — Attractive Valuation",
            f"Nifty 50 PE at {pe:.1f}x = below historical mean. "
            "Cheap valuation provides a cushion on downside. "
            "Supports bullish thesis if other signals align.", "📊")
    return None   # fair value (16-28) = neutral, no signal


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTIONS — Multi-Expiry (Nifty 50 exclusive, signals 18-20)
# ═══════════════════════════════════════════════════════════════════════════════

def _sig_multi_expiry_pcr(
    weekly_pcr: Optional[float],
    monthly_pcr: Optional[float],
    weekly_exp: Optional[date],
    monthly_exp: Optional[date],
    trade_date: date,
) -> Optional[IndexSignal]:
    """
    Nifty 50 exclusive — cross-expiry PCR divergence.
    Nifty is unique: weekly OI reflects this-week sentiment; monthly OI reflects
    institutional positioning for the whole month. When the two diverge, the
    longer timeframe (monthly) typically resolves correctly.
    """
    if weekly_pcr is None or monthly_pcr is None: return None
    if weekly_exp is None or monthly_exp is None: return None

    w_dte = (weekly_exp  - trade_date).days
    m_dte = (monthly_exp - trade_date).days
    w_tag = f"W-PCR {weekly_pcr:.2f} ({w_dte}d)"
    m_tag = f"M-PCR {monthly_pcr:.2f} ({m_dte}d)"

    # Both extreme fear = broad capitulation → contrarian bullish
    if weekly_pcr > 1.2 and monthly_pcr > 1.2:
        return IndexSignal(
            "Cross-Expiry Fear — Broad Capitulation (Nifty)", "Options OI", 1, 2.0,
            f"Both W+M PCR extreme: {w_tag}  |  {m_tag}",
            f"Weekly AND monthly options both showing extreme put buying (PCR >{weekly_pcr:.2f} / "
            f"{monthly_pcr:.2f}). Institutional hedging is synchronized across timeframes — a rare "
            "event that marks a fear peak. Contrarian: when everyone is hedged, downside is "
            "already priced in.", "🔄",
        )

    # Both below 0.7 = broad complacency → dangerous
    if weekly_pcr < 0.70 and monthly_pcr < 0.70:
        return IndexSignal(
            "Cross-Expiry Complacency — Broad Risk (Nifty)", "Options OI", -1, -2.0,
            f"Both W+M PCR low: {w_tag}  |  {m_tag}",
            f"Weekly AND monthly PCR below 0.70 — no hedging at either timeframe. "
            f"Participants are buying calls and ignoring puts across horizons. "
            "When everyone is positioned long across expiries, who is left to buy? "
            "Double-complacency is a rare and dangerous setup preceding sharp corrections.", "⚠️",
        )

    # Near-term fear but medium-term calm = this week's move, not a trend
    if weekly_pcr > 1.15 and monthly_pcr < 0.90:
        return IndexSignal(
            "Near-Term Fear / Medium-Term Calm (Nifty)", "Options OI", 1, 1.5,
            f"Weekly hedging ({w_tag}) vs monthly calm ({m_tag})",
            f"Weekly PCR {weekly_pcr:.2f} = heavy put buying for this week's expiry. "
            f"Monthly PCR {monthly_pcr:.2f} = institutions are NOT extending hedges to next month. "
            "This divergence tells us the fear is short-horizon, not systemic — "
            "the pressure tends to clear once the weekly expiry passes.", "📉",
        )

    # Near-term unhedged but monthly is hedged = institutions see risk retail doesn't
    if weekly_pcr < 0.75 and monthly_pcr > 1.05:
        return IndexSignal(
            "Near-Term Complacency / Institutional Monthly Hedge (Nifty)", "Options OI", -1, -1.5,
            f"Weekly unhedged ({w_tag}) vs monthly cautious ({m_tag})",
            f"Weekly PCR {weekly_pcr:.2f} = retail buying calls for this expiry. "
            f"Monthly PCR {monthly_pcr:.2f} = institutions quietly buying monthly puts. "
            "Information asymmetry: retail plays this week; institutions hedge the bigger move. "
            "This is the most actionable divergence — follow the monthly, not the weekly.", "🐻",
        )

    return None


def _sig_max_pain_convergence(
    weekly_max_pain: Optional[float],
    monthly_max_pain: Optional[float],
    spot_close: Optional[float],
    dte_weekly: int,
    dte_monthly: int,
) -> Optional[IndexSignal]:
    """
    Nifty 50 exclusive — dual max pain gravity.
    When weekly and monthly max pain are close to each other and to spot,
    both sets of option writers are defending the same level — creating a
    doubly strong gravitational pin. When they diverge, post-expiry drift
    toward the monthly max pain is a high-probability directional trade.
    """
    if weekly_max_pain is None or monthly_max_pain is None or not spot_close: return None

    mp_gap_pct     = abs(weekly_max_pain - monthly_max_pain) / spot_close * 100
    weekly_gap_pct = (weekly_max_pain  - spot_close) / spot_close * 100
    monthly_gap_pct= (monthly_max_pain - spot_close) / spot_close * 100

    # Strong convergence: both max pains near each other AND near spot
    if mp_gap_pct < 0.6 and abs(weekly_gap_pct) < 1.2 and dte_weekly <= 5:
        return IndexSignal(
            "Dual Max Pain Convergence — Double Pin (Nifty)", "Options OI", 0, 0.0,
            f"W-MP {weekly_max_pain:.0f} ≈ M-MP {monthly_max_pain:.0f} (gap {mp_gap_pct:.2f}%)",
            f"Weekly max pain {weekly_max_pain:.0f} and monthly max pain {monthly_max_pain:.0f} "
            f"both converge near spot {spot_close:.0f} (gap {mp_gap_pct:.2f}%). "
            f"Two separate sets of option writers are defending the same level with {dte_weekly}d to weekly expiry. "
            "Doubly pinned: expect exceptionally tight range. Any directional move will face "
            "compounded pinning force from both expiry cycles.", "📌",
        )

    # Both max pains above spot + weekly DTE ≤ 3 → post-expiry drift upward
    if dte_weekly <= 3 and monthly_max_pain > spot_close and abs(monthly_gap_pct) > 0.8:
        return IndexSignal(
            "Post-Weekly Drift UP Toward Monthly MP (Nifty)", "Options OI", 1, 1.0,
            f"Monthly MP {monthly_max_pain:.0f} (+{monthly_gap_pct:.1f}%) above spot — post-expiry pull",
            f"Weekly expiry in {dte_weekly}d. After weekly pin resolves, monthly max pain "
            f"{monthly_max_pain:.0f} ({monthly_gap_pct:+.1f}%) creates a gravitational pull upward. "
            "Post-weekly expiry sessions historically drift toward the monthly max pain level. "
            f"Monthly expiry in {dte_monthly}d — the pull strengthens as monthly DTE shrinks.", "⬆️",
        )

    # Both max pains below spot + weekly DTE ≤ 3 → post-expiry drift downward
    if dte_weekly <= 3 and monthly_max_pain < spot_close and abs(monthly_gap_pct) > 0.8:
        return IndexSignal(
            "Post-Weekly Drift DOWN Toward Monthly MP (Nifty)", "Options OI", -1, -1.0,
            f"Monthly MP {monthly_max_pain:.0f} ({monthly_gap_pct:.1f}%) below spot — post-expiry pull",
            f"Weekly expiry in {dte_weekly}d. Monthly max pain {monthly_max_pain:.0f} "
            f"({monthly_gap_pct:+.1f}% below spot) will dominate after weekly pin dissolves. "
            "Post-weekly drift toward monthly max pain is a high-probability setup — "
            f"option writers shift attention to monthly strikes after weekly Tuesday expires.", "⬇️",
        )

    return None


def _sig_gamma_wall(
    weekly_call_oi: int, weekly_put_oi: int,
    monthly_call_oi: int, monthly_put_oi: int,
    gamma_ratio: Optional[float],
    dte_weekly: int,
) -> Optional[IndexSignal]:
    """
    Nifty 50 exclusive — weekly vs monthly OI dominance.
    gamma_ratio = weekly_OI / (weekly_OI + monthly_OI).
    High ratio near expiry = strong near-term pinning; low ratio = monthly
    expiry is the real battlefield, weekly gamma is thin.
    """
    if gamma_ratio is None: return None
    w_total = weekly_call_oi + weekly_put_oi
    m_total = monthly_call_oi + monthly_put_oi

    if gamma_ratio > 0.58 and dte_weekly <= 6:
        # Score 0.0: a gamma pin is a RANGE statement, not a bearish one. The old
        # -0.5 silently voted bearish in the composite every pre-expiry week.
        return IndexSignal(
            "Weekly Gamma Dominance — Range Compression (Nifty)", "Options OI", 0, 0.0,
            f"Weekly OI {gamma_ratio*100:.0f}% of total — {w_total:,} vs monthly {m_total:,}",
            f"Weekly expiry concentrates {gamma_ratio*100:.0f}% of total call+put OI "
            f"({w_total:,} weekly vs {m_total:,} monthly). "
            f"With {dte_weekly}d to weekly expiry, option writers have maximum financial incentive "
            "to keep spot pinned. At-the-money gamma is accelerating — each ±1% move forces "
            "delta hedging that works against the direction. Expect tight range, not a breakout.", "🎯",
        )

    if gamma_ratio < 0.28:
        return IndexSignal(
            "Monthly OI Dominates — Weekly Gamma Thin (Nifty)", "Options OI", 0, 0.0,
            f"Monthly OI {(1-gamma_ratio)*100:.0f}% of total — {m_total:,} vs weekly {w_total:,}",
            f"Monthly expiry controls {(1-gamma_ratio)*100:.0f}% of total OI ({m_total:,} contracts). "
            "Weekly gamma is thin — not enough near-term OI to pin the market. "
            "Directional moves are structurally possible this week. "
            "Monthly max pain and monthly put/call walls are the dominant support/resistance levels.", "📊",
        )

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL FUNCTIONS — Statistical Regime Detection (signals 21-23, all indices)
# ═══════════════════════════════════════════════════════════════════════════════

def _sig_market_memory(regime) -> Optional[IndexSignal]:
    """
    Signal 21: Hurst R/S + DFA — Market Memory / Persistence.

    H > 0.58 (trending): amplify current direction — history is repeating.
    H < 0.42 (mean-reverting): fade current direction — market self-corrects.
    H ≈ 0.50 (random walk): no memory-based edge; other signals dominate.

    Score is direction-sensitive (uses recent 5D return as directional anchor)
    so the signal fires WITH or AGAINST the current market move as appropriate.
    """
    if regime is None or regime.error or regime.data_points < 20:
        return None

    h     = regime.memory_avg
    score = regime.memory_score
    label = regime.memory_label

    if label == "Trending":
        direction = 1 if score > 0 else -1
        trend_word = "Bullish Persistence" if direction > 0 else "Bearish Persistence"
        return IndexSignal(
            f"Market Memory: Trending ({trend_word})", "Statistical Regime",
            direction, score,
            f"Hurst {h:.3f} — Autocorrelated Returns → Follow Momentum",
            regime.memory_note, "🌊",
        )

    if label == "Mean-Reverting":
        direction = 1 if score > 0 else -1
        return IndexSignal(
            "Market Memory: Mean-Reverting — Fade the Move", "Statistical Regime",
            direction, score,
            f"Hurst {h:.3f} — Anti-Persistent Returns → Fade Recent Move",
            regime.memory_note, "🔄",
        )

    return None   # Random walk — no memory-based directional signal


def _sig_hmm_regime(regime) -> Optional[IndexSignal]:
    """
    Signal 22: Hidden Markov Model — Latent Market Regime.

    3-state Gaussian HMM (Bull / Sideways / Bear) fitted on 90D returns.
    Score = p_bull × 3 − p_bear × 3 (probability-weighted, continuous).
    Only emits a signal when the leading-state probability ≥ 0.40.
    """
    if regime is None or regime.error or regime.hmm_state == "Unknown":
        return None
    if regime.hmm_prob < 0.40:
        return None

    state = regime.hmm_state
    score = regime.hmm_score
    prob  = regime.hmm_prob

    emoji_map = {"Bull": "📈", "Sideways": "↔️", "Bear": "📉"}
    dir_map   = {"Bull": 1,   "Sideways": 0,    "Bear": -1}
    emoji = emoji_map.get(state, "🤔")
    dirn  = dir_map.get(state, 0)

    if state == "Sideways" and abs(score) < 0.3:
        return None   # Sideways with no probability lean — not actionable

    return IndexSignal(
        f"HMM Regime: {state} State Detected", "Statistical Regime",
        dirn, score,
        f"HMM → {state} (p={prob:.0%}) | Score {score:+.2f}",
        regime.hmm_note, emoji,
    )


def _sig_entropy_regime(regime) -> Optional[IndexSignal]:
    """
    Signal 23: Permutation Entropy — Market Complexity / Predictability.

    Low PE (< 0.50): ordered return patterns — signal reliability elevated.
    High PE (> 0.72): chaotic, noisy returns — higher model uncertainty.
    Moderate PE: no signal (normal market conditions).

    Score is intentionally small (±0.8) — entropy is a confidence modifier,
    not a primary directional signal.
    """
    if regime is None or regime.error or regime.data_points < 15:
        return None

    pe    = regime.perm_entropy
    label = regime.entropy_label

    # Entropy is a CONFIDENCE modifier, not a directional vote. It is already
    # applied to confidence via regime.entropy_conf in _compute_verdict. Its
    # composite SCORE must be 0.0 — a non-zero score here silently votes a
    # direction on every chaotic/ordered day (the diagnostic found it firing
    # bearish 152/152 days, anchoring the whole engine bearish). Keep the signal
    # for UI/context display, but contribute nothing to the directional sum.
    if label == "Ordered":
        return IndexSignal(
            "Entropy: Low — Ordered / Predictable Market", "Statistical Regime",
            0, 0.0,
            f"PE={pe:.4f} — High Predictability (Ordered Patterns)",
            regime.entropy_note, "🎯",
        )
    if label == "Chaotic":
        return IndexSignal(
            "Entropy: High — Chaotic / Unpredictable Market", "Statistical Regime",
            0, 0.0,
            f"PE={pe:.4f} — Low Predictability (Disordered Patterns)",
            regime.entropy_note, "🌪️",
        )
    return None   # Moderate entropy — normal market, no additional signal


def _sig_memory_engine(mem) -> Optional[IndexSignal]:
    """
    Signal 24: Prediction Memory Engine — historical pattern calibration.

    Finds the top-20 most similar past market days (by 8-feature vector:
    PCR, FII net, VIX, carry, breadth, Hurst, entropy, OI score).
    Score = (up_pct − dn_pct) × 2.5 from those similar days' actual outcomes.

    Confirms or contradicts the current prediction based on historical hit rates.
    """
    if mem is None or mem.error or mem.similar_count < 10:
        return None

    up  = mem.memory_up_pct
    dn  = mem.memory_dn_pct
    sc  = mem.memory_score
    acc = mem.memory_accuracy
    avg = mem.avg_actual_return

    conf_str = "CONFIRMS" if mem.confirms_prediction is True else (
        "CONTRADICTS" if mem.confirms_prediction is False else "is UNCERTAIN about"
    )

    if sc >= 1.2:
        return IndexSignal(
            "Memory Engine: Bullish Pattern Match", "Memory",
            1, sc,
            f"Similar past days: UP {up:.0%}  DOWN {dn:.0%}  (avg +{avg:.2f}%)",
            f"{mem.memory_note}  Historical accuracy: {acc:.0%}.", "🧠",
        )
    if sc <= -1.2:
        return IndexSignal(
            "Memory Engine: Bearish Pattern Match", "Memory",
            -1, sc,
            f"Similar past days: DOWN {dn:.0%}  UP {up:.0%}  (avg {avg:.2f}%)",
            f"{mem.memory_note}  Historical accuracy: {acc:.0%}.", "🧠",
        )
    if abs(sc) >= 0.4:
        d = 1 if sc > 0 else -1
        label = "Mildly Bullish" if d > 0 else "Mildly Bearish"
        return IndexSignal(
            f"Memory Engine: {label} Pattern Match", "Memory",
            d, sc,
            f"Similar past days: UP {up:.0%}  DOWN {dn:.0%}  (avg {avg:+.2f}%)",
            f"{mem.memory_note}  Historical accuracy: {acc:.0%}.", "🧠",
        )
    # Neutral / contradicting — add a warning signal
    if mem.confirms_prediction is False:
        return IndexSignal(
            "Memory Engine: Contradicts Current Prediction", "Memory",
            0, -0.5,
            f"Memory {conf_str} — reduce position size",
            f"{mem.memory_note}  Historical accuracy: {acc:.0%}.", "⚠️",
        )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

# Seven "Institutional" signals (FII institutional / options-delta / flow /
# 5D-cumulative / OI-buildup / position-change / short-squeeze) all read the same
# underlying fact: FII index-futures positioning. FIIs are structurally net-short
# index futures as a hedge on their cash longs, so these signals are highly
# collinear and almost always fire together in the same direction. Summing them
# raw over-counts ONE fact up to 7× and anchors the engine permanently bearish
# (the per-signal diagnostic confirmed several firing 100% of days, all bearish).
# We cap the NET institutional contribution so it counts as one strong vote.
_INSTITUTIONAL_CAP = 3.0

# Per-family caps on the NET contribution of each signal family to the composite.
#
# WHY: signals within a family are highly collinear (they slice the SAME underlying
# fact), so an uncapped linear sum over-counts one fact N times and manufactures
# false "STRONG" conviction — the mechanism behind the saturated −20 bearish
# misfires (e.g. 53 "Options OI" signals all firing off one OI snapshot). Capping
# each family's net score bounds it to "one strong vote" regardless of member count,
# de-correlating the composite.
#
# CAP SIZE is scaled by the family's MEASURED next-day information coefficient
# (walk-forward over the live track record — see backtest_signals.py).
# Re-measured 2026-06-19 over 381 days × 4 indices, AFTER pruning the non-index-F&O
# families (VWAP/RSI/Sector/VIX/constituents) out of the vote:
#     Price Action +0.121 (66% hit, but rare — z-score extremes only) → most room
#     Options OI +0.020, Statistical Regime +0.015, Futures OI +0.014 → weak +: moderate
#     Institutional −0.009 (net bearish leak from raw FII position-change signals;
#         Pro-Desk members carry the edge, +0.142/+0.052)            → capped, see below
#     Carry −0.004, Memory −0.019, Sector −0.01                      → EXCLUDED (cap 0)
#     Market Context −0.13 (VIX wrong-sign as a vote)                → EXCLUDED (cap 0)
# NOTE: composite IC after the prune is +0.059 NIFTY / +0.002–0.022 others — positive
# but within ~1 SE of zero. Caps were NOT re-tuned off this single in-sample run
# (overfitting guard); only chronically wrong-sign signals were demoted to context.
_FAMILY_CAPS: dict[str, float] = {
    "Price Action":       7.0,
    "Institutional":      _INSTITUTIONAL_CAP,
    "Statistical Regime": 4.0,
    "Options OI":         5.0,
    "Futures OI":         4.0,
    # Sector / Carry / Market Context EXCLUDED (cap 0): each has ≤0 measured next-day
    # IC (Sector −0.01, Carry −0.04, Market Context −0.13) and is NOT pure index-F&O
    # data the engine is now scoped to. Any surviving member shows as context only.
    "Sector":             0.0,
    "Carry":              0.0,
    "Market Context":     0.0,
    "Memory":             0.0,   # excluded from composite — context only
}
_FAMILY_CAP_DEFAULT = 4.0


def _aggregate_composite(signals: list[IndexSignal]) -> float:
    """IC-weighted, family-capped composite directional score.

    Each signal family's NET score is clamped to a per-family cap (``_FAMILY_CAPS``)
    before summing, so collinear signals within a family count as one bounded vote
    instead of over-counting a single fact N times. Cap size reflects the family's
    measured next-day IC; the Memory family (cap 0) is excluded from the composite
    and surfaced as context only. See ``_FAMILY_CAPS`` for the rationale.
    """
    fam_net: dict[str, float] = {}
    for s in signals:
        fam_net[s.category] = fam_net.get(s.category, 0.0) + s.score
    total = 0.0
    for cat, net in fam_net.items():
        cap = _FAMILY_CAPS.get(cat, _FAMILY_CAP_DEFAULT)
        if cap <= 0:
            continue   # family excluded from the composite
        total += max(-cap, min(cap, net))
    return float(total)


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

# Per-index empirical calibration of the 1σ expected-move band. India VIX measures
# NIFTY IV, so the realized+VIX blend systematically UNDER-sizes the wider non-NIFTY
# indices (worst for midcaps). Audited on prediction_log via
# scripts/validate_range_coverage.py: the raw ±1σ band contains the next close ~68%
# pooled (NIFTY 72% · BANKNIFTY 68% · FINNIFTY 70% · MIDCPNIFTY 63%). These widen the
# band per index toward a consistent, honest ~1σ. Values are SHRUNK toward 1.0 and
# checked out-of-sample: a 75% target did NOT generalise (return tails are
# non-stationary), but the MIDCAP widening robustly did (+8.5pp held-out coverage).
_EM_CALIBRATION: dict[str, float] = {
    "NIFTY": 1.00, "BANKNIFTY": 1.06, "FINNIFTY": 1.08, "MIDCPNIFTY": 1.18,
}

# RiskMetrics EWMA decay for the realized-vol leg. λ=0.94 → α=0.06, center-of-mass ≈ 16
# sessions (≈32-day span). Reacts faster to vol-regime shifts than a flat 20-day window
# and avoids the "vol cliff" (a single jump inflating σ equally for 20 days, then dropping
# off the window edge). Coverage ≡ flat-20D in backtest, so _EM_CALIBRATION is unaffected.
_EWMA_LAMBDA = 0.94

# ── Single conviction axis: verdict ⇄ directional target ⇄ sideways band ─────────
# conviction = composite / _CONVICTION_DIVISOR (clamped ±1). The model is "directional"
# iff |conviction| ≥ _C_SIDE; below that the projected move sits inside the noise band,
# so there is NO point target and the verdict is SIDEWAYS. Because the verdict threshold,
# the sideways-band width, and the target's zero-point are ALL derived from this one
# boundary, the three can never contradict (the old design used 3/7/12 for the verdict,
# ÷20 for the target, and ×0.40 for the band — independent knobs that routinely disagreed,
# e.g. a green "UP" target of +75 pts printed inside a ±78 pt "range-bound" band).
_CONVICTION_DIVISOR = 20.0
_C_SIDE             = 0.40                                   # sideways/directional boundary (σ units)
_DIR_COMPOSITE_MIN  = _CONVICTION_DIVISOR * _C_SIDE          # = 8.0 composite to call a direction


def _compute_expected_move(
    idx_hist: pd.DataFrame,
    spot_close: Optional[float],
    vix_close: Optional[float],
    composite: float,
    dte: int,
    levels: IndexKeyLevels,
    is_nifty: bool,
    gamma_ratio: Optional[float] = None,
    cal_mult: float = 1.0,
) -> dict:
    """
    Quantify the MAGNITUDE of tomorrow's expected move — "how many points".

    Methodology (best-practice quant, blends forward + backward looking vol):
      1. Realized 1σ daily move — std of last 20 daily returns × spot.
         Index-specific (Bank Nifty naturally wider than Nifty). Always available.
      2. VIX-implied 1σ daily move — India VIX is annualised IV; daily = VIX/√252.
         Forward-looking (what the options market is pricing). India VIX measures
         Nifty IV directly, so it anchors Nifty; for others realized vol leads.
      3. Blend: Nifty → 50/50 realized+VIX; other indices → 70/30 (realized leads,
         VIX as a market-wide vol-regime nudge).
      4. Expiry compression: DTE ≤ 1 gamma pinning shrinks the range (×0.70).
      5. OI wall anchoring: when the weekly call wall or put wall falls INSIDE the
         statistical 1σ range, the range bound is compressed to the wall level.
         Option writers actively defend those strikes — the effective range is tighter
         than the unconditional vol estimate.

    The expected_move is the 1σ band (~68% of days close within ±expected_move).
    The directional target skews within that band by conviction = composite/20.

    Returns a dict of the IndexPrediction expected-move fields (or empties).
    """
    out = dict(
        expected_move_pts=None, expected_move_pct=None,
        range_low=None, range_high=None,
        target_move_pts=None, target_close=None,
        sideways_band_pts=None, move_basis="",
    )
    if not spot_close or spot_close <= 0 or idx_hist.empty:
        return out

    closes = idx_hist.sort_values("trade_date")["close_val"].dropna()
    if len(closes) < 10:
        return out

    rets = closes.pct_change().dropna()
    if len(rets) < 5:
        return out
    # Realized 1σ via RiskMetrics EWMA (λ=0.94, zero-mean): σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}.
    # Exponentially-weighted, so recent vol dominates and a single jump decays out smoothly
    # (no flat-window cliff). Backtested next-day coverage ≡ flat-20D std (≈73% pooled, CIs
    # overlap → _EM_CALIBRATION still valid). Falls back to flat std on short history.
    # See scripts/backtest_sigma_estimator.py.
    if len(rets) >= 20:
        ewma_var = (rets ** 2).ewm(alpha=1 - _EWMA_LAMBDA, adjust=False).mean()
        realized_sigma_pct = float(ewma_var.iloc[-1] ** 0.5) * 100.0   # daily 1σ, %
    else:
        realized_sigma_pct = float(rets.tail(20).std()) * 100.0        # daily 1σ, %

    vix_sigma_pct = None
    if vix_close and vix_close > 0:
        vix_sigma_pct = float(vix_close) / (252.0 ** 0.5)    # annualised → daily 1σ, %

    if vix_sigma_pct is not None:
        w_realized = 0.5 if is_nifty else 0.7
        sigma_pct  = w_realized * realized_sigma_pct + (1 - w_realized) * vix_sigma_pct
        basis_src  = (f"50/50 EWMA+VIX" if is_nifty else "70/30 EWMA+VIX")
    else:
        sigma_pct  = realized_sigma_pct
        basis_src  = "EWMA realized vol"

    # Per-index empirical calibration (see _EM_CALIBRATION). Applied to sigma_pct so it
    # flows consistently into range / sideways / target / breakout / weekly downstream.
    if cal_mult and cal_mult > 0:
        sigma_pct *= cal_mult

    expected_move_pts = spot_close * sigma_pct / 100.0

    # Expiry-day gamma pinning compresses the expected range
    pin_note = ""
    if dte <= 1:
        expected_move_pts *= 0.70
        sigma_pct          *= 0.70
        pin_note = " · expiry pin ×0.7"

    # Data-driven sideways band: the conviction boundary × the 1σ move (volatility-scaled
    # "±X pts = sideways, beyond = directional"). Same _C_SIDE that gates the verdict.
    sideways_band = _C_SIDE * expected_move_pts

    # Directional target on the SAME conviction axis as the band and the verdict, so the
    # three can't contradict. Below the boundary there is no point target (the projected
    # move is inside the noise band → range-bound). At/above it, the target interpolates
    # from the band edge (at the boundary) up to the 1σ edge (at full conviction), which
    # guarantees  sideways_band ≤ |target| ≤ expected_move  whenever a target is shown.
    conviction = max(-1.0, min(1.0, composite / _CONVICTION_DIVISOR))
    if abs(conviction) < _C_SIDE:
        target_move = 0.0                                   # sub-band → no directional target
    else:
        span = (abs(conviction) - _C_SIDE) / (1.0 - _C_SIDE)   # 0 at boundary → 1 at full conv.
        mag  = sideways_band + span * (expected_move_pts - sideways_band)
        target_move = mag if conviction > 0 else -mag
    target_close = spot_close + target_move

    # ── OI wall context (informational — does NOT change the statistical range) ──
    # Call/put walls show where option writers are concentrated and delta-hedging
    # creates friction.  They are RESISTANCE / SUPPORT zones, not probability
    # ceilings or floors.  In a gamma-squeeze or strong trend the wall is blown
    # through — capping range_high at the call wall would understate upside on
    # exactly those days.  Show the wall positions as context so the trader knows
    # where friction sits WITHIN the 1σ band, but keep the 68% claim intact.
    range_low_stat  = spot_close - expected_move_pts
    range_high_stat = spot_close + expected_move_pts
    wall_notes: list[str] = []

    call_wall = levels.top_call_strike if levels else None
    put_wall  = levels.top_put_strike  if levels else None

    if call_wall and call_wall > spot_close:
        loc = "inside range — resistance" if call_wall < range_high_stat else "beyond range"
        wall_notes.append(f"call wall {call_wall:.0f} ({loc})")

    if put_wall and put_wall < spot_close:
        loc = "inside range — support" if put_wall > range_low_stat else "beyond range"
        wall_notes.append(f"put wall {put_wall:.0f} ({loc})")

    # Weekly OI concentration note
    if gamma_ratio is not None and gamma_ratio > 0.58 and 2 <= dte <= 6:
        wall_notes.append(f"weekly OI {gamma_ratio*100:.0f}% of total — near-expiry pin zone")

    wall_suffix = (" · " + " · ".join(wall_notes)) if wall_notes else ""

    out.update(
        expected_move_pts=round(expected_move_pts, 0),
        expected_move_pct=round(sigma_pct, 2),
        range_low=round(range_low_stat, 0),
        range_high=round(range_high_stat, 0),
        target_move_pts=round(target_move, 0),
        target_close=round(target_close, 0),
        sideways_band_pts=round(sideways_band, 0),
        move_basis=(
            f"1σ ≈ {sigma_pct:.2f}% ({basis_src}{pin_note}); "
            f"≈68% of days close within ±{expected_move_pts:.0f} pts "
            f"(empirical 1σ band, per-index vol-calibrated)"
            f"{wall_suffix}"
        ),
    )
    return out


# Breakout confirmation buffer as a FRACTION OF THE 1σ EXPECTED MOVE (volatility-
# based, not a fixed % of price). 0.40σ = the same noise zone as the sideways band,
# so a "new range" needs a vol-scaled break beyond the 1σ range — auto-scaling with
# the VIX/realized-vol regime (wide buffer when vol is high, tight when calm).
_BREAKOUT_SIGMA_FRAC = 0.40


def _compute_breakout_extensions(
    spot_close: float,
    expected_move_pts: float,
    range_low: float,
    range_high: float,
    levels: IndexKeyLevels,
    composite: float,
    fii_net: Optional[int] = None,
    pcr: Optional[float] = None,
    carry_pct_ann: Optional[float] = None,
    dte: int = 30,
    volume_ratio: Optional[float] = None,
) -> dict:
    """
    Compute the SECOND range — where the index heads if it breaks the first range.

    Logic (asymmetric quant framework):
      Confirmation buffer = 0.40 × the 1σ expected move (VOLATILITY-based, not a
      fixed % of price) AND the trigger lifts to clear the nearest OI wall just
      beyond the range. So a "new range" requires a vol-scaled, structure-aware
      break — wide in a high-VIX regime, tight when calm.

      UPSIDE EXTENSION (base = spot + 2σ):
        FII short squeeze amplification — when FII is massively net short, covering
        cascades create a CONVEX payoff: each uptick forces more covering, pushing
        further than statistical vol predicts.
          fii_net < -200k → ×1.20 (extend to 2.4σ — large squeeze risk)
          fii_net < -100k → ×1.12 (extend to 2.24σ — moderate squeeze)

      DOWNSIDE EXTENSION (base = spot - 2σ):
        PCR put-writing floor — when PCR is elevated, put writers (short puts)
        delta-hedge by BUYING futures as price falls, creating a real floor.
          pcr > 1.30 → ×0.80 (compress to 1.6σ — strong put-writing floor)
          pcr > 1.00 → ×0.90 (compress to 1.8σ — mild floor)
        Backwardation override: futures stress deepens on downside breaks;
          carry_pct_ann < 0 partially offsets PCR floor compression.

      OI WALL OVERRIDE: if a call/put wall sits inside the computed extension zone,
        it replaces the statistical cap as the structural target.
    """
    if expected_move_pts <= 0:
        return {}
    # Volatility-based confirmation buffer (fraction of 1σ) — scales with the regime.
    buffer         = round(_BREAKOUT_SIGMA_FRAC * expected_move_pts, 0)
    two_sigma      = expected_move_pts * 2.0

    call_wall = levels.top_call_strike if levels else None
    put_wall  = levels.top_put_strike  if levels else None

    up_trigger     = range_high + buffer
    dn_trigger     = range_low  - buffer
    # OI-wall confirmation: a genuine breakout must ALSO clear the nearest option
    # wall sitting just beyond the range (the structural OI resistance/support). If
    # such a wall is within ~one buffer of the statistical trigger, lift the trigger
    # to the wall — price hasn't really "broken out" until that OI barrier is taken.
    wall_note_up = wall_note_dn = ""
    if call_wall and range_high < call_wall <= up_trigger + buffer:
        if call_wall > up_trigger:
            wall_note_up = f" (clears call wall {call_wall:.0f})"
        up_trigger = max(up_trigger, call_wall)
    if put_wall and dn_trigger - buffer <= put_wall < range_low:
        if put_wall < dn_trigger:
            wall_note_dn = f" (clears put wall {put_wall:.0f})"
        dn_trigger = min(dn_trigger, put_wall)
    threshold = buffer   # reported as the minimum break-confirmation distance

    # ── Upside extension: FII short squeeze amplification ────────────────────
    up_sigma_mult = 1.0
    up_adj_note   = ""
    if fii_net is not None:
        if fii_net < -200_000:
            up_sigma_mult = 1.20
            up_adj_note   = f"FII {fii_net:,} short → squeeze extends to 2.4σ"
        elif fii_net < -100_000:
            up_sigma_mult = 1.12
            up_adj_note   = f"FII {fii_net:,} short → squeeze risk 2.2σ"

    two_sigma_high = spot_close + two_sigma * up_sigma_mult

    # ── Downside extension: PCR put-writing floor + backwardation offset ─────
    dn_sigma_mult = 1.0
    dn_adj_note   = ""
    if pcr is not None:
        if pcr > 1.30:
            dn_sigma_mult = 0.80
            dn_adj_note   = f"PCR {pcr:.2f} put-writing → floor at 1.6σ"
        elif pcr > 1.00:
            dn_sigma_mult = 0.90
            dn_adj_note   = f"PCR {pcr:.2f} put-writing → floor at 1.8σ"
    # Backwardation adds institutional selling pressure — partially offsets PCR floor
    if carry_pct_ann is not None and carry_pct_ann < 0:
        dn_sigma_mult = min(dn_sigma_mult + 0.07, 1.0)   # loosen floor by 7% of 2σ
        bkwd_note = f"backwardation {carry_pct_ann:.1f}% adds sell pressure"
        dn_adj_note = (dn_adj_note + " · " + bkwd_note) if dn_adj_note else bkwd_note

    two_sigma_low = spot_close - two_sigma * dn_sigma_mult

    # ── OI wall override: structural level beats statistical estimate ─────────
    call_walls_in_zone = sorted([
        w for w in [levels.top_call_strike, levels.second_call_strike]
        if w and up_trigger < w <= two_sigma_high
    ])
    if call_walls_in_zone:
        up_end = call_walls_in_zone[0]
        up_src = f"call wall {up_end:.0f}"
        if up_adj_note:
            up_src += f" · {up_adj_note}"
    else:
        up_end = two_sigma_high
        sigma_label = f"{up_sigma_mult * 2:.1f}σ"
        up_src = f"{sigma_label} statistical" + (f" · {up_adj_note}" if up_adj_note else "")

    put_walls_in_zone = sorted([
        w for w in [levels.top_put_strike, levels.second_put_strike]
        if w and two_sigma_low <= w < dn_trigger
    ], reverse=True)
    if put_walls_in_zone:
        dn_start = put_walls_in_zone[0]
        dn_src = f"put wall {dn_start:.0f}"
        if dn_adj_note:
            dn_src += f" · {dn_adj_note}"
    else:
        dn_start = two_sigma_low
        sigma_label = f"{dn_sigma_mult * 2:.1f}σ"
        dn_src = f"{sigma_label} statistical" + (f" · {dn_adj_note}" if dn_adj_note else "")

    # ── Max pain gravity note (DTE ≤ 7: gravity is meaningful) ───────────────
    mp = levels.max_pain if levels else None
    if mp and dte <= 7:
        mp_gap_up = mp - up_trigger
        mp_gap_dn = dn_trigger - mp
        if mp_gap_up < 0:   # max pain BELOW the upside breakout trigger
            up_src += f" · max pain {mp:.0f} ({abs(mp_gap_up):.0f} pts below break → gravity)"
        if mp_gap_dn < 0:   # max pain ABOVE the downside breakout trigger
            dn_src += f" · max pain {mp:.0f} ({abs(mp_gap_dn):.0f} pts above break → gravity)"

    # ── Volume confirmation (validated +6pp continuation edge) ───────────────
    vol_note = ""
    if volume_ratio is not None:
        if volume_ratio >= 1.15:
            vol_note = f" · vol {volume_ratio:.1f}× avg → expansion confirms a break (~58% follow-through)"
        elif volume_ratio <= 0.85:
            vol_note = f" · vol {volume_ratio:.1f}× avg → thin; breaks prone to fade back into range"

    return dict(
        breakout_threshold_pts=round(threshold, 0),
        breakout_up_start=round(up_trigger, 0),
        breakout_up_end=round(up_end, 0),
        breakout_dn_start=round(dn_start, 0),
        breakout_dn_end=round(dn_trigger, 0),
        breakout_up_src=up_src + wall_note_up + vol_note,
        breakout_dn_src=dn_src + wall_note_dn + vol_note,
    )


def _compute_weekly_range(
    fno_today: pd.DataFrame,
    spot_close: Optional[float],
    sigma_daily_pct: Optional[float],
    trade_date: date,
    is_nifty: bool,
    target_expiry: Optional[date] = None,
    prefix: str = "wk",
) -> dict:
    """
    Forward to-expiry range — where price can travel until a given option expiry,
    built PURELY from that expiry's own option chain (ATM straddle + OI walls + max pain).

    Default (``target_expiry=None``): targets the nearest expiry STRICTLY AFTER today,
    so on an expiry day it automatically ROLLS to the next cycle (Nifty → next Tuesday
    weekly; Bank/Fin/Midcap → next monthly, their only series). Recomputed each EOD —
    the range shrinks as expiry nears and resets after it. Output keys: ``wk_*``.

    Pass ``target_expiry`` (with ``prefix="mn"``) to build a SEPARATE range to a later
    expiry off ITS chain — e.g. the NIFTY monthly range from monthly-expiry OI, distinct
    from the weekly. Returns empty if that expiry isn't listed today.

    Width = blend of the ATM STRADDLE premium (the market's implied move to expiry,
    forward-looking — the institutional measure) and the statistical σ·√T move,
    bounded by the OI walls (call wall = resistance cap, put wall = support floor)
    and annotated with max pain (the pin/gravity into expiry). Point-in-time.
    """
    out: dict = {}
    if not spot_close or spot_close <= 0 or fno_today is None or fno_today.empty:
        return out
    opt = fno_today[fno_today["instrument"] == "OPTIDX"]
    if opt.empty:
        return out
    fut_exp = sorted({e for e in opt["expiry_date"].tolist() if e and e > trade_date})
    if not fut_exp:
        return out
    if target_expiry is not None:
        if target_expiry not in fut_exp:
            return out   # requested expiry not listed today → no range
        target = target_expiry
    else:
        target = fut_exp[0]
    chain = opt[opt["expiry_date"] == target]
    if chain.empty:
        return out

    days_cal = (target - trade_date).days
    days_trd = max(1, round(days_cal * 5.0 / 7.0))
    # Monthly = the last listed expiry within its calendar month — holiday-proof,
    # unlike the "last Tuesday" calendar rule (expiry shifts earlier on holidays).
    # Relative to TARGET (not the nearest), so an explicit monthly target_expiry is
    # labelled "monthly" even though earlier weeklies exist in the same month.
    _later_same_month = any(
        e.year == target.year and e.month == target.month and e > target
        for e in fut_exp
    )
    kind = "monthly" if (not is_nifty or not _later_same_month) else "weekly"

    # ATM straddle = market-implied move to expiry
    strikes  = [float(k) for k in chain["strike_price"].unique()]
    atm      = min(strikes, key=lambda k: abs(k - spot_close)) if strikes else None
    straddle = None
    if atm is not None:
        ce = chain[(chain["strike_price"] == atm) & (chain["option_type"] == "CE")]["close_price"]
        pe = chain[(chain["strike_price"] == atm) & (chain["option_type"] == "PE")]["close_price"]
        if not ce.empty and not pe.empty:
            s = float(ce.iloc[0]) + float(pe.iloc[0])
            straddle = s if s > 0 else None

    stat_move = (spot_close * (sigma_daily_pct or 0) / 100.0) * (days_trd ** 0.5)
    if straddle:
        weekly_move = 0.6 * straddle + 0.4 * stat_move
        basis = f"0.6×ATM straddle {straddle:.0f} + 0.4×σ√{days_trd}d {stat_move:.0f}"
    elif stat_move > 0:
        weekly_move = stat_move
        basis = f"σ√{days_trd}d statistical {stat_move:.0f} (no ATM straddle)"
    else:
        return out

    levels    = _compute_key_levels(chain, spot_close)
    call_cap  = (levels.top_call_strike if levels.top_call_strike and levels.top_call_strike > spot_close else None)
    put_floor = (levels.top_put_strike  if levels.top_put_strike  and levels.top_put_strike  < spot_close else None)
    range_low, range_high = spot_close - weekly_move, spot_close + weekly_move

    notes = []
    if call_cap:
        notes.append(f"call wall {call_cap:.0f} ({'caps range' if call_cap <= range_high else 'beyond range'})")
    if put_floor:
        notes.append(f"put wall {put_floor:.0f} ({'floors range' if put_floor >= range_low else 'beyond range'})")
    if levels.max_pain:
        notes.append(f"max pain {levels.max_pain:.0f}")

    out.update({
        f"{prefix}_expiry": target, f"{prefix}_dte_cal": days_cal, f"{prefix}_kind": kind,
        f"{prefix}_move_pts": round(weekly_move, 0),
        f"{prefix}_range_low": round(range_low, 0), f"{prefix}_range_high": round(range_high, 0),
        f"{prefix}_straddle_pts": round(straddle, 0) if straddle else None,
        f"{prefix}_put_floor": put_floor, f"{prefix}_call_cap": call_cap,
        f"{prefix}_max_pain": levels.max_pain,
        f"{prefix}_basis": f"to {target:%d %b} · {days_cal}d {kind} · {basis}"
                 + ((" · " + " · ".join(notes)) if notes else ""),
    })
    return out


def _compute_verdict(
    composite: float,
    signals: list[IndexSignal],
    pcr: Optional[float],
    carry_pct_ann: Optional[float],
    dte: int,
    spot_close: Optional[float],
    levels: IndexKeyLevels,
    ctx: MarketContext,
    entropy_conf: float = 1.0,
) -> tuple[str, str, str, str, str, str]:
    primary        = max(signals, key=lambda s: abs(s.score)) if signals else None
    primary_driver = primary.headline if primary else f"Composite {composite:+.1f}"

    oversold        = any(s.category == "Price Action" and s.direction == 1 and "Oversold" in s.name for s in signals)
    squeeze_setup   = any("Short Squeeze" in s.name for s in signals)
    # Require a live vote (score != 0): "FII Covering Shorts" is now context-only (demoted
    # as wrong-sign, IC −0.168), so it must NOT drive the "upside wildcard" verdict narrative.
    fii_covering    = any("Covering" in s.name and s.category == "Institutional"
                          and s.direction == 1 and s.score != 0 for s in signals)
    # Name-matched: family+direction alone also caught "VIX Fear Zone" (a contrarian
    # bullish read fired while VIX is SPIKING) and the PE-undervalued signal.
    vix_falling     = any("VIX Falling" in s.name for s in signals)
    fii_bearish     = any(s.category == "Institutional" and s.direction < 0 and abs(s.score) >= 2 for s in signals)
    broad_rally     = any("Broad-Based Rally" in s.name for s in signals)

    # ── Expiry day: gamma pin dominates ──────────────────────────────────────
    if dte <= 1:
        if spot_close and levels.max_pain:
            gap = (levels.max_pain - spot_close) / spot_close * 100
            if gap > 0.15:
                return ("UP", "LOW", "#69F0AE",
                        f"Expiry day — max pain {levels.max_pain:.0f} above spot ({gap:.1f}% pull)",
                        f"Max pain {levels.max_pain:.0f}; gamma pin pulls up",
                        "Gamma pin breaks on macro gap — low conviction.")
            if gap < -0.15:
                return ("DOWN", "LOW", "#FF6D00",
                        f"Expiry day — max pain {levels.max_pain:.0f} below spot ({abs(gap):.1f}% pull)",
                        f"Max pain {levels.max_pain:.0f}; gamma pin pulls down",
                        "Gamma pin breaks on macro gap — low conviction.")
        return ("SIDEWAYS", "MEDIUM", "#FFD600",
                "Expiry day — gamma pinning; expect narrow range near current levels",
                "DTE ≤ 1 — theta decay pins market near max pain",
                "Gap on global news overrides the pin.")

    # ── Entropy confidence gate ───────────────────────────────────────────────
    # When market is chaotic (PE > 0.72, conf_mult = 0.72), cap final confidence
    # at MEDIUM — we can detect direction but cannot claim HIGH certainty in a
    # disordered regime. Ordered markets (conf_mult > 1.0) allow normal HIGH.
    _entropy_chaotic  = entropy_conf < 0.85
    _entropy_ordered  = entropy_conf > 1.10

    def _cap_confidence(conf: str) -> str:
        if _entropy_chaotic and conf == "HIGH":
            return "MEDIUM"   # Entropy too noisy for high-confidence call
        return conf

    # ── Score-driven verdict — directional ONLY above the conviction band ─────
    # A direction is called iff |composite| ≥ _DIR_COMPOSITE_MIN (⇔ conviction ≥ _C_SIDE),
    # the SAME boundary that sets the sideways band and the directional target's zero-point.
    # Below it the projected move lives inside the noise band, so the verdict is SIDEWAYS
    # and any lean is shown only as the likely BREAK side / wildcard — never a green arrow
    # inside a "range-bound" band. HIGH keeps its ~6-aligned-signal threshold (±12).
    if composite >= 12:
        risk = "FII short position remains key headwind." if fii_bearish else "Global macro reversal could override."
        return ("UP", _cap_confidence("HIGH"), "#00C853",
                f"Strong multi-signal bullish alignment (score {composite:+.1f})",
                primary_driver, risk)
    if composite >= _DIR_COMPOSITE_MIN:
        return ("UP", _cap_confidence("MEDIUM"), "#69F0AE",
                f"Moderate bullish tilt — bias is UP (score {composite:+.1f})",
                primary_driver,
                "Watch FII position for sustained follow-through." if fii_bearish else
                "Watch carry and PCR for confirmation.")
    if composite <= -12:
        return ("DOWN", _cap_confidence("HIGH"), "#FF5252",
                f"Strong multi-signal bearish alignment (score {composite:+.1f})",
                primary_driver, "Check oversold bounce / DII support floor before shorting.")
    if composite <= -_DIR_COMPOSITE_MIN:
        return ("DOWN", _cap_confidence("MEDIUM"), "#FF6D00",
                f"Moderate bearish tilt — bias is DOWN (score {composite:+.1f})",
                primary_driver, "Mean-reverting market; deep scores can reverse sharply.")

    # ── Sub-band (|composite| < boundary) → RANGE-BOUND. Mild leans and the
    #    squeeze/oversold/covering setups are wildcards = the likely break side if the
    #    range fails, not a directional base case. Keeps the chip consistent with a
    #    target that is, by construction, inside the sideways band. ──────────────────
    if squeeze_setup and broad_rally:
        return ("SIDEWAYS", "LOW", "#FFD600",
                "Range-bound — short squeeze + broad rally is the upside wildcard",
                "Short squeeze setup: extreme PCR + HOD close + FII trapped",
                "Break side is up if the range fails — FII can add shorts; not a base case.")
    if oversold and vix_falling:
        return ("SIDEWAYS", "LOW", "#FFD600",
                "Range-bound — oversold + falling VIX; mean-reversion bounce is the upside wildcard",
                "Oversold z-score + falling VIX (mean-reversion bias)",
                "Break side is up if the range fails — oversold can persist.")
    if oversold:
        return ("SIDEWAYS", "LOW", "#FFD600",
                "Range-bound — oversold mean-reversion bounce is the upside wildcard",
                "Oversold 3D z-score (mean-reversion bias)",
                "Break side is up if the range fails — oversold can persist.")
    if fii_covering:
        return ("SIDEWAYS", "LOW", "#FFD600",
                "Range-bound — FII short covering is the key wildcard",
                "FII covering shorts creates buying pressure",
                "Break side is up if FII keep covering — monitor daily FAO data.")
    if composite >= 3:
        squeeze_note = " Short squeeze is an upside wildcard." if squeeze_setup else ""
        return ("SIDEWAYS", "LOW", "#FFD600",
                f"Range-bound — mild bullish lean below the directional band (score {composite:+.1f}){squeeze_note}",
                primary_driver, "Sub-band conviction — trade the range; upside is the likely break side.")
    if composite <= -3:
        return ("SIDEWAYS", "LOW", "#FFD600",
                f"Range-bound — mild bearish lean below the directional band (score {composite:+.1f})",
                primary_driver, "Sub-band conviction — trade the range; downside is the likely break side.")
    return ("SIDEWAYS", "LOW", "#78909C",
            f"No clear directional edge — range-bound likely (score {composite:+.1f})",
            "Composite score in neutral zone — signals mixed",
            "Large OI build in either direction breaks the range.")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_prediction(
    trade_date: date,
    fno_symbol: str,
    display_name: str,
    index_name: str,
    market_ctx: MarketContext,
    persist: bool = True,
) -> IndexPrediction:
    _no_data = dict(
        fno_symbol=fno_symbol, display_name=display_name, as_of_date=trade_date,
        spot_close=None, prev_close=None, day_change_pct=None, high=None, low=None,
        near_expiry=None, days_to_expiry=99, futures_price=None,
        carry_pts=None, carry_pct_ann=None, carry_label="No Data",
        fut_oi=0, fut_oi_chg=0, call_oi=0, put_oi=0, pcr=None,
        today=None, yesterday=None, levels=IndexKeyLevels(),
        direction="SIDEWAYS", confidence="LOW", direction_color="#78909C",
        composite_score=0.0, headline="No F&O data available",
        key_driver="F&O bhavcopy not fetched yet",
        key_risk="Run python -m src.cli daily to fetch today's data",
        data_available=False, note="F&O data not available.",
        market_context=market_ctx,
    )

    idx_hist = _load_index_history(index_name, trade_date)
    today_fno_date, prev_fno_date = _get_two_fno_dates(fno_symbol, trade_date)
    fno_today = _load_fno_for_date(fno_symbol, today_fno_date) if today_fno_date else pd.DataFrame()
    fno_prev  = _load_fno_for_date(fno_symbol, prev_fno_date)  if prev_fno_date  else pd.DataFrame()

    if fno_today.empty:
        return IndexPrediction(**_no_data)

    # ── Spot price ────────────────────────────────────────────────────────────
    spot_close = prev_close_price = day_change_pct = high_val = low_val = None
    if not idx_hist.empty:
        today_row = idx_hist[idx_hist["trade_date"] == trade_date]
        if today_row.empty: today_row = idx_hist.tail(1)
        if not today_row.empty:
            r = today_row.iloc[0]
            spot_close       = float(r["close_val"])  if pd.notna(r.get("close_val"))  else None
            prev_close_price = float(r["prev_close"]) if pd.notna(r.get("prev_close")) else None
            day_change_pct   = float(r["pct_chg"])    if pd.notna(r.get("pct_chg"))    else None
            high_val         = float(r["high_val"])   if pd.notna(r.get("high_val"))   else None
            low_val          = float(r["low_val"])    if pd.notna(r.get("low_val"))    else None

    # ── Active futures (OI > 0, not yet expired) ─────────────────────────────
    # Filter: expiry_date > trade_date — this is the critical "next-day forecast" gate.
    # The prediction is computed from EOD data on trade_date and is USED on trade_date+1.
    # Including trade_date's expiring contract would make dte=0, firing "Expiry day —
    # gamma pinning" for a day where expiry has already resolved.  By excluding contracts
    # that expire on or before trade_date, the prediction always reflects the NEXT active
    # cycle — exactly what a trader needs for tomorrow.
    # Side-effect: expiry-day verdict (dte≤1) now fires only on DTE=1 (day before expiry),
    # which correctly predicts "tomorrow will have gamma pinning."
    near_expiry = fut_expiry = None
    days_to_expiry = 30
    futures_price = carry_pts = carry_pct_ann = None
    carry_label = "No Data"
    fut_oi = fut_oi_chg = 0
    fut_oi_total = fut_oi_total_chg = 0   # rollover-immune basis for the OI-price matrix
    roll_share: Optional[float] = None    # next+far OI as a share of total (rollover gauge)
    total_oi_comparable = False           # is a prior-day total available to diff against?

    fut_rows = fno_today[
        (fno_today["instrument"] == "FUTIDX") &
        (fno_today["open_interest"] > 0) &
        (fno_today["expiry_date"] > trade_date)     # exclude expired / expiring-today
    ].sort_values("expiry_date")
    if not fut_rows.empty:
        nr = fut_rows.iloc[0]
        fut_expiry     = _to_date(nr["expiry_date"])
        days_to_expiry = max((fut_expiry - trade_date).days, 0) if fut_expiry else 30
        futures_price  = float(nr["settle_price"]) if pd.notna(nr["settle_price"]) else None
        fut_oi         = int(nr["open_interest"])    # near month (shown in the snapshot row)
        # Near-month OI change, matched by SAME expiry_date — kept for the snapshot field.
        if not fno_prev.empty:
            prev_futs = fno_prev[
                (fno_prev["instrument"] == "FUTIDX") &
                (fno_prev["expiry_date"] == nr["expiry_date"])
            ]
            if not prev_futs.empty:
                fut_oi_chg = fut_oi - int(prev_futs["open_interest"].sum())

        # ── Total-OI (rollover-immune) basis for the OI-Price matrix ──────────────
        # Near-month-only OI falsely reads "Long Unwinding / Short Covering" during
        # rollover week: positions MIGRATE to the next month rather than close, so the
        # near contract's OI can fall ~20%/day while TOTAL futures OI is flat. Summing
        # all active forward expiries (same >trade_date gate on today AND prev = an
        # apples-to-apples set) keeps OI continuous through the roll, so the matrix
        # reflects genuine net positioning. roll_share surfaces how far the roll has gone.
        fut_oi_total = int(fut_rows["open_interest"].sum())
        if fut_oi_total > 0:
            roll_share = (fut_oi_total - fut_oi) / fut_oi_total   # OI beyond the near month
        if not fno_prev.empty:
            prev_active = fno_prev[
                (fno_prev["instrument"] == "FUTIDX") &
                (fno_prev["open_interest"] > 0) &
                (fno_prev["expiry_date"] > trade_date)            # same forward gate as today
            ]
            if not prev_active.empty:
                fut_oi_total_chg    = fut_oi_total - int(prev_active["open_interest"].sum())
                total_oi_comparable = True

        if spot_close and futures_price and days_to_expiry >= 3:
            carry_pts     = futures_price - spot_close
            carry_pct_ann = (carry_pts / spot_close) * (365.0 / max(days_to_expiry, 1)) * 100
            carry_label   = _carry_label(carry_pct_ann)
        elif futures_price and not spot_close:
            spot_close = futures_price

    # ── Options near-expiry (weekly Tuesday for NIFTY; monthly last-Tuesday for others) ─
    # Same expiry_date > trade_date guard as fut_rows — use only future contracts.
    opt_rows = fno_today[
        (fno_today["instrument"] == "OPTIDX") &
        (fno_today["expiry_date"] > trade_date)     # exclude expired / expiring-today
    ].sort_values("expiry_date")
    near_expiry = _to_date(opt_rows.iloc[0]["expiry_date"]) if not opt_rows.empty else fut_expiry

    call_oi = put_oi = 0; pcr = None
    if near_expiry:
        opt = fno_today[(fno_today["instrument"] == "OPTIDX") & (fno_today["expiry_date"] == near_expiry)]
        call_oi = int(opt[opt["option_type"] == "CE"]["open_interest"].sum())
        put_oi  = int(opt[opt["option_type"] == "PE"]["open_interest"].sum())
        pcr     = round(put_oi / call_oi, 2) if call_oi > 0 else None

    dte_options = (near_expiry - trade_date).days if near_expiry else 30

    levels = IndexKeyLevels()
    if near_expiry and spot_close:
        levels = _compute_key_levels(
            fno_today[(fno_today["instrument"] == "OPTIDX") & (fno_today["expiry_date"] == near_expiry)],
            spot_close,
        )

    # ── Multi-expiry structure (Nifty 50: weekly + monthly bifurcation) ─────────
    monthly_exp_me: Optional[date] = None
    monthly_call_oi_me = monthly_put_oi_me = 0
    monthly_pcr_val: Optional[float] = None
    monthly_max_pain_lvl: Optional[float] = None
    gamma_ratio_val: Optional[float] = None
    near_is_monthly = False   # True for ~7 days before the last Tuesday (once near_expiry = last in month)

    if fno_symbol == "NIFTY" and not opt_rows.empty:
        all_exp = sorted(_to_date(e) for e in opt_rows["expiry_date"].unique() if e is not None)
        # Identify monthly expiry: last expiry in each calendar month
        month_last: dict = {}
        for e in all_exp:
            ym = (e.year, e.month)
            if ym not in month_last or e > month_last[ym]:
                month_last[ym] = e
        monthly_candidates = sorted(month_last.values())
        near_is_monthly = near_expiry in set(monthly_candidates)
        # Monthly for cross-analysis = nearest monthly that differs from near expiry.
        # When near_expiry IS the monthly (monthly expiry week), monthly_exp_me becomes
        # next month's monthly — PCR divergence signal is suppressed in that case (below).
        monthly_exp_me = next(
            (m for m in monthly_candidates if m != near_expiry), None
        )
        if monthly_exp_me is not None:
            m_opt = fno_today[
                (fno_today["instrument"] == "OPTIDX") &
                (fno_today["expiry_date"] == monthly_exp_me)
            ]
            monthly_call_oi_me = int(m_opt[m_opt["option_type"] == "CE"]["open_interest"].sum())
            monthly_put_oi_me  = int(m_opt[m_opt["option_type"] == "PE"]["open_interest"].sum())
            if monthly_call_oi_me > 0:
                monthly_pcr_val = round(monthly_put_oi_me / monthly_call_oi_me, 2)
            if spot_close and not m_opt.empty:
                monthly_max_pain_lvl = _compute_key_levels(m_opt, spot_close).max_pain
            total_oi_both = (call_oi + put_oi) + (monthly_call_oi_me + monthly_put_oi_me)
            if total_oi_both > 0:
                gamma_ratio_val = (call_oi + put_oi) / total_oi_both

    # ── Wyckoff range position (used in short squeeze signal) ────────────────
    range_pos: Optional[float] = None
    if spot_close and high_val and low_val and (high_val - low_val) > 5:
        range_pos = (spot_close - low_val) / (high_val - low_val)

    # Prefer options-based near_expiry for display: on expiry day the near-futures
    # contract often has OI=0 (fully settled) so fut_rows picks up the NEXT month's
    # contract, making fut_expiry=next_month while options still reference today.
    # Using near_expiry first keeps the card showing today's expiry correctly.
    near_expiry_display = near_expiry or fut_expiry

    # ── Snapshots ─────────────────────────────────────────────────────────────
    today_snap = _build_snapshot(trade_date, idx_hist, fno_today, near_expiry,
                                 spot_close, futures_price, carry_pts, carry_pct_ann)
    yesterday_snap = None
    if prev_fno_date and not fno_prev.empty:
        pr     = idx_hist[idx_hist["trade_date"] == prev_fno_date]
        pspot  = float(pr.iloc[0]["close_val"]) if not pr.empty and pd.notna(pr.iloc[0]["close_val"]) else None
        pfut_p = pcarry = pcarry_ann = None; popt_exp = None
        # Use the same expiry_date > trade_date gate so yesterday_snap references the
        # same contract as today_snap — making the Today vs Yesterday comparison apples-to-apples.
        pfut   = fno_prev[
            (fno_prev["instrument"] == "FUTIDX") &
            (fno_prev["open_interest"] > 0) &
            (fno_prev["expiry_date"] > trade_date)
        ].sort_values("expiry_date")
        if not pfut.empty:
            pnr = pfut.iloc[0]; pne = _to_date(pnr["expiry_date"])
            pfut_p = float(pnr["settle_price"]) if pd.notna(pnr["settle_price"]) else None
            if pspot and pfut_p and pne:
                pT = max((pne - prev_fno_date).days, 1) / 365.0
                if pT > 3 / 365:
                    pcarry = pfut_p - pspot
                    pcarry_ann = (pcarry / pspot) * (1.0 / pT) * 100
        por = fno_prev[
            (fno_prev["instrument"] == "OPTIDX") &
            (fno_prev["expiry_date"] > trade_date)
        ].sort_values("expiry_date")
        if not por.empty: popt_exp = _to_date(por.iloc[0]["expiry_date"])
        yesterday_snap = _build_snapshot(prev_fno_date, idx_hist, fno_prev, popt_exp,
                                         pspot, pfut_p, pcarry, pcarry_ann)

    # ═══════════════════════════════════════════════════════════════════════════
    # BUILD SIGNAL LIST
    # ═══════════════════════════════════════════════════════════════════════════
    sigs: list[IndexSignal] = []
    add = lambda s: s and sigs.append(s)  # noqa: E731

    # ── Near-expiry option slices (used by Signal 5) ─────────────────────────
    _opt_near_t = (
        fno_today[(fno_today["instrument"] == "OPTIDX") & (fno_today["expiry_date"] == near_expiry)]
        if near_expiry is not None else pd.DataFrame()
    )
    _opt_near_p = (
        fno_prev[(fno_prev["instrument"] == "OPTIDX") & (fno_prev["expiry_date"] == near_expiry)]
        if (near_expiry is not None and not fno_prev.empty) else pd.DataFrame()
    )

    # — Signals 1-7: index-specific price/futures/options —
    add(_sig_price_action(idx_hist))
    # VWAP + RSI removed: walk-forward (pooled 4 indices, point-in-time) found no
    # standalone next-day edge at 1/5/10-day horizons — in fact slightly mean-reverting.
    # They were already context-only (zero vote); now dropped from the output entirely.
    # OI-Price matrix runs on TOTAL futures OI (rollover-immune); roll_note flags when a
    # roll is underway so the verdict isn't read as positions closing.
    _roll_note = ""
    if roll_share is not None and roll_share >= 0.20:
        _roll_note = (f" · 🔄 Rollover underway — {roll_share * 100:.0f}% of futures OI now sits "
                      f"in the next/far month; near-month OI alone would misread this as unwinding.")
    sigs.append(_sig_oi_price_matrix(
        day_change_pct, fut_oi_total_chg, fut_oi_total,
        oi_uncomparable=not total_oi_comparable, roll_note=_roll_note))
    # Carry sanity-bound: suppress the directional carry signal when the annualised carry
    # is outside the plausible arbitrage band (thin/stale settle artifact). None → keep
    # (the no-data path); a real-but-implausible value (e.g. Midcap 38%) → suppress.
    if carry_pct_ann is None or _CARRY_MIN_ANN <= carry_pct_ann <= _CARRY_MAX_ANN:
        add(_sig_carry(carry_pct_ann, carry_pts))

    # ── Options-chain liquidity gate (PCR / max-pain / OI-premium) ───────────────
    _opt_oi_total = int((call_oi or 0) + (put_oi or 0))
    _opt_top_strike_oi = 0
    if not _opt_near_t.empty:
        _opt_top_strike_oi = int(
            _opt_near_t.groupby(["option_type", "strike_price"])["open_interest"].sum().max() or 0
        )
    _opt_conc = (_opt_top_strike_oi / _opt_oi_total) if _opt_oi_total > 0 else 1.0
    _opt_chain_liquid = (_opt_oi_total >= _OPT_OI_MIN) and (_opt_conc <= _OPT_CONC_MAX)
    # Compare PCR only against the SAME expiry's prior-day OI (via _opt_near_p, which is
    # fno_prev filtered to today's near_expiry).  On the first day of a new cycle,
    # _opt_near_p is empty → _prev_pcr = None → no crossing detection fires.
    # Using yesterday_snap.pcr was wrong: it comes from yesterday's OWN near_expiry
    # (the contract that just expired), giving false fresh_spike / fresh_drop signals.
    # After Sep 2025 NSE change: NIFTY weekly = Tuesday (new cycle starts Wednesday);
    # BANKNIFTY/FINNIFTY/MIDCPNIFTY monthly-only (new cycle starts first Wednesday
    # of each month after the last-Tuesday monthly expiry).
    _prev_pcr: Optional[float] = None
    if not _opt_near_p.empty:
        _p_call_oi = int(_opt_near_p[_opt_near_p["option_type"] == "CE"]["open_interest"].sum())
        _p_put_oi  = int(_opt_near_p[_opt_near_p["option_type"] == "PE"]["open_interest"].sum())
        if _p_call_oi > 0:
            _prev_pcr = round(_p_put_oi / _p_call_oi, 2)
    if _opt_chain_liquid:
        add(_sig_pcr(pcr, _prev_pcr))
        add(_sig_opt_oi_premium(_opt_near_t, _opt_near_p))   # Signal 5 — OI-Premium matrix
    if spot_close and dte_options <= 5 and _opt_chain_liquid:
        add(_sig_max_pain(levels.max_pain, spot_close, dte_options))
    # Range-position (Closed Near HOD/LOD) removed from the output.
    # Signal 6b — S/R Confluence: swing-pivot price levels from daily OHLC, read
    # against the last 4 sessions (hold / reject / break) with OI-wall, volume and
    # FII-flow confirmation. OI walls are passed only when the chain is liquid
    # enough to trust them (same gate as PCR / max pain).
    # CONTEXT-ONLY, same verdict as VWAP/RSI: walk-forward (scripts/
    # validate_sr_signal.py) measured pooled IC +0.02 / hit 52% over 293 days ×
    # 4 indices price-only, and IC −0.11 / hit 43% with full confluence on the
    # recent 47-day F&O window — no validated next-day directional edge. The
    # LEVELS themselves (sr_support / sr_resistance) stay on the prediction as
    # the range/risk map; the read informs but does not vote.
    _sr = _sr_level_read(
        idx_hist, spot_close,
        levels=levels if _opt_chain_liquid else None,
        fii_5d=market_ctx.fii_cumul_flows_5d.get(fno_symbol),
    )
    # S/R Confluence (Edwards & Magee support/resistance) removed from the output.

    # — Signals 7-13: FII / institutional —
    # The aggregate FII index-futures POSITION (fao_participant fut_idx net, e.g. −271,979)
    # is a single pooled number across ALL index futures — ~76% NIFTY / 17% BankNifty by OI.
    # Attributing it to an index where FIIs barely trade futures (FinNifty ~0.1%, Midcap ~6%
    # of FII index-fut OI) is misattribution and a prime driver of the collinear all-four-
    # bearish failures. Gate the position-LEVEL signals to indices with a MATERIAL share of
    # FII index-futures OI; the per-index FLOW signals (sourced per-symbol from
    # fii_derivatives_stats) always fire. Fails OPEN when per-index OI data is unavailable.
    _fii_oi_by_sym = market_ctx.fii_oi_cr_latest or {}
    _fii_oi_total  = sum(float(v) for v in _fii_oi_by_sym.values() if v)
    _fii_oi_this   = float(_fii_oi_by_sym.get(fno_symbol, 0.0) or 0.0)
    _fii_participates = (
        (_fii_oi_this / _fii_oi_total) >= _FII_FUT_OI_SHARE_MIN
        if _fii_oi_total > 0 else True   # no per-index data → preserve prior behaviour
    )

    add(_sig_fii_flow(market_ctx, fno_symbol))            # per-index flow — always
    add(_sig_fii_cumulative_flow(market_ctx, fno_symbol)) # per-index 5D flow — always
    if _fii_participates:
        add(_sig_fii_institutional(market_ctx))
        add(_sig_fii_options_delta(market_ctx))
        add(_sig_options_participant(market_ctx))   # retail (Client) options contrarian + Pro confirm
        add(_sig_fii_oi_buildup(market_ctx, fno_symbol))
        # FII 1-day position-change read demoted to CONTEXT: walk-forward (381d × 4idx,
        # 2026-06-19) measured both its members wrong-sign — "FII Covering Shorts →
        # Reversal" IC −0.168 over 306 fires (~2.9 SE) and "FII Adding to Short" IC
        # −0.099 over 487 fires. A naive directional read of FII index-futures position
        # change is exactly the contrarian-noise the prior documents, so it informs but
        # does not vote (Pro-Desk + flow + short-squeeze keep the real Institutional edge).
        add(_context_only(_sig_fii_position_change(market_ctx)))
        add(_sig_short_squeeze_setup(market_ctx, fno_symbol, day_change_pct, range_pos, pcr))

    # — Signals 14-17: market context —
    # Pruned to keep the engine purely index-F&O focused (next-day / week / month
    # index move from index options, index futures, FII/DII/Pro/Client positioning).
    # Removed as NOT index-derivative data (and ≤0 measured next-day IC):
    #   • VIX regime  — directional vote IC −0.13 (wrong-sign). VIX still sizes the
    #     expected-move RANGE directly (vix_close), just no longer votes on direction.
    #   • Sector breadth / defensive-cyclical / RS vs Nifty — cash-equity rotation.
    #   • Constituent stock fut/opt OI (17c/d/e) — single-STOCK F&O; FIIs trade the
    #     INDEX, not the basket, so these dilute the index read.

    # — Signals 18-20: Nifty 50 multi-expiry (fires only when weekly ≠ monthly) —
    if fno_symbol == "NIFTY" and monthly_exp_me is not None:
        dte_monthly = (monthly_exp_me - trade_date).days
        # PCR divergence requires a genuine weekly vs monthly distinction.
        # When near_expiry IS the monthly (last ~7 days before the last Tuesday), "monthly_exp_me"
        # is next month's expiry — comparing same-horizon PCR to next-month PCR
        # produces a meaningless "weekly vs monthly" label. Suppress in that case.
        if not near_is_monthly:
            add(_sig_multi_expiry_pcr(pcr, monthly_pcr_val, near_expiry, monthly_exp_me, trade_date))
        add(_sig_max_pain_convergence(
            levels.max_pain, monthly_max_pain_lvl,
            spot_close, dte_options, dte_monthly,
        ))
        add(_sig_gamma_wall(
            call_oi, put_oi, monthly_call_oi_me, monthly_put_oi_me,
            gamma_ratio_val, dte_options,
        ))

    # — Signals 21-23: Statistical Regime Detection (all indices) —
    regime = None
    try:
        from src.analytics.regime_detection import get_regime_signals
        regime = get_regime_signals(fno_symbol, index_name, trade_date)
        add(_sig_market_memory(regime))
        add(_sig_hmm_regime(regime))
        # Entropy regime (PE / predictability) removed from the output.
    except Exception:
        pass   # Regime signals are supplementary — never block the core prediction

    # ── Two-pass verdict + memory signal ─────────────────────────────────────
    # Pass 1: compute preliminary verdict from signals 1-23 (no memory yet).
    #         This gives us the actual direction to pass to the memory engine
    #         so confirms_prediction is meaningful.
    entropy_conf = float(regime.entropy_conf) if regime and not regime.error else 1.0
    composite_prelim = _aggregate_composite(sigs)
    direction_prelim, _, _, _, _, _ = _compute_verdict(
        composite_prelim, sigs, pcr, carry_pct_ann, dte_options,
        spot_close, levels, market_ctx, entropy_conf=entropy_conf,
    )

    # Signal 24: Memory Engine — query with REAL direction from pass 1
    mem_signal = None
    try:
        from src.analytics.memory_engine import get_memory_signal, extract_features

        class _FeatProxy:
            """Minimal proxy so extract_features can read the values it needs.

            MUST expose every attribute that extract_features() reads from pred:
              pcr, carry_pct_ann, composite_score — direct scalar features
              market_context, regime             — sub-objects (ctx, regime)
              spot_close, levels                 — for feat_max_pain_dist
              fno_symbol                         — for feat_fii_5d_cumul (symbol-specific lookup)

            Missing any of these silently zeros out the corresponding feature dimension
            in the similarity query, causing nearest-neighbour distances to be wrong.
            """
            def __init__(self):
                self.pcr              = pcr
                self.carry_pct_ann    = carry_pct_ann
                self.composite_score  = composite_prelim
                self.market_context   = market_ctx
                self.regime           = regime
                self.spot_close       = spot_close   # feat_max_pain_dist numerator
                self.levels           = levels       # feat_max_pain_dist denominator
                self.fno_symbol       = fno_symbol   # feat_fii_5d_cumul symbol key

        mem_signal = get_memory_signal(
            fno_symbol, trade_date,
            _FeatProxy(),
            direction_pred=direction_prelim,
        )
        if mem_signal and not mem_signal.error and mem_signal.similar_count >= 10:
            add(_sig_memory_engine(mem_signal))
    except Exception:
        pass   # Memory engine is supplementary — never blocks core prediction

    # Pass 2: recompute composite + final verdict including Signal 24
    composite = _aggregate_composite(sigs)
    sigs.sort(key=lambda s: abs(s.score), reverse=True)

    direction, confidence, color, headline, driver, risk = _compute_verdict(
        composite, sigs, pcr, carry_pct_ann, dte_options, spot_close, levels, market_ctx,
        entropy_conf=entropy_conf,
    )

    # ── Expected-Move Forecast (magnitude) ────────────────────────────────────
    em = _compute_expected_move(
        idx_hist, spot_close, market_ctx.vix_close, composite,
        dte_options, levels, is_nifty=(fno_symbol == "NIFTY"),
        gamma_ratio=gamma_ratio_val,
        cal_mult=_EM_CALIBRATION.get(fno_symbol, 1.0),
    )

    # Volume confirmation: today's traded value vs its trailing 20-day average.
    # Validated: a >1σ move on >=1.15x volume continues next day ~58% vs ~52% on
    # weak volume — so expansion confirms a break, thin volume → fade-back risk.
    vol_ratio: Optional[float] = None
    try:
        _vh = idx_hist.sort_values("trade_date")
        _vs = _vh["turnover_cr"].fillna(0.0)
        if float(_vs.tail(20).sum()) <= 0:
            _vs = _vh["volume"].fillna(0.0)
        _base = float(_vs.iloc[-21:-1].mean()) if len(_vs) >= 21 else float(_vs.iloc[:-1].mean())
        if _base > 0:
            vol_ratio = round(float(_vs.iloc[-1]) / _base, 2)
    except Exception:
        vol_ratio = None

    # ── Breakout Extension Ranges (second range if first range is broken) ──────
    bo: dict = {}
    if (em.get("range_low") is not None and em.get("range_high") is not None
            and em.get("expected_move_pts") is not None and spot_close):
        bo = _compute_breakout_extensions(
            spot_close, em["expected_move_pts"],
            em["range_low"], em["range_high"], levels, composite,
            fii_net=market_ctx.fii_fut_idx_net,
            pcr=pcr,
            carry_pct_ann=carry_pct_ann,
            dte=dte_options,
            volume_ratio=vol_ratio,
        )

    # ── Weekly (to-expiry) range — forward range until the next expiry ─────────
    wk = _compute_weekly_range(
        fno_today, spot_close, em.get("expected_move_pct"),
        trade_date, is_nifty=(fno_symbol == "NIFTY"),
    )

    # ── Monthly (to monthly-expiry) range — NIFTY only, from the MONTHLY chain ──
    # Built off the monthly expiry's own straddle + OI walls + max pain, a separate
    # wider scenario band beside the weekly. Skipped when near_expiry already IS the
    # monthly (then wk_* already runs to the monthly, so a second copy is redundant),
    # and for Bank/Fin/Midcap (monthly-only — their wk_* range IS the monthly).
    mn: dict = {}
    if fno_symbol == "NIFTY" and monthly_exp_me is not None and not near_is_monthly:
        mn = _compute_weekly_range(
            fno_today, spot_close, em.get("expected_move_pct"),
            trade_date, is_nifty=True,
            target_expiry=monthly_exp_me, prefix="mn",
        )

    # Surface silent F&O staleness: when the bhavcopy lags the index date, every
    # OI/PCR/walls read is from the older session — say so on the verdict card.
    _note = ""
    if today_fno_date and today_fno_date < trade_date:
        _fno_lag = (trade_date - today_fno_date).days
        risk  = (f"F&O data is from {today_fno_date:%d %b} ({_fno_lag}d old) — "
                 f"OI/PCR/wall reads are stale. ") + risk
        _note = f"F&O bhavcopy {today_fno_date:%d %b} is {_fno_lag}d behind the index data."

    pred_out = IndexPrediction(
        fno_symbol=fno_symbol, display_name=display_name, as_of_date=trade_date,
        note=_note,
        spot_close=spot_close, prev_close=prev_close_price,
        day_change_pct=day_change_pct, high=high_val, low=low_val,
        near_expiry=near_expiry_display, days_to_expiry=dte_options,
        futures_price=futures_price, carry_pts=carry_pts,
        carry_pct_ann=carry_pct_ann, carry_label=carry_label,
        fut_oi=fut_oi, fut_oi_chg=fut_oi_chg, call_oi=call_oi, put_oi=put_oi, pcr=pcr,
        today=today_snap, yesterday=yesterday_snap, levels=levels,
        direction=direction, confidence=confidence, direction_color=color,
        composite_score=composite, headline=headline,
        key_driver=driver, key_risk=risk, signals=sigs,
        data_available=True, market_context=market_ctx,
        regime=regime, mem_signal=mem_signal,
        # Multi-expiry fields (Nifty 50 only)
        weekly_expiry=near_expiry  if fno_symbol == "NIFTY" else None,
        monthly_expiry=monthly_exp_me,
        weekly_pcr=pcr             if fno_symbol == "NIFTY" else None,
        monthly_pcr=monthly_pcr_val,
        weekly_call_oi=call_oi     if fno_symbol == "NIFTY" else 0,
        weekly_put_oi=put_oi       if fno_symbol == "NIFTY" else 0,
        monthly_call_oi=monthly_call_oi_me,
        monthly_put_oi=monthly_put_oi_me,
        monthly_max_pain=monthly_max_pain_lvl,
        gamma_ratio=gamma_ratio_val,
        # Expected-move forecast (magnitude)
        expected_move_pts=em["expected_move_pts"],
        expected_move_pct=em["expected_move_pct"],
        range_low=em["range_low"],
        range_high=em["range_high"],
        target_move_pts=em["target_move_pts"],
        target_close=em["target_close"],
        sideways_band_pts=em["sideways_band_pts"],
        move_basis=em["move_basis"],
        # Breakout extension ranges
        breakout_threshold_pts=bo.get("breakout_threshold_pts"),
        breakout_up_start=bo.get("breakout_up_start"),
        breakout_up_end=bo.get("breakout_up_end"),
        breakout_dn_start=bo.get("breakout_dn_start"),
        breakout_dn_end=bo.get("breakout_dn_end"),
        breakout_up_src=bo.get("breakout_up_src", ""),
        breakout_dn_src=bo.get("breakout_dn_src", ""),
        # Weekly (to-expiry) range
        wk_expiry=wk.get("wk_expiry"), wk_dte_cal=wk.get("wk_dte_cal"),
        wk_kind=wk.get("wk_kind", ""), wk_move_pts=wk.get("wk_move_pts"),
        wk_range_low=wk.get("wk_range_low"), wk_range_high=wk.get("wk_range_high"),
        wk_straddle_pts=wk.get("wk_straddle_pts"), wk_put_floor=wk.get("wk_put_floor"),
        wk_call_cap=wk.get("wk_call_cap"), wk_max_pain=wk.get("wk_max_pain"),
        wk_basis=wk.get("wk_basis", ""),
        # Monthly (to monthly-expiry) range — NIFTY only
        mn_expiry=mn.get("mn_expiry"), mn_dte_cal=mn.get("mn_dte_cal"),
        mn_kind=mn.get("mn_kind", ""), mn_move_pts=mn.get("mn_move_pts"),
        mn_range_low=mn.get("mn_range_low"), mn_range_high=mn.get("mn_range_high"),
        mn_straddle_pts=mn.get("mn_straddle_pts"), mn_put_floor=mn.get("mn_put_floor"),
        mn_call_cap=mn.get("mn_call_cap"), mn_max_pain=mn.get("mn_max_pain"),
        mn_basis=mn.get("mn_basis", ""),
        # Price-structure S/R (Signal 6b)
        sr_support=_sr.get("support"), sr_resistance=_sr.get("resistance"),
        sr_support_touches=_sr.get("sup_touches", 0),
        sr_resistance_touches=_sr.get("res_touches", 0),
    )

    # Auto-store prediction in memory engine (non-fatal).
    # Temporarily set composite_score = composite_prelim before storing so that
    # feat_oi_score in the DB fingerprint uses the SAME value that was used in the
    # similarity query (_FeatProxy.composite_score = composite_prelim). Without this,
    # the stored feat_oi_score reflects the final composite (including Signal 24 ±2.5)
    # while every future query uses the pre-memory composite — a systematic mismatch.
    # persist=False on the read-only dashboard path avoids a prediction_log WRITE
    # (and its exclusive DuckDB lock) on every render. Daily logging is guaranteed
    # by cmd_daily, so viewing never needs to write.
    if persist:
        try:
            from src.analytics.memory_engine import store_prediction
            pred_out.composite_score = composite_prelim
            store_prediction(pred_out, trade_date)
            pred_out.composite_score = composite   # restore final for the caller
        except Exception:
            pred_out.composite_score = composite   # always restore on error path

    return pred_out
