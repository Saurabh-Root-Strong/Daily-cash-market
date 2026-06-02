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
  17c.Constituent Breadth      Nifty 50 ONLY — stock-level breadth + delivery quality
                               Turnover-weighted advance ratio across all 50 index members
                               Score ≥ 75% turnover advancing = +2.5 exceptional breadth
                               Delivery ≥ 55% on advancers = +0.5 institutional accumulation
                               Reveals WHO is buying (institutions vs day traders)
  17d.Constituent Fut OI       Nifty 50 ONLY — stock futures OI-Price matrix consensus
                               Fresh long/short buildup across 50 constituent stocks
                               Stock futures = always directional (no hedge use case)
                               net bull ≥ 30% above bear = +2.0 basket long conviction
  17e.Constituent Opt OI-Prem  Nifty 50 ONLY — stock options OI-Premium matrix basket
                               C.Buy+P.Write = Net Long per stock; aggregated across 50
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
    "Nifty Bank", "Nifty Auto", "Nifty Realty", "Nifty Metal",
    "Nifty Infrastructure", "Nifty PSU Bank", "Nifty Private Bank",
]
_DEFENSIVE_SECTORS = [
    "Nifty FMCG", "Nifty Pharma", "Nifty Healthcare",
    "Nifty IT", "Nifty Consumer Durables",
]

_INDIA_REPO = 6.5   # annualised % — RBI repo for fair-value carry

# ── Nifty 50 constituent symbols ─────────────────────────────────────────────
# These 50 stocks directly drive the Nifty 50 index (free-float market-cap weighted).
# Used to compute a stock-level breadth + delivery quality signal — an edge that
# sector-index breadth cannot provide because sector indices are themselves derived
# from the same underlying move, while individual stock delivery reveals WHO is buying.
# Source: NSE ind_nifty50list.csv  |  Last reviewed: Jun 2026
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

    # ── Nifty 50 constituent stock data — cash market (Signal 17c) ──────────────
    n50_coverage:      int   = 0
    n50_advancing:     int   = 0
    n50_declining:     int   = 0
    n50_adv_turn_pct:  Optional[float] = None
    n50_adv_deliv_avg: Optional[float] = None
    n50_dec_deliv_avg: Optional[float] = None

    # ── Nifty 50 constituent stock futures OI-Price matrix (Signal 17d) ─────────
    # Unlike index futures (which FII use as portfolio hedges), individual stock
    # futures are almost always directional bets. When 30+ Nifty 50 stocks show
    # fresh long buildup simultaneously, institutions are buying the BASKET.
    n50_fut_valid:       int = 0   # stocks with valid today vs yesterday comparison
    n50_fut_fresh_long:  int = 0   # price↑ + OI↑ = new money entering longs
    n50_fut_fresh_short: int = 0   # price↓ + OI↑ = new money entering shorts
    n50_fut_long_unwind: int = 0   # price↓ + OI↓ = longs exiting
    n50_fut_short_cov:   int = 0   # price↑ + OI↓ = shorts covering

    # ── Nifty 50 constituent stock options OI-Premium matrix (Signal 17e) ───────
    # OI-Premium matrix applied to the options chains of all 50 constituent stocks.
    # Disambiguates WHAT is happening in options: buying (OI↑+Prem↑) vs writing
    # (OI↑+Prem↓). Aggregate basket configuration reveals institutional intent.
    n50_opt_valid:     int = 0   # stocks with valid call + put classification
    n50_opt_net_long:  int = 0   # call buying and/or put writing (institutional bull)
    n50_opt_net_short: int = 0   # put buying and/or call writing (institutional bear)
    n50_opt_straddle:  int = 0   # both call + put buying (big move expected, no direction)
    n50_opt_range:     int = 0   # both call + put writing (low vol, range play)


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
    weekly_expiry: Optional[date] = None    # near options expiry (weekly for Nifty)
    monthly_expiry: Optional[date] = None   # last Thursday of calendar month
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


# ── Public entry point ────────────────────────────────────────────────────────

def get_index_predictions(trade_date: date) -> list[IndexPrediction]:
    """Compute next-day predictions for all major F&O indices."""
    ctx = _build_market_context(trade_date)
    return [
        _compute_prediction(trade_date, sym, name, idx_name, ctx)
        for sym, (name, idx_name) in _INDEX_MAP.items()
    ]


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

def _load_index_history(index_name: str, trade_date: date, lookback: int = 40) -> pd.DataFrame:
    start = trade_date - timedelta(days=lookback * 2)
    df = query_dataframe("""
        SELECT trade_date, open_val, high_val, low_val, close_val, prev_close, pct_chg
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
    return query_dataframe("""
        SELECT index_name, close_val, pct_chg FROM index_data
        WHERE trade_date = ?
          AND index_name NOT IN (
            'India VIX','Nifty 50','Nifty Bank',
            'Nifty Financial Services','Nifty Midcap Select'
          )
    """, [trade_date])


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
            elif ct == "DII":    ctx.dii_fut_idx_net    = fut_net
            elif ct == "CLIENT": ctx.client_fut_idx_net = fut_net
            elif ct in ("PRO", "PROPRIETORY", "PROP"): ctx.pro_fut_idx_net = fut_net

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

    # ── Nifty 50 constituent breadth + delivery quality (Signal 17c) ─────────
    # Query cash-market data for all 50 constituent stocks. Turnover-weighted
    # advance ratio is a more faithful proxy for index impact than a simple count
    # because the index is free-float market-cap weighted (Reliance ~10%, BEL ~0.5%).
    _sym_ph = ", ".join("?" * len(_NIFTY50_SYMBOLS))
    _stocks = query_dataframe(f"""
        SELECT
            symbol,
            CASE WHEN prev_close > 0
                 THEN (close_price - prev_close) / prev_close * 100
                 ELSE NULL END AS pct_chg,
            deliv_per,
            turnover_lacs
        FROM daily_data
        WHERE symbol IN ({_sym_ph})
          AND trade_date = ?
          AND series IN ('EQ', 'SM', 'ST')
    """, [*_NIFTY50_SYMBOLS, trade_date])
    if not _stocks.empty:
        _stocks = _stocks.dropna(subset=["pct_chg", "turnover_lacs"]).copy()
        _stocks = _stocks[_stocks["turnover_lacs"] > 0]
        ctx.n50_coverage  = len(_stocks)
        _adv = _stocks[_stocks["pct_chg"] > 0]
        _dec = _stocks[_stocks["pct_chg"] < 0]
        ctx.n50_advancing = len(_adv)
        ctx.n50_declining = len(_dec)
        _total_turn = float(_stocks["turnover_lacs"].sum())
        if _total_turn > 0:
            ctx.n50_adv_turn_pct = float(_adv["turnover_lacs"].sum()) / _total_turn * 100
        if len(_adv) >= 3:
            ctx.n50_adv_deliv_avg = float(_adv["deliv_per"].dropna().mean()) if "deliv_per" in _adv else None
        if len(_dec) >= 3:
            ctx.n50_dec_deliv_avg = float(_dec["deliv_per"].dropna().mean()) if "deliv_per" in _dec else None

    # ── Nifty 50 constituent stock futures OI-Price matrix (Signal 17d) ──────
    # Load near-expiry FUTSTK rows for the last 2 trading dates for all 50 symbols.
    # Near expiry = lowest expiry_date with open_interest > 0 (same logic as index futures).
    _sym_ph_f = ", ".join("?" * len(_NIFTY50_SYMBOLS))
    _n50_fut_df = query_dataframe(f"""
        WITH last2 AS (
            SELECT DISTINCT trade_date
            FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK'
              AND symbol IN ({_sym_ph_f})
              AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 2
        )
        SELECT f.symbol, f.trade_date, f.expiry_date,
               f.open_interest, f.settle_price
        FROM fno_bhavcopy f
        WHERE f.instrument = 'FUTSTK'
          AND f.symbol IN ({_sym_ph_f})
          AND f.trade_date IN (SELECT trade_date FROM last2)
          AND f.open_interest > 0
        ORDER BY f.symbol, f.trade_date DESC, f.expiry_date
    """, [*_NIFTY50_SYMBOLS, trade_date, *_NIFTY50_SYMBOLS])

    if not _n50_fut_df.empty:
        _n50_fut_df["trade_date"] = pd.to_datetime(_n50_fut_df["trade_date"]).dt.date
        _n50_fut_df["expiry_date"] = pd.to_datetime(_n50_fut_df["expiry_date"]).dt.date
        _fut_dates = sorted(_n50_fut_df["trade_date"].unique(), reverse=True)
        _td = _fut_dates[0] if _fut_dates else None
        _pd = _fut_dates[1] if len(_fut_dates) >= 2 else None

        if _td and _pd:
            # Take near expiry (first row per symbol × date after sort by expiry_date)
            _today_fut = (
                _n50_fut_df[_n50_fut_df["trade_date"] == _td]
                .groupby("symbol", as_index=False).first()
            )
            _prev_fut = (
                _n50_fut_df[_n50_fut_df["trade_date"] == _pd]
                .groupby("symbol", as_index=False).first()
            )
            _prev_idx = _prev_fut.set_index("symbol")

            _fl = _sc = _lu = _fs = _valid = 0
            for _, tr in _today_fut.iterrows():
                sym = tr["symbol"]
                if sym not in _prev_idx.index:
                    continue
                pr = _prev_idx.loc[sym]
                # Only compare same expiry (skip rollover days for this stock)
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

            ctx.n50_fut_valid       = _valid
            ctx.n50_fut_fresh_long  = _fl
            ctx.n50_fut_fresh_short = _fs
            ctx.n50_fut_long_unwind = _lu
            ctx.n50_fut_short_cov   = _sc

    # ── Nifty 50 constituent stock options OI-Premium matrix (Signal 17e) ────
    # Aggregate near-expiry call + put OI change and volume-weighted premium change
    # across all 50 constituent stocks. settle_price for OPTSTK = previous-day
    # settlement (NSE DAT column 9 = "previous close price"), so
    # (close_price - settle_price) = today's premium change. Same as Signal 5.
    _sym_ph_o = ", ".join("?" * len(_NIFTY50_SYMBOLS))
    _n50_opt_df = query_dataframe(f"""
        WITH last2 AS (
            SELECT DISTINCT trade_date
            FROM fno_bhavcopy
            WHERE instrument = 'OPTSTK'
              AND symbol IN ({_sym_ph_o})
              AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 2
        ),
        near_exp_date AS (
            SELECT symbol, trade_date, MIN(expiry_date) AS near_expiry
            FROM fno_bhavcopy
            WHERE instrument = 'OPTSTK'
              AND symbol IN ({_sym_ph_o})
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
    """, [*_NIFTY50_SYMBOLS, trade_date, *_NIFTY50_SYMBOLS])

    if not _n50_opt_df.empty:
        _n50_opt_df["trade_date"]  = pd.to_datetime(_n50_opt_df["trade_date"]).dt.date
        _n50_opt_df["near_expiry"] = pd.to_datetime(_n50_opt_df["near_expiry"]).dt.date
        _opt_dates = sorted(_n50_opt_df["trade_date"].unique(), reverse=True)
        _opt_td = _opt_dates[0] if _opt_dates else None
        _opt_pd = _opt_dates[1] if len(_opt_dates) >= 2 else None

        if _opt_td and _opt_pd:
            # Build lookup: (symbol, option_type) → row  for each date
            _opt_today = _n50_opt_df[_n50_opt_df["trade_date"] == _opt_td].set_index(["symbol", "option_type"])
            _opt_prev  = _n50_opt_df[_n50_opt_df["trade_date"] == _opt_pd].set_index(["symbol", "option_type"])
            _MIN_CONTR = 100   # min contracts for premium change to be meaningful

            _on = _os = _ostr = _or = _ov = 0

            # Collect all symbols that have both CE and PE data today
            _syms_today = {s for s, _ in _opt_today.index}
            for _sym in _syms_today:
                try:
                    t_ce = _opt_today.loc[(_sym, "CE")]
                    t_pe = _opt_today.loc[(_sym, "PE")]
                except KeyError:
                    continue   # missing CE or PE for this symbol today

                # Rollover guard: skip if near expiry changed vs yesterday
                try:
                    p_ce = _opt_prev.loc[(_sym, "CE")]
                    p_pe = _opt_prev.loc[(_sym, "PE")]
                except KeyError:
                    continue   # no previous data for this stock

                if t_ce["near_expiry"] != p_ce["near_expiry"]:
                    continue   # monthly expiry rollover for this stock

                # OI change %
                _c_oi_t = float(t_ce["total_oi"]); _c_oi_p = float(p_ce["total_oi"])
                _p_oi_t = float(t_pe["total_oi"]); _p_oi_p = float(p_pe["total_oi"])
                if _c_oi_p <= 0 or _p_oi_p <= 0:
                    continue

                c_oi_chg = (_c_oi_t - _c_oi_p) / _c_oi_p * 100
                p_oi_chg = (_p_oi_t - _p_oi_p) / _p_oi_p * 100
                c_prem   = t_ce.get("wt_prem_chg"); c_contr = float(t_ce.get("total_contracts", 0) or 0)
                p_prem   = t_pe.get("wt_prem_chg"); p_contr = float(t_pe.get("total_contracts", 0) or 0)

                # Classify each side (OI must change meaningfully; premium needs volume)
                c_oi_up = c_oi_chg >  0.5
                p_oi_up = p_oi_chg >  0.5
                c_prem_up = (c_prem is not None and not pd.isna(c_prem) and
                             float(c_prem) > 0 and c_contr >= _MIN_CONTR)
                p_prem_up = (p_prem is not None and not pd.isna(p_prem) and
                             float(p_prem) > 0 and p_contr >= _MIN_CONTR)

                c_buy = c_oi_up and c_prem_up    # call OI↑ + premium↑ = call buying
                c_wrt = c_oi_up and not c_prem_up # call OI↑ + premium↓ = call writing
                p_buy = p_oi_up and p_prem_up     # put OI↑ + premium↑ = put buying
                p_wrt = p_oi_up and not p_prem_up  # put OI↑ + premium↓ = put writing

                # Skip if neither side has significant OI change
                if not (c_oi_up or p_oi_up):
                    continue

                _ov += 1
                if   c_buy and p_wrt:  _on   += 1   # both sides bullish → net long
                elif c_wrt and p_buy:  _os   += 1   # both sides bearish → net short
                elif c_buy and p_buy:  _ostr += 1   # both buying → straddle
                elif c_wrt and p_wrt:  _or   += 1   # both writing → range
                elif c_buy or p_wrt:   _on   += 1   # single-sided bullish
                elif p_buy or c_wrt:   _os   += 1   # single-sided bearish

            ctx.n50_opt_valid    = _ov
            ctx.n50_opt_net_long = _on
            ctx.n50_opt_net_short= _os
            ctx.n50_opt_straddle = _ostr
            ctx.n50_opt_range    = _or

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
    """3D return z-score. NSE backtest: z ≤ -1.8 → 67% next-day bounce."""
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
                "NSE backtest: 67% next-day bounce probability after multi-day declines at this intensity. "
                "Selling is stretched — mean reversion expected."
            ), emoji="🔄",
        )
    if z >= 3.0:
        return IndexSignal(
            name="Extreme Overbought (Caution)", category="Price Action",
            direction=-1, score=-1.0,
            headline=f"Extreme Run +{cum_3d:.1f}% / 3D (z=+{z:.1f})",
            description=(
                f"3-day gain {cum_3d:.1f}% statistically extreme. "
                "Indian market shows 72% momentum continuation — mild caution only."
            ), emoji="⚠️",
        )
    return None


def _sig_oi_price_matrix(
    pct_chg: Optional[float],
    fut_oi_chg: int,
    fut_oi_base: int = 0,
    is_rollover: bool = False,
) -> IndexSignal:
    """OI-Price matrix — Murphy. Most reliable institutional directional signal.

    Uses percentage-based OI threshold (>0.5% change) to be scale-invariant across
    both old zip format (OI in lots) and new DAT format (OI in underlying units).

    is_rollover=True: today's near contract differs from yesterday's (monthly expiry).
    OI comparison is meaningless on rollover day — the signal falls back to pure price
    direction at half score, labeled explicitly so the user is not misled.
    """
    pct = pct_chg or 0.0
    up  = pct > 0.10; dn = pct < -0.10

    # ── Rollover day: OI comparison is invalid — brand-new contract, no prev ──
    if is_rollover:
        if up:
            return IndexSignal(
                "Expiry Rollover — Price Up (OI N/A)", "Futures OI", 1, 1.0,
                f"Rollover day — price +{pct:.2f}%, OI change not comparable",
                f"Monthly futures rollover: near contract changed today. OI change crosses contracts "
                f"and is meaningless. Price direction only: +{pct:.2f}% (mild bullish lean). "
                "Wait 1–2 days for OI to settle into new contract.", "🔄",
            )
        if dn:
            return IndexSignal(
                "Expiry Rollover — Price Down (OI N/A)", "Futures OI", -1, -1.0,
                f"Rollover day — price {pct:.2f}%, OI change not comparable",
                f"Monthly futures rollover: near contract changed today. OI change crosses contracts "
                f"and is meaningless. Price direction only: {pct:.2f}% (mild bearish lean). "
                "Wait 1–2 days for OI to settle into new contract.", "🔄",
            )
        return IndexSignal(
            "Expiry Rollover — Flat (OI N/A)", "Futures OI", 0, 0.0,
            "Rollover day — price flat, OI not comparable across contracts",
            "Monthly rollover: OI comparison invalid. No directional edge today.", "🔄",
        )

    oi_chg_pct = (fut_oi_chg / max(fut_oi_base, 1)) * 100 if fut_oi_base > 0 else 0.0
    oi_up = oi_chg_pct > 0.5
    oi_dn = oi_chg_pct < -0.5
    if up and oi_up:
        return IndexSignal("Fresh Long Buildup", "Futures OI", 1, 3.0, "Fresh Long Build",
            f"Price +{pct:.2f}% AND OI +{oi_chg_pct:.1f}%. New money entering longs — "
            "strongest bullish confirmation. Institutions building directional long.", "🟢")
    if up and oi_dn:
        return IndexSignal("Short Covering Rally", "Futures OI", 1, 1.0, "Short Covering",
            f"Price +{pct:.2f}% but OI {oi_chg_pct:+.1f}% (falling). "
            "Short covering rally — real but lacks fresh conviction.", "🟡")
    if dn and oi_up:
        return IndexSignal("Fresh Short Buildup", "Futures OI", -1, -3.0, "Fresh Short Build",
            f"Price {pct:.2f}% AND OI +{oi_chg_pct:.1f}%. "
            "New shorts added into decline — strongest bearish confirmation.", "🔴")
    if dn and oi_dn:
        return IndexSignal("Long Unwinding", "Futures OI", -1, -1.0, "Long Unwinding",
            f"Price {pct:.2f}% and OI {oi_chg_pct:+.1f}% (falling). "
            "Longs closing — bearish but self-limiting.", "🟡")
    return IndexSignal("Sideways / Indecisive", "Futures OI", 0, 0.0, "Indecisive OI",
        f"Price {pct:.2f}% with small OI change ({oi_chg_pct:+.1f}%). No directional conviction.", "⚪")


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
        return IndexSignal("Futures Slight Discount", "Carry", -1, -1.0,
            f"Carry {carry_pct_ann:.1f}% ann — Slight Discount",
            f"Futures at {carry_pts:.0f} pts ({carry_pct_ann:.1f}% ann). Below fair value.", "🟡")
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
    FII net index futures OI vs Client divergence.
    Schwager: follow smart money. FAO divergence resolves in FII's direction >90% of cases.
    Score decays with data age via _lag_mult so 3-day-old FAO fires at 40% weight.
    """
    if ctx.fao_date is None: return None
    fii = ctx.fii_fut_idx_net; cli = ctx.client_fut_idx_net; dii = ctx.dii_fut_idx_net
    lag = (ctx.trade_date - ctx.fao_date).days
    lm  = _lag_mult(lag)
    tag = f" [FAO {ctx.fao_date.strftime('%d %b')}{', ' + str(lag) + 'd lag' if lag else ''}]"
    dii_txt = f" | DII: {dii:+,}" if dii != 0 else ""

    # Hedge awareness: an FII index-futures SHORT that is offset by net-long stock
    # futures is a market-neutral HEDGE, not a directional bearish bet.
    # Guard: only apply hedge dampening when FAO and FII-stats data are from the
    # SAME day (≤1-day gap). When dates diverge, stock-fut book may have changed,
    # and mixing asynchronous data risks misclassifying a genuine directional short
    # as a hedge. When dates align, dampen bearish score to 40%.
    _stats_fao_gap = (
        abs((ctx.fii_stats_date - ctx.fao_date).days)
        if ctx.fii_stats_date and ctx.fao_date else 99
    )
    is_hedge = bool(
        _stats_fao_gap <= 1 and
        ctx.fii_stock_fut_net_cr and ctx.fii_stock_fut_net_cr > 500
    )
    bear_mult = 0.4 if is_hedge else 1.0
    neutral_note = ""
    if is_hedge:
        neutral_note = (f" Note: FII also net BUYING stock futures (+Rs{ctx.fii_stock_fut_net_cr:,.0f}Cr) "
                        "= index short is a hedge, not pure bearish — score dampened.")

    FII_T, CLI_T = 80_000, 100_000
    if fii < -FII_T and cli > CLI_T:
        return IndexSignal("FII Short vs Retail Long — Smart Money BEARISH",
            "Institutional", -1, round(-3.0 * bear_mult * lm, 2),
            f"FII {fii:+,} | Client {cli:+,}{tag}",
            f"FII {abs(fii):,} net SHORT vs Client net LONG {cli:,}{dii_txt}. "
            "Institutional-retail divergence resolves in FII's direction (>90% historical rate). "
            f"Institutions positioned against retail longs.{neutral_note}", "🐻")
    if fii > FII_T and cli < -CLI_T:
        return IndexSignal("FII Long vs Retail Short — Smart Money BULLISH",
            "Institutional", 1, round(3.0 * lm, 2),
            f"FII {fii:+,} | Client {cli:+,}{tag}",
            f"FII net LONG {fii:,} vs Client net SHORT {abs(cli):,}{dii_txt}. "
            "Smart money bullish vs retail shorts — short squeeze setup.", "🐂")
    if fii < -FII_T:
        return IndexSignal("FII Net Short — Institutional Bearish Bias",
            "Institutional", -1, round(-2.0 * bear_mult * lm, 2),
            f"FII Net {fii:+,} contracts (SHORT){tag}",
            f"FII carrying {abs(fii):,} net short index futures{dii_txt}.{neutral_note} "
            "Institutional short position creates overhead pressure on rallies.", "📉")
    if fii > FII_T:
        return IndexSignal("FII Net Long — Institutional Bullish Bias",
            "Institutional", 1, round(2.0 * lm, 2),
            f"FII Net {fii:+,} contracts (LONG){tag}",
            f"FII carrying {fii:,} net long index futures{dii_txt}. "
            "Institutional long positioning provides momentum support.", "📈")
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
    oi_now = ctx.fii_oi_cr_latest.get(fno_symbol)
    oi_ago = ctx.fii_oi_cr_5d_ago.get(fno_symbol)
    if oi_now is None or oi_ago is None or oi_ago == 0: return None

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

    # Only signal on meaningful moves (>3,000 contracts = ~Rs500Cr at Nifty levels)
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
    When shorts are trapped with no escape, forced covering creates explosive rallies.
    Historical win rate: ~72% next-day positive when all 3 criteria met.
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
        f"Confirmations: {', '.join(criteria)}. Historical win rate: ~72% next-day positive.", "💥")


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
        "when everyone has sold, no sellers remain. >22 VIX → 70% bounce probability in 5D.", "🔥")


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


def _sig_constituent_breadth(ctx: MarketContext) -> Optional[IndexSignal]:
    """
    Signal 17c: Nifty 50 constituent breadth + delivery quality — NIFTY exclusive.

    WHY THIS IS POWERFUL:
      Nifty 50 is free-float market-cap weighted across its 50 member stocks.
      Sector-index breadth tells us how many sector INDICES advanced — but each
      sector index is already derived from the same underlying stock moves, making
      it a coarser proxy. This signal goes one level deeper: the ACTUAL 50 stocks.

      Delivery % distinguishes REAL money from speculation:
        High delivery + advancing = institutions accumulating the index basket
        Low delivery + advancing  = leveraged/intraday momentum (less durable)
        High delivery + declining = institutions distributing (more lasting weakness)

    SCORING (turnover-weighted advance ratio, 0–100%):
      ≥ 75%  → +2.5  Exceptional breadth: almost the entire index basket rising
      ≥ 60%  → +1.5  Strong breadth: majority rising with broad participation
      40–60% → None  Mixed: no edge, stocks split roughly evenly
      ≤ 40%  → -1.5  Weak breadth: majority declining
      ≤ 25%  → -2.5  Exceptional weakness: near-universal decline in basket

    DELIVERY MODIFIER (applied after breadth score):
      +0.5 if advancing stocks avg delivery ≥ 55% → institutional accumulation
      -0.3 if advancing stocks avg delivery ≤ 25% → speculative rally, low conviction
      -0.5 if declining stocks avg delivery ≥ 55% → institutional distribution
    """
    if ctx.n50_coverage < 30 or ctx.n50_adv_turn_pct is None:
        return None

    twar = ctx.n50_adv_turn_pct   # turnover-weighted advance %, 0–100
    adv  = ctx.n50_advancing
    dec  = ctx.n50_declining
    cov  = ctx.n50_coverage
    adv_del = ctx.n50_adv_deliv_avg
    dec_del = ctx.n50_dec_deliv_avg

    if twar >= 75:
        base  = 2.5
        name  = "N50 Exceptional Breadth — Broad-Based Index Rally"
        head  = f"{adv}/{cov} Nifty 50 stocks up ({twar:.0f}% of turnover) — Exceptional"
        emoji = "🟢"
    elif twar >= 60:
        base  = 1.5
        name  = "N50 Strong Breadth — Majority of Index Basket Rising"
        head  = f"{adv}/{cov} Nifty 50 stocks up ({twar:.0f}% of turnover) — Broad Participation"
        emoji = "🟢"
    elif twar <= 25:
        base  = -2.5
        name  = "N50 Exceptional Weakness — Near-Universal Index Basket Decline"
        head  = f"{dec}/{cov} Nifty 50 stocks down ({100-twar:.0f}% of turnover) — Broad Sell-Off"
        emoji = "🔴"
    elif twar <= 40:
        base  = -1.5
        name  = "N50 Weak Breadth — Majority of Index Basket Declining"
        head  = f"{dec}/{cov} Nifty 50 stocks down ({100-twar:.0f}% of turnover) — Weak Participation"
        emoji = "🔴"
    else:
        return None   # 40–60%: mixed, no directional edge

    # Delivery quality modifier
    deliv_note = ""
    if base > 0:
        if adv_del is not None and adv_del >= 55:
            base += 0.5
            deliv_note = (
                f" Advancing stocks: {adv_del:.0f}% avg delivery = institutional quality — "
                "real money is buying the index basket, not just intraday momentum."
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
                "real money is selling the index basket, not panic or forced selling."
            )
            emoji = "💀"
        elif dec_del is not None:
            deliv_note = f" Declining stocks: {dec_del:.0f}% avg delivery."

    direction = 1 if base > 0 else -1
    flat = cov - adv - dec
    desc = (
        f"Nifty 50 constituent breadth: {adv} advancing / {dec} declining / "
        f"{flat} flat out of {cov} stocks tracked. "
        f"Turnover-weighted advance ratio: {twar:.1f}% — large-cap stocks "
        f"(Reliance, HDFC Bank, ICICI Bank) contribute proportionally to their "
        f"index weight, so this ratio closely tracks actual index driver quality. "
        f"{deliv_note}"
        f"\nBreadth edge: when >60% turnover-weight advances + institutional delivery, "
        f"the rally is in the INDEX BASKET (not just 2-3 heavyweights lifting the index) "
        f"— historically more durable than index-level momentum alone."
    )

    return IndexSignal(name, "Sector", direction, round(base, 2), head, desc, emoji)


def _sig_constituent_fut_oi(ctx: MarketContext) -> Optional[IndexSignal]:
    """
    Signal 17d: Nifty 50 constituent stock futures OI-Price matrix — NIFTY exclusive.

    WHY THIS OUTPERFORMS THE INDEX-LEVEL OI SIGNAL:
      Index futures (FUTIDX) are routinely used by FIIs as portfolio hedges against
      their large cash equity holdings. A massive FII short in NIFTY index futures
      is usually a hedge, not a directional call (the diagnostic confirmed this fires
      bearish on ~100% of days). Stock futures (FUTSTK) carry NO such ambiguity —
      you do not hedge a cash HDFC Bank position by shorting HDFC Bank futures;
      that would just close the position. Every FUTSTK OI change is a directional bet.

      When 30+ of the 50 index stocks show Fresh Long Buildup simultaneously,
      institutions are constructing BASKET LONGS via individual names. This is the
      institutional "all-in" signal that the index OI matrix cannot detect.

    OI-PRICE MATRIX applied per constituent stock (near-expiry futures only):
      Price ↑ + OI ↑ = Fresh Long Build  — new money entering; highest conviction
      Price ↑ + OI ↓ = Short Covering    — shorts forced out; bullish but finite
      Price ↓ + OI ↓ = Long Unwinding    — longs exiting; bearish but self-limiting
      Price ↓ + OI ↑ = Fresh Short Build — new money entering shorts; highest conviction

    SCORING (based on net_balance = bull_pct − bear_pct):
      net ≥ 0.45 + fresh_long ≥ 30%  → +3.0  Exceptional basket conviction long
      net ≥ 0.30                      → +2.0  Broadly bullish stock basket
      net ≥ 0.15                      → +1.0  Mild bullish lean in basket
      net ≤ −0.45 + fresh_short ≥ 30% → −3.0  Exceptional basket conviction short
      net ≤ −0.30                      → −2.0  Broadly bearish stock basket
      net ≤ −0.15                      → −1.0  Mild bearish lean
    """
    valid = ctx.n50_fut_valid
    if valid < 20:
        return None   # fewer than 20 comparable stocks — data too thin for a read

    fl = ctx.n50_fut_fresh_long
    sc = ctx.n50_fut_short_cov
    lu = ctx.n50_fut_long_unwind
    fs = ctx.n50_fut_fresh_short

    bull_pct = (fl + sc) / valid   # all upward-priced outcomes (regardless of mechanism)
    bear_pct = (fs + lu) / valid   # all downward-priced outcomes
    fl_pct   = fl / valid          # purely fresh longs (highest conviction, new money in)
    fs_pct   = fs / valid          # purely fresh shorts (highest conviction, new money in)
    net      = bull_pct - bear_pct

    if net >= 0.45 and fl_pct >= 0.30:
        score = 3.0
        name  = "N50 Basket Fresh Long Build — Institutional Conviction BUY"
        head  = (f"{fl}/{valid} N50 stocks Fresh Long + {sc} Short Cov "
                 f"= {bull_pct:.0%} of basket bulls (net {net:+.0%})")
        emoji = "🔥"
    elif net >= 0.30:
        score = 2.0
        name  = "N50 Basket Broadly Bullish — Stock Futures Net Long"
        head  = (f"{fl} fresh long + {sc} short cov vs {fs} fresh short + {lu} unwind "
                 f"({bull_pct:.0%} bulls, net {net:+.0%})")
        emoji = "🟢"
    elif net >= 0.15:
        score = 1.0
        name  = "N50 Basket Mild Bullish — More Stock Longs than Shorts"
        head  = f"N50 stock futures: {bull_pct:.0%} bullish vs {bear_pct:.0%} bearish (mild bull lean)"
        emoji = "🟡"
    elif net <= -0.45 and fs_pct >= 0.30:
        score = -3.0
        name  = "N50 Basket Fresh Short Build — Institutional Conviction SELL"
        head  = (f"{fs}/{valid} N50 stocks Fresh Short + {lu} Long Unwind "
                 f"= {bear_pct:.0%} of basket bears (net {net:+.0%})")
        emoji = "💀"
    elif net <= -0.30:
        score = -2.0
        name  = "N50 Basket Broadly Bearish — Stock Futures Net Short"
        head  = (f"{fs} fresh short + {lu} unwind vs {fl} fresh long + {sc} cov "
                 f"({bear_pct:.0%} bears, net {net:+.0%})")
        emoji = "🔴"
    elif net <= -0.15:
        score = -1.0
        name  = "N50 Basket Mild Bearish — More Stock Shorts than Longs"
        head  = f"N50 stock futures: {bear_pct:.0%} bearish vs {bull_pct:.0%} bullish (mild bear lean)"
        emoji = "🟠"
    else:
        return None   # net between −0.15 and +0.15: mixed, no directional edge

    direction = 1 if score > 0 else -1
    desc = (
        f"OI-Price matrix applied to {valid} Nifty 50 constituent stocks with active futures:\n"
        f"  Fresh Long Build  : {fl:2d} ({fl_pct:.0%}) — price ↑ + OI ↑ → new money entering longs\n"
        f"  Short Covering    : {sc:2d} — price ↑ + OI ↓ → shorts forced to buy back\n"
        f"  Long Unwinding    : {lu:2d} — price ↓ + OI ↓ → longs choosing to exit\n"
        f"  Fresh Short Build : {fs:2d} ({fs_pct:.0%}) — price ↓ + OI ↑ → new money entering shorts\n"
        f"\n"
        f"INTERPRETATION: Index futures can be hedges — when FII shorts NIFTY futures they "
        f"are often protecting a long cash portfolio (hence the bearish bias every day). "
        f"STOCK futures have no such hedge role. When {fl} index components simultaneously "
        f"show fresh long buildup in their individual stock futures, institutions are "
        f"constructing the index basket via stock-level names — a directional conviction "
        f"call that is independent of and complementary to the index-level OI signal."
    )

    return IndexSignal(name, "Futures OI", direction, score, head, desc, emoji)


def _sig_constituent_opt_oi_prem(ctx: MarketContext) -> Optional[IndexSignal]:
    """
    Signal 17e: Nifty 50 constituent stock options OI-Premium matrix — NIFTY exclusive.

    WHAT IT MEASURES:
      For each Nifty 50 stock: aggregate all near-expiry call and put strikes into
      a single OI-change + volume-weighted premium change reading. Then classify
      each stock's options configuration using the same 4-quadrant matrix as Signal 5:

        OI ↑ + Premium ↑  →  Buying    (directional bet — paying UP for the option)
        OI ↑ + Premium ↓  →  Writing   (premium collection — taking the OPPOSITE side)
        OI ↓ + Premium ↑  →  SC        (writers buying back — options squeeze)
        OI ↓ + Premium ↓  →  LE        (buyers closing out)

    COMBINED BASKET CLASSIFICATION PER STOCK:
        C.Buy  + P.Write = Net Long  → institutions bullish (both sides confirm)
        C.Write + P.Buy  = Net Short → institutions bearish (both sides confirm)
        C.Buy  + P.Buy   = Straddle  → big move expected, no direction
        C.Write + P.Write = Range   → low vol / theta decay expected

    WHY THIS IS DIFFERENT FROM SIGNAL 5 (Index PCR / OI-Premium):
      Signal 5 analyses the options ON THE INDEX ITSELF (OPTIDX for NIFTY).
      This signal analyses the options on the 50 individual stocks that MAKE UP the index.
      When 30+ constituent stocks simultaneously show Call Buying across their own
      options chains, institutions are buying individual-stock upside exposure —
      the most specific expression of a bullish Nifty view possible.

    SCORING (based on basket net_long_pct vs net_short_pct):
      net ≥ 0.45  → +3.0  Exceptional: >45% more net-long stocks than net-short
      net ≥ 0.25  → +2.0  Strong basket bullish options configuration
      net ≥ 0.12  → +1.0  Mild bullish lean in constituent options
      net ≤ −0.45 → −3.0  Exceptional: institutional bearish across basket
      net ≤ −0.25 → −2.0  Strong basket bearish options
      net ≤ −0.12 → −1.0  Mild bearish lean
    """
    valid = ctx.n50_opt_valid
    if valid < 15:
        return None   # need at least 15 classifiable stocks

    nl   = ctx.n50_opt_net_long
    ns   = ctx.n50_opt_net_short
    ostr = ctx.n50_opt_straddle
    orng = ctx.n50_opt_range

    nl_pct = nl / valid
    ns_pct = ns / valid
    net    = nl_pct - ns_pct

    # Straddle dominance = big-move expectation, not directional
    straddle_dominant = ostr / valid >= 0.30

    if net >= 0.45:
        score = 3.0
        name  = "N50 Basket Options: Exceptional Call Buying / Put Writing"
        head  = (f"{nl}/{valid} N50 stocks Net Long options config "
                 f"({nl_pct:.0%} bull vs {ns_pct:.0%} bear, net {net:+.0%})")
        emoji = "🔥"
    elif net >= 0.25:
        score = 2.0
        name  = "N50 Basket Options: Broadly Bullish Configuration"
        head  = (f"{nl} net-long vs {ns} net-short stocks in options "
                 f"({nl_pct:.0%} vs {ns_pct:.0%}, net {net:+.0%})")
        emoji = "🟢"
    elif net >= 0.12:
        score = 1.0
        name  = "N50 Basket Options: Mild Bullish Lean"
        head  = f"N50 options basket: {nl_pct:.0%} bullish vs {ns_pct:.0%} bearish (mild bull)"
        emoji = "🟡"
    elif net <= -0.45:
        score = -3.0
        name  = "N50 Basket Options: Exceptional Put Buying / Call Writing"
        head  = (f"{ns}/{valid} N50 stocks Net Short options config "
                 f"({ns_pct:.0%} bear vs {nl_pct:.0%} bull, net {net:+.0%})")
        emoji = "💀"
    elif net <= -0.25:
        score = -2.0
        name  = "N50 Basket Options: Broadly Bearish Configuration"
        head  = (f"{ns} net-short vs {nl} net-long stocks in options "
                 f"({ns_pct:.0%} vs {nl_pct:.0%}, net {net:+.0%})")
        emoji = "🔴"
    elif net <= -0.12:
        score = -1.0
        name  = "N50 Basket Options: Mild Bearish Lean"
        head  = f"N50 options basket: {ns_pct:.0%} bearish vs {nl_pct:.0%} bullish (mild bear)"
        emoji = "🟠"
    else:
        return None   # net between -0.12 and +0.12: mixed, no directional edge

    # Straddle note: large straddle count means big-move expectations even on directional signal
    straddle_note = ""
    if straddle_dominant:
        straddle_note = (
            f" Note: {ostr}/{valid} stocks also show straddle config (both call + put buying) — "
            "options market expects high volatility regardless of direction."
        )

    direction = 1 if score > 0 else -1
    desc = (
        f"OI-Premium matrix applied to near-expiry options chains of {valid} Nifty 50 stocks:\n"
        f"  Net Long  (C.Buy/P.Write) : {nl:2d} ({nl_pct:.0%}) — both sides confirm bullish\n"
        f"  Net Short (C.Write/P.Buy) : {ns:2d} ({ns_pct:.0%}) — both sides confirm bearish\n"
        f"  Straddle  (C.Buy+P.Buy)   : {ostr:2d} — big move expected, no direction\n"
        f"  Range     (C.Write+P.Write): {orng:2d} — low vol / theta play\n"
        f"\n"
        f"OI-Premium matrix disambiguates PCR: a high PCR could mean put buying (bearish hedge) "
        f"OR put writing (bullish premium collection) — completely opposite signals from the "
        f"same raw number. By checking whether premium rises or falls WITH the OI change, "
        f"we determine WHO is driving the OI. "
        f"Applied to 50 constituent stocks individually, this reveals whether the options "
        f"activity is directional conviction or just mechanical hedging."
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
            "already priced in. Historical pattern: double-PCR spike = 68% 5-day bounce.", "🔄",
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
            "This divergence tells us the fear is short-horizon, not systemic. "
            "Pattern: weekly-fear / monthly-calm → bounce after weekly expiry in ~70% of cases.", "📉",
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
            f"option writers shift attention to monthly strikes after Thursday.", "⬇️",
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
        return IndexSignal(
            "Weekly Gamma Dominance — Range Compression (Nifty)", "Options OI", 0, -0.5,
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


def _aggregate_composite(signals: list[IndexSignal]) -> float:
    """Composite directional score with the collinear Institutional block capped.

    Non-institutional signals sum normally; the institutional signals are summed
    then clamped to ±_INSTITUTIONAL_CAP before being added. This removes the
    structural over-counting without discarding institutional information.
    """
    inst  = sum(s.score for s in signals if s.category == "Institutional")
    other = sum(s.score for s in signals if s.category != "Institutional")
    inst_capped = max(-_INSTITUTIONAL_CAP, min(_INSTITUTIONAL_CAP, inst))
    return float(other + inst_capped)


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_expected_move(
    idx_hist: pd.DataFrame,
    spot_close: Optional[float],
    vix_close: Optional[float],
    composite: float,
    dte: int,
    levels: IndexKeyLevels,
    is_nifty: bool,
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

    rets = closes.pct_change().dropna().tail(20)
    if len(rets) < 5:
        return out
    realized_sigma_pct = float(rets.std()) * 100.0           # daily 1σ, %

    vix_sigma_pct = None
    if vix_close and vix_close > 0:
        vix_sigma_pct = float(vix_close) / (252.0 ** 0.5)    # annualised → daily 1σ, %

    if vix_sigma_pct is not None:
        w_realized = 0.5 if is_nifty else 0.7
        sigma_pct  = w_realized * realized_sigma_pct + (1 - w_realized) * vix_sigma_pct
        basis_src  = (f"50/50 realized+VIX" if is_nifty else "70/30 realized+VIX")
    else:
        sigma_pct  = realized_sigma_pct
        basis_src  = "realized 20D vol"

    expected_move_pts = spot_close * sigma_pct / 100.0

    # Expiry-day gamma pinning compresses the expected range
    pin_note = ""
    if dte <= 1:
        expected_move_pts *= 0.70
        sigma_pct          *= 0.70
        pin_note = " · expiry pin ×0.7"

    # Data-driven sideways band: 40% of the 1σ move (matches the user's
    # "±40 pts = sideways, beyond = directional" intuition, but volatility-scaled).
    sideways_band = 0.40 * expected_move_pts

    # Directional target: conviction (composite/20, clamped ±1) × expected move
    conviction  = max(-1.0, min(1.0, composite / 20.0))
    target_move = conviction * expected_move_pts
    target_close = spot_close + target_move

    out.update(
        expected_move_pts=round(expected_move_pts, 0),
        expected_move_pct=round(sigma_pct, 2),
        range_low=round(spot_close - expected_move_pts, 0),
        range_high=round(spot_close + expected_move_pts, 0),
        target_move_pts=round(target_move, 0),
        target_close=round(target_close, 0),
        sideways_band_pts=round(sideways_band, 0),
        move_basis=(
            f"1σ ≈ {sigma_pct:.2f}% ({basis_src}{pin_note}); "
            f"68% of days close within ±{expected_move_pts:.0f} pts"
        ),
    )
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
    fii_covering    = any("Covering" in s.name and s.category == "Institutional" and s.direction == 1 for s in signals)
    vix_falling     = any(s.category == "Market Context" and s.direction > 0 for s in signals)
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

    # ── Score-driven verdict (23-signal system; max ≈ ±35) ───────────────────
    # Thresholds calibrated so HIGH requires ~6+ signals aligned
    if composite >= 12:
        risk = "FII short position remains key headwind." if fii_bearish else "Global macro reversal could override."
        return ("UP", _cap_confidence("HIGH"), "#00C853",
                f"Strong multi-signal bullish alignment (score {composite:+.1f})",
                primary_driver, risk)
    if composite >= 7:
        return ("UP", _cap_confidence("MEDIUM"), "#69F0AE",
                f"Moderate bullish tilt — bias is UP (score {composite:+.1f})",
                primary_driver,
                "Watch FII position for sustained follow-through." if fii_bearish else
                "Watch carry and PCR for confirmation.")
    if composite >= 3:
        squeeze_note = " Short squeeze risk elevated." if squeeze_setup else ""
        return ("UP", "LOW", "#B9F6CA",
                f"Mild bullish bias (score {composite:+.1f}){squeeze_note}",
                primary_driver, "Low conviction — tight stops if long.")
    if composite <= -12:
        return ("DOWN", _cap_confidence("HIGH"), "#FF5252",
                f"Strong multi-signal bearish alignment (score {composite:+.1f})",
                primary_driver, "Check oversold bounce / DII support floor before shorting.")
    if composite <= -7:
        return ("DOWN", _cap_confidence("MEDIUM"), "#FF6D00",
                f"Moderate bearish tilt — bias is DOWN (score {composite:+.1f})",
                primary_driver, "Mean-reverting market; deep scores can reverse sharply.")
    if composite <= -3:
        squeeze_note = " BUT short squeeze risk is real." if squeeze_setup else ""
        return ("DOWN", "LOW", "#FF8A65",
                f"Mild bearish bias (score {composite:+.1f}){squeeze_note}",
                primary_driver,
                "Short squeeze risk elevated — FII covering would reverse quickly." if squeeze_setup else
                "Watch for FII short covering as reversal trigger.")

    # Neutral zone — tiebreakers
    if squeeze_setup and broad_rally:
        return ("UP", "LOW", "#B9F6CA",
                "Neutral overall — but short squeeze + broad rally = upside bias",
                "Short squeeze setup: extreme PCR + HOD close + FII trapped",
                "FII can add more shorts — squeeze not guaranteed.")
    if oversold and vix_falling:
        return ("UP", "LOW", "#B9F6CA",
                "Neutral F&O — Oversold + falling VIX; mean-reversion bounce likely",
                "Oversold z-score + falling VIX (67% bounce probability)",
                "Oversold can persist — use stops.")
    if oversold:
        return ("UP", "LOW", "#B9F6CA",
                "Neutral F&O — oversold mean-reversion bounce likely",
                "Oversold 3D z-score (67% bounce probability, NSE backtest)",
                "Oversold can persist — use stops.")
    if fii_covering:
        return ("UP", "LOW", "#B9F6CA",
                "Neutral F&O — FII short covering is the key wildcard",
                "FII covering shorts creates buying pressure",
                "FII can pause or reverse coverage — monitor daily FAO data.")
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

    # ── Active futures (OI > 0) ───────────────────────────────────────────────
    near_expiry = fut_expiry = None
    days_to_expiry = 30
    futures_price = carry_pts = carry_pct_ann = None
    carry_label = "No Data"
    fut_oi = fut_oi_chg = 0

    fut_rows = fno_today[(fno_today["instrument"] == "FUTIDX") & (fno_today["open_interest"] > 0)].sort_values("expiry_date")
    if not fut_rows.empty:
        nr = fut_rows.iloc[0]
        fut_expiry     = _to_date(nr["expiry_date"])
        days_to_expiry = max((fut_expiry - trade_date).days, 0) if fut_expiry else 30
        futures_price  = float(nr["settle_price"]) if pd.notna(nr["settle_price"]) else None
        fut_oi         = int(nr["open_interest"])
        # Compute OI change: matched by SAME expiry_date (not rank) so rollover
        # days don't create false OI spikes by comparing different contracts.
        # When prev_futs is empty, the near contract is brand-new (just rolled);
        # is_rollover=True flags this so the signal is labeled correctly.
        fut_oi_chg  = 0
        is_rollover = False
        if not fno_prev.empty:
            prev_futs = fno_prev[
                (fno_prev["instrument"] == "FUTIDX") &
                (fno_prev["expiry_date"] == nr["expiry_date"])
            ]
            if not prev_futs.empty:
                prev_oi    = int(prev_futs["open_interest"].sum())
                fut_oi_chg = fut_oi - prev_oi
            else:
                # Same expiry_date not found in yesterday → contract just rolled
                is_rollover = True
        if spot_close and futures_price and days_to_expiry >= 3:
            carry_pts     = futures_price - spot_close
            carry_pct_ann = (carry_pts / spot_close) * (365.0 / max(days_to_expiry, 1)) * 100
            carry_label   = _carry_label(carry_pct_ann)
        elif futures_price and not spot_close:
            spot_close = futures_price

    # ── Options expiry (weekly, for PCR / max pain) ───────────────────────────
    opt_rows = fno_today[fno_today["instrument"] == "OPTIDX"].sort_values("expiry_date")
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
    near_is_monthly = False   # True during the ~4 days of the monthly expiry week

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

    near_expiry_display = fut_expiry or near_expiry

    # ── Snapshots ─────────────────────────────────────────────────────────────
    today_snap = _build_snapshot(trade_date, idx_hist, fno_today, near_expiry,
                                 spot_close, futures_price, carry_pts, carry_pct_ann)
    yesterday_snap = None
    if prev_fno_date and not fno_prev.empty:
        pr     = idx_hist[idx_hist["trade_date"] == prev_fno_date]
        pspot  = float(pr.iloc[0]["close_val"]) if not pr.empty and pd.notna(pr.iloc[0]["close_val"]) else None
        pfut_p = pcarry = pcarry_ann = None; popt_exp = None
        pfut   = fno_prev[(fno_prev["instrument"] == "FUTIDX") & (fno_prev["open_interest"] > 0)].sort_values("expiry_date")
        if not pfut.empty:
            pnr = pfut.iloc[0]; pne = _to_date(pnr["expiry_date"])
            pfut_p = float(pnr["settle_price"]) if pd.notna(pnr["settle_price"]) else None
            if pspot and pfut_p and pne:
                pT = max((pne - prev_fno_date).days, 1) / 365.0
                if pT > 3 / 365:
                    pcarry = pfut_p - pspot
                    pcarry_ann = (pcarry / pspot) * (1.0 / pT) * 100
        por = fno_prev[fno_prev["instrument"] == "OPTIDX"].sort_values("expiry_date")
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
    sigs.append(_sig_oi_price_matrix(day_change_pct, fut_oi_chg, fut_oi, is_rollover=is_rollover))
    add(_sig_carry(carry_pct_ann, carry_pts))
    add(_sig_pcr(pcr, yesterday_snap.pcr if yesterday_snap else None))
    add(_sig_opt_oi_premium(_opt_near_t, _opt_near_p))   # Signal 5 — OI-Premium matrix
    if spot_close and dte_options <= 5:
        add(_sig_max_pain(levels.max_pain, spot_close, dte_options))
    add(_sig_range_position(spot_close, high_val, low_val, day_change_pct))

    # — Signals 7-13: FII / institutional (all use MarketContext) —
    add(_sig_fii_institutional(market_ctx))
    add(_sig_fii_options_delta(market_ctx))
    add(_sig_fii_flow(market_ctx, fno_symbol))
    add(_sig_fii_cumulative_flow(market_ctx, fno_symbol))
    add(_sig_fii_oi_buildup(market_ctx, fno_symbol))
    add(_sig_fii_position_change(market_ctx))
    add(_sig_short_squeeze_setup(market_ctx, fno_symbol, day_change_pct, range_pos, pcr))

    # — Signals 14-17: market context —
    add(_sig_vix_regime(market_ctx))
    add(_sig_sector_breadth(market_ctx))
    add(_sig_defensive_cyclical(market_ctx))
    add(_sig_valuation_pe(market_ctx))
    # Signal 17b: sector RS vs Nifty — fires for BankNifty / FinNifty / MidcapNifty only
    add(_sig_index_relative_strength(fno_symbol, day_change_pct, market_ctx))
    # Signals 17c/d/e: Nifty 50 constituent data — NIFTY only
    if fno_symbol == "NIFTY":
        add(_sig_constituent_breadth(market_ctx))       # 17c cash breadth + delivery
        add(_sig_constituent_fut_oi(market_ctx))         # 17d stock futures OI-Price
        add(_sig_constituent_opt_oi_prem(market_ctx))    # 17e stock options OI-Premium

    # — Signals 18-20: Nifty 50 multi-expiry (fires only when weekly ≠ monthly) —
    if fno_symbol == "NIFTY" and monthly_exp_me is not None:
        dte_monthly = (monthly_exp_me - trade_date).days
        # PCR divergence requires a genuine weekly vs monthly distinction.
        # When near_expiry IS the monthly (last ~4 days of the month), "monthly_exp_me"
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
        add(_sig_entropy_regime(regime))
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
    )

    pred_out = IndexPrediction(
        fno_symbol=fno_symbol, display_name=display_name, as_of_date=trade_date,
        spot_close=spot_close, prev_close=prev_close_price,
        day_change_pct=day_change_pct, high=high_val, low=low_val,
        near_expiry=near_expiry_display, days_to_expiry=days_to_expiry,
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
    )

    # Auto-store prediction in memory engine (non-fatal).
    # Temporarily set composite_score = composite_prelim before storing so that
    # feat_oi_score in the DB fingerprint uses the SAME value that was used in the
    # similarity query (_FeatProxy.composite_score = composite_prelim). Without this,
    # the stored feat_oi_score reflects the final composite (including Signal 24 ±2.5)
    # while every future query uses the pre-memory composite — a systematic mismatch.
    try:
        from src.analytics.memory_engine import store_prediction
        pred_out.composite_score = composite_prelim
        store_prediction(pred_out, trade_date)
        pred_out.composite_score = composite   # restore final for the caller
    except Exception:
        pred_out.composite_score = composite   # always restore on error path

    return pred_out
