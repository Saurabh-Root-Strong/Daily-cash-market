"""
Market Intelligence Engine — pattern-based quant signals from F&O participant data.

Methodology (world-class quant research principles):
  1. OI-Price Context  — classify today's Nifty move vs FII OI change
     (Fresh Long / Short Covering / Long Unwinding / Fresh Short)
  2. FII Futures Trend — z-score of 5D position change vs 30D baseline
  3. FII Volume Spike  — z-score of today's FII activity vs 20D average;
     combined with OI direction to classify as Accumulation vs Distribution
  4. PCR Trend         — 3D put-call ratio direction; contrarian at extremes
  5. Options Stance    — FII options delta shift over 5 days
  6. FII-DII Alignment — both buying = strong; divergence = caution
  7. Retail Contrarian — client extreme positioning (z-score vs 30D)
  8. FII Money Flow    — if fii_derivatives_stats available: net Rs. Cr in Index F&O

Composite score: weighted sum → market view label + bias reasoning.

Signal scoring:
  strength 3 → ±3 pts   strength 2 → ±2 pts   strength 1 → ±1 pt
  Total range: ±18 (all signals max, with FII stats) or ±14 (without)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_market_intelligence", "MarketIntelligence", "Signal", "FIITodaySnapshot"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Signal:
    name: str
    category: str           # "Futures OI" | "Options OI" | "Volume" | "PCR" | "Retail" | "FII Flow"
    direction: int          # +1 bullish, -1 bearish, 0 neutral
    strength: int           # 1 weak, 2 moderate, 3 strong
    score: int              # direction * strength
    headline: str           # short label shown in dashboard badge
    description: str        # full explanation
    emoji: str = ""


@dataclass
class WeeklyExpiryView:
    days_to_expiry: int
    expiry_date: date
    pcr_at_expiry_window: Optional[float]
    fii_net_oi_direction: str           # "Long" | "Short" | "Flat"
    bias: str
    reasoning: str


@dataclass
class TomorrowVerdict:
    direction: str        # "UP" | "DOWN" | "SIDEWAYS"
    confidence: str       # "HIGH" | "MEDIUM" | "LOW"
    headline: str         # one-line plain-English call
    key_driver: str       # the single most impactful signal
    key_risk: str         # what would invalidate this view
    direction_color: str  # hex colour for the direction label
    squeeze_risk: bool = False           # FII massively short + DII floor = squeeze fuel
    short_covering_active: bool = False  # FII is actively reducing large short (= net buying)


@dataclass
class FIITodaySnapshot:
    """
    Structured view of what FIIs actually DID today — all three data layers combined.
    Used for display in the dashboard; does not contribute to composite score.

    Combines:
      • OI layer   — gross long/short changes vs yesterday (absolute position moves)
      • Volume layer — today's trading activity vs baseline (conviction signal)
      • Rupee flow  — ₹Cr net from fii_derivatives_stats (real money confirmation)
    """
    as_of_date: date
    # OI layer (futures only — directional signal)
    long_change: int            # fut_idx_long today − yesterday (+ve = added longs)
    short_change: int           # fut_idx_short today − yesterday (+ve = added shorts)
    oi_net_change: int          # long_change − short_change (+ve = bullish)
    cumulative_long: int        # current total open long contracts
    cumulative_short: int       # current total open short contracts
    cumulative_net: int         # current net = cumulative_long − cumulative_short
    # Volume layer
    vol_today: int              # total activity contracts today
    vol_z: float                # z-score vs 20D baseline (0.0 if < 5 data points)
    # Rupee flow (from fii_derivatives_stats; 0.0 if not available)
    fut_net_cr: float           # index futures rupee net (buy−sell)
    opt_net_cr: float           # index options rupee net
    total_net_cr: float         # total index F&O net ₹Cr
    # Classification
    action_label: str           # "SHORT COVERING" | "ADDING LONGS" | "ADDING SHORTS" | etc.
    action_color: str           # hex
    action_emoji: str
    covering_pct: float = 0.0  # if short: % of total short covered today


@dataclass
class MarketIntelligence:
    as_of_date: date
    composite_score: float
    market_view: str            # label
    view_color: str             # hex color
    bias_reasoning: str         # 2-3 sentence plain-English summary
    signals: list[Signal] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    weekly_expiry: Optional[WeeklyExpiryView] = None
    fii_flow_available: bool = False
    tomorrow_verdict: Optional[TomorrowVerdict] = None
    fii_today: Optional[FIITodaySnapshot] = None


# ── Helper: next Thursday expiry ─────────────────────────────────────────────

def _next_thursday(from_date: date) -> date:
    """Next Thursday (weekly expiry) strictly after from_date."""
    days = (3 - from_date.weekday()) % 7
    if days == 0:
        days = 7
    return from_date + timedelta(days=days)


# ── Helper: safe z-score ──────────────────────────────────────────────────────

def _zscore(value: float, series: pd.Series) -> float:
    std = float(series.std())
    mean = float(series.mean())
    if std < 1e-9:
        return 0.0
    return (value - mean) / std


# ── Main entry point ──────────────────────────────────────────────────────────

def get_market_intelligence(
    as_of_date: date,
    lookback_days: int = 45,
) -> MarketIntelligence:
    """
    Compute all signals and return a MarketIntelligence object.

    lookback_days: how far back to load data for baseline statistics.
    """
    start = as_of_date - timedelta(days=lookback_days * 2)  # extra buffer for weekends/holidays

    oi_df  = _load_fao(start, as_of_date, "OI")
    vol_df = _load_fao(start, as_of_date, "Vol")
    idx_df = _load_nifty(start, as_of_date)
    fii_stats_df = _load_fii_stats(start, as_of_date)

    signals: list[Signal] = []
    alerts:  list[str]    = []

    if oi_df.empty:
        return MarketIntelligence(
            as_of_date=as_of_date,
            composite_score=0,
            market_view="No Data",
            view_color="#888888",
            bias_reasoning="Insufficient F&O participant data. Run backfill first.",
        )

    # ── Signal 1: OI-Price Context ────────────────────────────────────────────
    sig = _signal_oi_price_context(oi_df, idx_df, as_of_date)
    if sig:
        signals.append(sig)

    # ── Signal 2: FII Futures Trend (5D position change z-score) ─────────────
    sig = _signal_fii_futures_trend(oi_df, as_of_date)
    if sig:
        signals.append(sig)

    # ── Signal 3: Consecutive FII Positioning Pattern ─────────────────────────
    sig = _signal_consecutive_pattern(oi_df, as_of_date)
    if sig:
        signals.append(sig)

    # ── Signal 4: FII-DII Alignment ──────────────────────────────────────────
    sig = _signal_fii_dii_alignment(oi_df, as_of_date)
    if sig:
        signals.append(sig)

    # ── Signal 5: FII Volume Spike Detection ─────────────────────────────────
    sig, alert = _signal_volume_spike(vol_df, oi_df, as_of_date)
    if sig:
        signals.append(sig)
    if alert:
        alerts.append(alert)

    # ── Signal 6: PCR Trend ───────────────────────────────────────────────────
    sig = _signal_pcr_trend(oi_df, as_of_date)
    if sig:
        signals.append(sig)

    # ── Signal 7: FII Options Stance Shift ───────────────────────────────────
    sig = _signal_options_stance(oi_df, as_of_date)
    if sig:
        signals.append(sig)

    # ── Signal 8: Retail Contrarian ───────────────────────────────────────────
    sig = _signal_retail_contrarian(oi_df, as_of_date)
    if sig:
        signals.append(sig)

    # ── Signal 9: FII Money Flow (fii_derivatives_stats) ────────────────────
    fii_flow_available = not fii_stats_df.empty
    if fii_flow_available:
        sig = _signal_fii_money_flow(fii_stats_df, as_of_date)
        if sig:
            signals.append(sig)

    # ── PCR Extreme Alert ─────────────────────────────────────────────────────
    pcr_alert = _pcr_extreme_alert(oi_df, as_of_date)
    if pcr_alert:
        alerts.append(pcr_alert)

    # ── Composite Score & Market View ─────────────────────────────────────────
    composite = float(sum(s.score for s in signals))
    market_view, view_color = _score_to_view(composite)
    bias_reasoning = _build_reasoning(signals, composite, as_of_date)

    # Sort signals by impact (highest |score| first) so UI shows priority order
    signals.sort(key=lambda s: abs(s.score), reverse=True)

    # ── Weekly Expiry View ────────────────────────────────────────────────────
    weekly = _weekly_expiry_view(oi_df, as_of_date)

    # ── Tomorrow's Verdict ────────────────────────────────────────────────────
    verdict = _compute_tomorrow_verdict(signals, composite, weekly, oi_df)

    # ── Squeeze / Short Covering Alert ────────────────────────────────────────
    if verdict and not oi_df.empty:
        today_oi_chk = oi_df[oi_df["trade_date"] == oi_df["trade_date"].max()]
        fii_r = today_oi_chk[today_oi_chk["client_type"] == "FII"]
        if not fii_r.empty:
            fn = float(fii_r.iloc[0]["fut_idx_net"])
            if verdict.short_covering_active:
                alerts.append(
                    f"SHORT COVERING ACTIVE: FII net {fn:+,.0f} contracts and REDUCING. "
                    "Covering = forced buying = near-term bullish. Watch for acceleration."
                )
            elif verdict.squeeze_risk:
                alerts.append(
                    f"SQUEEZE RISK: FII holds {fn:+,.0f} massive short. "
                    "Any positive catalyst forces covering = sudden UP move. "
                    "Do not initiate fresh shorts — squeeze risk elevated."
                )

    fii_today = _compute_fii_today_snapshot(oi_df, vol_df, fii_stats_df)

    return MarketIntelligence(
        as_of_date=as_of_date,
        composite_score=composite,
        market_view=market_view,
        view_color=view_color,
        bias_reasoning=bias_reasoning,
        signals=signals,
        alerts=alerts,
        weekly_expiry=weekly,
        fii_flow_available=fii_flow_available,
        tomorrow_verdict=verdict,
        fii_today=fii_today,
    )


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_fao(start: date, end: date, data_type: str) -> pd.DataFrame:
    return query_dataframe("""
        SELECT
            trade_date, client_type,
            fut_idx_long, fut_idx_short,
            fut_idx_long - fut_idx_short AS fut_idx_net,
            opt_idx_call_long, opt_idx_call_short,
            opt_idx_put_long,  opt_idx_put_short,
            opt_idx_call_long - opt_idx_call_short AS opt_call_net,
            opt_idx_put_long  - opt_idx_put_short  AS opt_put_net,
            (opt_idx_call_long - opt_idx_call_short)
            - (opt_idx_put_long - opt_idx_put_short) AS opt_delta,
            total_long, total_short,
            total_long + total_short AS total_activity
        FROM fao_participant
        WHERE data_type = ?
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date, client_type
    """, [data_type, start, end])


def _load_nifty(start: date, end: date) -> pd.DataFrame:
    return query_dataframe("""
        SELECT trade_date, close_val, pct_chg
        FROM index_data
        WHERE index_name = 'NIFTY 50'
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date
    """, [start, end])


def _load_fii_stats(start: date, end: date) -> pd.DataFrame:
    return query_dataframe("""
        SELECT
            trade_date, category,
            buy_contracts, sell_contracts,
            buy_contracts - sell_contracts AS net_contracts,
            buy_value_cr, sell_value_cr,
            buy_value_cr - sell_value_cr   AS net_value_cr,
            oi_contracts
        FROM fii_derivatives_stats
        WHERE trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date, category
    """, [start, end])


# ── Individual signal functions ───────────────────────────────────────────────

def _fii_oi_series(oi_df: pd.DataFrame) -> pd.Series:
    """FII daily fut_idx_net series, sorted by date, indexed by trade_date."""
    fii = (
        oi_df[oi_df["client_type"] == "FII"]
        .sort_values("trade_date")
        .set_index("trade_date")["fut_idx_net"]
    )
    return fii


def _signal_oi_price_context(
    oi_df: pd.DataFrame, idx_df: pd.DataFrame, as_of_date: date
) -> Optional[Signal]:
    """
    OI-Price Matrix (most reliable institutional signal):
      Price UP  + FII OI UP   → Fresh Long  (bullish, new money in)
      Price UP  + FII OI DOWN → Short Cover (bullish but weak, positions closing)
      Price DOWN + FII OI DOWN → Long Unwind (bearish but limited, profit-taking)
      Price DOWN + FII OI UP   → Fresh Short (strongly bearish, new shorts)
    """
    fii = _fii_oi_series(oi_df)
    if len(fii) < 2 or idx_df.empty:
        return None

    nifty = idx_df.sort_values("trade_date").set_index("trade_date")

    # Get last common date
    common_dates = sorted(set(fii.index) & set(nifty.index))
    if len(common_dates) < 2:
        return None

    today_d  = common_dates[-1]
    prev_d   = common_dates[-2]

    price_up = float(nifty.loc[today_d, "pct_chg"]) > 0.10
    price_dn = float(nifty.loc[today_d, "pct_chg"]) < -0.10
    oi_up    = float(fii.loc[today_d]) > float(fii.loc[prev_d])
    oi_dn    = float(fii.loc[today_d]) < float(fii.loc[prev_d])
    pct      = float(nifty.loc[today_d, "pct_chg"])

    if price_up and oi_up:
        return Signal(
            name="OI-Price: Fresh Long",
            category="Futures OI",
            direction=1, strength=3, score=3,
            headline="Fresh Long Build",
            description=(
                f"Nifty {pct:+.2f}% AND FII Index Futures OI rising. "
                "New money entering longs — the strongest bullish confirmation. "
                "FII are not just covering shorts; they are adding fresh bullish exposure."
            ),
            emoji="🟢",
        )
    elif price_up and oi_dn:
        return Signal(
            name="OI-Price: Short Covering",
            category="Futures OI",
            direction=1, strength=1, score=1,
            headline="Short Covering Rally",
            description=(
                f"Nifty {pct:+.2f}% but FII Index Futures OI falling. "
                "Rally is driven by shorts closing, not fresh longs. "
                "Momentum is real but may lack follow-through — watch for OI to start rising."
            ),
            emoji="🟡",
        )
    elif price_dn and oi_dn:
        return Signal(
            name="OI-Price: Long Unwinding",
            category="Futures OI",
            direction=-1, strength=1, score=-1,
            headline="Long Unwinding",
            description=(
                f"Nifty {pct:+.2f}% and FII Index Futures OI falling. "
                "Existing longs are being closed — profit-taking or stop-losses. "
                "Bearish but self-limiting; once positions are cleaned up, selling pressure reduces."
            ),
            emoji="🟡",
        )
    elif price_dn and oi_up:
        return Signal(
            name="OI-Price: Fresh Short",
            category="Futures OI",
            direction=-1, strength=3, score=-3,
            headline="Fresh Short Build",
            description=(
                f"Nifty {pct:+.2f}% AND FII Index Futures OI rising (short side). "
                "New bearish positions being added into the decline — most dangerous signal. "
                "FII are actively betting on further downside, not just closing longs."
            ),
            emoji="🔴",
        )
    else:
        # Flat price — OI context inconclusive
        return Signal(
            name="OI-Price: Sideways",
            category="Futures OI",
            direction=0, strength=1, score=0,
            headline="Sideways / Indecisive",
            description=(
                f"Nifty {pct:+.2f}% — price movement too small for OI context to be definitive. "
                "Market awaiting a catalyst. Watch for OI buildup direction to set next move."
            ),
            emoji="⚪",
        )


def _signal_fii_futures_trend(oi_df: pd.DataFrame, as_of_date: date) -> Optional[Signal]:
    """
    Z-score of FII 5D net position change vs 30D rolling baseline.
    High z-score = unusual accumulation/distribution in progress.
    """
    fii = _fii_oi_series(oi_df)
    if len(fii) < 8:
        return None

    today_net = float(fii.iloc[-1])
    net_5d_ago = float(fii.iloc[-6]) if len(fii) >= 6 else float(fii.iloc[0])
    delta_5d = today_net - net_5d_ago

    # Rolling 5D deltas for baseline
    baseline_deltas = fii.diff(5).dropna()
    if len(baseline_deltas) < 5:
        return None

    z = _zscore(delta_5d, baseline_deltas)

    if z >= 2.0:
        still_short = today_net < -20_000
        label = "Aggressive Short Covering" if still_short else "Aggressive Long Build"
        desc = (
            f"FII net {today_net:+,.0f} contracts (still SHORT) but COVERED "
            f"{delta_5d:+,.0f} contracts in 5 days (z-score: {z:.1f}σ above normal). "
            "Unusually intense short covering = massive net BUYING. "
            "This is the most powerful near-term bullish signal: FIIs are FORCED to buy "
            "to close their short positions. Rally is self-reinforcing until shorts are covered."
            if still_short else
            f"FII Index Futures net position changed by {delta_5d:+,.0f} contracts in 5 days "
            f"(z-score: {z:.1f}σ above normal). Unusual intensity of long accumulation — "
            "historically this precedes sustained up-moves. Smart money is building "
            "large directional exposure at this level."
        )
        return Signal(
            name=f"FII Futures: {label}",
            category="Futures OI",
            direction=1, strength=3, score=3,
            headline=label,
            description=desc,
            emoji="⚡" if still_short else "🔥",
        )
    elif z >= 1.0:
        still_short = today_net < -20_000
        label = "Short Covering in Progress" if still_short else "Steady Long Addition"
        desc = (
            f"FII net {today_net:+,.0f} contracts (still SHORT), "
            f"but REDUCED position by {delta_5d:+,.0f} contracts over 5 days (z: {z:.1f}σ). "
            "SHORT COVERING = net BUY transactions = near-term bullish trigger. "
            "IMPORTANT: The direction of change matters more than the absolute level. "
            "FII is buying (to close shorts) — this drives market UP even while positioning stays bearish."
            if still_short else
            f"FII Index Futures net +{delta_5d:+,.0f} contracts over 5D (z: {z:.1f}σ). "
            "Above-average pace of long building — conviction improving. "
            "Not yet extreme but trend is clearly constructive."
        )
        return Signal(
            name=f"FII Futures: {label}",
            category="Futures OI",
            direction=1, strength=2, score=2,
            headline=label,
            description=desc,
            emoji="🔄" if still_short else "🟢",
        )
    elif z <= -2.0:
        return Signal(
            name="FII Futures: Aggressive Distribution",
            category="Futures OI",
            direction=-1, strength=3, score=-3,
            headline="Aggressive Short Build",
            description=(
                f"FII Index Futures net changed {delta_5d:+,.0f} contracts in 5 days "
                f"(z-score: {z:.1f}σ below normal). Unusual intensity of short build/long exit. "
                "Institutional players are aggressively reducing bullish exposure or actively "
                "building directional shorts — high-conviction bearish signal."
            ),
            emoji="💀",
        )
    elif z <= -1.0:
        return Signal(
            name="FII Futures: Steady Distribution",
            category="Futures OI",
            direction=-1, strength=2, score=-2,
            headline="Position Unwinding",
            description=(
                f"FII Index Futures net {delta_5d:+,.0f} contracts over 5D (z: {z:.1f}σ). "
                "Above-average pace of position reduction — caution warranted. "
                "Could be profit-taking or early stage of repositioning to short."
            ),
            emoji="🔴",
        )
    else:
        return Signal(
            name="FII Futures: Normal Activity",
            category="Futures OI",
            direction=0, strength=1, score=0,
            headline="Normal Positioning",
            description=(
                f"FII 5D futures net change: {delta_5d:+,.0f} contracts (z: {z:.1f}σ). "
                "Within normal range — no unusual accumulation or distribution pattern. "
                "Market in wait-and-see mode."
            ),
            emoji="⚪",
        )


def _signal_consecutive_pattern(oi_df: pd.DataFrame, as_of_date: date) -> Optional[Signal]:
    """Detect 3+ consecutive days of FII position change in one direction."""
    fii = _fii_oi_series(oi_df)
    if len(fii) < 4:
        return None

    daily_changes = fii.diff().tail(4).dropna()
    if len(daily_changes) < 3:
        return None

    last3 = daily_changes.tail(3)
    all_positive = all(v > 100 for v in last3)
    all_negative = all(v < -100 for v in last3)
    total = float(last3.sum())

    if all_positive:
        return Signal(
            name="Pattern: 3D Consecutive Long Build",
            category="Futures OI",
            direction=1, strength=2, score=2,
            headline="3-Day Long Streak",
            description=(
                f"FII has added to Index Futures long exposure for 3 consecutive days "
                f"(total: {total:+,.0f} contracts). Sustained accumulation — not a one-day event. "
                "Institutional players are systematically building a long position, "
                "suggesting conviction in near-term upside."
            ),
            emoji="📈",
        )
    elif all_negative:
        return Signal(
            name="Pattern: 3D Consecutive Short Build",
            category="Futures OI",
            direction=-1, strength=2, score=-2,
            headline="3-Day Short Streak",
            description=(
                f"FII has been reducing/shorting Index Futures for 3 consecutive days "
                f"(total: {total:+,.0f} contracts). Systematic distribution phase — "
                "not noise but a deliberate directional stance. Risk-off signal."
            ),
            emoji="📉",
        )
    return None


def _signal_fii_dii_alignment(oi_df: pd.DataFrame, as_of_date: date) -> Optional[Signal]:
    """Alignment between FII and DII positioning."""
    today_oi = oi_df[oi_df["trade_date"] == oi_df["trade_date"].max()]
    if today_oi.empty:
        return None

    fii_row = today_oi[today_oi["client_type"] == "FII"]
    dii_row = today_oi[today_oi["client_type"] == "DII"]
    if fii_row.empty or dii_row.empty:
        return None

    fii_net = float(fii_row.iloc[0]["fut_idx_net"])
    dii_net = float(dii_row.iloc[0]["fut_idx_net"])

    if fii_net > 0 and dii_net > 0:
        return Signal(
            name="Alignment: FII+DII Both Long",
            category="Futures OI",
            direction=1, strength=3, score=3,
            headline="FII+DII Both Long",
            description=(
                f"Both FII (+{fii_net:,.0f}) and DII (+{dii_net:,.0f}) are net long Index Futures. "
                "This is the highest-conviction bullish setup. Foreign and domestic institutions "
                "rarely agree on direction — when they do, the move tends to be sustained."
            ),
            emoji="🤝",
        )
    elif fii_net < 0 and dii_net < 0:
        return Signal(
            name="Alignment: FII+DII Both Short",
            category="Futures OI",
            direction=-1, strength=3, score=-3,
            headline="FII+DII Both Short",
            description=(
                f"Both FII ({fii_net:,.0f}) and DII ({dii_net:,.0f}) are net short Index Futures. "
                "Rare and bearish — when both institutional camps align on the short side, "
                "downside risk is elevated. Avoid leveraged long exposure."
            ),
            emoji="⚠️",
        )
    elif fii_net < 0 and dii_net > 0:
        is_squeeze = fii_net < -75_000 and dii_net > 20_000
        return Signal(
            name="Alignment: FII Short, DII Support" + (" ⚡ SQUEEZE SETUP" if is_squeeze else ""),
            category="Futures OI",
            direction=1 if is_squeeze else 0,
            strength=2 if is_squeeze else 1,
            score=1 if is_squeeze else 0,
            headline="FII Short + DII Floor = SQUEEZE SETUP" if is_squeeze else "FII Short / DII Support",
            description=(
                f"FII massively short ({fii_net:+,.0f}) while DII is actively buying ({dii_net:+,.0f}). "
                "⚡ SQUEEZE SETUP: DII creates a floor that prevents the bearish thesis from playing out. "
                "FII cannot push market down while DII keeps buying. "
                "Any positive catalyst forces FII to cover shorts (= BUY) → sudden sharp UP move. "
                "CRITICAL: This setup produces the sharpest short-term rallies."
                if is_squeeze else
                f"FII net short ({fii_net:,.0f}) but DII net long ({dii_net:+,.0f}). "
                "DII (LIC, MFs) typically buy on dips to deploy domestic inflows. "
                "Market range-bound — FII selling creates ceiling, DII buying creates floor."
            ),
            emoji="⚡" if is_squeeze else "⚖️",
        )
    else:
        return Signal(
            name="Alignment: FII Long, DII Hedging",
            category="Futures OI",
            direction=1, strength=1, score=1,
            headline="FII Long / DII Hedged",
            description=(
                f"FII net long ({fii_net:+,.0f}) while DII net short ({dii_net:,.0f}). "
                "DII short is typically hedging their large equity portfolio (not a bearish bet). "
                "FII is the directional driver here — net positive signal."
            ),
            emoji="🟩",
        )


def _signal_volume_spike(
    vol_df: pd.DataFrame, oi_df: pd.DataFrame, as_of_date: date
) -> tuple[Optional[Signal], Optional[str]]:
    """FII volume z-score. If spike + OI direction → classify as accumulation/distribution."""
    if vol_df.empty:
        return None, None

    fii_vol = (
        vol_df[vol_df["client_type"] == "FII"]
        .sort_values("trade_date")
        .set_index("trade_date")["total_activity"]
    )
    if len(fii_vol) < 5:
        return None, None

    today_vol = float(fii_vol.iloc[-1])
    baseline  = fii_vol.iloc[:-1]
    z = _zscore(today_vol, baseline)

    if abs(z) < 1.8:
        return None, None

    # Determine OI direction on same day
    fii_oi = _fii_oi_series(oi_df)
    oi_direction = 0
    today_d = fii_vol.index[-1]
    if today_d in fii_oi.index and len(fii_oi) >= 2:
        prev_d = fii_oi.index[fii_oi.index.get_loc(today_d) - 1]
        oi_change = float(fii_oi.loc[today_d]) - float(fii_oi.loc[prev_d])
        oi_direction = 1 if oi_change > 0 else (-1 if oi_change < 0 else 0)

    vol_k = today_vol / 1000
    alert = None

    if z >= 2.5:
        alert = (
            f"VOLUME ALERT: FII activity {today_vol:,.0f} contracts "
            f"({z:.1f}σ above 20D avg) — potential major position change"
        )

    if z >= 1.8 and oi_direction > 0:
        sig = Signal(
            name="Volume: FII Accumulation",
            category="Volume",
            direction=1, strength=2 if z < 2.5 else 3, score=2 if z < 2.5 else 3,
            headline=f"FII Accumulation ({z:.1f}σ spike)",
            description=(
                f"FII traded {today_vol:,.0f} contracts today ({z:.1f}σ above normal). "
                "OI also increased — these are NEW positions being built, not old ones closing. "
                "High-volume OI increase = institutional accumulation, not noise."
            ),
            emoji="🚨",
        )
        return sig, alert
    elif z >= 1.8 and oi_direction < 0:
        sig = Signal(
            name="Volume: FII Distribution",
            category="Volume",
            direction=-1, strength=2 if z < 2.5 else 3, score=-(2 if z < 2.5 else 3),
            headline=f"FII Distribution ({z:.1f}σ spike)",
            description=(
                f"FII traded {today_vol:,.0f} contracts today ({z:.1f}σ above normal). "
                "OI fell on same day — existing positions being CLOSED in bulk. "
                "Large-scale position exits create follow-through pressure."
            ),
            emoji="🚨",
        )
        return sig, alert
    elif z >= 1.8:
        sig = Signal(
            name="Volume: FII Activity Spike",
            category="Volume",
            direction=0, strength=1, score=0,
            headline=f"FII Activity Spike ({z:.1f}σ)",
            description=(
                f"Unusual FII volume ({today_vol:,.0f} contracts, {z:.1f}σ above baseline). "
                "OI direction unclear — could be rolling positions or mixed activity. "
                "Watch next 1-2 days for OI to reveal the direction."
            ),
            emoji="👀",
        )
        return sig, alert

    return None, None


def _signal_pcr_trend(oi_df: pd.DataFrame, as_of_date: date) -> Optional[Signal]:
    """Put-Call Ratio level and 3D trend direction."""
    daily_pcr = (
        oi_df.groupby("trade_date")[["opt_idx_put_long", "opt_idx_call_long"]]
        .sum()
        .reset_index()
        .sort_values("trade_date")
    )
    daily_pcr["pcr"] = (
        daily_pcr["opt_idx_put_long"]
        / daily_pcr["opt_idx_call_long"].replace(0, float("nan"))
    )
    daily_pcr = daily_pcr.dropna(subset=["pcr"])

    if len(daily_pcr) < 4:
        return None

    current_pcr = float(daily_pcr["pcr"].iloc[-1])
    pcr_3d_ago  = float(daily_pcr["pcr"].iloc[-4])
    pcr_rising  = current_pcr > pcr_3d_ago

    if current_pcr > 1.25 and pcr_rising:
        return Signal(
            name="PCR: Extreme Put Build (Contrarian Bullish)",
            category="PCR",
            direction=1, strength=2, score=2,
            headline=f"PCR {current_pcr:.2f} — Contrarian BUY",
            description=(
                f"PCR at {current_pcr:.2f} and rising. When put-buying reaches extremes, "
                "it signals peak fear/hedging — the market has already priced in the downside. "
                "Contrarian signal: markets rarely fall when everyone is already protected. "
                "Historically PCR > 1.25 marks short-term bottoms."
            ),
            emoji="🔄",
        )
    elif current_pcr > 1.15:
        return Signal(
            name="PCR: Elevated Put Activity",
            category="PCR",
            direction=1, strength=1, score=1,
            headline=f"PCR {current_pcr:.2f} — Mildly Bullish",
            description=(
                f"PCR at {current_pcr:.2f} — above neutral zone (0.8–1.1). "
                "Elevated put activity suggests defensive positioning. "
                "Not at extreme yet, but building toward a contrarian buy setup."
            ),
            emoji="🟡",
        )
    elif current_pcr < 0.72 and not pcr_rising:
        return Signal(
            name="PCR: Call Euphoria (Contrarian Bearish)",
            category="PCR",
            direction=-1, strength=2, score=-2,
            headline=f"PCR {current_pcr:.2f} — Contrarian SELL",
            description=(
                f"PCR at {current_pcr:.2f} and falling. Excessive call-buying signals complacency. "
                "When retail rushes to buy calls at highs, smart money sells into the euphoria. "
                "PCR < 0.72 historically precedes short-term corrections."
            ),
            emoji="🔄",
        )
    elif current_pcr < 0.82:
        return Signal(
            name="PCR: Low Put Activity (Caution)",
            category="PCR",
            direction=-1, strength=1, score=-1,
            headline=f"PCR {current_pcr:.2f} — Mild Caution",
            description=(
                f"PCR at {current_pcr:.2f} — below neutral. Participants under-hedged. "
                "Not extreme yet, but drift toward complacency territory."
            ),
            emoji="🟡",
        )
    else:
        return Signal(
            name="PCR: Neutral",
            category="PCR",
            direction=0, strength=1, score=0,
            headline=f"PCR {current_pcr:.2f} — Neutral",
            description=(
                f"PCR at {current_pcr:.2f} — in neutral zone (0.82–1.15). "
                "No strong contrarian signal from options market. Market balanced."
            ),
            emoji="⚪",
        )


def _signal_options_stance(oi_df: pd.DataFrame, as_of_date: date) -> Optional[Signal]:
    """FII options delta shift over 5 days (call_net - put_net)."""
    fii = oi_df[oi_df["client_type"] == "FII"].sort_values("trade_date")
    if len(fii) < 6:
        return None

    today_delta = float(fii.iloc[-1]["opt_delta"])
    delta_5d_ago = float(fii.iloc[-6]["opt_delta"]) if len(fii) >= 6 else 0
    shift = today_delta - delta_5d_ago

    historical_shifts = fii["opt_delta"].diff(5).dropna()
    if len(historical_shifts) < 3:
        return None
    z = _zscore(shift, historical_shifts)

    if today_delta > 0 and z > 0.8:
        return Signal(
            name="Options: FII Bullish Stance Strengthening",
            category="Options OI",
            direction=1, strength=2, score=2,
            headline="FII Options: Bullish Shift",
            description=(
                f"FII options delta (Call Net − Put Net) = {today_delta:+,.0f} and improving "
                f"(5D shift: {shift:+,.0f}, z: {z:.1f}σ). "
                "FII buying more calls than puts — directional bullish options bet. "
                "Combined with futures positioning: dual confirmation."
            ),
            emoji="📞",
        )
    elif today_delta < 0 and z < -0.8:
        return Signal(
            name="Options: FII Bearish Stance Strengthening",
            category="Options OI",
            direction=-1, strength=2, score=-2,
            headline="FII Options: Bearish Shift",
            description=(
                f"FII options delta = {today_delta:+,.0f} and deteriorating "
                f"(5D shift: {shift:+,.0f}, z: {z:.1f}σ). "
                "FII holding more puts than calls — defensive/bearish options stance. "
                "When futures AND options both bearish → highest conviction sell signal."
            ),
            emoji="🔻",
        )
    elif today_delta > 0:
        return Signal(
            name="Options: FII Mildly Bullish",
            category="Options OI",
            direction=1, strength=1, score=1,
            headline="FII Options: Mildly Bullish",
            description=(
                f"FII options delta = {today_delta:+,.0f} (positive = more calls than puts). "
                "Moderate bullish options stance. Not at conviction levels but directionally positive."
            ),
            emoji="🟢",
        )
    else:
        return Signal(
            name="Options: FII Defensive",
            category="Options OI",
            direction=-1, strength=1, score=-1,
            headline="FII Options: Defensive",
            description=(
                f"FII options delta = {today_delta:+,.0f} (negative = more puts than calls). "
                "FII hedging or positioning for downside via options. Cautious signal."
            ),
            emoji="🔴",
        )


def _signal_retail_contrarian(oi_df: pd.DataFrame, as_of_date: date) -> Optional[Signal]:
    """Retail (Client) extreme positioning — contrarian indicator."""
    client = (
        oi_df[oi_df["client_type"] == "Client"]
        .sort_values("trade_date")
        .set_index("trade_date")["fut_idx_net"]
    )
    if len(client) < 5:
        return None

    today_val = float(client.iloc[-1])
    z = _zscore(today_val, client.iloc[:-1])

    if z <= -1.8:
        return Signal(
            name="Retail: Extreme Short (Contrarian Buy)",
            category="Retail",
            direction=1, strength=2, score=2,
            headline="Retail Extreme Short",
            description=(
                f"Client (retail) Index Futures net: {today_val:+,.0f} contracts ({z:.1f}σ). "
                "Retail is extremely bearish. Contrarian signal: retail is typically wrong "
                "at turning points — extreme retail shorts historically mark short-term bottoms. "
                "Smart money often fades extreme retail positioning."
            ),
            emoji="🔄",
        )
    elif z >= 1.8:
        return Signal(
            name="Retail: Extreme Long (Contrarian Sell)",
            category="Retail",
            direction=-1, strength=2, score=-2,
            headline="Retail Extreme Long",
            description=(
                f"Client (retail) Index Futures net: {today_val:+,.0f} contracts ({z:.1f}σ). "
                "Retail is excessively bullish. Contrarian sell signal — extreme retail longs "
                "historically precede short-term corrections. 'When everyone's in, it's time to get out.'"
            ),
            emoji="🔄",
        )
    return None


def _signal_fii_money_flow(fii_stats_df: pd.DataFrame, as_of_date: date) -> Optional[Signal]:
    """FII net buy/sell value in Index Futures + Options from fii_derivatives_stats."""
    # Categories are stored UPPERCASE (as NSE provides them)
    idx_cats = fii_stats_df[
        fii_stats_df["category"].isin([
            "INDEX FUTURES", "INDEX OPTIONS",
            "NIFTY FUTURES", "BANKNIFTY FUTURES",
            "NIFTY OPTIONS", "BANKNIFTY OPTIONS",
        ])
    ]
    if idx_cats.empty:
        return None

    latest_date = idx_cats["trade_date"].max()
    today = idx_cats[idx_cats["trade_date"] == latest_date]

    # Aggregate net value across all index F&O categories
    total_net = float(today["net_value_cr"].sum())

    # Historical baseline (all dates except today)
    hist = (
        idx_cats[idx_cats["trade_date"] != latest_date]
        .groupby("trade_date")["net_value_cr"]
        .sum()
    )
    if len(hist) < 3:
        return None

    z = _zscore(total_net, hist)

    if z >= 2.0:
        return Signal(
            name="FII Flow: Massive Net Buying",
            category="FII Flow",
            direction=1, strength=3, score=3,
            headline=f"FII Net Buy Rs.{total_net:+,.0f}Cr ({z:.1f}σ)",
            description=(
                f"FII net bought Rs.{total_net:,.0f} Cr in Index F&O today ({z:.1f}σ above normal). "
                "Exceptional money flow into Index derivatives — high conviction institutional buying. "
                "Rupee value confirms OI signal: real capital is being deployed."
            ),
            emoji="💰",
        )
    elif z >= 1.0:
        return Signal(
            name="FII Flow: Above-Average Buying",
            category="FII Flow",
            direction=1, strength=2, score=2,
            headline=f"FII Net Buy Rs.{total_net:+,.0f}Cr",
            description=(
                f"FII net bought Rs.{total_net:,.0f} Cr in Index F&O ({z:.1f}σ above baseline). "
                "Healthy institutional inflow — supports directional move higher."
            ),
            emoji="💸",
        )
    elif z <= -2.0:
        return Signal(
            name="FII Flow: Massive Net Selling",
            category="FII Flow",
            direction=-1, strength=3, score=-3,
            headline=f"FII Net Sell Rs.{total_net:,.0f}Cr ({z:.1f}σ)",
            description=(
                f"FII net sold Rs.{abs(total_net):,.0f} Cr in Index F&O today ({z:.1f}σ below normal). "
                "Massive capital outflow from index derivatives — institutional distribution at extreme scale. "
                "Strongest bearish confirmation: the money is leaving."
            ),
            emoji="🩸",
        )
    elif z <= -1.0:
        return Signal(
            name="FII Flow: Above-Average Selling",
            category="FII Flow",
            direction=-1, strength=2, score=-2,
            headline=f"FII Net Sell Rs.{abs(total_net):,.0f}Cr",
            description=(
                f"FII net sold Rs.{abs(total_net):,.0f} Cr in Index F&O ({z:.1f}σ below baseline). "
                "Notable outflow — caution for leveraged longs."
            ),
            emoji="📤",
        )
    else:
        return Signal(
            name="FII Flow: Normal Activity",
            category="FII Flow",
            direction=0, strength=1, score=0,
            headline=f"FII Flow Rs.{total_net:+,.0f}Cr (Normal)",
            description=(
                f"FII net flow Rs.{total_net:+,.0f} Cr in Index F&O ({z:.1f}σ) — within normal range. "
                "No unusual capital deployment to signal conviction in either direction."
            ),
            emoji="⚪",
        )


def _pcr_extreme_alert(oi_df: pd.DataFrame, as_of_date: date) -> Optional[str]:
    """Return alert string if PCR is in extreme zone."""
    daily_pcr = (
        oi_df.groupby("trade_date")[["opt_idx_put_long", "opt_idx_call_long"]]
        .sum()
        .reset_index()
        .sort_values("trade_date")
    )
    daily_pcr["pcr"] = (
        daily_pcr["opt_idx_put_long"]
        / daily_pcr["opt_idx_call_long"].replace(0, float("nan"))
    )
    if daily_pcr.empty:
        return None
    current_pcr = float(daily_pcr["pcr"].iloc[-1])
    if current_pcr > 1.4:
        return f"PCR EXTREME HIGH: {current_pcr:.2f} — contrarian BUY zone (fear peak)"
    elif current_pcr < 0.65:
        return f"PCR EXTREME LOW: {current_pcr:.2f} — contrarian SELL zone (complacency peak)"
    return None


# ── FII Today Snapshot ───────────────────────────────────────────────────────

def _compute_fii_today_snapshot(
    oi_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    fii_stats_df: pd.DataFrame,
) -> Optional[FIITodaySnapshot]:
    """
    Combine OI change + Volume + Rupee flow into a single structured snapshot
    of what FIIs did TODAY.  Used only for display — no score contribution.

    Three data layers:
      OI layer:    how many long/short contracts were added or closed today
      Volume:      was today unusually active (z-score vs 20D baseline)?
      Rupee flow:  did real ₹ move in index F&O (from fii_derivatives_stats)?
    """
    fii_daily = (
        oi_df[oi_df["client_type"] == "FII"]
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    if len(fii_daily) < 2:
        return None

    today = fii_daily.iloc[-1]
    prev  = fii_daily.iloc[-2]

    # ── OI layer ─────────────────────────────────────────────────────────────
    long_today  = int(today.get("fut_idx_long",  0) or 0)
    short_today = int(today.get("fut_idx_short", 0) or 0)
    long_prev   = int(prev.get("fut_idx_long",   0) or 0)
    short_prev  = int(prev.get("fut_idx_short",  0) or 0)

    long_change  = long_today  - long_prev    # +ve = added longs today
    short_change = short_today - short_prev   # +ve = added shorts today
    oi_net_change = long_change - short_change # +ve = net bullish move
    cum_net = long_today - short_today

    # ── Volume layer ─────────────────────────────────────────────────────────
    vol_today_cnt = 0
    vol_z = 0.0
    if not vol_df.empty:
        fii_vol = (
            vol_df[vol_df["client_type"] == "FII"]
            .sort_values("trade_date")
            .set_index("trade_date")["total_activity"]
        )
        if len(fii_vol) >= 2:
            vol_today_cnt = int(fii_vol.iloc[-1])
            if len(fii_vol) >= 6:
                vol_z = _zscore(float(fii_vol.iloc[-1]), fii_vol.iloc[:-1])

    # ── Rupee flow layer ─────────────────────────────────────────────────────
    fut_net_cr = 0.0
    opt_net_cr = 0.0
    if not fii_stats_df.empty:
        latest_d   = fii_stats_df["trade_date"].max()
        today_cr   = fii_stats_df[fii_stats_df["trade_date"] == latest_d].copy()
        if "net_value_cr" not in today_cr.columns:
            today_cr["net_value_cr"] = today_cr["buy_value_cr"] - today_cr["sell_value_cr"]
        fut_rows = today_cr[today_cr["category"].str.contains("FUTURES", case=False, na=False)]
        opt_rows = today_cr[today_cr["category"].str.contains("OPTIONS", case=False, na=False)]
        fut_net_cr = float(fut_rows["net_value_cr"].sum())
        opt_net_cr = float(opt_rows["net_value_cr"].sum())
    total_net_cr = fut_net_cr + opt_net_cr

    # ── Classify action ──────────────────────────────────────────────────────
    is_massive_short = cum_net < -50_000
    is_long          = cum_net >  10_000
    covering_pct     = 0.0

    if oi_net_change > 1_000 and is_massive_short:
        action_label = "SHORT COVERING"
        action_color = "#69f0ae"
        action_emoji = "⚡"
        if (long_prev - short_prev) != 0:
            covering_pct = abs(oi_net_change) / abs(long_prev - short_prev) * 100
    elif oi_net_change > 1_000 and is_long:
        action_label = "ADDING LONGS"
        action_color = "#00c853"
        action_emoji = "🟢"
    elif oi_net_change > 1_000:
        action_label = "NET BUYER"
        action_color = "#69f0ae"
        action_emoji = "🟡"
    elif oi_net_change < -1_000 and is_massive_short:
        action_label = "ADDING SHORTS"
        action_color = "#ff5252"
        action_emoji = "💀"
    elif oi_net_change < -1_000 and is_long:
        action_label = "LONG UNWINDING"
        action_color = "#ff9800"
        action_emoji = "📉"
    elif oi_net_change < -1_000:
        action_label = "NET SELLER"
        action_color = "#ff5252"
        action_emoji = "🔴"
    else:
        action_label = "HOLDING"
        action_color = "#888888"
        action_emoji = "⚪"

    # ── Rupee flow override: when ₹ and OI disagree, note it ─────────────────
    # (display layer can flag this as "conflicting signals")

    return FIITodaySnapshot(
        as_of_date=fii_daily.iloc[-1]["trade_date"],
        long_change=long_change,
        short_change=short_change,
        oi_net_change=oi_net_change,
        cumulative_long=long_today,
        cumulative_short=short_today,
        cumulative_net=cum_net,
        vol_today=vol_today_cnt,
        vol_z=vol_z,
        fut_net_cr=fut_net_cr,
        opt_net_cr=opt_net_cr,
        total_net_cr=total_net_cr,
        action_label=action_label,
        action_color=action_color,
        action_emoji=action_emoji,
        covering_pct=covering_pct,
    )


# ── Tomorrow's Verdict ────────────────────────────────────────────────────────

def _compute_tomorrow_verdict(
    signals: list[Signal],
    composite_score: float,
    weekly_expiry: Optional[WeeklyExpiryView],
    oi_df: pd.DataFrame,
) -> TomorrowVerdict:
    """
    Synthesise all signals into a single directional verdict for tomorrow.

    Priority hierarchy (highest wins):
      1. Expiry day (DTE ≤ 1) → pin/consolidation effect dominates
      1.5 Short covering active → FII reducing massive short = net buying = UP
      2. FII + DII institutional divergence → tug-of-war → range (with squeeze risk)
      3. Composite score magnitude → directional call

    Key insight: "FII is bearish" (existing short position) ≠ "market will fall".
    A massive FII short is FUEL for a rally when FIIs cover (buy back). The direction
    of OI change (delta_5d) matters more than the absolute level for predicting tomorrow.
    """
    today_oi   = oi_df[oi_df["trade_date"] == oi_df["trade_date"].max()]
    fii_row    = today_oi[today_oi["client_type"] == "FII"]
    dii_row    = today_oi[today_oi["client_type"] == "DII"]
    client_row = today_oi[today_oi["client_type"] == "Client"]

    fii_net    = float(fii_row.iloc[0]["fut_idx_net"])    if not fii_row.empty    else 0.0
    dii_net    = float(dii_row.iloc[0]["fut_idx_net"])    if not dii_row.empty    else 0.0
    client_net = float(client_row.iloc[0]["fut_idx_net"]) if not client_row.empty else 0.0

    # ── 5D delta: direction of FII position change ────────────────────────────
    fii_series = _fii_oi_series(oi_df)
    delta_5d = 0.0
    if len(fii_series) >= 6:
        delta_5d = float(fii_series.iloc[-1]) - float(fii_series.iloc[-6])

    # ── Squeeze and covering detection ────────────────────────────────────────
    # short_covering_active: FII is massively short AND actively REDUCING the short
    # Each covered contract = 1 BUY transaction → drives market UP
    short_covering_active = fii_net < -75_000 and delta_5d > 10_000

    # squeeze_risk: structural setup where covering can trigger sudden rally
    # Conditions: massive short + NOT aggressively adding more + DII buying floor
    retail_extreme_short = client_net < -20_000
    squeeze_risk = (
        fii_net < -75_000 and
        delta_5d > -15_000 and                              # not aggressively adding shorts
        (dii_net > 10_000 or retail_extreme_short)          # floor or contrarian signal
    )

    fii_dii_diverge = (fii_net < -5_000 and dii_net > 5_000) or \
                      (fii_net > 5_000 and dii_net < -5_000)
    dte  = weekly_expiry.days_to_expiry if weekly_expiry else 7
    pcr  = weekly_expiry.pcr_at_expiry_window if weekly_expiry else 1.0

    # Primary signal = highest |score| after sort
    primary = signals[0] if signals else None
    primary_driver = primary.headline if primary else "Composite score"

    # ── Rule 1: Expiry day (tomorrow IS expiry) ───────────────────────────────
    if dte <= 1:
        if fii_dii_diverge:
            return TomorrowVerdict(
                direction="SIDEWAYS",
                confidence="HIGH",
                headline="Expiry day + institutional tug-of-war → tight range, no trend",
                key_driver=(
                    f"DTE={dte}D — gamma pinning. "
                    f"FII net {fii_net:+,.0f} vs DII net {dii_net:+,.0f} contracts cancel out."
                ),
                key_risk="Global trigger (block deal, macro surprise) could break the range.",
                direction_color="#FFD600",
                squeeze_risk=squeeze_risk,
            )
        elif composite_score <= -3:
            return TomorrowVerdict(
                direction="DOWN",
                confidence="MEDIUM",
                headline="Expiry day with bearish setup → selling on rallies, limited upside",
                key_driver=primary_driver,
                key_risk=(
                    "⚡ Squeeze risk — short covering into expiry close can produce late surge."
                    if squeeze_risk else
                    "Short covering into close could produce a late squeeze."
                ),
                direction_color="#FF6D00",
                squeeze_risk=squeeze_risk,
            )
        else:
            return TomorrowVerdict(
                direction="SIDEWAYS",
                confidence="MEDIUM",
                headline="Expiry day — max pain pin dominates, expect narrow range",
                key_driver=f"PCR {pcr:.2f}, DTE {dte}D — theta decay pins strikes",
                key_risk="Gap open on global news could shift the max pain calculus.",
                direction_color="#FFD600",
            )

    # ── Rule 1.5: Short covering ACTIVELY IN PROGRESS ────────────────────────
    # FII is massively short AND reducing position = net buying = drives market UP.
    # This overrides the bearish composite score because the actual TRANSACTIONS are bullish.
    if short_covering_active:
        return TomorrowVerdict(
            direction="UP",
            confidence="MEDIUM",
            headline=(
                f"FII short covering in progress — net {fii_net:+,.0f} reducing by "
                f"+{delta_5d:,.0f} over 5D. Every covered short = 1 BUY transaction."
            ),
            key_driver=(
                f"Short covering: FII {fii_net:+,.0f} → 5D change +{delta_5d:,.0f} contracts. "
                "Covering is net buying pressure regardless of absolute positioning level."
            ),
            key_risk=(
                "Covering may pause on fresh negative catalyst. "
                "Not a sustainable trend — watch for OI to stabilise and reverse."
            ),
            direction_color="#69F0AE",
            squeeze_risk=True,
            short_covering_active=True,
        )

    # ── Rule 2: Institutional divergence in neutral score zone ────────────────
    if fii_dii_diverge and -4 < composite_score < 4:
        fii_side = "selling" if fii_net < 0 else "buying"
        dii_side = "buying"  if dii_net > 0 else "selling"
        if squeeze_risk:
            return TomorrowVerdict(
                direction="SIDEWAYS",
                confidence="MEDIUM",
                headline=(
                    f"FII {fii_side} vs DII {dii_side} — range-bound. "
                    f"⚡ SQUEEZE RISK: FII holds {fii_net:+,.0f} shorts with DII floor."
                ),
                key_driver=(
                    f"Institutional standoff: FII {fii_net:+,.0f} (SHORT) vs DII {dii_net:+,.0f} (LONG). "
                    "DII floor prevents bearish thesis from playing out at current levels."
                ),
                key_risk=(
                    "⚡ SQUEEZE TRIGGER: Any positive catalyst forces FII to cover shorts = "
                    "sudden BUY pressure = SHARP RALLY. "
                    "Do NOT build fresh short positions at this level — squeeze risk is high."
                ),
                direction_color="#FFD600",
                squeeze_risk=True,
            )
        return TomorrowVerdict(
            direction="SIDEWAYS",
            confidence="MEDIUM",
            headline=f"FII {fii_side} vs DII {dii_side} — institutional standoff, expect consolidation",
            key_driver=(
                f"FII {fii_net:+,.0f} vs DII {dii_net:+,.0f} — opposite positions "
                "create a natural floor and ceiling."
            ),
            key_risk="Decisive FII action (large block buy/sell) could break the range.",
            direction_color="#FFD600",
        )

    # ── Rule 3: Score-driven directional verdict ──────────────────────────────
    #
    # KEY INSIGHT (backtest-validated): A bearish composite score (FII massively short)
    # does NOT predict market will fall tomorrow. Historically 0/7 DOWN calls were correct —
    # in 5 cases the market went UP strongly because deep FII short = squeeze fuel.
    # Rule: Only UP is score-driven. DOWN requires active fresh short BUILDING (delta_5d negative)
    # PLUS no squeeze setup. Otherwise the bearish stance is a CAUTION, not a trade signal.

    if composite_score >= 7:
        return TomorrowVerdict(
            direction="UP",
            confidence="HIGH",
            headline="Strong institutional alignment bullish — upside expected",
            key_driver=primary_driver,
            key_risk="Any adverse global macro or RBI action could reverse FII positioning.",
            direction_color="#00C853",
        )
    elif composite_score >= 3:
        return TomorrowVerdict(
            direction="UP",
            confidence="MEDIUM",
            headline="Moderate institutional bullish tilt — bias is up but not high conviction",
            key_driver=primary_driver,
            key_risk="FII positioning could reverse if global risk-off sentiment increases.",
            direction_color="#69F0AE",
        )
    elif composite_score <= -7 and delta_5d < -10_000 and not squeeze_risk:
        # Only DOWN when FII is ACTIVELY BUILDING fresh shorts aggressively (not just holding):
        # delta_5d < -10K = added 10K+ new short contracts in last 5 days (clear fresh build)
        # AND no squeeze floor (DII not buying, retail not extreme short)
        # Historically rare — most bearish periods are holding + covering, not fresh building
        return TomorrowVerdict(
            direction="DOWN",
            confidence="MEDIUM",
            headline=(
                f"FII aggressively building fresh shorts — net {fii_net:+,.0f}, "
                f"added {abs(delta_5d):,.0f} new shorts in 5D. Selling pressure likely."
            ),
            key_driver=(
                f"Fresh short build: delta_5d={delta_5d:+,.0f}. "
                "FII is not holding existing shorts — they are actively adding. "
                "This is directional selling, not a squeeze setup."
            ),
            key_risk=(
                "Global macro surprise or large DII buying spree could reverse this sharply. "
                "Tight stop-loss essential — short squeezes are violent when they start."
            ),
            direction_color="#FF6D00",
            squeeze_risk=False,
        )
    elif composite_score <= -3:
        # Bearish score but NOT an active fresh short build → NOT a trade signal.
        # FII holding a large short = potential squeeze fuel. Market could go either way.
        # Backtest shows: these DOWN calls were 0% accurate. Stay flat and watch for covering.
        _squeeze_warning = (
            "⚡ SQUEEZE RISK: FII holds massive short with DII floor in place. "
            "Any positive catalyst forces covering = sudden rally. DO NOT SHORT."
            if squeeze_risk else
            "FII bearish but not actively adding shorts. "
            "Existing short position is squeeze fuel — market direction unclear."
        )
        _dii_note = (
            f" DII providing {dii_net:+,.0f} contract support floor."
            if dii_net > 10_000 else ""
        )
        return TomorrowVerdict(
            direction="SIDEWAYS",
            confidence="LOW",
            headline=(
                f"FII bearish bias (score {composite_score:+.0f}) but NOT a short trade signal — "
                "bearish positioning = squeeze fuel, not guaranteed downside."
            ),
            key_driver=(
                f"FII net {fii_net:+,.0f} (bearish stance).{_dii_note} "
                "delta_5d={delta_5d:+,.0f} — not aggressively building, so existing short "
                "is ammunition for a covering rally, not confirmation of fresh selling."
            ).format(delta_5d=delta_5d),
            key_risk=_squeeze_warning,
            direction_color="#FFD600",
            squeeze_risk=squeeze_risk,
        )
    else:
        return TomorrowVerdict(
            direction="SIDEWAYS",
            confidence="LOW",
            headline="No clear institutional edge — await a decisive directional signal",
            key_driver="Composite score in neutral zone (−3 to +3)",
            key_risk="Any large FII block trade or macro event could trigger a trend.",
            direction_color="#78909C",
            squeeze_risk=squeeze_risk,
        )


# ── Score → View label ────────────────────────────────────────────────────────

def _score_to_view(score: float) -> tuple[str, str]:
    if score >= 9:
        return "STRONG BULLISH", "#00c853"
    elif score >= 5:
        return "BULLISH", "#69f0ae"
    elif score >= 2:
        return "CAUTIOUSLY BULLISH", "#b9f6ca"
    elif score >= -1:
        return "NEUTRAL", "#ffca28"
    elif score >= -4:
        return "CAUTIOUSLY BEARISH", "#ffab40"
    elif score >= -8:
        return "BEARISH", "#ff5252"
    else:
        return "STRONG BEARISH", "#d50000"


# ── Bias reasoning ────────────────────────────────────────────────────────────

def _build_reasoning(signals: list[Signal], score: float, as_of_date: date) -> str:
    if not signals:
        return "Insufficient data to generate market view."

    bullish = [s for s in signals if s.direction > 0 and s.strength >= 2]
    bearish = [s for s in signals if s.direction < 0 and s.strength >= 2]

    parts: list[str] = []

    if bullish:
        bull_names = ", ".join(s.headline for s in bullish[:3])
        parts.append(f"Bull case supported by: {bull_names}.")
    if bearish:
        bear_names = ", ".join(s.headline for s in bearish[:3])
        parts.append(f"Bear risk from: {bear_names}.")

    if score > 3:
        parts.append("Weight of evidence is clearly bullish — institutional smart money is positioned long.")
    elif score > 0:
        parts.append("Slight bullish edge, but not high conviction — await confirmation before adding longs.")
    elif score < -3:
        parts.append("Institutional players positioned for downside — elevated risk for long positions.")
    elif score < 0:
        parts.append("Slight bearish tilt — caution warranted, keep position sizes moderate.")
    else:
        parts.append("Market balanced — no institutional edge in either direction, range-bound likely.")

    return " ".join(parts)


# ── Weekly Expiry View ────────────────────────────────────────────────────────

def _weekly_expiry_view(oi_df: pd.DataFrame, as_of_date: date) -> WeeklyExpiryView:
    expiry = _next_thursday(as_of_date)
    days_to = (expiry - as_of_date).days

    # PCR in last 3D before expiry window
    daily_pcr = (
        oi_df.groupby("trade_date")[["opt_idx_put_long", "opt_idx_call_long"]]
        .sum()
        .reset_index()
        .sort_values("trade_date")
    )
    daily_pcr["pcr"] = (
        daily_pcr["opt_idx_put_long"]
        / daily_pcr["opt_idx_call_long"].replace(0, float("nan"))
    )
    current_pcr = float(daily_pcr["pcr"].iloc[-1]) if not daily_pcr.empty else None

    fii = _fii_oi_series(oi_df)
    fii_direction = "Flat"
    if not fii.empty:
        net = float(fii.iloc[-1])
        if net > 5000:
            fii_direction = "Long"
        elif net < -5000:
            fii_direction = "Short"

    # Expiry bias logic
    if days_to <= 2:
        # Final 2 days — theta decay dominates, market gravitates toward max pain
        if current_pcr and current_pcr > 1.2:
            bias = "Range-bound (Put writers defending)"
            reasoning = (
                f"Expiry in {days_to} day(s). High PCR ({current_pcr:.2f}) means "
                "put writers have incentive to pin the market above their strike. "
                "Expect low volatility and range-bound action near current levels."
            )
        elif current_pcr and current_pcr < 0.8:
            bias = "Call writers defend upper levels"
            reasoning = (
                f"Expiry in {days_to} day(s). Low PCR ({current_pcr:.2f}) — "
                "call writers will resist moves above their sold strikes. "
                "Overhead resistance likely near Call OI concentration."
            )
        else:
            bias = "Expiry Week Consolidation"
            reasoning = f"Expiry in {days_to} day(s). Typical pre-expiry range compression expected."
    else:
        if fii_direction == "Long" and current_pcr and current_pcr > 1.0:
            bias = "Bullish into Expiry"
            reasoning = (
                f"FII net long with PCR {current_pcr:.2f}. Put writers will defend support; "
                "FII longs provide upward pressure. Bias: market holds or drifts higher into expiry."
            )
        elif fii_direction == "Short" and current_pcr and current_pcr < 0.9:
            bias = "Bearish into Expiry"
            reasoning = (
                f"FII net short with low PCR {current_pcr:.2f}. Few puts means market "
                "is under-hedged — any negative catalyst gets amplified. "
                "Elevated downside risk into expiry."
            )
        else:
            bias = "Watch for Directional Move"
            reasoning = (
                f"FII {fii_direction.lower()} positioned. "
                f"PCR: {current_pcr:.2f}. "
                "No clear expiry-specific setup — follow OI build direction mid-week."
            )

    return WeeklyExpiryView(
        days_to_expiry=days_to,
        expiry_date=expiry,
        pcr_at_expiry_window=current_pcr,
        fii_net_oi_direction=fii_direction,
        bias=bias,
        reasoning=reasoning,
    )
