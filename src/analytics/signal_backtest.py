"""
Signal Backtest Engine — walk-forward accuracy test for the Market Intelligence system.

Methodology (no look-ahead bias):
  For each historical trading date D:
    1. Compute what the 9-signal engine would have said at CLOSE of day D
       using ONLY data available up to and including day D.
    2. Get actual Nifty 50 % change on the NEXT trading day D+1.
    3. Score: CORRECT if verdict direction matches actual market direction.

Thresholds (configurable):
  Nifty next-day > +threshold  → market "went UP"
  Nifty next-day < -threshold  → market "went DOWN"
  Otherwise                    → market "was SIDEWAYS"

Simulated P&L (1-unit trading model):
  UP verdict     → go long  1 unit → P&L = actual next-day Nifty %
  DOWN verdict   → go short 1 unit → P&L = -(actual next-day Nifty %)
  SIDEWAYS       → stay flat       → P&L = 0

This is the most honest evaluation of the signal system:
  - No data leakage (signal only uses history ≤ D)
  - Fair threshold (±0.25% excludes noise while capturing real moves)
  - P&L simulation shows real trading value, not just hit-rate vanity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["run_signal_backtest", "BacktestRecord", "BacktestSummary"]

# Default verdict-correctness thresholds
_DEFAULT_THRESHOLD = 0.25   # ±0.25% Nifty change = "directional"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BacktestRecord:
    """One row in the backtest table: signal D → outcome D+1."""
    signal_date: date        # evening of this date the signal was generated
    next_date: date          # trading day being predicted
    verdict: str             # "UP" | "DOWN" | "SIDEWAYS"
    confidence: str          # "HIGH" | "MEDIUM" | "LOW"
    composite_score: float   # raw composite score
    market_view: str         # "BEARISH" | "CAUTIOUSLY BEARISH" etc.
    squeeze_risk: bool
    short_covering: bool
    next_day_pct: float      # actual Nifty 50 % change on next_date
    actual_direction: str    # "UP" | "DOWN" | "SIDEWAYS" (classified from pct)
    outcome: str             # "CORRECT" | "WRONG"
    pnl_sim: float           # simulated 1-unit P&L from following the signal
    # Key signals that drove the verdict
    key_driver: str = ""
    key_risk: str = ""


@dataclass
class BacktestSummary:
    """Aggregate accuracy statistics over the backtest window."""
    total_signals: int
    up_signals: int
    down_signals: int
    sideways_signals: int
    # Directional accuracy (UP + DOWN signals only — most tradeable)
    directional_signals: int
    correct_directional: int
    wrong_directional: int
    correct_up: int
    correct_down: int
    correct_sideways: int
    # Rates
    hit_rate_overall: float       # correct / total
    hit_rate_directional: float   # correct directional / directional signals
    hit_rate_up: float
    hit_rate_down: float
    hit_rate_sideways: float
    # By confidence
    high_conf_signals: int
    high_conf_correct: int
    hit_rate_high: float
    med_conf_signals: int
    med_conf_correct: int
    hit_rate_med: float
    low_conf_signals: int
    low_conf_correct: int
    hit_rate_low: float
    # P&L simulation
    cumulative_pnl: float
    avg_pnl_per_signal: float
    sharpe_sim: float             # (mean return / std) * sqrt(252) — rough Sharpe
    max_drawdown: float           # worst underwater stretch in % P&L
    # Streaks
    longest_correct_streak: int
    longest_wrong_streak: int
    # Notable calls
    best_call: Optional[BacktestRecord]
    worst_call: Optional[BacktestRecord]
    squeeze_accuracy: float       # accuracy rate when squeeze_risk = True
    covering_accuracy: float      # accuracy rate when short_covering = True
    # Full records (sorted chronologically)
    records: list[BacktestRecord] = field(default_factory=list)
    threshold_pct: float = _DEFAULT_THRESHOLD


# ── Main backtest function ────────────────────────────────────────────────────

def run_signal_backtest(
    end_date: date,
    backtest_days: int = 60,
    threshold_pct: float = _DEFAULT_THRESHOLD,
    signal_lookback: int = 45,
) -> BacktestSummary:
    """
    Walk-forward signal accuracy backtest.

    Parameters
    ----------
    end_date       : last date to include (usually today or most recent trading date)
    backtest_days  : how many historical signal dates to evaluate
    threshold_pct  : ±% to classify a day as UP / DOWN / SIDEWAYS
    signal_lookback: lookback window passed to market_intelligence (days of history used)

    Returns BacktestSummary with all records and aggregate stats.
    """
    from src.analytics.market_intelligence import get_market_intelligence

    # ── Load all available signal dates ──────────────────────────────────────
    buffer_start = end_date - timedelta(days=backtest_days * 2 + signal_lookback * 2)

    sig_dates_df = query_dataframe("""
        SELECT DISTINCT trade_date FROM fao_participant
        WHERE data_type = 'OI'
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date
    """, [buffer_start, end_date])

    if sig_dates_df.empty:
        return _empty_summary(threshold_pct)

    # ── Load Nifty 50 returns for outcome lookup ──────────────────────────────
    nifty_df = query_dataframe("""
        SELECT trade_date, pct_chg, close_val
        FROM index_data
        WHERE index_name = 'Nifty 50'
          AND trade_date >= ?
          AND trade_date <= ?
        ORDER BY trade_date
    """, [buffer_start, end_date + timedelta(days=10)])

    if nifty_df.empty:
        return _empty_summary(threshold_pct)

    nifty_pct   = nifty_df.set_index("trade_date")["pct_chg"].to_dict()
    nifty_dates = sorted(nifty_pct.keys())

    # Keep only the most recent backtest_days signal dates
    all_sig_dates = sorted(sig_dates_df["trade_date"].tolist())
    # Require at least signal_lookback // 2 prior dates for baseline stats
    min_history = max(10, signal_lookback // 2)
    eval_dates = all_sig_dates[min_history:][-backtest_days:]

    records: list[BacktestRecord] = []

    for sig_date in eval_dates:
        # Next trading day with Nifty data
        future = [d for d in nifty_dates if d > sig_date]
        if not future:
            continue
        next_date = future[0]
        next_pct  = float(nifty_pct.get(next_date, 0.0))

        # Classify actual direction
        if next_pct > threshold_pct:
            actual_dir = "UP"
        elif next_pct < -threshold_pct:
            actual_dir = "DOWN"
        else:
            actual_dir = "SIDEWAYS"

        # ── Compute signal (walk-forward — only data ≤ sig_date) ─────────────
        try:
            mi = get_market_intelligence(sig_date, lookback_days=signal_lookback)
        except Exception:
            continue

        if mi.market_view == "No Data" or mi.tomorrow_verdict is None:
            continue

        verd  = mi.tomorrow_verdict
        verdict    = verd.direction
        confidence = verd.confidence

        # ── Score ──────────────────────────────────────────────────────────
        correct = (verdict == actual_dir)

        # Simulated P&L (1-unit leveraged trade)
        if verdict == "UP":
            pnl = next_pct
        elif verdict == "DOWN":
            pnl = -next_pct
        else:
            pnl = 0.0

        records.append(BacktestRecord(
            signal_date=sig_date,
            next_date=next_date,
            verdict=verdict,
            confidence=confidence,
            composite_score=mi.composite_score,
            market_view=mi.market_view,
            squeeze_risk=verd.squeeze_risk,
            short_covering=verd.short_covering_active,
            next_day_pct=next_pct,
            actual_direction=actual_dir,
            outcome="CORRECT" if correct else "WRONG",
            pnl_sim=pnl,
            key_driver=verd.key_driver[:80] if verd.key_driver else "",
            key_risk=verd.key_risk[:80] if verd.key_risk else "",
        ))

    if not records:
        return _empty_summary(threshold_pct)

    return _compute_summary(records, threshold_pct)


# ── Summary computation ───────────────────────────────────────────────────────

def _compute_summary(records: list[BacktestRecord], threshold_pct: float) -> BacktestSummary:
    total = len(records)

    up_r   = [r for r in records if r.verdict == "UP"]
    dn_r   = [r for r in records if r.verdict == "DOWN"]
    sw_r   = [r for r in records if r.verdict == "SIDEWAYS"]
    dir_r  = up_r + dn_r

    def n_correct(subset): return sum(1 for r in subset if r.outcome == "CORRECT")
    def hit(subset): return n_correct(subset) / len(subset) if subset else 0.0

    high_r = [r for r in records if r.confidence == "HIGH"]
    med_r  = [r for r in records if r.confidence == "MEDIUM"]
    low_r  = [r for r in records if r.confidence == "LOW"]

    sq_r   = [r for r in dir_r if r.squeeze_risk]
    cov_r  = [r for r in dir_r if r.short_covering]

    # ── P&L and risk stats ────────────────────────────────────────────────────
    pnls = [r.pnl_sim for r in records]
    cum_pnl = sum(pnls)
    avg_pnl = cum_pnl / total

    # Rough Sharpe: annualised mean / std of per-signal returns
    pnl_series = pd.Series(pnls)
    pnl_std = float(pnl_series.std())
    sharpe = (avg_pnl / pnl_std * (252 ** 0.5)) if pnl_std > 1e-9 else 0.0

    # Max drawdown on cumulative P&L curve
    cum_curve = pnl_series.cumsum()
    running_max = cum_curve.cummax()
    drawdowns = cum_curve - running_max
    max_dd = float(drawdowns.min())

    # ── Streaks ───────────────────────────────────────────────────────────────
    max_correct_streak = cur_c = 0
    max_wrong_streak   = cur_w = 0
    for r in records:
        if r.outcome == "CORRECT":
            cur_c += 1; cur_w = 0
        else:
            cur_w += 1; cur_c = 0
        max_correct_streak = max(max_correct_streak, cur_c)
        max_wrong_streak   = max(max_wrong_streak,   cur_w)

    # Best/worst single directional call by P&L
    dir_correct = [r for r in dir_r if r.outcome == "CORRECT"]
    dir_wrong   = [r for r in dir_r if r.outcome == "WRONG"]
    best  = max(dir_correct, key=lambda r: r.pnl_sim, default=None)
    worst = min(dir_wrong,   key=lambda r: r.pnl_sim, default=None)

    return BacktestSummary(
        total_signals=total,
        up_signals=len(up_r), down_signals=len(dn_r), sideways_signals=len(sw_r),
        directional_signals=len(dir_r),
        correct_directional=n_correct(dir_r),
        wrong_directional=len(dir_r) - n_correct(dir_r),
        correct_up=n_correct(up_r), correct_down=n_correct(dn_r), correct_sideways=n_correct(sw_r),
        hit_rate_overall=hit(records),
        hit_rate_directional=hit(dir_r),
        hit_rate_up=hit(up_r),
        hit_rate_down=hit(dn_r),
        hit_rate_sideways=hit(sw_r),
        high_conf_signals=len(high_r), high_conf_correct=n_correct(high_r), hit_rate_high=hit(high_r),
        med_conf_signals=len(med_r),   med_conf_correct=n_correct(med_r),   hit_rate_med=hit(med_r),
        low_conf_signals=len(low_r),   low_conf_correct=n_correct(low_r),   hit_rate_low=hit(low_r),
        cumulative_pnl=cum_pnl,
        avg_pnl_per_signal=avg_pnl,
        sharpe_sim=sharpe,
        max_drawdown=max_dd,
        longest_correct_streak=max_correct_streak,
        longest_wrong_streak=max_wrong_streak,
        best_call=best,
        worst_call=worst,
        squeeze_accuracy=hit(sq_r),
        covering_accuracy=hit(cov_r),
        records=records,
        threshold_pct=threshold_pct,
    )


def _empty_summary(threshold_pct: float) -> BacktestSummary:
    return BacktestSummary(
        total_signals=0, up_signals=0, down_signals=0, sideways_signals=0,
        directional_signals=0, correct_directional=0, wrong_directional=0,
        correct_up=0, correct_down=0, correct_sideways=0,
        hit_rate_overall=0.0, hit_rate_directional=0.0,
        hit_rate_up=0.0, hit_rate_down=0.0, hit_rate_sideways=0.0,
        high_conf_signals=0, high_conf_correct=0, hit_rate_high=0.0,
        med_conf_signals=0,  med_conf_correct=0,  hit_rate_med=0.0,
        low_conf_signals=0,  low_conf_correct=0,  hit_rate_low=0.0,
        cumulative_pnl=0.0, avg_pnl_per_signal=0.0, sharpe_sim=0.0, max_drawdown=0.0,
        longest_correct_streak=0, longest_wrong_streak=0,
        best_call=None, worst_call=None,
        squeeze_accuracy=0.0, covering_accuracy=0.0,
        threshold_pct=threshold_pct,
    )
