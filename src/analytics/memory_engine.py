"""
Prediction Memory Engine — adaptive, self-improving prediction calibration.

PURPOSE: Every day the system predicts tomorrow's direction. The memory engine
stores that prediction with a 12-dimensional market fingerprint. The next day
it fills in what actually happened. When making today's prediction, it searches
all past days whose fingerprint closely matched and asks: "In similar conditions,
what did the market actually do the next day?"

HOW IT IMPROVES PREDICTIONS:
  Today's 12-dim fingerprint = [PCR, FII_5D_flow, FII_options_delta, carry,
                                  VIX, VIX_trend, breadth, max_pain_dist,
                                  FII_net_position, hurst, entropy, oi_score]

  Similarity search over all stored days → top-20 most similar.
  Distribution of outcomes → "UP 65%, DOWN 20%, SIDEWAYS 15%"
  → memory_score = (0.65 − 0.20) × 2.5 = +1.125  →  added to composite
  → "Memory confirms bullish prediction with 65% historical rate"

12-FEATURE FINGERPRINT (vs original 8):
  Original 8: PCR, carry, FII_net, VIX, breadth, Hurst, entropy, OI_score
  Added 4   : FII_5D_cumulative (sustained pressure), FII_options_delta
              (directional options bias), VIX_5D_change (fear trend),
              max_pain_distance (gravity from option writers' ideal)

  WHY EACH MATTERS:
    FII_5D_cumul  — FII buying for 5 days ≠ FII just net long today. Sustained
                    accumulation/distribution is the stronger signal.
    FII_delta     — FII options positioning (call_net - put_net) tells whether
                    FII is directionally betting via options, independent of futures.
    VIX_5D_chg   — A falling VIX market behaves differently to a rising VIX
                    even at the same absolute level.
    max_pain_dist — How far spot is from option writers' ideal (max pain).
                    Near expiry with spot far from max pain = strong gravity.

No external libraries — pure numpy + data layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import get_repository

__all__ = [
    "MemorySignal",
    "AccuracyReport",
    "store_prediction",
    "update_outcomes",
    "get_memory_signal",
    "get_accuracy_report",
    "get_prediction_log",
    "get_pending_predictions",
    "backfill_predictions",
]

# ── 12-dimensional feature fingerprint ───────────────────────────────────────
# Each feature clipped to [lo, hi] then divided to [0, 1] for similarity search.
# Order is FIXED — both _feat_vector (from DB) and _today_vector must use
# the same key iteration order. Python 3.7+ dicts preserve insertion order.

_FEAT_RANGES: dict[str, tuple[float, float]] = {
    # ── Options market (short-term sentiment, highest predictive value) ───────
    "feat_pcr":          (0.3,   2.5),   # Put-Call Ratio across all options
    "feat_max_pain_dist":(-3.0,  3.0),   # (spot − max_pain) / max_pain × 100
                                          # negative = spot below max pain (upside pull)
                                          # positive = spot above (downside gravity)
    # ── Institutional futures flow ────────────────────────────────────────────
    "feat_carry":        (-8.0,  18.0),  # annualised futures carry %
    "feat_fii_net":      (-1.0,  1.0),   # FII net futures position / 400K → ±1
    "feat_fii_5d_cumul": (-1.0,  1.0),   # FII 5D cumulative futures flow / 5000 Cr → ±1
                                          # sustained buying/selling vs one-day position
    "feat_fii_delta":    (-1.0,  1.0),   # FII options: (call_net − put_net) / 100K → ±1
                                          # positive = FII buying calls (bullish options bet)
    # ── Risk / volatility ─────────────────────────────────────────────────────
    "feat_vix":          (8.0,   40.0),  # India VIX absolute level
    "feat_vix_5d_chg":   (-30.0, 30.0), # VIX 5D % change (rising fear vs falling fear)
    # ── Market breadth ────────────────────────────────────────────────────────
    "feat_breadth":      (0.0,   100.0), # % of sectors advancing today
    # ── Statistical regime ────────────────────────────────────────────────────
    "feat_hurst":        (0.0,   1.0),   # Hurst exponent (trending vs mean-reverting)
    "feat_entropy":      (0.0,   1.0),   # Permutation entropy (ordered vs chaotic)
    # ── Overall signal alignment ──────────────────────────────────────────────
    "feat_oi_score":     (-1.0,  1.0),   # composite OI/price score / 20 → ±1
}

# Weights — must sum to 1.0
# Higher weight = this dimension matters more for "similarity"
_FEAT_WEIGHTS: dict[str, float] = {
    "feat_pcr":          0.16,   # options positioning — highest short-term predictive
    "feat_max_pain_dist":0.04,   # gamma gravity
    "feat_carry":        0.06,   # futures term structure
    "feat_fii_net":      0.13,   # FII current position
    "feat_fii_5d_cumul": 0.11,   # FII sustained flow — more reliable than single day
    "feat_fii_delta":    0.06,   # FII options directional bet
    "feat_vix":          0.10,   # risk regime
    "feat_vix_5d_chg":   0.06,   # fear trend direction
    "feat_breadth":      0.08,   # market participation quality
    "feat_hurst":        0.05,   # regime memory
    "feat_entropy":      0.05,   # complexity
    "feat_oi_score":     0.10,   # composite signal alignment
}

assert abs(sum(_FEAT_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"
assert set(_FEAT_RANGES) == set(_FEAT_WEIGHTS), "Range and weight keys must match"

_TOP_K            = 20    # top-N similar days to retrieve
_MIN_HISTORY      = 15    # minimum filled predictions to activate memory signal
_MIN_SIMILARITY   = 0.15  # discard very dissimilar days (similarity < 0.15)
_DIRECTION_THRESH = 0.15  # % return threshold: UP > +0.15%, DOWN < -0.15%
                           # tighter than 0.20% — aligns with composite ≥ ±3 prediction threshold


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SimilarDay:
    trade_date:       date
    similarity:       float
    direction_pred:   str
    direction_actual: str
    was_correct:      bool
    actual_return:    float
    composite_score:  float
    hmm_state:        str
    # Key context values for display
    pcr:              float = 0.0
    fii_net_norm:     float = 0.0
    fii_5d_cumul:     float = 0.0
    vix:              float = 0.0
    breadth:          float = 0.0


@dataclass
class MemorySignal:
    similar_count:        int
    similar_days:         list[SimilarDay] = field(default_factory=list)
    memory_up_pct:        float = 0.0
    memory_dn_pct:        float = 0.0
    memory_sw_pct:        float = 0.0
    memory_score:         float = 0.0           # ±2.5 Signal 24 score
    avg_actual_return:    float = 0.0
    avg_similarity:       float = 0.0
    min_similarity:       float = 0.0
    confirms_prediction:  Optional[bool] = None  # True/False/None (uncertain)
    memory_accuracy:      float = 0.0
    memory_note:          str = ""
    # Feature values shown to user for transparency
    today_features:       dict = field(default_factory=dict)
    error:                Optional[str] = None


@dataclass
class AccuracyReport:
    symbol:            str
    total_predictions: int
    correct:           int
    overall_accuracy:  float
    by_direction:      dict = field(default_factory=dict)
    by_confidence:     dict = field(default_factory=dict)
    by_regime:         dict = field(default_factory=dict)
    accuracy_30d:      Optional[float] = None
    accuracy_60d:      Optional[float] = None
    accuracy_90d:      Optional[float] = None
    avg_return_correct:   float = 0.0
    avg_return_incorrect: float = 0.0
    note:              str = ""


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(pred) -> dict:
    """
    Build the 12-dimensional feature fingerprint from an IndexPrediction-like object.

    Returns two dicts merged:
      - raw:  {feat_pcr: 0.76, feat_vix: 15.0, ...}         (stored in DB)
      - norm: {feat_pcr_n: 0.215, feat_vix_n: 0.259, ...}   (used in similarity)

    All attribute accesses are None-safe with sensible neutral defaults so missing
    data (e.g. no regime computed yet) does not distort similarity scores.
    """
    ctx = pred.market_context
    r   = pred.regime

    # ── Options ───────────────────────────────────────────────────────────────
    pcr = float(getattr(pred, "pcr", None) or 0.90)

    # Max pain distance: (spot − max_pain) / max_pain × 100
    spot      = float(getattr(pred, "spot_close", None) or 0.0)
    lv        = getattr(pred, "levels", None)
    max_pain  = float(getattr(lv, "max_pain", None) or 0.0) if lv else 0.0
    if max_pain > 0 and spot > 0:
        max_pain_dist = (spot - max_pain) / max_pain * 100.0
    else:
        max_pain_dist = 0.0

    # ── Futures ───────────────────────────────────────────────────────────────
    carry      = float(getattr(pred, "carry_pct_ann", None) or 0.0)
    fii_net_r  = float((getattr(ctx, "fii_fut_idx_net", None) or 0) if ctx else 0)
    fii_net    = max(-1.0, min(1.0, fii_net_r / 400_000))

    # FII 5D cumulative (₹ Cr normalised)
    fii_5d_r   = 0.0
    if ctx and hasattr(ctx, "fii_cumul_flows_5d"):
        fno_sym = getattr(pred, "fno_symbol", "")
        fii_5d_r = float(
            (ctx.fii_cumul_flows_5d.get(fno_sym) or 0) if ctx.fii_cumul_flows_5d else 0
        )
    fii_5d     = max(-1.0, min(1.0, fii_5d_r / 5_000))  # ±5000 Cr → ±1

    # FII options delta (call_net - put_net, normalised)
    fii_call_n = float((getattr(ctx, "fii_opt_call_net", None) or 0) if ctx else 0)
    fii_put_n  = float((getattr(ctx, "fii_opt_put_net",  None) or 0) if ctx else 0)
    fii_delta  = max(-1.0, min(1.0, (fii_call_n - fii_put_n) / 100_000))

    # ── Risk / volatility ─────────────────────────────────────────────────────
    vix        = float((getattr(ctx, "vix_close",     None) or 15.0) if ctx else 15.0)
    vix_5d_chg = float((getattr(ctx, "vix_5d_chg_pct",None) or 0.0) if ctx else 0.0)

    # ── Breadth ───────────────────────────────────────────────────────────────
    breadth    = float((getattr(ctx, "breadth_pct_advancing", None) or 50.0) if ctx else 50.0)

    # ── Statistical regime ────────────────────────────────────────────────────
    r_ok       = r and not getattr(r, "error", True)
    hurst      = float((getattr(r, "memory_avg",   None) or 0.5) if r_ok else 0.5)
    entropy    = float((getattr(r, "perm_entropy", None) or 0.9) if r_ok else 0.9)

    # ── Composite score ───────────────────────────────────────────────────────
    raw_score  = float(getattr(pred, "composite_score", None) or 0.0)
    oi_score   = max(-1.0, min(1.0, raw_score / 20.0))

    raw = {
        "feat_pcr":           pcr,
        "feat_max_pain_dist": round(max_pain_dist, 3),
        "feat_carry":         carry,
        "feat_fii_net":       round(fii_net,   4),
        "feat_fii_5d_cumul":  round(fii_5d,    4),
        "feat_fii_delta":     round(fii_delta,  4),
        "feat_vix":           vix,
        "feat_vix_5d_chg":    round(vix_5d_chg, 3),
        "feat_breadth":       breadth,
        "feat_hurst":         round(hurst,    4),
        "feat_entropy":       round(entropy,  4),
        "feat_oi_score":      round(oi_score, 4),
    }

    norm: dict = {}
    for key, val in raw.items():
        lo, hi = _FEAT_RANGES[key]
        norm[key + "_n"] = float(np.clip((val - lo) / max(hi - lo, 1e-9), 0.0, 1.0))

    return {**raw, **norm}


def _feat_vector(row: pd.Series) -> np.ndarray:
    """
    Normalised feature vector from a prediction_log DB row.
    Iterates _FEAT_RANGES in insertion order — must match _today_vector.
    Missing columns (old rows before schema migration) default to mid-range (0.5).
    """
    vec = []
    for key in _FEAT_RANGES:
        lo, hi = _FEAT_RANGES[key]
        raw = row.get(key, None)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            vec.append(0.5)   # mid-range neutral for missing old features
        else:
            vec.append(float(np.clip((float(raw) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)))
    return np.array(vec)


def _today_vector(feats: dict) -> np.ndarray:
    """
    Normalised feature vector from extract_features() output.
    Reads the pre-normalised _n values — same order as _feat_vector.
    """
    return np.array([float(feats.get(key + "_n", 0.5)) for key in _FEAT_RANGES])


def _weighted_distance(a: np.ndarray, b: np.ndarray) -> float:
    w = np.array([_FEAT_WEIGHTS[k] for k in _FEAT_RANGES])
    return float(math.sqrt(np.dot(w, (a - b) ** 2)))


def _similarity(dist: float) -> float:
    """Gaussian kernel: distance → similarity in (0, 1]."""
    return float(math.exp(-dist * 2.5))


def _classify_return(ret: float) -> str:
    if ret > _DIRECTION_THRESH:
        return "UP"
    if ret < -_DIRECTION_THRESH:
        return "DOWN"
    return "SIDEWAYS"


# ── Core API ──────────────────────────────────────────────────────────────────

def store_prediction(pred, trade_date: date) -> None:
    """
    Persist one IndexPrediction to prediction_log. Idempotent (DELETE + INSERT).
    Called automatically at the end of _compute_prediction().
    Stores pre-memory composite score (composite_score field on pred already
    includes Signal 24 when called, but _FeatProxy uses composite_prelim).
    """
    try:
        feats = extract_features(pred)
        r     = pred.regime
        row = {
            "trade_date":       trade_date,
            "fno_symbol":       pred.fno_symbol,
            "direction_pred":   pred.direction,
            "confidence_pred":  pred.confidence,
            "composite_score":  round(pred.composite_score, 3),
            "signal_count":     len(pred.signals),
            # Raw feature values
            "feat_pcr":           feats["feat_pcr"],
            "feat_max_pain_dist": feats["feat_max_pain_dist"],
            "feat_carry":         feats["feat_carry"],
            "feat_fii_net":       feats["feat_fii_net"],
            "feat_fii_5d_cumul":  feats["feat_fii_5d_cumul"],
            "feat_fii_delta":     feats["feat_fii_delta"],
            "feat_vix":           feats["feat_vix"],
            "feat_vix_5d_chg":    feats["feat_vix_5d_chg"],
            "feat_breadth":       feats["feat_breadth"],
            "feat_hurst":         feats["feat_hurst"],
            "feat_entropy":       feats["feat_entropy"],
            "feat_oi_score":      feats["feat_oi_score"],
            # Regime labels
            "hmm_state":    r.hmm_state   if r and not r.error else None,
            "memory_label": r.memory_label if r and not r.error else None,
            "outcome_filled": False,
        }
        get_repository().upsert_prediction(row)
    except Exception:
        pass   # Memory logging is non-fatal


def update_outcomes(as_of_date: date) -> int:
    """
    Fill actual outcomes for all unfilled predictions whose next trading day
    has data available. Called in run_daily_job() after each fetch.
    Returns count of rows filled.
    """
    _INDEX_NAME_MAP = {
        "NIFTY":      "Nifty 50",
        "BANKNIFTY":  "Nifty Bank",
        "FINNIFTY":   "Nifty Financial Services",
        "MIDCPNIFTY": "Nifty Midcap Select",
    }
    repo     = get_repository()
    unfilled = repo.get_unfilled_predictions()
    if unfilled.empty:
        return 0

    filled = 0
    for _, row in unfilled.iterrows():
        pred_date = row["trade_date"]
        if hasattr(pred_date, "date"):
            pred_date = pred_date.date()
        symbol    = str(row["fno_symbol"])
        idx_name  = _INDEX_NAME_MAP.get(symbol)
        if not idx_name:
            continue

        # Next trading day's actual return
        df = repo.query("""
            SELECT pct_chg FROM index_data
            WHERE index_name = ? AND trade_date > ?
            ORDER BY trade_date ASC LIMIT 1
        """, [idx_name, pred_date])

        if df.empty or df["pct_chg"].isna().all():
            continue

        actual_ret  = float(df["pct_chg"].iloc[0])
        dir_actual  = _classify_return(actual_ret)
        dir_pred    = str(row["direction_pred"])
        correct     = (dir_pred == dir_actual)

        repo.fill_prediction_outcome(
            trade_date       = pred_date,
            fno_symbol       = symbol,
            actual_return    = round(actual_ret, 4),
            direction_actual = dir_actual,
            was_correct      = correct,
        )
        filled += 1

    return filled


def get_memory_signal(
    fno_symbol: str,
    trade_date: date,
    pred,
    direction_pred: str = "",
    top_k: int = _TOP_K,
) -> MemorySignal:
    """
    Find the top_k most similar historical days and compute the memory signal.

    MECHANISM:
      1. Build today's 12-dim normalised feature vector
      2. Compute weighted Euclidean distance to every historical day
      3. Top-K most similar → look at their actual next-day outcomes
      4. Distribution: UP X%, DOWN Y%, SIDEWAYS Z%
      5. memory_score = (up_pct − dn_pct) × 2.5   →  Signal 24
      6. confirms_prediction = True if dominant direction matches prediction

    `direction_pred` must be the ACTUAL computed direction from the preliminary
    verdict (not a placeholder), so confirms_prediction is meaningful.
    """
    result = MemorySignal(similar_count=0)
    try:
        repo    = get_repository()
        history = repo.get_filled_predictions(fno_symbol, limit=500)

        if history.empty or len(history) < _MIN_HISTORY:
            result.error = (
                f"Insufficient history: {len(history)} filled predictions "
                f"(need {_MIN_HISTORY} to activate)."
            )
            return result

        # Exclude today and the immediately preceding day (safe guard against
        # same-session re-runs; outcome_filled=TRUE already prevents today's row)
        cutoff  = trade_date - timedelta(days=1)
        history = history[history["trade_date"].apply(
            lambda d: (d.date() if hasattr(d, "date") else d) < cutoff
        )].reset_index(drop=True)

        if len(history) < _MIN_HISTORY:
            result.error = f"Only {len(history)} usable historical days after date cutoff."
            return result

        # Build today's feature vector
        feats   = extract_features(pred)
        today_v = _today_vector(feats)

        # Compute similarity to every historical day
        sims = []
        for _, row in history.iterrows():
            hist_v = _feat_vector(row)
            dist   = _weighted_distance(today_v, hist_v)
            sim    = _similarity(dist)
            if sim >= _MIN_SIMILARITY:
                sims.append((sim, row))

        if not sims:
            result.error = "No days found above minimum similarity threshold."
            return result

        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[:top_k]

        # Aggregate outcomes
        up_count = sw_count = dn_count = 0
        correct_count = 0
        ret_sum = 0.0
        sim_sum = 0.0
        similar_days_out: list[SimilarDay] = []

        for sim, row in top:
            d_actual = str(row.get("direction_actual", "SIDEWAYS"))
            correct  = bool(row.get("was_correct", False))
            ret      = float(row.get("actual_return", 0) or 0)
            pred_d   = str(row.get("direction_pred", "SIDEWAYS"))
            hmm      = str(row.get("hmm_state") or "Unknown")
            score    = float(row.get("composite_score", 0) or 0)
            td       = row["trade_date"]
            if hasattr(td, "date"):
                td = td.date()

            if d_actual == "UP":      up_count += 1
            elif d_actual == "DOWN":  dn_count += 1
            else:                     sw_count += 1

            if correct:
                correct_count += 1
            ret_sum += ret
            sim_sum += sim

            if len(similar_days_out) < 8:
                similar_days_out.append(SimilarDay(
                    trade_date=td, similarity=round(sim, 4),
                    direction_pred=pred_d, direction_actual=d_actual,
                    was_correct=correct, actual_return=round(ret, 3),
                    composite_score=round(score, 2), hmm_state=hmm,
                    pcr=float(row.get("feat_pcr", 0) or 0),
                    fii_net_norm=float(row.get("feat_fii_net", 0) or 0),
                    fii_5d_cumul=float(row.get("feat_fii_5d_cumul", 0) or 0),
                    vix=float(row.get("feat_vix", 0) or 0),
                    breadth=float(row.get("feat_breadth", 0) or 0),
                ))

        k       = len(top)
        up_pct  = up_count  / k
        dn_pct  = dn_count  / k
        sw_pct  = sw_count  / k
        acc     = correct_count / k
        avg_ret = ret_sum / k
        avg_sim = sim_sum / k
        min_sim = min(s for s, _ in top)

        memory_score = round((up_pct - dn_pct) * 2.5, 3)

        # Confirms/contradicts using REAL direction (not placeholder)
        real_dir = direction_pred or (
            getattr(pred, "direction", "") if hasattr(pred, "direction") else ""
        )
        if real_dir == "UP":
            dominant_pct = up_pct
        elif real_dir == "DOWN":
            dominant_pct = dn_pct
        else:
            dominant_pct = sw_pct

        confirms = (True  if dominant_pct >= 0.60 else
                    False if dominant_pct <= 0.35 else None)

        # Human-readable note
        dom  = "UP" if up_pct >= dn_pct and up_pct >= sw_pct else (
               "DOWN" if dn_pct >= up_pct and dn_pct >= sw_pct else "SIDEWAYS")
        conf = ("CONFIRMS" if confirms is True else
                "CONTRADICTS" if confirms is False else "is UNCERTAIN about")

        note = (
            f"Top {k} similar past days — "
            f"UP {up_pct:.0%}  DOWN {dn_pct:.0%}  SIDEWAYS {sw_pct:.0%}. "
            f"Avg next-day return: {avg_ret:+.2f}%. "
            f"Hit rate in similar conditions: {acc:.0%}. "
            f"Memory {conf} the {real_dir or 'current'} prediction."
        )

        # Store today's raw feature values for dashboard transparency panel
        today_feats_display = {
            "PCR":             round(feats["feat_pcr"], 2),
            "FII Net (norm)":  round(feats["feat_fii_net"], 3),
            "FII 5D Flow":     round(feats["feat_fii_5d_cumul"], 3),
            "FII Opt Delta":    round(feats["feat_fii_delta"], 3),
            "VIX":             round(feats["feat_vix"], 1),
            "VIX 5D Chg%":     round(feats["feat_vix_5d_chg"], 2),
            "Breadth%":        round(feats["feat_breadth"], 1),
            "Carry%":          round(feats["feat_carry"], 2),
            "Max Pain Dist%":  round(feats["feat_max_pain_dist"], 2),
            "Hurst":           round(feats["feat_hurst"], 3),
            "Entropy":         round(feats["feat_entropy"], 3),
        }

        result.similar_count       = k
        result.similar_days        = similar_days_out
        result.memory_up_pct       = round(up_pct, 4)
        result.memory_dn_pct       = round(dn_pct, 4)
        result.memory_sw_pct       = round(sw_pct, 4)
        result.memory_score        = memory_score
        result.avg_actual_return   = round(avg_ret, 4)
        result.avg_similarity      = round(avg_sim, 4)
        result.min_similarity      = round(min_sim, 4)
        result.confirms_prediction = confirms
        result.memory_accuracy     = round(acc, 4)
        result.memory_note         = note
        result.today_features      = today_feats_display

    except Exception as exc:
        result.error = str(exc)

    return result


def get_accuracy_report(fno_symbol: str, days: int = 90) -> AccuracyReport:
    """Rolling accuracy analysis — overall + by direction / confidence / regime."""
    report = AccuracyReport(symbol=fno_symbol, total_predictions=0,
                            correct=0, overall_accuracy=0.0)
    try:
        history = get_repository().get_filled_predictions(fno_symbol, limit=500)
        if history.empty:
            report.note = "No prediction history yet."
            return report

        history["trade_date"] = history["trade_date"].apply(
            lambda d: d.date() if hasattr(d, "date") else d
        )
        today   = date.today()
        cutoff  = today - timedelta(days=days)
        window  = history[history["trade_date"] >= cutoff]

        if window.empty:
            report.note = f"No predictions in the last {days} days."
            return report

        report.total_predictions = len(window)
        report.correct           = int(window["was_correct"].sum())
        report.overall_accuracy  = round(report.correct / report.total_predictions, 4)

        for d in ["UP", "DOWN", "SIDEWAYS"]:
            sub = window[window["direction_pred"] == d]
            if len(sub) > 0:
                report.by_direction[d] = {
                    "count":             len(sub),
                    "correct":           int(sub["was_correct"].sum()),
                    "accuracy":          round(sub["was_correct"].mean(), 4),
                    "avg_actual_return": round(sub["actual_return"].mean(), 4),
                }

        for conf in ["HIGH", "MEDIUM", "LOW"]:
            sub = window[window["confidence_pred"] == conf]
            if len(sub) > 0:
                report.by_confidence[conf] = {
                    "count":    len(sub),
                    "correct":  int(sub["was_correct"].sum()),
                    "accuracy": round(sub["was_correct"].mean(), 4),
                }

        for regime in ["Bull", "Sideways", "Bear"]:
            sub = window[window["hmm_state"] == regime]
            if len(sub) > 0:
                report.by_regime[regime] = {
                    "count":    len(sub),
                    "correct":  int(sub["was_correct"].sum()),
                    "accuracy": round(sub["was_correct"].mean(), 4),
                }

        for d_win, attr in [(30, "accuracy_30d"), (60, "accuracy_60d"), (90, "accuracy_90d")]:
            sub = history[history["trade_date"] >= today - timedelta(days=d_win)]
            if len(sub) >= 5:
                setattr(report, attr, round(sub["was_correct"].mean(), 4))

        cr = window[window["was_correct"] == True]["actual_return"].dropna()
        wr = window[window["was_correct"] == False]["actual_return"].dropna()
        if len(cr) > 0:
            report.avg_return_correct   = round(float(cr.mean()), 4)
        if len(wr) > 0:
            report.avg_return_incorrect = round(float(wr.mean()), 4)

        report.note = (
            f"{report.total_predictions} predictions | "
            f"{report.overall_accuracy:.0%} accurate | "
            f"Avg return correct: {report.avg_return_correct:+.2f}%"
        )

    except Exception as exc:
        report.note = f"Error: {exc}"

    return report


def get_prediction_log(fno_symbol: str, limit: int = 60) -> pd.DataFrame:
    """
    Analytics gateway: return filled prediction history for one symbol.
    Dashboard views must call this — never call get_repository() directly.
    """
    df = get_repository().get_filled_predictions(fno_symbol, limit=limit)
    if not df.empty:
        df["trade_date"] = df["trade_date"].apply(
            lambda d: d.date() if hasattr(d, "date") else d
        )
    return df


def get_pending_predictions(fno_symbol: str) -> pd.DataFrame:
    """
    Analytics gateway: return unfilled (pending) predictions for one symbol.
    Dashboard views must call this — never call get_repository() directly.
    """
    unfilled = get_repository().get_unfilled_predictions()
    if unfilled.empty:
        return pd.DataFrame()
    df = unfilled[unfilled["fno_symbol"] == fno_symbol].copy()
    if not df.empty:
        df["trade_date"] = df["trade_date"].apply(
            lambda d: d.date() if hasattr(d, "date") else d
        )
    return df


def backfill_predictions(from_date: date, to_date: date) -> dict:
    """
    Recompute and store predictions for every FNO trading day in [from_date, to_date].
    Also fills outcomes for all completed days.

    _compute_prediction() auto-stores each prediction — we only count here.
    """
    from src.analytics.index_prediction import get_index_predictions
    from src.data.repository import query_dataframe

    dates_df = query_dataframe("""
        SELECT DISTINCT trade_date FROM fno_bhavcopy
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
    """, [from_date, to_date])

    if dates_df.empty:
        return {"processed": 0, "stored": 0, "error": "No FNO data in range"}

    trade_dates = [
        d.date() if hasattr(d, "date") else d
        for d in dates_df["trade_date"].tolist()
    ]

    stored = errors = 0
    for td in trade_dates:
        try:
            preds  = get_index_predictions(td)
            # _compute_prediction() stores each prediction automatically at the end.
            # Do NOT call store_prediction() again here (double-store bug).
            stored += sum(1 for p in preds if p.data_available)
        except Exception:
            errors += 1

    filled = update_outcomes(to_date)

    return {
        "processed":       len(trade_dates),
        "stored":          stored,
        "outcomes_filled": filled,
        "errors":          errors,
    }
