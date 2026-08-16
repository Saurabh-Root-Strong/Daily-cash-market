"""
Sector Rotation Memory Engine — self-improving signal calibration.

PURPOSE
-------
The sector rotation page classifies sectors as "Confirmed Accumulation",
"Secret Accumulation", etc. based on today's delivery/Z-score signals.
But is the signal historically reliable?  In what regime conditions does
"Confirmed Accumulation" actually deliver positive absolute or relative
returns?

The memory engine answers this by recording every day's sector signals
with a 9-dimensional fingerprint, then filling in the actual forward
outcomes after 7 / 14 / 30 calendar days.  When analysing today, it
finds the K most similar past conditions and reports:

  "In the 14 most similar past setups for IT + Confirmed Accumulation:
   ─ Next 1W:  median +0.8%  (range −1.4% to +3.2%)  |  Positive: 9/14 (64%)
   ─ Next 2W:  median +1.9%  (range −2.1% to +5.6%)  |  Positive: 10/14 (71%)
   ─ RS vs Nifty 2W: median +2.8%  |  Outperformed: 12/14 (86%)"

This directly answers the user's core question:
  "If I buy this sector now, based on past history, will I make or lose money?"

9-DIMENSIONAL FINGERPRINT
-------------------------
Sector delivery conditions (3 dims):
  feat_dv_n     — normalised DV ratio (today's delivery / 100D avg)
  feat_zpct_n   — Z-score cross-sectional percentile (already 0–1)
  feat_rs_n     — sector RS vs Nifty in 1W at signal time

Market regime conditions (6 dims — from prediction_log / regime engine):
  feat_ema20    — Nifty above(+1) / below(–1) 20D EMA
  feat_ema_x    — Golden cross (+1) / Death cross (–1) / Neutral (0)
  feat_vix_n    — VIX normalised 8–40 → 0–1
  feat_fii_n    — FII 5D net flow normalised ±20000 Cr → 0–1
  feat_hmm_n    — HMM state: Bull=1, Sideways=0.5, Bear=0
  feat_pcr_n    — PCR normalised 0.3–2.5 → 0–1

TABLES (in market_data.duckdb)
-------------------------------
  sector_rotation_log     One row per (trade_date, sector)
  sector_regime_log       One row per trade_date

WORKFLOW
--------
  Daily (7:30 PM after nightly_sync):
    1. python -m scripts.sector_memory_daily record          → logs today's signals
    2. python -m scripts.sector_memory_daily fill-outcomes   → fills 7/14/30D outcomes for old records

  One-time backfill:
    python -m scripts.sector_memory_backfill --days 365

  Streamlit (cached call):
    memory = cached_sector_memory_context(trade_date, sector, signal, regime)
    → shows historical outcome stats in _sector_card()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe, get_repository

__all__ = [
    "SectorMemoryContext",
    "MemoryEdge",
    "record_daily_snapshot",
    "fill_forward_outcomes",
    "get_sector_memory_context",
    "compute_memory_edge",
    "apply_memory_overlay",
    "backfill_sector_memory",
    "SCHEMA_DDL",
]

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS sector_rotation_log (
    trade_date       DATE    NOT NULL,
    sector           VARCHAR NOT NULL,
    -- Sector signal at time of recording
    signal           VARCHAR,
    accum_score      REAL,
    dv_ratio         REAL,
    dv_ratio_5d      REAL,
    z_score          REAL,
    z_pct            REAL,
    breadth          REAL,
    price_1w         REAL,
    rs_1w            REAL,
    -- 9-dimensional normalised fingerprint (for similarity search)
    feat_dv_n        REAL,
    feat_zpct_n      REAL,
    feat_rs_n        REAL,
    feat_ema20       REAL,
    feat_ema_x       REAL,
    feat_vix_n       REAL,
    feat_fii_n       REAL,
    feat_hmm_n       REAL,
    feat_pcr_n       REAL,
    -- Regime label snapshot
    regime_label     VARCHAR,
    -- Forward outcomes (filled retrospectively by fill_forward_outcomes)
    fwd_ret_1w       REAL,
    fwd_ret_2w       REAL,
    fwd_ret_1m       REAL,
    fwd_nifty_1w     REAL,
    fwd_nifty_2w     REAL,
    fwd_nifty_1m     REAL,
    fwd_rs_1w        REAL,
    fwd_rs_2w        REAL,
    fwd_rs_1m        REAL,
    outcome_filled   BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (trade_date, sector)
);

CREATE TABLE IF NOT EXISTS sector_regime_log (
    trade_date       DATE PRIMARY KEY,
    regime_label     VARCHAR,
    regime_score     REAL,
    ema20_above      INTEGER,
    ema_cross_bull   INTEGER,
    vix              REAL,
    vix_5d_chg       REAL,
    fii_5d_cr        REAL,
    hmm_state        VARCHAR,
    pcr              REAL,
    market_breadth   REAL
);
"""

# ── Feature weights — must sum to 1.0 ────────────────────────────────────────
# Sector delivery dims weighted more than regime because the SECTOR-SPECIFIC
# delivery signal is what's being tested (regime is shared context).
_FEAT_WEIGHTS: dict[str, float] = {
    "feat_dv_n":    0.22,   # delivery ratio vs own 100D norm (core accumulation signal)
    "feat_zpct_n":  0.18,   # cross-sectional delivery rank (how abnormal vs all sectors)
    "feat_rs_n":    0.10,   # relative price strength vs Nifty at signal time
    "feat_ema20":   0.12,   # market short-term trend direction
    "feat_ema_x":   0.08,   # golden / death cross (trend inflection)
    "feat_vix_n":   0.10,   # fear level (regime quality)
    "feat_fii_n":   0.10,   # institutional stance (regime quality)
    "feat_hmm_n":   0.06,   # statistical regime memory
    "feat_pcr_n":   0.04,   # options forward-looking bias
}
assert abs(sum(_FEAT_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"

_TOP_K           = 20   # retrieve up to 20 most similar INDEPENDENT setups
_MIN_SIMILARITY  = 0.40 # discard very dissimilar days (< 40% similarity)
_MIN_RECORDS     = 5    # need at least 5 similar filled records to show memory
_EPISODE_GAP_DAYS = 10  # min calendar gap between retained neighbours so their
                        # forward windows don't overlap (independent-episode k_eff)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SimilarSetup:
    trade_date:    date
    sector:        str
    signal:        str
    regime_label:  str
    similarity:    float
    dv_ratio:      float
    z_pct:         float
    # Forward outcomes
    fwd_ret_1w:    Optional[float]
    fwd_ret_2w:    Optional[float]
    fwd_ret_1m:    Optional[float]
    fwd_rs_1w:     Optional[float]
    fwd_rs_2w:     Optional[float]
    fwd_rs_1m:     Optional[float]


@dataclass
class SectorMemoryContext:
    """
    Historical outcome statistics for a given (sector, signal, conditions) combination.
    Returned by get_sector_memory_context() and shown on each sector card.
    """
    sector:           str
    signal:           str
    n_similar:        int            # number of similar setups found
    n_filled:         int            # number with outcome data
    n_episodes:       int = 0        # INDEPENDENT episodes after overlap de-dup (k_eff)
    avg_similarity:   float = 0.0    # mean similarity of the retained neighbours [0,1]
    similar_setups:   list[SimilarSetup] = field(default_factory=list)

    # 1-week outcomes
    ret_1w_median:    Optional[float] = None
    ret_1w_p25:       Optional[float] = None
    ret_1w_p75:       Optional[float] = None
    ret_1w_pos_pct:   Optional[float] = None   # % of setups with positive absolute return
    rs_1w_median:     Optional[float] = None   # median RS vs Nifty
    rs_1w_pos_pct:    Optional[float] = None   # % of setups that outperformed Nifty

    # 2-week outcomes
    ret_2w_median:    Optional[float] = None
    ret_2w_p25:       Optional[float] = None
    ret_2w_p75:       Optional[float] = None
    ret_2w_pos_pct:   Optional[float] = None
    rs_2w_median:     Optional[float] = None
    rs_2w_pos_pct:    Optional[float] = None

    # 1-month outcomes
    ret_1m_median:    Optional[float] = None
    ret_1m_pos_pct:   Optional[float] = None
    rs_1m_median:     Optional[float] = None

    # Regime breakdown of the similar setups
    by_regime:        dict = field(default_factory=dict)  # {"BEAR": 8, "SIDEWAYS": 4, ...}

    note:             str = ""
    error:            Optional[str] = None


# ── Feature normalisation ─────────────────────────────────────────────────────

def _normalize(
    dv_ratio:       float,
    z_pct:          float,
    rs_1w:          Optional[float],
    ema20_above:    bool,
    ema_cross_bull: Optional[bool],
    vix:            Optional[float],
    fii_5d_cr:      Optional[float],
    hmm_state:      Optional[str],
    pcr:            Optional[float],
) -> dict[str, float]:
    """
    Map raw features to [0, 1] for Euclidean similarity search.
    None / missing values → 0.5 (neutral centre) so they contribute
    minimally to distance rather than dominating it.
    """
    def _clip(v, lo, hi):
        return max(lo, min(hi, v))

    return {
        # Sector delivery
        "feat_dv_n":   _clip((dv_ratio - 0.5) / 2.5, 0.0, 1.0),   # 0.5–3.0 → 0–1
        "feat_zpct_n": float(z_pct),                                 # already 0–1
        "feat_rs_n":   _clip(((rs_1w or 0.0) + 20) / 40, 0.0, 1.0), # ±20% → 0–1
        # Regime
        "feat_ema20":  1.0 if ema20_above else 0.0,
        "feat_ema_x":  (1.0 if ema_cross_bull is True
                        else 0.0 if ema_cross_bull is False
                        else 0.5),
        "feat_vix_n":  _clip(((vix or 16) - 8) / 32, 0.0, 1.0),     # 8–40 → 0–1
        "feat_fii_n":  _clip(((fii_5d_cr or 0.0) + 20_000) / 40_000, 0.0, 1.0),
        "feat_hmm_n":  {"Bull": 1.0, "Sideways": 0.5, "Bear": 0.0}.get(hmm_state or "Sideways", 0.5),
        "feat_pcr_n":  _clip(((pcr or 1.0) - 0.3) / 2.2, 0.0, 1.0),  # 0.3–2.5 → 0–1
    }


def _similarity(f1: dict[str, float], f2: dict[str, float]) -> float:
    """
    Weighted similarity score in [0, 1].
    1.0 = identical fingerprints, 0.0 = maximally different.

    Uses inverse weighted Euclidean distance normalised to [0, 1]:
      distance = sqrt( Σ w_i × (f1_i - f2_i)² )   (max possible = sqrt(Σ w_i) = 1.0)
      similarity = 1 - distance
    """
    dist_sq = sum(
        _FEAT_WEIGHTS[k] * (f1.get(k, 0.5) - f2.get(k, 0.5)) ** 2
        for k in _FEAT_WEIGHTS
    )
    distance = math.sqrt(dist_sq)   # range 0–1 (max = sqrt(sum(weights × 1)) = 1.0)
    return round(max(0.0, 1.0 - distance), 4)


def _similarity_many(f1: dict[str, float], feats: pd.DataFrame) -> np.ndarray:
    """_similarity for a whole frame of candidate rows at once.

    `feats` must have exactly the _FEAT_WEIGHTS columns. Returns one rounded
    similarity per row, in row order.

    This exists because get_sector_memory_context is called once per sector
    (~24 per overlay) over that sector's whole filled history, and scoring the
    rows one at a time was 16,368 iterrows and the largest single cost in the
    Smart Money panel. _similarity remains the scalar definition of the metric;
    tests/test_sector_memory_sim.py pins the two together over random inputs
    including NaNs, so this cannot quietly drift into a different metric.

    NaN rows come out NaN here, where the scalar path returns 0.0 — Python's
    max(0.0, nan) keeps 0.0 while np.maximum propagates. Both then fail
    `>= _MIN_SIMILARITY`, so the row is dropped either way; the caller must
    compare with >= and never with `not <`.
    """
    cols = list(_FEAT_WEIGHTS.keys())
    w = np.array([_FEAT_WEIGHTS[k] for k in cols], dtype=np.float64)
    t = np.array([f1.get(k, 0.5) for k in cols], dtype=np.float64)
    # float64, not the stored float32: the scalar path widened via float()
    # before doing the arithmetic, and float32 would round differently.
    f = feats[cols].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.round(
            np.maximum(0.0, 1.0 - np.sqrt((((f - t) ** 2) * w).sum(axis=1))), 4)


# ── Schema initialisation ─────────────────────────────────────────────────────

def _ensure_schema() -> None:
    """Create sector_rotation_log and sector_regime_log if absent."""
    try:
        repo = get_repository()
        with repo._cm.connect() as conn:
            for stmt in SCHEMA_DDL.strip().split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(s)
    except Exception as exc:
        import sys
        print(f"[SectorMemory] schema init failed: {exc}", file=sys.stderr)


# ── Recording ─────────────────────────────────────────────────────────────────

def record_daily_snapshot(
    as_of_date:       date,
    min_turnover_lacs: float = 1.0,
    replace:          bool = False,
) -> int:
    """
    Compute today's sector rotation signals and market regime, then write
    one row per sector to sector_rotation_log and one row to sector_regime_log.

    This is the SINGLE source of truth for the fingerprint. The historical
    backfill replays this exact function per past date, so stored history is
    byte-for-byte consistent with the live daily job — there is no second
    feature implementation to drift out of sync.

    Parameters
    ----------
    replace : if False (daily default) skip dates already recorded — idempotent.
              if True (backfill) delete the date's existing rows first and
              re-record, so a corrected fingerprint definition can be rebuilt.

    Returns the number of sector rows written (0 on failure or if already done).
    Called by sector_memory_daily.py after nightly_sync completes.
    """
    _ensure_schema()

    try:
        existing = query_dataframe(
            "SELECT COUNT(*) AS n FROM sector_rotation_log WHERE trade_date = ?",
            [as_of_date],
        )
        n_existing = int(existing["n"].iloc[0]) if not existing.empty else 0
    except Exception:
        n_existing = 0

    if n_existing > 0:
        if not replace:
            return 0   # idempotent: already recorded, leave matured outcomes intact
        # replace=True → wipe this date's rows in both tables before re-recording
        try:
            repo = get_repository()
            with repo._cm.connect() as conn:
                conn.execute("DELETE FROM sector_rotation_log WHERE trade_date = ?", [as_of_date])
                conn.execute("DELETE FROM sector_regime_log   WHERE trade_date = ?", [as_of_date])
        except Exception as exc:
            import sys
            print(f"[SectorMemory] replace-delete failed for {as_of_date}: {exc}",
                  file=sys.stderr)

    try:
        from src.analytics.sector_rotation import get_sector_rotation, get_market_regime
        rot    = get_sector_rotation(as_of_date, min_turnover_lacs=min_turnover_lacs)
        regime = get_market_regime(as_of_date)
    except Exception as exc:
        import sys
        print(f"[SectorMemory] sector rotation compute failed for {as_of_date}: {exc}",
              file=sys.stderr)
        return 0

    if rot.empty:
        return 0

    # Regime fingerprint
    r_ema20   = regime.get("nifty_vs_ema20", "—")
    r_ema_x   = regime.get("nifty_vs_ema50", "—")
    r_vix     = regime.get("vix")
    r_fii     = regime.get("fii_5d_cr")
    r_hmm     = regime.get("hmm_state", "Sideways")
    r_pcr     = None   # fetch from prediction_log below
    r_label   = regime.get("regime", "SIDEWAYS")
    r_score   = regime.get("score", 5.0)

    try:
        pcr_row = query_dataframe("""
            SELECT feat_pcr FROM prediction_log
            WHERE fno_symbol = 'NIFTY'
              AND trade_date <= ?
              AND trade_date >= (? - INTERVAL 7 DAY)
            ORDER BY trade_date DESC LIMIT 1
        """, [as_of_date, as_of_date])
        if not pcr_row.empty:
            r_pcr = float(pcr_row["feat_pcr"].iloc[0])
    except Exception:
        pass

    ema20_above    = (r_ema20 == "ABOVE")
    ema_cross_bull = (True if r_ema_x == "ABOVE" else False if r_ema_x == "BELOW" else None)

    # Write regime log row
    try:
        repo = get_repository()
        with repo._cm.connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sector_regime_log VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, [
                as_of_date, r_label, r_score,
                1 if ema20_above else 0,
                (1 if ema_cross_bull is True else 0 if ema_cross_bull is False else None),
                r_vix, None,   # vix_5d_chg: numeric Δ not exposed by get_market_regime
                r_fii, r_hmm, r_pcr,
                None,   # market_breadth — filled separately if available
            ])
    except Exception as exc:
        import sys
        print(f"[SectorMemory] regime_log write failed: {exc}", file=sys.stderr)

    # Write sector rows
    rows_written = 0
    try:
        repo = get_repository()
        with repo._cm.connect() as conn:
            for _, row in rot.iterrows():
                sector  = str(row["sector"])
                signal  = str(row.get("signal", ""))
                dv      = float(row.get("dv_ratio",    1.0) or 1.0)
                dv5d    = row.get("dv_ratio_5d")
                z       = float(row.get("z_score",     0.0) or 0.0)
                zpct    = float(row.get("z_pct",       0.5) or 0.5)
                breadth = row.get("breadth")
                p1w     = row.get("price_1w")
                rs1w    = row.get("rs_1w")
                score   = float(row.get("accum_score", 50.0) or 50.0)

                feat = _normalize(
                    dv_ratio       = dv,
                    z_pct          = zpct,
                    rs_1w          = float(rs1w) if rs1w is not None and not pd.isna(rs1w) else None,
                    ema20_above    = ema20_above,
                    ema_cross_bull = ema_cross_bull,
                    vix            = r_vix,
                    fii_5d_cr      = r_fii,
                    hmm_state      = r_hmm,
                    pcr            = r_pcr,
                )

                conn.execute("""
                    INSERT OR REPLACE INTO sector_rotation_log (
                        trade_date, sector, signal, accum_score,
                        dv_ratio, dv_ratio_5d, z_score, z_pct, breadth,
                        price_1w, rs_1w,
                        feat_dv_n, feat_zpct_n, feat_rs_n,
                        feat_ema20, feat_ema_x, feat_vix_n,
                        feat_fii_n, feat_hmm_n, feat_pcr_n,
                        regime_label,
                        outcome_filled
                    ) VALUES (?,?,?,?, ?,?,?,?,?, ?,?, ?,?,?, ?,?,?, ?,?,?, ?, ?)
                """, [
                    as_of_date, sector, signal, score,
                    dv,
                    float(dv5d) if dv5d is not None and not (isinstance(dv5d, float) and pd.isna(dv5d)) else None,
                    z, zpct,
                    float(breadth) if breadth is not None and not (isinstance(breadth, float) and pd.isna(breadth)) else None,
                    float(p1w)  if p1w  is not None and not (isinstance(p1w,  float) and pd.isna(p1w))  else None,
                    float(rs1w) if rs1w is not None and not (isinstance(rs1w, float) and pd.isna(rs1w)) else None,
                    feat["feat_dv_n"], feat["feat_zpct_n"], feat["feat_rs_n"],
                    feat["feat_ema20"], feat["feat_ema_x"], feat["feat_vix_n"],
                    feat["feat_fii_n"], feat["feat_hmm_n"], feat["feat_pcr_n"],
                    r_label,
                    False,
                ])
                rows_written += 1

    except Exception as exc:
        import sys
        print(f"[SectorMemory] sector_log write failed: {exc}", file=sys.stderr)

    return rows_written


# ── Outcome filling ───────────────────────────────────────────────────────────

def _sector_forward_return(
    sector: str,
    signal_date: date,
    horizon_days: int,
    min_turnover_lacs: float = 1.0,
) -> Optional[float]:
    """
    Compute the turnover-weighted average price return for a sector over
    the `horizon_days` calendar days following `signal_date`.

    Uses close_price from daily_data on the last available date ≤ signal_date
    and the last available date ≤ signal_date + horizon_days.
    """
    try:
        end_date = signal_date + timedelta(days=horizon_days)
        df = query_dataframe("""
            WITH start_prices AS (
                SELECT b.symbol, b.close_price AS price_start, b.turnover_lacs
                FROM daily_data b
                JOIN v_sector_master sm ON b.symbol = sm.symbol
                WHERE sm.sector = ?
                  AND b.series = 'EQ'
                  AND b.turnover_lacs >= ?
                  AND b.trade_date = (
                      SELECT MAX(trade_date) FROM daily_data
                      WHERE trade_date <= ? AND series = 'EQ'
                  )
            ),
            end_prices AS (
                SELECT b.symbol, b.close_price AS price_end
                FROM daily_data b
                WHERE b.series = 'EQ'
                  AND b.trade_date = (
                      SELECT MAX(trade_date) FROM daily_data
                      WHERE trade_date <= ? AND series = 'EQ'
                  )
            )
            SELECT
                SUM((e.price_end - s.price_start) / NULLIF(s.price_start, 0) * 100
                    * s.turnover_lacs)
                / NULLIF(SUM(s.turnover_lacs), 0) AS fwd_ret
            FROM start_prices s
            JOIN end_prices   e ON s.symbol = e.symbol
            WHERE e.price_end IS NOT NULL AND s.price_start > 0
        """, [sector, min_turnover_lacs, signal_date, end_date])

        if df.empty or df["fwd_ret"].isna().all():
            return None
        return round(float(df["fwd_ret"].iloc[0]), 3)
    except Exception:
        return None


def _nifty_forward_return(signal_date: date, horizon_days: int) -> Optional[float]:
    """Nifty 50 return over the same horizon (for RS computation)."""
    try:
        end_date = signal_date + timedelta(days=horizon_days)
        df = query_dataframe("""
            SELECT
                (SELECT close_val FROM index_data
                 WHERE index_name = 'Nifty 50' AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1)
                AS price_end,
                (SELECT close_val FROM index_data
                 WHERE index_name = 'Nifty 50' AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1)
                AS price_start
        """, [end_date, signal_date])

        if df.empty:
            return None
        p_start = df["price_start"].iloc[0]
        p_end   = df["price_end"].iloc[0]
        if p_start and p_start > 0:
            return round((float(p_end) - float(p_start)) / float(p_start) * 100, 3)
        return None
    except Exception:
        return None


def fill_forward_outcomes(
    as_of_date:        date,
    min_turnover_lacs: float = 1.0,
) -> int:
    """
    For all unfilled sector_rotation_log rows whose signal_date is old enough
    for the outcome to be measurable, compute and store the forward returns.

    Fills 1W outcomes for records ≥ 8 calendar days old.
    Fills 2W outcomes for records ≥ 15 calendar days old.
    Fills 1M outcomes for records ≥ 31 calendar days old.
    Marks outcome_filled=TRUE only when ALL three horizons are filled.

    Returns the number of rows updated.
    """
    _ensure_schema()

    try:
        pending = query_dataframe("""
            SELECT trade_date, sector
            FROM sector_rotation_log
            WHERE outcome_filled = FALSE
              AND trade_date <= ?
            ORDER BY trade_date
        """, [as_of_date - timedelta(days=7)])
    except Exception:
        return 0

    if pending.empty:
        return 0

    updated = 0
    nifty_cache: dict[tuple, Optional[float]] = {}

    for _, prow in pending.iterrows():
        sig_date = (prow["trade_date"].date()
                    if hasattr(prow["trade_date"], "date")
                    else prow["trade_date"])
        sector   = str(prow["sector"])
        age      = (as_of_date - sig_date).days

        # Only fill the horizons that are old enough
        vals: dict[str, Optional[float]] = {}

        for horizon, col_ret, col_nifty, col_rs, min_age in [
            (7,  "fwd_ret_1w", "fwd_nifty_1w", "fwd_rs_1w", 8),
            (14, "fwd_ret_2w", "fwd_nifty_2w", "fwd_rs_2w", 15),
            (30, "fwd_ret_1m", "fwd_nifty_1m", "fwd_rs_1m", 31),
        ]:
            if age < min_age:
                continue

            ret   = _sector_forward_return(sector, sig_date, horizon, min_turnover_lacs)
            n_key = (sig_date, horizon)
            if n_key not in nifty_cache:
                nifty_cache[n_key] = _nifty_forward_return(sig_date, horizon)
            nifty = nifty_cache[n_key]

            vals[col_ret]   = ret
            vals[col_nifty] = nifty
            vals[col_rs]    = (round(ret - nifty, 3)
                               if ret is not None and nifty is not None else None)

        if not vals:
            continue

        # Mark fully filled when all 3 horizons are populated
        all_filled = (
            vals.get("fwd_ret_1w") is not None and
            vals.get("fwd_ret_2w") is not None and
            vals.get("fwd_ret_1m") is not None
        ) if age >= 31 else False

        set_clauses = ", ".join(f"{k} = ?" for k in vals)
        set_vals    = list(vals.values())
        if all_filled:
            set_clauses += ", outcome_filled = TRUE"

        try:
            repo = get_repository()
            with repo._cm.connect() as conn:
                conn.execute(
                    f"UPDATE sector_rotation_log SET {set_clauses} "
                    f"WHERE trade_date = ? AND sector = ?",
                    set_vals + [sig_date, sector],
                )
            updated += 1
        except Exception as exc:
            import sys
            print(f"[SectorMemory] outcome fill failed for {sig_date} {sector}: {exc}",
                  file=sys.stderr)

    return updated


# ── Memory retrieval ──────────────────────────────────────────────────────────

# Columns the similarity retrieval needs. ONE definition: the per-sector read
# and the whole-log snapshot must return the same shape, or narrowing the
# snapshot in pandas would not be interchangeable with querying per sector.
_MEMORY_LOG_COLS = """trade_date, sector, signal, regime_label, accum_score,
                   dv_ratio, z_pct,
                   feat_dv_n, feat_zpct_n, feat_rs_n,
                   feat_ema20, feat_ema_x, feat_vix_n,
                   feat_fii_n, feat_hmm_n, feat_pcr_n,
                   fwd_ret_1w, fwd_ret_2w, fwd_ret_1m,
                   fwd_rs_1w,  fwd_rs_2w,  fwd_rs_1m"""


def read_memory_log_snapshot(cutoff: date) -> pd.DataFrame:
    """Every filled log row before `cutoff`, for all sectors, in one read.

    apply_memory_overlay asks get_sector_memory_context for ~24 sectors with the
    same as_of_date, and each call was issuing its own single-sector query. The
    rows are the same either way: this is the identical predicate minus the
    `sector = ?` term, with the same ORDER BY, so masking by sector afterwards
    yields the same frame in the same order.

    Callers must treat the result as read-only — it is shared across sectors.
    """
    return query_dataframe(f"""
        SELECT {_MEMORY_LOG_COLS}
        FROM sector_rotation_log
        WHERE outcome_filled = TRUE
          AND trade_date < ?
        ORDER BY trade_date DESC
    """, [cutoff])


def get_sector_memory_context(
    as_of_date:    date,
    sector:        str,
    signal:        str,
    regime_label:  str,
    dv_ratio:      float,
    z_pct:         float,
    rs_1w:         Optional[float],
    ema20_above:   bool,
    ema_cross_bull: Optional[bool],
    vix:           Optional[float],
    fii_5d_cr:     Optional[float],
    hmm_state:     Optional[str],
    pcr:           Optional[float],
    same_signal_only: bool = False,
    _log_snapshot: Optional[pd.DataFrame] = None,
) -> SectorMemoryContext:
    """
    Find the K most similar historical setups for this (sector, conditions)
    and return statistical outcome summaries.

    Parameters
    ----------
    same_signal_only : if True, only consider records with the exact same
                       signal label; if False, consider all records for this
                       sector (broader sample but less specific).

    Returns SectorMemoryContext.  On failure returns an empty context with
    .error set so the caller can degrade gracefully.
    """
    ctx = SectorMemoryContext(sector=sector, signal=signal, n_similar=0, n_filled=0)

    try:
        today_feat = _normalize(
            dv_ratio       = dv_ratio,
            z_pct          = z_pct,
            rs_1w          = rs_1w,
            ema20_above    = ema20_above,
            ema_cross_bull = ema_cross_bull,
            vix            = vix,
            fii_5d_cr      = fii_5d_cr,
            hmm_state      = hmm_state,
            pcr            = pcr,
        )
    except Exception as exc:
        ctx.error = f"Feature normalisation failed: {exc}"
        return ctx

    # Fetch all filled historical records for this sector.
    #
    # POINT-IN-TIME GATE: a setup is only usable if its forward window had fully
    # RESOLVED by as_of_date — otherwise viewing a historical date would consume
    # outcomes that were not yet knowable then (look-ahead leakage). outcome_filled
    # alone is insufficient because it is set at fill/backfill time (≈ now), not
    # relative to as_of. So we additionally require trade_date ≤ as_of − 31 days
    # (the longest, 1-month, horizon). For live use (as_of = today) this is a no-op
    # since sub-31-day setups are not yet outcome_filled; for historical / backtest
    # use it is what makes the memory honest.
    _OUTCOME_HORIZON_DAYS = 31
    try:
        _cutoff = as_of_date - timedelta(days=_OUTCOME_HORIZON_DAYS)
        if _log_snapshot is not None:
            # Caller already read the whole log for this cutoff (see
            # apply_memory_overlay). Narrowing in pandas selects the same rows
            # in the same order — the snapshot is fetched ORDER BY trade_date
            # DESC and boolean masking is order-preserving.
            hist = _log_snapshot[_log_snapshot["sector"] == sector]
            if same_signal_only:
                hist = hist[hist["signal"] == signal]
        else:
            sig_filter = "AND signal = ?" if same_signal_only else ""
            params     = [sector, _cutoff]
            if same_signal_only:
                params.insert(1, signal)

            hist = query_dataframe(f"""
                SELECT {_MEMORY_LOG_COLS}
                FROM sector_rotation_log
                WHERE sector = ?
                  {sig_filter}
                  AND outcome_filled = TRUE
                  AND trade_date < ?
                ORDER BY trade_date DESC
            """, params)
    except Exception as exc:
        ctx.error = f"DB read failed: {exc}"
        return ctx

    if hist.empty:
        ctx.note = (f"No historical records for {sector} yet. "
                    "Memory will build up automatically after 30+ trading days.")
        return ctx

    ctx.n_similar = len(hist)

    # These columns are stored float32. The old path read rows with iterrows(),
    # which upcast them to Python floats before they were handed to
    # SimilarSetup; .iloc[] does not, so without this the setups would carry
    # float32 and could format differently at high precision. Widen once for the
    # whole frame rather than per row.
    _f32 = list(hist.select_dtypes("float32").columns)
    if _f32:
        hist = hist.astype({c: "float64" for c in _f32})

    # Similarity for every historical record, scored as an array rather than by
    # iterating rows. This is called once per sector (~24 per overlay) over that
    # sector's whole filled history, and the row loop was 16,368 iterrows and
    # the single largest cost in the Smart Money panel.
    #
    # It is the same arithmetic as _similarity, which is retained as the scalar
    # definition and pinned against this path in tests/test_sector_memory_sim.py:
    #     distance = sqrt( Sum w_i (f1_i - f2_i)^2 ),  sim = round(1 - distance, 4)
    #
    # NULL handling matches by construction, though the two get there
    # differently. These columns are DuckDB floats, so a NULL arrives as NaN,
    # never as None: the scalar path then puts NaN into hist_feat (NaN is not
    # None), every comparison against it is False, and max(0.0, 1.0 - NaN)
    # returns 0.0, which fails the _MIN_SIMILARITY test. Here the NaN propagates
    # and `NaN >= _MIN_SIMILARITY` is False. Both drop the row.
    #
    # The `len(hist_feat) < 6` skip in the old loop counted non-None entries, so
    # with float columns it could never fire; it is not reproduced. If this
    # table ever gains an object-dtype feature column that holds real Nones, the
    # two paths WOULD diverge (the scalar one defaults missing keys to 0.5) and
    # this needs revisiting.
    feat_cols = list(_FEAT_WEIGHTS.keys())
    _sims = _similarity_many(today_feat, hist[feat_cols])
    with np.errstate(invalid="ignore"):
        _ok = np.flatnonzero(_sims >= _MIN_SIMILARITY)

    if _ok.size == 0:
        ctx.note = (f"No sufficiently similar historical setups found for {sector} + {signal}. "
                    "Current conditions may be unprecedented in the logged history.")
        return ctx

    # Descending by similarity. STABLE, because list.sort(reverse=True) leaves
    # equal elements in their original order (trade_date DESC) rather than
    # reversing them — and ties decide which episode the gap rule keeps.
    _order = _ok[np.argsort(-_sims[_ok], kind="stable")]

    # Rows are materialised lazily in the walk below: it stops at _TOP_K kept
    # episodes, so building a Series for every survivor would be wasted work.
    scored = ((float(_sims[i]), hist.iloc[i]) for i in _order)

    # ── Independent-episode de-duplication ────────────────────────────────────
    # Recording every trading day means neighbours are often consecutive days
    # whose forward windows overlap ~80% — counting them all fakes statistical
    # confidence ("10/14 positive" backed by ~2 real events). Greedily keep, in
    # similarity order, only setups ≥ _EPISODE_GAP_DAYS apart so each contributes
    # an (almost) independent forward outcome. n_episodes (k_eff) is the honest
    # sample size used downstream for shrinkage.
    kept: list[tuple[float, pd.Series]] = []
    kept_dates: list[date] = []
    for sim, hrow in scored:
        td_raw = hrow["trade_date"]
        td_d = td_raw.date() if hasattr(td_raw, "date") else td_raw
        if all(abs((td_d - kd).days) >= _EPISODE_GAP_DAYS for kd in kept_dates):
            kept.append((sim, hrow))
            kept_dates.append(td_d)
        if len(kept) >= _TOP_K:
            break

    top = kept
    ctx.n_similar   = len(top)
    ctx.n_episodes  = len(top)   # already de-duplicated → these ARE independent episodes
    ctx.avg_similarity = round(sum(s for s, _ in top) / len(top), 4) if top else 0.0

    # Build similar setup list and compute stats
    filled: list[SimilarSetup] = []
    by_regime: dict[str, int] = {}

    for sim, hrow in top:
        td = hrow["trade_date"]
        if hasattr(td, "date"):
            td = td.date()

        setup = SimilarSetup(
            trade_date   = td,
            sector       = sector,
            signal       = str(hrow.get("signal", "")),
            regime_label = str(hrow.get("regime_label", "—")),
            similarity   = round(sim, 3),
            dv_ratio     = float(hrow.get("dv_ratio") or 1.0),
            z_pct        = float(hrow.get("z_pct") or 0.5),
            fwd_ret_1w   = hrow.get("fwd_ret_1w"),
            fwd_ret_2w   = hrow.get("fwd_ret_2w"),
            fwd_ret_1m   = hrow.get("fwd_ret_1m"),
            fwd_rs_1w    = hrow.get("fwd_rs_1w"),
            fwd_rs_2w    = hrow.get("fwd_rs_2w"),
            fwd_rs_1m    = hrow.get("fwd_rs_1m"),
        )
        rl = setup.regime_label
        by_regime[rl] = by_regime.get(rl, 0) + 1
        filled.append(setup)

    ctx.similar_setups = filled
    ctx.n_filled       = len(filled)
    ctx.by_regime      = by_regime

    if len(filled) < _MIN_RECORDS:
        ctx.note = (f"Only {len(filled)} similar setups found — "
                    f"memory will improve as more data accumulates.")

    # Compute statistics from filled outcomes
    def _stats(values: list[float]) -> tuple[float, float, float, float]:
        """Return (median, p25, p75, pct_positive)."""
        a = np.array(values, dtype=float)
        return (
            round(float(np.median(a)),        2),
            round(float(np.percentile(a, 25)), 2),
            round(float(np.percentile(a, 75)), 2),
            round(float((a > 0).mean() * 100), 1),
        )

    rets_1w = [s.fwd_ret_1w for s in filled if s.fwd_ret_1w is not None]
    rets_2w = [s.fwd_ret_2w for s in filled if s.fwd_ret_2w is not None]
    rets_1m = [s.fwd_ret_1m for s in filled if s.fwd_ret_1m is not None]
    rs_1w_l = [s.fwd_rs_1w  for s in filled if s.fwd_rs_1w  is not None]
    rs_2w_l = [s.fwd_rs_2w  for s in filled if s.fwd_rs_2w  is not None]
    rs_1m_l = [s.fwd_rs_1m  for s in filled if s.fwd_rs_1m  is not None]

    if len(rets_1w) >= _MIN_RECORDS:
        ctx.ret_1w_median, ctx.ret_1w_p25, ctx.ret_1w_p75, ctx.ret_1w_pos_pct = _stats(rets_1w)
    if len(rets_2w) >= _MIN_RECORDS:
        ctx.ret_2w_median, ctx.ret_2w_p25, ctx.ret_2w_p75, ctx.ret_2w_pos_pct = _stats(rets_2w)
    if len(rets_1m) >= _MIN_RECORDS:
        ctx.ret_1m_median, _, _, ctx.ret_1m_pos_pct = _stats(rets_1m)
    if len(rs_1w_l) >= _MIN_RECORDS:
        ctx.rs_1w_median = round(float(np.median(rs_1w_l)), 2)
        ctx.rs_1w_pos_pct = round(float((np.array(rs_1w_l) > 0).mean() * 100), 1)
    if len(rs_2w_l) >= _MIN_RECORDS:
        ctx.rs_2w_median = round(float(np.median(rs_2w_l)), 2)
        ctx.rs_2w_pos_pct = round(float((np.array(rs_2w_l) > 0).mean() * 100), 1)
    if len(rs_1m_l) >= _MIN_RECORDS:
        ctx.rs_1m_median = round(float(np.median(rs_1m_l)), 2)

    return ctx


# ── Bulk backfill ─────────────────────────────────────────────────────────────

def backfill_sector_memory(
    end_date:          date,
    start_date:        Optional[date]  = None,
    min_turnover_lacs: float = 1.0,
    fill_outcomes:     bool  = True,
    verbose:           bool  = True,
    replace:           bool  = True,
) -> dict[str, int]:
    """
    Backfill sector_rotation_log + sector_regime_log for every trading day in
    [start_date, end_date] by REPLAYING record_daily_snapshot() per date.

    Why replay rather than a bulk SQL pass: the stored fingerprint must match the
    live daily job EXACTLY. The previous bulk implementation re-derived features
    independently and drifted out of sync — it un-normalised FII flow with the
    wrong scale, never computed 1-week relative strength (feat_rs_n was dead),
    produced degenerate signal labels (price unknown → no Secret/Confirmed), and
    sourced regime dims only from prediction_log, which starts 2026-04. Replaying
    the real recording path makes history identical to live by construction.
    Regime VIX/FII/EMA come from raw tables via get_market_regime() and so are
    populated across the full range; only HMM/PCR fall back to neutral on the
    older dates that genuinely predate the prediction engine.

    Parameters
    ----------
    end_date        : most recent date to include (typically today)
    start_date      : earliest date; defaults to end_date − 365 days
    min_turnover_lacs : turnover filter passed through to get_sector_rotation
    fill_outcomes   : whether to fill forward returns after recording
    verbose         : print progress
    replace         : re-record dates already present (True for a clean rebuild)

    Returns
    -------
    dict with keys 'dates_processed', 'rows_recorded', 'outcomes_filled'
    """
    _ensure_schema()

    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # Trading days actually present in daily_data within the window.
    try:
        dates_df = query_dataframe("""
            SELECT DISTINCT trade_date
            FROM daily_data
            WHERE trade_date >= ? AND trade_date <= ? AND series = 'EQ'
            ORDER BY trade_date
        """, [start_date, end_date])
    except Exception as exc:
        if verbose:
            print(f"[SectorMemory] Trading-date scan failed: {exc}")
        return {"dates_processed": 0, "rows_recorded": 0, "outcomes_filled": 0}

    if dates_df.empty:
        if verbose:
            print("[SectorMemory] No trading dates in range.")
        return {"dates_processed": 0, "rows_recorded": 0, "outcomes_filled": 0}

    trade_dates = [
        d.date() if hasattr(d, "date") else d
        for d in dates_df["trade_date"].tolist()
    ]
    if verbose:
        print(f"[SectorMemory] Backfill: replaying {len(trade_dates)} dates "
              f"{trade_dates[0]} → {trade_dates[-1]}")

    rows_recorded = 0
    processed     = 0
    for td in trade_dates:
        try:
            rows_recorded += record_daily_snapshot(
                td, min_turnover_lacs=min_turnover_lacs, replace=replace,
            )
            processed += 1
            if verbose and processed % 25 == 0:
                print(f"[SectorMemory]   {processed}/{len(trade_dates)} dates "
                      f"({rows_recorded} rows)")
        except Exception as exc:
            if verbose:
                print(f"[SectorMemory] Record failed for {td}: {exc}")

    if verbose:
        print(f"[SectorMemory] Recorded {rows_recorded} rows across {processed} dates")

    outcomes_filled = 0
    if fill_outcomes:
        if verbose:
            print("[SectorMemory] Filling forward outcomes…")
        outcomes_filled = fill_forward_outcomes(end_date, min_turnover_lacs)
        if verbose:
            print(f"[SectorMemory] Filled outcomes for {outcomes_filled} records")

    return {
        "dates_processed": processed,
        "rows_recorded":   rows_recorded,
        "outcomes_filled": outcomes_filled,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Memory-sharpened signal overlay
# ──────────────────────────────────────────────────────────────────────────────
# Turns the passive "here's what history did" readout into an ACTIVE conviction
# adjustment on the cross-sectional accumulation score. Design contract:
#   • Memory is a SHRUNK overlay, never a replacement. It tilts the IC-validated
#     accum_score by at most ±_ALPHA and only in proportion to evidence quality.
#   • It scales conviction; it NEVER flips a signal's direction.
#   • Thin / autocorrelated / low-similarity evidence → edge collapses to 0 and
#     the base score is used untouched ("Unproven").
# ══════════════════════════════════════════════════════════════════════════════

_ALPHA                  = 0.25   # max fractional tilt memory may apply to accum_score
_K0                     = 8.0    # episode count at which sample-confidence = 0.5
_RS_SCALE               = 2.0    # forward RS% that ~saturates the tanh effect size
_SPREAD_SCALE           = 6.0    # outcome IQR% reference for the consistency factor
_MIN_EPISODES_FOR_EDGE  = 5      # below this k_eff → edge forced to 0 (Unproven)
_HIGH_CONV_EPISODES     = 8      # k_eff needed to qualify for High Conviction

_BULL_SIGNALS = {"🔥 Secret Accumulation", "✅ Confirmed Accumulation", "👀 Early Accumulation"}
_BEAR_SIGNALS = {"❌ Active Selling", "⚠️ Distribution Trap", "📉 Weakening"}


@dataclass
class MemoryEdge:
    """Memory-derived conviction adjustment for one sector's current setup."""
    edge:             float          # shrunk signed edge in [-1, +1]
    confidence:       float          # evidence-quality weight in [0, 1]
    conviction:       str            # HIGH | CONFIRM | NEUTRAL | DISAGREE | UNPROVEN
    score_multiplier: float          # 1 + _ALPHA*edge — multiply accum_score by this
    expected_rs:      Optional[float] = None   # median forward RS vs Nifty (2W) %
    hit_rate:         Optional[float] = None   # % of episodes that outperformed Nifty
    n_episodes:       int = 0                   # independent episodes (k_eff)
    basis:            str = ""                  # human-readable evidence summary


def _classify_conviction(edge: float, k: int, base_bull: bool, base_bear: bool) -> str:
    """Label the edge RELATIVE to the base signal's intended direction."""
    if base_bull:
        if edge >= 0.30 and k >= _HIGH_CONV_EPISODES:
            return "HIGH"        # history strongly rewards this accumulation setup
        if edge >= 0.10:
            return "CONFIRM"
        if edge <= -0.20:
            return "DISAGREE"    # looks like a buy today but historically faded
        return "NEUTRAL"
    if base_bear:
        # For an avoid/weakness signal, NEGATIVE forward RS confirms the thesis.
        if edge <= -0.30 and k >= _HIGH_CONV_EPISODES:
            return "HIGH"        # history strongly rewards avoiding/shorting this
        if edge <= -0.10:
            return "CONFIRM"
        if edge >= 0.20:
            return "DISAGREE"    # flagged weak today but historically bounced
        return "NEUTRAL"
    return "NEUTRAL"


def compute_memory_edge(ctx: SectorMemoryContext, base_signal: str) -> MemoryEdge:
    """
    Convert a SectorMemoryContext into a shrunk conviction adjustment.

    edge = (0.6·tanh(rs/RS_SCALE) + 0.4·(2·hit−1)) × confidence
    confidence = k/(k+K0) × avg_similarity × consistency,  all ∈ [0,1]

    Below _MIN_EPISODES_FOR_EDGE independent episodes the edge is forced to 0 so
    a thin sample can never move the validated base score.
    """
    rs_med = ctx.rs_2w_median if ctx.rs_2w_median is not None else ctx.rs_1w_median
    hit    = ctx.rs_2w_pos_pct if ctx.rs_2w_pos_pct is not None else ctx.rs_1w_pos_pct
    k      = int(ctx.n_episodes or 0)

    if k < _MIN_EPISODES_FOR_EDGE or rs_med is None:
        return MemoryEdge(
            edge=0.0, confidence=0.0, conviction="UNPROVEN", score_multiplier=1.0,
            expected_rs=rs_med, hit_rate=hit, n_episodes=k,
            basis=f"only {k} independent episode(s) — using base score",
        )

    eff = math.tanh(rs_med / _RS_SCALE)                       # effect size  [-1,1]
    hr  = (2.0 * (hit / 100.0) - 1.0) if hit is not None else 0.0  # hit rate [-1,1]
    raw = 0.6 * eff + 0.4 * hr

    samp = k / (k + _K0)
    consistency = 1.0
    if ctx.ret_2w_p25 is not None and ctx.ret_2w_p75 is not None:
        spread = abs(ctx.ret_2w_p75 - ctx.ret_2w_p25)
        consistency = 1.0 / (1.0 + spread / _SPREAD_SCALE)
    conf = max(0.0, min(1.0, samp * float(ctx.avg_similarity or 0.0) * consistency))

    edge = max(-1.0, min(1.0, raw * conf))
    mult = 1.0 + _ALPHA * edge

    conviction = _classify_conviction(
        edge, k, base_signal in _BULL_SIGNALS, base_signal in _BEAR_SIGNALS,
    )
    basis = (f"{k} episodes · {rs_med:+.1f}% med RS·2W · "
             f"{(hit or 0):.0f}% outperformed · conf {conf:.2f}")
    return MemoryEdge(
        edge=round(edge, 3), confidence=round(conf, 3), conviction=conviction,
        score_multiplier=round(mult, 4), expected_rs=round(rs_med, 2),
        hit_rate=hit, n_episodes=k, basis=basis,
    )


def apply_memory_overlay(
    rotation_df: pd.DataFrame,
    as_of_date:  date,
    regime:      dict,
) -> pd.DataFrame:
    """
    Attach a memory-sharpened conviction layer to a get_sector_rotation() frame.

    Adds columns: memory_edge, memory_conf, conviction, expected_rs_2w,
    mem_episodes, mem_basis, adj_score (= accum_score × score_multiplier).
    Returns a NEW frame re-sorted by adj_score descending. Pure analytics — no
    Streamlit, no lookahead (retrieval is gated to trade_date < as_of_date).
    """
    if rotation_df is None or rotation_df.empty:
        return rotation_df

    df = rotation_df.copy()

    ema20_above    = (regime.get("nifty_vs_ema20") == "ABOVE")
    _ema_x         = regime.get("nifty_vs_ema50")
    ema_cross_bull = (True if _ema_x == "ABOVE" else False if _ema_x == "BELOW" else None)
    vix            = float(regime["vix"])       if regime.get("vix")       is not None else None
    fii_5d_cr      = float(regime["fii_5d_cr"]) if regime.get("fii_5d_cr") is not None else None
    hmm_state      = regime.get("hmm_state")
    regime_label   = regime.get("regime", "SIDEWAYS")

    # One read of the log for every sector below, instead of one per sector.
    # Kept best-effort: if it fails, each call falls back to its own query and
    # the overlay still renders — this is a fetch optimisation, not a
    # behavioural gate.
    try:
        _snapshot = read_memory_log_snapshot(as_of_date - timedelta(days=31))
    except Exception:
        _snapshot = None

    edges, confs, convs, exp_rs, eps, bases, adj = [], [], [], [], [], [], []
    for _, row in df.iterrows():
        signal = str(row.get("signal", ""))
        rs1w   = row.get("rs_1w")
        rs1w   = (float(rs1w) if rs1w is not None
                  and not (isinstance(rs1w, float) and pd.isna(rs1w)) else None)
        try:
            ctx = get_sector_memory_context(
                as_of_date     = as_of_date,
                sector         = str(row["sector"]),
                signal         = signal,
                regime_label   = regime_label,
                dv_ratio       = float(row.get("dv_ratio", 1.0) or 1.0),
                z_pct          = float(row.get("z_pct", 0.5) or 0.5),
                rs_1w          = rs1w,
                ema20_above    = ema20_above,
                ema_cross_bull = ema_cross_bull,
                vix            = vix,
                fii_5d_cr      = fii_5d_cr,
                hmm_state      = hmm_state,
                pcr            = None,
                _log_snapshot  = _snapshot,
            )
            me = compute_memory_edge(ctx, signal)
        except Exception:
            me = MemoryEdge(edge=0.0, confidence=0.0, conviction="UNPROVEN",
                            score_multiplier=1.0, basis="memory unavailable")

        base_score = float(row.get("accum_score", 0.0) or 0.0)
        edges.append(me.edge);        confs.append(me.confidence)
        convs.append(me.conviction);  exp_rs.append(me.expected_rs)
        eps.append(me.n_episodes);    bases.append(me.basis)
        adj.append(round(base_score * me.score_multiplier, 1))

    df["memory_edge"]    = edges
    df["memory_conf"]    = confs
    df["conviction"]     = convs
    df["expected_rs_2w"] = exp_rs
    df["mem_episodes"]   = eps
    df["mem_basis"]      = bases
    df["adj_score"]      = adj

    return df.sort_values("adj_score", ascending=False).reset_index(drop=True)
