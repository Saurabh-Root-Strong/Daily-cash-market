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
    "record_daily_snapshot",
    "fill_forward_outcomes",
    "get_sector_memory_context",
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

_TOP_K          = 20    # retrieve up to 20 most similar historical setups
_MIN_SIMILARITY = 0.40  # discard very dissimilar days (< 40% similarity)
_MIN_RECORDS    = 5     # need at least 5 similar filled records to show memory


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
) -> int:
    """
    Compute today's sector rotation signals and market regime, then write
    one row per sector to sector_rotation_log and one row to sector_regime_log.

    Returns the number of sector rows written (0 on failure or if already done).
    Called by sector_memory_daily.py after nightly_sync completes.
    """
    _ensure_schema()

    # Idempotency guard — skip if today is already recorded
    try:
        existing = query_dataframe(
            "SELECT COUNT(*) AS n FROM sector_rotation_log WHERE trade_date = ?",
            [as_of_date],
        )
        if not existing.empty and int(existing["n"].iloc[0]) > 0:
            return 0
    except Exception:
        pass

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
                r_vix, regime.get("vix_trend"),
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
                JOIN sector_master sm ON b.symbol = sm.symbol
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

    # Fetch all filled historical records for this sector
    try:
        sig_filter = "AND signal = ?" if same_signal_only else ""
        params     = [sector, as_of_date - timedelta(days=1)]
        if same_signal_only:
            params.insert(1, signal)

        hist = query_dataframe(f"""
            SELECT trade_date, sector, signal, regime_label, accum_score,
                   dv_ratio, z_pct,
                   feat_dv_n, feat_zpct_n, feat_rs_n,
                   feat_ema20, feat_ema_x, feat_vix_n,
                   feat_fii_n, feat_hmm_n, feat_pcr_n,
                   fwd_ret_1w, fwd_ret_2w, fwd_ret_1m,
                   fwd_rs_1w,  fwd_rs_2w,  fwd_rs_1m
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

    # Compute similarity for each historical record
    scored: list[tuple[float, pd.Series]] = []
    feat_cols = list(_FEAT_WEIGHTS.keys())

    for _, hrow in hist.iterrows():
        hist_feat = {k: float(hrow[k]) for k in feat_cols if hrow[k] is not None}
        if len(hist_feat) < 6:   # skip rows with too many nulls
            continue
        sim = _similarity(today_feat, hist_feat)
        if sim >= _MIN_SIMILARITY:
            scored.append((sim, hrow))

    if not scored:
        ctx.note = (f"No sufficiently similar historical setups found for {sector} + {signal}. "
                    "Current conditions may be unprecedented in the logged history.")
        return ctx

    # Sort by similarity descending, take top K
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:_TOP_K]
    ctx.n_similar = len(top)

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
) -> dict[str, int]:
    """
    Backfill sector_rotation_log for all available trading dates in [start_date, end_date].

    Uses DuckDB window functions to compute rolling delivery metrics efficiently
    across all sectors and all dates in a SINGLE SQL pass — O(n) not O(n²).

    After recording all signals, optionally fills forward outcomes for records
    old enough to have measurable outcomes.

    Parameters
    ----------
    end_date        : most recent date to include (typically today)
    start_date      : earliest date; defaults to end_date − 365 days
    min_turnover_lacs : turnover filter for delivery data
    fill_outcomes   : whether to fill forward returns after recording
    verbose         : print progress

    Returns
    -------
    dict with keys 'dates_processed', 'rows_recorded', 'outcomes_filled'
    """
    _ensure_schema()

    if start_date is None:
        start_date = end_date - timedelta(days=365)

    # ── Step 1: Compute rolling sector metrics for ALL dates in a single query ─
    # This is O(n) using DuckDB window functions — far more efficient than
    # calling get_sector_rotation() for each date individually (O(n × k) queries).
    if verbose:
        print(f"[SectorMemory] Backfill: computing rolling metrics {start_date} → {end_date}")

    try:
        rolling_df = query_dataframe("""
            WITH daily_sector AS (
                SELECT sm.sector, b.trade_date,
                       SUM(b.deliv_per * b.turnover_lacs)
                           / NULLIF(SUM(b.turnover_lacs), 0)             AS wtd_deliv_pct,
                       SUM(b.deliv_per / 100.0 * b.turnover_lacs) / 100  AS deliv_value_cr,
                       COUNT(DISTINCT b.symbol)                           AS n_stocks,
                       SUM(CASE WHEN b.deliv_per > 0 THEN 1 ELSE 0 END) * 1.0
                           / NULLIF(COUNT(*), 0)                         AS breadth_raw
                FROM daily_data b
                JOIN sector_master sm ON b.symbol = sm.symbol
                WHERE b.series IN ('EQ', 'SM', 'ST')
                  AND sm.sector NOT IN ('ETF', 'Others')
                  AND b.turnover_lacs >= ?
                  AND b.trade_date >= (? - INTERVAL 150 DAY)
                  AND b.trade_date <= ?
                GROUP BY sm.sector, b.trade_date
            ),
            rolling AS (
                SELECT *,
                       AVG(deliv_value_cr)    OVER w100 AS avg_100d,
                       STDDEV(deliv_value_cr) OVER w100 AS std_100d,
                       AVG(deliv_value_cr)    OVER w5   AS avg_5d,
                       AVG(wtd_deliv_pct)     OVER w100 AS avg_wtd_pct_100d
                FROM daily_sector
                WINDOW
                    w100 AS (PARTITION BY sector ORDER BY trade_date
                             ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING),
                    w5   AS (PARTITION BY sector ORDER BY trade_date
                             ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
            )
            SELECT sector, trade_date,
                   deliv_value_cr,
                   avg_100d, std_100d, avg_5d, avg_wtd_pct_100d,
                   CASE WHEN avg_100d > 0
                        THEN deliv_value_cr / avg_100d
                        ELSE NULL END                                     AS dv_ratio,
                   CASE WHEN std_100d > 0
                        THEN (deliv_value_cr - avg_100d) / std_100d
                        ELSE NULL END                                     AS z_score,
                   CASE WHEN avg_100d > 0
                        THEN (avg_5d / (avg_100d / 100 * 5))
                        ELSE NULL END                                     AS dv_ratio_5d,
                   breadth_raw                                            AS breadth,
                   wtd_deliv_pct
            FROM rolling
            WHERE avg_100d IS NOT NULL
              AND trade_date >= ?
              AND trade_date <= ?
            ORDER BY trade_date, sector
        """, [min_turnover_lacs, start_date, end_date, start_date, end_date])
    except Exception as exc:
        if verbose:
            print(f"[SectorMemory] Rolling metrics query failed: {exc}")
        return {"dates_processed": 0, "rows_recorded": 0, "outcomes_filled": 0}

    if rolling_df.empty:
        if verbose:
            print("[SectorMemory] No data returned from rolling metrics query.")
        return {"dates_processed": 0, "rows_recorded": 0, "outcomes_filled": 0}

    # ── Step 2: Get Nifty 50 returns for each date (for RS computation) ──────
    try:
        nifty_df = query_dataframe("""
            SELECT trade_date, close_val,
                   LAG(close_val, 7)  OVER (ORDER BY trade_date) AS close_7d_ago
            FROM index_data
            WHERE index_name = 'Nifty 50'
              AND trade_date >= (? - INTERVAL 15 DAY)
              AND trade_date <= ?
            ORDER BY trade_date
        """, [start_date, end_date])
        nifty_df["trade_date"] = pd.to_datetime(nifty_df["trade_date"]).dt.date
        nifty_1w_map = {}
        for _, nr in nifty_df.iterrows():
            if nr["close_7d_ago"] and nr["close_7d_ago"] > 0:
                nifty_1w_map[nr["trade_date"]] = round(
                    (nr["close_val"] - nr["close_7d_ago"]) / nr["close_7d_ago"] * 100, 2)
    except Exception:
        nifty_1w_map = {}

    # ── Step 3: Get regime fingerprints from prediction_log ──────────────────
    try:
        regime_df = query_dataframe("""
            SELECT trade_date, hmm_state, feat_pcr,
                   feat_vix, feat_fii_5d_cumul, feat_breadth
            FROM prediction_log
            WHERE fno_symbol = 'NIFTY'
              AND trade_date >= ?  AND trade_date <= ?
            ORDER BY trade_date
        """, [start_date, end_date])
        regime_df["trade_date"] = pd.to_datetime(regime_df["trade_date"]).dt.date
        regime_map = regime_df.set_index("trade_date").to_dict("index")
    except Exception:
        regime_map = {}

    # ── Step 4: Get EMA positions for each date ───────────────────────────────
    try:
        nifty_hist = query_dataframe("""
            SELECT trade_date, close_val FROM index_data
            WHERE index_name = 'Nifty 50'
              AND trade_date >= (? - INTERVAL 200 DAY)
              AND trade_date <= ?
            ORDER BY trade_date
        """, [start_date, end_date])
        nifty_hist["trade_date"] = pd.to_datetime(nifty_hist["trade_date"]).dt.date
        nifty_hist["ema20"] = nifty_hist["close_val"].ewm(span=20, adjust=False).mean()
        nifty_hist["ema50"] = nifty_hist["close_val"].ewm(span=50, adjust=False).mean()
        nifty_hist["ema20_above"] = nifty_hist["close_val"] > nifty_hist["ema20"]
        nifty_hist["ema_cross"]   = nifty_hist["ema20"] > nifty_hist["ema50"]
        ema_map = nifty_hist.set_index("trade_date")[
            ["ema20_above", "ema_cross"]].to_dict("index")
    except Exception:
        ema_map = {}

    # ── Step 5: Process each date and write to sector_rotation_log ───────────
    rolling_df["trade_date"] = pd.to_datetime(rolling_df["trade_date"]).dt.date
    all_dates   = sorted(rolling_df["trade_date"].unique())
    rows_written = 0

    for td in all_dates:
        day_df = rolling_df[rolling_df["trade_date"] == td].copy()
        if day_df.empty:
            continue

        # Cross-sectional z-percentile for this date
        z_scores = day_df["z_score"].dropna()
        if len(z_scores) > 1:
            day_df["z_pct"] = day_df["z_score"].rank(pct=True)
        else:
            day_df["z_pct"] = 0.5

        # Nifty RS for this date
        nifty_1w = nifty_1w_map.get(td)

        # Regime features
        rg       = regime_map.get(td, {})
        hmm      = rg.get("hmm_state", "Sideways")
        pcr      = rg.get("feat_pcr")
        vix_raw  = rg.get("feat_vix")
        fii_5d   = rg.get("feat_fii_5d_cumul")
        if fii_5d is not None:
            fii_5d = float(fii_5d) * 10_000  # feat_fii_5d_cumul is normalised; undo

        em       = ema_map.get(td, {})
        ema20_ab = bool(em.get("ema20_above", False))
        ema_cr   = (True if em.get("ema_cross") else False) if em else None

        # Derive regime label from score (simplified for backfill)
        score = 5.0
        if ema20_ab:   score += 1.5
        else:          score -= 1.5
        if ema_cr is True:  score += 1.0
        elif ema_cr is False: score -= 1.0
        if hmm == "Bull":  score += 1.0
        elif hmm == "Bear": score -= 1.0
        score = max(0, min(10, score))
        r_label = ("BULL" if score >= 7.5 else "CAUTIOUS BULL" if score >= 6.0
                   else "SIDEWAYS" if score >= 4.0 else "CAUTION" if score >= 2.5 else "BEAR")

        try:
            repo = get_repository()
            with repo._cm.connect() as conn:
                for _, sr in day_df.iterrows():
                    sector  = str(sr["sector"])
                    dv      = sr["dv_ratio"]
                    dv5d    = sr["dv_ratio_5d"]
                    z       = sr["z_score"]
                    zpct    = sr["z_pct"]
                    breadth = sr["breadth"]
                    p1w     = None   # not computable in bulk without another query
                    rs1w    = None
                    if dv is None or pd.isna(dv):
                        continue

                    # Classify signal (same gates as get_sector_rotation)
                    z_pct_v = float(zpct) if not pd.isna(zpct) else 0.5
                    dv5d_v  = float(dv5d) if dv5d is not None and not pd.isna(dv5d) else 1.0
                    wtd_pct = sr.get("wtd_deliv_pct")
                    avg_pct = sr.get("avg_wtd_pct_100d")
                    pct_ok  = (float(wtd_pct) >= float(avg_pct) * 0.85
                               if wtd_pct is not None and avg_pct is not None
                                  and not pd.isna(wtd_pct) and not pd.isna(avg_pct)
                                  and avg_pct > 0
                               else True)
                    d_surge_conf = (z_pct_v >= 0.50 and dv5d_v >= 1.15) or (z_pct_v >= 0.90 and dv5d_v >= 0.9)
                    d_weak_conf  = z_pct_v <= 0.25 and dv5d_v <= 0.90
                    d_surge      = z_pct_v >= 0.50
                    p1w_val      = 0.0   # unknown at backfill time

                    if d_surge_conf and pct_ok:
                        signal = ("🔥 Secret Accumulation" if p1w_val < -1.0
                                  else "✅ Confirmed Accumulation" if p1w_val > 1.0
                                  else "👀 Early Accumulation")
                    elif d_surge and not pct_ok:
                        signal = "📊 Volume Spike"
                    elif p1w_val > 1.0 and d_weak_conf:
                        signal = "⚠️ Distribution Trap"
                    elif p1w_val < -1.0 and d_weak_conf:
                        signal = "❌ Active Selling"
                    elif z_pct_v <= 0.25 and dv5d_v < 1.05:
                        signal = "📉 Weakening"
                    else:
                        signal = "⚖️ Neutral"

                    feat = _normalize(
                        dv_ratio=float(dv), z_pct=z_pct_v, rs_1w=None,
                        ema20_above=ema20_ab, ema_cross_bull=ema_cr,
                        vix=float(vix_raw) if vix_raw else None,
                        fii_5d_cr=fii_5d, hmm_state=hmm, pcr=float(pcr) if pcr else None,
                    )

                    conn.execute("""
                        INSERT OR IGNORE INTO sector_rotation_log (
                            trade_date, sector, signal, accum_score,
                            dv_ratio, dv_ratio_5d, z_score, z_pct, breadth,
                            price_1w, rs_1w,
                            feat_dv_n, feat_zpct_n, feat_rs_n,
                            feat_ema20, feat_ema_x, feat_vix_n,
                            feat_fii_n, feat_hmm_n, feat_pcr_n,
                            regime_label, outcome_filled
                        ) VALUES (?,?,?,?, ?,?,?,?,?, ?,?, ?,?,?, ?,?,?, ?,?,?, ?, ?)
                    """, [
                        td, sector, signal, 50.0,
                        float(dv),
                        float(dv5d) if dv5d is not None and not pd.isna(dv5d) else None,
                        float(z) if not pd.isna(z) else None, z_pct_v,
                        float(breadth) if breadth is not None and not pd.isna(breadth) else None,
                        p1w, rs1w,
                        feat["feat_dv_n"], feat["feat_zpct_n"], feat["feat_rs_n"],
                        feat["feat_ema20"], feat["feat_ema_x"], feat["feat_vix_n"],
                        feat["feat_fii_n"], feat["feat_hmm_n"], feat["feat_pcr_n"],
                        r_label, False,
                    ])
                    rows_written += 1
        except Exception as exc:
            if verbose:
                print(f"[SectorMemory] Write failed for {td}: {exc}")

    if verbose:
        print(f"[SectorMemory] Recorded {rows_written} rows across {len(all_dates)} dates")

    # ── Step 6: Fill forward outcomes for old records ─────────────────────────
    outcomes_filled = 0
    if fill_outcomes:
        if verbose:
            print("[SectorMemory] Filling forward outcomes…")
        outcomes_filled = fill_forward_outcomes(end_date, min_turnover_lacs)
        if verbose:
            print(f"[SectorMemory] Filled outcomes for {outcomes_filled} records")

    return {
        "dates_processed": len(all_dates),
        "rows_recorded":   rows_written,
        "outcomes_filled": outcomes_filled,
    }
