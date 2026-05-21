"""
Institutional-grade multi-factor F&O signal system — v2.

Factor weights (must sum to 1.0):

  1. OI-Price Matrix    35%  — THE primary futures signal
                              Long Build / Short Build = conviction plays
                              Short Cover / Long Unwind = weaker reversals
                              Both direction + magnitude of OI% and price% matter

  2. Cost of Carry      20%  — Annualised basis vs India fair value (~7% p.a.)
                              Premium = demand / bullish; Discount = hedging / bearish

  3. PCR Contrarian     20%  — Put/Call OI ratio, contrarian interpretation
                              PCR > 1.5 = panic puts = institutional floor (bullish)
                              PCR < 0.5 = call frenzy = institutional ceiling (bearish)

  4. Rollover Signal    15%  — Near vs Next month OI dynamics
                              Rolling Fwd + price direction = continuation
                              Both building = high conviction
                              Both unwinding = exit / reduce

  5. Max Pain Gravity   10%  — Expiry-proximity weighted (matters only in final 7 days)

Signal thresholds on composite score (-2.0 to +2.0):
  STRONG BUY  ≥  1.2
  BUY         ≥  0.5
  HOLD        > -0.5
  SELL        ≤ -0.5
  STRONG SELL ≤ -1.2
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_fno_composite_signals"]

_WEIGHTS = {
    "oi_matrix":  0.35,
    "coc":        0.20,
    "pcr":        0.20,
    "rollover":   0.15,
    "max_pain":   0.10,
}

# India equity futures fair value = risk-free (repo) – dividend yield ≈ 6–7% p.a.
_FAIR_CARRY_ANN = 7.0


def _as_date(d) -> date:
    if isinstance(d, date) and not hasattr(d, "hour"):
        return d
    if hasattr(d, "date"):
        return d.date()
    return date.fromisoformat(str(d)[:10])


def _compute_max_pain(grp: pd.DataFrame) -> float:
    """Strike where total option-writer payout is minimised."""
    calls = grp[grp["option_type"] == "CE"][["strike_price", "open_interest"]]
    puts  = grp[grp["option_type"] == "PE"][["strike_price", "open_interest"]]
    if calls.empty or puts.empty:
        return float("nan")
    candidates = np.unique(grp["strike_price"].values)
    best, min_pay = candidates[0], np.inf
    for p in candidates:
        pay = (
            ((p - calls["strike_price"]).clip(lower=0) * calls["open_interest"]).sum()
            + ((puts["strike_price"] - p).clip(lower=0) * puts["open_interest"]).sum()
        )
        if pay < min_pay:
            min_pay, best = pay, p
    return float(best)


def _oi_matrix(price_chg: float, oi_chg_pct: float) -> tuple[str, float]:
    """
    The 4-cell OI-Price matrix — primary futures intelligence.

    Returns (signal_name, raw_score in -2.0…+2.0).

    Cell classification:
      Price↑ + OI↑ → Long Buildup   (+2) — fresh longs, strong conviction
      Price↓ + OI↑ → Short Buildup  (-2) — fresh shorts, strong conviction
      Price↑ + OI↓ → Short Cover    (+1) — shorts exiting, weaker signal
      Price↓ + OI↓ → Long Unwind    (-1) — longs exiting, weaker signal

    Magnitude scaling: larger price + larger OI change → score closer to ±2.0.
    Flat moves count as Neutral (dead-zone: |price|<0.25%, |OI%|<0.5%).
    """
    price_up   = price_chg > 0.25
    price_down = price_chg < -0.25
    oi_up      = oi_chg_pct > 0.5
    oi_down    = oi_chg_pct < -0.5

    # Magnitude modifier: scale by how strong each move is
    p_mag = min(abs(price_chg) / 2.5, 1.0)   # 2.5% price = max
    o_mag = min(abs(oi_chg_pct) / 6.0, 1.0)  # 6% OI chg = max
    mag   = 0.6 + 0.4 * (p_mag + o_mag) / 2  # always ≥ 0.6

    if price_up and oi_up:
        return "Long Buildup",   +2.0 * mag
    if price_down and oi_up:
        return "Short Buildup",  -2.0 * mag
    if price_up and oi_down:
        return "Short Cover",    +1.0 * mag   # weaker: not fresh conviction
    if price_down and oi_down:
        return "Long Unwind",    -1.0 * mag
    if price_up:
        return "Bullish",         +0.4
    if price_down:
        return "Bearish",         -0.4
    return "Neutral", 0.0


def _rollover_score(
    near_chg_pct: float,
    next_oi: float | None,
    next_chg: float | None,
    price_chg: float,
) -> float:
    """
    Cross-expiry rollover intelligence.

    Rolling Fwd (near↓ next↑): participation moving to next contract.
      Price direction confirms whether longs or shorts are rolling.
    Both building: high conviction — reinforces the OI matrix signal.
    Both unwinding: position exit — bearish structural signal (-1).
    """
    has_next = (
        next_oi is not None
        and not (isinstance(next_oi, float) and np.isnan(next_oi))
        and next_oi > 0
    )
    if not has_next:
        return 0.0

    next_chg = float(next_chg) if next_chg is not None else 0.0
    near_falling = near_chg_pct < -2.0
    near_rising  = near_chg_pct > 2.0
    next_rising  = next_chg > 0
    next_falling = next_chg < 0
    price_up     = price_chg > 0

    # Rolling Fwd: near↓ + next↑ = textbook rollover
    if near_falling and next_rising:
        # Follow price direction for rollover interpretation
        return +1.5 if price_up else -1.5

    # Both building across expiries: very high conviction
    if near_rising and next_rising:
        return +2.0 if price_up else -2.0

    # Both unwinding: complete position exit (bearish regardless of price)
    if near_falling and next_falling:
        return -1.5

    # Near building only (late expiry positioning)
    if near_rising:
        return +1.0 if price_up else -1.0

    return 0.0


def get_fno_composite_signals(
    trade_date: date,
    min_fut_oi: int = 50_000,
) -> pd.DataFrame:
    """
    6-factor composite F&O signal for each stock.

    Returns columns:
        symbol, company_name, sector, industry,
        fut_oi, spot_price, fut_close, settle_price,
        chg_in_oi, oi_chg_pct, next_oi, next_chg_oi,
        price_chg_pct, coc_pct, coc_ann, days_to_expiry,
        stock_pcr, max_pain, mp_distance_pct, value_lacs,
        oi_matrix_signal,
        score_oi, score_coc, score_pcr, score_roll, score_mp,
        composite_score, signal_label
    """
    trade_date = _as_date(trade_date)

    # ── 1. Near month + next month futures (single query) ─────────────────────
    fut_df = query_dataframe(
        """
        WITH per_sym_exp AS (
            SELECT symbol, expiry_date,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY expiry_date) AS rn
            FROM (
                SELECT DISTINCT symbol, expiry_date
                FROM fno_bhavcopy
                WHERE trade_date  = ?
                  AND instrument  = 'FUTSTK'
                  AND expiry_date >= ?
            )
        ),
        near_data AS (
            SELECT f.symbol,
                   f.close_price   AS fut_close,
                   f.settle_price,
                   f.open_interest AS fut_oi,
                   f.chg_in_oi,
                   f.value_lacs,
                   p.expiry_date
            FROM fno_bhavcopy f
            JOIN per_sym_exp p ON f.symbol = p.symbol
                               AND f.expiry_date = p.expiry_date
                               AND p.rn = 1
            WHERE f.trade_date = ?
              AND f.instrument = 'FUTSTK'
              AND f.open_interest >= ?
        ),
        next_data AS (
            SELECT f.symbol,
                   f.open_interest AS next_oi,
                   f.chg_in_oi     AS next_chg_oi
            FROM fno_bhavcopy f
            JOIN per_sym_exp p ON f.symbol = p.symbol
                               AND f.expiry_date = p.expiry_date
                               AND p.rn = 2
            WHERE f.trade_date = ?
              AND f.instrument = 'FUTSTK'
        )
        SELECT nd.*, nx.next_oi, nx.next_chg_oi
        FROM   near_data nd
        LEFT JOIN next_data nx ON nd.symbol = nx.symbol
        """,
        [
            trade_date, trade_date,  # per_sym_exp
            trade_date, min_fut_oi,  # near_data
            trade_date,              # next_data
        ],
    )

    if fut_df.empty:
        return pd.DataFrame()

    # ── 2. Near-month options (PCR + Max Pain) ─────────────────────────────────
    opt_df = query_dataframe(
        """
        WITH near AS (
            SELECT MIN(expiry_date) AS exp
            FROM fno_bhavcopy
            WHERE trade_date = ? AND instrument = 'OPTSTK'
        )
        SELECT f.symbol, f.strike_price, f.option_type, f.open_interest
        FROM fno_bhavcopy f, near n
        WHERE f.trade_date = ?
          AND f.instrument = 'OPTSTK'
          AND f.expiry_date = n.exp
          AND f.open_interest > 0
        """,
        [trade_date, trade_date],
    )

    # ── 3. Spot prices from cash market ────────────────────────────────────────
    cash_df = query_dataframe(
        """
        SELECT symbol, close_price AS spot_price
        FROM daily_data
        WHERE trade_date = (
            SELECT MAX(trade_date) FROM daily_data WHERE trade_date <= ?
        )
        """,
        [trade_date],
    )

    # ── 4. Sector master ───────────────────────────────────────────────────────
    sec_df = query_dataframe(
        "SELECT symbol, company_name, sector, industry FROM sector_master", [],
    )

    # ── 5. Per-stock PCR and Max Pain ──────────────────────────────────────────
    pcr_map: dict[str, float] = {}
    mp_map:  dict[str, float] = {}

    if not opt_df.empty:
        for sym, grp in opt_df.groupby("symbol"):
            call_oi = grp.loc[grp["option_type"] == "CE", "open_interest"].sum()
            put_oi  = grp.loc[grp["option_type"] == "PE", "open_interest"].sum()
            if call_oi > 0:
                pcr_map[sym] = float(put_oi) / float(call_oi)
            if len(grp) >= 4:
                mp_map[sym] = _compute_max_pain(grp)

    # ── 6. Build master frame ──────────────────────────────────────────────────
    df = fut_df.merge(cash_df, on="symbol", how="inner")
    df = df.merge(sec_df, on="symbol", how="left")

    df["stock_pcr"] = df["symbol"].map(pcr_map)
    df["max_pain"]  = df["symbol"].map(mp_map)
    df["sector"]    = df["sector"].fillna("Unknown")
    df["industry"]  = df["industry"].fillna("Unknown")

    # ── 7. Derived metrics ─────────────────────────────────────────────────────
    sp   = df["settle_price"].replace(0, np.nan)
    spot = df["spot_price"].replace(0, np.nan)

    df["price_chg_pct"] = ((df["fut_close"] - sp) / sp * 100).round(3)
    df["coc_pct"]       = ((df["fut_close"] - spot) / spot * 100).round(3)

    # OI change %: chg_in_oi / previous_OI
    prev_oi = (df["fut_oi"] - df["chg_in_oi"]).replace(0, np.nan)
    df["oi_chg_pct"] = (df["chg_in_oi"] / prev_oi * 100).round(2)

    # Days to expiry (from expiry_date column added in query)
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
        df["days_to_expiry"] = df["expiry_date"].apply(
            lambda d: (d - trade_date).days if d is not None else 30
        )
    else:
        df["days_to_expiry"] = 30

    # Annualised cost of carry: raw_coc * (365 / days_to_expiry)
    safe_days = df["days_to_expiry"].clip(lower=1)
    df["coc_ann"] = (df["coc_pct"] * (365.0 / safe_days)).round(2)

    # Max pain distance
    mp = df["max_pain"].replace(0, np.nan)
    df["mp_distance_pct"] = ((df["spot_price"] - mp) / mp * 100).round(2)

    # ── 8. Factor scores (-2 … +2) ────────────────────────────────────────────

    # --- Factor 1: OI-Price Matrix (35%) ---
    matrix_results = df.apply(
        lambda r: _oi_matrix(
            float(r["price_chg_pct"]) if pd.notna(r["price_chg_pct"]) else 0.0,
            float(r["oi_chg_pct"])    if pd.notna(r["oi_chg_pct"])    else 0.0,
        ),
        axis=1,
        result_type="expand",
    )
    df["oi_matrix_signal"] = matrix_results[0]
    df["score_oi"]         = matrix_results[1].clip(-2, 2)

    # --- Factor 2: Annualised Cost of Carry (20%) ---
    # Fair value ~7% p.a. for India equities
    # Score:  coc_ann - fair_carry → shift so "fair = neutral"
    coc_shifted = df["coc_ann"].fillna(0) - _FAIR_CARRY_ANN
    # Map: >+5% above fair=+2, +2–+5%=+1, −2–+2%=0, −5–−2%=−1, <−5%=−2
    df["score_coc"] = np.select(
        [coc_shifted >= 5.0, coc_shifted >= 2.0,
         coc_shifted > -2.0,
         coc_shifted > -5.0],
        [2, 1, 0, -1],
        default=-2,
    ).astype(float)

    # --- Factor 3: PCR Contrarian (20%) ---
    df["score_pcr"] = np.select(
        [df["stock_pcr"].fillna(1.0) >= 1.8,
         df["stock_pcr"].fillna(1.0) >= 1.2,
         df["stock_pcr"].fillna(1.0) >  0.7,
         df["stock_pcr"].fillna(1.0) >  0.5],
        [2, 1, 0, -1],
        default=-2,
    ).astype(float)

    # --- Factor 4: Rollover Signal (15%) ---
    df["score_roll"] = df.apply(
        lambda r: _rollover_score(
            float(r["oi_chg_pct"])   if pd.notna(r["oi_chg_pct"])  else 0.0,
            float(r["next_oi"])      if pd.notna(r.get("next_oi")) else None,
            float(r["next_chg_oi"])  if pd.notna(r.get("next_chg_oi")) else None,
            float(r["price_chg_pct"]) if pd.notna(r["price_chg_pct"]) else 0.0,
        ),
        axis=1,
    ).clip(-2, 2).astype(float)

    # --- Factor 5: Max Pain Gravity (10%, expiry-proximity weighted) ---
    # Negate: spot BELOW max pain → positive (magnetic pull up = bullish)
    mp_raw = np.select(
        [-df["mp_distance_pct"].fillna(0) >= 3.0,
         -df["mp_distance_pct"].fillna(0) >= 1.0,
         -df["mp_distance_pct"].fillna(0) > -1.0,
         -df["mp_distance_pct"].fillna(0) > -3.0],
        [2, 1, 0, -1],
        default=-2,
    ).astype(float)

    # Reduce max pain weight when far from expiry (only matters last 7 days)
    mp_proximity = np.select(
        [df["days_to_expiry"] <= 3,
         df["days_to_expiry"] <= 7,
         df["days_to_expiry"] <= 14],
        [1.0, 0.6, 0.3],
        default=0.15,  # very little weight when >14 days away
    )
    df["score_mp"] = (pd.Series(mp_raw) * pd.Series(mp_proximity)).clip(-2, 2)

    # ── 9. Composite score ─────────────────────────────────────────────────────
    df["composite_score"] = (
        df["score_oi"]   * _WEIGHTS["oi_matrix"]
        + df["score_coc"]  * _WEIGHTS["coc"]
        + df["score_pcr"]  * _WEIGHTS["pcr"]
        + df["score_roll"] * _WEIGHTS["rollover"]
        + df["score_mp"]   * _WEIGHTS["max_pain"]
    ).round(3)

    # ── 10. Signal label ───────────────────────────────────────────────────────
    def _label(s: float) -> str:
        if s >= 1.2:   return "STRONG BUY"
        if s >= 0.5:   return "BUY"
        if s > -0.5:   return "HOLD"
        if s > -1.2:   return "SELL"
        return "STRONG SELL"

    df["signal_label"] = df["composite_score"].apply(_label)

    # ── 11. Round display columns ──────────────────────────────────────────────
    for col in ["price_chg_pct", "coc_pct", "coc_ann", "mp_distance_pct", "composite_score"]:
        df[col] = df[col].round(2)
    df["stock_pcr"] = df["stock_pcr"].round(2)

    # ── 12. Select and sort ────────────────────────────────────────────────────
    out_cols = [
        "symbol", "company_name", "sector", "industry",
        "fut_oi", "spot_price", "fut_close", "settle_price",
        "chg_in_oi", "oi_chg_pct", "next_oi", "next_chg_oi",
        "price_chg_pct", "coc_pct", "coc_ann", "days_to_expiry",
        "stock_pcr", "max_pain", "mp_distance_pct", "value_lacs",
        "oi_matrix_signal",
        "score_oi", "score_coc", "score_pcr", "score_roll", "score_mp",
        "composite_score", "signal_label",
    ]
    df = df[[c for c in out_cols if c in df.columns]]
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
