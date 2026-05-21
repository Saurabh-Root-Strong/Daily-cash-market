"""
Institutional-grade multi-factor F&O signal system.

5 Factors (methodology inspired by leading quant firms):
  1. Price Direction  (25%) — today's futures move vs yesterday's settlement
  2. Cost of Carry    (25%) — (fut_close - spot_close) / spot_close, raw %
  3. PCR Contrarian   (20%) — put/call near-expiry OI ratio (crowd vs market)
  4. Max Pain Gravity (20%) — spot distance from max pain strike
  5. Volume Activity  (10%) — futures turnover magnitude × price direction

Signal thresholds on composite score (-2.0 to +2.0):
  STRONG BUY  ≥  1.0
  BUY         ≥  0.4
  HOLD        > -0.4
  SELL        > -1.0
  STRONG SELL ≤ -1.0
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_fno_composite_signals"]

_WEIGHTS = {
    "price_direction": 0.25,
    "cost_of_carry":   0.25,
    "pcr_contrarian":  0.20,
    "max_pain":        0.20,
    "volume_activity": 0.10,
}


def _as_date(d) -> date:
    if isinstance(d, date) and not hasattr(d, "hour"):
        return d
    if hasattr(d, "date"):
        return d.date()
    return date.fromisoformat(str(d)[:10])


def _score(series: pd.Series, thresholds: list[float], scores: list[int]) -> pd.Series:
    """
    Map a continuous series to discrete scores.
    thresholds must be in descending order; len(scores) == len(thresholds) + 1.
    """
    conditions = [series >= t for t in thresholds]
    return pd.Series(
        np.select(conditions, scores[:-1], default=scores[-1]),
        index=series.index,
        dtype=float,
    )


def _compute_max_pain(grp: pd.DataFrame) -> float:
    """
    Max Pain = strike price where total option-writer payout is minimised.

    Writer payout for test price P:
        call_payout = Σ call_OI × max(P - K_call, 0)
        put_payout  = Σ put_OI  × max(K_put  - P, 0)
    """
    calls = grp[grp["option_type"] == "CE"][["strike_price", "open_interest"]]
    puts  = grp[grp["option_type"] == "PE"][["strike_price", "open_interest"]]

    if calls.empty or puts.empty:
        return float("nan")

    candidates = np.unique(grp["strike_price"].values)
    best_strike = candidates[0]
    min_payout  = np.inf

    for p in candidates:
        call_pay = (
            ((p - calls["strike_price"]).clip(lower=0) * calls["open_interest"]).sum()
        )
        put_pay = (
            ((puts["strike_price"] - p).clip(lower=0) * puts["open_interest"]).sum()
        )
        total = call_pay + put_pay
        if total < min_payout:
            min_payout = total
            best_strike = p

    return float(best_strike)


def get_fno_composite_signals(
    trade_date: date,
    min_fut_oi: int = 50_000,
) -> pd.DataFrame:
    """
    Compute a 5-factor composite F&O signal for each stock.

    Returns columns:
        symbol, company_name, sector, industry,
        fut_oi, spot_price, fut_close, settle_price,
        price_chg_pct, coc_pct, stock_pcr, max_pain, mp_distance_pct,
        value_lacs,
        score_price, score_coc, score_pcr, score_mp, score_vol,
        composite_score, signal_label
    """
    trade_date = _as_date(trade_date)

    # ── 1. Near-month futures ─────────────────────────────────────────────
    fut_df = query_dataframe(
        """
        WITH near AS (
            SELECT MIN(expiry_date) AS exp
            FROM fno_bhavcopy
            WHERE trade_date = ? AND instrument = 'FUTSTK'
        )
        SELECT
            f.symbol,
            f.close_price   AS fut_close,
            f.settle_price,
            f.open_interest AS fut_oi,
            f.value_lacs
        FROM fno_bhavcopy f, near n
        WHERE f.trade_date = ?
          AND f.instrument = 'FUTSTK'
          AND f.expiry_date = n.exp
          AND f.open_interest >= ?
        """,
        [trade_date, trade_date, min_fut_oi],
    )

    if fut_df.empty:
        return pd.DataFrame()

    # ── 2. Near-month options (for PCR + Max Pain) ────────────────────────
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

    # ── 3. Same-day spot prices ───────────────────────────────────────────
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

    # ── 4. Sector master ──────────────────────────────────────────────────
    sec_df = query_dataframe(
        "SELECT symbol, company_name, sector, industry FROM sector_master",
        [],
    )

    # ── 5. Per-stock PCR and Max Pain ─────────────────────────────────────
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

    # ── 6. Build master frame ─────────────────────────────────────────────
    df = fut_df.merge(cash_df, on="symbol", how="inner")
    df = df.merge(sec_df,  on="symbol", how="left")

    df["stock_pcr"] = df["symbol"].map(pcr_map)
    df["max_pain"]  = df["symbol"].map(mp_map)

    df["sector"]  = df["sector"].fillna("Unknown")
    df["industry"] = df["industry"].fillna("Unknown")

    # ── 7. Derived metrics ────────────────────────────────────────────────
    sp = df["settle_price"].replace(0, np.nan)
    df["price_chg_pct"]   = (df["fut_close"] - sp) / sp * 100

    spot = df["spot_price"].replace(0, np.nan)
    df["coc_pct"]         = (df["fut_close"] - spot) / spot * 100

    mp = df["max_pain"].replace(0, np.nan)
    # positive  = spot above max pain (bearish gravity)
    # negative  = spot below max pain (bullish gravity)
    df["mp_distance_pct"] = (df["spot_price"] - mp) / mp * 100

    # ── 8. Factor scores  (-2 … +2) ───────────────────────────────────────
    # Price Direction: strong up=+2, up=+1, flat=0, down=-1, strong down=-2
    df["score_price"] = _score(
        df["price_chg_pct"].fillna(0),
        [2.0, 0.5, -0.5, -2.0],
        [2, 1, 0, -1, -2],
    )

    # Cost of Carry: premium>3%=+2, 1-3%=+1, neutral=0, discount=−1/-2
    df["score_coc"] = _score(
        df["coc_pct"].fillna(0),
        [3.0, 1.0, -0.5, -2.0],
        [2, 1, 0, -1, -2],
    )

    # PCR Contrarian: >1.5=+2 (heavy put bias → contrarian bullish)
    # fillna(1.0) = neutral when no options data
    df["score_pcr"] = _score(
        df["stock_pcr"].fillna(1.0),
        [1.5, 1.0, 0.7, 0.5],
        [2, 1, 0, -1, -2],
    )

    # Max Pain Gravity: spot below MP (negative distance) → bullish pull → +score
    # Negate distance so that "spot below MP" maps to high values (bullish)
    df["score_mp"] = _score(
        -df["mp_distance_pct"].fillna(0),
        [3.0, 1.0, -1.0, -3.0],
        [2, 1, 0, -1, -2],
    )

    # Volume Activity: above-median turnover combined with price direction
    vol_median = float(df["value_lacs"].median())
    high_vol   = df["value_lacs"] > vol_median * 1.5
    price_up   = df["price_chg_pct"].fillna(0)

    df["score_vol"] = np.where(
        high_vol & (price_up > 1.0),  2,
        np.where(high_vol & (price_up > 0),   1,
        np.where(high_vol & (price_up < -1.0), -2,
        np.where(high_vol & (price_up < 0),    -1, 0))),
    ).astype(float)

    # ── 9. Composite score ────────────────────────────────────────────────
    df["composite_score"] = (
        df["score_price"] * _WEIGHTS["price_direction"]
        + df["score_coc"]   * _WEIGHTS["cost_of_carry"]
        + df["score_pcr"]   * _WEIGHTS["pcr_contrarian"]
        + df["score_mp"]    * _WEIGHTS["max_pain"]
        + df["score_vol"]   * _WEIGHTS["volume_activity"]
    )

    # ── 10. Signal label ──────────────────────────────────────────────────
    def _label(s: float) -> str:
        if s >= 1.0:  return "STRONG BUY"
        if s >= 0.4:  return "BUY"
        if s > -0.4:  return "HOLD"
        if s > -1.0:  return "SELL"
        return "STRONG SELL"

    df["signal_label"] = df["composite_score"].apply(_label)

    # ── 11. Round for display ─────────────────────────────────────────────
    for col in ["price_chg_pct", "coc_pct", "mp_distance_pct", "composite_score"]:
        df[col] = df[col].round(2)
    df["stock_pcr"] = df["stock_pcr"].round(2)

    # ── 12. Select and sort ───────────────────────────────────────────────
    out_cols = [
        "symbol", "company_name", "sector", "industry",
        "fut_oi", "spot_price", "fut_close", "settle_price",
        "price_chg_pct", "coc_pct", "stock_pcr", "max_pain",
        "mp_distance_pct", "value_lacs",
        "score_price", "score_coc", "score_pcr", "score_mp", "score_vol",
        "composite_score", "signal_label",
    ]
    df = df[[c for c in out_cols if c in df.columns]]

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)
