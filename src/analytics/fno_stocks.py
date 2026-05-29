"""
FNO Stock Signals — OI-based buy/sell classification per stock, by sector/subsector.

Signal logic (standard F&O analysis):
  Long Buildup:   OI ↑ + Price ↑  → fresh longs entering  (Bullish)
  Short Buildup:  OI ↑ + Price ↓  → fresh shorts entering  (Bearish)
  Short Covering: OI ↓ + Price ↑  → shorts exiting         (Mildly Bullish)
  Long Unwinding: OI ↓ + Price ↓  → longs exiting          (Mildly Bearish)

Price direction uses near-month FUTSTK: close_price vs settle_price
(settle_price confirmed = previous day's settlement, ~equal to cash prev_close).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_fno_stock_oi_signals",
    "get_sector_oi_summary",
]

_SIGNAL_SCORE = {
    "Long Buildup":   2,
    "Short Covering": 1,
    "Neutral":        0,
    "Long Unwinding": -1,
    "Short Buildup":  -2,
}


def _as_date(d) -> date:
    return d.date() if hasattr(d, "date") else d


def _classify_signal(price_chg: float | None, oi_chg: float | None) -> str:
    if price_chg is None or oi_chg is None:
        return "Neutral"
    if oi_chg > 0 and price_chg > 0:
        return "Long Buildup"
    if oi_chg > 0 and price_chg <= 0:
        return "Short Buildup"
    if oi_chg < 0 and price_chg > 0:
        return "Short Covering"
    if oi_chg < 0 and price_chg <= 0:
        return "Long Unwinding"
    return "Neutral"


def get_fno_stock_oi_signals(
    trade_date: date,
    min_fut_oi: int = 50_000,
) -> pd.DataFrame:
    """
    OI-based buy/sell signals for all F&O stocks on trade_date.

    Returns columns:
        symbol, company_name, sector, industry,
        fut_oi, chg_in_oi, oi_chg_pct,
        close_price, settle_price, price_chg_pct,
        call_oi, put_oi, stock_pcr,
        fut_volume, call_vol, put_vol,
        oi_signal, signal_score
    """
    trade_date = _as_date(trade_date)

    df = query_dataframe("""
        WITH near_expiry AS (
            SELECT symbol, MIN(expiry_date) AS near_exp
            FROM fno_bhavcopy
            WHERE trade_date   = ?
              AND instrument   = 'FUTSTK'
              AND expiry_date >= ?
            GROUP BY symbol
        ),
        prev_date AS (
            SELECT MAX(trade_date) AS prev_dt
            FROM fno_bhavcopy
            WHERE trade_date < ?
              AND instrument  = 'FUTSTK'
        ),
        near_futures AS (
            SELECT f.symbol,
                   SUM(f.open_interest)  AS fut_oi,
                   SUM(f.contracts)      AS fut_volume,
                   SUM(f.close_price  * GREATEST(f.contracts, 1)) /
                       SUM(GREATEST(f.contracts, 1))              AS close_price,
                   SUM(f.settle_price * GREATEST(f.contracts, 1)) /
                       SUM(GREATEST(f.contracts, 1))              AS settle_price
            FROM fno_bhavcopy f
            INNER JOIN near_expiry ne
                    ON f.symbol = ne.symbol AND f.expiry_date = ne.near_exp
            WHERE f.trade_date  = ?
              AND f.instrument  = 'FUTSTK'
            GROUP BY f.symbol
        ),
        prev_futures AS (
            SELECT f.symbol,
                   SUM(f.open_interest)  AS prev_oi
            FROM fno_bhavcopy f
            INNER JOIN near_expiry ne
                    ON f.symbol = ne.symbol AND f.expiry_date = ne.near_exp
            CROSS JOIN prev_date pd
            WHERE f.trade_date  = pd.prev_dt
              AND f.instrument  = 'FUTSTK'
            GROUP BY f.symbol
        ),
        stock_options AS (
            SELECT symbol,
                   SUM(CASE WHEN option_type='CE' THEN open_interest ELSE 0 END) AS call_oi,
                   SUM(CASE WHEN option_type='PE' THEN open_interest ELSE 0 END) AS put_oi,
                   SUM(CASE WHEN option_type='CE' THEN contracts     ELSE 0 END) AS call_vol,
                   SUM(CASE WHEN option_type='PE' THEN contracts     ELSE 0 END) AS put_vol
            FROM fno_bhavcopy
            WHERE trade_date  = ?
              AND instrument  = 'OPTSTK'
            GROUP BY symbol
        )
        SELECT
            nf.symbol,
            COALESCE(sm.company_name, nf.symbol)                    AS company_name,
            COALESCE(sm.sector,   'Others')                         AS sector,
            COALESCE(sm.industry, 'Others')                         AS industry,
            nf.fut_oi,
            COALESCE(pf.prev_oi, nf.fut_oi)                        AS prev_oi,
            nf.fut_oi - COALESCE(pf.prev_oi, nf.fut_oi)            AS chg_in_oi,
            nf.fut_volume,
            nf.close_price,
            nf.settle_price,
            COALESCE(so.call_oi,  0)                                AS call_oi,
            COALESCE(so.put_oi,   0)                                AS put_oi,
            COALESCE(so.call_vol, 0)                                AS call_vol,
            COALESCE(so.put_vol,  0)                                AS put_vol
        FROM near_futures nf
        LEFT JOIN prev_futures pf ON nf.symbol = pf.symbol
        LEFT JOIN stock_options so ON nf.symbol = so.symbol
        LEFT JOIN sector_master sm ON nf.symbol = sm.symbol
        WHERE nf.fut_oi >= ?
        ORDER BY nf.fut_oi DESC
    """, [
        trade_date, trade_date,   # near_expiry
        trade_date,               # prev_date
        trade_date,               # near_futures join
        trade_date,               # stock_options
        min_fut_oi,
    ])

    if df.empty:
        return df

    # Price change from futures (settle_price = previous day settlement)
    df["price_chg_pct"] = (
        (df["close_price"] - df["settle_price"])
        / df["settle_price"].replace(0, float("nan")) * 100
    ).round(2)

    # OI change % (chg_in_oi = today_oi - prev_oi, computed in SQL from actual yesterday data)
    df["oi_chg_pct"] = (
        df["chg_in_oi"] / df["prev_oi"].replace(0, float("nan")) * 100
    ).round(2)

    # Stock-level PCR (options only)
    df["stock_pcr"] = (
        df["put_oi"] / df["call_oi"].replace(0, float("nan"))
    ).round(2)

    # OI signal using percentage change (scale-invariant across lots vs units formats)
    df["oi_signal"] = df.apply(
        lambda r: _classify_signal(r["price_chg_pct"], r["oi_chg_pct"]),
        axis=1,
    )
    df["signal_score"] = df["oi_signal"].map(_SIGNAL_SCORE).fillna(0).astype(int)

    # Price signal: today's futures directional move (always available)
    def _price_signal(chg: float | None) -> str:
        if chg is None or pd.isna(chg):
            return "Neutral"
        if chg > 1.0:
            return "Bullish"
        if chg < -1.0:
            return "Bearish"
        if chg > 0.25:
            return "Mildly Bullish"
        if chg < -0.25:
            return "Mildly Bearish"
        return "Neutral"

    df["price_signal"] = df["price_chg_pct"].apply(_price_signal)

    # PCR signal: contrarian interpretation
    def _pcr_signal(pcr) -> str:
        if pcr is None or pd.isna(pcr):
            return "—"
        if pcr > 1.3:
            return "Put Heavy"    # bearish hedging / contrarian bullish
        if pcr < 0.5:
            return "Call Heavy"   # bullish speculation / contrarian bearish
        return "Neutral"

    df["pcr_signal"] = df["stock_pcr"].apply(_pcr_signal)

    # Combined signal: price direction + PCR confirmation
    def _combined_signal(row) -> str:
        p = row["price_signal"]
        pcr = row["stock_pcr"]
        if p in ("Bullish", "Mildly Bullish") and isinstance(pcr, float) and pcr > 1.0:
            return "Long Buildup"      # price up + put heavy = real buying
        if p in ("Bearish", "Mildly Bearish") and isinstance(pcr, float) and pcr < 0.7:
            return "Short Buildup"     # price down + call heavy = real selling
        if p in ("Bullish", "Mildly Bullish"):
            return "Bullish"
        if p in ("Bearish", "Mildly Bearish"):
            return "Bearish"
        return "Neutral"

    df["combined_signal"] = df.apply(_combined_signal, axis=1)

    return df.reset_index(drop=True)


def get_sector_oi_summary(
    trade_date: date,
    min_fut_oi: int = 50_000,
) -> pd.DataFrame:
    """
    Sector-level aggregation of OI signals.

    Returns per sector:
        sector, stock_count, long_buildup, short_buildup,
        short_covering, long_unwinding, neutral,
        total_fut_oi, dominant_signal, net_score,
        bullish_pct
    """
    df = get_fno_stock_oi_signals(trade_date, min_fut_oi=min_fut_oi)
    if df.empty:
        return pd.DataFrame()

    records = []
    for sector, grp in df.groupby("sector"):
        sig_counts  = grp["combined_signal"].value_counts().to_dict()
        dominant    = grp["combined_signal"].mode().iloc[0] if not grp.empty else "Neutral"
        bullish     = sig_counts.get("Long Buildup", 0) + sig_counts.get("Bullish", 0)
        bearish     = sig_counts.get("Short Buildup", 0) + sig_counts.get("Bearish", 0)
        total       = len(grp)
        # net_score: +2 LB, +1 Bullish, 0 Neutral, -1 Bearish, -2 SB
        cscore_map  = {"Long Buildup": 2, "Bullish": 1, "Neutral": 0, "Bearish": -1, "Short Buildup": -2}
        net_score   = int(grp["combined_signal"].map(cscore_map).fillna(0).sum())
        records.append({
            "sector":          sector,
            "stock_count":     total,
            "long_buildup":    sig_counts.get("Long Buildup",  0),
            "bullish":         sig_counts.get("Bullish",       0),
            "short_buildup":   sig_counts.get("Short Buildup", 0),
            "bearish":         sig_counts.get("Bearish",       0),
            "neutral":         sig_counts.get("Neutral",       0),
            "total_fut_oi":    int(grp["fut_oi"].sum()),
            "dominant_signal": dominant,
            "net_score":       net_score,
            "bullish_pct":     round(bullish / total * 100, 1) if total else 0.0,
        })

    result = pd.DataFrame(records)
    result = result.sort_values("net_score", ascending=False).reset_index(drop=True)
    return result
