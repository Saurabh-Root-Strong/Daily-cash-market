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

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_fno_stock_oi_signals",
    "get_sector_oi_summary",
    "get_fno_positioning_by_symbol",
    "get_sector_fno_aggregate",
    "get_fno_expiry_breakdown_by_symbol",
    "get_expiry_oi_trend",
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


_POST_EXPIRY_WINDOW = 3   # trading days after a monthly roll where futures OI-change is unreliable


def _trading_days_since_roll(as_of_date: date) -> int | None:
    """
    Trading days since the most recent monthly-expiry roll, or None if no roll in
    the recent window.

    A "roll" is the trade_date on which the market-wide near (minimum) FUTSTK
    expiry_date stepped UP to a new month — i.e. the old monthly contract expired.
    For ~_POST_EXPIRY_WINDOW trading days after that, stock-futures OI is still
    migrating into the new contract, so day-over-day OI change is noise, not
    conviction. Returns 0 on the roll day itself, 1 the next session, etc.
    """
    df = query_dataframe("""
        SELECT trade_date, MIN(expiry_date) AS near_exp
        FROM fno_bhavcopy
        WHERE instrument = 'FUTSTK' AND trade_date <= ?
        GROUP BY trade_date
        ORDER BY trade_date DESC
        LIMIT 12
    """, [_as_date(as_of_date)])
    if df.empty or len(df) < 2:
        return None

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["near_exp"]   = pd.to_datetime(df["near_exp"]).dt.date
    rows = df.to_dict("records")   # newest-first

    # Walk newest→older; the roll is where the older session had a SMALLER near
    # expiry than the newer one (the contract stepped up a month).
    for i in range(len(rows) - 1):
        if rows[i + 1]["near_exp"] < rows[i]["near_exp"]:
            return i   # i sessions elapsed since the roll session (rows[i] is roll day)
    return None


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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR-ROTATION F&O OVERLAY  (per-symbol positioning + sector aggregate)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Designed to overlay onto the Sector Rotation page: each F&O stock in a sector's
# drill-down gets a Futures-position and Options-position read, and each sector
# gets an aggregate "F&O edge" badge. Non-F&O stocks simply have no row here
# (the caller LEFT JOINs, so they render blank).
#
# CRITICAL: fno_bhavcopy.chg_in_oi is 0 for every row (the NSE DAT column is not
# populated). Day-over-day OI change MUST be recomputed as today_OI - prev_OI on
# the SAME (near) contract. Forgetting this silently yields "Neutral" everywhere.

_FUT_SIGNAL_SCORE = {
    "Long Buildup": 2, "Short Covering": 1, "Neutral": 0,
    "Long Unwinding": -1, "Short Buildup": -2,
}


def _fut_signal(price_chg: float | None, oi_chg_pct: float | None) -> str:
    """OI-price matrix on near-month futures (Murphy). Needs a real move on both
    axes (>0.1% price, >0.5% OI) so noise near zero stays Neutral."""
    if price_chg is None or oi_chg_pct is None or pd.isna(price_chg) or pd.isna(oi_chg_pct):
        return "Neutral"
    p_up, p_dn = price_chg > 0.1, price_chg < -0.1
    oi_up, oi_dn = oi_chg_pct > 0.5, oi_chg_pct < -0.5
    if oi_up and p_up:  return "Long Buildup"
    if oi_up and p_dn:  return "Short Buildup"
    if oi_dn and p_up:  return "Short Covering"
    if oi_dn and p_dn:  return "Long Unwinding"
    return "Neutral"


def _opt_signal(pcr: float | None) -> str:
    """Near-month stock PCR — DESCRIPTIVE only, no directional claim.

    The contrarian reading (high PCR = bullish) was tested in the Phase-3 IC
    diagnostic and came out NEGATIVE (IC ~-0.08 at 5/20d) — i.e. on this data
    high PCR went with LOWER forward returns, not a contrarian bounce. The sign
    is unstable on a small sample, so we make no directional call here and just
    report the raw positioning. Read it as context, not a buy/sell signal.
    """
    if pcr is None or pd.isna(pcr):
        return "—"
    if pcr > 1.3:  return "Put Heavy"
    if pcr < 0.6:  return "Call Heavy"
    return "Balanced"


def get_fno_positioning_by_symbol(as_of_date: date) -> pd.DataFrame:
    """
    Per-symbol F&O positioning for every F&O stock on as_of_date — the building
    block for the Sector Rotation overlay.

    One row per F&O underlying with:
        symbol,
        fut_oi (near), fut_oi_total (near+next+far), fut_oi_value_cr,
        fut_oi_chg_pct (recomputed vs prev day, near contract),
        fut_price_chg_pct, fut_signal,
        call_oi, put_oi, pcr, opt_signal,
        near_oi, next_oi, far_oi   (per-expiry futures OI, for hover)
    """
    as_of_date = _as_date(as_of_date)

    df = query_dataframe("""
        WITH expiries AS (   -- rank each symbol's futures expiries: 1=near,2=next,3=far
            SELECT symbol, expiry_date,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY expiry_date) AS exp_rank
            FROM (
                SELECT DISTINCT symbol, expiry_date
                FROM fno_bhavcopy
                WHERE trade_date = ? AND instrument = 'FUTSTK' AND expiry_date >= ?
            ) e
        ),
        prev_date AS (
            SELECT MAX(trade_date) AS prev_dt
            FROM fno_bhavcopy WHERE trade_date < ? AND instrument = 'FUTSTK'
        ),
        fut AS (   -- today's futures, joined to expiry rank
            SELECT f.symbol, x.exp_rank,
                   f.open_interest AS oi, f.value_lacs,
                   f.close_price, f.settle_price
            FROM fno_bhavcopy f
            JOIN expiries x ON f.symbol = x.symbol AND f.expiry_date = x.expiry_date
            WHERE f.trade_date = ? AND f.instrument = 'FUTSTK'
        ),
        fut_agg AS (
            SELECT symbol,
                   SUM(oi)                                          AS fut_oi_total,
                   SUM(value_lacs) / 100.0                          AS fut_oi_value_cr,
                   SUM(CASE WHEN exp_rank = 1 THEN oi ELSE 0 END)   AS near_oi,
                   SUM(CASE WHEN exp_rank = 2 THEN oi ELSE 0 END)   AS next_oi,
                   SUM(CASE WHEN exp_rank = 3 THEN oi ELSE 0 END)   AS far_oi,
                   -- near/next-month price (one expiry each, but guard against dups).
                   -- settle_price = prev-day close for futures, so (close-settle) = today's move.
                   MAX(CASE WHEN exp_rank = 1 THEN close_price  END) AS near_close,
                   MAX(CASE WHEN exp_rank = 1 THEN settle_price END) AS near_settle,
                   MAX(CASE WHEN exp_rank = 2 THEN close_price  END) AS next_close,
                   MAX(CASE WHEN exp_rank = 2 THEN settle_price END) AS next_settle
            FROM fut GROUP BY symbol
        ),
        near_contract AS (   -- the specific expiry_date that is "near" TODAY
            SELECT symbol, expiry_date FROM expiries WHERE exp_rank = 1
        ),
        prev_near AS (
            -- prev-day OI on the SAME contract (matched by expiry_date, not by
            -- rank). After a monthly expiry the "near" label jumps to a new
            -- contract, so rank-matching would compare two DIFFERENT contracts
            -- and report the rollover as a huge OI surge. Matching the expiry_date
            -- compares the June contract to the June contract. If that contract
            -- did not yet trade on the prev day (brand-new far month), prev is
            -- absent and we treat OI change as unknown, not as a spike.
            SELECT f.symbol, SUM(f.open_interest) AS prev_near_oi
            FROM fno_bhavcopy f
            JOIN near_contract nc ON f.symbol = nc.symbol AND f.expiry_date = nc.expiry_date
            CROSS JOIN prev_date pd
            WHERE f.trade_date = pd.prev_dt AND f.instrument = 'FUTSTK'
            GROUP BY f.symbol
        ),
        opt AS (   -- near-month stock options call/put OI
            SELECT o.symbol,
                   SUM(CASE WHEN o.option_type = 'CE' THEN o.open_interest ELSE 0 END) AS call_oi,
                   SUM(CASE WHEN o.option_type = 'PE' THEN o.open_interest ELSE 0 END) AS put_oi
            FROM fno_bhavcopy o
            JOIN expiries x ON o.symbol = x.symbol AND o.expiry_date = x.expiry_date
            WHERE o.trade_date = ? AND o.instrument = 'OPTSTK' AND x.exp_rank = 1
            GROUP BY o.symbol
        )
        SELECT a.symbol,
               a.near_oi AS fut_oi, a.fut_oi_total, a.fut_oi_value_cr,
               a.near_oi, a.next_oi, a.far_oi,
               a.near_close, a.near_settle, a.next_close, a.next_settle,
               p.prev_near_oi,   -- NULL when the near contract didn't trade prev day (post-expiry)
               COALESCE(o.call_oi, 0) AS call_oi,
               COALESCE(o.put_oi, 0)  AS put_oi
        FROM fut_agg a
        LEFT JOIN prev_near p ON a.symbol = p.symbol
        LEFT JOIN opt       o ON a.symbol = o.symbol
    """, [as_of_date, as_of_date, as_of_date, as_of_date, as_of_date])

    if df.empty:
        return df

    df["fut_price_chg_pct"] = (
        (df["near_close"] - df["near_settle"])
        / df["near_settle"].replace(0, float("nan")) * 100
    ).round(2)
    # Next-month price change — computed from TODAY's next-month close/settle (settle =
    # prev-day close for futures). Previously the breakdown derived this from prev-day
    # rows in supp_fut, which yielded YESTERDAY's move (corr ~0 with today, 43% sign-flips).
    df["next_price_chg_pct"] = (
        (df["next_close"] - df["next_settle"])
        / df["next_settle"].replace(0, float("nan")) * 100
    ).round(2)
    # OI change vs SAME contract prev day. NaN when prev OI is missing (the near
    # contract is brand-new, i.e. we are right after a monthly expiry) — in that
    # window OI change is not meaningful and must NOT be turned into a signal.
    df["fut_oi_chg_pct"] = (
        (df["fut_oi"] - df["prev_near_oi"])
        / df["prev_near_oi"].replace(0, float("nan")) * 100
    ).round(2)
    df["pcr"] = (df["put_oi"] / df["call_oi"].replace(0, float("nan"))).round(2)

    # Market-wide post-expiry window: for ~3 sessions after a monthly roll, every
    # stock's futures OI is migrating into the new contract, so OI-change is noise
    # for ALL of them — not just rows with a missing prev contract. In that window
    # suppress the futures-OI signal entirely (the options/PCR signal is unaffected
    # and still shown). Per-row missing prev OI is the other unreliable case.
    days_since_roll = _trading_days_since_roll(as_of_date)
    in_post_expiry  = days_since_roll is not None and days_since_roll <= _POST_EXPIRY_WINDOW
    df["oi_reliable"] = df["prev_near_oi"].notna() & (not in_post_expiry)

    def _row_fut_signal(r):
        if not r["oi_reliable"] or pd.isna(r["fut_oi_chg_pct"]):
            return "OI settling (post-expiry)" if in_post_expiry else "OI N/A"
        return _fut_signal(r["fut_price_chg_pct"], r["fut_oi_chg_pct"])

    df["fut_signal"]      = df.apply(_row_fut_signal, axis=1)
    df["opt_signal"]      = df["pcr"].apply(_opt_signal)
    df["fut_score"]       = df["fut_signal"].map(_FUT_SIGNAL_SCORE).fillna(0).astype(int)
    df["post_expiry"]     = in_post_expiry
    df["days_since_roll"] = days_since_roll if days_since_roll is not None else -1

    return df.drop(
        columns=["near_close", "near_settle", "next_close", "next_settle", "prev_near_oi"]
    ).reset_index(drop=True)


def get_sector_fno_aggregate(as_of_date: date) -> pd.DataFrame:
    """
    Roll up per-symbol F&O positioning to the sector level — the "F&O edge" badge.

    One row per sector (only sectors containing F&O stocks) with:
        sector, fno_stock_count,
        n_long_buildup, n_short_buildup, n_short_covering, n_long_unwinding,
        fut_oi_value_cr, sector_pcr,
        fno_net_score (sum of per-stock fut_score),
        fno_bias  (LONG BUILDUP / SHORT BUILDUP / MIXED ... — dominant read)
    """
    pos = get_fno_positioning_by_symbol(as_of_date)
    if pos.empty:
        return pd.DataFrame()

    # Attach sector via the analytics-layer aggregator's master (keeps SQL in data layer)
    sec = query_dataframe(
        "SELECT symbol, sector FROM sector_master WHERE sector IS NOT NULL", []
    )
    pos = pos.merge(sec, on="symbol", how="left")
    pos = pos[~pos["sector"].isin(["ETF", "Others"]) & pos["sector"].notna()]
    if pos.empty:
        return pd.DataFrame()

    in_post_expiry = bool(pos["post_expiry"].iloc[0]) if "post_expiry" in pos.columns else False

    records = []
    for sector, g in pos.groupby("sector"):
        n = len(g)
        sc = g["fut_signal"].value_counts().to_dict()
        call = float(g["call_oi"].sum())
        put = float(g["put_oi"].sum())
        net = int(g["fut_score"].sum())
        avg = net / n if n else 0.0
        sector_pcr = round(put / call, 2) if call > 0 else None

        if in_post_expiry:
            # Futures OI unreliable this window — derive bias from sector PCR
            # (contrarian): high PCR = downside hedged = bullish lean, low = bearish.
            if sector_pcr is None:
                bias = "⚪ OI settling (post-expiry)"
            elif sector_pcr >= 1.1:
                bias = "🟡 Bullish (PCR, fut OI settling)"
            elif sector_pcr <= 0.6:
                bias = "🟠 Bearish (PCR, fut OI settling)"
            else:
                bias = "⚪ Neutral (fut OI settling)"
        else:
            bias = ("🟢 Long Buildup" if avg >= 0.75 else
                    "🔴 Short Buildup" if avg <= -0.75 else
                    "🟡 Mild Long" if avg >= 0.25 else
                    "🟠 Mild Short" if avg <= -0.25 else
                    "⚪ Mixed / Neutral")
        records.append({
            "sector":            sector,
            "fno_stock_count":   n,
            "n_long_buildup":    sc.get("Long Buildup", 0),
            "n_short_buildup":   sc.get("Short Buildup", 0),
            "n_short_covering":  sc.get("Short Covering", 0),
            "n_long_unwinding":  sc.get("Long Unwinding", 0),
            "fut_oi_value_cr":   round(float(g["fut_oi_value_cr"].sum()), 1),
            "sector_pcr":        sector_pcr,
            "fno_net_score":     net,
            "fno_avg_score":     round(avg, 2),
            "fno_bias":          bias,
            "post_expiry":       in_post_expiry,
        })

    sort_col = "sector_pcr" if in_post_expiry else "fno_avg_score"
    return pd.DataFrame(records).sort_values(
        sort_col, ascending=False, na_position="last"
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-EXPIRY F&O BREAKDOWN  (near / next / far for each stock)
#
# Architecture: built on get_fno_positioning_by_symbol() which is proven to work
# on both local and Streamlit Cloud. Two small supplementary queries add the
# extra data (next/far prev OI + options per expiry + premium change).
# ═══════════════════════════════════════════════════════════════════════════════
#
# Futures signal: OI-price matrix (same as Murphy's standard framework).
#
# Options signal: OI-PREMIUM matrix — the missing link PCR alone cannot provide.
#
#   PCR (put/call ratio) is a CONTRARIAN indicator that tells you the ratio of
#   existing put to call OI. The problem: a LOW PCR means heavy call OI, but
#   does not tell you if that OI was built by BUYERS (bullish) or WRITERS (bearish).
#
#   The OI-Premium matrix disambiguates:
#   ┌─────────────────────────────────────────────────────────────┐
#   │  OI Change │ Premium Change │ Interpretation                │
#   │────────────┼───────────────┼───────────────────────────────│
#   │  OI ↑      │ Premium ↑     │ BUYING  — demand drives both  │
#   │  OI ↑      │ Premium ↓     │ WRITING — supply: writers sell │
#   │  OI ↓      │ Premium ↑     │ SHORT COVERING — writers exit  │
#   │  OI ↓      │ Premium ↓     │ LONG EXITING — buyers exit     │
#   └─────────────────────────────────────────────────────────────┘
#
#   Premium change uses (close_price - settle_price) from today's bhavcopy row,
#   which IS yesterday's settlement — so no separate prev-day options query needed.
#   OI change still needs prev-day OI (opt_prev CTE).
#
#   Combined call+put signal:
#     Call Buying  + Put Writing  → 🔥 Strong Bull (net long)
#     Call Writing + Put Buying   → ❄️ Strong Bear (net short)
#     Call Buying  + Put Buying   → ⚡ Straddle (volatility bet)
#     Call Writing + Put Writing  → 📊 Range Play (low vol / theta)


def _compact_fut_label(signal: str, oi_chg_pct: float | None) -> str:
    """Single-cell text for a futures expiry: emoji + signal abbrev + OI change %."""
    if signal in ("OI N/A", "OI settling (post-expiry)"):
        return "⟳ rolling"
    icons = {
        "Long Buildup":   "🟢 LB",
        "Short Buildup":  "🔴 SB",
        "Short Covering": "🔵 SC",
        "Long Unwinding": "🟠 LU",
        "Neutral":        "⚪",
    }
    base = icons.get(signal, "⚪")
    if oi_chg_pct is not None and not pd.isna(oi_chg_pct):
        return f"{base} {oi_chg_pct:+.0f}%"
    return base


def _compact_pcr_label(pcr: float | None) -> str:
    """Single-cell text for options PCR: value + directional label."""
    if pcr is None or pd.isna(pcr):
        return "—"
    if pcr > 1.3:
        return f"{pcr:.1f} Put↑"
    if pcr < 0.6:
        return f"{pcr:.1f} Call↑"
    return f"{pcr:.1f} Bal"


def _opt_oi_prem_signal(
    oi_chg_pct: float | None,
    prem_chg: float | None,
    opt_type: str,   # "CE" or "PE"
) -> str:
    """
    OI-premium matrix for a single option type (calls or puts).
    Premium change = volume-weighted (close - settle) across all strikes at this expiry.
    settle_price in NSE bhavcopy = previous-day settlement → no extra query needed.
    """
    if any(x is None or (isinstance(x, float) and pd.isna(x)) for x in [oi_chg_pct, prem_chg]):
        return "—"
    t     = "C" if opt_type == "CE" else "P"
    oi_up = oi_chg_pct  >  0.5
    oi_dn = oi_chg_pct  < -0.5
    pr_up = prem_chg    >  0
    if oi_up and pr_up:   return f"{t}.Buying"    # demand → OI↑, premium↑
    if oi_up and not pr_up: return f"{t}.Writing"  # supply → OI↑, premium↓
    if oi_dn and pr_up:   return f"{t}.SC"         # short covering → OI↓, premium↑
    if oi_dn and not pr_up: return f"{t}.LE"       # long exiting → OI↓, premium↓
    return f"{t}.Neutral"


# Option-footprint sentiment (price-supportive = bullish). Puts INVERT vs calls:
# writing or closing puts is bullish; buying or uncovering puts is bearish.
#   Calls: Buying ↑bull · Writing ↓bear · SC (writers buy back) ↑bull · LE (buyers exit) ↓bear
#   Puts:  Writing ↑bull · Buying ↓bear · LE (buyers exit) ↑bull · SC (writers buy back) ↓bear
_OPT_BULLISH = {"C.Buying", "C.SC", "P.Writing", "P.LE"}
_OPT_BEARISH = {"C.Writing", "C.LE", "P.Buying", "P.SC"}


def _opt_sentiment(sig: str) -> int:
    """+1 bullish / −1 bearish / 0 neutral for a single call- or put-leg signal."""
    if sig in _OPT_BULLISH:
        return 1
    if sig in _OPT_BEARISH:
        return -1
    return 0


def _combined_opt_label(call_sig: str, put_sig: str, pcr: float | None) -> str:
    """
    Combine call + put OI-premium signals into a single actionable read.
    PCR is appended as supporting context (not the decision driver).

    Bullishness is scored via _opt_sentiment so put legs are read correctly
    (put writing/closing = bullish; put buying/uncovering = bearish) — naive
    substring matching ("Buying"/"SC") is call-side only and mis-colours puts.
    """
    if call_sig == "—" and put_sig == "—":
        return "—"

    pcr_s = f" PCR:{pcr:.1f}" if pcr is not None and not pd.isna(pcr) else ""

    # Volatility structure — same ACTION on both legs, direction-neutral. Check first.
    if call_sig == "C.Buying" and put_sig == "P.Buying":
        return f"⚡ Vol Bet C+P.Buy{pcr_s}"      # both bought → long volatility
    if call_sig == "C.Writing" and put_sig == "P.Writing":
        return f"📊 Range C+P.Wrt{pcr_s}"        # both written → short volatility / range

    cb, pb = _opt_sentiment(call_sig), _opt_sentiment(put_sig)
    # Net directional bias: both legs agree.
    if cb > 0 and pb > 0:
        return f"🔥 Bull C.Buy+P.Wrt{pcr_s}"
    if cb < 0 and pb < 0:
        return f"❄️ Bear C.Wrt+P.Buy{pcr_s}"

    # Single-sided / mixed → lead with the informative leg, colour by TRUE sentiment.
    dominant = call_sig if call_sig not in ("—", "C.Neutral") else put_sig
    sent = _opt_sentiment(dominant)
    dot = "🟢" if sent > 0 else "🔴" if sent < 0 else "⚪"
    return f"{dot} {dominant}{pcr_s}"


def get_fno_expiry_breakdown_by_symbol(as_of_date: date) -> pd.DataFrame:
    """
    Per-symbol futures OI signal + options OI-premium matrix for near/next/far expiries.

    Built on get_fno_positioning_by_symbol() (proven on local + Streamlit Cloud) +
    two small focused queries for next/far data and options per expiry.
    Avoids one large complex CTE that caused Cloud-specific DuckDB failures.

    Returns one row per F&O underlying:
        symbol, post_expiry,
        near/next/far_fut_label  — compact futures signal string
        near/next_opt_label      — call+put OI-premium combined signal
        far_opt_label            — far-month PCR only
    """
    as_of_date = _as_date(as_of_date)

    # ── Step 1: Base — proven working function with near-month data ───────────
    base = get_fno_positioning_by_symbol(as_of_date)
    if base.empty:
        return pd.DataFrame()

    days_since_roll = _trading_days_since_roll(as_of_date)
    in_post_expiry  = days_since_roll is not None and days_since_roll <= _POST_EXPIRY_WINDOW

    # ── Step 2: Supplementary — next/far prev OI for OI change % ─────────────
    supp_fut = query_dataframe("""
        WITH expiries AS (
            SELECT symbol, expiry_date,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY expiry_date) AS exp_rank
            FROM (
                SELECT DISTINCT symbol, expiry_date
                FROM fno_bhavcopy
                WHERE trade_date = ? AND instrument = 'FUTSTK' AND expiry_date >= ?
            ) e
        ),
        prev_date AS (
            SELECT MAX(trade_date) AS prev_dt
            FROM fno_bhavcopy WHERE trade_date < ? AND instrument = 'FUTSTK'
        )
        SELECT f.symbol,
               SUM(CASE WHEN x.exp_rank = 2 THEN f.open_interest END) AS next_prev_oi,
               SUM(CASE WHEN x.exp_rank = 3 THEN f.open_interest END) AS far_prev_oi
        FROM fno_bhavcopy f
        JOIN expiries x ON f.symbol = x.symbol AND f.expiry_date = x.expiry_date
        CROSS JOIN prev_date pd
        WHERE f.trade_date = pd.prev_dt AND f.instrument = 'FUTSTK'
        GROUP BY f.symbol
    """, [as_of_date, as_of_date, as_of_date])

    if not supp_fut.empty:
        base = base.merge(supp_fut, on="symbol", how="left")
    else:
        for c in ["next_prev_oi", "far_prev_oi"]:
            base[c] = float("nan")

    # ── Step 3: Supplementary — options OI + premium change per expiry ────────
    supp_opt = query_dataframe("""
        WITH expiries AS (
            SELECT symbol, expiry_date,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY expiry_date) AS exp_rank
            FROM (
                SELECT DISTINCT symbol, expiry_date
                FROM fno_bhavcopy
                WHERE trade_date = ? AND instrument = 'FUTSTK' AND expiry_date >= ?
            ) e
        ),
        prev_date AS (
            SELECT MAX(trade_date) AS prev_dt
            FROM fno_bhavcopy WHERE trade_date < ? AND instrument = 'FUTSTK'
        ),
        opt_prev_strike AS (   -- prev-day option close per strike (true premium-change base)
            SELECT o.symbol, o.expiry_date, o.option_type, o.strike_price,
                   o.close_price AS prev_close
            FROM fno_bhavcopy o
            CROSS JOIN prev_date pd
            WHERE o.trade_date = pd.prev_dt AND o.instrument = 'OPTSTK'
        ),
        opt_today AS (
            -- Premium change = TODAY's close − the SAME strike's PREVIOUS-day close,
            -- contracts-weighted per leg. settle_price is unreliable for options
            -- (NSE settles many strikes theoretically), so we join per strike instead.
            -- LEFT JOIN on the 4-key (symbol, expiry, type, strike) is 1:1 — no fan-out.
            -- Only strikes that traded both days (prev_close > 0) contribute; brand-new
            -- strikes have no measurable day-over-day change and are excluded.
            SELECT o.symbol, x.exp_rank,
                   SUM(CASE WHEN o.option_type='CE' THEN o.open_interest ELSE 0 END) AS call_oi,
                   SUM(CASE WHEN o.option_type='PE' THEN o.open_interest ELSE 0 END) AS put_oi,
                   SUM(CASE WHEN o.option_type='CE' AND pp.prev_close > 0
                            THEN (o.close_price - pp.prev_close) * o.contracts ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN o.option_type='CE' AND pp.prev_close > 0
                            THEN o.contracts ELSE 0 END), 0) AS call_prem_chg,
                   SUM(CASE WHEN o.option_type='PE' AND pp.prev_close > 0
                            THEN (o.close_price - pp.prev_close) * o.contracts ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN o.option_type='PE' AND pp.prev_close > 0
                            THEN o.contracts ELSE 0 END), 0) AS put_prem_chg
            FROM fno_bhavcopy o
            JOIN expiries x ON o.symbol = x.symbol AND o.expiry_date = x.expiry_date
            LEFT JOIN opt_prev_strike pp
                   ON o.symbol = pp.symbol AND o.expiry_date = pp.expiry_date
                  AND o.option_type = pp.option_type AND o.strike_price = pp.strike_price
            WHERE o.trade_date = ? AND o.instrument = 'OPTSTK'
            GROUP BY o.symbol, x.exp_rank
        ),
        opt_prev AS (
            SELECT o.symbol, x.exp_rank,
                   SUM(CASE WHEN o.option_type='CE' THEN o.open_interest ELSE 0 END) AS call_prev_oi,
                   SUM(CASE WHEN o.option_type='PE' THEN o.open_interest ELSE 0 END) AS put_prev_oi
            FROM fno_bhavcopy o
            JOIN expiries x ON o.symbol = x.symbol AND o.expiry_date = x.expiry_date
            CROSS JOIN prev_date pd
            WHERE o.trade_date = pd.prev_dt AND o.instrument = 'OPTSTK'
            GROUP BY o.symbol, x.exp_rank
        )
        SELECT t.symbol,
               MAX(CASE WHEN t.exp_rank=1 THEN t.call_oi       END) AS near_call_oi,
               MAX(CASE WHEN t.exp_rank=1 THEN t.put_oi        END) AS near_put_oi,
               MAX(CASE WHEN t.exp_rank=1 THEN t.call_prem_chg END) AS near_call_prem_chg,
               MAX(CASE WHEN t.exp_rank=1 THEN t.put_prem_chg  END) AS near_put_prem_chg,
               MAX(CASE WHEN t.exp_rank=1 THEN p.call_prev_oi  END) AS near_call_prev_oi,
               MAX(CASE WHEN t.exp_rank=1 THEN p.put_prev_oi   END) AS near_put_prev_oi,
               MAX(CASE WHEN t.exp_rank=2 THEN t.call_oi       END) AS next_call_oi,
               MAX(CASE WHEN t.exp_rank=2 THEN t.put_oi        END) AS next_put_oi,
               MAX(CASE WHEN t.exp_rank=2 THEN t.call_prem_chg END) AS next_call_prem_chg,
               MAX(CASE WHEN t.exp_rank=2 THEN t.put_prem_chg  END) AS next_put_prem_chg,
               MAX(CASE WHEN t.exp_rank=2 THEN p.call_prev_oi  END) AS next_call_prev_oi,
               MAX(CASE WHEN t.exp_rank=2 THEN p.put_prev_oi   END) AS next_put_prev_oi,
               MAX(CASE WHEN t.exp_rank=3 THEN t.call_oi       END) AS far_call_oi,
               MAX(CASE WHEN t.exp_rank=3 THEN t.put_oi        END) AS far_put_oi
        FROM opt_today t
        LEFT JOIN opt_prev p ON t.symbol = p.symbol AND t.exp_rank = p.exp_rank
        GROUP BY t.symbol
    """, [as_of_date, as_of_date, as_of_date, as_of_date])

    if not supp_opt.empty:
        base = base.merge(supp_opt, on="symbol", how="left")
    else:
        for c in ["near_call_oi", "near_put_oi", "near_call_prem_chg", "near_put_prem_chg",
                  "near_call_prev_oi", "near_put_prev_oi",
                  "next_call_oi", "next_put_oi", "next_call_prem_chg", "next_put_prem_chg",
                  "next_call_prev_oi", "next_put_prev_oi", "far_call_oi", "far_put_oi"]:
            base[c] = float("nan")

    df = base.copy()
    df["post_expiry"] = in_post_expiry
    # Day 0 of a new cycle: the near contract changed identity since the previous
    # session. Used by the options block below — see the note there.
    on_roll_day = days_since_roll == 0

    # ── Futures labels (near uses proven base data; next/far computed here) ───
    # Near: base already has fut_signal (near) and fut_oi_chg_pct
    df["near_fut_label"] = df.apply(
        lambda r: _compact_fut_label(
            str(r.get("fut_signal", "Neutral")), r.get("fut_oi_chg_pct")
        ), axis=1,
    )
    # Next: OI change vs prev-day same contract
    df["next_oi_chg_pct"] = (
        (df["next_oi"] - df["next_prev_oi"])
        / df["next_prev_oi"].replace(0, float("nan")) * 100
    ).round(1)
    # next_price_chg_pct now comes from the base query (today's next-month close/settle),
    # not from supp_fut's prev-day rows — see get_fno_positioning_by_symbol.
    df["next_fut_label"] = df.apply(
        lambda r: _compact_fut_label(
            _fut_signal(r.get("next_price_chg_pct"), r.get("next_oi_chg_pct")),
            r.get("next_oi_chg_pct"),
        ), axis=1,
    )
    # Far: OI change only (no price — far month price less meaningful)
    df["far_oi_chg_pct"] = (
        (df["far_oi"] - df["far_prev_oi"])
        / df["far_prev_oi"].replace(0, float("nan")) * 100
    ).round(1)
    df["far_fut_label"] = df["far_oi_chg_pct"].apply(
        lambda v: (f"⚪ {v:+.0f}%" if abs(v) < 0.5 else
                   f"🟢 +{v:.0f}%" if v > 0 else f"🔴 {v:.0f}%")
        if pd.notna(v) else "—"
    )

    # ── Options OI-premium signals (near / next; far = PCR only) ─────────────
    for pfx in ("near", "next"):
        for opt in ("call", "put"):
            df[f"{pfx}_{opt}_oi_chg_pct"] = (
                (df[f"{pfx}_{opt}_oi"] - df[f"{pfx}_{opt}_prev_oi"])
                / df[f"{pfx}_{opt}_prev_oi"].replace(0, float("nan")) * 100
            ).round(1)
        df[f"{pfx}_call_sig"] = df.apply(
            lambda r, p=pfx: _opt_oi_prem_signal(
                r.get(f"{p}_call_oi_chg_pct"), r.get(f"{p}_call_prem_chg"), "CE"
            ), axis=1,
        )
        df[f"{pfx}_put_sig"] = df.apply(
            lambda r, p=pfx: _opt_oi_prem_signal(
                r.get(f"{p}_put_oi_chg_pct"), r.get(f"{p}_put_prem_chg"), "PE"
            ), axis=1,
        )
        # ROLL DAY: today's near month was the BACK month yesterday, so this
        # day-over-day comparison measures a contract against itself while it was
        # still filling up. OI mechanically jumps and the matrix reads it as
        # demand. Measured across the whole archive, the median rank-1 CE OI
        # change is +1.0% on 492 ordinary sessions and +25.8% on the 26 roll
        # days -- a 26x inflation confined to that one session (day+1 and day+2
        # sit inside the ordinary population, so the guard is deliberately one
        # day wide, not _POST_EXPIRY_WINDOW wide).
        #
        # The futures cycle read already refuses this comparison via
        # _TREND_MIN_BASE; the options path had no equivalent and printed a
        # confident bull label on exactly the days the futures columns blanked.
        # Both legs to "—" so _combined_opt_label resolves to "—": we do not
        # know, which is not the same as no positioning.
        if on_roll_day:
            df[f"{pfx}_call_sig"] = "—"
            df[f"{pfx}_put_sig"]  = "—"
        call = df.get(f"{pfx}_call_oi", pd.Series(0, index=df.index)).fillna(0)
        put  = df.get(f"{pfx}_put_oi",  pd.Series(0, index=df.index)).fillna(0)
        df[f"{pfx}_pcr"] = (put / call.replace(0, float("nan"))).round(2)
        df[f"{pfx}_opt_label"] = df.apply(
            lambda r, p=pfx: _combined_opt_label(
                r[f"{p}_call_sig"], r[f"{p}_put_sig"], r.get(f"{p}_pcr")
            ), axis=1,
        )

    far_call = df.get("far_call_oi", pd.Series(0, index=df.index)).fillna(0)
    far_put  = df.get("far_put_oi",  pd.Series(0, index=df.index)).fillna(0)
    df["far_pcr"]       = (far_put / far_call.replace(0, float("nan"))).round(2)
    df["far_opt_label"] = df["far_pcr"].apply(_compact_pcr_label)

    # ── Expiry-cycle trend per expiry: futures, then options ─────────────────
    # Label columns must always be strings. A symbol present here but absent from a
    # trend frame would otherwise render as NaN in the table instead of "—".
    _LABEL_COLS = ["near_trend_label", "next_trend_label", "far_trend_label",
                   "near_opt_trend_label", "next_opt_trend_label", "far_opt_trend_label"]
    for extra in (get_expiry_oi_trend(as_of_date), get_expiry_opt_trend(as_of_date)):
        if not extra.empty:
            df = df.merge(extra, on="symbol", how="left")
    for c in _LABEL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("—").astype(str)

    keep = [
        "symbol", "post_expiry",
        "near_fut_label", "next_fut_label", "far_fut_label",
        "near_oi_chg_pct", "next_oi_chg_pct", "far_oi_chg_pct",
        "near_opt_label", "next_opt_label", "far_opt_label",
        "near_call_sig", "near_put_sig", "next_call_sig", "next_put_sig",
        "near_trend_label", "next_trend_label", "far_trend_label",
        "near_oi_trend_pct", "next_oi_trend_pct", "far_oi_trend_pct",
        "near_opt_trend_label", "next_opt_trend_label", "far_opt_trend_label",
        "near_opt_bull_z", "next_opt_bull_z",
        "trend_window",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-DAY OI TREND PER EXPIRY  (near / next / far)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY this is not simply "the 1-day signal with a bigger number".
#
# Measured over all 503 F&O sessions (310,099 symbol-expiry-days, 2026-08-05):
#
# 1. The 1-day label is not a trend. It changes on 70.9% of sessions and outright
#    REVERSES (LB<->SB / SC<->LU) on 22.1%. A 5-session version of the identical
#    matrix flips 41.2% / reverses 7.4%; normalised (below) 36.6% / 3.4%.
#
# 2. A raw OI % change is only meaningful for the NEAR month. Median |OI 5d| is
#    10.4% near, but 97.3% next and 81.8% far — because next/far contracts are in
#    their FILL-UP phase, so a raw % measures the contract's age, not conviction.
#    Fix: subtract that day's cross-sectional median for the SAME expiry rank, and
#    scale by its MAD. Every contract at the same rank shares the same lifecycle
#    stage, so the common drift cancels and what is left is this stock's build
#    RELATIVE to its peers. One |z| deadband then works for all three buckets
#    (|z|<=0.5 -> ~26% Neutral in each).
#
# 3. Monthly rollover corrupts a fixed 5-session window on the NEAR contract:
#    median |OI 5d| is 7.63x higher when the window spans a roll (everyone rolls
#    at once). Normalisation alone only cuts that to 3.57x — not enough. Fix: when
#    the roll is k<W sessions back, ANCHOR the window to the roll and trend over k
#    sessions instead, reporting the shorter window. NEXT is unaffected (0.83x) and
#    a brand-new FAR contract simply has too little history and is blanked.
#
# 4. FAR is barely traded: median 33 contracts/day vs 4,192 near, and only 24.3%
#    of far rows clear 100 contracts. It gets an OI-only read behind a volume gate,
#    never a directional colour (its price is stale).
#
# NOTE ON EXPECTATIONS: this makes the panel stable and honest, NOT predictive.
# The same matrix at 1 / 5 / 10 days separates 5-day forward returns by under
# 0.08pp — there is no edge here to harvest, only a description to get right.

_TREND_MAX_LOOKBACK = 26  # sessions to pull; the longest observed cycle is 25
_SEQ_N          = 4      # daily steps shown in the comma-separated sequence
_TREND_MIN_DAYS = 1      # cycle days needed before a cumulative read means anything
_TREND_Z        = 0.5    # |z| deadband on peer-relative OI change
_TREND_PX_PCT   = 0.5    # % price deadband over the cycle
_FAR_MIN_VOL    = 100    # median contracts/day required to label the far month
_VOL_WINDOW     = 5      # sessions the liquidity median is taken over (fixed, see below)
_SEQ_CLIP       = 999    # daily % steps are clipped for display (OI can go 0 -> inf)
# Sessions before expiry where the near-month cumulative stops meaning anything.
# Measured across all 25 cycles, median near-month cumulative OI by sessions left:
#   10 left +1.1% | 7 +0.1% | 5 -2.3% | 4 -6.5% | 3 -29.2% | 2 -55.4% | 1 -78.1% | 0 -91.6%
# and the share of stocks below -50% goes 0% -> 4% -> 65% -> 89% -> 94%. That is
# everyone leaving the expiring contract at once, not conviction. The peer-relative
# z is immune (the grey share holds at ~26% throughout, because the collapse is
# common to every stock) so the LABEL never lied — but the number on screen would
# read -80% on nearly every row, so the cumulative is withheld in this window.
_PRE_EXPIRY_WINDOW = 3
# The peer-relative z is only as good as the peer set it is ranked against. The
# fill-up guard masks most next-month names late in a cycle, which on ~15% of
# sessions left 1-19 valid peers — a median/MAD from five names is noise, not a
# ranking. Below this count the label is withheld rather than guessed.
_MIN_PEERS = 20
# A contract that held under this fraction of its current OI at the start of the
# window was effectively EMPTY then — it is filling up, not trending. Right after a
# roll the next/far months routinely print +6,000% or +20,000% because the base was
# a few hundred lots. Those are lifecycle ramps and get blanked, not labelled.
_TREND_MIN_BASE = 0.20

_TREND_CODE = {
    "Long Buildup":   ("🟢", "LB"),
    "Short Buildup":  ("🔴", "SB"),
    "Short Covering": ("🔵", "SC"),
    "Long Unwinding": ("🟠", "LU"),
    "Neutral":        ("⚪", ""),
}


@lru_cache(maxsize=1)
def _nse_holidays() -> frozenset:
    """Weekday trading holidays from config/nse_holidays.yaml (empty set if absent)."""
    try:
        import yaml
        from src.core.config import PROJECT_ROOT
        path = Path(PROJECT_ROOT) / "config" / "nse_holidays.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out = set()
        for _year, entries in (raw.get("holidays") or {}).items():
            for e in entries or []:
                d = e.get("date") if isinstance(e, dict) else e
                if d:
                    out.add(date.fromisoformat(str(d)))
        return frozenset(out)
    except Exception:                       # never let a config problem break the page
        return frozenset()


def _sessions_until(as_of: date, target: date) -> int:
    """
    Trading sessions strictly after `as_of` up to and including `target`.

    Counting bare weekdays gets this wrong whenever a holiday falls in the final
    stretch — across the history that happened on 7 sessions, each reporting 4
    sessions left when only 3 remained, so the pre-expiry guard fired a day late and
    showed a rollover-corrupted number for one session.

    Uses the EXCHANGE CALENDAR only (weekdays minus configured holidays), never the
    tape. An earlier version counted sessions already present in fno_bhavcopy, which is
    more accurate but reads rows dated AFTER as_of: harmless in live use, where they do
    not exist, but it made a replayed date behave differently from the live one. The
    forward trading calendar is public information at any as_of, so estimating it is
    legitimate; silently depending on how much history the database happens to hold is
    not.

    This puts weight on `config/nse_holidays.yaml`, which had a phantom holiday
    (2026-03-20, on which NSE actually traded) and a missing one (2026-06-26) the first
    time anything read it. Both fixed; keep it current or this guard drifts by a day.
    """
    if target <= as_of:
        return 0
    hol, n, d = _nse_holidays(), 0, as_of
    while d < target:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            n += 1
    return n


def _px_band(cycle_days: int) -> float:
    """
    Price deadband, widened with the length of the cycle so far.

    A fixed 0.5% band is a real gate on day 1 (27% of stocks sit inside it) but
    nearly none by day 20 (5%) — over three weeks almost everything has moved more
    than half a percent, so the price axis silently stops filtering and the label
    becomes OI-only. Price wanders as ~sqrt(time), so the band follows it.
    """
    return _TREND_PX_PCT * float(np.sqrt(max(cycle_days, 1)))


def _trend_signal(px_pct: float | None, z: float | None, px_band: float) -> str:
    """OI-price matrix over the window, on PEER-RELATIVE OI change (z) not raw %."""
    if px_pct is None or z is None or pd.isna(px_pct) or pd.isna(z):
        return "Neutral"
    p_up, p_dn = px_pct > px_band, px_pct < -px_band
    oi_up, oi_dn = z > _TREND_Z, z < -_TREND_Z
    if oi_up and p_up:  return "Long Buildup"
    if oi_up and p_dn:  return "Short Buildup"
    if oi_dn and p_up:  return "Short Covering"
    if oi_dn and p_dn:  return "Long Unwinding"
    return "Neutral"


def _seq_str(steps) -> str:
    """
    "+2, +3, +1, +0" — the day-by-day path, NEWEST FIRST.

    `steps` arrives in chronological order and is reversed here, so the leftmost
    figure is today. That puts today's step next to the label and the ⚠ fade flag,
    which is what ⚠ is about. A missing session shows as "-".
    """
    out = []
    for v in steps:
        if v is None or pd.isna(v) or np.isinf(v):
            out.append("-")
        else:
            out.append(f"{max(min(v, _SEQ_CLIP), -_SEQ_CLIP):+.0f}")
    return ", ".join(reversed(out))


def _trend_label(signal: str, cum_pct: float | None, fading: bool, seq: str) -> str:
    """
    "🟢 LB +8% | +2, -1, +3, +2"

    Left of the dot: the CUMULATIVE OI change since this expiry cycle began (the
    monthly-expiry basis) — the code is the peer-relative read, the number is raw.
    Right of the dot: the last few daily steps, so a build that is still running
    is visually distinct from one that stalled a week ago.
    "⚠" = TODAY moved against the cycle build.
    """
    if cum_pct is None or pd.isna(cum_pct):
        return f"— | {seq}" if seq else "—"
    icon, code = _TREND_CODE.get(signal, ("⚪", ""))
    body = f"{icon} {code} {cum_pct:+.0f}%".replace("  ", " ")
    if fading and signal != "Neutral":
        body += "⚠"
    return f"{body} | {seq}" if seq else body


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS OVER AN EXPIRY CYCLE
#
# The futures matrix ports to a 20-day window because a future has one price and no
# decay. An option does NOT, and porting it naively would have been wrong. Measured
# on the July 2026 cycle (300,319 option rows, 210 symbols, 20 sessions):
#
#   Median premium change from cycle start to expiry, on a FIXED strike:
#       CE ATM -95.2% (70% of strikes lost value) | CE OTM -98.4% (87%)
#       PE ATM -93.8% (69%)                       | PE OTM -98.0% (90%)
#   So "OI up + premium down" over a cycle would classify nearly the whole chain as
#   WRITING — that is theta running to expiry, not order flow. At ONE day the same
#   measurement is clean: median premium change +0.0%, only 37-38% losing value.
#   (Delta matters too: cycle premium change correlates +0.56 CE / -0.44 PE with the
#   underlying's move, R2 20-31%.)
#
# So the classification stays DAILY, where it is valid, and the CYCLE is summarised by
# COUNTING those daily verdicts. Two further corrections the data forced:
#
#   1. Expiry mechanics swamp the daily verdict near the end: with 15 sessions left the
#      mix is SB 44% / LB 25%, but at 1 session left it is LU 65% / SC 30% — 95% of the
#      chain "unwinding" because everyone closes out. Sessions inside
#      _PRE_EXPIRY_WINDOW are excluded from the count.
#   2. Writing dominates the cross-section: the modal state is SB for 125 of 210 CE and
#      114 of 210 PE. A raw label would read "short buildup" for most stocks and carry
#      no information. The directional score is therefore ranked against peers, exactly
#      as the futures cumulative is.
#
# Aggregating OI across strikes IS safe over a cycle: 89% of strikes trading at cycle
# end already existed at the start, and they hold 98% of end open interest.
# ═══════════════════════════════════════════════════════════════════════════════

_OPT_OI_DEAD   = 1.0     # |daily OI %| below this is Neutral
_OPT_PREM_DEAD = 2.0     # |daily premium %| below this is Neutral
_OPT_MIN_DAYS  = 4       # classified sessions needed before a cycle count is shown
# A day's verdict is only as good as the chain it was computed from. Measured per
# symbol-side-day since 2026-05-01: NEAR trades a median of 22 strikes / 7,454
# contracts, NEXT only 7 strikes / 117 contracts, and FAR a median of ZERO of both
# (82% of far symbol-side-days have no volume at all; on the latest session just 2 of
# 207 symbols cleared 100 far contracts). Sessions below these floors are not counted,
# which is why the far month gets no options column at all — there is no market there
# to describe, only one or two trades to over-interpret.
_OPT_MIN_STRIKES = 3     # traded strikes that must contribute to the premium change
_OPT_MIN_VOL     = 50    # contracts traded on that side that session

# What each daily state means for DIRECTION, per side. Call writing caps upside;
# put writing supports. Buying is directional the obvious way.
_OPT_BULL_SIGN = {
    ("CE", "LB"): +1,   ("CE", "SB"): -1,   ("CE", "SC"): +1,   ("CE", "LU"): -1,
    ("PE", "LB"): -1,   ("PE", "SB"): +1,   ("PE", "SC"): -1,   ("PE", "LU"): +1,
}
_OPT_CODE = {"LB": "Buy", "SB": "Wrt", "SC": "Cov", "LU": "Exit", "Neutral": "—"}
# Deterministic tie-break when two states held the same number of sessions. Fresh
# positioning outranks closing, since that is the more informative read.
_OPT_TIE_ORDER = ["SB", "LB", "SC", "LU"]


def _opt_day_state(oi_pct: float | None, prem_pct: float | None) -> str:
    """
    One session's OI x PREMIUM verdict for one side of one symbol's chain.
      OI up   + premium up   -> LB  buyers accumulating
      OI up   + premium down -> SB  writers accumulating (supply)
      OI down + premium up   -> SC  writers buying back
      OI down + premium down -> LU  buyers exiting
    """
    if oi_pct is None or prem_pct is None or pd.isna(oi_pct) or pd.isna(prem_pct):
        return "Neutral"
    if abs(oi_pct) <= _OPT_OI_DEAD or abs(prem_pct) <= _OPT_PREM_DEAD:
        return "Neutral"
    if oi_pct > 0:
        return "LB" if prem_pct > 0 else "SB"
    return "SC" if prem_pct > 0 else "LU"


def get_expiry_opt_trend(as_of_date: date) -> pd.DataFrame:
    """
    Options positioning over the current expiry cycle, per symbol, for the near and
    next monthly expiries.

    Daily OI x premium verdicts (same strike vs same strike, contracts-weighted) are
    counted across the cycle; the resulting bull/bear score is ranked against peers.

    Returns one row per symbol:
        near/next_opt_trend_label   "🔴 Bear · CE Wrt9 PE Buy6 /16d"
        near/next_opt_bull_z        peer-relative directional score
    """
    as_of_date = _as_date(as_of_date)

    # Bind the count to THIS cycle. Without this the window is just "the last N
    # sessions", which drags the previous cycle's verdicts into the tally — the
    # near contract was already trading as next month back then.
    cs = query_dataframe("""
        WITH near_by_day AS (
            SELECT trade_date, MIN(expiry_date) AS near_exp FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK' AND expiry_date >= trade_date AND trade_date <= ?
            GROUP BY trade_date
        )
        SELECT MIN(trade_date) AS cycle_start FROM near_by_day
        WHERE near_exp = (SELECT near_exp FROM near_by_day WHERE trade_date = ? LIMIT 1)
    """, [as_of_date, as_of_date])
    cycle_start = (_as_date(pd.to_datetime(cs["cycle_start"].iloc[0]))
                   if not cs.empty and pd.notna(cs["cycle_start"].iloc[0]) else as_of_date)

    daily = query_dataframe("""
        WITH expiries AS (
            SELECT symbol, expiry_date,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY expiry_date) AS exp_rank
            FROM (SELECT DISTINCT symbol, expiry_date FROM fno_bhavcopy
                  WHERE trade_date = ? AND instrument = 'FUTSTK' AND expiry_date >= ?) e
        ),
        cyc AS (   -- sessions of the CURRENT cycle only, ending at as_of
            SELECT DISTINCT trade_date FROM fno_bhavcopy
            WHERE instrument = 'OPTSTK' AND trade_date <= ? AND trade_date >= ?
        ),
        o AS (
            SELECT b.trade_date, b.symbol, x.exp_rank, b.option_type,
                   b.strike_price, b.open_interest, b.contracts, b.close_price
            FROM fno_bhavcopy b
            JOIN expiries x ON b.symbol = x.symbol AND b.expiry_date = x.expiry_date
            JOIN cyc      c ON c.trade_date = b.trade_date
            WHERE b.instrument = 'OPTSTK' AND x.exp_rank <= 3
        ),
        prev AS (   -- each session's predecessor inside the window
            SELECT trade_date,
                   LAG(trade_date) OVER (ORDER BY trade_date) AS prev_dt
            FROM cyc
        )
        SELECT a.trade_date, a.symbol, a.exp_rank, a.option_type,
               SUM(a.open_interest)                                        AS oi,
               SUM(p.open_interest)                                        AS prev_oi,
               SUM(a.contracts)                                            AS side_vol,
               COUNT(*) FILTER (WHERE p.close_price > 0 AND a.contracts > 0)
                                                                           AS n_strikes,
               SUM(CASE WHEN p.close_price > 0
                        THEN (a.close_price - p.close_price) * a.contracts END)
                 / NULLIF(SUM(CASE WHEN p.close_price > 0
                        THEN a.contracts END), 0)                          AS prem_chg,
               SUM(CASE WHEN p.close_price > 0 THEN p.close_price * a.contracts END)
                 / NULLIF(SUM(CASE WHEN p.close_price > 0
                        THEN a.contracts END), 0)                          AS prem_base
        FROM o a
        JOIN prev pr ON pr.trade_date = a.trade_date
        JOIN o    p  ON p.symbol = a.symbol AND p.exp_rank = a.exp_rank
                    AND p.option_type = a.option_type
                    AND p.strike_price = a.strike_price
                    AND p.trade_date = pr.prev_dt
        GROUP BY 1, 2, 3, 4
    """, [as_of_date, as_of_date, as_of_date, cycle_start])

    # A frame of "—" for every F&O name, so the columns never vanish from the table
    # on days when nothing can be said (the roll session, or a cycle too young).
    universe = query_dataframe("""
        SELECT DISTINCT symbol FROM fno_bhavcopy
        WHERE trade_date = ? AND instrument = 'FUTSTK' ORDER BY symbol
    """, [as_of_date])

    def _blank() -> pd.DataFrame:
        if universe.empty:
            return pd.DataFrame()
        z = universe.copy()
        for p in ("near", "next", "far"):
            z[f"{p}_opt_trend_label"] = "—"
            z[f"{p}_opt_bull_z"] = np.nan
        return z

    if daily.empty:
        return _blank()

    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    # Drop the expiry-mechanics window: near the end the daily verdict is 95%
    # "unwinding" for everyone, which is closure, not positioning.
    near_exp = query_dataframe("""
        SELECT MIN(expiry_date) AS exp FROM fno_bhavcopy
        WHERE trade_date = ? AND instrument = 'FUTSTK' AND expiry_date >= ?
    """, [as_of_date, as_of_date])
    if not near_exp.empty and pd.notna(near_exp["exp"].iloc[0]):
        exp_d = _as_date(pd.to_datetime(near_exp["exp"].iloc[0]))
        # Sessions-left for each cycle session WITHOUT a query per session: it is the
        # count from as_of, plus the sessions between that session and as_of. Calling
        # _sessions_until() in a loop cost 2 SQL round-trips per cycle day.
        cyc_sessions = sorted(daily["trade_date"].unique())
        left_at_asof = _sessions_until(as_of_date, exp_d)
        keep = [d for i, d in enumerate(cyc_sessions)
                if left_at_asof + (len(cyc_sessions) - 1 - i) > _PRE_EXPIRY_WINDOW]
        daily = daily[daily["trade_date"].isin(keep)]
    if daily.empty:
        return _blank()

    daily["oi_pct"] = ((daily["oi"] - daily["prev_oi"])
                       / daily["prev_oi"].replace(0, np.nan) * 100)
    daily["prem_pct"] = (daily["prem_chg"]
                         / daily["prem_base"].replace(0, np.nan) * 100)
    # Thin-chain gate: a verdict drawn from one or two trades is not a verdict.
    thin = ((daily["n_strikes"].fillna(0) < _OPT_MIN_STRIKES)
            | (daily["side_vol"].fillna(0) < _OPT_MIN_VOL))
    daily["state"] = [_opt_day_state(a, b)
                      for a, b in zip(daily["oi_pct"], daily["prem_pct"])]
    daily.loc[thin, "state"] = "Thin"          # excluded from the count entirely
    daily = daily[daily["state"] != "Thin"]
    if daily.empty:
        return pd.DataFrame()
    daily["bull"] = [_OPT_BULL_SIGN.get((t, s), 0)
                     for t, s in zip(daily["option_type"], daily["state"])]

    out = pd.DataFrame({"symbol": sorted(daily["symbol"].unique())})
    for rank, pfx in ((1, "near"), (2, "next"), (3, "far")):
        R = daily[daily["exp_rank"] == rank]
        if R.empty:
            out[f"{pfx}_opt_trend_label"] = "—"
            out[f"{pfx}_opt_bull_z"] = float("nan")
            continue

        n_days = R.groupby("symbol")["trade_date"].nunique()
        score = R.groupby("symbol")["bull"].sum() / n_days.replace(0, np.nan)

        # Dominant non-Neutral state per side, and how many sessions it held.
        # TIES MUST BREAK DETERMINISTICALLY: with a count of 1 several states tie and
        # value_counts() ordering is not stable, which made the same input render
        # "PE Wrt1" on one call and "PE Exit1" on the next. Sort by count, then by a
        # fixed state order.
        def _side(sym_grp, side):
            s = sym_grp[(sym_grp.option_type == side) & (sym_grp.state != "Neutral")]
            if s.empty:
                return "—", 0
            vc = s["state"].value_counts()
            top = sorted(vc.items(), key=lambda kv: (-kv[1], _OPT_TIE_ORDER.index(kv[0])))[0]
            return _OPT_CODE.get(top[0], "—"), int(top[1])

        rows = {}
        for sym, g in R.groupby("symbol"):
            ce_c, ce_n = _side(g, "CE")
            pe_c, pe_n = _side(g, "PE")
            rows[sym] = (ce_c, ce_n, pe_c, pe_n, int(n_days.get(sym, 0)))

        med = score.median()
        mad = (score - med).abs().median()
        usable = (score.notna().sum() >= _MIN_PEERS) and pd.notna(mad) and mad > 0
        z = (score - med) / mad if usable else pd.Series(np.nan, index=score.index)

        def _sides(ce_c, ce_n, pe_c, pe_n) -> str:
            """Name only the sides that actually had a verdict ('PE —0' reads as noise)."""
            parts = []
            if ce_n:
                parts.append(f"CE {ce_c}{ce_n}")
            if pe_n:
                parts.append(f"PE {pe_c}{pe_n}")
            return " ".join(parts) if parts else "quiet"

        lab = {}
        for sym, (ce_c, ce_n, pe_c, pe_n, nd) in rows.items():
            if nd < _OPT_MIN_DAYS:
                lab[sym] = "—"
                continue
            sides = _sides(ce_c, ce_n, pe_c, pe_n)
            if rank == 3:
                # Far month: counts only, no Bull/Bear tint. Peer ranking needs a real
                # cross-section and the far chain rarely has one (on a typical session
                # only a handful of the 200+ symbols trade a far option at all), so a
                # ranked verdict there would be a comparison against almost nobody.
                lab[sym] = f"⚪ {sides} /{nd}d"
                continue
            if pd.isna(z.get(sym, np.nan)):
                lab[sym] = "—"
                continue
            zz = z[sym]
            icon, word = (("🟢", "Bull") if zz > _TREND_Z else
                          ("🔴", "Bear") if zz < -_TREND_Z else ("⚪", "Bal"))
            lab[sym] = f"{icon} {word} | {sides} /{nd}d"

        out[f"{pfx}_opt_trend_label"] = out["symbol"].map(lab).fillna("—").astype(str)
        out[f"{pfx}_opt_bull_z"] = pd.to_numeric(out["symbol"].map(z), errors="coerce")

    # Symbols that trade futures but had no usable option data still need a row, or
    # the merge downstream leaves NaN where the table expects a string.
    if not universe.empty:
        out = universe.merge(out, on="symbol", how="left")
        for p in ("near", "next", "far"):
            out[f"{p}_opt_trend_label"] = out[f"{p}_opt_trend_label"].fillna("—").astype(str)
            out[f"{p}_opt_bull_z"] = pd.to_numeric(out[f"{p}_opt_bull_z"], errors="coerce")
    return out


def get_expiry_oi_trend(as_of_date: date, seq_n: int = _SEQ_N) -> pd.DataFrame:
    """
    Per-expiry OI + price trend measured on the MONTHLY EXPIRY CYCLE basis:
    cumulative from the session the current cycle began (the last monthly roll) to
    as_of_date, tracked on the SAME contract (matched by expiry_date), plus the
    last `seq_n` daily steps so the path is visible, not just the endpoint.

    Returns one row per symbol:
        symbol,
        near/next/far_oi_trend_pct   cumulative OI change % since the cycle start
        near/next/far_trend_label    "🟢 LB +8% | +2, -1, +3, +2"
        trend_window                 cycle days elapsed (the cumulative's basis)
    """
    as_of_date = _as_date(as_of_date)

    # Cycle start = the FIRST session on which today's near expiry was the market's
    # nearest expiry. Derived from the contract's own identity, not by scanning back
    # for a roll: _trading_days_since_roll() caps its lookback at 12 sessions (right
    # for its own 3-day post-expiry job) and returns None past cycle day 12, which
    # would silently anchor the cumulative in the PREVIOUS cycle for the whole second
    # half of every month. Cycles run 16-25 sessions.
    cyc = query_dataframe("""
        WITH near_by_day AS (
            SELECT trade_date, MIN(expiry_date) AS near_exp
            FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK' AND expiry_date >= trade_date AND trade_date <= ?
            GROUP BY trade_date
        )
        SELECT MIN(trade_date) AS cycle_start,
               COUNT(*) - 1     AS cycle_days
        FROM near_by_day
        WHERE near_exp = (SELECT near_exp FROM near_by_day
                          WHERE trade_date = ? LIMIT 1)
    """, [as_of_date, as_of_date])
    cycle_days = (int(cyc["cycle_days"].iloc[0])
                  if not cyc.empty and pd.notna(cyc["cycle_days"].iloc[0]) else None)
    lookback = _TREND_MAX_LOOKBACK if cycle_days is None else max(cycle_days + 1, seq_n + 1)

    raw = query_dataframe("""
        WITH expiries AS (   -- rank as of TODAY; we then follow each contract back
            SELECT symbol, expiry_date,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY expiry_date) AS exp_rank
            FROM (
                SELECT DISTINCT symbol, expiry_date
                FROM fno_bhavcopy
                WHERE trade_date = ? AND instrument = 'FUTSTK' AND expiry_date >= ?
            ) e
        ),
        win AS (             -- enough sessions to reach back to the cycle start
            SELECT trade_date FROM (
                SELECT DISTINCT trade_date FROM fno_bhavcopy
                WHERE instrument = 'FUTSTK' AND trade_date <= ?
                ORDER BY trade_date DESC LIMIT ?
            ) s
        )
        SELECT f.trade_date, f.symbol, x.exp_rank,
               f.open_interest AS oi, f.contracts AS vol, f.close_price
        FROM fno_bhavcopy f
        JOIN expiries x ON f.symbol = x.symbol AND f.expiry_date = x.expiry_date
        JOIN win      w ON w.trade_date = f.trade_date
        WHERE f.instrument = 'FUTSTK' AND x.exp_rank <= 3
    """, [as_of_date, as_of_date, as_of_date, int(min(lookback, _TREND_MAX_LOOKBACK))])

    if raw.empty:
        return pd.DataFrame()

    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    sessions = sorted(raw["trade_date"].unique())
    latest = sessions[-1]

    # Sessions left on the near contract — drives the pre-expiry guard below.
    near_exp = query_dataframe("""
        SELECT MIN(expiry_date) AS exp FROM fno_bhavcopy
        WHERE trade_date = ? AND instrument = 'FUTSTK' AND expiry_date >= ?
    """, [as_of_date, as_of_date])
    sessions_left = None
    if not near_exp.empty and pd.notna(near_exp["exp"].iloc[0]):
        sessions_left = _sessions_until(
            as_of_date, _as_date(pd.to_datetime(near_exp["exp"].iloc[0])))
    pre_expiry = sessions_left is not None and sessions_left <= _PRE_EXPIRY_WINDOW
    # Cycle start = `cycle_days` sessions back. Clamp to what we actually pulled.
    n_back = len(sessions) - 1 if cycle_days is None else min(cycle_days, len(sessions) - 1)
    base_dt = sessions[-(n_back + 1)] if n_back >= _TREND_MIN_DAYS else None

    out = pd.DataFrame({"symbol": sorted(raw["symbol"].unique())})
    out["trend_window"] = n_back

    for rank, pfx in ((1, "near"), (2, "next"), (3, "far")):
        R = raw[raw["exp_rank"] == rank]
        if R.empty:
            out[f"{pfx}_oi_trend_pct"] = float("nan")
            out[f"{pfx}_trend_label"] = "—"
            continue

        now = R[R["trade_date"] == latest].set_index("symbol")
        # Liquidity is judged over a FIXED recent window, not the cycle so far. Using
        # the whole cycle made the gate a different statistic on different days — a
        # 2-session median on cycle day 1 versus a 21-session median on day 20, which
        # is why the far month's pass rate drifted 13% -> 41% through a cycle.
        recent = sessions[-_VOL_WINDOW:]
        med_vol = R[R["trade_date"].isin(recent)].groupby("symbol")["vol"].median()
        idx = now.index
        f = pd.DataFrame(index=idx)
        f["med_vol"] = med_vol.reindex(idx)

        # ── cumulative since the cycle began, same contract ───────────────────
        if base_dt is not None:
            was = R[R["trade_date"] == base_dt].set_index("symbol")
            common = idx.intersection(was.index)
            b_oi = was["oi"].reindex(idx)
            b_px = was["close_price"].reindex(idx)
            f["oi_pct"] = (now["oi"] - b_oi) / b_oi.replace(0, float("nan")) * 100
            f["px_pct"] = (now["close_price"] / b_px.replace(0, float("nan")) - 1) * 100
            # Fill-up guard: a contract that held under _TREND_MIN_BASE of its
            # current OI at the cycle start is RAMPING, not trending. This is what
            # blanks the next month late in a cycle and the far month nearly always.
            f.loc[b_oi < _TREND_MIN_BASE * now["oi"], "oi_pct"] = float("nan")
            f.loc[b_oi.isna(), "oi_pct"] = float("nan")
            # Pre-expiry guard: in the last few sessions the near contract is being
            # closed out market-wide, so its cumulative is a rollover artefact.
            if rank == 1 and pre_expiry:
                f["oi_pct"] = float("nan")
        else:
            f["oi_pct"] = float("nan")
            f["px_pct"] = float("nan")

        # ── the day-by-day path (always valid, cycle or not) ──────────────────
        piv = R.pivot_table(index="symbol", columns="trade_date", values="oi",
                            aggfunc="last").reindex(idx)
        steps = piv.pct_change(axis=1) * 100
        tail = steps[steps.columns[-seq_n:]]
        f["seq"] = [_seq_str(r) for r in tail.to_numpy()]

        # ── peer-relative: strip the day's common lifecycle drift for this rank ─
        # Withheld when the surviving peer set is too small or degenerate (every
        # name identical => MAD 0). z stays NaN there, which renders as "—" rather
        # than "Neutral": we do not know, which is not the same as no build.
        n_peers = int(f["oi_pct"].notna().sum())
        med = f["oi_pct"].median()
        mad = (f["oi_pct"] - med).abs().median()
        usable = n_peers >= _MIN_PEERS and pd.notna(mad) and mad > 0
        f["z"] = ((f["oi_pct"] - med) / mad) if usable else float("nan")
        if not usable and rank != 3:
            f["oi_pct"] = float("nan")   # no ranking => no cumulative claim either

        # ── is TODAY pushing with the cycle build, or against it? ─────────────
        last_step = tail[tail.columns[-1]] if len(tail.columns) else pd.Series(np.nan, index=idx)
        fading = (np.sign(last_step) * np.sign(f["oi_pct"])) < 0

        if rank == 3:
            # Far month: OI only (its price is stale), and only where it trades.
            liquid = f["med_vol"] >= _FAR_MIN_VOL
            f.loc[~liquid, "oi_pct"] = float("nan")
            f["label"] = [
                ((f"{'🟢' if v > 0 else '🔴' if v < 0 else '⚪'} {v:+.0f}% | {s}")
                 if pd.notna(v) else (f"— | {s}" if ok else "—"))
                for v, s, ok in zip(f["oi_pct"], f["seq"], liquid)
            ]
        else:
            band = _px_band(n_back)
            sig = [_trend_signal(p, z, band) for p, z in zip(f["px_pct"], f["z"])]
            f["label"] = [_trend_label(sg, v, bool(fd), s)
                          for sg, v, fd, s in zip(sig, f["oi_pct"], fading, f["seq"])]

        out = out.merge(
            f[["oi_pct", "label"]].rename(
                columns={"oi_pct": f"{pfx}_oi_trend_pct", "label": f"{pfx}_trend_label"}
            ).rename_axis("symbol").reset_index(),
            on="symbol", how="left",
        )
        out[f"{pfx}_trend_label"] = out[f"{pfx}_trend_label"].fillna("—")

    return out
