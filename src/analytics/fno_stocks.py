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
    "get_fno_positioning_by_symbol",
    "get_sector_fno_aggregate",
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
    """Near-month stock PCR, contrarian. PCR>1.3 = put-heavy (downside hedged →
    contrarian bullish); PCR<0.6 = call-heavy (complacent → contrarian bearish)."""
    if pcr is None or pd.isna(pcr):
        return "—"
    if pcr > 1.3:  return "Put Heavy"
    if pcr < 0.6:  return "Call Heavy"
    return "Neutral"


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
                   -- near-month price (one expiry, but guard against dups)
                   MAX(CASE WHEN exp_rank = 1 THEN close_price  END) AS near_close,
                   MAX(CASE WHEN exp_rank = 1 THEN settle_price END) AS near_settle
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
               a.near_close, a.near_settle,
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

    return df.drop(columns=["near_close", "near_settle", "prev_near_oi"]).reset_index(drop=True)


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
