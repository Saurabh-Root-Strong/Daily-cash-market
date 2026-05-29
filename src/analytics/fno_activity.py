"""
FNO Activity Analytics — expiry structure, OI analysis, and instrument ranking.

All SQL stays in the repository; this layer builds DataFrames on top of it
and adds business logic (expiry classification, PCR, near/mid/far labelling).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "is_monthly_expiry",
    "classify_expiry_rank",
    "get_fno_dates_available",
    "get_fno_summary_stats",
    "get_expiry_calendar",
    "get_index_expiry_oi",
    "get_index_futures_rollover",
    "get_stock_oi_leaders",
    "get_index_symbols_active",
    "get_expiry_oi_history",
]

def is_monthly_expiry(d: date) -> bool:
    """
    True if d is the last occurrence of its weekday in its month.

    NSE uses different weekdays for different indices (NIFTY = Tuesday,
    BANKNIFTY = Wednesday, stocks = Thursday historically), so we check
    last-of-weekday rather than last-of-Thursday specifically.
    """
    return (d + timedelta(days=7)).month != d.month


def classify_expiry_rank(expiry_date: date, sorted_expiries: list[date]) -> str:
    """
    Return Near/Mid/Far label ranked WITHIN TYPE (monthly vs weekly separately).

    Monthly: Near Month / Mid Month / Far Month / Far+
    Weekly:  Near Week  / Mid Week  / Far Week  / Far Week

    Ranking across all expiries (old behaviour) gave wrong labels for NIFTY —
    its 4 weekly options before the June monthly claimed Near/Mid/Far Month,
    hiding the real monthly futures structure in "Far+".
    """
    is_monthly = is_monthly_expiry(expiry_date)
    same_type = [d for d in sorted_expiries if is_monthly_expiry(d) == is_monthly]
    try:
        idx = same_type.index(expiry_date)
    except ValueError:
        return "Far+" if is_monthly else "Far Week"
    if is_monthly:
        return {0: "Near Month", 1: "Mid Month", 2: "Far Month"}.get(idx, "Far+")
    else:
        return {0: "Near Week", 1: "Mid Week", 2: "Far Week"}.get(idx, "Far Week")


def get_fno_dates_available(from_date: Optional[date] = None) -> list[date]:
    """All trade_dates present in fno_bhavcopy, newest first."""
    if from_date is not None:
        df = query_dataframe(
            "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE trade_date >= ? ORDER BY trade_date DESC",
            [from_date],
        )
    else:
        df = query_dataframe(
            "SELECT DISTINCT trade_date FROM fno_bhavcopy ORDER BY trade_date DESC"
        )
    if df.empty:
        return []
    return [
        d.date() if hasattr(d, "date") else d
        for d in df["trade_date"].tolist()
    ]


def _as_date(d) -> date:
    """Coerce Timestamp or date to plain date."""
    return d.date() if hasattr(d, "date") else d


def get_fno_summary_stats(trade_date: date) -> dict:
    """KPI-level summary for the selected trade date."""
    trade_date = _as_date(trade_date)
    df = query_dataframe("""
        SELECT
            COUNT(DISTINCT symbol)                                         AS total_symbols,
            COUNT(DISTINCT expiry_date)                                    AS total_expiries,
            SUM(contracts)                                                 AS total_volume,
            SUM(value_lacs) / 100                                          AS total_value_cr,
            SUM(open_interest)                                             AS total_oi,
            SUM(CASE WHEN instrument IN ('OPTIDX','OPTSTK')
                      AND option_type = 'CE' THEN open_interest ELSE 0
                 END)                                                      AS call_oi,
            SUM(CASE WHEN instrument IN ('OPTIDX','OPTSTK')
                      AND option_type = 'PE' THEN open_interest ELSE 0
                 END)                                                      AS put_oi,
            SUM(CASE WHEN instrument IN ('FUTIDX','FUTSTK')
                     THEN open_interest ELSE 0 END)                        AS fut_oi,
            COUNT(DISTINCT CASE WHEN instrument IN ('FUTIDX','OPTIDX')
                                THEN symbol END)                           AS index_symbols,
            COUNT(DISTINCT CASE WHEN instrument IN ('FUTSTK','OPTSTK')
                                THEN symbol END)                           AS stock_symbols
        FROM fno_bhavcopy
        WHERE trade_date = ?
    """, [trade_date])

    if df.empty or df.iloc[0]["total_symbols"] is None:
        return {}

    row = df.iloc[0]
    call_oi = float(row["call_oi"] or 0)
    put_oi  = float(row["put_oi"]  or 0)
    return {
        "total_symbols":  int(row["total_symbols"]  or 0),
        "index_symbols":  int(row["index_symbols"]  or 0),
        "stock_symbols":  int(row["stock_symbols"]  or 0),
        "total_expiries": int(row["total_expiries"] or 0),
        "total_volume":   int(row["total_volume"]   or 0),
        "total_value_cr": round(float(row["total_value_cr"] or 0), 1),
        "total_oi":       int(row["total_oi"]       or 0),
        "call_oi":        int(call_oi),
        "put_oi":         int(put_oi),
        "fut_oi":         int(row["fut_oi"]         or 0),
        "overall_pcr":    round(put_oi / call_oi, 2) if call_oi > 0 else None,
    }


def get_expiry_calendar(trade_date: date) -> pd.DataFrame:
    """
    All active expiry dates for the given trade_date with OI breakdown.

    Adds: expiry_type (Weekly/Monthly), days_to_expiry, expiry_label, pcr.
    """
    trade_date = _as_date(trade_date)
    df = query_dataframe("""
        SELECT
            expiry_date,
            SUM(open_interest)                                              AS total_oi,
            SUM(contracts)                                                  AS total_volume,
            SUM(value_lacs) / 100                                           AS value_cr,
            SUM(CASE WHEN instrument IN ('OPTIDX','OPTSTK')
                      AND option_type = 'CE' THEN open_interest ELSE 0 END) AS call_oi,
            SUM(CASE WHEN instrument IN ('OPTIDX','OPTSTK')
                      AND option_type = 'PE' THEN open_interest ELSE 0 END) AS put_oi,
            SUM(CASE WHEN instrument IN ('FUTIDX','FUTSTK')
                     THEN open_interest ELSE 0 END)                         AS fut_oi,
            COUNT(DISTINCT symbol)                                          AS symbols
        FROM fno_bhavcopy
        WHERE trade_date = ?
          AND expiry_date >= ?
        GROUP BY expiry_date
        ORDER BY expiry_date
    """, [trade_date, trade_date])

    if df.empty:
        return df

    df["expiry_date"]  = pd.to_datetime(df["expiry_date"]).dt.date
    df["days_to_expiry"] = df["expiry_date"].apply(lambda d: (d - trade_date).days)
    df["is_monthly"]   = df["expiry_date"].apply(is_monthly_expiry)
    df["expiry_type"]  = df["is_monthly"].map({True: "Monthly", False: "Weekly"})
    df["expiry_label"] = df["expiry_date"].apply(lambda d: d.strftime("%d %b '%y"))
    df["pcr"] = df.apply(
        lambda r: round(r["put_oi"] / r["call_oi"], 2) if (r["call_oi"] or 0) > 0 else None,
        axis=1,
    )

    # Near/Mid/Far rank (sorted position)
    sorted_expiries = sorted(df["expiry_date"].tolist())
    df["expiry_rank"] = df["expiry_date"].apply(
        lambda d: classify_expiry_rank(d, sorted_expiries)
    )
    return df


def get_index_expiry_oi(trade_date: date, symbol: str = "NIFTY") -> pd.DataFrame:
    """
    Per-expiry OI breakdown for a single index symbol.

    Returns columns: expiry_date, expiry_label, expiry_type, days_to_expiry,
                     fut_oi, call_oi, put_oi, total_oi, volume, pcr, expiry_rank
    """
    trade_date = _as_date(trade_date)
    df = query_dataframe("""
        SELECT
            expiry_date,
            SUM(CASE WHEN instrument = 'FUTIDX' THEN open_interest ELSE 0 END) AS fut_oi,
            SUM(CASE WHEN option_type = 'CE'    THEN open_interest ELSE 0 END) AS call_oi,
            SUM(CASE WHEN option_type = 'PE'    THEN open_interest ELSE 0 END) AS put_oi,
            SUM(open_interest)                                                  AS total_oi,
            SUM(contracts)                                                      AS volume,
            SUM(value_lacs) / 100                                               AS value_cr
        FROM fno_bhavcopy
        WHERE trade_date = ?
          AND symbol     = ?
          AND expiry_date >= ?
        GROUP BY expiry_date
        ORDER BY expiry_date
    """, [trade_date, symbol, trade_date])

    if df.empty:
        return df

    df["expiry_date"]  = pd.to_datetime(df["expiry_date"]).dt.date
    df["days_to_expiry"] = df["expiry_date"].apply(lambda d: (d - trade_date).days)
    df["is_monthly"]   = df["expiry_date"].apply(is_monthly_expiry)
    df["expiry_type"]  = df["is_monthly"].map({True: "Monthly", False: "Weekly"})
    df["expiry_label"] = df["expiry_date"].apply(lambda d: d.strftime("%d %b '%y"))
    df["pcr"] = df.apply(
        lambda r: round(r["put_oi"] / r["call_oi"], 2) if (r["call_oi"] or 0) > 0 else None,
        axis=1,
    )

    sorted_expiries = sorted(df["expiry_date"].tolist())
    df["expiry_rank"] = df["expiry_date"].apply(
        lambda d: classify_expiry_rank(d, sorted_expiries)
    )
    return df


def get_index_futures_rollover(trade_date: date, symbol: str) -> pd.DataFrame:
    """
    FUTIDX settle prices and OI per monthly expiry — for cost-of-carry / rollover analysis.

    Returns only monthly expiries with open_interest > 0.
    Added columns:
      oi_pct         — this expiry's OI as % of total FUTIDX OI
      carry_pts      — settle_price minus previous expiry's settle (NaN for near month)
      carry_pct_ann  — annualised carry % vs previous expiry (NaN for near month)

    Typical India fair-value carry ≈ (repo - dividend) * T ≈ 6–7% ann → ~0.5% per month.
    Positive carry > 7%  = premium (bullish demand)
    Positive carry 3–7%  = near fair value (neutral)
    Positive carry 0–3%  = slight discount (mild bearish)
    Negative carry       = backwardation (stress / bearish)
    """
    trade_date = _as_date(trade_date)
    df = query_dataframe("""
        SELECT
            expiry_date,
            settle_price,
            open_interest,
            chg_in_oi,
            contracts
        FROM fno_bhavcopy
        WHERE trade_date   = ?
          AND symbol       = ?
          AND instrument   = 'FUTIDX'
          AND settle_price > 0
          AND open_interest > 0
        ORDER BY expiry_date
    """, [trade_date, symbol])

    if df.empty:
        return df

    df["expiry_date"]    = pd.to_datetime(df["expiry_date"]).dt.date
    df["days_to_expiry"] = df["expiry_date"].apply(lambda d: (d - trade_date).days)
    df["is_monthly"]     = df["expiry_date"].apply(is_monthly_expiry)
    df["expiry_type"]    = df["is_monthly"].map({True: "Monthly", False: "Weekly"})
    df["expiry_label"]   = df["expiry_date"].apply(lambda d: d.strftime("%d %b '%y"))

    # Keep only monthly expiries — index futures don't have weekly contracts
    df = df[df["is_monthly"]].reset_index(drop=True)
    if df.empty:
        return df

    sorted_exp = sorted(df["expiry_date"].tolist())
    df["expiry_rank"] = df["expiry_date"].apply(
        lambda d: classify_expiry_rank(d, sorted_exp)
    )

    total_oi = float(df["open_interest"].sum())
    df["oi_pct"] = (df["open_interest"] / total_oi * 100).round(1) if total_oi > 0 else 0.0

    # Cost of carry: each monthly expiry vs the previous one
    df["carry_pts"]     = float("nan")
    df["carry_pct_ann"] = float("nan")
    for i in range(1, len(df)):
        s_near        = float(df.loc[i - 1, "settle_price"])
        s_far         = float(df.loc[i,     "settle_price"])
        days_between  = (df.loc[i, "expiry_date"] - df.loc[i - 1, "expiry_date"]).days
        carry_pts     = s_far - s_near
        carry_ann     = (carry_pts / s_near) * (365.0 / max(days_between, 1)) * 100
        df.loc[i, "carry_pts"]     = round(carry_pts, 2)
        df.loc[i, "carry_pct_ann"] = round(carry_ann, 2)

    return df


def get_index_symbols_active(trade_date: date) -> list[str]:
    trade_date = _as_date(trade_date)
    """Index symbols (FUTIDX/OPTIDX) active on this trade date."""
    df = query_dataframe("""
        SELECT DISTINCT symbol
        FROM fno_bhavcopy
        WHERE trade_date = ?
          AND instrument IN ('FUTIDX', 'OPTIDX')
        ORDER BY symbol
    """, [trade_date])
    return df["symbol"].tolist() if not df.empty else []


def get_stock_oi_leaders(trade_date: date, top_n: int = 25) -> pd.DataFrame:
    trade_date = _as_date(trade_date)
    """Top N stocks by total OI with futures/call/put breakdown."""
    df = query_dataframe("""
        SELECT
            symbol,
            SUM(CASE WHEN instrument = 'FUTSTK' THEN open_interest ELSE 0 END) AS fut_oi,
            SUM(CASE WHEN instrument = 'OPTSTK' AND option_type = 'CE'
                     THEN open_interest ELSE 0 END)                             AS call_oi,
            SUM(CASE WHEN instrument = 'OPTSTK' AND option_type = 'PE'
                     THEN open_interest ELSE 0 END)                             AS put_oi,
            SUM(open_interest)                                                  AS total_oi,
            SUM(contracts)                                                      AS total_volume,
            SUM(value_lacs) / 100                                               AS value_cr
        FROM fno_bhavcopy
        WHERE trade_date = ?
          AND instrument IN ('FUTSTK', 'OPTSTK')
        GROUP BY symbol
        ORDER BY total_oi DESC
        LIMIT ?
    """, [trade_date, top_n])

    if df.empty:
        return df

    df["pcr"] = df.apply(
        lambda r: round(r["put_oi"] / r["call_oi"], 2) if (r["call_oi"] or 0) > 0 else None,
        axis=1,
    )
    return df


def get_expiry_oi_history(symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
    """
    Historical OI per expiry date for a symbol (for OI buildup chart).
    Returns: trade_date, expiry_date, expiry_label, call_oi, put_oi, fut_oi, total_oi
    All OI columns are in ₹ Crores (value_lacs/100) for scale consistency across
    old zip format (OI in lots) and new DAT format (OI in underlying units).
    """
    df = query_dataframe("""
        SELECT
            trade_date,
            expiry_date,
            SUM(CASE WHEN instrument IN ('FUTIDX','FUTSTK')
                     THEN value_lacs ELSE 0 END) / 100 AS fut_oi,
            SUM(CASE WHEN option_type = 'CE'
                     THEN value_lacs ELSE 0 END) / 100 AS call_oi,
            SUM(CASE WHEN option_type = 'PE'
                     THEN value_lacs ELSE 0 END) / 100 AS put_oi,
            SUM(value_lacs) / 100                       AS total_oi
        FROM fno_bhavcopy
        WHERE symbol     = ?
          AND trade_date >= ?
          AND trade_date <= ?
        GROUP BY trade_date, expiry_date
        ORDER BY trade_date, expiry_date
    """, [symbol, from_date, to_date])

    if not df.empty:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
        df["expiry_label"] = df["expiry_date"].apply(lambda d: d.strftime("%d %b '%y"))
    return df
