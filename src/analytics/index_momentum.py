"""
Index momentum analytics — rolling returns, momentum rank, relative performance.

All functions read from index_data table via repository.query_dataframe.
No I/O, no Streamlit imports.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_index_snapshot",
    "get_index_history",
    "get_index_heatmap",
]

# ── Key indices shown prominently on the tracker page ────────────────────────
TRACKED_INDICES = [
    "Nifty 50",
    "Nifty Bank",
    "Nifty Private Bank",
    "Nifty PSU Bank",
    "Nifty IT",
    "Nifty Pharma",
    "Nifty Realty",
    "Nifty Auto",
    "Nifty FMCG",
    "Nifty Metal",
    "Nifty Energy",
    "Nifty Infrastructure",
    "Nifty Financial Services",
    "Nifty Oil & Gas",
    "Nifty Healthcare Index",
    "Nifty Media",
    "Nifty Midcap 50",
    "NIFTY Midcap 100",
    "Nifty Midcap 150",
    "NIFTY Smallcap 100",
    "Nifty Smallcap 250",
    "Nifty Next 50",
    "Nifty Commodities",
    "Nifty India Defence",
    "Nifty Mobility",
    "Nifty Consumer Durables",
    "Nifty Capital Markets",
    "Nifty Housing",
    "Nifty India Manufacturing",
    "India VIX",
]


def get_index_snapshot(trade_date: date) -> pd.DataFrame:
    """
    Return one row per tracked index for the given date with rolling returns.

    Columns: index_name, close_val, pct_chg (1D), ret_1w, ret_1m, ret_3m,
             vs_nifty50_1m, pe_ratio, pb_ratio, div_yield
    """
    # Pull 65 trading days of history to compute 3M return
    raw = query_dataframe(f"""
        SELECT trade_date, index_name, close_val, pct_chg, pe_ratio, pb_ratio, div_yield
        FROM index_data
        WHERE trade_date <= '{trade_date}'
          AND trade_date >= ('{trade_date}'::DATE - INTERVAL 100 DAY)
        ORDER BY index_name, trade_date
    """)

    if raw.empty:
        return pd.DataFrame()

    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date

    rows = []
    grouped = raw.groupby("index_name")

    # Get Nifty 50 series for relative performance
    nifty50 = grouped.get_group("Nifty 50") if "Nifty 50" in grouped.groups else None

    def _nifty_ret(n_days: int) -> float | None:
        if nifty50 is None or len(nifty50) < n_days:
            return None
        tail = nifty50.sort_values("trade_date").tail(n_days)
        first, last = tail["close_val"].iloc[0], tail["close_val"].iloc[-1]
        return (last - first) / first * 100 if first else None

    nifty_1m = _nifty_ret(22)
    nifty_3m = _nifty_ret(65)

    for idx_name in TRACKED_INDICES:
        if idx_name not in grouped.groups:
            continue
        grp = grouped.get_group(idx_name).sort_values("trade_date")
        latest = grp[grp["trade_date"] == trade_date]
        if latest.empty:
            # Use the most recent available row up to trade_date
            latest = grp.tail(1)

        row = latest.iloc[-1]

        def _ret(n: int) -> float | None:
            if len(grp) < 2:
                return None
            tail = grp.tail(n)
            if len(tail) < 2:
                return None
            first, last = tail["close_val"].iloc[0], tail["close_val"].iloc[-1]
            return (last - first) / first * 100 if first else None

        r1w = _ret(6)
        r1m = _ret(22)
        r3m = _ret(65)

        vs_nifty = (r1m - nifty_1m) if (r1m is not None and nifty_1m is not None) else None

        rows.append({
            "index_name":    idx_name,
            "close_val":     float(row["close_val"]) if pd.notna(row["close_val"]) else None,
            "pct_chg_1d":    float(row["pct_chg"])   if pd.notna(row["pct_chg"])   else None,
            "ret_1w":        r1w,
            "ret_1m":        r1m,
            "ret_3m":        r3m,
            "vs_nifty50":    vs_nifty,
            "pe_ratio":      float(row["pe_ratio"])  if pd.notna(row["pe_ratio"])  else None,
            "pb_ratio":      float(row["pb_ratio"])  if pd.notna(row["pb_ratio"])  else None,
            "div_yield":     float(row["div_yield"]) if pd.notna(row["div_yield"]) else None,
        })

    return pd.DataFrame(rows)


def get_index_history(
    index_name: str,
    trade_date: date,
    lookback_days: int = 120,
) -> pd.DataFrame:
    """
    Return daily OHLC + pct_chg history for one index.
    """
    df = query_dataframe(f"""
        SELECT trade_date, open_val, high_val, low_val, close_val,
               points_chg, pct_chg, turnover_cr, pe_ratio
        FROM index_data
        WHERE index_name = '{index_name.replace("'", "''")}'
          AND trade_date <= '{trade_date}'
          AND trade_date >= ('{trade_date}'::DATE - INTERVAL {lookback_days} DAY)
        ORDER BY trade_date
    """)
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def get_index_heatmap(trade_date: date) -> pd.DataFrame:
    """
    All indices for the given date — used for the full heatmap view.
    Returns index_name, close_val, pct_chg sorted by pct_chg desc.
    """
    df = query_dataframe(f"""
        SELECT index_name, close_val, pct_chg, turnover_cr
        FROM index_data
        WHERE trade_date = '{trade_date}'
        ORDER BY pct_chg DESC
    """)
    return df
