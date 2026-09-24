"""
SLB (Securities Lending & Borrowing) analytics.

WHAT THE DATA IS
    slb_open_positions  shares still out on loan per (symbol, series) at end of day.
                        Cash-equity shorts can only be carried overnight in India
                        by borrowing here, so this is the closest thing NSE
                        publishes to per-stock short interest.
    slb_bhavcopy        per-contract fee prints. A fee is RUPEES PER SHARE FOR THE
                        WHOLE REMAINING TENURE; annualise before comparing.

WHAT THE BACKTEST SAYS (scripts/build_slb_study.py, 2018-03..2026-09, 2,109 sessions)
    Heavy / fresh borrowing is followed by UNDERPERFORMANCE, and it survives the
    controls that killed earlier signals in this project:
      - dividend-adjusted returns (NSE's prev_close is not adjusted for dividends)
      - ex-date windows removed, sector-neutral target
      - past 5d / 21d return, volatility, liquidity in a Fama-MacBeth regression
      - futures basis (reverse cash-futures arbitrage is a real part of borrowing,
        corr -0.13..-0.16, but the signal keeps ~90% of its size with it controlled)
      - dropping the worst 5-20 names from EVERY decile (it is not DHFL + YESBANK)
    It lives in the TOP DECILE only; deciles 1-8 are flat. And it does NOT pay as a
    short: the top decile's own borrow fee is ~4-5% a year (~35 bps a month) plus
    ~20 bps STT round trip, against a ~50 bps a month RELATIVE gap. Use it as an
    AVOID / TRIM flag on longs, which costs nothing.

    The live flag reproduces the study's exact definition: top decile of 5-session
    borrowing growth OR outstanding / 20d delivery, among top-300-liquidity names
    that have SLB outstanding and no corporate action in the last 25 sessions.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "latest_slb_date", "borrow_board", "lend_board", "market_history",
    "symbol_history", "foreclosures", "load_evidence", "series_mix",
]

_STUDY = Path(__file__).resolve().parents[2] / "data" / "slb_study" / "summary.json"


def latest_slb_date(on_or_before: date) -> date | None:
    df = query_dataframe(
        "SELECT max(trade_date) d FROM slb_open_positions WHERE trade_date <= ?", [on_or_before])
    v = df["d"].iat[0] if len(df) else None
    return None if v is None or pd.isna(v) else pd.Timestamp(v).date()


def _sessions(d: date, n: int) -> list[date]:
    df = query_dataframe(
        "SELECT DISTINCT trade_date FROM slb_open_positions WHERE trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT ?", [d, n])
    return [pd.Timestamp(x).date() for x in df["trade_date"]]


def borrow_board(d: date) -> pd.DataFrame:
    """One row per stock with SLB outstanding on d, with the study's features."""
    sess = _sessions(d, 21)
    if not sess:
        return pd.DataFrame()
    d = sess[0]
    d5 = sess[5] if len(sess) > 5 else None
    d20 = sess[20] if len(sess) > 20 else None
    px = query_dataframe(
        """
        WITH px AS (
            SELECT trade_date, symbol, close_price, prev_close,
                   coalesce(deliv_qty, 0) AS deliv_qty, turnover_lacs
            FROM daily_data
            WHERE series = 'EQ' AND close_price > 0 AND prev_close > 0
              AND trade_date BETWEEN ? AND ?
        ), w AS (
            SELECT *,
              avg(deliv_qty) OVER (PARTITION BY symbol ORDER BY trade_date
                                   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS dq20,
              avg(turnover_lacs) OVER (PARTITION BY symbol ORDER BY trade_date
                                       ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS adv60,
              close_price / lag(close_price, 21) OVER p - 1 AS r21,
              CASE WHEN abs(prev_close / nullif(lag(close_price) OVER p, 0) - 1) > 0.02
                   THEN 1 ELSE 0 END AS ca_day
            FROM px WINDOW p AS (PARTITION BY symbol ORDER BY trade_date)
        ), w2 AS (
            SELECT *, max(ca_day) OVER (PARTITION BY symbol ORDER BY trade_date
                                        ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS ca_recent
            FROM w
        )
        SELECT w2.symbol, close_price, dq20, adv60, r21, ca_recent, turnover_lacs,
               coalesce(m.sector, '') AS sector
        FROM w2 LEFT JOIN v_sector_master m USING (symbol)
        WHERE trade_date = ?
        """,
        [d - timedelta(days=150), d, d],
    )
    oi = query_dataframe(
        """
        SELECT trade_date, symbol, sum(open_qty) AS oi,
               sum(CASE WHEN series_kind = 'NO_AGM' THEN open_qty ELSE 0 END) AS oi_x
        FROM slb_open_positions WHERE trade_date IN (?, ?, ?)
        GROUP BY 1, 2
        """,
        [d, d5 or d, d20 or d],
    )
    oi["trade_date"] = pd.to_datetime(oi["trade_date"]).dt.date
    now = oi.loc[oi["trade_date"] == d].set_index("symbol")
    if now.empty:
        return pd.DataFrame()
    prev5 = oi.loc[oi["trade_date"] == d5].set_index("symbol")["oi"] if d5 else pd.Series(dtype=float)
    prev20 = oi.loc[oi["trade_date"] == d20].set_index("symbol")["oi"] if d20 else pd.Series(dtype=float)

    fee = query_dataframe(
        """
        SELECT b.symbol,
               sum((b.value / b.quantity) / p.close_price * 365.0
                   / greatest(date_diff('day', b.trade_date, b.expiry_date), 1)
                   * b.quantity * p.close_price) / nullif(sum(b.quantity * p.close_price), 0) AS fee_ann,
               max(b.trade_date) AS last_fee_date
        FROM slb_bhavcopy b
        JOIN daily_data p ON p.symbol = b.symbol AND p.trade_date = b.trade_date AND p.series = 'EQ'
        WHERE b.trade_date BETWEEN ? AND ? AND b.series_kind <> 'ROLLOVER' AND b.quantity > 0
          AND date_diff('day', b.trade_date, b.expiry_date) >= 10
          AND b.quantity * p.close_price >= 100000
        GROUP BY 1
        """,
        [sess[-1], d],
    ).set_index("symbol")

    df = now[["oi", "oi_x"]].join(px.set_index("symbol"), how="left").join(fee, how="left")
    df["oi_5"] = prev5.reindex(df.index).fillna(0)
    df["oi_20"] = prev20.reindex(df.index).fillna(0)
    df["oi_cr"] = df["oi"] * df["close_price"] / 1e7
    df["dtc"] = df["oi"] / df["dq20"].replace(0, np.nan)
    df["doi5"] = (df["oi"] - df["oi_5"]) / df["dq20"].replace(0, np.nan)
    df["chg20_pct"] = np.where(df["oi_20"] > 0, (df["oi"] / df["oi_20"] - 1) * 100, np.nan)
    df["x_share"] = df["oi_x"] / df["oi"]

    # Study universe: top-300 by lagged 60d turnover across ALL EQ names that day.
    rank = px.set_index("symbol")["adv60"].rank(ascending=False, method="first")
    df["liq_rank"] = rank.reindex(df.index)
    elig = (df["liq_rank"] <= 300) & (df["ca_recent"] == 0)
    df["in_study"] = elig
    for sig in ("doi5", "dtc"):
        v = df.loc[elig & df[sig].notna(), sig]
        df[f"{sig}_pct"] = v.rank(pct=True).reindex(df.index)
    df["flag_fresh"] = df["doi5_pct"] > 0.9
    df["flag_heavy"] = df["dtc_pct"] > 0.9
    df.attrs["as_of"] = d
    df.attrs["n_study"] = int(elig.sum())
    return df.reset_index().rename(columns={"index": "symbol"})


def lend_board(d: date) -> pd.DataFrame:
    """Every fee print on d, annualised against the cash close: the lender's view."""
    df = query_dataframe(
        """
        SELECT b.symbol, b.series, b.series_kind, b.expiry_date,
               date_diff('day', b.trade_date, b.expiry_date) AS days,
               b.value / b.quantity AS fee_vwap, b.low_fee, b.high_fee,
               b.quantity, b.trades, b.value, p.close_price
        FROM slb_bhavcopy b
        LEFT JOIN daily_data p ON p.symbol = b.symbol AND p.trade_date = b.trade_date
                              AND p.series = 'EQ'
        WHERE b.trade_date = ? AND b.quantity > 0
        """,
        [d],
    )
    if df.empty:
        return df
    df["notional_cr"] = df["quantity"] * df["close_price"] / 1e7
    df["yield_ann_pct"] = (df["fee_vwap"] / df["close_price"] * 365
                           / df["days"].clip(lower=1) * 100)
    return df.sort_values("value", ascending=False).reset_index(drop=True)


def market_history(start: date, end: date) -> pd.DataFrame:
    """Daily market totals: outstanding value, names, fee rupees, fresh lending."""
    return query_dataframe(
        """
        WITH o AS (
            SELECT o.trade_date, o.symbol, sum(o.open_qty) AS q
            FROM slb_open_positions o WHERE o.trade_date BETWEEN ? AND ?
            GROUP BY 1, 2
        ), v AS (
            SELECT o.trade_date, sum(o.q * p.close_price) / 1e7 AS oi_cr, count(*) AS names
            FROM o JOIN daily_data p ON p.symbol = o.symbol AND p.trade_date = o.trade_date
                                    AND p.series = 'EQ'
            GROUP BY 1
        ), f AS (
            SELECT b.trade_date, sum(b.value) / 1e7 AS fee_cr,
                   sum(CASE WHEN b.series_kind <> 'ROLLOVER'
                            THEN b.quantity * p.close_price END) / 1e7 AS lent_cr
            FROM slb_bhavcopy b
            LEFT JOIN daily_data p ON p.symbol = b.symbol AND p.trade_date = b.trade_date
                                  AND p.series = 'EQ'
            WHERE b.trade_date BETWEEN ? AND ?
            GROUP BY 1
        )
        SELECT v.trade_date, v.oi_cr, v.names, coalesce(f.fee_cr, 0) AS fee_cr,
               coalesce(f.lent_cr, 0) AS lent_cr
        FROM v LEFT JOIN f USING (trade_date)
        ORDER BY 1
        """,
        [start, end, start, end],
    )


def series_mix(start: date, end: date) -> pd.DataFrame:
    """Share of fee rupees by contract family per year (regular vs AGM-proof vs roll)."""
    return query_dataframe(
        """
        SELECT year(trade_date) AS year, series_kind, sum(value) / 1e7 AS fee_cr
        FROM slb_bhavcopy WHERE trade_date BETWEEN ? AND ?
        GROUP BY 1, 2 ORDER BY 1, 2
        """,
        [start, end],
    )


def symbol_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    df = query_dataframe(
        """
        WITH o AS (
            SELECT trade_date, sum(open_qty) AS oi FROM slb_open_positions
            WHERE symbol = ? AND trade_date BETWEEN ? AND ? GROUP BY 1
        ), f AS (
            SELECT trade_date, sum(value) / nullif(sum(quantity), 0) AS fee_vwap,
                   sum(quantity) AS lent_qty,
                   min(date_diff('day', trade_date, expiry_date)) AS min_days
            FROM slb_bhavcopy
            WHERE symbol = ? AND trade_date BETWEEN ? AND ? AND series_kind <> 'ROLLOVER'
            GROUP BY 1
        )
        SELECT p.trade_date, p.close_price, coalesce(o.oi, 0) AS oi, p.deliv_qty,
               f.fee_vwap, f.lent_qty, f.min_days
        FROM daily_data p
        LEFT JOIN o USING (trade_date)
        LEFT JOIN f USING (trade_date)
        WHERE p.symbol = ? AND p.series = 'EQ' AND p.trade_date BETWEEN ? AND ?
          AND p.trade_date IN (SELECT DISTINCT trade_date FROM slb_open_positions)
        ORDER BY 1
        """,
        [symbol, start, end, symbol, start, end, symbol, start, end],
    )
    if not df.empty:
        df["oi_cr"] = df["oi"] * df["close_price"] / 1e7
    return df


def foreclosures(d: date) -> pd.DataFrame:
    try:
        return query_dataframe(
            "SELECT * FROM slb_foreclosure WHERE record_date >= ? ORDER BY record_date",
            [d - timedelta(days=21)])
    except Exception:
        return pd.DataFrame()


def load_evidence() -> dict:
    try:
        return json.loads(_STUDY.read_text())
    except Exception:
        return {}
