"""
FPI Capital Flow Analytics — NSDL FPI investment flow signals.

Key concepts:
  Equity net flow > 0  → foreign money entering Indian equity market → bullish pressure
  Equity net flow < 0  → FPI selling → supply / bearish pressure

  Risk Appetite Score  = equity_net_15d / (|equity| + |debt| + |hybrid| + ε) × 100
    100% = 100% of net flow in equity (max risk-on)
    0%   = neutral / mixed
    <0%  = net selling equity while buying debt (risk-off flight)

  15-20 Day Outlook = composite of:
    1. Cumulative equity net flow over last 15 dates
    2. Risk appetite score trend
    3. Capital flight signal (simultaneous equity + debt outflows)
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_fpi_summary",
    "get_fpi_category_breakdown",
    "get_fpi_risk_appetite",
    "get_fpi_15d_outlook",
    "get_fpi_available_dates",
    "get_fpi_date_range",
]

_EQUITY_CATS = {"Equity"}
_DEBT_CATS   = {"Debt", "Debt-VRR"}
_RISK_CATS   = {"Equity", "Hybrid"}


def get_fpi_available_dates() -> list[date]:
    """All distinct dates with FPI data, most recent first."""
    df = query_dataframe(
        "SELECT DISTINCT trade_date FROM fpi_nsdl_flows ORDER BY trade_date DESC"
    )
    return list(df["trade_date"]) if not df.empty else []


def get_fpi_date_range() -> tuple[date | None, date | None]:
    """Return (min_date, max_date) of data in DB, or (None, None) if empty."""
    df = query_dataframe(
        "SELECT MIN(trade_date) AS min_d, MAX(trade_date) AS max_d FROM fpi_nsdl_flows"
    )
    if df.empty:
        return None, None
    min_d = df["min_d"].iloc[0]
    max_d = df["max_d"].iloc[0]
    # DuckDB returns NaT/NaN when table is empty
    try:
        import pandas as pd
        if pd.isna(min_d) or pd.isna(max_d):
            return None, None
    except (TypeError, ValueError):
        pass
    return min_d, max_d


def get_fpi_summary(as_of_date: date, lookback_days: int = 180) -> pd.DataFrame:
    """
    Daily FPI flows for all categories, last N calendar days up to as_of_date.
    Returns columns: trade_date, category, gross_purchase_cr, gross_sales_cr, net_investment_cr
    """
    start = as_of_date - timedelta(days=lookback_days)
    return query_dataframe("""
        SELECT trade_date, category, gross_purchase_cr, gross_sales_cr, net_investment_cr
        FROM fpi_nsdl_flows
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date, category
    """, [start, as_of_date])


def get_fpi_category_breakdown(as_of_date: date, lookback_days: int = 15) -> pd.DataFrame:
    """
    Cumulative flow per category over last N calendar days.
    Returns: category, gross_purchase_cr, gross_sales_cr, net_investment_cr, net_pct
    """
    start = as_of_date - timedelta(days=lookback_days * 2)  # buffer for weekends
    df = query_dataframe("""
        SELECT category,
               SUM(gross_purchase_cr) AS gross_purchase_cr,
               SUM(gross_sales_cr)    AS gross_sales_cr,
               SUM(net_investment_cr) AS net_investment_cr
        FROM fpi_nsdl_flows
        WHERE trade_date >= ? AND trade_date <= ?
        GROUP BY category
        ORDER BY net_investment_cr DESC
    """, [start, as_of_date])

    if df.empty:
        return df

    total_abs = df["net_investment_cr"].abs().sum()
    df["net_pct"] = (df["net_investment_cr"] / (total_abs + 1e-9) * 100).round(1)
    return df


def get_fpi_risk_appetite(as_of_date: date, lookback_days: int = 90) -> pd.DataFrame:
    """
    Daily risk appetite score time series (last N calendar days).
    Columns: trade_date, equity_net, debt_net, hybrid_net, total_net, risk_score
    risk_score = equity_net / (|equity| + |debt| + |hybrid| + ε) × 100
    """
    start = as_of_date - timedelta(days=lookback_days)
    raw = query_dataframe("""
        SELECT trade_date, category, net_investment_cr
        FROM fpi_nsdl_flows
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
    """, [start, as_of_date])

    if raw.empty:
        return pd.DataFrame()

    pivot = raw.pivot_table(
        index="trade_date", columns="category",
        values="net_investment_cr", aggfunc="sum", fill_value=0,
    ).reset_index()

    pivot["equity_net"]  = pivot.get("Equity", 0)
    pivot["debt_net"]    = pivot.get("Debt", 0) + pivot.get("Debt-VRR", 0)
    pivot["hybrid_net"]  = pivot.get("Hybrid", 0)
    pivot["others_net"]  = pivot.get("Others", 0)
    pivot["total_net"]   = (
        pivot["equity_net"] + pivot["debt_net"] +
        pivot["hybrid_net"] + pivot["others_net"]
    )
    denom = (
        pivot["equity_net"].abs() + pivot["debt_net"].abs() +
        pivot["hybrid_net"].abs() + 1e-9
    )
    pivot["risk_score"] = (pivot["equity_net"] / denom * 100).round(1)

    cols = ["trade_date", "equity_net", "debt_net", "hybrid_net", "others_net",
            "total_net", "risk_score"]
    return pivot[[c for c in cols if c in pivot.columns]].sort_values("trade_date")


def get_fpi_15d_outlook(as_of_date: date) -> dict:
    """
    Compute 15-20 day market outlook from FPI flow patterns.

    Returns dict with:
      signal       — "STRONGLY BULLISH" / "BULLISH" / "NEUTRAL" / "BEARISH" / "STRONGLY BEARISH"
      score        — integer −5 to +5
      equity_15d   — cumulative equity net flow over last 15 data dates (Cr)
      risk_score   — avg risk appetite score (%) over last 15 data dates
      capital_flight — bool: simultaneous equity + debt outflows
      days_of_data — number of trading dates available for the calculation
      rationale    — plain-English explanation
    """
    appetite = get_fpi_risk_appetite(as_of_date, lookback_days=60)
    if appetite.empty:
        return _empty_outlook("No FPI data available")

    # Use last 15 actual data dates (not calendar days)
    last15 = appetite.tail(15)
    n = len(last15)

    equity_15d   = float(last15["equity_net"].sum())
    debt_15d     = float(last15["debt_net"].sum())
    hybrid_15d   = float(last15.get("hybrid_net", pd.Series([0])).sum())
    avg_risk     = float(last15["risk_score"].mean())
    capital_flight = (equity_15d < -2000) and (debt_15d < -2000)

    # Score components
    # 1. Equity momentum (−3 to +3)
    if equity_15d > 12_000:
        eq_score = 3
    elif equity_15d > 5_000:
        eq_score = 2
    elif equity_15d > 1_500:
        eq_score = 1
    elif equity_15d < -12_000:
        eq_score = -3
    elif equity_15d < -5_000:
        eq_score = -2
    elif equity_15d < -1_500:
        eq_score = -1
    else:
        eq_score = 0

    # 2. Risk appetite tilt (−2 to +2)
    if avg_risk > 60:
        ra_score = 2
    elif avg_risk > 40:
        ra_score = 1
    elif avg_risk < -20:
        ra_score = -2
    elif avg_risk < 10:
        ra_score = -1
    else:
        ra_score = 0

    # 3. Capital flight penalty
    cf_score = -2 if capital_flight else 0

    total = eq_score + ra_score + cf_score

    if total >= 4:
        signal = "STRONGLY BULLISH"
    elif total >= 2:
        signal = "BULLISH"
    elif total >= 1:
        signal = "MILDLY BULLISH"
    elif total <= -4:
        signal = "STRONGLY BEARISH"
    elif total <= -2:
        signal = "BEARISH"
    elif total <= -1:
        signal = "MILDLY BEARISH"
    else:
        signal = "NEUTRAL"

    rationale = _build_rationale(signal, equity_15d, avg_risk, capital_flight, debt_15d, n)

    return {
        "signal":        signal,
        "score":         total,
        "equity_15d":    round(equity_15d, 1),
        "debt_15d":      round(debt_15d, 1),
        "risk_score":    round(avg_risk, 1),
        "capital_flight": capital_flight,
        "days_of_data":  n,
        "rationale":     rationale,
    }


def _empty_outlook(reason: str) -> dict:
    return {
        "signal": "N/A", "score": 0,
        "equity_15d": 0.0, "debt_15d": 0.0, "risk_score": 0.0,
        "capital_flight": False, "days_of_data": 0, "rationale": reason,
    }


def _build_rationale(
    signal: str, equity_15d: float, risk_score: float,
    capital_flight: bool, debt_15d: float, n: int,
) -> str:
    dir_word = "bought" if equity_15d >= 0 else "sold"
    eq_abs   = abs(equity_15d)
    parts = [
        f"FPIs {dir_word} ₹{eq_abs:,.0f} Cr of equity over last {n} sessions.",
        f"Risk appetite: {risk_score:+.1f}% ({'risk-on' if risk_score > 40 else 'risk-off' if risk_score < 10 else 'mixed'}).",
    ]
    if capital_flight:
        parts.append("Capital flight alert: selling both equity AND debt simultaneously.")
    elif debt_15d > 3_000:
        parts.append(f"Debt also seeing inflows (₹{debt_15d:,.0f} Cr) — mixed risk appetite.")
    parts.append(f"15-20 day outlook: {signal}.")
    return " ".join(parts)
