"""
Sector Signal Backtest — historical accuracy of the 1–2 Week Sector Outlook signal.

Walk-forward methodology (no look-ahead bias):
  For each past trading date D and each sector:
    1. Compute the outlook signal using ONLY data available up to and including day D.
       Rolling 100-day baseline uses shift(1) before rolling — excludes day D itself.
    2. Measure the sector's cumulative price return over the next 5 trading days.
    3. Score: CORRECT if signal direction matched actual next-5D return.

Signal → Outcome mapping:
  Accumulating / Buying Dips → Correct if fwd_5d_pct > 0%  (bullish thesis)
  Distributing               → Correct if fwd_5d_pct < 0%  (bearish thesis)
  Weak Rally / Neutral       → Not directional → N/A (not scored)

Simplified signal (no breadth):
  The live dashboard also weighs per-stock breadth (fraction of stocks above their own
  100D norm). Computing that per-day for 300 days × 30 sectors × 50 stocks is expensive.
  The backtest approximates "broad" via extreme z/dv thresholds, making it slightly more
  conservative (fewer Accumulating calls) but fast and bias-free.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_sector_signal_log", "get_sector_accuracy_summary"]

_MIN_PERIODS  = 20   # minimum rolling days before we trust the 100D stat
_FWD_DAYS     = 5    # evaluate next-5-trading-day cumulative return


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_sector_daily(
    from_date: date, to_date: date, min_turnover_lacs: float
) -> pd.DataFrame:
    return query_dataframe("""
        SELECT
            d.trade_date,
            sm.sector,
            SUM(d.turnover_lacs * d.deliv_per / 10000.0)  AS daily_dv_cr,
            SUM(
                CASE WHEN d.prev_close > 0
                THEN ((d.close_price - d.prev_close) / d.prev_close * 100.0)
                     * d.turnover_lacs
                ELSE 0.0 END
            ) / NULLIF(
                SUM(CASE WHEN d.prev_close > 0 THEN d.turnover_lacs ELSE 0.0 END),
                0.0
            ) AS daily_price_chg_pct
        FROM daily_data d
        JOIN sector_master sm ON d.symbol = sm.symbol
        WHERE d.trade_date >= ?
          AND d.trade_date <= ?
          AND d.series IN ('EQ', 'SM', 'ST')
          AND d.turnover_lacs >= ?
        GROUP BY d.trade_date, sm.sector
        ORDER BY sm.sector, d.trade_date
    """, [from_date, to_date, max(min_turnover_lacs, 0.0)])


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 100-day rolling mean and std per sector with shift(1) to exclude
    the current day from its own baseline — no look-ahead bias.
    """
    df = df.sort_values(["sector", "trade_date"]).copy()

    mean_parts: list[pd.Series] = []
    std_parts:  list[pd.Series] = []

    for _sector, grp in df.groupby("sector", sort=False):
        shifted = grp["daily_dv_cr"].shift(1)
        rolling = shifted.rolling(100, min_periods=_MIN_PERIODS)
        mean_parts.append(rolling.mean())
        std_parts.append(rolling.std())

    df["mean_100d"] = pd.concat(mean_parts)
    df["std_100d"]  = pd.concat(std_parts)

    mean_safe = df["mean_100d"].replace(0, float("nan"))
    std_safe  = df["std_100d"].replace(0, float("nan"))
    df["dv_ratio"] = df["daily_dv_cr"] / mean_safe
    df["z_score"]  = (df["daily_dv_cr"] - df["mean_100d"]) / std_safe
    return df


def _classify_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirrors sector_performance._sig() without per-day breadth.

    "broad" is approximated by z >= 1.5 or dv >= 1.8, matching the live
    signal's behaviour when breadth >= 50% of stocks.
    """
    def _sig(row) -> str:
        dv = row["dv_ratio"]
        z  = row["z_score"]
        pm = row["daily_price_chg_pct"]

        if pd.isna(dv) or pd.isna(z):
            return "Insufficient Data"

        pm = pm if not pd.isna(pm) else 0.0

        # Abnormal: same thresholds as live signal
        abnormal = z >= 1.5 or dv >= 1.5 or (z >= 1.0 and dv >= 1.2)
        # Broad proxy (without per-stock breadth): z >= 1.5 ≈ top ~7% of days per sector
        broad    = z >= 1.5 or dv >= 1.8
        strong   = abnormal and broad

        if strong and pm > 0:              return "Accumulating"
        if strong:                         return "Buying Dips"
        if dv >= 0.75 and pm > 0:         return "Weak Rally"
        if dv >= 0.60 and abs(pm) <= 1.5: return "Neutral"
        return "Distributing"

    df["signal"] = df.apply(_sig, axis=1)
    return df


def _compute_forward_returns(df: pd.DataFrame, fwd_days: int = _FWD_DAYS) -> pd.DataFrame:
    """Sum of next `fwd_days` daily price changes per sector (in-sample future)."""
    df = df.sort_values(["sector", "trade_date"]).copy()

    fwd_parts: list[pd.DataFrame] = []
    for _sector, grp in df.groupby("sector", sort=False):
        grp = grp.sort_values("trade_date").copy()
        fwd_cols: list[str] = []
        for i in range(1, fwd_days + 1):
            col = f"_fwd_{i}d"
            grp[col] = grp["daily_price_chg_pct"].shift(-i)
            fwd_cols.append(col)
        grp["n_fwd_days"] = grp[fwd_cols].notna().sum(axis=1)
        grp["fwd_5d_pct"] = grp[fwd_cols].sum(axis=1, min_count=1)
        fwd_parts.append(grp.drop(columns=fwd_cols))

    return pd.concat(fwd_parts)


def _classify_outcomes(df: pd.DataFrame, fwd_days: int = _FWD_DAYS) -> pd.DataFrame:
    def _outcome(row) -> str:
        sig   = row["signal"]
        n_fwd = int(row.get("n_fwd_days", 0))
        fwd   = row.get("fwd_5d_pct", float("nan"))

        if sig in ("Insufficient Data", "Weak Rally", "Neutral"):
            return "N/A"

        if n_fwd == 0 or pd.isna(fwd):
            return "Pending"
        if n_fwd < fwd_days:
            return f"Partial ({n_fwd}/{fwd_days}d)"

        # Directional scoring — any positive/negative return counts as correct
        if sig in ("Accumulating", "Buying Dips"):
            return "Correct" if fwd > 0.0 else "Wrong"
        if sig == "Distributing":
            return "Correct" if fwd < 0.0 else "Wrong"
        return "N/A"

    df["outcome"] = df.apply(_outcome, axis=1)
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def get_sector_signal_log(
    as_of_date: date,
    min_turnover_lacs: float = 0,
    lookback_dates: int = 30,
    fwd_days: int = _FWD_DAYS,
) -> pd.DataFrame:
    """
    Historical sector signal log with backtest outcomes.

    Returns one row per (trade_date, sector) for the `lookback_dates` most
    recent trading dates up to and including as_of_date.

    Columns: trade_date, sector, signal, dv_ratio, z_score,
             daily_price_chg_pct, fwd_5d_pct, n_fwd_days, outcome
    """
    # Need 100D rolling baseline + eval window + forward returns buffer
    # 460 calendar days ≈ 320 trading days — comfortably covers 100D baseline + 200D eval
    query_start = as_of_date - timedelta(days=460)

    raw = _load_sector_daily(query_start, as_of_date, min_turnover_lacs)
    if raw.empty:
        return pd.DataFrame()

    df = _compute_rolling_stats(raw)
    df = _classify_signals(df)
    df = _compute_forward_returns(df, fwd_days=fwd_days)
    df = _classify_outcomes(df, fwd_days=fwd_days)

    # Keep only the `lookback_dates` most recent trading dates.
    # df["trade_date"].unique() yields pandas Timestamps; as_of_date is a
    # datetime.date — normalize to Timestamp so the comparison doesn't raise.
    as_of_ts = pd.Timestamp(as_of_date)
    all_dates = sorted(df["trade_date"].unique())
    eval_dates = [d for d in all_dates if d <= as_of_ts]
    eval_dates = set(eval_dates[-lookback_dates:])

    result = df[df["trade_date"].isin(eval_dates)].copy()
    cols = [
        "trade_date", "sector", "signal", "dv_ratio", "z_score",
        "daily_price_chg_pct", "fwd_5d_pct", "n_fwd_days", "outcome",
    ]
    result = result[[c for c in cols if c in result.columns]]
    return (
        result
        .sort_values(["trade_date", "sector"], ascending=[False, True])
        .reset_index(drop=True)
    )


def get_sector_accuracy_summary(
    as_of_date: date,
    min_turnover_lacs: float = 0,
    lookback_dates: int = 60,
) -> dict:
    """
    Aggregate win-rate stats by signal type for the last `lookback_dates` trading dates.

    Only counts completed outcomes (Correct / Wrong).  Pending and Partial rows
    are excluded so early signals do not artificially inflate win rates.

    Returns dict keyed by signal name ("Accumulating", "Buying Dips", "Distributing",
    "__overall__") with keys: n, win_rate, avg_fwd_5d, median_fwd_5d.
    """
    log = get_sector_signal_log(as_of_date, min_turnover_lacs, lookback_dates, fwd_days=5)
    if log.empty:
        return {}

    completed = log[log["outcome"].isin(["Correct", "Wrong"])].copy()
    if completed.empty:
        return {}

    stats: dict = {}
    for sig in ("Accumulating", "Buying Dips", "Distributing"):
        subset = completed[completed["signal"] == sig]
        if subset.empty:
            continue
        n       = len(subset)
        correct = int((subset["outcome"] == "Correct").sum())
        stats[sig] = {
            "n":             n,
            "win_rate":      correct / n,
            "avg_fwd_5d":    float(subset["fwd_5d_pct"].mean()),
            "median_fwd_5d": float(subset["fwd_5d_pct"].median()),
        }

    # Overall directional accuracy
    dir_completed = completed[completed["signal"].isin(("Accumulating", "Buying Dips", "Distributing"))]
    if not dir_completed.empty:
        n_total   = len(dir_completed)
        n_correct = int((dir_completed["outcome"] == "Correct").sum())
        stats["__overall__"] = {
            "n":        n_total,
            "win_rate": n_correct / n_total,
        }

    return stats
