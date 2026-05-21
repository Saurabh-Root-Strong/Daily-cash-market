from datetime import date, timedelta
from typing import Optional, Dict
import pandas as pd

from src.analytics.delivery_signals import get_stock_metrics
from src.analytics.base import get_weighting_method, get_min_turnover_filter
from src.data.repository import query_dataframe
from src.logging_setup import get_logger

log = get_logger(__name__)


def aggregate_by_sector(
    trade_date: date,
    weighting: Optional[str] = None,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    if weighting is None:
        weighting = get_weighting_method()
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    df = get_stock_metrics(trade_date, min_turnover_lacs=min_turnover_lacs)
    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=["sector"])

    acc_threshold, dist_threshold = 1.2, 0.8

    records = []
    for sector, grp in df.groupby("sector"):
        total_turnover = grp["turnover_lacs"].sum()
        total_deliv_value = grp["deliv_value_lacs"].sum()
        stock_count = len(grp)

        valid_price = grp.dropna(subset=["price_change_pct"])
        valid_deliv = grp.dropna(subset=["deliv_per"])

        simple_price = valid_price["price_change_pct"].mean()
        simple_deliv = valid_deliv["deliv_per"].mean()

        if weighting == "turnover" and total_turnover > 0:
            w = valid_price["turnover_lacs"] / valid_price["turnover_lacs"].sum()
            wtd_price = (valid_price["price_change_pct"] * w).sum() if not valid_price.empty else None

            w2 = valid_deliv["turnover_lacs"] / valid_deliv["turnover_lacs"].sum()
            wtd_deliv = (valid_deliv["deliv_per"] * w2).sum() if not valid_deliv.empty else None
        else:
            wtd_price = simple_price
            wtd_deliv = simple_deliv

        top_deliv = grp.dropna(subset=["deliv_per"])
        top_deliv_symbol = (
            top_deliv.nlargest(1, "deliv_per")["symbol"].iloc[0]
            if not top_deliv.empty else None
        )

        acc_count = int((grp["deliv_ratio"] >= acc_threshold).sum()) if "deliv_ratio" in grp.columns else 0
        dist_count = int((grp["deliv_ratio"] < dist_threshold).sum()) if "deliv_ratio" in grp.columns else 0

        records.append({
            "sector": sector,
            "stock_count": stock_count,
            "simple_price_change_pct": simple_price,
            "simple_deliv_per": simple_deliv,
            "wtd_price_change_pct": wtd_price,
            "wtd_deliv_per": wtd_deliv,
            "top_delivery_symbol": top_deliv_symbol,
            "accumulation_count": acc_count,
            "distribution_count": dist_count,
            "total_turnover_lacs": total_turnover,
            "total_deliv_value_lacs": total_deliv_value,
        })

    result = pd.DataFrame(records)
    result = result.sort_values("wtd_deliv_per", ascending=False).reset_index(drop=True)
    return result


def get_sector_drilldown(trade_date: date, sector_name: str, top_n: int = 10) -> Dict:
    df = get_stock_metrics(trade_date)
    if df.empty:
        return {}

    sector_df = df[df["sector"] == sector_name].copy()
    if sector_df.empty:
        return {}

    total_turnover = sector_df["turnover_lacs"].sum()
    total_deliv_value = sector_df["deliv_value_lacs"].sum()

    sector_df["turnover_share_pct"] = (sector_df["turnover_lacs"] / total_turnover * 100).round(2)
    sector_df["deliv_value_share_pct"] = (sector_df["deliv_value_lacs"] / total_deliv_value * 100).round(2)

    top_by_delivery_pct = sector_df.nlargest(top_n, "deliv_per")
    top_by_delivery_value = sector_df.nlargest(top_n, "deliv_value_lacs")
    top_by_turnover = sector_df.nlargest(top_n, "turnover_lacs")
    contribution_table = sector_df.nlargest(top_n, "turnover_lacs")

    # Sub-sector summary — turnover-weighted delivery % across ALL stocks
    # Simple avg(deliv_per) is misleading: a ₹5000 stock with 60% delivery
    # delivers far more real money than a ₹50 stock with 70% delivery.
    # Weighting by turnover (price × qty) gives the true conviction signal.
    def _agg_subsector(g: pd.DataFrame) -> pd.Series:
        total_to = g["turnover_lacs"].sum()
        wtd_deliv = (
            (g["deliv_per"] * g["turnover_lacs"]).sum() / total_to
            if total_to > 0 else g["deliv_per"].mean()
        )
        return pd.Series({
            "wtd_deliv_per":         wtd_deliv,
            "simple_deliv_per":      g["deliv_per"].mean(),
            "avg_price_chg":         g["price_change_pct"].mean(),
            "stock_count":           len(g),
            "total_turnover_lacs":   total_to,
            "total_deliv_value_lacs": g["deliv_value_lacs"].sum(),
        })

    subsector_summary = (
        sector_df.dropna(subset=["industry"])
        .groupby("industry")
        .apply(_agg_subsector)
        .reset_index()
        .sort_values("wtd_deliv_per", ascending=False)
        .reset_index(drop=True)
    )

    sector_summary = {
        "stock_count": len(sector_df),
        "total_turnover_lacs": total_turnover,
        "total_deliv_value_lacs": total_deliv_value,
        "avg_price_change_pct": sector_df["price_change_pct"].mean(),
        "avg_deliv_per": sector_df["deliv_per"].mean(),
    }

    return {
        "top_by_delivery_pct": top_by_delivery_pct,
        "top_by_delivery_value": top_by_delivery_value,
        "top_by_turnover": top_by_turnover,
        "contribution_table": contribution_table,
        "subsector_summary": subsector_summary,
        "sector_summary": sector_summary,
    }


def get_sector_history(sector_name: str, days: int = 60) -> pd.DataFrame:
    min_turnover_lacs = get_min_turnover_filter()
    sql = """
        SELECT
            b.trade_date,
            SUM(b.deliv_per * b.turnover_lacs) / NULLIF(SUM(b.turnover_lacs), 0)
                AS avg_deliv_per,
            SUM(
                CASE WHEN b.prev_close > 0
                THEN (b.close_price - b.prev_close) / b.prev_close * 100
                END * b.turnover_lacs
            ) / NULLIF(SUM(CASE WHEN b.prev_close > 0 THEN b.turnover_lacs END), 0)
                AS avg_price_change_pct,
            SUM(b.turnover_lacs) / 100 AS total_turnover_cr,
            COUNT(DISTINCT b.symbol) AS stock_count
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ?
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
        GROUP BY b.trade_date
        ORDER BY b.trade_date DESC
        LIMIT ?
    """
    df = query_dataframe(sql, [sector_name, min_turnover_lacs, days])
    return df.sort_values("trade_date").reset_index(drop=True)


def get_sector_master_performance(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    # ── Cumulative price return: end_close vs start_close, weighted by end turnover ──
    # start_close = closest available trading day on or before period start date
    _price_sql = """
        WITH end_prices AS (
            SELECT b.symbol, b.close_price, b.turnover_lacs
            FROM daily_data b
            WHERE b.trade_date = ?
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
        ),
        start_prices AS (
            SELECT b.symbol, b.close_price
            FROM daily_data b
            INNER JOIN (
                SELECT symbol, MAX(trade_date) AS td
                FROM daily_data
                WHERE trade_date <= ?
                  AND series IN ('EQ', 'SM', 'ST')
                GROUP BY symbol
            ) t ON b.symbol = t.symbol AND b.trade_date = t.td
        )
        SELECT
            s.sector,
            SUM(
                CASE WHEN sp.close_price > 0
                THEN (ep.close_price - sp.close_price) / sp.close_price * 100
                     * ep.turnover_lacs
                END
            ) / NULLIF(SUM(CASE WHEN sp.close_price > 0 THEN ep.turnover_lacs END), 0)
                AS price_chg_pct
        FROM end_prices ep
        INNER JOIN start_prices sp ON ep.symbol = sp.symbol
        INNER JOIN sector_master s ON ep.symbol = s.symbol
        WHERE s.sector IS NOT NULL
        GROUP BY s.sector
    """

    # ── Total delivered value (₹ Cr) over the period — real money flow ──────
    _deliv_sql = """
        SELECT
            s.sector,
            SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date <= ?
        GROUP BY s.sector
    """

    def _fetch(label: str, start: date) -> pd.DataFrame:
        price_df = query_dataframe(_price_sql, [as_of_date, min_turnover_lacs, start])
        price_df.columns = ["sector", f"{label}_price_chg_pct"]
        deliv_df = query_dataframe(_deliv_sql, [min_turnover_lacs, start, as_of_date])
        deliv_df.columns = ["sector", f"{label}_deliv_cr"]
        return price_df.merge(deliv_df, on="sector", how="outer")

    # ── Layer 1: 1W/2W/1M/3M delivery value ──────────────────────────────────
    w  = _fetch("1W", as_of_date - timedelta(days=7))
    tw = _fetch("2W", as_of_date - timedelta(days=14))
    m  = _fetch("1M", as_of_date - timedelta(days=30))
    q  = _fetch("3M", as_of_date - timedelta(days=90))

    result = (w.merge(tw, on="sector", how="outer")
               .merge(m,  on="sector", how="outer")
               .merge(q,  on="sector", how="outer"))

    # ── Layer 2 & 3: Pure historical 100-trading-day baseline ────────────────────
    # Baseline EXCLUDES today — today is the signal; history is the reference.
    # OFFSET 100 from dates BEFORE today → 101st most-recent pre-today date.
    # Window: > cutoff AND < as_of_date = exactly 100 trading days, none of which is today.
    cutoff_row = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data "
        "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1 OFFSET 100",
        [as_of_date],
    )
    cutoff_100d = (
        pd.to_datetime(cutoff_row["trade_date"].iloc[0]).date()
        if not cutoff_row.empty
        else as_of_date - timedelta(days=200)
    )

    # 100D total DV — pure history, today excluded
    _baseline_sql = """
        SELECT s.sector,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date < ?
        GROUP BY s.sector
    """
    baseline_df = query_dataframe(_baseline_sql, [min_turnover_lacs, cutoff_100d, as_of_date])
    baseline_df.columns = ["sector", "100D_deliv_cr"]
    result = result.merge(baseline_df, on="sector", how="left")

    # Today's single-day delivery per sector (DV_i = Turnover_i x Delivery%)
    _today_dv_sql = """
        SELECT s.sector,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS today_dv_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date = ?
        GROUP BY s.sector
    """
    today_df = query_dataframe(_today_dv_sql, [min_turnover_lacs, as_of_date])
    today_df.columns = ["sector", "today_dv_cr"]
    result = result.merge(today_df, on="sector", how="left")

    # ── Layer 3: exact mean/stddev over the same pure historical 100D window ──────
    # Must run BEFORE dv_ratio so we can use mean_100d_dv as the denominator.
    # Using total/100 is wrong when actual trading days N ≠ 100 (weekends, holidays).
    _stats_sql = """
        SELECT sector,
               AVG(daily_dv)         AS mean_100d_dv,
               STDDEV_SAMP(daily_dv) AS std_100d_dv
        FROM (
            SELECT s.sector,
                   b.trade_date,
                   SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS daily_dv
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE s.sector IS NOT NULL
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date < ?
            GROUP BY s.sector, b.trade_date
        ) daily_stats
        GROUP BY sector
    """
    stats_df = query_dataframe(_stats_sql, [min_turnover_lacs, cutoff_100d, as_of_date])
    stats_df.columns = ["sector", "mean_100d_dv", "std_100d_dv"]
    result = result.merge(stats_df, on="sector", how="left")

    # Today's sector turnover-weighted delivery % — conviction quality check.
    # If delivery VALUE (₹) is high but delivery % fell below its 100D avg, it's a
    # speculative volume spike, not institutional accumulation. Both metrics together
    # distinguish genuine institutional conviction from "heavy trading, low holding."
    _today_pct_sql = """
        SELECT s.sector,
               SUM(b.deliv_per * b.turnover_lacs) / NULLIF(SUM(b.turnover_lacs), 0)
                   AS today_wtd_deliv_pct
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date = ?
        GROUP BY s.sector
    """
    today_pct_df = query_dataframe(_today_pct_sql, [min_turnover_lacs, as_of_date])
    today_pct_df.columns = ["sector", "today_wtd_deliv_pct"]
    result = result.merge(today_pct_df, on="sector", how="left")

    # 100D average sector turnover-weighted delivery % (pure history, today excluded)
    _avg_pct_sql = """
        SELECT sector, AVG(daily_wtd_deliv_pct) AS avg_wtd_deliv_pct_100d
        FROM (
            SELECT s.sector, b.trade_date,
                   SUM(b.deliv_per * b.turnover_lacs) / NULLIF(SUM(b.turnover_lacs), 0)
                       AS daily_wtd_deliv_pct
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE s.sector IS NOT NULL
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date < ?
            GROUP BY s.sector, b.trade_date
        ) daily_pct
        GROUP BY sector
    """
    avg_pct_df = query_dataframe(_avg_pct_sql, [min_turnover_lacs, cutoff_100d, as_of_date])
    avg_pct_df.columns = ["sector", "avg_wtd_deliv_pct_100d"]
    result = result.merge(avg_pct_df, on="sector", how="left")

    # DV Ratio = today_dv / mean_100d_dv (exact daily mean, not total/100 approximation)
    result["dv_ratio"] = (
        result["today_dv_cr"] /
        result["mean_100d_dv"].replace(0, float("nan"))
    ).replace([float("inf"), -float("inf")], float("nan"))

    result["z_score"] = (
        (result["today_dv_cr"] - result["mean_100d_dv"]) /
        result["std_100d_dv"].replace(0, float("nan"))
    ).replace([float("inf"), -float("inf")], float("nan"))

    # ── Breadth: fraction of stocks where today DV_i > stock's own 100D avg daily DV ──
    # Stocks without history excluded from denominator (not penalised for having no baseline).
    _breadth_sql = """
        WITH today_dv AS (
            SELECT b.symbol,
                   b.turnover_lacs * b.deliv_per / 100.0 / 100.0 AS today_dv_cr
            FROM daily_data b
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date = ?
        ),
        hist_avg AS (
            SELECT b.symbol,
                   AVG(b.turnover_lacs * b.deliv_per / 100.0 / 100.0) AS avg_dv_cr
            FROM daily_data b
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date < ?
            GROUP BY b.symbol
        )
        SELECT s.sector,
               CAST(SUM(CASE WHEN h.avg_dv_cr IS NOT NULL AND t.today_dv_cr > h.avg_dv_cr
                             THEN 1 ELSE 0 END) AS DOUBLE)
                   / NULLIF(SUM(CASE WHEN h.avg_dv_cr IS NOT NULL THEN 1 ELSE 0 END), 0)
                   AS breadth
        FROM today_dv t
        INNER JOIN sector_master s ON t.symbol = s.symbol
        LEFT JOIN hist_avg h ON t.symbol = h.symbol
        WHERE s.sector IS NOT NULL
        GROUP BY s.sector
    """
    breadth_df = query_dataframe(
        _breadth_sql,
        [min_turnover_lacs, as_of_date, min_turnover_lacs, cutoff_100d, as_of_date],
    )
    breadth_df.columns = ["sector", "breadth"]
    result = result.merge(breadth_df, on="sector", how="left")

    result = result.sort_values("dv_ratio", ascending=False).reset_index(drop=True)
    return result


def get_subsector_master_performance(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """Same as get_sector_master_performance but grouped by sector + industry."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    _price_sql = """
        WITH end_prices AS (
            SELECT b.symbol, b.close_price, b.turnover_lacs
            FROM daily_data b
            WHERE b.trade_date = ?
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
        ),
        start_prices AS (
            SELECT b.symbol, b.close_price
            FROM daily_data b
            INNER JOIN (
                SELECT symbol, MAX(trade_date) AS td
                FROM daily_data
                WHERE trade_date <= ?
                  AND series IN ('EQ', 'SM', 'ST')
                GROUP BY symbol
            ) t ON b.symbol = t.symbol AND b.trade_date = t.td
        )
        SELECT
            s.sector,
            COALESCE(s.industry, 'Others') AS industry,
            SUM(
                CASE WHEN sp.close_price > 0
                THEN (ep.close_price - sp.close_price) / sp.close_price * 100
                     * ep.turnover_lacs
                END
            ) / NULLIF(SUM(CASE WHEN sp.close_price > 0 THEN ep.turnover_lacs END), 0)
                AS price_chg_pct
        FROM end_prices ep
        INNER JOIN start_prices sp ON ep.symbol = sp.symbol
        INNER JOIN sector_master s ON ep.symbol = s.symbol
        WHERE s.sector IS NOT NULL
        GROUP BY s.sector, COALESCE(s.industry, 'Others')
    """

    # Two separate delivery queries: one with stock_count, one without
    _deliv_sql = """
        SELECT
            s.sector,
            COALESCE(s.industry, 'Others') AS industry,
            SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr,
            COUNT(DISTINCT b.symbol) AS stock_count
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date <= ?
        GROUP BY s.sector, COALESCE(s.industry, 'Others')
    """
    _deliv_sql_no_count = """
        SELECT
            s.sector,
            COALESCE(s.industry, 'Others') AS industry,
            SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date <= ?
        GROUP BY s.sector, COALESCE(s.industry, 'Others')
    """

    def _fetch(label: str, start: date) -> pd.DataFrame:
        price_df = query_dataframe(_price_sql, [as_of_date, min_turnover_lacs, start])
        price_df.columns = ["sector", "industry", f"{label}_price_chg_pct"]
        if label == "1W":
            deliv_df = query_dataframe(_deliv_sql, [min_turnover_lacs, start, as_of_date])
            deliv_df.columns = ["sector", "industry", f"{label}_deliv_cr", "stock_count"]
        else:
            deliv_df = query_dataframe(_deliv_sql_no_count, [min_turnover_lacs, start, as_of_date])
            deliv_df.columns = ["sector", "industry", f"{label}_deliv_cr"]
        return price_df.merge(deliv_df, on=["sector", "industry"], how="outer")

    w  = _fetch("1W", as_of_date - timedelta(days=7))
    tw = _fetch("2W", as_of_date - timedelta(days=14))
    m  = _fetch("1M", as_of_date - timedelta(days=30))
    q  = _fetch("3M", as_of_date - timedelta(days=90))

    result = (w.merge(tw, on=["sector", "industry"], how="outer")
               .merge(m,  on=["sector", "industry"], how="outer")
               .merge(q,  on=["sector", "industry"], how="outer"))

    # ── Layer 2 & 3: Pure historical 100-trading-day baseline per sub-sector ───
    # Baseline EXCLUDES today — same logic as sector function.
    cutoff_row = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data "
        "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1 OFFSET 100",
        [as_of_date],
    )
    cutoff_100d = (
        pd.to_datetime(cutoff_row["trade_date"].iloc[0]).date()
        if not cutoff_row.empty
        else as_of_date - timedelta(days=200)
    )

    _baseline_no_count_sql = """
        SELECT s.sector,
               COALESCE(s.industry, 'Others') AS industry,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date < ?
        GROUP BY s.sector, COALESCE(s.industry, 'Others')
    """
    baseline_df = query_dataframe(_baseline_no_count_sql,
                                  [min_turnover_lacs, cutoff_100d, as_of_date])
    baseline_df.columns = ["sector", "industry", "100D_deliv_cr"]
    result = result.merge(baseline_df, on=["sector", "industry"], how="left")

    # Today's single-day delivery per sub-sector
    _today_dv_subsector_sql = """
        SELECT s.sector,
               COALESCE(s.industry, 'Others') AS industry,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS today_dv_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date = ?
        GROUP BY s.sector, COALESCE(s.industry, 'Others')
    """
    today_df = query_dataframe(_today_dv_subsector_sql, [min_turnover_lacs, as_of_date])
    today_df.columns = ["sector", "industry", "today_dv_cr"]
    result = result.merge(today_df, on=["sector", "industry"], how="left")

    # Layer 3: exact mean/stddev per sub-sector — must run before dv_ratio
    _stats_subsector_sql = """
        SELECT sector, industry,
               AVG(daily_dv)         AS mean_100d_dv,
               STDDEV_SAMP(daily_dv) AS std_100d_dv
        FROM (
            SELECT s.sector,
                   COALESCE(s.industry, 'Others') AS industry,
                   b.trade_date,
                   SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS daily_dv
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE s.sector IS NOT NULL
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date < ?
            GROUP BY s.sector, COALESCE(s.industry, 'Others'), b.trade_date
        ) daily_stats
        GROUP BY sector, industry
    """
    stats_df = query_dataframe(_stats_subsector_sql,
                               [min_turnover_lacs, cutoff_100d, as_of_date])
    stats_df.columns = ["sector", "industry", "mean_100d_dv", "std_100d_dv"]
    result = result.merge(stats_df, on=["sector", "industry"], how="left")

    # DV Ratio = today_dv / mean_100d_dv (exact daily mean, not total/100 approximation)
    result["dv_ratio"] = (
        result["today_dv_cr"] /
        result["mean_100d_dv"].replace(0, float("nan"))
    ).replace([float("inf"), -float("inf")], float("nan"))

    result["z_score"] = (
        (result["today_dv_cr"] - result["mean_100d_dv"]) /
        result["std_100d_dv"].replace(0, float("nan"))
    ).replace([float("inf"), -float("inf")], float("nan"))

    # ── Breadth per sub-sector ────────────────────────────────────────────────
    _breadth_subsector_sql = """
        WITH today_dv AS (
            SELECT b.symbol,
                   b.turnover_lacs * b.deliv_per / 100.0 / 100.0 AS today_dv_cr
            FROM daily_data b
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date = ?
        ),
        hist_avg AS (
            SELECT b.symbol,
                   AVG(b.turnover_lacs * b.deliv_per / 100.0 / 100.0) AS avg_dv_cr
            FROM daily_data b
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date < ?
            GROUP BY b.symbol
        )
        SELECT s.sector,
               COALESCE(s.industry, 'Others') AS industry,
               CAST(SUM(CASE WHEN h.avg_dv_cr IS NOT NULL AND t.today_dv_cr > h.avg_dv_cr
                             THEN 1 ELSE 0 END) AS DOUBLE)
                   / NULLIF(SUM(CASE WHEN h.avg_dv_cr IS NOT NULL THEN 1 ELSE 0 END), 0)
                   AS breadth
        FROM today_dv t
        INNER JOIN sector_master s ON t.symbol = s.symbol
        LEFT JOIN hist_avg h ON t.symbol = h.symbol
        WHERE s.sector IS NOT NULL
        GROUP BY s.sector, COALESCE(s.industry, 'Others')
    """
    breadth_sub_df = query_dataframe(
        _breadth_subsector_sql,
        [min_turnover_lacs, as_of_date, min_turnover_lacs, cutoff_100d, as_of_date],
    )
    breadth_sub_df.columns = ["sector", "industry", "breadth"]
    result = result.merge(breadth_sub_df, on=["sector", "industry"], how="left")

    result = result.sort_values(["sector", "dv_ratio"], ascending=[True, False])
    return result.reset_index(drop=True)


def get_all_stocks() -> pd.DataFrame:
    """All EQ/SM/ST stocks with company name — used to populate the search selectbox."""
    sql = """
        SELECT
            b.symbol,
            COALESCE(s.company_name, b.symbol) AS company_name,
            COALESCE(s.sector, 'Others') AS sector,
            COALESCE(s.industry, 'Others') AS industry,
            MAX(b.turnover_lacs) AS recent_turnover
        FROM daily_data b
        LEFT JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
        GROUP BY b.symbol, COALESCE(s.company_name, b.symbol),
                 COALESCE(s.sector, 'Others'), COALESCE(s.industry, 'Others')
        ORDER BY company_name
    """
    return query_dataframe(sql, [])


def search_stock_suggestions(query: str, limit: int = 15) -> pd.DataFrame:
    """Fast symbol/name lookup — returns symbol + company for dropdown suggestions."""
    pattern = f"%{query.upper()}%"
    sql = """
        SELECT
            b.symbol,
            COALESCE(s.company_name, b.symbol) AS company_name,
            COALESCE(s.sector, 'Others') AS sector,
            COALESCE(s.industry, 'Others') AS industry,
            MAX(b.turnover_lacs) AS recent_turnover
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND (UPPER(b.symbol) LIKE ? OR UPPER(COALESCE(s.company_name, b.symbol)) LIKE ?)
        GROUP BY b.symbol, COALESCE(s.company_name, b.symbol),
                 COALESCE(s.sector, 'Others'), COALESCE(s.industry, 'Others')
        ORDER BY recent_turnover DESC
        LIMIT ?
    """
    return query_dataframe(sql, [pattern, pattern, limit])


def search_stocks_performance(
    as_of_date: date,
    query: str,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """Multi-period stock performance for all stocks matching symbol or company name."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    pattern = f"%{query.upper()}%"

    today_sql = """
        WITH base AS (
            SELECT b.symbol,
                   b.trade_date,
                   COALESCE(s.company_name, b.symbol) AS company_name,
                   COALESCE(s.sector, 'Others') AS sector,
                   COALESCE(s.industry, 'Others') AS industry,
                   COALESCE(s.category, '') AS category,
                   b.close_price, b.turnover_lacs, b.deliv_per,
                   b.ttl_trd_qnty,
                   CASE WHEN b.prev_close > 0
                   THEN (b.close_price - b.prev_close) / b.prev_close * 100
                   ELSE NULL END AS price_change_pct,
                   AVG(b.deliv_per) OVER (
                       PARTITION BY b.symbol, b.series
                       ORDER BY b.trade_date
                       ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                   ) AS deliv_per_10d_avg,
                   AVG(b.ttl_trd_qnty) OVER (
                       PARTITION BY b.symbol, b.series
                       ORDER BY b.trade_date
                       ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                   ) AS vol_20d_avg
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND (UPPER(b.symbol) LIKE ? OR UPPER(COALESCE(s.company_name, b.symbol)) LIKE ?)
        )
        SELECT symbol, company_name, sector, industry, category,
               close_price, price_change_pct, turnover_lacs, deliv_per,
               (deliv_per / NULLIF(deliv_per_10d_avg, 0)) AS deliv_ratio,
               (ttl_trd_qnty / NULLIF(vol_20d_avg, 0))    AS vol_ratio
        FROM base
        WHERE trade_date = ?
        ORDER BY turnover_lacs DESC
    """
    df = query_dataframe(today_sql, [pattern, pattern, as_of_date])
    if df.empty:
        return df

    symbols = df["symbol"].tolist()
    ph = ",".join("?" * len(symbols))

    price_hist_sql = f"""
        SELECT b.symbol, b.close_price AS start_price
        FROM daily_data b
        INNER JOIN (
            SELECT symbol, MAX(trade_date) AS td
            FROM daily_data
            WHERE trade_date <= ?
              AND series IN ('EQ', 'SM', 'ST')
              AND symbol IN ({ph})
            GROUP BY symbol
        ) t ON b.symbol = t.symbol AND b.trade_date = t.td
        WHERE b.symbol IN ({ph})
    """

    deliv_hist_sql = f"""
        SELECT b.symbol,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
        FROM daily_data b
        WHERE b.trade_date > ?
          AND b.trade_date <= ?
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.symbol IN ({ph})
        GROUP BY b.symbol
    """

    for label, cal_days in [("1W", 7), ("2W", 14), ("1M", 30), ("3M", 90)]:
        start = as_of_date - timedelta(days=cal_days)

        price_df = query_dataframe(price_hist_sql, [start] + symbols + symbols)
        price_df.columns = ["symbol", "start_price"]
        df = df.merge(price_df, on="symbol", how="left")
        df[f"{label}_price_chg_pct"] = (
            (df["close_price"] - df["start_price"])
            / df["start_price"].replace(0, float("nan")) * 100
        )
        df.drop(columns=["start_price"], inplace=True)

        deliv_df = query_dataframe(deliv_hist_sql, [start, as_of_date] + symbols)
        deliv_df.columns = ["symbol", f"{label}_deliv_cr"]
        df = df.merge(deliv_df, on="symbol", how="left")

    return df.reset_index(drop=True)


def get_subsector_stocks_performance(
    as_of_date: date,
    sector_name: str,
    industry_name: str,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """Stock-level cumulative price returns + delivery for a given sub-sector."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()

    today_sql = """
        WITH base AS (
            SELECT b.symbol,
                   COALESCE(s.company_name, b.symbol) AS company_name,
                   COALESCE(s.category, '') AS category,
                   b.trade_date,
                   b.close_price, b.turnover_lacs, b.deliv_per,
                   b.ttl_trd_qnty,
                   AVG(b.deliv_per) OVER (
                       PARTITION BY b.symbol, b.series
                       ORDER BY b.trade_date
                       ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                   ) AS deliv_per_10d_avg,
                   AVG(b.ttl_trd_qnty) OVER (
                       PARTITION BY b.symbol, b.series
                       ORDER BY b.trade_date
                       ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                   ) AS vol_20d_avg
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE b.series IN ('EQ', 'SM', 'ST')
              AND s.sector = ?
              AND COALESCE(s.industry, 'Others') = ?
        )
        SELECT symbol, company_name, category, close_price, turnover_lacs, deliv_per,
               (deliv_per / NULLIF(deliv_per_10d_avg, 0)) AS deliv_ratio,
               (ttl_trd_qnty / NULLIF(vol_20d_avg, 0))    AS vol_ratio
        FROM base
        WHERE trade_date = ?
          AND turnover_lacs >= ?
        ORDER BY turnover_lacs DESC
    """
    df = query_dataframe(today_sql, [sector_name, industry_name, as_of_date, min_turnover_lacs])
    if df.empty:
        return df

    # ── Cumulative price return per period ───────────────────────────────────
    price_hist_sql = """
        SELECT b.symbol, b.close_price AS start_price
        FROM daily_data b
        INNER JOIN (
            SELECT symbol, MAX(trade_date) AS td
            FROM daily_data
            WHERE trade_date <= ?
              AND series IN ('EQ', 'SM', 'ST')
            GROUP BY symbol
        ) t ON b.symbol = t.symbol AND b.trade_date = t.td
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector = ?
          AND COALESCE(s.industry, 'Others') = ?
    """

    # ── Total delivered value (₹ Cr) over period ─────────────────────────────
    deliv_hist_sql = """
        SELECT b.symbol,
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.trade_date > ?
          AND b.trade_date <= ?
          AND b.series IN ('EQ', 'SM', 'ST')
          AND s.sector = ?
          AND COALESCE(s.industry, 'Others') = ?
        GROUP BY b.symbol
    """

    for label, cal_days in [("1W", 7), ("2W", 14), ("1M", 30), ("3M", 90)]:
        start = as_of_date - timedelta(days=cal_days)

        # Price change
        ph = query_dataframe(price_hist_sql, [start, sector_name, industry_name])
        ph.columns = ["symbol", "start_price"]
        df = df.merge(ph, on="symbol", how="left")
        df[f"{label}_price_chg_pct"] = (
            (df["close_price"] - df["start_price"])
            / df["start_price"].replace(0, float("nan")) * 100
        )
        df.drop(columns=["start_price"], inplace=True)

        # Delivered value over period
        dh = query_dataframe(deliv_hist_sql, [start, as_of_date, sector_name, industry_name])
        dh.columns = ["symbol", f"{label}_deliv_cr"]
        df = df.merge(dh, on="symbol", how="left")

    return df.reset_index(drop=True)
