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

    # ── Average delivery % over the period ───────────────────────────────────
    _deliv_sql = """
        SELECT
            s.sector,
            SUM(b.deliv_per * b.turnover_lacs) / NULLIF(SUM(b.turnover_lacs), 0)
                AS deliv_pct
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date <= ?
        GROUP BY s.sector
    """

    def _fetch(label: str, calendar_days: int) -> pd.DataFrame:
        start = as_of_date - timedelta(days=calendar_days)
        price_df = query_dataframe(_price_sql, [as_of_date, min_turnover_lacs, start])
        price_df.columns = ["sector", f"{label}_price_chg_pct"]

        deliv_df = query_dataframe(_deliv_sql, [min_turnover_lacs, start, as_of_date])
        deliv_df.columns = ["sector", f"{label}_deliv_pct"]

        return price_df.merge(deliv_df, on="sector", how="outer")

    w  = _fetch("1W", 7)
    tw = _fetch("2W", 14)
    m  = _fetch("1M", 30)
    q  = _fetch("3M", 90)

    result = (w.merge(tw, on="sector", how="outer")
               .merge(m,  on="sector", how="outer")
               .merge(q,  on="sector", how="outer"))
    result = result.sort_values("3M_deliv_pct", ascending=False).reset_index(drop=True)
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

    _deliv_sql = """
        SELECT
            s.sector,
            COALESCE(s.industry, 'Others') AS industry,
            SUM(b.deliv_per * b.turnover_lacs) / NULLIF(SUM(b.turnover_lacs), 0)
                AS deliv_pct,
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

    def _fetch(label: str, calendar_days: int) -> pd.DataFrame:
        start = as_of_date - timedelta(days=calendar_days)
        price_df = query_dataframe(_price_sql, [as_of_date, min_turnover_lacs, start])
        price_df.columns = ["sector", "industry", f"{label}_price_chg_pct"]
        deliv_df = query_dataframe(_deliv_sql, [min_turnover_lacs, start, as_of_date])
        deliv_df.columns = ["sector", "industry", f"{label}_deliv_pct", "stock_count"]
        # keep stock_count only from the shortest period (most current)
        if label == "1W":
            merged = price_df.merge(deliv_df, on=["sector", "industry"], how="outer")
        else:
            merged = price_df.merge(
                deliv_df.drop(columns=["stock_count"]), on=["sector", "industry"], how="outer"
            )
        return merged

    w  = _fetch("1W", 7)
    tw = _fetch("2W", 14)
    m  = _fetch("1M", 30)
    q  = _fetch("3M", 90)

    result = (w.merge(tw, on=["sector", "industry"], how="outer")
               .merge(m,  on=["sector", "industry"], how="outer")
               .merge(q,  on=["sector", "industry"], how="outer"))
    result = result.sort_values(["sector", "3M_deliv_pct"], ascending=[True, False])
    return result.reset_index(drop=True)


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
        SELECT symbol, company_name, close_price, turnover_lacs, deliv_per,
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

    # ── Average delivery % over period ────────────────────────────────────────
    deliv_hist_sql = """
        SELECT b.symbol,
               AVG(b.deliv_per) AS deliv_pct
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

        # Avg delivery over period
        dh = query_dataframe(deliv_hist_sql, [start, as_of_date, sector_name, industry_name])
        dh.columns = ["symbol", f"{label}_deliv_pct"]
        df = df.merge(dh, on="symbol", how="left")

    return df.reset_index(drop=True)
