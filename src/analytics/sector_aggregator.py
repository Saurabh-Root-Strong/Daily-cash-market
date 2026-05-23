from datetime import date, timedelta
from typing import Optional, Dict
import pandas as pd

from src.analytics.delivery_signals import get_stock_metrics
from src.analytics.base import get_weighting_method, get_min_turnover_filter, get_thresholds
from src.data.repository import query_dataframe
from src.logging_setup import get_logger

log = get_logger(__name__)


# ── Shared helpers: sector vs subsector grouping ──────────────────────────────

def _g_sel(by_industry: bool) -> str:
    """Extra SELECT column when grouping by industry."""
    return ", COALESCE(s.industry, 'Others') AS industry" if by_industry else ""


def _g_by(by_industry: bool) -> str:
    """Extra GROUP BY expression when grouping by industry."""
    return ", COALESCE(s.industry, 'Others')" if by_industry else ""


def _merge_on(by_industry: bool):
    return ["sector", "industry"] if by_industry else "sector"


def _get_cutoff_100d(as_of_date: date) -> date:
    """Oldest date bounding a strict 100-trading-day window ending before as_of_date."""
    row = query_dataframe(
        "SELECT DISTINCT trade_date FROM daily_data "
        "WHERE trade_date < ? ORDER BY trade_date DESC LIMIT 1 OFFSET 100",
        [as_of_date],
    )
    return (
        pd.to_datetime(row["trade_date"].iloc[0]).date()
        if not row.empty
        else as_of_date - timedelta(days=200)
    )


# ── Public aggregation functions ──────────────────────────────────────────────

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
    acc_threshold, dist_threshold = get_thresholds()

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
            "sector":                  sector,
            "stock_count":             stock_count,
            "simple_price_change_pct": simple_price,
            "simple_deliv_per":        simple_deliv,
            "wtd_price_change_pct":    wtd_price,
            "wtd_deliv_per":           wtd_deliv,
            "top_delivery_symbol":     top_deliv_symbol,
            "accumulation_count":      acc_count,
            "distribution_count":      dist_count,
            "total_turnover_lacs":     total_turnover,
            "total_deliv_value_lacs":  total_deliv_value,
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

    sector_df["turnover_share_pct"] = (
        (sector_df["turnover_lacs"] / total_turnover * 100).round(2)
        if total_turnover > 0 else 0.0
    )
    sector_df["deliv_value_share_pct"] = (
        (sector_df["deliv_value_lacs"] / total_deliv_value * 100).round(2)
        if total_deliv_value > 0 else 0.0
    )

    top_by_delivery_pct   = sector_df.nlargest(top_n, "deliv_per")
    top_by_delivery_value = sector_df.nlargest(top_n, "deliv_value_lacs")
    top_by_turnover       = sector_df.nlargest(top_n, "turnover_lacs")
    contribution_table    = sector_df.nlargest(top_n, "turnover_lacs")

    def _agg_subsector(g: pd.DataFrame) -> pd.Series:
        total_to = g["turnover_lacs"].sum()
        wtd_deliv = (
            (g["deliv_per"] * g["turnover_lacs"]).sum() / total_to
            if total_to > 0 else g["deliv_per"].mean()
        )
        return pd.Series({
            "wtd_deliv_per":          wtd_deliv,
            "simple_deliv_per":       g["deliv_per"].mean(),
            "avg_price_chg":          g["price_change_pct"].mean(),
            "stock_count":            len(g),
            "total_turnover_lacs":    total_to,
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
        "stock_count":           len(sector_df),
        "total_turnover_lacs":   total_turnover,
        "total_deliv_value_lacs": total_deliv_value,
        "avg_price_change_pct":  sector_df["price_change_pct"].mean(),
        "avg_deliv_per":         sector_df["deliv_per"].mean(),
    }

    return {
        "top_by_delivery_pct":   top_by_delivery_pct,
        "top_by_delivery_value": top_by_delivery_value,
        "top_by_turnover":       top_by_turnover,
        "contribution_table":    contribution_table,
        "subsector_summary":     subsector_summary,
        "sector_summary":        sector_summary,
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


# ── Master performance — unified sector + subsector implementation ─────────────

def _build_master_performance(
    as_of_date: date,
    min_turnover_lacs: float,
    by_industry: bool,
) -> pd.DataFrame:
    """
    Shared core for get_sector_master_performance (by_industry=False) and
    get_subsector_master_performance (by_industry=True).  The only structural
    difference between the two callers is the GROUP BY dimension.

    f-string interpolation here is safe: gs/gb/outer_g are hard-coded constant
    strings derived from a boolean — never from user input.
    """
    gs       = _g_sel(by_industry)
    gb       = _g_by(by_industry)
    mo       = _merge_on(by_industry)
    grp_cols = ["sector"] + (["industry"] if by_industry else [])
    outer_g  = ", industry" if by_industry else ""

    # ── Period price + delivery ───────────────────────────────────────────────
    price_sql = f"""
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
        SELECT s.sector{gs},
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
        GROUP BY s.sector{gb}
    """

    def _fetch(label: str, start: date) -> pd.DataFrame:
        include_count = by_industry and label == "1W"
        count_col = ", COUNT(DISTINCT b.symbol) AS stock_count" if include_count else ""
        deliv_sql = f"""
            SELECT s.sector{gs},
                SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
                {count_col}
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE s.sector IS NOT NULL
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date <= ?
            GROUP BY s.sector{gb}
        """
        p_df = query_dataframe(price_sql, [as_of_date, min_turnover_lacs, start])
        p_df.columns = grp_cols + [f"{label}_price_chg_pct"]

        d_df = query_dataframe(deliv_sql, [min_turnover_lacs, start, as_of_date])
        d_df.columns = grp_cols + [f"{label}_deliv_cr"] + (["stock_count"] if include_count else [])

        return p_df.merge(d_df, on=mo, how="outer")

    w  = _fetch("1W", as_of_date - timedelta(days=7))
    tw = _fetch("2W", as_of_date - timedelta(days=14))
    m  = _fetch("1M", as_of_date - timedelta(days=30))
    q  = _fetch("3M", as_of_date - timedelta(days=90))
    result = (w.merge(tw, on=mo, how="outer")
               .merge(m,  on=mo, how="outer")
               .merge(q,  on=mo, how="outer"))

    # ── 100-trading-day baseline (today excluded) ─────────────────────────────
    cutoff_100d = _get_cutoff_100d(as_of_date)

    baseline_sql = f"""
        SELECT s.sector{gs},
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS deliv_val_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date > ?
          AND b.trade_date < ?
        GROUP BY s.sector{gb}
    """
    bdf = query_dataframe(baseline_sql, [min_turnover_lacs, cutoff_100d, as_of_date])
    bdf.columns = grp_cols + ["100D_deliv_cr"]
    result = result.merge(bdf, on=mo, how="left")

    today_dv_sql = f"""
        SELECT s.sector{gs},
               SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS today_dv_cr
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE s.sector IS NOT NULL
          AND b.series IN ('EQ', 'SM', 'ST')
          AND b.turnover_lacs >= ?
          AND b.trade_date = ?
        GROUP BY s.sector{gb}
    """
    tdf = query_dataframe(today_dv_sql, [min_turnover_lacs, as_of_date])
    tdf.columns = grp_cols + ["today_dv_cr"]
    result = result.merge(tdf, on=mo, how="left")

    stats_sql = f"""
        SELECT sector{outer_g},
               AVG(daily_dv)         AS mean_100d_dv,
               STDDEV_SAMP(daily_dv) AS std_100d_dv
        FROM (
            SELECT s.sector{gs},
                   b.trade_date,
                   SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0 AS daily_dv
            FROM daily_data b
            INNER JOIN sector_master s ON b.symbol = s.symbol
            WHERE s.sector IS NOT NULL
              AND b.series IN ('EQ', 'SM', 'ST')
              AND b.turnover_lacs >= ?
              AND b.trade_date > ?
              AND b.trade_date < ?
            GROUP BY s.sector{gb}, b.trade_date
        ) daily_stats
        GROUP BY sector{outer_g}
    """
    sdf = query_dataframe(stats_sql, [min_turnover_lacs, cutoff_100d, as_of_date])
    sdf.columns = grp_cols + ["mean_100d_dv", "std_100d_dv"]
    result = result.merge(sdf, on=mo, how="left")

    # ── Sector-level only: today wtd delivery % + 100D average ───────────────
    # Delivery VALUE (₹) and delivery % tell different stories: VALUE can be high
    # while % falls (volume spike, not conviction). Both together distinguish
    # genuine institutional accumulation from speculative noise.
    if not by_industry:
        today_pct_sql = """
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
        tpdf = query_dataframe(today_pct_sql, [min_turnover_lacs, as_of_date])
        tpdf.columns = ["sector", "today_wtd_deliv_pct"]
        result = result.merge(tpdf, on="sector", how="left")

        avg_pct_sql = """
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
        apdf = query_dataframe(avg_pct_sql, [min_turnover_lacs, cutoff_100d, as_of_date])
        apdf.columns = ["sector", "avg_wtd_deliv_pct_100d"]
        result = result.merge(apdf, on="sector", how="left")

    # ── Derived metrics ───────────────────────────────────────────────────────
    _nan = float("nan")
    _inf = [float("inf"), -float("inf")]

    result["dv_ratio"] = (
        result["today_dv_cr"] / result["mean_100d_dv"].replace(0, _nan)
    ).replace(_inf, _nan)

    result["z_score"] = (
        (result["today_dv_cr"] - result["mean_100d_dv"])
        / result["std_100d_dv"].replace(0, _nan)
    ).replace(_inf, _nan)

    # ── Breadth: fraction of stocks where today DV > own 100D average ─────────
    breadth_sql = f"""
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
        SELECT s.sector{gs},
               CAST(SUM(CASE WHEN h.avg_dv_cr IS NOT NULL AND t.today_dv_cr > h.avg_dv_cr
                             THEN 1 ELSE 0 END) AS DOUBLE)
                   / NULLIF(SUM(CASE WHEN h.avg_dv_cr IS NOT NULL THEN 1 ELSE 0 END), 0)
                   AS breadth
        FROM today_dv t
        INNER JOIN sector_master s ON t.symbol = s.symbol
        LEFT JOIN hist_avg h ON t.symbol = h.symbol
        WHERE s.sector IS NOT NULL
        GROUP BY s.sector{gb}
    """
    brdf = query_dataframe(
        breadth_sql,
        [min_turnover_lacs, as_of_date, min_turnover_lacs, cutoff_100d, as_of_date],
    )
    brdf.columns = grp_cols + ["breadth"]
    result = result.merge(brdf, on=mo, how="left")

    if by_industry:
        result = result.sort_values(["sector", "dv_ratio"], ascending=[True, False])
    else:
        result = result.sort_values("dv_ratio", ascending=False)

    return result.reset_index(drop=True)


def get_sector_master_performance(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()
    return _build_master_performance(as_of_date, min_turnover_lacs, by_industry=False)


def get_subsector_master_performance(
    as_of_date: date,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """Same as get_sector_master_performance but grouped by sector + industry."""
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()
    return _build_master_performance(as_of_date, min_turnover_lacs, by_industry=True)


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


def get_stock_close_prices(symbols: tuple, trade_date: date) -> dict:
    """Return {symbol: close_price} for a list of symbols on the given date."""
    if not symbols:
        return {}
    ph = ", ".join("?" * len(symbols))
    df = query_dataframe(
        f"SELECT symbol, close_price FROM daily_data WHERE trade_date = ? AND symbol IN ({ph})",
        [trade_date] + list(symbols),
    )
    return {} if df.empty else dict(zip(df["symbol"], df["close_price"]))


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

        ph = query_dataframe(price_hist_sql, [start, sector_name, industry_name])
        ph.columns = ["symbol", "start_price"]
        df = df.merge(ph, on="symbol", how="left")
        df[f"{label}_price_chg_pct"] = (
            (df["close_price"] - df["start_price"])
            / df["start_price"].replace(0, float("nan")) * 100
        )
        df.drop(columns=["start_price"], inplace=True)

        dh = query_dataframe(deliv_hist_sql, [start, as_of_date, sector_name, industry_name])
        dh.columns = ["symbol", f"{label}_deliv_cr"]
        df = df.merge(dh, on="symbol", how="left")

    return df.reset_index(drop=True)
