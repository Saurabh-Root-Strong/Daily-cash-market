from datetime import date
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
    sql = """
        SELECT
            b.trade_date,
            COALESCE(s.sector, 'Others') AS sector,
            AVG(CASE WHEN b.prev_close > 0 THEN (b.close_price - b.prev_close)/b.prev_close*100 END) AS avg_price_change_pct,
            AVG(b.deliv_per) AS avg_deliv_per,
            SUM(b.turnover_lacs) AS total_turnover_lacs,
            COUNT(*) AS stock_count
        FROM daily_data b
        LEFT JOIN sector_master s ON b.symbol = s.symbol
        WHERE COALESCE(s.sector, 'Others') = ?
        GROUP BY b.trade_date, COALESCE(s.sector, 'Others')
        ORDER BY b.trade_date DESC
        LIMIT ?
    """
    df = query_dataframe(sql, [sector_name, days])
    return df.sort_values("trade_date").reset_index(drop=True)
