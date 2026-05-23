"""
NiftyIndices daily snapshot fetcher.

Source: https://www.niftyindices.com/Daily_Snapshot/ind_close_all_DDMMYYYY.csv
One file per trading day — 147 indices, 13 columns.

No cookie priming needed for niftyindices.com (unlike nseindia.com).
"""
from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd
import requests

from src.core.logging import get_logger

__all__ = ["IndexFetcher"]

log = get_logger(__name__)

_BASE_URL = "https://www.niftyindices.com/Daily_Snapshot/ind_close_all_{date}.csv"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.niftyindices.com/reports/daily-reports",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_COL_MAP = {
    "Index Name":          "index_name",
    "Open Index Value":    "open_val",
    "High Index Value":    "high_val",
    "Low Index Value":     "low_val",
    "Closing Index Value": "close_val",
    "Points Change":       "points_chg",
    "Change(%)":           "pct_chg",
    "Volume":              "volume",
    "Turnover (Rs. Cr.)":  "turnover_cr",
    "P/E":                 "pe_ratio",
    "P/B":                 "pb_ratio",
    "Div Yield":           "div_yield",
}

# Indices to EXCLUDE (bond indices, derivatives indices, shariah, leverage products)
_EXCLUDE_PATTERNS = [
    "G-Sec", "BHARAT Bond", "1D Rate", "Shariah", "1x Inverse",
    "2x Leverage", "1x Inverse", "PR 1x", "PR 2x", "TR 1x", "TR 2x",
    "USD", "Dividend Points", "Arbitrage", "Futures Index", "Futures TR",
]
_EXCLUDE_RE = re.compile("|".join(re.escape(p) for p in _EXCLUDE_PATTERNS))


class IndexFetcher:
    """Fetch and parse the NiftyIndices daily snapshot CSV for one date."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    def fetch(self, trade_date: date) -> pd.DataFrame:
        """
        Return a cleaned DataFrame ready for upsert_index_data().
        Returns empty DataFrame if no data for this date (holiday/weekend).
        """
        date_str = trade_date.strftime("%d%m%Y")
        url = _BASE_URL.format(date=date_str)

        try:
            resp = self._session.get(url, timeout=20)
            if resp.status_code == 404:
                log.debug("No index data for %s (404)", trade_date)
                return pd.DataFrame()
            resp.raise_for_status()
            if len(resp.content) < 200:
                return pd.DataFrame()
        except Exception as exc:
            log.warning("IndexFetcher: failed to fetch %s: %s", trade_date, exc)
            return pd.DataFrame()

        return self._parse(resp.text, trade_date)

    def _parse(self, csv_text: str, trade_date: date) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as exc:
            log.warning("IndexFetcher: CSV parse error for %s: %s", trade_date, exc)
            return pd.DataFrame()

        # Rename columns
        df = df.rename(columns=_COL_MAP)
        if "index_name" not in df.columns:
            return pd.DataFrame()

        # Drop excluded index types (bonds, derivatives products)
        mask = ~df["index_name"].str.contains(_EXCLUDE_RE, na=False)
        df = df[mask].copy()

        # Add trade_date
        df["trade_date"] = trade_date

        # Add prev_close: close - points_chg
        if "close_val" in df.columns and "points_chg" in df.columns:
            df["prev_close"] = df["close_val"] - df["points_chg"]
        else:
            df["prev_close"] = None

        # Coerce numeric columns
        num_cols = ["open_val", "high_val", "low_val", "close_val", "prev_close",
                    "points_chg", "pct_chg", "turnover_cr", "pe_ratio", "pb_ratio",
                    "div_yield"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("Int64")

        # Select only columns the table expects
        keep = ["trade_date", "index_name", "open_val", "high_val", "low_val",
                "close_val", "prev_close", "points_chg", "pct_chg", "volume",
                "turnover_cr", "pe_ratio", "pb_ratio", "div_yield"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.dropna(subset=["index_name", "close_val"])

        log.info("IndexFetcher: %d indices parsed for %s", len(df), trade_date)
        return df
