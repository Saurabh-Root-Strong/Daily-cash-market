"""
BhavCopyFetcher — downloads the UDiFF bhavcopy zip and transforms it to schema.

UDiFF format (post July 2024):
  URL: BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
  Columns: abbreviated (TckrSymb, TtlTrfVal, etc.)
  TtlTrfVal is in rupees — divide by 100 000 for lakhs.
  Delivery data is NOT included — fetched separately via DeliveryFetcher.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd

from src.core.exceptions import ParseError
from src.core.logging import get_logger
from src.ingestion.base import BaseFetcher

__all__ = ["BhavCopyFetcher", "fetch_bhavcopy", "transform_to_schema"]

log = get_logger(__name__)

_EQUITY_SERIES = frozenset({"EQ", "SM", "ST"})

_COL_MAP: dict[str, str] = {
    "TradDt":          "trade_date",
    "TckrSymb":        "symbol",
    "SctySrs":         "series",
    "PrvsClsgPric":    "prev_close",
    "OpnPric":         "open_price",
    "HghPric":         "high_price",
    "LwPric":          "low_price",
    "LastPric":        "last_price",
    "ClsPric":         "close_price",
    "TtlTradgVol":     "ttl_trd_qnty",
    "TtlTrfVal":       "turnover_lacs",   # renamed; divide by 100 000 below
    "TtlNbOfTxsExctd": "no_of_trades",
}

_SCHEMA_COLS = [
    "trade_date", "symbol", "series",
    "prev_close", "open_price", "high_price", "low_price",
    "last_price", "close_price", "avg_price",
    "ttl_trd_qnty", "turnover_lacs", "no_of_trades",
    "deliv_qty", "deliv_per",
]


class BhavCopyFetcher(BaseFetcher):
    """Downloads and parses the NSE CM bhavcopy for a given trade date."""

    @property
    def name(self) -> str:
        return "BhavCopy"

    def build_url(self, trade_date: date) -> str:
        from src.core.config import get_config
        return get_config().ingestion.bhavcopy_url.format(
            date=trade_date.strftime("%Y%m%d")
        )

    def _url_candidates(self, trade_date: date) -> list:
        """Ordered fallback chain for bhavcopy ZIP."""
        d = trade_date.strftime("%Y%m%d")
        dm = trade_date.strftime("%d%m%Y")
        return [
            self.build_url(trade_date),    # primary: nsearchives (from config)
            # Fallback 1: archives domain alias
            f"https://archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d}_F_0000.csv.zip",
            # Fallback 2: older UDiFF format variant
            f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d}_F_0000.zip",
            # Fallback 3: very old bhavcopy format (pre-UDiFF, still on NSE archives)
            f"https://archives.nseindia.com/content/historical/EQUITIES/"
            f"{trade_date.year}/{trade_date.strftime('%b').upper()}/cm{dm}bhav.csv.zip",
        ]

    def fetch(self, trade_date: date) -> pd.DataFrame:
        for url in self._url_candidates(trade_date):
            log.info("Fetching bhavcopy: %s", url)
            raw_bytes = self._client.get_bytes(url, expect_404_ok=True)
            if raw_bytes is None:
                continue
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw_bytes))
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                raw_df = pd.read_csv(zf.open(csv_name))
                return transform_to_schema(raw_df, trade_date)
            except StopIteration:
                raise ParseError(f"No CSV inside bhavcopy zip for {trade_date}")
            except Exception as exc:
                log.warning("Bhavcopy parse failed via %s: %s", url[-60:], exc)
                continue

        log.info("Bhavcopy not available for %s (all sources empty — holiday/weekend)", trade_date)
        return pd.DataFrame()


def transform_to_schema(raw_df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Filter, rename, and cast raw bhavcopy CSV to the daily_data schema."""
    df = raw_df.copy()

    # Keep only CM segment equity series
    if "Sgmt" in df.columns:
        df = df[df["Sgmt"] == "CM"]
    if "SctySrs" in df.columns:
        df = df[df["SctySrs"].isin(_EQUITY_SERIES)]

    df = df.rename(columns=_COL_MAP)

    # Rupees → lakhs
    df["turnover_lacs"] = pd.to_numeric(df["turnover_lacs"], errors="coerce") / 100_000

    # avg_price = turnover_rupees / volume
    turnover_rupees = df["turnover_lacs"] * 100_000
    volume = pd.to_numeric(df["ttl_trd_qnty"], errors="coerce")
    df["avg_price"] = turnover_rupees / volume.replace(0, pd.NA)

    df["trade_date"] = trade_date
    df["deliv_qty"] = None
    df["deliv_per"] = None

    for col in ("prev_close", "open_price", "high_price", "low_price",
                "last_price", "close_price", "turnover_lacs", "avg_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("ttl_trd_qnty", "no_of_trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    existing = [c for c in _SCHEMA_COLS if c in df.columns]
    return df[existing].reset_index(drop=True)


# ── Backward-compatible module-level function ─────────────────────────────────

def fetch_bhavcopy(trade_date: date, client) -> pd.DataFrame | None:
    """Legacy call-site wrapper — prefer BhavCopyFetcher(client).fetch(date)."""
    result = BhavCopyFetcher(client).fetch(trade_date)
    return None if result.empty else result
