"""
FIIStatsFetcher — FII Derivatives Statistics report from NSE Archives.

NSE serves this as an older Excel (.xls / OLE2) binary, not CSV.
Report name in Archives API: "F&O - FII Derivatives Statistics"

Per-index breakdown — shows WHERE FII is buying/selling:
  NIFTY FUTURES, BANKNIFTY FUTURES, FINNIFTY FUTURES, MIDCPNIFTY FUTURES,
  NIFTY OPTIONS, BANKNIFTY OPTIONS, FINNIFTY OPTIONS, MIDCPNIFTY OPTIONS,
  STOCK FUTURES, STOCK OPTIONS

7 data columns per row:
  Buy No. of Contracts | Buy Amt (Cr) |
  Sell No. of Contracts | Sell Amt (Cr) |
  OI No. of Contracts | OI Amt (Cr)

net_value_cr = buy_value_cr - sell_value_cr
  Positive → FII net buyer (money flowing IN)
  Negative → FII net seller (money flowing OUT)
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from src.core.exceptions import ParseError
from src.core.logging import get_logger
from src.ingestion.base import BaseFetcher
from src.ingestion.fao_fetcher import (
    _build_archive_url,
    _DERIVATIVES_PAGE,
)

__all__ = ["FIIStatsFetcher"]

log = get_logger(__name__)

_FII_STATS_NAME = "F&O - FII Derivatives Statistics"

# All valid category names from NSE FII stats report (UPPERCASE as stored in DB)
_VALID_CATEGORIES = {
    "INDEX FUTURES",
    "NIFTY FUTURES",
    "BANKNIFTY FUTURES",
    "FINNIFTY FUTURES",
    "MIDCPNIFTY FUTURES",
    "NIFTYNXT50 FUTURES",
    "INDEX OPTIONS",
    "NIFTY OPTIONS",
    "BANKNIFTY OPTIONS",
    "FINNIFTY OPTIONS",
    "MIDCPNIFTY OPTIONS",
    "NIFTYNXT50 OPTIONS",
    "STOCK FUTURES",
    "STOCK OPTIONS",
}

_NUMERIC_COLS = [
    "buy_contracts", "buy_value_cr",
    "sell_contracts", "sell_value_cr",
    "oi_contracts", "oi_value_cr",
]

# OLE2 (XLS) magic bytes
_XLS_MAGIC  = b"\xd0\xcf\x11\xe0"
# XLSX (ZIP) magic bytes
_XLSX_MAGIC = b"PK\x03\x04"


class FIIStatsFetcher(BaseFetcher):
    """Downloads and parses the FII Derivatives Statistics XLS for one date."""

    def __init__(self, client) -> None:
        super().__init__(client)
        self._derivatives_primed = False

    @property
    def name(self) -> str:
        return "FII Derivatives Statistics"

    def _prime(self) -> None:
        """Hit the derivatives page once per fetcher instance to prime cookies."""
        if self._derivatives_primed:
            return
        try:
            self._client.get(_DERIVATIVES_PAGE, expect_404_ok=True)
            log.debug("Primed NSE derivatives session for FII stats")
            self._derivatives_primed = True
        except Exception as exc:
            log.debug("FII stats prime failed (non-fatal): %s", exc)

    def fetch(self, trade_date: date) -> pd.DataFrame:
        """Returns one row per index category. Empty DataFrame when no data."""
        self._prime()

        date_str = trade_date.strftime("%d-%m-%Y")
        url  = _build_archive_url(_FII_STATS_NAME, date_str)
        data = self._client.get_bytes(url, expect_404_ok=True)

        if not data:
            log.debug("FII Stats not available for %s (holiday/weekend/404)", trade_date)
            return pd.DataFrame()

        if not (data[:4] == _XLS_MAGIC or data[:4] == _XLSX_MAGIC):
            log.debug("FII Stats for %s: non-Excel response (len=%d)", trade_date, len(data))
            return pd.DataFrame()

        try:
            return _parse_fii_stats_xls(data, trade_date)
        except ParseError as exc:
            log.warning("FII Stats parse error for %s: %s", trade_date, exc)
            return pd.DataFrame()


def _parse_fii_stats_xls(data: bytes, trade_date: date) -> pd.DataFrame:
    """
    Parse NSE FII Derivatives Statistics XLS.

    XLS layout:
      Row 0:  Title ("FII DERIVATIVES STATISTICS FOR DD-Mon-YYYY")
      Row 1:  Group headers: [nan, BUY, BUY, SELL, SELL, OPEN INTEREST..., nan]
      Row 2:  Sub-headers: [nan, No. of contracts, Amt in Crores, ...]
      Row 3+: Data rows (category in col 0, 6 numeric cols follow)
    """
    try:
        engine = "xlrd" if data[:4] == _XLS_MAGIC else "openpyxl"
        xl  = pd.ExcelFile(io.BytesIO(data), engine=engine)
        raw = xl.parse(xl.sheet_names[0], header=None)
    except Exception as exc:
        raise ParseError(f"Cannot open FII Stats file: {exc}") from exc

    if raw.shape[1] < 6:
        raise ParseError(f"FII Stats has only {raw.shape[1]} columns (expected 7)")

    # Assign fixed column names — structure is stable across report versions
    # Col 0: category | 1: buy_contracts | 2: buy_value_cr |
    # 3: sell_contracts | 4: sell_value_cr | 5: oi_contracts | 6: oi_value_cr
    col_names = [
        "category",
        "buy_contracts", "buy_value_cr",
        "sell_contracts", "sell_value_cr",
        "oi_contracts",   "oi_value_cr",
    ]
    # Some files have 7 cols, some might have 6 (no oi_value_cr)
    raw = raw.iloc[:, :len(col_names)]
    raw.columns = col_names[:raw.shape[1]]

    # Pad missing cols
    for col in col_names:
        if col not in raw.columns:
            raw[col] = 0

    # Drop the title + header rows (rows 0–2) and any row whose category
    # is not in our valid set
    raw["category"] = raw["category"].astype(str).str.strip().str.upper()
    df = raw[raw["category"].isin(_VALID_CATEGORIES)].copy()

    if df.empty:
        raise ParseError(f"No valid category rows found in FII Stats for {trade_date}")

    df["trade_date"] = trade_date

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce",
                )
                .fillna(0)
            )
        else:
            df[col] = 0.0

    for col in ("buy_contracts", "sell_contracts", "oi_contracts"):
        df[col] = df[col].astype("int64")
    for col in ("buy_value_cr", "sell_value_cr", "oi_value_cr"):
        df[col] = df[col].astype("float64")

    return df[["trade_date", "category"] + _NUMERIC_COLS]
