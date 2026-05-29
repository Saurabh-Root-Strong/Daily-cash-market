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
import json
import re
import urllib.parse
from datetime import date
from pathlib import Path

import pandas as pd

from src.core.exceptions import ParseError
from src.core.logging import get_logger
from src.ingestion.base import BaseFetcher

__all__ = ["FIIStatsFetcher"]

_DERIVATIVES_PAGE = "https://www.nseindia.com/all-reports-derivatives"

log = get_logger(__name__)

def _fii_stats_url_candidates(trade_date: date) -> list:
    """
    Ordered fallback chain for FII Derivatives Statistics XLS.
    Tries direct archive domains first, then NSE API as last resort.
    """
    dmon       = trade_date.strftime("%d-%b-%Y")   # 26-May-2026
    ddmmyyyy   = trade_date.strftime("%d%m%Y")     # 26052026
    dd_mm_yyyy = trade_date.strftime("%d-%m-%Y")   # 26-05-2026

    def _api():
        a = json.dumps([{"name": "F&O - FII Derivatives Statistics",
                         "type": "archives", "category": "derivatives",
                         "section": "equity"}])
        p = urllib.parse.urlencode({"archives": a, "date": dd_mm_yyyy,
                                    "type": "equity", "mode": "single"})
        return f"https://www.nseindia.com/api/reports?{p}"

    return [
        f"https://nsearchives.nseindia.com/content/fo/fii_stats_{dmon}.xls",   # primary
        f"https://archives.nseindia.com/content/fo/fii_stats_{dmon}.xls",      # fallback 1
        f"https://nsearchives.nseindia.com/content/fo/fii_stats_{dmon}.xlsx",  # fallback 2 (.xlsx)
        f"https://archives.nseindia.com/content/fo/fii_stats_{dmon}.xlsx",
        _api(),                                                                  # fallback 3 (API)
    ]

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

        for url in _fii_stats_url_candidates(trade_date):
            try:
                data = self._client.get_bytes(url, expect_404_ok=True)
            except Exception as exc:
                log.debug("FII Stats fallback failed %s: %s", url[-60:], exc)
                continue

            if not data:
                continue
            if not (data[:4] == _XLS_MAGIC or data[:4] == _XLSX_MAGIC):
                continue  # HTML/JSON error page — try next

            try:
                df = _parse_fii_stats_xls(data, trade_date)
                log.debug("FII Stats for %s fetched via %s", trade_date, url[-60:])
                return df
            except ParseError as exc:
                log.warning("FII Stats parse error %s via %s: %s",
                            trade_date, url[-60:], exc)
                continue  # Try next URL — maybe different format

        log.debug("FII Stats not available for %s (all sources returned empty)", trade_date)
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


# ── Folder import (manual drop) ───────────────────────────────────────────────

_DATE_FROM_TITLE = re.compile(
    r"\b(\d{1,2})[- ]([A-Za-z]{3,9})[- ](\d{4})\b"
)

def _date_from_title_row(raw_df: pd.DataFrame) -> date | None:
    """Extract trade date from the NSE title row: 'FII DERIVATIVES STATISTICS FOR 26-May-2026'."""
    title = str(raw_df.iloc[0, 0]) if not raw_df.empty else ""
    m = _DATE_FROM_TITLE.search(title)
    if not m:
        return None
    try:
        return date(int(m.group(3)),
                    list(__import__("calendar").month_abbr).index(m.group(2).capitalize()),
                    int(m.group(1)))
    except Exception:
        return None


def import_fii_stats_folder(folder: str | Path | None = None) -> dict:
    """
    Parse all .xls / .xlsx files in data/fii_imports/ and upsert into DB.
    Returns summary dict with files_processed, rows_inserted, errors.
    """
    from src.data.repository import get_repository

    if folder is None:
        from src.core.config import PROJECT_ROOT
        folder = PROJECT_ROOT / "data" / "fii_imports"

    folder = Path(folder)
    repo = get_repository()

    files_processed = 0
    rows_inserted = 0
    errors: list[str] = []

    for f in sorted(folder.glob("*.xls*")):
        try:
            data = f.read_bytes()
            engine = "xlrd" if data[:4] == b"\xd0\xcf\x11\xe0" else "openpyxl"
            xl = pd.ExcelFile(io.BytesIO(data), engine=engine)
            raw = xl.parse(xl.sheet_names[0], header=None)

            trade_date = _date_from_title_row(raw)
            if trade_date is None:
                errors.append(f"{f.name}: could not parse date from title row")
                continue

            df = _parse_fii_stats_xls(data, trade_date)
            if df.empty:
                errors.append(f"{f.name}: no valid rows parsed")
                continue

            repo.upsert_fii_stats(df)
            rows_inserted += len(df)
            files_processed += 1
            log.info("Imported %d FII stats rows for %s from %s", len(df), trade_date, f.name)

        except Exception as exc:
            errors.append(f"{f.name}: {exc}")

    return {"files_processed": files_processed, "rows_inserted": rows_inserted, "errors": errors}
