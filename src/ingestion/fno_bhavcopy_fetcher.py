"""
FNOBhavCopyFetcher — F&O Bhavcopy DAT file from NSE Archives.

Daily snapshot of all active F&O instruments (futures + options).
File name pattern: FNO_BC{DDMMYYYY}.DAT
Report name in Archives API: "F&O - Bhavcopy File (DAT)"

File is a headerless CSV (no column names in file).

Column layout (0-indexed, 26 total columns):
  0:  full_ticker     — packed symbol (e.g. BANKNIFTY26JANFUT)
  1:  instrument      — FUTIDX | FUTSTK | OPTIDX | OPTSTK
  2:  symbol          — underlying (e.g. NIFTY, BANKNIFTY, RELIANCE)
  3:  expiry_date     — DDMMYYYY (e.g. 27012026 = 27-Jan-2026)
  4:  strike_price    — 0 for futures
  5:  option_type     — CE / PE / '' (blank → XX for futures)
  9:  settle_price    — settlement / previous close price
  10: open_price
  11: high_price
  12: low_price
  13: close_price
  15: contracts       — number of individual contracts traded
  16: value_rupees    — traded value in rupees → ÷100 000 for lakhs
  18: open_interest   — contracts
  19: chg_in_oi       — change in OI contracts
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd

from src.core.exceptions import ParseError
from src.core.logging import get_logger
from src.ingestion.base import BaseFetcher
import urllib.parse

from src.ingestion.fao_fetcher import (
    _build_archive_url,
    _primed_derivatives,
    _DERIVATIVES_PAGE,
    _ARCHIVE_BASE,
)

__all__ = ["FNOBhavCopyFetcher"]


def _build_archive_url_no_date(report_name: str) -> str:
    """Build NSE Archives API URL for a report WITHOUT a date (returns latest file)."""
    import json
    archives = json.dumps([{
        "name":     report_name,
        "type":     "archives",
        "category": "derivatives",
        "section":  "equity",
    }])
    params = urllib.parse.urlencode({
        "archives": archives,
        "type":     "equity",
        "mode":     "single",
    })
    return f"{_ARCHIVE_BASE}?{params}"

log = get_logger(__name__)

_FNO_BHAVCOPY_NAME = "F&O - Bhavcopy File (DAT)"

_VALID_INSTRUMENTS = {"FUTIDX", "OPTIDX", "FUTSTK", "OPTSTK"}

# Column indices in the headerless CSV
_COL_INSTRUMENT   = 1
_COL_SYMBOL       = 2
_COL_EXPIRY       = 3
_COL_STRIKE       = 4
_COL_OPTTYPE      = 5
_COL_SETTLE       = 9
_COL_OPEN         = 10
_COL_HIGH         = 11
_COL_LOW          = 12
_COL_CLOSE        = 13
_COL_CONTRACTS    = 15
_COL_VALUE_RUPEES = 16
_COL_OI           = 18
_COL_CHG_OI       = 19

_SCHEMA_COLS = [
    "trade_date", "instrument", "symbol", "expiry_date",
    "strike_price", "option_type",
    "open_price", "high_price", "low_price", "close_price", "settle_price",
    "contracts", "value_lacs", "open_interest", "chg_in_oi",
]


class FNOBhavCopyFetcher(BaseFetcher):
    """Downloads and parses the NSE FNO Bhavcopy DAT file for one date."""

    @property
    def name(self) -> str:
        return "FNO BhavCopy"

    def _prime(self) -> None:
        global _primed_derivatives
        if _primed_derivatives:
            return
        try:
            self._client.get(_DERIVATIVES_PAGE, expect_404_ok=True)
            import src.ingestion.fao_fetcher as _m
            _m._primed_derivatives = True
            log.debug("Primed NSE derivatives session for FNO bhavcopy")
        except Exception as exc:
            log.debug("FNO bhavcopy prime failed (non-fatal): %s", exc)

    def fetch(self, trade_date: date) -> pd.DataFrame:
        """
        Returns one row per F&O instrument. Empty DataFrame when no data.

        Note: NSE Archives API for this report is only reliable when called
        WITHOUT a date parameter (returns today's file).  Date-specific requests
        are broken — they all map to January of the same year.
        We therefore omit the date parameter; the returned filename confirms the
        actual date in the Content-Disposition header / col[21] field.
        """
        self._prime()

        # Omit date to get the latest available file from NSE
        url  = _build_archive_url_no_date(_FNO_BHAVCOPY_NAME)
        data = self._client.get_bytes(url, expect_404_ok=True)

        if not data:
            log.debug("FNO bhavcopy not available for %s (holiday/weekend/404)", trade_date)
            return pd.DataFrame()

        try:
            text = _decode_response(data)
        except Exception as exc:
            log.warning("FNO bhavcopy: cannot decode response for %s: %s", trade_date, exc)
            return pd.DataFrame()

        if not text:
            log.debug("FNO bhavcopy for %s: empty after decode", trade_date)
            return pd.DataFrame()

        try:
            return _parse_fno_dat(text, trade_date)
        except ParseError as exc:
            log.warning("FNO bhavcopy parse error for %s: %s", trade_date, exc)
            return pd.DataFrame()


def _decode_response(data: bytes) -> str:
    """Handle both plain CSV and zip-wrapped DAT files."""
    if data[:2] == b"PK":
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
            dat_name = next(
                (n for n in zf.namelist() if n.upper().endswith(".DAT")),
                zf.namelist()[0] if zf.namelist() else None,
            )
            if not dat_name:
                return ""
            return zf.open(dat_name).read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise ParseError(f"Cannot extract zip: {exc}") from exc

    return data.decode("utf-8", errors="replace")


_COL_TRADE_DATE = 21   # DD/Mon/YYYY — the actual trade date embedded in the file

def _parse_fno_dat(text: str, trade_date: date) -> pd.DataFrame:
    """
    Parse NSE FNO Bhavcopy DAT (headerless CSV) into a normalised DataFrame.

    The actual trade date is read from col[21] (format DD/Mon/YYYY) rather
    than using the requested trade_date, because the no-date API may return
    a file for a date slightly different from today (e.g. previous trading day
    after market hours).
    """
    if not text or not text.strip():
        raise ParseError("Empty FNO bhavcopy response")

    try:
        raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, low_memory=False)
    except Exception as exc:
        raise ParseError(f"Cannot parse FNO DAT: {exc}") from exc

    if raw.shape[1] < 20:
        raise ParseError(
            f"FNO DAT has only {raw.shape[1]} columns (expected ≥20)"
        )

    df = pd.DataFrame()
    df["instrument"]   = raw[_COL_INSTRUMENT].str.strip()
    df["symbol"]       = raw[_COL_SYMBOL].str.strip()
    df["_expiry_raw"]  = raw[_COL_EXPIRY].str.strip()
    df["strike_price"] = raw[_COL_STRIKE].str.strip()
    df["option_type"]  = raw[_COL_OPTTYPE].str.strip().str.upper()
    df["settle_price"] = raw[_COL_SETTLE].str.strip()
    df["open_price"]   = raw[_COL_OPEN].str.strip()
    df["high_price"]   = raw[_COL_HIGH].str.strip()
    df["low_price"]    = raw[_COL_LOW].str.strip()
    df["close_price"]  = raw[_COL_CLOSE].str.strip()
    df["contracts"]    = raw[_COL_CONTRACTS].str.strip()
    df["_value_rupees"]= raw[_COL_VALUE_RUPEES].str.strip()
    df["open_interest"]= raw[_COL_OI].str.strip()
    df["chg_in_oi"]    = raw[_COL_CHG_OI].str.strip()
    df["_date_col"]    = raw[_COL_TRADE_DATE].str.strip()

    # Keep only recognised instrument types
    df = df[df["instrument"].isin(_VALID_INSTRUMENTS)].copy()
    if df.empty:
        raise ParseError(f"No valid instrument rows in FNO bhavcopy for {trade_date}")

    # Extract actual trade date from col[21] (DD/Mon/YYYY format)
    # e.g. "20/MAY/2026" → 2026-05-20
    actual_dates = pd.to_datetime(
        df["_date_col"], format="%d/%b/%Y", errors="coerce"
    ).dt.date
    most_common = actual_dates.dropna().mode()
    if not most_common.empty:
        actual_trade_date = most_common.iloc[0]
        if actual_trade_date != trade_date:
            log.debug(
                "FNO bhavcopy: file date %s differs from requested %s — using file date",
                actual_trade_date, trade_date,
            )
    else:
        actual_trade_date = trade_date

    # Parse expiry date: DDMMYYYY → date
    df["expiry_date"] = pd.to_datetime(
        df["_expiry_raw"], format="%d%m%Y", errors="coerce"
    ).dt.date
    df = df.dropna(subset=["expiry_date"]).copy()

    # Normalise option_type: blank/NAN → XX for futures
    df["option_type"] = df["option_type"].replace({"": "XX", "NAN": "XX"})
    df["option_type"] = df["option_type"].fillna("XX")

    df["trade_date"] = actual_trade_date

    # Numeric casts
    for col in ("open_price", "high_price", "low_price", "close_price",
                "settle_price", "strike_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("contracts", "open_interest", "chg_in_oi"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    df["value_lacs"] = (
        pd.to_numeric(df["_value_rupees"], errors="coerce").fillna(0) / 100_000
    )

    df["strike_price"] = df["strike_price"].fillna(0.0)

    return df[[c for c in _SCHEMA_COLS if c in df.columns]].reset_index(drop=True)
