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

_DERIVATIVES_PAGE = "https://www.nseindia.com/all-reports-derivatives"
_ARCHIVE_BASE     = "https://www.nseindia.com/api/reports"

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

    def __init__(self, client) -> None:
        super().__init__(client)
        self._derivatives_primed = False

    @property
    def name(self) -> str:
        return "FNO BhavCopy"

    def _prime(self) -> None:
        """Hit the derivatives page once per fetcher instance to prime cookies."""
        if self._derivatives_primed:
            return
        try:
            self._client.get(_DERIVATIVES_PAGE, expect_404_ok=True)
            log.debug("Primed NSE derivatives session for FNO bhavcopy")
            self._derivatives_primed = True
        except Exception as exc:
            log.debug("FNO bhavcopy prime failed (non-fatal): %s", exc)

    def _fno_url_candidates(self, trade_date: date) -> list:
        """
        Ordered fallback chain for FNO Bhavcopy.
        API no-date is primary (confirmed working); direct archive URLs as fallbacks.
        Direct archive URLs only work for the last ~7 days.
        """
        dm   = trade_date.strftime("%d%m%Y")
        dmon = trade_date.strftime("%d-%b-%Y").upper()   # 26-MAY-2026
        return [
            _build_archive_url_no_date(_FNO_BHAVCOPY_NAME),        # primary: API latest
            f"https://nsearchives.nseindia.com/content/fo/FNO_BC{dm}.DAT",
            f"https://archives.nseindia.com/content/fo/FNO_BC{dm}.DAT",
            f"https://nsearchives.nseindia.com/content/fo/fo{dm}bhav.csv.zip",
            f"https://archives.nseindia.com/content/fo/fo{dm}bhav.csv.zip",
        ]

    def fetch(self, trade_date: date) -> pd.DataFrame:
        """Returns one row per F&O instrument. Empty DataFrame when no data."""
        self._prime()

        from datetime import date as _date
        candidates = self._fno_url_candidates(trade_date)
        # The no-date API (index 0) always returns today's file regardless of the
        # requested date.  For historical dates skip it so the direct archive URLs
        # (which work for the last ~7 days) get tried instead.
        if trade_date != _date.today():
            candidates = candidates[1:]

        for url in candidates:
            try:
                data = self._client.get_bytes(url, expect_404_ok=True)
            except Exception as exc:
                log.debug("FNO fallback failed %s: %s", url[-60:], exc)
                continue

            if not data:
                continue

            try:
                text = _decode_response(data)
            except Exception:
                continue

            if not text:
                continue

            try:
                df = _parse_fno_dat(text, trade_date)
                log.debug("FNO bhavcopy for %s fetched via %s", trade_date, url[-60:])
                return df
            except ParseError as exc:
                log.warning("FNO bhavcopy parse error %s via %s: %s",
                            trade_date, url[-60:], exc)
                continue

        log.debug("FNO bhavcopy not available for %s (all sources returned empty)", trade_date)
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


# ── Folder import (manual drop of fo*.zip files) ─────────────────────────────

import re as _re
from pathlib import Path as _Path

_CONTRACT_RE = _re.compile(
    r'^(FUTIDX|FUTSTK|OPTIDX|OPTSTK)'   # instrument (6 chars)
    r'(.+?)'                              # symbol (non-greedy)
    r'(\d{2}-[A-Z]{3}-\d{4})'            # expiry DD-MMM-YYYY
    r'(CE|PE)?'                           # option type (absent for futures)
    r'(\d+\.?\d*)?$'                      # strike (absent for futures)
)

_MONTH_MAP = {m: i+1 for i, m in enumerate(
    ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
)}


def _parse_contract_d(series: "pd.Series") -> "pd.DataFrame":
    """Parse CONTRACT_D into instrument, symbol, expiry_date, option_type, strike_price."""
    from datetime import date as _date
    rows = []
    for val in series:
        m = _CONTRACT_RE.match(str(val).strip())
        if not m:
            rows.append(None)
            continue
        instr, sym, exp_str, opt, strike = m.groups()
        d_str, mo_str, yr_str = exp_str.split('-')
        exp = _date(int(yr_str), _MONTH_MAP[mo_str], int(d_str))
        rows.append({
            "instrument":  instr,
            "symbol":      sym,
            "expiry_date": exp,
            "option_type": opt if opt else "XX",
            "strike_price": float(strike) if strike else 0.0,
        })
    return pd.DataFrame([r for r in rows if r is not None], index=[i for i,r in enumerate(rows) if r is not None])


def _parse_old_fno_zip(zip_path: _Path, trade_date: "date") -> "pd.DataFrame":
    """Parse old-format NSE FNO zip (fo*.csv + op*.csv with CONTRACT_D column)."""
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith('.csv'):
                continue
            with zf.open(name) as f:
                raw = pd.read_csv(f, dtype=str, low_memory=False)

            raw.columns = [c.strip() for c in raw.columns]
            if 'CONTRACT_D' not in raw.columns:
                continue

            parsed = _parse_contract_d(raw['CONTRACT_D'])
            if parsed.empty:
                continue

            raw = raw.iloc[parsed.index].reset_index(drop=True)
            parsed = parsed.reset_index(drop=True)

            df = parsed.copy()
            df['trade_date']    = trade_date
            df['open_price']    = pd.to_numeric(raw.get('OPEN_PRICE',  pd.Series()), errors='coerce')
            df['high_price']    = pd.to_numeric(raw.get('HIGH_PRICE',  pd.Series()), errors='coerce')
            df['low_price']     = pd.to_numeric(raw.get('LOW_PRICE',   pd.Series()), errors='coerce')
            df['close_price']   = pd.to_numeric(raw.get('CLOSE_PRIC',  pd.Series()), errors='coerce')
            df['settle_price']  = pd.to_numeric(raw.get('SETTLEMENT',  pd.Series()), errors='coerce')
            df['open_interest'] = pd.to_numeric(raw.get('OI_NO_CON',   pd.Series()), errors='coerce').fillna(0).astype('int64')
            df['contracts']     = pd.to_numeric(raw.get('TRD_NO_CON',  pd.Series()), errors='coerce').fillna(0).astype('int64')
            df['value_lacs']    = pd.to_numeric(raw.get('TRADED_VAL',  pd.Series()), errors='coerce').fillna(0) / 100_000
            df['chg_in_oi']     = 0

            frames.append(df[_SCHEMA_COLS])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def import_fno_folder(folder: "_Path | str | None" = None) -> dict:
    """
    Parse all fo*.zip files in data/fii_imports/ (or given folder) and upsert into DB.
    Filename must match foDD MM YY.zip  e.g. fo140526.zip → 2026-05-14
    """
    from src.core.config import PROJECT_ROOT
    from src.data.repository import get_repository
    from datetime import date

    if folder is None:
        folder = PROJECT_ROOT / "data" / "fii_imports"
    folder = _Path(folder)
    repo = get_repository()

    files_processed = 0
    rows_inserted   = 0
    errors: list[str] = []

    for f in sorted(folder.glob("fo*.zip")):
        # Filename: foDDMMYY.zip  e.g. fo140526 → day=14 month=05 year=2026
        stem = f.stem  # fo140526
        try:
            dd, mm, yy = int(stem[2:4]), int(stem[4:6]), int(stem[6:8])
            trade_date = date(2000 + yy, mm, dd)
        except Exception:
            errors.append(f"{f.name}: cannot parse date from filename")
            continue

        try:
            df = _parse_old_fno_zip(f, trade_date)
            if df.empty:
                errors.append(f"{f.name}: no rows parsed")
                continue
            repo.upsert_fno_bhavcopy(df)
            rows_inserted += len(df)
            files_processed += 1
            log.info("Imported %d FNO rows for %s from %s", len(df), trade_date, f.name)
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")

    return {"files_processed": files_processed, "rows_inserted": rows_inserted, "errors": errors}
