"""
DealsFetcher — NSE bulk deals / block deals / short-selling archives.

ENDPOINT (the only one that serves history; do not "fix" it to /api/historical/)
    https://www.nseindia.com/api/historicalOR/bulk-block-short-deals
        ?optionType=bulk_deals|block_deals|short_selling
        &from=DD-MM-YYYY&to=DD-MM-YYYY
        &csv=true

WHY csv=true IS MANDATORY
    The JSON form of this endpoint is paginated and hard-capped at 70 rows with
    no page parameter that works (page / pageNo / start+length / limit were all
    verified inert: each returned the same first 70 rows). `csv=true` returns the
    WHOLE range in one response — a full calendar year of bulk deals comes back
    complete (2019: 7,565 rows across 244 sessions). So the unit of fetch here is
    a year, not a day.

    /content/equities/bulk.csv on nsearchives is NOT a substitute: it carries the
    latest session only (verified 2026-09-11, 218 rows, one date).

RETENTION
    Verified 200 with real rows back to 2010 (16,202 bulk rows in 2010). The
    practical floor for any study is daily_data's own start, 2018-01-01.

FORMAT TRAPS
    - Headers carry a trailing space: "Date ", "Symbol ", "Buy / Sell ".
    - Quantities are Indian-grouped strings: "1,40,003" — a plain int() fails
      and pandas leaves them as object dtype.
    - The feed contains genuine duplicate rows (same date/symbol/client/qty/price):
      two separate trades reported identically. They are NOT de-duplicated here —
      dropping them would understate the day's traded quantity. Hence no primary
      key on the table; idempotency comes from delete-by-date-range on write.
    - Dates are DD-MON-YYYY.
"""
from __future__ import annotations

import io
import time
from datetime import date

import pandas as pd
import requests

from src.core.logging import get_logger

__all__ = ["DealsFetcher", "DEAL_TYPES", "_to_date"]

log = get_logger(__name__)

_BASE = "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals"
_REFERER = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"

# option-type token -> deal_type stored in the DB
DEAL_TYPES = {"bulk_deals": "BULK", "block_deals": "BLOCK"}

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _REFERER,
}


# NSE writes dates as DD-MON-YYYY. pandas' %b is LOCALE-DEPENDENT: under a non-English
# LC_TIME every row parses to NaT and the loader would silently drop the whole file with
# only a warning. Map the months explicitly so the parse can never depend on the machine.
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _to_date(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.strip().str.upper().str.split("-", n=2, expand=True)
    if parts.shape[1] < 3:
        return pd.Series(pd.NaT, index=s.index)
    return pd.to_datetime(
        {
            "year":  pd.to_numeric(parts[2], errors="coerce"),
            "month": parts[1].map(_MONTHS),
            "day":   pd.to_numeric(parts[0], errors="coerce"),
        },
        errors="coerce",
    )


def _to_int(s: pd.Series) -> pd.Series:
    """'1,40,003' -> 140003."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                         errors="coerce").astype("Int64")


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(),
                         errors="coerce")


class DealsFetcher:
    """Range fetcher for bulk / block / short-selling archives."""

    name = "Bulk & Block deals"

    def __init__(self, session: requests.Session | None = None,
                 polite_delay: float = 2.0) -> None:
        self._s = session or self._new_session()
        self._delay = polite_delay

    @staticmethod
    def _new_session() -> requests.Session:
        s = requests.Session()
        s.headers.update(_HEADERS)
        # Cookie priming. nseindia.com/ itself answers 403 to this UA but still
        # sets the Akamai cookies; the report page then answers 200. Both calls
        # are needed before the api/ host will serve anything but a 503.
        for url in ("https://www.nseindia.com/", _REFERER):
            try:
                s.get(url, timeout=30)
            except Exception as exc:  # priming is best-effort
                log.debug("Cookie priming failed for %s: %s", url, exc)
            time.sleep(1)
        return s

    # ── raw ───────────────────────────────────────────────────────────────────

    def fetch_csv(self, option_type: str, start: date, end: date,
                  retries: int = 3) -> str:
        url = (f"{_BASE}?optionType={option_type}"
               f"&from={start.strftime('%d-%m-%Y')}&to={end.strftime('%d-%m-%Y')}"
               f"&csv=true")
        last = ""
        for attempt in range(1, retries + 1):
            r = self._s.get(url, timeout=180)
            if r.status_code == 200 and not r.text.lstrip().startswith("<"):
                return r.content.decode("utf-8-sig")
            last = f"HTTP {r.status_code}"
            log.warning("%s %s..%s attempt %d/%d: %s", option_type, start, end,
                        attempt, retries, last)
            self._s = self._new_session()
            time.sleep(self._delay * attempt)
        raise RuntimeError(f"{option_type} {start}..{end} failed: {last}")

    # ── parsed ────────────────────────────────────────────────────────────────

    def fetch_deals(self, option_type: str, start: date, end: date) -> pd.DataFrame:
        """bulk_deals / block_deals -> normalised deal rows."""
        text = self.fetch_csv(option_type, start, end)
        df = pd.read_csv(io.StringIO(text), dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if df.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "trade_date":    _to_date(df["Date"]).dt.date,
            "deal_type":     DEAL_TYPES[option_type],
            "symbol":        df["Symbol"].str.strip(),
            "security_name": df["Security Name"].str.strip(),
            "client_name":   df["Client Name"].str.strip(),
            "buy_sell":      df["Buy / Sell"].str.strip().str.upper(),
            "quantity":      _to_int(df["Quantity Traded"]),
            "trade_price":   _to_float(df["Trade Price / Wght. Avg. Price"]),
            "remarks":       df.get("Remarks", pd.Series([""] * len(df))).fillna("").str.strip(),
        })
        bad = out["trade_date"].isna() | out["quantity"].isna() | out["trade_price"].isna()
        if bad.any():
            log.warning("Dropping %d unparseable %s rows (%s..%s)",
                        int(bad.sum()), option_type, start, end)
        return out.loc[~bad].reset_index(drop=True)

    def fetch_short_selling(self, start: date, end: date) -> pd.DataFrame:
        text = self.fetch_csv("short_selling", start, end)
        df = pd.read_csv(io.StringIO(text), dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if df.empty:
            return pd.DataFrame()
        out = pd.DataFrame({
            "trade_date":    _to_date(df["Date"]).dt.date,
            "symbol":        df["Symbol"].str.strip(),
            "security_name": df["Security Name"].str.strip(),
            "quantity":      _to_int(df["Quantity"]),
        })
        bad = out["trade_date"].isna() | out["quantity"].isna()
        return out.loc[~bad].reset_index(drop=True)
