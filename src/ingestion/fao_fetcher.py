"""
FAOParticipantFetcher — F&O Participant-wise OI and Volume reports from NSE.

NSE publishes two CSV files after each trading session, available directly at:
  https://archives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv
  https://archives.nseindia.com/content/nsccl/fao_participant_vol_DDMMYYYY.csv

Participants: Client (retail), DII, FII, Pro (proprietary)

Columns per file:
  Client Type, Future Index Long/Short, Future Stock Long/Short,
  Option Index Call/Put Long/Short, Option Stock Call/Put Long/Short,
  Total Long Contracts, Total Short Contracts
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from src.core.exceptions import ParseError
from src.core.logging import get_logger
from src.ingestion.base import BaseFetcher

__all__ = ["FAOParticipantFetcher"]

log = get_logger(__name__)

_DERIVATIVES_PAGE = "https://www.nseindia.com/all-reports-derivatives"

# Fallback chain for each file type — tried in order until one returns data
# Primary: nsearchives (confirmed direct download)
# Fallback 1: archives (old domain alias — may redirect)
# Fallback 2: NSE /api/reports (Archive API — worked before domain change)
def _oi_url_candidates(date_ddmmyyyy: str, date_dd_mm_yyyy: str) -> list:
    import json, urllib.parse
    def _api(name):
        a = json.dumps([{"name": name, "type": "archives",
                         "category": "derivatives", "section": "equity"}])
        p = urllib.parse.urlencode({"archives": a, "date": date_dd_mm_yyyy,
                                    "type": "equity", "mode": "single"})
        return f"https://www.nseindia.com/api/reports?{p}"
    return [
        f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{date_ddmmyyyy}.csv",
        f"https://archives.nseindia.com/content/nsccl/fao_participant_oi_{date_ddmmyyyy}.csv",
        _api("F&O - Participant wise Open Interest(csv)"),
        _api("F&O-Participant wise Open Interest (csv)"),
    ]

def _vol_url_candidates(date_ddmmyyyy: str, date_dd_mm_yyyy: str) -> list:
    import json, urllib.parse
    def _api(name):
        a = json.dumps([{"name": name, "type": "archives",
                         "category": "derivatives", "section": "equity"}])
        p = urllib.parse.urlencode({"archives": a, "date": date_dd_mm_yyyy,
                                    "type": "equity", "mode": "single"})
        return f"https://www.nseindia.com/api/reports?{p}"
    return [
        f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_vol_{date_ddmmyyyy}.csv",
        f"https://archives.nseindia.com/content/nsccl/fao_participant_vol_{date_ddmmyyyy}.csv",
        _api("F&O - Participant wise Trading Volumes(csv)"),
        _api("F&O-Participant wise Trading Volumes (csv)"),
    ]

_VALID_TYPES = {"Client", "DII", "FII", "Pro"}

_COL_MAP = {
    "Client Type":              "client_type",
    "Client type":              "client_type",
    "Future Index Long":        "fut_idx_long",
    "Future Index Short":       "fut_idx_short",
    "Future Stock Long":        "fut_stk_long",
    "Future Stock Short":       "fut_stk_short",
    "Option Index Call Long":   "opt_idx_call_long",
    "Option Index Call Short":  "opt_idx_call_short",
    "Option Index Put Long":    "opt_idx_put_long",
    "Option Index Put Short":   "opt_idx_put_short",
    "Option Stock Call Long":   "opt_stk_call_long",
    "Option Stock Call Short":  "opt_stk_call_short",
    "Option Stock Put Long":    "opt_stk_put_long",
    "Option Stock Put Short":   "opt_stk_put_short",
    "Total Long Contracts":     "total_long",
    "Total Short Contracts":    "total_short",
}

_NUMERIC_COLS = [
    "fut_idx_long", "fut_idx_short",
    "fut_stk_long", "fut_stk_short",
    "opt_idx_call_long", "opt_idx_call_short",
    "opt_idx_put_long",  "opt_idx_put_short",
    "opt_stk_call_long", "opt_stk_call_short",
    "opt_stk_put_long",  "opt_stk_put_short",
    "total_long", "total_short",
]

class FAOParticipantFetcher(BaseFetcher):
    """Downloads and parses both OI + Volume participant CSVs for one date."""

    def __init__(self, client) -> None:
        super().__init__(client)
        self._derivatives_primed = False

    @property
    def name(self) -> str:
        return "F&O Participant"

    def _prime_derivatives(self) -> None:
        """Hit the derivatives reports page once per fetcher instance to prime cookies."""
        if self._derivatives_primed:
            return
        try:
            self._client.get(_DERIVATIVES_PAGE, expect_404_ok=True)
            log.debug("Primed NSE derivatives session")
            self._derivatives_primed = True
        except Exception as exc:
            log.debug("Derivatives prime failed (non-fatal): %s", exc)

    def _fetch_csv(self, urls: list, label: str, trade_date: date) -> str | None:
        """Try each URL in order; return first valid CSV text, else None."""
        for url in urls:
            try:
                text = self._client.get_text(url, expect_404_ok=True)
            except Exception as exc:
                log.debug("FAO %s fallback failed %s: %s", label, url[-60:], exc)
                continue
            if not text or not text.strip():
                continue
            first = text.strip().splitlines()[0]
            if "<" in first or "{" in first:
                continue
            log.debug("FAO %s for %s fetched via %s", label, trade_date, url[-60:])
            return text
        return None

    def fetch(self, trade_date: date) -> pd.DataFrame:
        """Returns one row per (client_type x data_type).  Empty if no data."""
        self._prime_derivatives()

        ddmmyyyy   = trade_date.strftime("%d%m%Y")
        dd_mm_yyyy = trade_date.strftime("%d-%m-%Y")

        frames: list[pd.DataFrame] = []
        for data_type, candidates in (
            ("OI",  _oi_url_candidates(ddmmyyyy, dd_mm_yyyy)),
            ("Vol", _vol_url_candidates(ddmmyyyy, dd_mm_yyyy)),
        ):
            text = self._fetch_csv(candidates, data_type, trade_date)
            if text is None:
                log.debug("F&O %s not available for %s (all sources returned empty)",
                          data_type, trade_date)
                continue
            try:
                frames.append(_parse_fao_csv(text, trade_date, data_type))
            except ParseError as exc:
                log.warning("Parse error F&O %s %s: %s", data_type, trade_date, exc)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)


def _parse_fao_csv(text: str, trade_date: date, data_type: str) -> pd.DataFrame:
    """Parse one NSE F&O participant CSV into a normalised DataFrame."""
    try:
        # Row 0 is a title ("Participant wise Open Interest..."), row 1 is the header
        df = pd.read_csv(io.StringIO(text), skiprows=1)
    except Exception as exc:
        raise ParseError(f"Cannot parse F&O CSV: {exc}") from exc

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_COL_MAP)

    if "client_type" not in df.columns:
        raise ParseError(
            f"Missing 'Client Type' column in F&O {data_type} for {trade_date}. "
            f"Got: {list(df.columns)}"
        )

    df["client_type"] = df["client_type"].astype(str).str.strip()
    df = df[df["client_type"].isin(_VALID_TYPES)].copy()

    if df.empty:
        raise ParseError(f"No participant rows in F&O {data_type} for {trade_date}")

    df["trade_date"] = trade_date
    df["data_type"]  = data_type

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.strip(),
                    errors="coerce",
                )
                .fillna(0)
                .astype("int64")
            )
        else:
            df[col] = 0

    return df[["trade_date", "client_type", "data_type"] + _NUMERIC_COLS]
