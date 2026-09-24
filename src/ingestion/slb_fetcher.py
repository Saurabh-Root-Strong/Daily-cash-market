"""
SLBFetcher — NSE Securities Lending & Borrowing Scheme (SLBS) daily archives.

WHAT SLB IS (NSE Clearing FAQ, 2025-07; SEBI SLB framework)
    A lender parks shares with the clearing corporation for a fixed tenure and earns
    a market-determined fee; a borrower takes delivery of those shares, pays the fee,
    and must return the same quantity on the reverse-leg date. It is the ONLY way in
    Indian cash equity to carry a short sale past the day it was made, which is why
    outstanding SLB borrowing is read as a short-interest proxy. It is not a pure
    one: borrowed stock also funds reverse cash-futures arbitrage (short cash, long
    futures: market-neutral) and covers delivery shortfalls.

ENDPOINTS (listed by /api/merged-daily-reports?key=SLBS; the capital-market list
does not carry them)
    archives/slbs/bhavcopy/SLBM_BC_DDMMYYYY.DAT       per-contract fee OHLC + volume
    archives/slbs/open_pos/slb_openpos_DDMMYYYY.csv   outstanding quantity per series
    content/slbs/Forclosure_SLB_YYYYMMDD.CSV          corporate-action foreclosures
    (the file name really is "Forclosure": NSE's spelling)

RETENTION
    Both daily files verified 200 back to 2013 with an unchanged layout (2013, 2018,
    2021, 2024, 2025, 2026 sampled). A 404 is a holiday (15-Jan-2026 was one).

FORMAT TRAPS
    - The bhavcopy DAT has NO header row. The 17 columns were identified by
      reconciling rows: value / quantity sits inside [low, high] on every contract,
      and the two columns after value are the contract-LIFE high / low, not 52-week.
    - PRICES ARE A FEE, not a share price: rupees per share for the WHOLE remaining
      tenure of the contract. Rs 2 on a 3-day contract and Rs 2 on a 10-month contract
      are wildly different costs, so a fee is only comparable once annualised against
      the cash close and the days left to the reverse leg.
    - The reverse-leg date moved from the first THURSDAY (2018-2025, still what the
      NCL FAQ says) to the first TUESDAY (the 2026 files show 06-OCT-2026 and
      03-NOV-2026). Always read the expiry from the file, never compute it.
    - Two contract families per month (NCL FAQ Q9): numeric 01..12 are foreclosed on
      every corporate action incl. AGM/EGM; X1..XD (XO=Oct, XN=Nov, XD=Dec) are NOT
      foreclosed on AGM/EGM. Two-letter series (OD/ON, O1/O2, F1 ...) are ROLLOVER
      contracts: a fee to extend an existing position, not a fresh borrow.
    - Series are plain two-character strings: "01" must never become the int 1.
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from src.core.logging import get_logger
from src.ingestion.http_client import NSEHttpClient

__all__ = ["SLBFetcher", "parse_bhavcopy", "parse_open_positions",
           "parse_foreclosure", "series_kind"]

log = get_logger(__name__)

_ARCH = "https://nsearchives.nseindia.com/"
_BHAV_URL = _ARCH + "archives/slbs/bhavcopy/SLBM_BC_{d}.DAT"
_OPEN_URL = _ARCH + "archives/slbs/open_pos/slb_openpos_{d}.csv"
_FORE_URL = _ARCH + "content/slbs/Forclosure_SLB_{ymd}.CSV"

_BHAV_COLS = ["security_name", "symbol", "series", "expiry_date", "market_type",
              "prev_fee", "open_fee", "high_fee", "low_fee", "close_fee", "_blank",
              "quantity", "value", "life_high_fee", "life_low_fee", "trade_date",
              "trades"]

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _to_date(s: pd.Series) -> pd.Series:
    """DD-MON-YYYY -> date, without the locale-dependent %b."""
    parts = s.astype(str).str.strip().str.upper().str.split("-", n=2, expand=True)
    if parts.shape[1] < 3:
        return pd.Series(pd.NaT, index=s.index)
    return pd.to_datetime(
        {"year": pd.to_numeric(parts[2], errors="coerce"),
         "month": parts[1].map(_MONTHS),
         "day": pd.to_numeric(parts[0], errors="coerce")},
        errors="coerce",
    ).dt.date


def series_kind(series: pd.Series) -> pd.Series:
    """REGULAR (01..12, foreclosed on AGM/EGM) | NO_AGM (X1..XD) | ROLLOVER (other)."""
    s = series.astype(str).str.strip().str.upper()
    regular = s.str.fullmatch(r"(0[1-9]|1[0-2])")
    no_agm = s.str.fullmatch(r"X[1-9ODN]")
    return pd.Series(
        [("REGULAR" if r else "NO_AGM" if n else "ROLLOVER") for r, n in zip(regular, no_agm)],
        index=series.index,
    )


def parse_bhavcopy(text: str) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), header=None, dtype=str, skipinitialspace=True)
    if df.shape[1] != len(_BHAV_COLS):
        raise ValueError(f"SLB bhavcopy has {df.shape[1]} columns, expected {len(_BHAV_COLS)}")
    df.columns = _BHAV_COLS
    df = df.apply(lambda c: c.str.strip())
    out = pd.DataFrame({
        "trade_date": _to_date(df["trade_date"]),
        "symbol": df["symbol"],
        "series": df["series"].str.upper(),
        "expiry_date": _to_date(df["expiry_date"]),
        "security_name": df["security_name"],
    })
    out["series_kind"] = series_kind(out["series"])
    for c in ["prev_fee", "open_fee", "high_fee", "low_fee", "close_fee",
              "life_high_fee", "life_low_fee", "value"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    out["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    ok = out["trade_date"].notna() & out["symbol"].ne("") & out["quantity"].notna()
    return out.loc[ok].reset_index(drop=True)


def parse_open_positions(text: str, trade_date: date) -> pd.DataFrame:
    if not text or not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    qcol = next(c for c in df.columns if c.startswith("outstanding"))
    out = pd.DataFrame({
        "trade_date": trade_date,
        "symbol": df["security"].str.strip(),
        "series": df["series"].str.strip().str.upper(),
        "open_qty": pd.to_numeric(df[qcol].str.replace(",", ""), errors="coerce"),
    })
    out["series_kind"] = series_kind(out["series"])
    out = out.loc[out["symbol"].notna() & out["open_qty"].notna()]
    out["open_qty"] = out["open_qty"].astype("int64")
    # One row per (date, symbol, series) — NSE has never been seen to repeat one,
    # but summing is the only safe reading if it ever does.
    return (out.groupby(["trade_date", "symbol", "series", "series_kind"], as_index=False)
               ["open_qty"].sum())


def parse_foreclosure(text: str) -> pd.DataFrame:
    """The file has a legend appended after a blank line; keep only the data block."""
    if not text or not text.strip():
        return pd.DataFrame()
    block = text.split("\n\n")[0].strip()
    df = pd.read_csv(io.StringIO(block), dtype=str)
    df.columns = [c.strip().rstrip("*").strip().lower().replace(" ", "_") for c in df.columns]
    if "security" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "symbol": df["security"].str.strip(),
        "series_scope": df["series"].str.strip(),
        "record_date": _to_date(df["record_date"]),
        "ex_date": _to_date(df["ex_date"]),
        "foreclosure_date": _to_date(df["foreclosure_date"]),
        "shut_start": _to_date(df["shut_period_start_date"]),
        "shut_end": _to_date(df["shut_period_end_date"]),
        "description": df["corporate_action_description"].str.split(r"\s{2,}").str[0].str.strip(),
    })
    return out.loc[out["symbol"].notna()].reset_index(drop=True)


class SLBFetcher:
    def __init__(self, client: NSEHttpClient | None = None) -> None:
        self._client = client or NSEHttpClient()

    def fetch_bhavcopy_text(self, d: date) -> str | None:
        return self._client.get_text(_BHAV_URL.format(d=d.strftime("%d%m%Y")), expect_404_ok=True)

    def fetch_open_positions_text(self, d: date) -> str | None:
        return self._client.get_text(_OPEN_URL.format(d=d.strftime("%d%m%Y")), expect_404_ok=True)

    def fetch_foreclosure_text(self, d: date) -> str | None:
        return self._client.get_text(_FORE_URL.format(ymd=d.strftime("%Y%m%d")), expect_404_ok=True)

    def fetch_day(self, d: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(bhavcopy, open positions) for one session; both empty on a holiday."""
        bc = self.fetch_bhavcopy_text(d)
        op = self.fetch_open_positions_text(d)
        bhav = parse_bhavcopy(bc) if bc else pd.DataFrame()
        opos = parse_open_positions(op, d) if op else pd.DataFrame()
        return bhav, opos
