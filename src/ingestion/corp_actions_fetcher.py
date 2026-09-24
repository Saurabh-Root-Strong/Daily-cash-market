"""
Corporate actions (dividends, splits, bonus, buybacks...) from NSE.

WHY THIS EXISTS
    daily_data.prev_close is NOT adjusted for ANY corporate action -- not dividends
    (VEDL / COALINDIA / ITC / ONGC 2023-2026) and not splits or bonuses either
    (BAJFINANCE 2025-06-16 prev_close 9,331 vs close 938; IRCTC 2021-10-28; HDFCBANK
    2025-08-26). So:
      - every return compounded from close / prev_close books an ex-dividend drop
        as a loss unless the dividend is added back;
      - "prev_close disagrees with yesterday's close" NEVER detects a split, and any
        guard built on it silently never fires;
      - share quantities (SLB outstanding, delivery) jump by the split factor on the
        ex-date while trailing averages stay in old shares.
    This feed is the only reliable record of all three.

ENDPOINT
    /api/corporates-corporateActions?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY
    JSON list; a quarter per call returns a few hundred rows. Needs primed cookies.
    Dividend amount is only in free text: "Dividend - Rs 7.50 Per Share",
    "Interim Dividend - Re 1 Per Share", "Final Dividend - Rs 2.50 Per Sh" ...
"""
from __future__ import annotations

import re
import time
from datetime import date

import pandas as pd
import requests

from src.core.logging import get_logger

__all__ = ["fetch_corporate_actions", "parse_dividend_amount", "parse_share_factor",
           "classify_action", "enrich"]

log = get_logger(__name__)

_URL = ("https://www.nseindia.com/api/corporates-corporateActions"
        "?index=equities&from_date={f}&to_date={t}")
_PRIME = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Subjects bundle several actions: "Interim Dividend- Rs 3 Per Share / Rights 1:16 @
# Premium Rs 554 Per Share", "Dividend - Rs 10 Per Share/Face Value Split ... From
# Rs 10/- Per Share To Re 1/- Per Share". So: split into clauses, keep only clauses
# that name a dividend, and take the number directly before "Per Share" in each
# ("Rs 7.50 Per Share", "- 0.40 Per Share"). "Special Dividend Rs 131 Per Share &
# Dividend Rs 525 Per Share" pays the SUM of its dividend clauses.
_CLAUSE = re.compile(r"[/&+,;]")
_AMT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(?:/-)?\s*PER\s*SH", re.I)


def parse_dividend_amount(subject: str) -> float:
    total = 0.0
    # "Rs 2/- Per Share": drop the "/-" before "/" is used as a clause separator.
    for clause in _CLAUSE.split((subject or "").replace("/-", " ")):
        if "DIVIDEND" not in clause.upper().replace(" ", ""):
            continue
        m = _AMT.search(clause)
        if m:
            total += float(m.group(1))
    return total


_FACE = re.compile(r"FROM\s+R[SE]\.?\s*([0-9.]+)\s*(?:/-)?.*?\bTO\s+R[SE]\.?\s*([0-9.]+)", re.I)
# "Bonus 1:1", "Bonus- 1:2". NOT "Bonus Ncrps 4:1" / "Bonus 1 Debenture For 1": those
# issue preference shares or debentures and leave the equity share count unchanged.
_BONUS = re.compile(r"BONUS[\s\-:]*(\d+)\s*:\s*(\d+)", re.I)


def parse_share_factor(subject: str) -> float:
    """How many shares one old share becomes on the ex-date (1.0 = unchanged).

    Split "From Rs 10/- ... To Rs 2/-" -> 5; consolidation "From Re 1 To Rs 10" -> 0.1;
    NSE "Bonus a:b" = a new shares for every b held -> (a + b) / b. A subject carrying
    both (BAJFINANCE 2025: split 2 -> 1 and bonus 4:1) multiplies them: 2 x 5 = 10.
    """
    s = (subject or "").upper()
    f = 1.0
    if "SPLIT" in s or "CONSOLIDAT" in s or "SUB-DIVISION" in s:
        m = _FACE.search(s)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            if a > 0 and b > 0:
                f *= a / b
    m = _BONUS.search(s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b > 0:
            f *= (a + b) / b
    return f


def classify_action(subject: str) -> str:
    """Coarse kind; share-count-changing kinds are what break quantity series."""
    s = (subject or "").upper()
    if "SPLIT" in s or "SUB-DIVISION" in s or "CONSOLIDAT" in s:
        return "SPLIT"
    if "BONUS" in s:
        return "BONUS"
    if "DEMERGER" in s:
        return "DEMERGER"
    if "RIGHTS" in s:
        return "RIGHTS"
    if "AMALGAMATION" in s or "ARRANGEMENT" in s or "MERGER" in s:
        return "SCHEME"
    if "BUY BACK" in s or "BUYBACK" in s:
        return "BUYBACK"
    if "DIVIDEND" in s.replace(" ", ""):
        return "DIVIDEND"
    return "OTHER"


def fetch_corporate_actions(start: date, end: date) -> pd.DataFrame:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Referer": _PRIME})
    s.get(_PRIME, timeout=30)
    rows = []
    # Chunk by calendar quarter starting AT `start`: pd.date_range(freq="QS") begins at
    # the first quarter START after it, which silently dropped a mid-quarter start.
    chunks, q = [], pd.Timestamp(start)
    while q.date() <= end:
        qe = min((q + pd.offsets.QuarterEnd(0)).date(), end)
        chunks.append((q, qe))
        q = pd.Timestamp(qe) + pd.Timedelta(days=1)
    for q, qe in chunks:
        url = _URL.format(f=q.strftime("%d-%m-%Y"), t=qe.strftime("%d-%m-%Y"))
        for attempt in range(4):
            try:
                r = s.get(url, timeout=60)
                r.raise_for_status()
                rows.extend(r.json())
                break
            except Exception as exc:
                log.warning("corp actions %s attempt %d: %s", q.date(), attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
                s.get(_PRIME, timeout=30)
        else:
            # Callers range-REPLACE the table; a chunk skipped here would be deleted
            # there. Fail the whole pull instead of returning a silently partial one.
            raise RuntimeError(f"corporate actions {q.date()}..{qe}: all retries failed")
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = pd.DataFrame({
        "symbol": df["symbol"].str.strip(),
        "series": df["series"].str.strip(),
        "subject": df["subject"].str.strip(),
        "ex_date": pd.to_datetime(df["exDate"], format="%d-%b-%Y", errors="coerce").dt.date,
        "record_date": pd.to_datetime(df["recDate"], format="%d-%b-%Y", errors="coerce").dt.date,
    })
    return enrich(out)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add dividend, share_factor and kind to a raw (symbol, subject, ex_date...) frame."""
    df = df.copy()
    df["dividend"] = df["subject"].map(parse_dividend_amount)
    df["share_factor"] = df["subject"].map(parse_share_factor)
    df["kind"] = df["subject"].map(classify_action)
    return df.drop_duplicates().reset_index(drop=True)
