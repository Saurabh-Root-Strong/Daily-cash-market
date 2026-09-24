"""
Corporate actions (dividends, splits, bonus, buybacks...) from NSE.

WHY THIS EXISTS
    NSE's bhavcopy prev_close is adjusted for splits / bonus, but NOT for ordinary
    dividends (verified: VEDL / COALINDIA / ITC / ONGC ex-dates 2023-2026 show zero
    adjustment). Every return compounded from close / prev_close therefore books the
    ex-dividend drop as a loss. Any signal that clusters around record dates (SLB
    borrowing does) reads that mechanical drop as "forecast underperformance" unless
    the dividend is added back.

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

__all__ = ["fetch_corporate_actions", "parse_dividend_amount"]

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


def fetch_corporate_actions(start: date, end: date) -> pd.DataFrame:
    s = requests.Session()
    s.headers.update({"User-Agent": _UA, "Referer": _PRIME})
    s.get(_PRIME, timeout=30)
    rows = []
    for q in pd.date_range(start, end, freq="QS"):
        qe = min((q + pd.offsets.QuarterEnd(0)).date(), end)
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
    out["dividend"] = out["subject"].map(parse_dividend_amount)
    return out.drop_duplicates().reset_index(drop=True)
