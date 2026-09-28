"""
Fetch the official 4-level industry classification for every stock in sector_master.

Taxonomy (NSE / AMFI harmonised, 2023+):
    Macro-Economic Sector  →  Sector  →  Industry  →  Basic Industry
    e.g. SUNPHARMA : Healthcare → Healthcare → Pharmaceuticals & Biotechnology → Pharmaceuticals
         APOLLOHOSP: Healthcare → Healthcare → Healthcare Services → Hospital

Source: Screener.in company pages (they publish NSE's classification; NSE's and
BSE's own JSON APIs are Akamai-blocked for scripts). robots.txt allows /company/*.

Polite + resumable: one request at a time, a delay between them, and every raw page
is cached under data/classification_raw/ so a re-run only fetches what is missing.
Output: config/stock_classification.csv (versioned, the durable artifact).
Load into DuckDB with scripts/load_stock_classification.py.

Run:
    python scripts/fetch_stock_classification.py            # fetch missing + parse all
    python scripts/fetch_stock_classification.py --parse-only
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR  = PROJECT_ROOT / "data" / "classification_raw"
OUT_CSV  = PROJECT_ROOT / "config" / "stock_classification.csv"
BASE_URL = "https://www.screener.in/company/{sym}/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_LEVELS = {"Broad Sector": "macro_sector", "Sector": "sector",
           "Broad Industry": "industry", "Industry": "basic_industry"}
_LEVEL_RE = re.compile(r'title="(Broad Sector|Sector|Broad Industry|Industry)"[^>]*>([^<]+)<')
_ETF_RE   = re.compile(r"\bETF\b|\bFund\b|\bBeES\b", re.I)
_NAME_RE  = re.compile(r"<title>\s*(.+?)\s+share price", re.S)


def _symbols(universe: str = "master") -> pd.DataFrame:
    """master = every sector_master symbol; all = master ∪ every EQ symbol traded in
    the last 60 sessions (new listings, SME migrations, renames, small caps)."""
    from src.data.repository import query_dataframe
    if universe == "master":
        return query_dataframe(
            "SELECT symbol, sector AS canonical_sector FROM v_sector_master ORDER BY symbol", []
        )
    return query_dataframe("""
        WITH recent AS (
            SELECT DISTINCT symbol FROM daily_data
            WHERE series = 'EQ' AND trade_date IN (
                SELECT DISTINCT trade_date FROM daily_data WHERE series = 'EQ'
                ORDER BY trade_date DESC LIMIT 60))
        SELECT u.symbol, m.sector AS canonical_sector
        FROM (SELECT symbol FROM v_sector_master UNION SELECT symbol FROM recent) u
        LEFT JOIN v_sector_master m USING (symbol)
        ORDER BY u.symbol
    """, [])


def _cache_path(sym: str) -> Path:
    return RAW_DIR / f"{re.sub(r'[^A-Za-z0-9_-]', '_', sym)}.html"


NOT_FOUND = "<!-- 404 -->"
NOT_FOUND_RETRY_DAYS = 7      # a fresh IPO may not be on the source on listing day
MAX_BAD_IN_A_ROW = 5          # consecutive non-company 200 pages ⇒ we are being blocked


def _age_days(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 86400


def _worklist(symbols: list[str], refresh_days: int | None) -> list[str]:
    """New symbols first, then expired 404 markers, then pages older than
    `refresh_days` (oldest first) so official reclassifications are picked up."""
    missing, retry_404, stale = [], [], []
    for sym in symbols:
        p = _cache_path(sym)
        if not p.exists():
            missing.append(sym)
            continue
        age = _age_days(p)
        if p.stat().st_size < 64 and NOT_FOUND in p.read_text(encoding="utf-8", errors="ignore"):
            if age >= NOT_FOUND_RETRY_DAYS:
                retry_404.append(sym)
        elif refresh_days is not None and age >= refresh_days:
            stale.append((age, sym))
    return missing + retry_404 + [s for _, s in sorted(stale, reverse=True)]


def _labels(text: str) -> tuple[str, ...]:
    found = {}
    for label, val in _LEVEL_RE.findall(text):
        found.setdefault(label, html.unescape(val).strip())
    return tuple(found.get(k, "") for k in _LEVELS)


def fetch(symbols: list[str], delay: float, max_fetch: int | None = None,
          refresh_days: int | None = None) -> None:
    import requests
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    todo = _worklist(symbols, refresh_days)
    if max_fetch is not None:
        todo = todo[:max_fetch]
    print(f"[fetch] {len(symbols)} symbols, {len(todo)} to fetch "
          f"(new / expired-404 / stale>{refresh_days}d)")
    s = requests.Session()
    bad_in_a_row = 0
    for i, sym in enumerate(todo, 1):
        url = BASE_URL.format(sym=quote(sym, safe=""))
        path = _cache_path(sym)
        for attempt in range(3):
            try:
                r = s.get(url, headers={"User-Agent": UA}, timeout=30)
            except Exception as exc:
                # Screener resets the connection when a long run goes too fast;
                # back off hard and start a fresh session rather than hammer it.
                print(f"  {sym}: error {exc}; backing off {30 * (attempt + 1)}s", flush=True)
                time.sleep(30 * (attempt + 1))
                s = requests.Session()
                continue
            if r.status_code == 429:
                print(f"  {sym}: 429 rate-limited; backing off 60s")
                time.sleep(60)
                continue
            if r.status_code == 200:
                # Only cache a real company page. A captcha / block / error page also
                # returns 200; caching it would pin that stock as unclassified forever.
                if not _NAME_RE.search(r.text):
                    bad_in_a_row += 1
                    print(f"  {sym}: 200 but not a company page — not cached", flush=True)
                    if bad_in_a_row >= MAX_BAD_IN_A_ROW:
                        print(f"[fetch] {bad_in_a_row} non-company pages in a row — "
                              "source is likely blocking; stopping this run", flush=True)
                        return
                    break
                bad_in_a_row = 0
                if path.exists() and path.stat().st_size > 64:
                    old = _labels(path.read_text(encoding="utf-8", errors="ignore"))
                    new_l = _labels(r.text)
                    if old != new_l and all(new_l):
                        print(f"  RECLASSIFIED {sym}: {' > '.join(old)}  →  {' > '.join(new_l)}",
                              flush=True)
                path.write_text(r.text, encoding="utf-8")
            elif r.status_code == 404:
                if not (path.exists() and path.stat().st_size > 64):
                    path.write_text(NOT_FOUND, encoding="utf-8")
                else:
                    print(f"  {sym}: 404 now — keeping the previously cached page", flush=True)
            else:
                print(f"  {sym}: HTTP {r.status_code}")
            break
        if i % 50 == 0:
            print(f"[fetch] {i}/{len(todo)}", flush=True)
        time.sleep(delay)


def parse(sym: str) -> dict:
    p = _cache_path(sym)
    row = {"symbol": sym, "company_name": "", "macro_sector": "", "sector": "",
           "industry": "", "basic_industry": "", "source": "", "status": "not_fetched",
           "fetched_at": ""}
    if not p.exists():
        return row
    h = p.read_text(encoding="utf-8", errors="ignore")
    if NOT_FOUND in h:
        row["status"] = "not_on_source"
        return row
    m = _NAME_RE.search(h)
    if m:
        row["company_name"] = html.unescape(m.group(1)).strip()
    for label, val in _LEVEL_RE.findall(h):
        col = _LEVELS[label]
        if not row[col]:
            row[col] = html.unescape(val).strip()
    row["fetched_at"] = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
    if row["basic_industry"]:
        row["source"], row["status"] = "screener.in (NSE classification)", "ok"
    elif _ETF_RE.search(row["company_name"]):
        row["source"], row["status"] = "screener.in (fund page)", "etf"
    else:
        row["status"] = "unclassified_on_source"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--universe", choices=["master", "all"], default="all",
                    help="master = sector_master only; all = + every EQ stock traded in 60 sessions")
    ap.add_argument("--max-fetch", type=int, default=None,
                    help="cap page fetches this run (daily incremental use)")
    ap.add_argument("--refresh-days", type=int, default=None,
                    help="also re-fetch cached pages older than this many days")
    args = ap.parse_args()

    syms = _symbols(args.universe)
    non_etf = syms[syms["canonical_sector"].fillna("") != "ETF"]["symbol"].tolist()
    if not args.parse_only:
        fetch(non_etf, args.delay, args.max_fetch, args.refresh_days)

    rows = [parse(s) for s in non_etf]
    df = pd.DataFrame(rows).merge(syms, on="symbol", how="left")
    print(df["status"].value_counts().to_string())
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[parse] wrote {len(df)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
