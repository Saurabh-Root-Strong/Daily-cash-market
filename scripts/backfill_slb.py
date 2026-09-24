"""Backfill NSE SLB bhavcopy + open positions into DuckDB.

    python scripts/backfill_slb.py --start 2017-01-01            # download + load
    python scripts/backfill_slb.py --load-only                   # re-load the cache

Raw files are cached under data/slb_raw/{bc,op}/ so a re-load never re-hits NSE.
A 404 (holiday / weekend) is cached as an empty ".none" marker so it is not
re-requested either; delete the marker to force a retry.
"""
import argparse
import os
import sys
import time
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.ingestion.slb_fetcher import parse_bhavcopy, parse_open_positions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "slb_raw")


def _path(kind: str, d: date) -> str:
    return os.path.join(RAW, kind, f"{d:%Y%m%d}.txt")


def download(start: date, end: date, workers: int = 6) -> None:
    """Parallel archive pull. nsearchives serves these files without the cookie
    priming the www API needs, so plain per-thread sessions are enough; the
    NSEHttpClient path costs ~2 s a file and would take hours for 9 years."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import requests

    from src.ingestion.slb_fetcher import _BHAV_URL, _OPEN_URL
    urls = {"bc": _BHAV_URL, "op": _OPEN_URL}
    for kind in urls:
        os.makedirs(os.path.join(RAW, kind), exist_ok=True)
    local = threading.local()

    def sess() -> "requests.Session":
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers.update({"User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")})
        return local.s

    def job(kind: str, d: date) -> str:
        p = _path(kind, d)
        for attempt in range(4):
            try:
                r = sess().get(urls[kind].format(d=d.strftime("%d%m%Y")), timeout=30)
                if r.status_code == 404:
                    open(p + ".none", "w").close()
                    return "404"
                r.raise_for_status()
                with open(p, "w", encoding="utf-8", newline="") as fh:
                    fh.write(r.text)
                return "ok"
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return "fail"

    todo = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            for kind in urls:
                p = _path(kind, d)
                if not (os.path.exists(p) or os.path.exists(p + ".none")):
                    todo.append((kind, d))
        d += timedelta(days=1)
    print(f"{len(todo)} files to fetch", flush=True)
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(lambda kd: job(*kd), todo), 1):
            counts[res] = counts.get(res, 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(todo)} {counts}", flush=True)
    print(f"download done: {counts}", flush=True)


def load() -> None:
    from src.data.schema import initialize_schema
    from src.data.repository import upsert_slb

    bhav, opos = [], []
    for name in sorted(os.listdir(os.path.join(RAW, "op"))):
        if not name.endswith(".txt"):
            continue
        d = date(int(name[:4]), int(name[4:6]), int(name[6:8]))
        opos.append(parse_open_positions(open(os.path.join(RAW, "op", name), encoding="utf-8").read(), d))
        bp = _path("bc", d)
        if os.path.exists(bp):
            b = parse_bhavcopy(open(bp, encoding="utf-8").read())
            if not b.empty:
                # The DAT carries its own date column; it must agree with the file name.
                bad = b["trade_date"] != d
                if bad.any():
                    print(f"  {d}: {int(bad.sum())} bhavcopy rows dated {b.loc[bad, 'trade_date'].unique()}")
                    b = b.loc[~bad]
                bhav.append(b)
    bhav_df = pd.concat(bhav, ignore_index=True) if bhav else pd.DataFrame()
    opos_df = pd.concat(opos, ignore_index=True) if opos else pd.DataFrame()
    print(f"parsed: bhavcopy {len(bhav_df):,} rows, open positions {len(opos_df):,} rows, "
          f"{opos_df['trade_date'].nunique() if len(opos_df) else 0} sessions", flush=True)
    initialize_schema()
    print("loaded:", upsert_slb(bhav_df, opos_df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()
    if not a.load_only:
        end = date.fromisoformat(a.end) if a.end else date.today()
        download(date.fromisoformat(a.start), end)
    if not a.download_only:
        load()


if __name__ == "__main__":
    main()
