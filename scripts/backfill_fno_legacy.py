"""Backfill fno_bhavcopy 2022-01 .. 2024-07 from NSE's LEGACY derivatives archive.

WHY THIS EXISTS
    fno_bhavcopy starts 2024-07-24, which is where the UDiFF-format capture began.
    That capped every F&O test in this repo at ~500 sessions - half a market cycle -
    and it is why the Index & Large Cap backtest could not answer the 2022-2026
    question that was actually asked of it.

    The legacy endpoint is still served for the whole period:
        /content/historical/DERIVATIVES/<YYYY>/<MON>/fo<DD><MON><YYYY>bhav.csv.zip
    Verified 200 on 2022-01-10, 2022-06-15, 2022-12-14, 2023-01-11, 2023-03-15,
    2023-09-13, 2024-03-13, 2024-06-12. A 404 on 2023-06-29 was NOT missing data -
    that was Bakri Id, a market holiday. Both formats are served in the overlap
    (2024-03-13 returns 200 on legacy AND UDiFF), so the boundary is not a cliff.

COLUMN MAP (legacy -> schema). The legacy file carries everything the panel needs,
including per-strike option OPEN_INT and CHG_IN_OI:
    INSTRUMENT SYMBOL EXPIRY_DT STRIKE_PR OPTION_TYP OPEN HIGH LOW CLOSE
    SETTLE_PR CONTRACTS VAL_INLAKH OPEN_INT CHG_IN_OI TIMESTAMP

SAFETY
    Writes through repository.upsert_fno_bhavcopy, so the _check_fno_sanity gate
    runs on every session before its DELETE. Existing dates are skipped, so this
    can never overwrite the UDiFF-era capture. Resumable: rerun to continue.
"""
import sys, os, io, time, zipfile, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime as dt
import pandas as pd
import requests

from src.data.repository import get_repository, upsert_fno_bhavcopy

MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT",
       "NOV", "DEC"]
BASE = ("https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
        "{y}/{m}/fo{d:02d}{m}{y}bhav.csv.zip")


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/all-reports-derivatives",
    })
    try:
        s.get("https://www.nseindia.com/", timeout=20)
    except requests.RequestException:
        pass
    return s


def fetch_day(s: requests.Session, d: dt.date, retries: int = 3):
    """-> DataFrame, or None when NSE has no file (holiday / weekend)."""
    url = BASE.format(y=d.year, m=MON[d.month - 1], d=d.day)
    for a in range(retries):
        try:
            r = s.get(url, timeout=45)
        except requests.RequestException:
            time.sleep(2 + 3 * a)
            continue
        if r.status_code == 404:
            return None                     # holiday, not a failure
        if r.status_code == 200 and r.content[:2] == b"PK":
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return pd.read_csv(z.open(z.namelist()[0]))
        time.sleep(2 + 3 * a)
    raise RuntimeError(f"{d}: gave up after {retries} attempts on {url}")


def normalise(raw: pd.DataFrame, d: dt.date) -> pd.DataFrame:
    df = raw.rename(columns={
        "INSTRUMENT": "instrument", "SYMBOL": "symbol", "EXPIRY_DT": "expiry_date",
        "STRIKE_PR": "strike_price", "OPTION_TYP": "option_type",
        "OPEN": "open_price", "HIGH": "high_price", "LOW": "low_price",
        "CLOSE": "close_price", "SETTLE_PR": "settle_price",
        "CONTRACTS": "contracts", "VAL_INLAKH": "value_lacs",
        "OPEN_INT": "open_interest", "CHG_IN_OI": "chg_in_oi"})
    keep = ["instrument", "symbol", "expiry_date", "strike_price", "option_type",
            "open_price", "high_price", "low_price", "close_price", "settle_price",
            "contracts", "value_lacs", "open_interest", "chg_in_oi"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["trade_date"] = d
    df["expiry_date"] = pd.to_datetime(df["expiry_date"],
                                       format="%d-%b-%Y", errors="coerce").dt.date
    # legacy writes XX for futures and so does the UDiFF-era capture already in
    # the table (FUTSTK/FUTIDX are 100% 'XX'), and the column is NOT NULL. Keep it.
    df["option_type"] = df["option_type"].fillna("XX")
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    for c in ("open_price", "high_price", "low_price", "close_price",
              "settle_price", "value_lacs"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("contracts", "open_interest", "chg_in_oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    return df.dropna(subset=["expiry_date", "symbol", "instrument"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2024-07-23")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--batch", type=int, default=20,
                    help="sessions per upsert. The write costs ~20s regardless of "
                         "size, so batching dominates total runtime.")
    a = ap.parse_args()
    start = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end)

    have = set(get_repository().get_distinct_dates("fno_bhavcopy"))
    have = {d if isinstance(d, dt.date) else pd.to_datetime(d).date() for d in have}
    todo = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in have:
            todo.append(d)
        d += dt.timedelta(days=1)
    print(f"already have {len(have)} F&O sessions; {len(todo)} weekdays to try "
          f"from {start} to {end}", flush=True)

    s = session()
    ok = hol = 0
    rows = 0
    t0 = time.time()
    buf: list = []

    def flush() -> None:
        """One upsert per batch. _check_fno_sanity still sees every session, and
        the DELETE is keyed on the dates present, so a batch is as safe as a day."""
        nonlocal buf, ok, rows
        if not buf:
            return
        df = pd.concat(buf, ignore_index=True)
        try:
            upsert_fno_bhavcopy(df)
            ok += df["trade_date"].nunique()
            rows += len(df)
        except Exception as exc:                              # noqa: BLE001
            print(f"  !! batch {df['trade_date'].min()}..{df['trade_date'].max()} "
                  f"rejected, retrying day by day: {exc}", flush=True)
            for d_, g in df.groupby("trade_date"):
                try:
                    upsert_fno_bhavcopy(g)
                    ok += 1
                    rows += len(g)
                except Exception as e2:                       # noqa: BLE001
                    print(f"  !! {d_} rejected: {e2}", flush=True)
        buf = []

    for i, d in enumerate(todo, 1):
        try:
            raw = fetch_day(s, d)
        except RuntimeError as exc:
            print(f"  !! {exc}", flush=True)
            continue
        if raw is None:
            hol += 1
        else:
            buf.append(normalise(raw, d))
            if len(buf) >= a.batch:
                flush()
        if i % 25 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  loaded {ok}  holidays {hol}  "
                  f"{rows:,} rows  {el/i:.2f}s/day  "
                  f"eta {(len(todo)-i)*el/i/60:.0f}m", flush=True)
        time.sleep(a.sleep)
    flush()
    print(f"DONE  loaded {ok} sessions ({rows:,} rows), {hol} holidays, "
          f"{time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
