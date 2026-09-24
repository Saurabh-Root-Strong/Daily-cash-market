"""Backfill NSE bulk deals / block deals / short selling into DuckDB.

    python scripts/backfill_deals.py --from-year 2010 --to-year 2026
    python scripts/backfill_deals.py --load-only        # re-load cached CSVs

Fetch unit is a CALENDAR YEAR (see src/ingestion/deals_fetcher for why the JSON
form is unusable). Raw responses are cached under data/deals_raw/ so a re-load
never re-hits NSE; delete a cached file to force a re-fetch of that year.
"""
import argparse
import io
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.schema import initialize_schema
from src.data.repository import upsert_deals, upsert_short_selling
from src.ingestion.deals_fetcher import DealsFetcher

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "deals_raw")
OPTIONS = ("bulk_deals", "block_deals", "short_selling")


def raw_path(option: str, year: int) -> str:
    return os.path.join(RAW_DIR, f"{option}_{year}.csv")


def download(fetcher: DealsFetcher, option: str, year: int, today: date) -> str:
    path = raw_path(option, year)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    start = date(year, 1, 1)
    end = min(date(year, 12, 31), today)
    text = fetcher.fetch_csv(option, start, end)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path


def parse(option: str, path: str) -> pd.DataFrame:
    """Parse a cached CSV with the same normalisation the fetcher applies."""
    from src.ingestion import deals_fetcher as df_mod
    text = io.open(path, encoding="utf-8-sig").read()
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if df.empty:
        return pd.DataFrame()
    if option == "short_selling":
        out = pd.DataFrame({
            "trade_date": df_mod._to_date(df["Date"]).dt.date,
            "symbol": df["Symbol"].str.strip(),
            "security_name": df["Security Name"].str.strip(),
            "quantity": df_mod._to_int(df["Quantity"]),
        })
        return out.loc[out["trade_date"].notna() & out["quantity"].notna()]
    out = pd.DataFrame({
        "trade_date": df_mod._to_date(df["Date"]).dt.date,
        "deal_type": df_mod.DEAL_TYPES[option],
        "symbol": df["Symbol"].str.strip(),
        "security_name": df["Security Name"].str.strip(),
        "client_name": df["Client Name"].str.strip(),
        "buy_sell": df["Buy / Sell"].str.strip().str.upper(),
        "quantity": df_mod._to_int(df["Quantity Traded"]),
        "trade_price": df_mod._to_float(df["Trade Price / Wght. Avg. Price"]),
        "remarks": df.get("Remarks", pd.Series([""] * len(df))).fillna("").astype(str).str.strip(),
    })
    ok = out["trade_date"].notna() & out["quantity"].notna() & out["trade_price"].notna()
    if (~ok).any():
        print(f"      dropped {int((~ok).sum())} unparseable rows")
    return out.loc[ok]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2010)
    ap.add_argument("--to-year", type=int, default=date.today().year)
    ap.add_argument("--load-only", action="store_true",
                    help="skip the network, load whatever is cached")
    ap.add_argument("--no-load", action="store_true", help="download only")
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    today = date.today()
    fetcher = None if args.load_only else DealsFetcher()

    for year in range(args.from_year, args.to_year + 1):
        for option in OPTIONS:
            path = raw_path(option, year)
            if not args.load_only:
                try:
                    download(fetcher, option, year, today)
                except Exception as exc:
                    print(f"  {option} {year}: FETCH FAILED {exc}")
                    continue
            if not os.path.exists(path):
                continue
            if args.no_load:
                print(f"  {option} {year}: cached {os.path.getsize(path):,} bytes")
                continue
            frame = parse(option, path)
            if frame.empty:
                print(f"  {option} {year}: 0 rows")
                continue
            n = (upsert_short_selling(frame) if option == "short_selling"
                 else upsert_deals(frame))
            print(f"  {option} {year}: {n:,} rows, "
                  f"{frame['trade_date'].nunique()} sessions")


if __name__ == "__main__":
    if "--no-load" not in sys.argv:
        initialize_schema()
    main()
