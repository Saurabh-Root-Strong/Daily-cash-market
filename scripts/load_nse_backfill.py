"""
Load staged NSE backfill (sec_bhavdata_full -> daily_data, ind_close_all -> index_data).

--dry-run (default): parse + validate ALL staged CSVs, report counts/ranges/sanity. NO DB.
--load            : INSERT the parsed rows into data/market_data.duckdb (additive, only
                    trade_date < 2024-12-26 so it never overlaps existing data). Requires
                    the Streamlit app to be stopped (DuckDB single-writer).

Authoritative trade_date = the DATE inside each file (not the filename), deduped on
(symbol, series, trade_date) so any holiday-served-stale files self-correct.
"""
from __future__ import annotations
import sys, os, glob
import numpy as np, pandas as pd

STAGE = r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/00fb1329-476b-4454-94e7-cefc7a5aa9d8/scratchpad/nse_backfill"
DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
CUTOFF = pd.Timestamp("2024-12-26")           # existing data starts here → stay strictly below

_DAILY_COLS = ["trade_date","symbol","series","prev_close","open_price","high_price",
               "low_price","last_price","close_price","avg_price","ttl_trd_qnty",
               "turnover_lacs","no_of_trades","deliv_qty","deliv_per"]

def _num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"-": np.nan, "": np.nan}), errors="coerce")

def parse_bhav() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(STAGE, "bhav", "*.csv"))):
        try:
            d = pd.read_csv(f, dtype=str)
        except Exception:
            continue
        d.columns = [c.strip().upper() for c in d.columns]
        if "SYMBOL" not in d.columns or "DELIV_PER" not in d.columns:
            continue
        out = pd.DataFrame({
            "trade_date": pd.to_datetime(d["DATE1"].str.strip(), format="%d-%b-%Y", errors="coerce"),
            "symbol": d["SYMBOL"].str.strip(),
            "series": d["SERIES"].str.strip(),
            "prev_close": _num(d["PREV_CLOSE"]), "open_price": _num(d["OPEN_PRICE"]),
            "high_price": _num(d["HIGH_PRICE"]), "low_price": _num(d["LOW_PRICE"]),
            "last_price": _num(d["LAST_PRICE"]), "close_price": _num(d["CLOSE_PRICE"]),
            "avg_price": _num(d["AVG_PRICE"]), "ttl_trd_qnty": _num(d["TTL_TRD_QNTY"]),
            "turnover_lacs": _num(d["TURNOVER_LACS"]), "no_of_trades": _num(d["NO_OF_TRADES"]),
            "deliv_qty": _num(d["DELIV_QTY"]), "deliv_per": _num(d["DELIV_PER"]),
        })
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["trade_date"].notna() & (df["trade_date"] < CUTOFF)]
    df = df.drop_duplicates(["symbol", "series", "trade_date"])
    for c in ("ttl_trd_qnty", "no_of_trades", "deliv_qty"):
        df[c] = df[c].round().astype("Int64")
    return df[_DAILY_COLS]

_IDX_COLS = ["trade_date","index_name","open_val","high_val","low_val","close_val",
             "prev_close","points_chg","pct_chg","volume","turnover_cr","pe_ratio",
             "pb_ratio","div_yield"]

def parse_index() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(STAGE, "index", "*.csv"))):
        try:
            d = pd.read_csv(f, dtype=str)
        except Exception:
            continue
        d.columns = [c.strip() for c in d.columns]
        if "Index Name" not in d.columns:
            continue
        cl = _num(d["Closing Index Value"]); pts = _num(d["Points Change"])
        out = pd.DataFrame({
            "trade_date": pd.to_datetime(d["Index Date"].str.strip(), format="%d-%m-%Y", errors="coerce"),
            "index_name": d["Index Name"].str.strip(),
            "open_val": _num(d["Open Index Value"]), "high_val": _num(d["High Index Value"]),
            "low_val": _num(d["Low Index Value"]), "close_val": cl,
            "prev_close": cl - pts, "points_chg": pts, "pct_chg": _num(d["Change(%)"]),
            "volume": _num(d["Volume"]), "turnover_cr": _num(d["Turnover (Rs. Cr.)"]),
            "pe_ratio": _num(d["P/E"]), "pb_ratio": _num(d["P/B"]), "div_yield": _num(d["Div Yield"]),
        })
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["trade_date"].notna() & (df["trade_date"] < CUTOFF)]
    df = df.drop_duplicates(["index_name", "trade_date"])
    df["volume"] = df["volume"].round().astype("Int64")
    return df[_IDX_COLS]

def report(name, df, keycol):
    print(f"\n{name}: {len(df):,} rows | dates {df['trade_date'].min().date()} -> "
          f"{df['trade_date'].max().date()} ({df['trade_date'].nunique()} days) | "
          f"{df[keycol].nunique()} distinct {keycol}")
    if name == "daily_data":
        dp = df["deliv_per"].dropna()
        print(f"  deliv_per: {dp.min():.1f}..{dp.max():.1f} (mean {dp.mean():.1f}) | "
              f"null close {df['close_price'].isna().mean()*100:.1f}% | "
              f"EQ rows {int((df['series']=='EQ').sum()):,}")
    print(f"  sample: {df.iloc[0][keycol]} {df.iloc[0]['trade_date'].date()} "
          f"close={df.iloc[0].get('close_price', df.iloc[0].get('close_val'))}")

if __name__ == "__main__":
    print("parsing staged bhav files (delivery)..."); daily = parse_bhav()
    print("parsing staged index files...");           idx = parse_index()
    report("daily_data", daily, "symbol")
    report("index_data", idx, "index_name")

    if "--load" in sys.argv:
        import duckdb
        print(f"\nLOADING into {DB} (additive, trade_date < {CUTOFF.date()}) ...")
        con = duckdb.connect(DB)
        pre = con.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]
        con.execute("INSERT INTO daily_data SELECT * FROM daily ")  # column order matches schema
        con.execute("INSERT INTO index_data SELECT * FROM idx ")
        post = con.execute("SELECT COUNT(*) FROM daily_data").fetchone()[0]
        span = con.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily_data").fetchone()
        con.close()
        print(f"daily_data {pre:,} -> {post:,} rows | span {span[0]} -> {span[1]}  DONE")
    else:
        print("\n[dry-run] parsed + validated only. Re-run with --load (after stopping "
              "Streamlit) to write to the DB.")
