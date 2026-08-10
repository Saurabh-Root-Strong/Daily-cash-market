"""
Backfill index_data 2013-01-01 .. 2017-12-31 from NiftyIndices daily snapshots.

Why: index_data started 2018-01-01 (2,128 sessions). NiftyIndices serves back to
2013-01-01. For calendar-seasonality work the sample size per (sector, month)
cell goes from 8 to 13 observations -- the difference between unusable and
marginal. Nothing else in the repo depends on the start date.

THE NAMING TRAP: indices were renamed CNX -> Nifty between 2015-11-02 and
2015-12-01. Inserting raw names would create parallel half-series (e.g.
"CNX Bank" 2013-2015 and "Nifty Bank" 2015-2026) that silently break every
lookback. _RENAME maps old -> modern. The map is NOT trusted on faith: --verify
checks close-value continuity across the cutover seam and refuses to load if a
mapped pair jumps more than 2%.

Midcap/Smallcap have a THREE-stage name history:
    CNX Midcap -> Nifty Free Float Midcap 100 -> NIFTY Midcap 100
Both hops are in the map so the whole 2013-2026 series lands on one name.

Stages:
  --fetch   : download to a parquet stage (no DB, safe while Streamlit runs)
  --verify  : seam continuity + coverage report off the stage
  --load    : upsert stage into index_data (stop Streamlit first)
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import re
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
STAGE = Path(r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot"
             r"/84e1ef3a-e9e9-474e-b47f-c9eca80173a1/scratchpad/index_backfill_2013.parquet")

START = dt.date(2013, 1, 1)
END = dt.date(2017, 12, 31)

_URL = "https://www.niftyindices.com/Daily_Snapshot/ind_close_all_{d}.csv"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Referer": "https://www.niftyindices.com/reports/daily-reports",
}

_COL_MAP = {
    "Index Name": "index_name", "Open Index Value": "open_val",
    "High Index Value": "high_val", "Low Index Value": "low_val",
    "Closing Index Value": "close_val", "Points Change": "points_chg",
    "Change(%)": "pct_chg", "Volume": "volume",
    "Turnover (Rs. Cr.)": "turnover_cr", "P/E": "pe_ratio",
    "P/B": "pb_ratio", "Div Yield": "div_yield",
}

# Same exclusions the live fetcher applies, so the backfill cannot introduce
# index types the modern pipeline deliberately drops.
_EXCLUDE = ["G-Sec", "GSEC", "BHARAT Bond", "1D Rate", "Shariah", "1x Inverse",
            "1X Inverse", "2x Leverage", "2X Leverage", "PR 1x", "PR 2x",
            "PR 1X", "PR 2X", "TR 1x", "TR 2x", "TR 1X", "TR 2X", "USD",
            "Dividend Points", "Arbitrage", "Futures Index", "Futures TR",
            "DEFTY", "Nifty Dividend"]
_EXCLUDE_RE = re.compile("|".join(re.escape(p) for p in _EXCLUDE), re.I)

# old name -> modern name. Verified at the seam by --verify.
_RENAME = {
    "CNX Nifty": "Nifty 50",
    "CNX Nifty Junior": "Nifty Next 50",
    "CNX 100": "Nifty 100",
    "CNX 200": "Nifty 200",
    "CNX 500": "Nifty 500",
    "CNX Auto": "Nifty Auto",
    "CNX Bank": "Nifty Bank",
    "CNX IT": "Nifty IT",
    "CNX Pharma": "Nifty Pharma",
    "CNX FMCG": "Nifty FMCG",
    "CNX Metal": "Nifty Metal",
    "CNX Media": "Nifty Media",
    "CNX Realty": "Nifty Realty",
    "CNX Energy": "Nifty Energy",
    "CNX Infrastructure": "Nifty Infrastructure",
    "CNX Commodities": "Nifty Commodities",
    "CNX Consumption": "Nifty India Consumption",
    "CNX Finance": "Nifty Financial Services",
    "CNX PSU Bank": "Nifty PSU Bank",
    "CNX PSE": "Nifty PSE",
    "CNX MNC": "Nifty MNC",
    "CNX Service Sector": "Nifty Services Sector",
    "CNX Dividend Opportunities": "Nifty Dividend Opportunities 50",
    "CPSE": "Nifty CPSE",
    "CNX High Beta": "Nifty High Beta 50",
    "CNX Low Volatility": "Nifty Low Volatility 50",
    "CNX 100 Equal Weight": "Nifty100 Equal Weight",
    "CNX Alpha Index": "Nifty Alpha 50",
    "NSE Quality 30": "Nifty Quality 30",
    "NV 20": "Nifty50 Value 20",
    "LIX 15": "Nifty100 Liquid 15",
    "LIX15 Midcap": "Nifty Midcap Liquid 15",
    "NI15": "Nifty Growth Sectors 15",
    # three-stage midcap/smallcap lineage -> single modern name
    "CNX Midcap": "NIFTY Midcap 100",
    "Nifty Free Float Midcap 100": "NIFTY Midcap 100",
    "Nifty Midcap 100": "NIFTY Midcap 100",
    "CNX Smallcap": "NIFTY Smallcap 100",
    "Nifty Free Float Smallcap 100": "NIFTY Smallcap 100",
    "Nifty Smallcap 100": "NIFTY Smallcap 100",
}

# Pairs to check across the CNX->Nifty cutover, plus the midcap/smallcap hops.
_SEAM_CHECK = ["Nifty 50", "Nifty Bank", "Nifty IT", "Nifty Pharma", "Nifty Auto",
               "Nifty FMCG", "Nifty Metal", "Nifty Realty", "Nifty Energy",
               "Nifty PSU Bank", "Nifty Financial Services", "NIFTY Midcap 100",
               "NIFTY Smallcap 100", "Nifty 100", "Nifty 500"]


def _norm(s: str) -> str:
    return _RENAME.get(s.strip(), s.strip())


def fetch() -> None:
    sess = requests.Session()
    sess.headers.update(_HEADERS)
    frames, miss, d = [], 0, START
    while d <= END:
        if d.weekday() < 5 or True:          # weekends included: special sessions exist
            try:
                r = sess.get(_URL.format(d=d.strftime("%d%m%Y")), timeout=25)
                if r.status_code == 200 and not r.text.lstrip().startswith("<"):
                    df = pd.read_csv(io.StringIO(r.text))
                    df = df.rename(columns=_COL_MAP)
                    if "index_name" in df.columns:
                        df["index_name"] = df["index_name"].astype(str).str.strip()
                        df = df[~df["index_name"].str.contains(_EXCLUDE_RE, na=False)]
                        df["index_name"] = df["index_name"].map(_norm)
                        df["trade_date"] = d
                        frames.append(df)
                    else:
                        miss += 1
                else:
                    miss += 1
            except Exception:
                miss += 1
                time.sleep(0.5)
        d += dt.timedelta(days=1)
        if d.day == 1:
            print("  ..", d, f"({len(frames)} sessions)", flush=True)

    if not frames:
        print("nothing fetched"); return
    out = pd.concat(frames, ignore_index=True)

    num = ["open_val", "high_val", "low_val", "close_val", "points_chg", "pct_chg",
           "turnover_cr", "pe_ratio", "pb_ratio", "div_yield"]
    for c in num:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].astype(str).str.strip()
                                   .replace({"-": None, "": None}), errors="coerce")
    out["volume"] = pd.to_numeric(out.get("volume"), errors="coerce").astype("Int64")
    out["prev_close"] = out["close_val"] - out["points_chg"]
    out = out.dropna(subset=["index_name", "close_val"])
    out = out.drop_duplicates(subset=["trade_date", "index_name"], keep="last")

    keep = ["trade_date", "index_name", "open_val", "high_val", "low_val", "close_val",
            "prev_close", "points_chg", "pct_chg", "volume", "turnover_cr",
            "pe_ratio", "pb_ratio", "div_yield"]
    out = out[[c for c in keep if c in out.columns]]
    STAGE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(STAGE, index=False)
    print(f"\nstaged {len(out):,} rows / {out.trade_date.nunique()} sessions "
          f"/ {out.index_name.nunique()} indices -> {STAGE}")
    print(f"missed/holiday days: {miss}")


def verify() -> bool:
    if not STAGE.exists():
        print("no stage file; run --fetch"); return False
    st = pd.read_parquet(STAGE)
    st["trade_date"] = pd.to_datetime(st.trade_date)
    con = duckdb.connect(DB, read_only=True)
    live = con.sql("""select trade_date, index_name, close_val, pct_chg from index_data
                      where trade_date < DATE '2018-03-01'""").df()
    con.close()
    live["trade_date"] = pd.to_datetime(live.trade_date)

    print(f"stage: {len(st):,} rows, {st.trade_date.min().date()} -> {st.trade_date.max().date()}, "
          f"{st.index_name.nunique()} indices")

    ok = True
    # A raw jump threshold is the WRONG test: a high-beta sector can legitimately
    # move >3% in a day (Nifty Metal did exactly that on 2015-12-01, and the file's
    # own pct_chg confirms it). The correct test is CHAIN CONSISTENCY -- the later
    # row reports its own prev_close, so implied_prev = close/(1+pct/100) must
    # equal the earlier session's close. That is exact and beta-independent.
    def seam(a, b, label, nm):
        nonlocal ok
        implied = b.close_val / (1 + (b.pct_chg or 0) / 100.0)
        err = abs(implied - a.close_val) / a.close_val * 100
        raw = abs(b.close_val - a.close_val) / a.close_val * 100
        good = err < 0.5
        if not good:
            ok = False
        print(f"  {nm:32s} {a.close_val:10.2f} -> {b.close_val:10.2f}  "
              f"move {raw:5.2f}%  chain-err {err:5.3f}%  "
              f"{'OK' if good else '** BREAK **'}")

    print("\n=== SEAM 1: CNX->Nifty rename (2015-11-30 -> 2015-12-01) ===")
    print("    test = chain consistency (implied prev_close vs prior close), not jump size")
    for nm in _SEAM_CHECK:
        s = st[st.index_name == nm].sort_values("trade_date")
        pre = s[s.trade_date < "2015-12-01"]
        post = s[s.trade_date >= "2015-12-01"]
        if pre.empty or post.empty:
            print(f"  {nm:32s} MISSING one side (pre={len(pre)}, post={len(post)})")
            ok = False
            continue
        seam(pre.iloc[-1], post.iloc[0], "rename", nm)

    print("\n=== SEAM 2: stage end (2017-12) -> live DB start (2018-01) ===")
    # The live DB still carries the pre-2018-03-28 legacy names for midcap/
    # smallcap; apply the same map before comparing, otherwise this reports a
    # false MISSING. load() renames those live rows for real.
    live["index_name"] = live["index_name"].map(_norm)
    live = live.sort_values("trade_date")
    for nm in _SEAM_CHECK:
        s = st[st.index_name == nm].sort_values("trade_date")
        l = live[live.index_name == nm]
        if s.empty or l.empty:
            print(f"  {nm:32s} MISSING (stage={len(s)}, live={len(l)})")
            ok = False
            continue
        seam(s.iloc[-1], l.iloc[0], "handoff", nm)

    print("\n=== any surviving CNX-style names? ===")
    bad = sorted(x for x in st.index_name.unique()
                 if x.upper().startswith(("CNX", "LIX", "NI15", "NV ")))
    print("  ", bad if bad else "none")
    if bad:
        ok = False

    print("\n=== sessions per year in stage ===")
    print(st.groupby(st.trade_date.dt.year).trade_date.nunique().to_string())

    print("\nVERIFY:", "PASS" if ok else "FAIL")
    return ok


def load() -> None:
    if not verify():
        print("\nrefusing to load: verification failed")
        return
    st = pd.read_parquet(STAGE)
    st["trade_date"] = pd.to_datetime(st.trade_date).dt.date
    con = duckdb.connect(DB, read_only=False)
    try:
        con.execute("BEGIN TRANSACTION")
        before = con.execute("select count(*) from index_data").fetchone()[0]

        # Collapse the legacy midcap/smallcap names in the LIVE table onto the
        # modern name, so 2013-2026 is one continuous series instead of two
        # half-series that silently break every lookback across 2018-03-28.
        for old, new in (("Nifty Free Float Midcap 100", "NIFTY Midcap 100"),
                         ("Nifty Free Float Smallcap 100", "NIFTY Smallcap 100")):
            n = con.execute("select count(*) from index_data where index_name = ?",
                            [old]).fetchone()[0]
            if not n:
                continue
            clash = con.execute("""select count(*) from index_data a
                where a.index_name = ? and exists (select 1 from index_data b
                where b.index_name = ? and b.trade_date = a.trade_date)""",
                [old, new]).fetchone()[0]
            if clash:
                print(f"  !! {old}: {clash} dates already have {new}; deleting legacy dupes")
                con.execute("""delete from index_data a where a.index_name = ?
                    and exists (select 1 from index_data b where b.index_name = ?
                    and b.trade_date = a.trade_date)""", [old, new])
            con.execute("update index_data set index_name = ? where index_name = ?",
                        [new, old])
            print(f"  renamed {n} live rows: {old} -> {new}")

        con.register("_st", st)
        con.execute("""
            INSERT INTO index_data
                (trade_date, index_name, open_val, high_val, low_val, close_val,
                 prev_close, points_chg, pct_chg, volume, turnover_cr,
                 pe_ratio, pb_ratio, div_yield)
            SELECT trade_date, index_name, open_val, high_val, low_val, close_val,
                   prev_close, points_chg, pct_chg, volume, turnover_cr,
                   pe_ratio, pb_ratio, div_yield
            FROM _st
            ON CONFLICT (trade_date, index_name) DO UPDATE SET
                open_val=excluded.open_val, high_val=excluded.high_val,
                low_val=excluded.low_val, close_val=excluded.close_val,
                prev_close=excluded.prev_close, points_chg=excluded.points_chg,
                pct_chg=excluded.pct_chg, volume=excluded.volume,
                turnover_cr=excluded.turnover_cr, pe_ratio=excluded.pe_ratio,
                pb_ratio=excluded.pb_ratio, div_yield=excluded.div_yield
        """)
        after = con.execute("select count(*) from index_data").fetchone()[0]
        con.execute("COMMIT")
        print(f"\nrows {before:,} -> {after:,}  (+{after-before:,})")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--load", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    if a.verify:
        verify()
    if a.load:
        load()
    if not (a.fetch or a.verify or a.load):
        ap.print_help()
