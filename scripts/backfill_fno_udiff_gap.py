"""Fill the 13 sessions the legacy archive does not cover, from the UDiFF endpoint.

The legacy derivatives archive stops at 2024-07-05 and the UDiFF-format capture in
this table starts 2024-07-24, leaving 2024-07-08..07-23 uncovered. 2024-01-20 and
2024-05-18 are special Saturday sessions the legacy path also missed.
"""
import sys, io, time, zipfile
sys.path.insert(0, "d:/Python Projects/Daily_Cash_Market")
import datetime as dt, pandas as pd, requests
from src.data.repository import upsert_fno_bhavcopy

MISS = ['2024-01-20', '2024-05-18', '2024-07-08', '2024-07-09', '2024-07-10',
        '2024-07-11', '2024-07-12', '2024-07-15', '2024-07-16', '2024-07-18',
        '2024-07-19', '2024-07-22', '2024-07-23']
s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                  "Referer": "https://www.nseindia.com/all-reports-derivatives"})
s.get("https://www.nseindia.com/", timeout=20)
buf = []
for ds in MISS:
    d = dt.date.fromisoformat(ds)
    u = ("https://nsearchives.nseindia.com/content/fo/"
         f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")
    r = s.get(u, timeout=45)
    if r.status_code != 200 or r.content[:2] != b"PK":
        print(f"  {ds}: HTTP {r.status_code}, skipped")
        continue
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = pd.read_csv(z.open(z.namelist()[0]))
    df = raw.rename(columns={
        "FinInstrmTp": "instrument", "TckrSymb": "symbol", "XpryDt": "expiry_date",
        "StrkPric": "strike_price", "OptnTp": "option_type", "OpnPric": "open_price",
        "HghPric": "high_price", "LwPric": "low_price", "ClsPric": "close_price",
        "SttlmPric": "settle_price", "TtlTradgVol": "contracts",
        "TtlTrfVal": "value_lacs", "OpnIntrst": "open_interest",
        "ChngInOpnIntrst": "chg_in_oi"})
    df = df[df["instrument"].isin(["STF", "IDF", "STO", "IDO"])].copy()
    df["instrument"] = df["instrument"].map(
        {"STF": "FUTSTK", "IDF": "FUTIDX", "STO": "OPTSTK", "IDO": "OPTIDX"})
    df["trade_date"] = d
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
    df["option_type"] = df["option_type"].fillna("XX")
    df.loc[~df["option_type"].isin(["CE", "PE"]), "option_type"] = "XX"
    # TtlTrfVal is in rupees in UDiFF; the schema stores lakhs, as the legacy
    # VAL_INLAKH column already did for every backfilled row.
    df["value_lacs"] = pd.to_numeric(df["value_lacs"], errors="coerce") / 100000.0
    for c in ("open_price", "high_price", "low_price", "close_price", "settle_price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # UDiFF leaves StrkPric blank on futures rows; the column is NOT NULL and every
    # FUTSTK/FUTIDX row already in the table stores 0.0.
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce").fillna(0.0)
    for c in ("contracts", "open_interest", "chg_in_oi"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    keep = ["trade_date", "instrument", "symbol", "expiry_date", "strike_price",
            "option_type", "open_price", "high_price", "low_price", "close_price",
            "settle_price", "contracts", "value_lacs", "open_interest", "chg_in_oi"]
    buf.append(df[keep].dropna(subset=["expiry_date"]))
    print(f"  {ds}: {len(df):,} rows")
    time.sleep(0.4)
if buf:
    all_ = pd.concat(buf, ignore_index=True)
    upsert_fno_bhavcopy(all_)
    print(f"upserted {len(all_):,} rows across {all_['trade_date'].nunique()} sessions")
