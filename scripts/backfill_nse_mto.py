"""
Pre-Oct-2019 delivery backfill via the OLD two-file format (cm-bhavcopy price + MTO
delivery), for the window before sec_bhavdata_full exists. Default Jan 2018 -> Sep 2019.

DOWNLOAD (default): stage cm{DD}{MON}{YYYY}bhav.csv.zip + MTO_{DDMMYYYY}.DAT +
ind_close_all_{DDMMYYYY}.csv. --load: parse, JOIN price+delivery on (symbol,series),
and INSERT only rows with trade_date < current DB min (guarantees no overlap).
"""
from __future__ import annotations
import sys, os, io, zipfile, glob, time, urllib.request, datetime as dt
import numpy as np, pandas as pd

STAGE = r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/00fb1329-476b-4454-94e7-cefc7a5aa9d8/scratchpad/nse_backfill"
CM = os.path.join(STAGE,"cmbhav"); MTO = os.path.join(STAGE,"mto"); IDXO = os.path.join(STAGE,"idx_old")
DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
MON = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
START = dt.date(2018,1,1); END = dt.date(2019,9,30)

def _get(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 400: return "skip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":UA,"Referer":"https://www.nseindia.com/all-reports"})
        with urllib.request.urlopen(req, timeout=30) as r: data = r.read()
        if len(data) < 400 or data[:9]==b"<!DOCTYPE": return "miss"
        open(dest,"wb").write(data); return "ok"
    except urllib.error.HTTPError as e: return "miss" if e.code==404 else "err"
    except Exception: return "err"

def download():
    for p in (CM,MTO,IDXO): os.makedirs(p, exist_ok=True)
    d=START; ok=miss=0; t0=time.time()
    while d<=END:
        if d.weekday()<5:
            ds=d.strftime("%d%m%Y"); cmn=f"cm{d.strftime('%d')}{MON[d.month-1]}{d.year}bhav.csv.zip"
            r1=_get(f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{d.year}/{MON[d.month-1]}/{cmn}", os.path.join(CM,f"{d.isoformat()}.zip"))
            r2=_get(f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{ds}.DAT", os.path.join(MTO,f"{d.isoformat()}.dat"))
            r3=_get(f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{ds}.csv", os.path.join(IDXO,f"{d.isoformat()}.csv"))
            ok += r1=="ok"; miss += r1=="miss"
            if r1 in ("ok","err"): time.sleep(0.15)
        if d.day==1: print(f"  {d.isoformat()} cm_ok={ok} miss={miss} {time.time()-t0:.0f}s", flush=True)
        d+=dt.timedelta(days=1)
    print(f"DOWNLOAD DONE cm_ok={ok} miss={miss} in {time.time()-t0:.0f}s", flush=True)

def _num(s): return pd.to_numeric(s.astype(str).str.strip().replace({"-":np.nan,"":np.nan}), errors="coerce")

def parse_combine():
    # delivery (MTO): record-type-20 rows = 20,srno,symbol,series,qty,deliv_qty,deliv_per
    deliv=[]
    for f in sorted(glob.glob(os.path.join(MTO,"*.dat"))):
        rows=[l.split(",") for l in open(f).read().splitlines() if l.startswith("20,")]
        if not rows: continue
        m=pd.DataFrame([r for r in rows if len(r)>=7]).iloc[:,:7]
        m.columns=["rt","sr","symbol","series","qty","dq","dp"]
        m["trade_date"]=pd.to_datetime(os.path.basename(f)[:10])
        deliv.append(m[["trade_date","symbol","series","dq","dp"]])
    D=pd.concat(deliv, ignore_index=True)
    D["symbol"]=D["symbol"].str.strip(); D["series"]=D["series"].str.strip()
    D["deliv_qty"]=_num(D["dq"]).round().astype("Int64"); D["deliv_per"]=_num(D["dp"])
    # price (cm bhav zip)
    px=[]
    for f in sorted(glob.glob(os.path.join(CM,"*.zip"))):
        try: z=zipfile.ZipFile(f); c=pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype=str)
        except Exception: continue
        c.columns=[x.strip().upper() for x in c.columns]
        px.append(pd.DataFrame({
            "trade_date":pd.to_datetime(c["TIMESTAMP"].str.strip(), format="%d-%b-%Y", errors="coerce"),
            "symbol":c["SYMBOL"].str.strip(), "series":c["SERIES"].str.strip(),
            "prev_close":_num(c["PREVCLOSE"]),"open_price":_num(c["OPEN"]),"high_price":_num(c["HIGH"]),
            "low_price":_num(c["LOW"]),"last_price":_num(c["LAST"]),"close_price":_num(c["CLOSE"]),
            "ttl_trd_qnty":_num(c["TOTTRDQTY"]),"tottrdval":_num(c["TOTTRDVAL"]),"no_of_trades":_num(c["TOTALTRADES"])}))
    P=pd.concat(px, ignore_index=True)
    m=P.merge(D[["trade_date","symbol","series","deliv_qty","deliv_per"]], on=["trade_date","symbol","series"], how="left")
    m["avg_price"]=(m["tottrdval"]/m["ttl_trd_qnty"]).round(2)
    m["turnover_lacs"]=(m["tottrdval"]/1e5).round(2)
    m["ttl_trd_qnty"]=m["ttl_trd_qnty"].round().astype("Int64"); m["no_of_trades"]=m["no_of_trades"].round().astype("Int64")
    cols=["trade_date","symbol","series","prev_close","open_price","high_price","low_price",
          "last_price","close_price","avg_price","ttl_trd_qnty","turnover_lacs","no_of_trades","deliv_qty","deliv_per"]
    return m[m["trade_date"].notna()].drop_duplicates(["symbol","series","trade_date"])[cols]

def parse_index_old():
    fr=[]
    for f in sorted(glob.glob(os.path.join(IDXO,"*.csv"))):
        try: d=pd.read_csv(f, dtype=str)
        except Exception: continue
        d.columns=[x.strip() for x in d.columns]
        if "Index Name" not in d.columns: continue
        cl=_num(d["Closing Index Value"]); pts=_num(d["Points Change"])
        fr.append(pd.DataFrame({"trade_date":pd.to_datetime(d["Index Date"].str.strip(),format="%d-%m-%Y",errors="coerce"),
            "index_name":d["Index Name"].str.strip(),"open_val":_num(d["Open Index Value"]),"high_val":_num(d["High Index Value"]),
            "low_val":_num(d["Low Index Value"]),"close_val":cl,"prev_close":cl-pts,"points_chg":pts,"pct_chg":_num(d["Change(%)"]),
            "volume":_num(d["Volume"]).round().astype("Int64"),"turnover_cr":_num(d["Turnover (Rs. Cr.)"]),
            "pe_ratio":_num(d["P/E"]),"pb_ratio":_num(d["P/B"]),"div_yield":_num(d["Div Yield"])}))
    ic=["trade_date","index_name","open_val","high_val","low_val","close_val","prev_close","points_chg","pct_chg","volume","turnover_cr","pe_ratio","pb_ratio","div_yield"]
    return pd.concat(fr, ignore_index=True).drop_duplicates(["index_name","trade_date"])[ic]

if __name__=="__main__":
    if "--load" in sys.argv:
        import duckdb
        daily=parse_combine(); idx=parse_index_old()
        con=duckdb.connect(DB)
        dmin=con.execute("SELECT MIN(trade_date) FROM daily_data").fetchone()[0]
        imin=con.execute("SELECT MIN(trade_date) FROM index_data").fetchone()[0]
        daily=daily[daily["trade_date"]<pd.Timestamp(dmin)].dropna(subset=["symbol","series","trade_date","close_price"])
        idx=idx[idx["trade_date"]<pd.Timestamp(imin)].dropna(subset=["index_name","trade_date","close_val"])
        print(f"loading daily {len(daily):,} (< {dmin}) | index {len(idx):,} (< {imin})")
        print(f"  daily span {daily['trade_date'].min().date()}->{daily['trade_date'].max().date()} | deliv null {daily['deliv_per'].isna().mean()*100:.1f}%")
        con.execute("INSERT INTO daily_data SELECT * FROM daily"); con.execute("INSERT INTO index_data SELECT * FROM idx")
        span=con.execute("SELECT MIN(trade_date),MAX(trade_date),COUNT(*) FROM daily_data").fetchone(); con.close()
        print(f"DONE daily_data now {span[0]}->{span[1]} ({span[2]:,} rows)")
    elif "--parse-check" in sys.argv:
        daily=parse_combine(); print(f"combined {len(daily):,} rows {daily['trade_date'].min().date()}->{daily['trade_date'].max().date()} deliv null {daily['deliv_per'].isna().mean()*100:.1f}%")
    else:
        download()
