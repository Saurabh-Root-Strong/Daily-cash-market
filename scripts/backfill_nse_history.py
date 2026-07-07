"""
Backfill NSE history (delivery bhavcopy + index OHLC) into staging CSVs.

DOWNLOAD-ONLY (non-destructive): pulls the raw archive files to a staging dir. A separate
loader step parses + inserts into DCM (daily_data / index_data) for the non-overlapping
pre-2024-12-26 window. Resumable: skips files already staged.

Sources (verified reachable from this env with a browser UA; the HTML site 403s but these
nsearchives file endpoints serve directly):
  delivery : https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
  index    : https://nsearchives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv
404 = market holiday / weekend → skipped.
"""
from __future__ import annotations
import sys, os, time, urllib.request, datetime as dt

STAGE = r"C:/Users/HP/AppData/Local/Temp/claude/d--Python-Projects-Tradebot/00fb1329-476b-4454-94e7-cefc7a5aa9d8/scratchpad/nse_backfill"
BHAV = os.path.join(STAGE, "bhav"); IDX = os.path.join(STAGE, "index")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
START = dt.date(2019, 10, 1)           # sec_bhavdata_full floor (pre-Oct-2019 → MTO path)
END   = dt.date(2024, 12, 25)          # DCM data starts 2024-12-26 → no overlap

def fetch(url: str, dest: str) -> str:
    """-> 'skip' | 'ok' | 'miss' (404/holiday) | 'err'."""
    if os.path.exists(dest) and os.path.getsize(dest) > 500:
        return "skip"
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Referer": "https://www.nseindia.com/all-reports"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if data[:9] == b"<!DOCTYPE" or len(data) < 500:
            return "miss"
        with open(dest, "wb") as f:
            f.write(data)
        return "ok"
    except urllib.error.HTTPError as e:
        return "miss" if e.code == 404 else "err"
    except Exception:
        return "err"

def main():
    os.makedirs(BHAV, exist_ok=True); os.makedirs(IDX, exist_ok=True)
    d = START; nb = ni = miss = err = skip = 0; t0 = time.time()
    while d <= END:
        if d.weekday() < 5:                          # Mon-Fri only
            ds = d.strftime("%d%m%Y")
            rb = fetch(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ds}.csv",
                       os.path.join(BHAV, f"{d.isoformat()}.csv"))
            ri = fetch(f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{ds}.csv",
                       os.path.join(IDX, f"{d.isoformat()}.csv"))
            for r in (rb, ri):
                if r == "ok": pass
            nb += rb == "ok"; ni += ri == "ok"; skip += (rb == "skip")
            miss += (rb == "miss"); err += (rb == "err")
            if rb in ("ok", "err"):                   # be polite only on real fetches
                time.sleep(0.15)
        if d.day == 1:
            print(f"  {d.isoformat()}  bhav={nb} idx={ni} skip={skip} holiday/miss={miss} "
                  f"err={err}  {time.time()-t0:.0f}s", flush=True)
        d += dt.timedelta(days=1)
    print(f"DONE  bhav_ok={nb} index_ok={ni} skipped={skip} misses={miss} errors={err} "
          f"in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
