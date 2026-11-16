"""
Repair fao_participant / fii_derivatives_stats rows that hold ANOTHER date's file.

Found 2026-11-15: 298 of 2,192 stored participant-OI days (and as many Vol days)
were byte-identical copies of a different date's file with the same day-of-month
(e.g. 09-Jan-2026's file also stored under 09-Feb-2026 and 09-Apr-2026). The NSE
/api/reports fallback in the fetchers served the wrong month and nothing checked
the file's own title date. Re-downloading from the primary nsearchives URL showed
the earliest copy is genuine and the later ones are wrong; holiday-dated rows are
copies too (NSE serves no file for a holiday).

What this does, per stored date (suspect dates first, then every other date):
  * download OI + Vol CSV and the FII-stats XLS from the PRIMARY URL ONLY
  * refuse any file whose title names a different date
  * if NSE's content differs from what is stored -> replace the stored rows
  * if NSE serves nothing AND the date is not a Nifty session -> delete the rows
  * if NSE serves nothing on a real session -> leave the rows, report them
A parquet backup of both tables is written before the first change.
Resumable: progress is kept in data/backup/repair_progress.json.

Usage:  python scripts/repair_participant_copies.py [--suspects-only] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                           # noqa: BLE001
    pass

import pandas as pd                                          # noqa: E402

from src.data.repository import get_repository, query_dataframe   # noqa: E402
from src.ingestion.fao_fetcher import _parse_fao_csv, title_date  # noqa: E402
from src.ingestion.fii_stats_fetcher import _parse_fii_stats_xls  # noqa: E402
from src.ingestion.http_client import NSEHttpClient                 # noqa: E402

BACKUP = ROOT / "data" / "backup"
PROGRESS = BACKUP / "repair_progress.json"
SLEEP = 0.3
FAO_URL = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_{kind}_{d:%d%m%Y}.csv"
FII_URL = "https://nsearchives.nseindia.com/content/fo/fii_stats_{d:%d-%b-%Y}.xls"
SIG_COLS = ["fut_idx_long", "fut_idx_short", "opt_idx_call_long", "opt_idx_put_short", "total_long"]


def _dates(table: str) -> list[date]:
    df = query_dataframe(f"SELECT DISTINCT trade_date FROM {table} ORDER BY 1")
    return [pd.Timestamp(x).date() for x in df["trade_date"]]


def _suspects() -> set[date]:
    """Dates whose participant content duplicates another date, plus non-session dates."""
    d = query_dataframe(
        "SELECT trade_date, string_agg(client_type || ':' || fut_idx_long || '/' || fut_idx_short"
        " || '/' || opt_idx_call_long, '|' ORDER BY client_type) sig"
        " FROM fao_participant WHERE data_type = 'OI' GROUP BY trade_date")
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date
    dup = d[d.duplicated("sig", keep=False)]["trade_date"]
    sess = set(pd.to_datetime(query_dataframe(
        "SELECT trade_date FROM index_data WHERE index_name = 'Nifty 50'")["trade_date"]).dt.date)
    off = {x for x in d["trade_date"] if x not in sess}
    return set(dup) | off


def _sessions() -> set[date]:
    return set(pd.to_datetime(query_dataframe(
        "SELECT trade_date FROM index_data WHERE index_name = 'Nifty 50'")["trade_date"]).dt.date)


def _fetch_fao(client, d: date) -> pd.DataFrame | None | str:
    frames = []
    for kind, dt in (("oi", "OI"), ("vol", "Vol")):
        txt = client.get_text(FAO_URL.format(kind=kind, d=d), expect_404_ok=True)
        if not txt or not txt.strip() or "<" in txt.strip().splitlines()[0]:
            return None
        stated = title_date(txt)
        if stated is not None and stated != d:
            return f"title says {stated}"
        frames.append(_parse_fao_csv(txt, d, dt))
        time.sleep(SLEEP)
    return pd.concat(frames, ignore_index=True)


def _fetch_fii(client, d: date):
    data = client.get_bytes(FII_URL.format(d=d), expect_404_ok=True)
    if not data or data[:4] not in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04"):
        return None
    try:
        return _parse_fii_stats_xls(data, d)
    except Exception as exc:                                # noqa: BLE001
        return f"parse: {exc}"


def _same_fao(new: pd.DataFrame, d: date) -> bool:
    old = query_dataframe("SELECT * FROM fao_participant WHERE trade_date = ?", [d])
    if len(old) != len(new):
        return False
    k = ["client_type", "data_type"]
    a = old.set_index(k)[SIG_COLS].sort_index()
    b = new.set_index(k)[SIG_COLS].sort_index()
    return a.index.equals(b.index) and (a.values == b.values).all()


def _same_fii(new: pd.DataFrame, d: date) -> bool:
    old = query_dataframe("SELECT category, oi_contracts, buy_contracts FROM fii_derivatives_stats "
                          "WHERE trade_date = ?", [d])
    if len(old) != len(new):
        return False
    a = old.set_index("category")[["oi_contracts", "buy_contracts"]].sort_index()
    b = new.set_index("category")[["oi_contracts", "buy_contracts"]].sort_index()
    return a.index.equals(b.index) and (a.values == b.values).all()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suspects-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    BACKUP.mkdir(parents=True, exist_ok=True)
    repo = get_repository()
    stamp = date.today().isoformat()
    for t in ("fao_participant", "fii_derivatives_stats"):
        out = BACKUP / f"{t}_before_repair_{stamp}.parquet"
        if not out.exists():
            query_dataframe(f"SELECT * FROM {t}").to_parquet(out, index=False)
            print(f"backup -> {out}", flush=True)

    prog = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {"fao": [], "fii": []}
    done_fao, done_fii = set(prog["fao"]), set(prog["fii"])
    sess = _sessions()
    sus = _suspects()
    fao_all = _dates("fao_participant")
    fii_all = _dates("fii_derivatives_stats")
    order = sorted(sus) + ([] if args.suspects_only else [d for d in fao_all if d not in sus])
    fii_order = sorted(sus & set(fii_all)) + (
        [] if args.suspects_only else [d for d in fii_all if d not in sus])
    print(f"suspects {len(sus)} · fao dates {len(fao_all)} · fii dates {len(fii_all)}", flush=True)

    client = NSEHttpClient()
    stats = {"fao_replaced": 0, "fao_deleted": 0, "fao_same": 0, "fao_missing_session": [],
             "fao_bad_title": [], "fii_replaced": 0, "fii_deleted": 0, "fii_same": 0,
             "fii_missing_session": [], "fii_bad": []}
    t0 = time.time()

    def save():
        PROGRESS.write_text(json.dumps({"fao": sorted(done_fao), "fii": sorted(done_fii)}))

    for i, d in enumerate(order):
        if d.isoformat() in done_fao:
            continue
        try:
            new = _fetch_fao(client, d)
        except Exception as exc:                            # noqa: BLE001
            print(f"  ! fao {d}: {exc}", flush=True)
            continue
        if isinstance(new, str):
            stats["fao_bad_title"].append((d.isoformat(), new))
        elif new is None:
            if d not in sess:
                if not args.dry_run:
                    with repo._cm.connect() as conn:        # noqa: SLF001
                        conn.execute("DELETE FROM fao_participant WHERE trade_date = ?", [d])
                stats["fao_deleted"] += 1
                print(f"  - fao {d}: no NSE file, not a session -> deleted", flush=True)
            else:
                stats["fao_missing_session"].append(d.isoformat())
        elif _same_fao(new, d):
            stats["fao_same"] += 1
        else:
            if not args.dry_run:
                repo.upsert_fao_data(new)
            stats["fao_replaced"] += 1
            print(f"  * fao {d}: replaced with NSE's own file", flush=True)
        done_fao.add(d.isoformat())
        if i % 25 == 0:
            save()
            print(f"  fao {i}/{len(order)}  {time.time()-t0:.0f}s  {stats['fao_replaced']} replaced "
                  f"{stats['fao_deleted']} deleted", flush=True)
        time.sleep(SLEEP)
    save()

    for i, d in enumerate(fii_order):
        if d.isoformat() in done_fii:
            continue
        try:
            new = _fetch_fii(client, d)
        except Exception as exc:                            # noqa: BLE001
            print(f"  ! fii {d}: {exc}", flush=True)
            continue
        if isinstance(new, str):
            stats["fii_bad"].append((d.isoformat(), new))
        elif new is None or new.empty:
            if d not in sess:
                if not args.dry_run:
                    with repo._cm.connect() as conn:        # noqa: SLF001
                        conn.execute("DELETE FROM fii_derivatives_stats WHERE trade_date = ?", [d])
                stats["fii_deleted"] += 1
            else:
                stats["fii_missing_session"].append(d.isoformat())
        elif _same_fii(new, d):
            stats["fii_same"] += 1
        else:
            if not args.dry_run:
                repo.upsert_fii_stats(new)
            stats["fii_replaced"] += 1
            print(f"  * fii {d}: replaced", flush=True)
        done_fii.add(d.isoformat())
        if i % 25 == 0:
            save()
            print(f"  fii {i}/{len(fii_order)}  {time.time()-t0:.0f}s  {stats['fii_replaced']} replaced",
                  flush=True)
        time.sleep(SLEEP)
    save()

    print("\nDONE", json.dumps({k: (v if not isinstance(v, list) else (len(v), v[:15]))
                                for k, v in stats.items()}, indent=1, default=str), flush=True)
    (BACKUP / f"repair_report_{stamp}.json").write_text(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
