"""
FII / DII daily CASH-market data.

Going-forward source: NSE provisional cash figures
  https://www.nseindia.com/api/fiidiiTradeReact   (returns the latest session)

Backfill source: Groww  https://groww.in/fii-dii-data  — load older days via a
CSV (date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net) using
import_fii_dii_csv(), or seed the recent window with backfill_fii_dii_seed().

All values are ₹ Crore. net = buy − sell. FII net negative = foreign selling.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.core.logging import get_logger
from src.data.repository import get_repository

log = get_logger(__name__)

_NSE_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


def _to_date(s: str):
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except ValueError:
            continue
    return pd.to_datetime(s, dayfirst=True, errors="coerce").date()


def _num(v) -> Optional[float]:
    try:
        return round(float(str(v).replace(",", "").strip()), 2)
    except (ValueError, TypeError):
        return None


# ── NSE provisional cash (latest session) ─────────────────────────────────────

def fetch_fii_dii_cash_nse() -> pd.DataFrame:
    """
    Fetch the latest FII/FPI + DII provisional cash figures from NSE.

    Self-contained session with gzip encoding (NSE's brotli responses aren't
    decoded by the shared client) and cookie priming.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://www.nseindia.com/reports/fii-dii",
    })
    try:
        s.get("https://www.nseindia.com/", timeout=15)
        time.sleep(1.0)
        r = s.get(_NSE_URL, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        log.warning("NSE FII/DII fetch failed: %s", exc)
        return pd.DataFrame()

    rec: dict = {}
    for row in data:
        cat = str(row.get("category", "")).upper()
        d = _to_date(row.get("date"))
        rec["trade_date"] = d
        if cat.startswith("FII") or "FPI" in cat:
            rec["fii_buy"], rec["fii_sell"], rec["fii_net"] = (
                _num(row.get("buyValue")), _num(row.get("sellValue")), _num(row.get("netValue")))
        elif cat.startswith("DII"):
            rec["dii_buy"], rec["dii_sell"], rec["dii_net"] = (
                _num(row.get("buyValue")), _num(row.get("sellValue")), _num(row.get("netValue")))
    if not rec.get("trade_date"):
        return pd.DataFrame()
    rec["source"] = "NSE"
    return pd.DataFrame([rec])


# ── CSV backfill (Groww / NSE archive exports) ────────────────────────────────

def import_fii_dii_csv(csv_path: str | Path) -> int:
    """
    Import historical FII/DII cash data from a CSV.

    Accepts flexible headers (case-insensitive): date + fii_buy/fii_sell/fii_net
    + dii_buy/dii_sell/dii_net. Net is derived from buy−sell when absent. Returns
    rows written.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col = {c: c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in col:
                return n
        return None
    c_date = pick("date", "trade_date")
    rows = []
    for _, r in df.iterrows():
        fb, fs = _num(r.get(pick("fii_buy", "fii_purchase"))), _num(r.get(pick("fii_sell", "fii_sale")))
        db, ds = _num(r.get(pick("dii_buy", "dii_purchase"))), _num(r.get(pick("dii_sell", "dii_sale")))
        fn = _num(r.get(pick("fii_net"))) if pick("fii_net") else (
            round(fb - fs, 2) if fb is not None and fs is not None else None)
        dn = _num(r.get(pick("dii_net"))) if pick("dii_net") else (
            round(db - ds, 2) if db is not None and ds is not None else None)
        d = _to_date(r.get(c_date))
        if d is None:
            continue
        rows.append({"trade_date": d, "fii_buy": fb, "fii_sell": fs, "fii_net": fn,
                     "dii_buy": db, "dii_sell": ds, "dii_net": dn, "source": "CSV"})
    if not rows:
        return 0
    return get_repository().upsert_fii_dii_cash(pd.DataFrame(rows))


# ── Recent-window seed (from Groww, ₹ Cr net) — initial backfill ──────────────
# buy/sell not provided by the Groww net view; net is what matters for flow.
_GROWW_SEED = [
    ("2026-06-05", -8776.25,  9133.57), ("2026-06-04", -4447.06, 4360.14),
    ("2026-06-03", -5616.56,  5740.89), ("2026-06-02", -8362.92, 9589.32),
    ("2026-06-01", -3911.68,  5109.13), ("2026-05-29", -21105.86, 16764.14),
    ("2026-05-27", -1042.70,  3821.00), ("2026-05-26", -2407.87, 1361.43),
    ("2026-05-25",   821.75,  3856.88), ("2026-05-22", -4440.47, 6003.53),
]


def backfill_fii_dii_seed() -> int:
    """Seed the recent FII/DII net window (idempotent). Returns rows written."""
    rows = [{"trade_date": _to_date(d), "fii_buy": None, "fii_sell": None,
             "fii_net": fn, "dii_buy": None, "dii_sell": None, "dii_net": dn,
             "source": "Groww"} for d, fn, dn in _GROWW_SEED]
    return get_repository().upsert_fii_dii_cash(pd.DataFrame(rows))


def run_fii_dii_cash_daily() -> int:
    """Fetch today's NSE provisional cash figures and upsert. Returns rows written."""
    df = fetch_fii_dii_cash_nse()
    if df.empty:
        return 0
    return get_repository().upsert_fii_dii_cash(df)
