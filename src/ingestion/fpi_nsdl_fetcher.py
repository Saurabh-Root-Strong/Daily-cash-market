"""
FPINSDLFetcher — NSDL FPI Investment data (automated + manual drop-folder).

Two automated data sources (no login required):
  Latest.aspx  — most recent trading day (fetched daily after market close)
  Monthly.aspx — all trading days in the current calendar month

Manual drop-folder fallback:
  data/fpi_imports/ — for historical months not covered by Monthly.aspx

Category mapping to DB schema:
  Equity              → Equity
  Debt-General Limit  → Debt   (aggregated)
  Debt-FAR            → Debt   (aggregated)
  Debt-VRR            → Debt-VRR
  Hybrid              → Hybrid
  Mutual Funds        → Others (aggregated)
  AIFs                → Others (aggregated)
"""
from __future__ import annotations

import io
import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from src.core.logging import get_logger
from src.data.repository import upsert_fpi_flows

__all__ = [
    "FPINSDLFetcher",
    "import_fpi_folder",
    "fetch_fpi_latest",
    "fetch_fpi_monthly",
]

log = get_logger(__name__)

_BASE_URL    = "https://www.fpi.nsdl.co.in"
_LATEST_URL  = f"{_BASE_URL}/web/Reports/Latest.aspx"
_MONTHLY_URL = f"{_BASE_URL}/web/Reports/Monthly.aspx"

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         _BASE_URL,
}

# Category mapping for HTML scraping — EXACT match only (case-insensitive).
# Substring matching causes false positives: "Primary market & others" → "Others".
_CATEGORY_MAP = {
    "equity":              "Equity",
    "debt-general limit":  "Debt",
    "debt-general":        "Debt",
    "debt-far":            "Debt",
    "debt far":            "Debt",
    "debt-vrr":            "Debt-VRR",
    "hybrid":              "Hybrid",
    "mutual funds":        "Others",
    "aifs":                "Others",
}

_XLS_MAGIC  = b"\xd0\xcf\x11\xe0"
_XLSX_MAGIC = b"PK\x03\x04"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_BASE_URL, timeout=15)
        time.sleep(0.8)
    except Exception as exc:
        log.debug("NSDL prime failed (non-fatal): %s", exc)
    return s


# ── Automated fetchers ────────────────────────────────────────────────────────

class FPINSDLFetcher:
    """Auto-fetches daily FPI data from NSDL (no login required)."""

    def __init__(self) -> None:
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = _make_session()
        return self._session

    def fetch_latest(self) -> pd.DataFrame:
        """Fetch the most recent trading day's FPI data from Latest.aspx."""
        try:
            r = self._get_session().get(_LATEST_URL, timeout=20)
            r.raise_for_status()
            return _parse_nsdl_html(r.text)
        except Exception as exc:
            log.warning("FPI Latest fetch failed: %s", exc)
            self._session = None
            return pd.DataFrame()

    def fetch_monthly(self) -> pd.DataFrame:
        """Fetch all trading days in the current calendar month from Monthly.aspx."""
        try:
            r = self._get_session().get(_MONTHLY_URL, timeout=20)
            r.raise_for_status()
            return _parse_nsdl_html(r.text)
        except Exception as exc:
            log.warning("FPI Monthly fetch failed: %s", exc)
            self._session = None
            return pd.DataFrame()


def fetch_fpi_latest() -> dict:
    """Fetch latest day, upsert to DB. Returns summary dict."""
    fetcher = FPINSDLFetcher()
    df = fetcher.fetch_latest()
    if df.empty:
        return {"rows_inserted": 0, "dates": [], "error": "Empty response from NSDL Latest"}
    rows = upsert_fpi_flows(df)
    dates = sorted(df["trade_date"].unique().tolist())
    log.info("FPI daily fetch: %d rows for dates %s", rows, dates)
    return {"rows_inserted": rows, "dates": dates, "error": None}


def fetch_fpi_monthly() -> dict:
    """Fetch current month's daily FPI data, upsert all to DB. Returns summary dict."""
    fetcher = FPINSDLFetcher()
    df = fetcher.fetch_monthly()
    if df.empty:
        return {"rows_inserted": 0, "dates": [], "error": "Empty response from NSDL Monthly"}
    rows = upsert_fpi_flows(df)
    dates = sorted(df["trade_date"].unique().tolist())
    log.info("FPI monthly fetch: %d rows for %d dates", rows, len(dates))
    return {"rows_inserted": rows, "dates": dates, "error": None}


# ── HTML parser ───────────────────────────────────────────────────────────────

def _parse_nsdl_html(html: str) -> pd.DataFrame:
    """
    Parse the FPI investment table from NSDL Latest.aspx or Monthly.aspx.

    The table uses rowspan for Date and Category cells. We track the current
    date and category context as we walk rows, and accumulate values per route,
    recording the "Sub-total" row for each (date, category) block.
    """
    # Find all <tr> elements in the relevant table section
    tbl_start = html.find("Gross Purchases")
    if tbl_start < 0:
        log.debug("NSDL HTML: 'Gross Purchases' header not found")
        return pd.DataFrame()

    # Walk back to find the enclosing <table>
    table_open = html.rfind("<table", 0, tbl_start)
    table_close_search = html.find("</table>", tbl_start)
    tbl_html = html[table_open:table_close_search + 8]

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl_html, re.DOTALL | re.IGNORECASE)

    records: list[dict] = []
    current_date: date | None = None
    current_cat:  str | None  = None

    for row_html in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.DOTALL | re.IGNORECASE)
        cells = [_strip_tags(c).strip() for c in cells]
        cells = [c for c in cells if c]  # drop empty

        if not cells:
            continue

        # Detect if first cell is a date
        candidate_date = _parse_date(cells[0])
        if candidate_date is not None:
            current_date = candidate_date
            cells = cells[1:]  # consume date cell

        if not cells:
            continue

        # Check if this row has a category + route + 4 numbers
        # OR just route + 4 numbers (category carried from above)
        # OR sub-total + 4 numbers
        # OR a new category on its own row
        first = cells[0]

        # "Sub-total" or "Total" row
        if first.lower() in ("sub-total", "sub total"):
            if current_date and current_cat and len(cells) >= 4:
                gp = _paren_float(cells[1])
                gs = _paren_float(cells[2])
                ni = _paren_float(cells[3])
                records.append({
                    "trade_date":        current_date,
                    "category":          current_cat,
                    "gross_purchase_cr": gp,
                    "gross_sales_cr":    gs,
                    "net_investment_cr": ni,
                })
            continue

        # Skip rows that look like pure totals
        if first.lower() in ("total",):
            continue

        # Check if first cell is a known category name
        cat_match = _match_category(first)
        if cat_match:
            current_cat = cat_match
            cells = cells[1:]  # consume category cell

        # Skip non-data routes (header rows, note rows)
        # Data rows must have ≥ 3 numeric values among remaining cells

    if not records:
        log.debug("NSDL HTML: no sub-total rows found")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Multiple categories can map to same canonical name — aggregate them
    df = (
        df.groupby(["trade_date", "category"], as_index=False)
        .agg({"gross_purchase_cr": "sum", "gross_sales_cr": "sum", "net_investment_cr": "sum"})
    )
    df["trade_date"]        = pd.to_datetime(df["trade_date"]).dt.date
    df["gross_purchase_cr"] = df["gross_purchase_cr"].astype("float64")
    df["gross_sales_cr"]    = df["gross_sales_cr"].astype("float64")
    df["net_investment_cr"] = df["net_investment_cr"].astype("float64")

    return df.reset_index(drop=True)


def _match_category(text: str) -> str | None:
    """Return canonical category name for an EXACT match only.
    Substring matching causes false positives (e.g. 'Primary market & others' → 'Others').
    """
    return _CATEGORY_MAP.get(text.strip().lower())


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def _paren_float(val: str) -> float:
    """Convert '(1234.56)' → -1234.56, '1234.56' → 1234.56."""
    s = val.strip().replace(",", "")
    if s.startswith("(") and s.endswith(")"):
        try:
            return -float(s[1:-1])
        except ValueError:
            return 0.0
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def _parse_date(val: str) -> date | None:
    val = val.strip()
    for fmt in ("%d-%b-%Y", "%d/%b/%Y", "%d %b %Y",
                "%d-%B-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(val, fmt).date()
        except ValueError:
            continue
    return None


# ── Manual drop-folder import ─────────────────────────────────────────────────

def import_fpi_folder(folder: Path | None = None) -> dict:
    """
    Scan folder for .xls/.xlsx files, parse each, upsert to DB.
    Used for historical months not covered by Monthly.aspx.
    Returns: {files_processed, rows_inserted, errors: list[str]}
    """
    if folder is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        folder = project_root / "data" / "fpi_imports"

    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return {"files_processed": 0, "rows_inserted": 0,
                "errors": [f"Created import folder: {folder}. Drop NSDL Excel files there."]}

    excel_files = sorted(list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))
    if not excel_files:
        return {"files_processed": 0, "rows_inserted": 0,
                "errors": [f"No .xls/.xlsx files found in {folder}"]}

    total_rows = 0
    errors: list[str] = []
    files_ok = 0

    for fpath in excel_files:
        try:
            data = fpath.read_bytes()
            df = _parse_fpi_excel(data, fpath.name)
            if df.empty:
                errors.append(f"{fpath.name}: no valid rows found")
                continue
            rows = upsert_fpi_flows(df)
            total_rows += rows
            files_ok += 1
            log.info("FPI import: %s → %d rows upserted", fpath.name, rows)
        except Exception as exc:
            errors.append(f"{fpath.name}: {exc}")
            log.warning("FPI import error in %s: %s", fpath.name, exc)

    return {"files_processed": files_ok, "rows_inserted": total_rows, "errors": errors}


# ── Excel parser (drop-folder) ────────────────────────────────────────────────

_EXCEL_CATEGORY_MAP = {
    "equity":    "Equity",
    "debt vrr":  "Debt-VRR",
    "debt-vrr":  "Debt-VRR",
    "vrr":       "Debt-VRR",
    "debt":      "Debt",
    "hybrid":    "Hybrid",
    "others":    "Others",
    "other":     "Others",
}
_SKIP_CATEGORIES = {"grand total", "total"}


def _parse_fpi_excel(data: bytes, source_name: str = "") -> pd.DataFrame:
    if data[:4] == _XLSX_MAGIC:
        engine = "openpyxl"
    elif data[:4] == _XLS_MAGIC:
        engine = "xlrd"
    else:
        raise ValueError(f"Not a recognised Excel file: {source_name}")

    try:
        xl = pd.ExcelFile(io.BytesIO(data), engine=engine)
    except Exception as exc:
        raise ValueError(f"Cannot open Excel: {exc}") from exc

    for sheet in xl.sheet_names:
        try:
            raw = xl.parse(sheet, header=None, dtype=str)
        except Exception:
            continue
        result = _try_parse_excel_sheet(raw)
        if result is not None and not result.empty:
            return result

    raise ValueError(f"Could not find FPI data in any sheet of {source_name}")


def _try_parse_excel_sheet(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw.empty or raw.shape[1] < 4:
        return None

    cat_row_idx = None
    for i in range(min(20, len(raw))):
        vals_lower = [str(v).strip().lower() for v in raw.iloc[i].values]
        if "equity" in vals_lower and "debt" in vals_lower:
            cat_row_idx = i
            break

    if cat_row_idx is None:
        return None

    cat_header = raw.iloc[cat_row_idx].fillna("")
    col_to_cat: dict[int, str] = {}
    for j, val in enumerate(cat_header):
        key = str(val).strip().lower()
        for raw_key in sorted(_EXCEL_CATEGORY_MAP.keys(), key=len, reverse=True):
            if raw_key in key and key not in _SKIP_CATEGORIES:
                canonical = _EXCEL_CATEGORY_MAP[raw_key]
                if canonical not in col_to_cat.values():
                    col_to_cat[j] = canonical
                break

    if not col_to_cat:
        return None

    data_start = cat_row_idx + 2
    rows: list[dict] = []

    for i in range(data_start, len(raw)):
        row = raw.iloc[i]
        date_raw = str(row.iloc[0]).strip()
        if not date_raw or date_raw.lower() in ("nan", ""):
            continue
        if any(skip in date_raw.lower() for skip in ("total", "grand")):
            continue
        trade_date = _parse_date(date_raw)
        if trade_date is None:
            continue
        for start_col, category in col_to_cat.items():
            try:
                gp = _to_float(row.iloc[start_col])
                gs = _to_float(row.iloc[start_col + 1])
                ni = _to_float(row.iloc[start_col + 2])
            except (IndexError, ValueError):
                continue
            rows.append({"trade_date": trade_date, "category": category,
                         "gross_purchase_cr": gp, "gross_sales_cr": gs,
                         "net_investment_cr": ni})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for col in ("gross_purchase_cr", "gross_sales_cr", "net_investment_cr"):
        df[col] = df[col].astype("float64")
    return df.reset_index(drop=True)


def _to_float(val) -> float:
    s = str(val).strip().replace(",", "")
    if not s or s.lower() in ("nan", "-", "n/a", ""):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0
