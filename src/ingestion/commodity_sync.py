"""Copy MCX commodity futures from the Commodity_Forex_Market (CFM) project.

CFM already owns every MCX trap (non-traded closes, tender-window settlement
prints, roll gaps, thin one-lot sessions -- see its panel.py). Re-deriving a
front-month series here would be a second implementation that silently drifts
from the one CFM tests, so this module calls CFM's own `front_month_panel` and
stores the result. The crude-vs-index study reads the same table the dashboard
reads, so the number on the screen is the number that was tested.

Read-only and brief: CFM's scheduled updater (4x a day) needs a write lock on
its file, and DuckDB refuses that while any other process has it open.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)

__all__ = ["CFM_DIR", "SYNC_COMMODITIES", "load_from_cfm", "sync_commodities"]

CFM_DIR = Path(os.environ.get("DCM_CFM_DIR",
                              r"D:\Python Projects\Commodity_Forex_Market"))

# Names as CFM stores them in contract_daily.commodity.
# CFM's own liquidity list: below these, the front month goes days without a
# trade and a "return" is a stale quote. NICKEL and LEAD are thin from 2024 --
# kept, but the study gates every session on turnover.
SYNC_COMMODITIES = ["CRUDE OIL", "GOLD", "SILVER", "NATURALGAS", "COPPER",
                    "ZINC", "ALUMINIUM", "LEAD", "NICKEL"]


def _open_cfm(db: Path, tries: int = 10) -> duckdb.DuckDBPyConnection:
    last: Exception | None = None
    for i in range(tries):
        try:
            return duckdb.connect(str(db), read_only=True)
        except Exception as exc:  # noqa: BLE001 -- CFM writer holds the file
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"CFM database busy after {tries} tries: {last}")


def load_from_cfm(commodities: list[str] | None = None,
                  cfm_dir: Path | None = None) -> pd.DataFrame:
    """Roll-safe front-month series per commodity, straight from CFM."""
    cfm_dir = Path(cfm_dir or CFM_DIR)
    db = cfm_dir / "data" / "cfm.duckdb"
    if not db.exists():
        raise FileNotFoundError(f"CFM database not found at {db} (set DCM_CFM_DIR)")
    if str(cfm_dir) not in sys.path:
        sys.path.insert(0, str(cfm_dir))
    from cfm.analytics.panel import front_month_panel   # noqa: E402 -- CFM's own code

    con = _open_cfm(db)
    try:
        p = front_month_panel(con, list(commodities or SYNC_COMMODITIES))
    finally:
        con.close()
    if p.empty:
        return p
    out = p[["trade_date", "commodity", "symbol", "expiry_date", "close",
             "ret1", "turnover_cr"]].copy()
    out["trade_date"] = pd.to_datetime(out.trade_date).dt.date
    out["expiry_date"] = pd.to_datetime(out.expiry_date).dt.date
    return out


def sync_commodities(commodities: list[str] | None = None) -> int:
    """Refresh commodity_daily from CFM. Returns rows written."""
    from src.data.repository import get_repository
    from src.data.schema import initialize_schema

    initialize_schema()
    df = load_from_cfm(commodities)
    n = get_repository().replace_commodity_daily(df)
    if n:
        last = df.groupby("commodity").trade_date.max()
        log.info("Commodity sync: %d rows; latest %s", n,
                 ", ".join(f"{k} {v}" for k, v in last.items()))
    return n
