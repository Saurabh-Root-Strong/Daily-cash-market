import io
import zipfile
from datetime import date
from typing import Optional

import pandas as pd

from src.ingestion.nse_client import NSEClient
from src.logging_setup import get_logger

log = get_logger(__name__)

_EQUITY_SERIES = {"EQ", "SM", "ST"}

_COL_MAP = {
    "TradDt": "trade_date",
    "TckrSymb": "symbol",
    "SctySrs": "series",
    "PrvsClsgPric": "prev_close",
    "OpnPric": "open_price",
    "HghPric": "high_price",
    "LwPric": "low_price",
    "LastPric": "last_price",
    "ClsPric": "close_price",
    "TtlTradgVol": "ttl_trd_qnty",
    "TtlTrfVal": "turnover_lacs",
    "TtlNbOfTxsExctd": "no_of_trades",
}


def build_url(trade_date: date) -> str:
    from src.config_loader import load_config
    template = load_config()["ingestion"]["bhavcopy_url"]
    return template.format(date=trade_date.strftime("%Y%m%d"))


def fetch_bhavcopy(trade_date: date, client: NSEClient) -> Optional[pd.DataFrame]:
    url = build_url(trade_date)
    log.info("Fetching bhavcopy: %s", url)
    resp = client.get(url, expect_404_ok=True)
    if resp is None:
        log.info("Bhavcopy not available for %s (404/holiday)", trade_date)
        return None
    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        raw_df = pd.read_csv(zf.open(csv_name))
        return transform_to_schema(raw_df, trade_date)
    except Exception as exc:
        log.error("Failed to parse bhavcopy for %s: %s", trade_date, exc)
        return None


def transform_to_schema(raw_df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    df = raw_df.copy()

    # Filter cash market equities
    if "Sgmt" in df.columns:
        df = df[df["Sgmt"] == "CM"]
    if "SctySrs" in df.columns:
        df = df[df["SctySrs"].isin(_EQUITY_SERIES)]

    df = df.rename(columns=_COL_MAP)

    # Convert turnover from rupees to lakhs
    df["turnover_lacs"] = pd.to_numeric(df["turnover_lacs"], errors="coerce") / 100_000

    # Compute avg_price = turnover_rupees / volume
    turnover_rupees = df["turnover_lacs"] * 100_000
    volume = pd.to_numeric(df["ttl_trd_qnty"], errors="coerce")
    df["avg_price"] = turnover_rupees / volume.replace(0, pd.NA)

    df["trade_date"] = trade_date
    df["deliv_qty"] = None
    df["deliv_per"] = None

    for col in ["prev_close", "open_price", "high_price", "low_price",
                "last_price", "close_price", "turnover_lacs", "avg_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["ttl_trd_qnty", "no_of_trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    keep = [
        "trade_date", "symbol", "series", "prev_close", "open_price",
        "high_price", "low_price", "last_price", "close_price", "avg_price",
        "ttl_trd_qnty", "turnover_lacs", "no_of_trades", "deliv_qty", "deliv_per",
    ]
    existing = [c for c in keep if c in df.columns]
    return df[existing].reset_index(drop=True)
