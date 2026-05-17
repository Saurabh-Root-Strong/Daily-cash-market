from datetime import datetime
from typing import Optional
import pandas as pd

from src.ingestion.nse_client import NSEClient
from src.logging_setup import get_logger

log = get_logger(__name__)

_BASE_URL = "https://nsearchives.nseindia.com/content/indices/ind_{filename}list.csv"

INDEX_DEFINITIONS = {
    "NIFTY PHARMA":             ("niftypharma",           "Pharma"),
    "NIFTY IT":                 ("niftyit",               "IT"),
    "NIFTY BANK":               ("niftybank",             "Bank"),
    "NIFTY AUTO":               ("niftyauto",             "Auto"),
    "NIFTY FMCG":               ("niftyfmcg",             "FMCG"),
    "NIFTY METAL":              ("niftymetal",            "Metal"),
    "NIFTY REALTY":             ("niftyrealty",           "Realty"),
    "NIFTY ENERGY":             ("niftyenergy",           "Energy"),
    "NIFTY MEDIA":              ("niftymedia",            "Media"),
    "NIFTY PSU BANK":           ("niftypsubank",          "PSU Bank"),
    "NIFTY PVT BANK":           ("niftyprivatebank",       "Pvt Bank"),
    "NIFTY FIN SERVICE":        ("niftyfinance",          "Fin Service"),
    "NIFTY CONSUMER DURABLES":  ("niftyconsumerdurables", "Consumer Durables"),
    "NIFTY HEALTHCARE":         ("niftyhealthcare",       "Healthcare"),
    "NIFTY OIL AND GAS":        ("niftyoilgas",           "Oil & Gas"),
}


def fetch_one_index(name: str, filename: str, sector: str, client: NSEClient) -> Optional[pd.DataFrame]:
    url = _BASE_URL.format(filename=filename)
    log.info("Fetching index constituents: %s from %s", name, url)
    resp = client.get(url, expect_404_ok=True)
    if resp is None:
        log.warning("Index file not found for %s", name)
        return None
    try:
        import io
        df = pd.read_csv(io.BytesIO(resp.content))
        df.columns = [c.strip() for c in df.columns]

        symbol_col = next(
            (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()),
            None
        )
        company_col = next(
            (c for c in df.columns if "company" in c.lower() or "name" in c.lower()),
            None
        )
        industry_col = next(
            (c for c in df.columns if "industry" in c.lower()),
            None
        )

        if symbol_col is None:
            log.warning("No symbol column found in %s", name)
            return None

        result = pd.DataFrame()
        result["symbol"] = df[symbol_col].str.strip()
        result["company_name"] = df[company_col].str.strip() if company_col else ""
        result["sector"] = sector
        result["industry"] = df[industry_col].str.strip() if industry_col else sector
        result["market_cap_category"] = ""
        result["last_updated"] = datetime.now()

        return result.dropna(subset=["symbol"])
    except Exception as exc:
        log.error("Failed to parse index %s: %s", name, exc)
        return None


def fetch_all_sectors(client: NSEClient) -> pd.DataFrame:
    frames = []
    for name, (filename, sector) in INDEX_DEFINITIONS.items():
        df = fetch_one_index(name, filename, sector, client)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        log.warning("No sector data fetched")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Keep first occurrence per symbol (highest-priority index)
    combined = combined.drop_duplicates(subset=["symbol"], keep="first")
    return combined
