"""
SectorFetcher — downloads NSE index constituent CSVs to build sector_master.

Each sectoral index (NIFTY PHARMA, NIFTY IT, …) maps symbols to a sector label.
INDEX_DEFINITIONS is the single source of truth for which indices we track.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import pandas as pd

from src.core.logging import get_logger
from src.ingestion.base import BaseFetcher

__all__ = ["SectorFetcher", "fetch_all_sectors", "fetch_one_index", "INDEX_DEFINITIONS"]

log = get_logger(__name__)

_CONSTITUENT_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_{filename}list.csv"
)

# (filename_slug, sector_label) — order matters: first match wins on dedup
INDEX_DEFINITIONS: dict[str, tuple[str, str]] = {
    "NIFTY PHARMA":            ("niftypharma",           "Pharma"),
    "NIFTY IT":                ("niftyit",               "IT"),
    "NIFTY BANK":              ("niftybank",             "Bank"),
    "NIFTY AUTO":              ("niftyauto",             "Auto"),
    "NIFTY FMCG":              ("niftyfmcg",             "FMCG"),
    "NIFTY METAL":             ("niftymetal",            "Metal"),
    "NIFTY REALTY":            ("niftyrealty",           "Realty"),
    "NIFTY ENERGY":            ("niftyenergy",           "Energy"),
    "NIFTY MEDIA":             ("niftymedia",            "Media"),
    "NIFTY PSU BANK":          ("niftypsubank",          "PSU Bank"),
    "NIFTY PVT BANK":          ("niftyprivatebank",      "Pvt Bank"),
    "NIFTY FIN SERVICE":       ("niftyfinance",          "Fin Service"),
    "NIFTY CONSUMER DURABLES": ("niftyconsumerdurables", "Consumer Durables"),
    "NIFTY HEALTHCARE":        ("niftyhealthcare",       "Healthcare"),
    "NIFTY OIL AND GAS":       ("niftyoilgas",           "Oil & Gas"),
}


class SectorFetcher(BaseFetcher):
    """Fetches all configured NSE index constituent files."""

    @property
    def name(self) -> str:
        return "SectorConstituents"

    def fetch(self, trade_date=None) -> pd.DataFrame:  # trade_date unused
        """Return combined sector_master DataFrame for all configured indices."""
        frames: list[pd.DataFrame] = []
        for index_name, (filename, sector) in INDEX_DEFINITIONS.items():
            df = self._fetch_one(index_name, filename, sector)
            if df is not None and not df.empty:
                frames.append(df)
        if not frames:
            log.warning("No sector constituent data fetched")
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return combined.drop_duplicates(subset=["symbol"], keep="first")

    def _fetch_one(
        self, index_name: str, filename: str, sector: str
    ) -> Optional[pd.DataFrame]:
        url = _CONSTITUENT_URL.format(filename=filename)
        log.info("Fetching constituents: %s", index_name)
        raw = self._client.get_bytes(url, expect_404_ok=True)
        if raw is None:
            log.warning("Constituent file not found for %s", index_name)
            return None
        try:
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip() for c in df.columns]

            symbol_col = next(
                (c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None
            )
            company_col = next(
                (c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None
            )
            industry_col = next(
                (c for c in df.columns if "industry" in c.lower()), None
            )
            if symbol_col is None:
                log.warning("No symbol column in %s", index_name)
                return None

            out = pd.DataFrame()
            out["symbol"]             = df[symbol_col].str.strip()
            out["company_name"]       = df[company_col].str.strip() if company_col else ""
            out["sector"]             = sector
            out["industry"]           = df[industry_col].str.strip() if industry_col else sector
            out["market_cap_category"] = ""
            out["last_updated"]       = datetime.now()
            return out.dropna(subset=["symbol"])
        except Exception as exc:
            log.error("Failed to parse %s: %s", index_name, exc)
            return None


# ── Backward-compatible module-level functions ────────────────────────────────

def fetch_all_sectors(client) -> pd.DataFrame:
    return SectorFetcher(client).fetch()


def fetch_one_index(
    name: str, filename: str, sector: str, client
) -> Optional[pd.DataFrame]:
    return SectorFetcher(client)._fetch_one(name, filename, sector)
