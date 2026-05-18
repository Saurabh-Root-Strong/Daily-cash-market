"""
DeliveryFetcher — downloads and parses the NSE MTO delivery file.

MTO format: plain-text DAT file; equity rows start with "20,".
Fields: record-type, sr-no, symbol, series, traded-qty, deliv-qty, deliv-pct.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.core.exceptions import ParseError
from src.core.logging import get_logger
from src.ingestion.base import BaseFetcher

__all__ = ["DeliveryFetcher", "fetch_delivery"]

log = get_logger(__name__)


class DeliveryFetcher(BaseFetcher):
    """Downloads and parses the MTO (Mark-to-Market delivery) DAT file."""

    @property
    def name(self) -> str:
        return "Delivery (MTO)"

    def build_url(self, trade_date: date) -> str:
        from src.core.config import get_config
        return get_config().ingestion.delivery_url.format(
            date=trade_date.strftime("%d%m%Y")
        )

    def fetch(self, trade_date: date) -> pd.DataFrame:
        url = self.build_url(trade_date)
        log.info("Fetching delivery data: %s", url)
        text = self._client.get_text(url, expect_404_ok=True)
        if text is None:
            log.info("Delivery data not available for %s (holiday/weekend)", trade_date)
            return pd.DataFrame()
        try:
            return _parse_mto(text, trade_date)
        except Exception as exc:
            raise ParseError(f"Failed to parse MTO for {trade_date}: {exc}") from exc


def _parse_mto(text: str, trade_date: date) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("20,"):
            continue
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            rows.append({
                "trade_date": trade_date,
                "symbol":     parts[2].strip(),
                "series":     parts[3].strip(),
                "deliv_qty":  int(parts[5].strip()),
                "deliv_per":  float(parts[6].strip()),
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        log.warning("No delivery rows parsed for %s", trade_date)
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Backward-compatible module-level function ─────────────────────────────────

def fetch_delivery(trade_date: date, client) -> pd.DataFrame | None:
    """Legacy call-site wrapper — prefer DeliveryFetcher(client).fetch(date)."""
    result = DeliveryFetcher(client).fetch(trade_date)
    return None if result.empty else result
