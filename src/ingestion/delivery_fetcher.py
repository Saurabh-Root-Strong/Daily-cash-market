from datetime import date
from typing import Optional
import io
import pandas as pd

from src.ingestion.nse_client import NSEClient
from src.logging_setup import get_logger

log = get_logger(__name__)


def build_url(trade_date: date) -> str:
    from src.config_loader import load_config
    template = load_config()["ingestion"]["delivery_url"]
    return template.format(date=trade_date.strftime("%d%m%Y"))


def fetch_delivery(trade_date: date, client: NSEClient) -> Optional[pd.DataFrame]:
    url = build_url(trade_date)
    log.info("Fetching delivery data: %s", url)
    text = client.get_text(url, expect_404_ok=True)
    if text is None:
        log.info("Delivery data not available for %s", trade_date)
        return None
    return _parse_mto(text, trade_date)


def _parse_mto(text: str, trade_date: date) -> Optional[pd.DataFrame]:
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
                "symbol": parts[2].strip(),
                "series": parts[3].strip(),
                "deliv_qty": int(parts[5].strip()),
                "deliv_per": float(parts[6].strip()),
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        log.warning("No delivery rows parsed for %s", trade_date)
        return None

    return pd.DataFrame(rows)
