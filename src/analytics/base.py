from typing import Tuple
from src.data.repository import get_latest_trade_date, get_available_dates


def get_min_turnover_filter() -> float:
    from src.config_loader import load_config
    return load_config()["analytics"]["min_turnover_lacs"]


def get_delivery_window() -> int:
    from src.config_loader import load_config
    return load_config()["analytics"]["delivery_avg_window"]


def get_volume_window() -> int:
    from src.config_loader import load_config
    return load_config()["analytics"]["volume_avg_window"]


def get_thresholds() -> Tuple[float, float]:
    from src.config_loader import load_config
    cfg = load_config()["analytics"]
    return cfg["accumulation_threshold"], cfg["distribution_threshold"]


def get_weighting_method() -> str:
    from src.config_loader import load_config
    return load_config()["analytics"]["weighting_method"]
