from functools import lru_cache
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_db_path() -> Path:
    cfg = load_config()
    return PROJECT_ROOT / cfg["database"]["path"]


def get_log_dir() -> Path:
    cfg = load_config()
    return PROJECT_ROOT / cfg["logging"]["log_dir"]
