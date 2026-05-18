"""
Backward-compatibility shim — delegates to src.core.config.

Existing imports like `from src.config_loader import load_config` keep working.
New code should import from src.core.config directly.
"""
from src.core.config import get_config, PROJECT_ROOT  # noqa: F401


def load_config() -> dict:
    """Return raw config as a plain dict for legacy callers."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with config_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)

# Attach cache_clear so test fixtures can call load_config.cache_clear()
load_config.cache_clear = lambda: None  # type: ignore[attr-defined]


def get_db_path():
    """Return Path to the DuckDB file."""
    return get_config().database.resolved_path


def get_log_dir():
    """Return Path to the log directory."""
    return get_config().logging.resolved_log_dir
