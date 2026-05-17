import logging
import sys
from datetime import date
from pathlib import Path

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        _setup_root_logger()
        _configured = True
    return logging.getLogger(name)


def _setup_root_logger() -> None:
    from src.config_loader import load_config, get_log_dir

    cfg = load_config()
    level = getattr(logging, cfg["logging"]["level"].upper(), logging.INFO)

    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"nse_dashboard_{date.today().isoformat()}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
