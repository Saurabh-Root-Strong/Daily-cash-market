"""
Centralised logging setup.

Call get_logger(__name__) in every module — idempotent, safe to call many times.
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

__all__ = ["get_logger"]

_configured: bool = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        _setup_root_logger()
        _configured = True
    return logging.getLogger(name)


def _setup_root_logger() -> None:
    from src.core.config import get_config

    cfg = get_config()
    level = getattr(logging, cfg.logging.level.upper(), logging.INFO)

    log_dir: Path = cfg.logging.resolved_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"nse_dashboard_{date.today().isoformat()}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    if not root.handlers:   # avoid duplicate handlers on reimport
        root.setLevel(level)
        root.addHandler(file_handler)
        root.addHandler(stream_handler)
