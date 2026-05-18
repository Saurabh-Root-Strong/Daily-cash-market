"""
Backward-compatibility shim — delegates to src.core.logging.

New code should import get_logger from src.core.logging directly.
"""
from src.core.logging import get_logger  # noqa: F401

__all__ = ["get_logger"]
