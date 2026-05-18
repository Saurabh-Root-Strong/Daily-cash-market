"""
Backward-compatibility shim.

New code should import NSEHttpClient from src.ingestion.http_client directly.
"""
from src.ingestion.http_client import NSEHttpClient as NSEClient  # noqa: F401

__all__ = ["NSEClient"]
