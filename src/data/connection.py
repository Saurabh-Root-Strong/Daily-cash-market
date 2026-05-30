"""
Database connection management.

ConnectionManager owns the db path and produces short-lived connections.
All callers open a connection, use it, and close it — no long-lived handles.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

__all__ = ["ConnectionManager"]


def _read_only() -> bool:
    """Open DuckDB read-only on the hosted snapshot (CLOUD_MODE=true).

    In cloud mode the DB is an ephemeral downloaded snapshot that must never be
    written to, and DuckDB read-only connections let multiple readers share one
    file. Locally we stay read-write so the daily fetch and the dashboard's
    prediction logging can still write. Read at connect time so tests can toggle it.
    """
    return os.environ.get("CLOUD_MODE", "").lower() == "true"


class ConnectionManager:
    """Produces DuckDB connections for a fixed database file."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextmanager
    def connect(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context-manager connection — always closed on exit."""
        conn = duckdb.connect(str(self._db_path), read_only=_read_only())
        try:
            yield conn
        finally:
            conn.close()

    def raw_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Non-managed connection for Streamlit's @st.cache_resource.
        Caller is responsible for closing.
        """
        return duckdb.connect(str(self._db_path), read_only=_read_only())
