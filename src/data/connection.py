"""
Database connection management.

ConnectionManager owns the db path and produces short-lived connections.
All callers open a connection, use it, and close it — no long-lived handles.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

__all__ = ["ConnectionManager"]


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
        conn = duckdb.connect(str(self._db_path))
        try:
            yield conn
        finally:
            conn.close()

    def raw_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Non-managed connection for Streamlit's @st.cache_resource.
        Caller is responsible for closing.
        """
        return duckdb.connect(str(self._db_path))
