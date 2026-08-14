"""
Database connection management.

ConnectionManager owns the db path and produces short-lived connections.
All callers open a connection, use it, and close it — no long-lived handles.

DuckDB allows only ONE read-write process to hold a database file at a time
(an exclusive file lock per connection; there is no cross-process MVCC). So
whenever the dashboard and a writer (daily fetch, prediction logging, sector
memory) touch the file in overlapping windows, the loser gets an OS-level
"file is being used by another process" IO error.

Rather than crash, `_open()` retries with exponential backoff + jitter. Almost
all real contention is sub-second (a short write transaction), so a handful of
retries turns a hard crash into a brief, invisible wait. Genuine errors (corrupt
file, missing directory, SQL errors) are NOT lock errors and propagate
immediately.
"""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

__all__ = ["ConnectionManager", "is_lock_error"]


def _read_only() -> bool:
    """Open DuckDB read-only on the hosted snapshot (CLOUD_MODE=true).

    In cloud mode the DB is an ephemeral downloaded snapshot that must never be
    written to, and DuckDB read-only connections let multiple readers share one
    file. Locally we stay read-write so the daily fetch and the dashboard's
    prediction logging can still write. Read at connect time so tests can toggle it.
    """
    return os.environ.get("CLOUD_MODE", "").lower() == "true"


# ── Lock-contention retry policy ──────────────────────────────────────────────
# Tunable via env so a long-running writer (e.g. the historical backfill) can be
# given a larger budget than an interactive dashboard query. Defaults give a
# worst case of roughly ~15 s of waiting, but typical contention clears in <1 s.
_MAX_RETRIES = int(os.environ.get("DUCKDB_LOCK_RETRIES", "30"))
_BASE_DELAY  = float(os.environ.get("DUCKDB_LOCK_BASE_DELAY", "0.1"))   # seconds
_MAX_DELAY   = float(os.environ.get("DUCKDB_LOCK_MAX_DELAY", "0.75"))   # seconds


def is_lock_error(exc: Exception) -> bool:
    """Public alias — callers outside this module need to recognise contention.

    The dashboard uses it to tell "the nightly ingest holds the file" (expected,
    transient, retry in a moment) apart from a genuine fault. Without that
    distinction a routine data refresh renders a full traceback to the user.
    """
    return _is_lock_error(exc)


def _is_lock_error(exc: Exception) -> bool:
    """True only for cross-process file-lock contention — never for real faults."""
    msg = str(exc).lower()
    return (
        "being used by another process" in msg          # Windows lock message
        or "file is already open" in msg                # DuckDB's own hint
        or "could not set lock" in msg                  # POSIX flock failure
        or "conflicting lock is held" in msg
    )


def _open(db_path: str, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection, retrying transient cross-process lock errors.

    Backoff is exponential with random jitter so two processes contending for
    the same file don't retry in lockstep (which would livelock the handoff).
    """
    delay = _BASE_DELAY
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return duckdb.connect(db_path, read_only=read_only)
        except Exception as exc:  # noqa: BLE001 — re-raised below unless it's a lock
            if not _is_lock_error(exc):
                raise
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            # jitter in [0.5, 1.0) of the current delay
            time.sleep(min(delay, _MAX_DELAY) * (0.5 + random.random() * 0.5))
            delay = min(delay * 1.6, _MAX_DELAY)
    # Exhausted the retry budget — surface the original lock error.
    raise last_exc  # type: ignore[misc]


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
        conn = _open(str(self._db_path), read_only=_read_only())
        try:
            yield conn
        finally:
            conn.close()

    def raw_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Non-managed connection for Streamlit's @st.cache_resource.
        Caller is responsible for closing.
        """
        return _open(str(self._db_path), read_only=_read_only())
