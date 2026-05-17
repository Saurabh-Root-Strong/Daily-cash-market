from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import duckdb


def _db_path() -> Path:
    from src.config_loader import get_db_path
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    conn = duckdb.connect(str(_db_path()))
    try:
        yield conn
    finally:
        conn.close()


def get_raw_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(_db_path()))
