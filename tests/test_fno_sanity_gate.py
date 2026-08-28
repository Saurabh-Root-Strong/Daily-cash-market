"""The F&O write-time sanity gate.

Regression cover for 2026-08-27, which landed in the archive with futures open
interest 98.6% below the prior session and traded contracts 207x above it. The
prices in that file were correct, so nothing looked obviously broken; and the
expiry-cycle read is peer-relative, so a whole-session error normalises away --
every name equally wrong, no outliers, every analytics guard satisfied. The
board rendered it as a normal session at stale_days=0.

The write is the only layer that can still see the anomaly, so that is where it
is caught.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.repository import get_repository, query_dataframe


def _session(d: date, oi: int, n: int = 20) -> pd.DataFrame:
    """A minimal but schema-complete FUTSTK session."""
    return pd.DataFrame({
        "trade_date": [d] * n,
        "instrument": ["FUTSTK"] * n,
        "symbol": [f"SYM{i}" for i in range(n)],
        "expiry_date": [date(d.year, d.month, 28)] * n,
        "strike_price": [0.0] * n,
        "option_type": ["XX"] * n,
        "open_price": [100.0] * n,
        "high_price": [101.0] * n,
        "low_price": [99.0] * n,
        "close_price": [100.5] * n,
        "settle_price": [100.0] * n,
        "contracts": [1000] * n,
        "value_lacs": [10.0] * n,
        "open_interest": [oi // n] * n,
        "chg_in_oi": [0] * n,
    })


def test_first_session_is_never_gated(temp_db):
    """Nothing to compare against — a backfill must not be blocked."""
    assert get_repository().upsert_fno_bhavcopy(_session(date(2026, 8, 26), 20_000_000)) == 20


def test_normal_session_passes(temp_db):
    repo = get_repository()
    repo.upsert_fno_bhavcopy(_session(date(2026, 8, 26), 20_000_000))
    # ~5% move, the largest observed across the violent August 2026 roll week
    assert repo.upsert_fno_bhavcopy(_session(date(2026, 8, 27), 19_000_000)) == 20


def test_the_2026_08_27_corruption_is_rejected(temp_db):
    repo = get_repository()
    repo.upsert_fno_bhavcopy(_session(date(2026, 8, 26), 20_000_000))
    with pytest.raises(ValueError, match="sanity gate"):
        repo.upsert_fno_bhavcopy(_session(date(2026, 8, 27), 280_000))  # 0.014x, as observed


def test_rejected_session_leaves_the_archive_untouched(temp_db):
    """The gate runs BEFORE the DELETE, so a bad file cannot destroy a good day."""
    repo = get_repository()
    repo.upsert_fno_bhavcopy(_session(date(2026, 8, 26), 20_000_000))
    with pytest.raises(ValueError):
        repo.upsert_fno_bhavcopy(_session(date(2026, 8, 26), 100_000))
    kept = query_dataframe(
        "SELECT SUM(open_interest) AS oi FROM fno_bhavcopy WHERE trade_date = ?",
        [date(2026, 8, 26)],
    )
    assert int(kept["oi"].iloc[0]) == 20_000_000


def test_override_forces_the_write(temp_db, monkeypatch):
    repo = get_repository()
    repo.upsert_fno_bhavcopy(_session(date(2026, 8, 26), 20_000_000))
    monkeypatch.setenv("FNO_SKIP_SANITY", "1")
    assert repo.upsert_fno_bhavcopy(_session(date(2026, 8, 27), 280_000)) == 20
