"""Guards for the failure that cost a whole trading session, silently.

On 2026-09-17 the daily job died on a DuckDB lock held by the running dashboard.
Three independent defects then conspired to hide it for four days:

  1. run_daily_job walks back only until ONE session succeeds, so the failed day was
     never revisited -- the next morning it fetched 18 Sep and stopped.
  2. run_health_check took its trading calendar FROM daily_data, the very table with
     the hole, so the missing session was "not a trading day" and every source
     scored complete. The daily log printed "Data health OK".
  3. The deal page derives its date picker from the scored tables, so the session
     simply never appeared as an option -- indistinguishable from a market holiday.

These tests pin the two structural repairs.
"""
from __future__ import annotations

import inspect
from datetime import date, timedelta

import pytest

from src.data.repository import query_dataframe


def test_health_calendar_is_corroborated_across_feeds():
    """daily_data must not be its own yardstick."""
    from src.data import health

    src = inspect.getsource(health.run_health_check)
    for table in ("index_data", "fno_bhavcopy", "bulk_block_deals"):
        assert table in src, (
            f"{table} is no longer part of the corroborated trading calendar; a hole "
            "in daily_data would again be invisible"
        )
    assert "daily_data" in health._SOURCES, "bhavcopy must itself be a CHECKED source"


def test_health_sees_sessions_that_only_the_deal_feed_recorded():
    """A session any feed saw is a trading day, even if daily_data lost it."""
    from src.data.health import run_health_check

    h = run_health_check(lookback_days=10)
    cal = {d.date() if hasattr(d, "date") else d for d in h.trading_days}
    if not cal:
        pytest.skip("no recent data")
    deal_days = query_dataframe(
        "SELECT DISTINCT trade_date FROM bulk_block_deals "
        "WHERE trade_date >= ? ORDER BY trade_date", [min(cal)])["trade_date"]
    deal_days = {d.date() if hasattr(d, "date") else d for d in deal_days}
    missed = deal_days - cal
    assert not missed, f"deal sessions absent from the health calendar: {missed}"


def test_daily_job_repairs_orphaned_sessions():
    from src.ingestion import orchestrator

    assert hasattr(orchestrator, "_repair_missing_sessions")
    assert "_repair_missing_sessions" in inspect.getsource(orchestrator.run_daily_job), (
        "run_daily_job no longer repairs missed sessions; a transient failure would "
        "orphan that day permanently again"
    )


def test_repair_refetches_a_hole(monkeypatch):
    """Functional: a date seen by another feed but absent from daily_data is fetched."""
    import pandas as pd

    from src.ingestion import orchestrator

    hole = date(2026, 9, 17)
    other = date(2026, 9, 18)

    class FakeRepo:
        def query(self, sql, params=None):
            return pd.DataFrame({"trade_date": [hole, other]})

        def get_dates_present(self, dates):
            return [d for d in dates if d != hole]   # the hole is missing

    called = []

    def fake_fetch(d, client, repo):
        called.append(d)
        return "success", 3104

    monkeypatch.setattr(orchestrator, "fetch_one_date", fake_fetch)
    orchestrator._repair_missing_sessions(client=object(), repo=FakeRepo())
    assert called == [hole], f"expected a re-fetch of {hole}, got {called}"


def test_no_hole_in_the_current_calendar():
    """End-to-end: every session any feed saw must be in daily_data."""
    df = query_dataframe(
        """
        SELECT DISTINCT trade_date FROM (
            SELECT trade_date FROM index_data       WHERE trade_date >= ?
            UNION SELECT trade_date FROM fno_bhavcopy     WHERE trade_date >= ?
            UNION SELECT trade_date FROM bulk_block_deals WHERE trade_date >= ?
        ) s
        WHERE trade_date NOT IN (SELECT DISTINCT trade_date FROM daily_data)
        ORDER BY 1
        """,
        [date.today() - timedelta(days=60)] * 3)
    assert df.empty, f"sessions missing from daily_data: {[str(d)[:10] for d in df.trade_date]}"
