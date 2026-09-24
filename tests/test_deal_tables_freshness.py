"""Guards for two defects that were invisible from the UI.

1. `deal_event_outcomes` required a CLOSED 21-session forward window, which silently
   dropped every event of the last ~21 sessions. The table sat a month behind the raw
   feed while the page rendered it as current, so "Recent deals & how they aged" had
   no recent deals at all. Unresolved horizons must be NULL, never a missing row.

2. Nothing rebuilt the analytics tables after the daily fetch, so the gap widened
   every day. The rebuild now runs inside run_daily_pipeline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.repository import query_dataframe


def _max_date(table: str):
    try:
        df = query_dataframe(f"SELECT max(trade_date) AS d FROM {table}", [])
    except Exception:
        pytest.skip(f"{table} not present")
    if df.empty or df["d"].iat[0] is None:
        pytest.skip(f"{table} empty")
    return pd.Timestamp(df["d"].iat[0])


def test_outcomes_keeps_pace_with_the_raw_deal_feed():
    raw = _max_date("bulk_block_deals")
    out = _max_date("deal_event_outcomes")
    gap = (raw - out).days
    assert gap <= 7, (
        f"deal_event_outcomes stops {gap} days before the raw feed "
        f"({out.date()} vs {raw.date()}). An event is only excluded when it has no "
        "NEXT session to enter on; requiring a closed forward window instead drops "
        "the most recent month and the page shows stale 'recent' deals."
    )


def test_client_events_keeps_pace_with_the_raw_deal_feed():
    raw = _max_date("bulk_block_deals")
    cev = _max_date("deal_client_events")
    assert (raw - cev).days <= 7


def test_recent_events_have_rows_but_unresolved_horizons():
    """The recent tail must EXIST with blank returns, not be absent."""
    # deal_event_outcomes carries m21/m42 only; the 5-horizon set lives in
    # deal_client_events. Assert on the longest horizon this table actually has.
    df = query_dataframe(
        "SELECT m21, m42 FROM deal_event_outcomes "
        "WHERE trade_date > (SELECT max(trade_date) - 14 FROM deal_event_outcomes)", [])
    if df.empty:
        pytest.skip("no recent rows")
    assert df["m42"].isna().all(), "a 2-month window cannot have closed for last fortnight"


def test_daily_pipeline_rebuilds_the_deal_tables():
    """The rebuild must be wired into the daily routine, not left to a human."""
    import inspect

    from src import cli

    src = inspect.getsource(cli.run_daily_pipeline)
    assert "build_deal_outcomes" in src and "build_client_events" in src, (
        "run_daily_pipeline no longer rebuilds the bulk/block analytics tables; "
        "they will drift behind the raw feed again"
    )


def test_entity_key_merges_space_only_spellings():
    from scripts.build_client_events import normalise

    assert normalise("ADITYA BIRLA SUN LIFE MUTUAL FUND") == \
           normalise("ADITYA BIRLA SUNLIFE MUTUAL FUND")
    assert normalise("BIRLA SUN LIFE INSURANCE") == normalise("BIRLA SUNLIFE INSURANCE")
    assert normalise("QE SECURITIES LLP") == normalise("QE SECURITIES")
    # ...but genuinely different firms must stay apart
    assert normalise("ABAKKUS ASSET MANAGER LLP") != normalise("ABAKKUS DIVERSIFIED ALPHA FUND")


def test_no_session_is_missing_from_daily_data():
    """A deal date with no price row means the bhavcopy ingest silently skipped a
    real session.

    How this was found: 17 Sep 2026 was absent from daily_data, index_data, the F&O
    bhavcopy and fii_dii_cash, while bulk_block_deals carried 216 rows for it. NSE
    served a 204 KB bhavcopy for that date, so the market had traded and the ingest
    had simply not run. Nothing surfaced it: the scored tables just skipped the day,
    the date picker's max_value is derived from those tables, and the page therefore
    greyed out 17 and 18 Sep as though NSE had published nothing.
    """
    df = query_dataframe(
        "SELECT DISTINCT b.trade_date FROM bulk_block_deals b "
        "LEFT JOIN (SELECT DISTINCT trade_date FROM daily_data) d "
        "  ON d.trade_date = b.trade_date "
        "WHERE d.trade_date IS NULL "
        "  AND b.trade_date >= (SELECT min(trade_date) FROM daily_data) "
        "ORDER BY 1", [])
    assert df.empty, (
        "sessions present in the deal feed but missing from daily_data - the price "
        f"ingest skipped them: {[str(x)[:10] for x in df.trade_date]}"
    )
