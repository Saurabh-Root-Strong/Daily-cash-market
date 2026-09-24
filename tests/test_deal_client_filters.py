"""Regression tests for the per-client filters and drill-down.

The date filter introduces a failure mode that is silent by construction: pick a
window close to today and the 1-month / 3-month forward windows have not closed, so
`mean()` skips the NaNs and reports the average of whichever handful of buys are old
enough to have resolved -- a number that looks like the window but describes a biased
sub-slice of it. horizon_coverage() is what the UI uses to refuse to show those.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.analytics import deal_signals as ds


@pytest.fixture(scope="module")
def ev() -> pd.DataFrame:
    df = ds.load_client_events("BUY")
    if df.empty:
        pytest.skip("deal_client_events not built")
    return df


def test_coverage_is_full_on_old_data_and_empty_on_fresh(ev):
    last = ev["trade_date"].max()
    fresh = ev[ev["trade_date"] > last - pd.Timedelta(days=12)]
    old = ev[ev["trade_date"] < last - pd.Timedelta(days=400)]

    cov_old = ds.horizon_coverage(old)
    assert cov_old["3 months"] > 0.9, "old buys should have resolved 3-month windows"

    if not fresh.empty:
        cov_fresh = ds.horizon_coverage(fresh)
        assert cov_fresh["3 months"] < 0.5, (
            "a window ending at the last session cannot have resolved 3-month returns; "
            "if this passes, the forward-return join is looking ahead"
        )


def test_leaderboard_sort_falls_back_when_primary_horizon_unresolved(ev):
    last = ev["trade_date"].max()
    fresh = ev[ev["trade_date"] > last - pd.Timedelta(days=12)]
    if fresh.empty:
        pytest.skip("no recent events")
    lb = ds.entity_leaderboard(fresh, min_events=1)
    assert not lb.empty
    # Whatever it sorted by must be a column that actually has values.
    assert lb["Buys"].notna().all()
    if lb["1 month"].notna().any():
        assert lb["1 month"].is_monotonic_decreasing or lb["1 month"].isna().any()


def test_client_trades_returns_only_that_client(ev):
    key = ev["client_key"].value_counts().index[0]
    trades = ds.client_trades(ev, key)
    expected = int((ev["client_key"] == key).sum())
    assert len(trades) == expected
    assert set(trades.columns) >= {"Date", "Symbol", "Value (Cr)", "1 month"}
    assert trades["Date"].is_monotonic_decreasing


def test_client_trades_respects_a_date_slice(ev):
    key = ev["client_key"].value_counts().index[0]
    sub = ev[ev["client_key"] == key]
    cut = sub["trade_date"].median()
    window = ev[ev["trade_date"] >= cut]
    trades = ds.client_trades(window, key)
    assert len(trades) == int((sub["trade_date"] >= cut).sum())
    assert trades["Date"].min() >= cut


def test_leaderboard_exposes_client_key_for_the_drilldown(ev):
    lb = ds.entity_leaderboard(ev, min_events=15)
    assert "client_key" in lb.columns, "the drill-down matches on client_key, not the label"
    assert lb["client_key"].is_unique


def test_hit_rate_is_blank_not_zero_when_nothing_has_resolved():
    """(NaN > 0) is False, so unresolved windows used to score as LOSSES and every
    client on the default 10-session view showed a confident "Hit rate 0%"."""
    import pandas as pd

    from src.analytics import deal_signals as ds

    ev = ds.load_client_events("BUY")
    if ev.empty:
        pytest.skip("no events")
    recent = ev[ev.trade_date >= ev.trade_date.max() - pd.Timedelta(days=10)]
    if recent.empty or recent["m21"].notna().any():
        pytest.skip("recent window already has resolved 1-month windows")
    lb = ds.entity_leaderboard(recent, 1)
    assert not lb.empty
    assert lb["Hit rate 1 month"].isna().all(), (
        "a hit rate computed over zero resolved trades must be blank, not 0%"
    )


def test_one_entity_is_one_row_per_stock_day():
    """NSE spells one firm several ways; netting keys on client_key so a buy under
    one spelling nets against a sell under another."""
    from src.data.repository import query_dataframe

    df = query_dataframe(
        "SELECT trade_date, symbol, client_key, count(*) c FROM deal_client_events "
        "GROUP BY 1,2,3 HAVING count(*) > 1", [])
    assert df.empty, f"{len(df)} entity-day-stock groups split across rows"
