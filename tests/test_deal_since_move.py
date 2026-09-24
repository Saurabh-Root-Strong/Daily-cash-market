"""Guards for the "move since buy" columns in deal_client_events.

Two traps this has to stay clear of:

1. Splits and bonuses. Raw last_close / deal price turns a 1:10 split into a -90%
   "loss". The move must be carried through the corrected cf chain.
2. Series gaps. A stock pushed into BE (trade-to-trade) vanishes from EQ for weeks.
   An EQ-only chain scores those weeks as 0%, which printed APOLLO at +7,257% since a
   2019 buy against a true ~+4,700%.
"""
from __future__ import annotations

import pytest

from src.data.repository import query_dataframe


def _cols() -> set[str]:
    try:
        df = query_dataframe("SELECT * FROM deal_client_events LIMIT 1", [])
    except Exception:
        pytest.skip("deal_client_events not built")
    return set(df.columns)


def test_since_columns_exist_and_are_filled():
    need = {"since_pct", "nifty_since_pct", "last_close", "px_date", "sessions_since"}
    assert need <= _cols()
    df = query_dataframe(
        "SELECT count(*) n, count(since_pct) s, count(px_date) p "
        "FROM deal_client_events WHERE side = 'BUY'", [])
    assert df.s[0] >= 0.99 * df.n[0], "since_pct should be blank only for bad deal prints"
    assert df.p[0] == df.n[0]


def test_latest_session_buys_have_zero_sessions_and_flat_nifty():
    _cols()
    df = query_dataframe(
        "SELECT sessions_since, nifty_since_pct FROM deal_client_events "
        "WHERE side = 'BUY' AND trade_date = (SELECT max(trade_date) FROM deal_client_events)"
        , [])
    if df.empty:
        pytest.skip("no rows")
    assert (df.sessions_since == 0).all()
    assert df.nifty_since_pct.abs().max() < 1e-9


def test_split_does_not_read_as_a_crash():
    """TEMBO: deal at 568, last close 56 after a split -- raw says -90%."""
    _cols()
    df = query_dataframe(
        "SELECT since_pct, last_close / vwap - 1 AS raw FROM deal_client_events "
        "WHERE side = 'BUY' AND symbol = 'TEMBO' AND trade_date = DATE '2026-07-28'", [])
    if df.empty:
        pytest.skip("TEMBO event not present")
    assert df.raw.iat[0] < -0.8
    assert df.since_pct.iat[0] > -0.5, "split leaked into the since-buy move"


def test_be_series_spells_are_followed():
    """APOLLO spent four spells in BE; true move since the 2019-12-19 buy is ~+4,700%.
    The EQ-only chain gave +7,257%. Allow a generous band either side."""
    _cols()
    df = query_dataframe(
        "SELECT since_pct FROM deal_client_events "
        "WHERE side = 'BUY' AND symbol = 'APOLLO' AND trade_date = DATE '2019-12-19'", [])
    if df.empty:
        pytest.skip("APOLLO event not present")
    assert 30 < df.since_pct.iat[0] < 60, df.since_pct.iat[0]


def test_leaderboard_and_drilldown_expose_since_columns():
    from src.analytics import deal_signals as ds

    ev = ds.load_client_events()
    if ev.empty:
        pytest.skip("no events")
    w = ev[ev.trade_date >= ev.trade_date.max() - __import__("pandas").Timedelta(days=20)]
    lb = ds.entity_leaderboard(w, 1)
    for c in ("Since buy median", "vs Nifty median", "In profit now %"):
        assert c in lb.columns
        assert lb[c].notna().any(), f"{c} blank on a recent window, which is its whole point"
    tr = ds.client_trades(w, lb.client_key.iat[0])
    for c in ("Since buy %", "Nifty same span %", "Sessions since", "Price as of", "Last price"):
        assert c in tr.columns
