"""Pins the batched Rotation Clock walk-forward against the per-date path.

get_rotation_clock_accuracy used to call get_sector_rotation_timeframe once per
signal date, and each of those issued its own ~200-day sector query plus a Nifty
query — ~42 dates x 2 queries x 3 windows, 24.6s of the panel's 29.6s. It now
slices one prefetched panel and answers the index return from one prefetched
series.

Two things have to hold for that to be safe, and neither is obvious from
reading the diff:

  1. The clock arithmetic must have exactly ONE implementation. If the backtest
     grows its own copy, the validation table stops describing the clock it sits
     under — which has happened in this repo before.
  2. _nifty_ret_from_closes must return None in exactly the cases the query
     version does. Its 7-day floor is what makes a gap in index_data produce
     None instead of silently reaching back to an older close, and index_data
     has 14 known continuity breaks.

These run offline on synthetic series. The end-to-end equality (identical dicts
across four as-of dates x three windows, plus early-history dates reaching into
the 2017 gaps) was verified against the pre-batch implementation at the time of
the change; it needs the real database and so is not repeated here.
"""
from __future__ import annotations

import inspect
from datetime import date, timedelta

import pandas as pd
import pytest

from src.analytics import sector_rotation as sr


def _series(pairs):
    if not pairs:
        return pd.Series(dtype="float64")
    idx = [d for d, _ in pairs]
    return pd.Series([v for _, v in pairs], index=idx)


D = date(2026, 6, 15)


def test_basic_window_return():
    s = _series([(D, 100.0), (D + timedelta(days=5), 110.0)])
    assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) == 10.0


def test_start_is_the_close_on_or_before_from_date():
    """Not the day before the window — the shipped definition, reproduced."""
    s = _series([(D - timedelta(days=1), 50.0), (D, 100.0),
                 (D + timedelta(days=5), 110.0)])
    assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) == 10.0


def test_gap_wider_than_seven_days_before_from_date_gives_none():
    """The trap. An older close must NOT be reached for.

    Mirrors the query version's `trade_date >= from_date - 7 days`.
    """
    s = _series([(D - timedelta(days=30), 90.0), (D + timedelta(days=5), 110.0)])
    assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) is None


def test_gap_within_seven_days_still_resolves():
    s = _series([(D - timedelta(days=6), 100.0), (D + timedelta(days=5), 110.0)])
    assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) == 10.0


@pytest.mark.parametrize("closes", [None, []])
def test_missing_series_gives_none(closes):
    s = None if closes is None else _series(closes)
    assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) is None


def test_no_rows_in_window_gives_none():
    s = _series([(D + timedelta(days=90), 110.0)])
    assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) is None


def test_zero_or_negative_start_gives_none():
    for bad in (0.0, -10.0):
        s = _series([(D, bad), (D + timedelta(days=5), 110.0)])
        assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) is None


def test_result_is_rounded_to_two_places():
    s = _series([(D, 100.0), (D + timedelta(days=5), 100.0 * (1 + 0.123456))])
    assert sr._nifty_ret_from_closes(s, D, D + timedelta(days=5)) == 12.35


# ── the single-implementation guard ──────────────────────────────────────────

def test_backtest_and_clock_share_one_panel_query():
    """Both paths must read _ROTATION_PANEL_SQL, not a private copy."""
    src = inspect.getsource(sr.get_rotation_clock_accuracy)
    assert "_ROTATION_PANEL_SQL" in src, (
        "the walk-forward no longer uses the clock's own panel SQL — a second "
        "copy means the validation table can drift from the clock above it")
    assert "SUM(b.deliv_per" not in src, (
        "the walk-forward has inlined its own sector-panel SQL")


def test_backtest_reuses_the_clock_computation():
    src = inspect.getsource(sr.get_rotation_clock_accuracy)
    assert "_rotation_clock_from_panel" in src, (
        "the walk-forward must call the same phase computation the panel "
        "renders, not reimplement it")


def test_lookback_is_not_duplicated():
    """The slice span and the query span must come from one place."""
    assert sr._rotation_lookback_cal(5) == max(5 * 3 + 45, 200)
    assert sr._rotation_lookback_cal(22) == max(22 * 3 + 45, 200)
    for fn in (sr.get_rotation_clock_accuracy, sr.get_sector_rotation_timeframe):
        src = inspect.getsource(fn)
        assert "_rotation_lookback_cal" in src, f"{fn.__name__} hardcodes the lookback"
        assert "* 3 + 45" not in src, f"{fn.__name__} inlines the lookback formula"


def test_panel_lower_bound_is_strict():
    """The SQL uses `> ?`; the pandas slice in the backtest must match."""
    assert "b.trade_date > ?" in sr._ROTATION_PANEL_SQL
    src = inspect.getsource(sr.get_rotation_clock_accuracy)
    assert '_spanel["trade_date"] >' in src and '_spanel["trade_date"] >=' not in src, (
        "the prefetched panel is sliced with >=, but the query it replaced used "
        "a strict >; the earliest day of every window would be included twice")
