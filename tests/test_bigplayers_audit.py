"""Regression tests for the 2026-11-16 audit of the Big Players analytics."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from tests.test_positioning_cycle import _seed, _sessions


def test_turn_window_is_forward_only():
    from src.analytics.positioning_cycle import _near
    sig = np.array([10])
    assert _near(sig, np.array([8])) == 0.0          # turn 2 sessions BEFORE the signal: not "coming"
    assert _near(sig, np.array([10])) == 0.0         # same day: already happened when the file lands
    assert _near(sig, np.array([11])) == 100.0
    assert _near(sig, np.array([20])) == 100.0
    assert _near(sig, np.array([21])) == 0.0


def test_percentile_restarts_at_regime_break():
    from src.analytics.positioning_cycle import _pct_rank, REGIME_BREAKS
    brk = REGIME_BREAKS[0]
    idx = pd.bdate_range(brk - pd.Timedelta(days=500), brk + pd.Timedelta(days=400))
    rng = np.random.default_rng(0)
    vals = np.where(idx < brk, 45 + rng.normal(0, 3, len(idx)), 11 + rng.normal(0, 2, len(idx)))
    pr = _pct_rank(pd.Series(vals, index=idx))
    post = pr[idx >= brk]
    assert post.iloc[:100].isna().all()               # not enough post-break history yet
    late = post.dropna()
    assert len(late) > 50
    assert 20 < late.median() < 80                    # ranked within its own era, not pinned near 0


def test_expiry_calendar_fallback_before_bhavcopy(temp_db):
    from src.analytics.positioning_cycle import expiry_calendar
    sess = pd.DatetimeIndex([d for d in pd.bdate_range("2019-01-01", "2019-03-31")
                             if d != pd.Timestamp("2019-03-28")])   # pretend last Thu of Mar is a holiday
    cal = expiry_calendar(date(2019, 3, 31), sess)
    assert date(2019, 1, 31) in cal                   # last Thursday
    assert date(2019, 2, 28) in cal
    assert date(2019, 3, 27) in cal                   # holiday -> previous session
    assert date(2019, 3, 28) not in cal


def test_cache_sees_new_rows_without_restart(temp_db):
    from src.analytics import big_players as bp
    days = _sessions(date(2024, 1, 1), 300)
    _seed(days[:-1])
    a1 = bp.get_player_actions(days[-1])
    _seed(days)                                       # tonight's file lands
    a2 = bp.get_player_actions(days[-1])
    assert a1["data_date"] == days[-2]
    assert a2["data_date"] == days[-1]


def test_partial_first_leg_and_extreme():
    from src.analytics.big_players import _swing_legs
    close = np.array([100, 101, 104, 108, 110, 106, 102, 100, 103, 107, 106], float)
    share = np.linspace(10, 20, len(close))
    idx = pd.date_range("2026-01-01", periods=len(close), freq="B")
    legs = _swing_legs(idx, close, share, share, None, 0.03)
    assert legs[0]["partial"] is True
    assert all(not g["partial"] for g in legs[1:])
    cur = legs[-1]
    assert cur["open"] and cur["extreme_close"] == 107.0   # reversal level must use the high, not today's 106
