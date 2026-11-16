"""Big Players analytics tests — temp DuckDB, reuses the positioning-cycle seeder."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from tests.test_positioning_cycle import _seed, _sessions


def _fresh():
    from src.analytics import big_players as bp, positioning_cycle as pc
    bp._load_cached.cache_clear()
    pc._frame_cached.cache_clear()
    return bp


@pytest.fixture
def days():
    return _sessions(date(2021, 1, 1), 400)   # clear of the Nov-2024 regime break


def test_flow_state_labels():
    from src.analytics.big_players import _flow_state
    assert _flow_state(100, 10) == "LB"
    assert _flow_state(-100, 10) == "LU"
    assert _flow_state(10, 100) == "SB"
    assert _flow_state(10, -100) == "SC"
    assert _flow_state(0, 0) == "FLAT"
    assert _flow_state(np.nan, 5) == "FLAT"


def test_actions_zero_sum_and_absorber(temp_db, days):
    _seed(days)
    bp = _fresh()
    a = bp.get_player_actions(days[-1])
    assert not a.get("error"), a.get("error")
    assert a["total_net"] == pytest.approx(0, abs=1e-6)
    fii = next(p for p in a["players"] if p["name"] == "FII")
    if fii["net_change"]:
        assert a["absorb"], "someone must take the other side of a non-zero FII change"
        assert sum(a["absorb"].values()) == pytest.approx(100)
        assert "FII" not in a["absorb"]


def test_actions_flags_incomplete_file(temp_db, days):
    _seed(days, imbalance_on=days[-1])
    bp = _fresh()
    a = bp.get_player_actions(days[-1])
    assert any("do not sum to zero" in w for w in a["warnings"])


def test_playbook_today_in_exactly_one_scenario(temp_db, days):
    _seed(days)
    bp = _fresh()
    pb = bp.get_scenario_playbook(days[-1])
    assert not pb.get("error"), pb.get("error")
    assert len(pb["scenarios"]) == 4
    assert sum(s["today"] for s in pb["scenarios"]) <= 1
    assert len(pb["detail"]) == 8
    assert pb["usable"] <= pb["sessions"]


def test_playbook_short_history_errors_cleanly(temp_db):
    d = _sessions(date(2024, 1, 1), 60)
    _seed(d)
    bp = _fresh()
    assert "error" in bp.get_scenario_playbook(d[-1])


def test_player_history_unknown_and_window(temp_db, days):
    _seed(days)
    bp = _fresh()
    assert "error" in bp.get_player_history(days[-1], "XYZ")
    h = bp.get_player_history(days[-1], "FII", sessions=100)
    assert len(h["df"]) == 100
    assert h["df"]["date"].iloc[-1].date() == days[-1]


def test_holiday_rows_dropped(temp_db, days):
    hol = date(2021, 6, 12)
    _seed(days, holiday=hol)
    bp = _fresh()
    h = bp.get_player_history(days[-1], "FII", sessions=400)
    assert hol not in set(h["df"]["date"].dt.date)


def test_swing_legs_classify_by_fii_share_change():
    import pandas as pd
    from src.analytics.big_players import _swing_legs
    close = np.array([100, 104, 108, 110, 106, 102, 100, 103, 106], float)
    share = np.array([10, 12, 14, 15, 13, 11, 10, 10.2, 10.5], float)   # FII follows, then flat
    idx = pd.date_range("2026-01-01", periods=len(close), freq="B")
    legs = _swing_legs(idx, close, share, share * 1000, 100 - share, 0.03)
    assert [g["dir"] for g in legs] == [1, -1, 1]
    assert legs[0]["scenario"] == "FII-backed rally"
    assert legs[1]["scenario"] == "FII-backed fall"
    assert legs[2]["scenario"] == "Rally, FII on sidelines"      # +0.5 pts < 1-point threshold
    assert legs[2]["open"] and not legs[0]["open"]
    assert legs[0]["client_share_chg"] < 0                       # client on the other side
