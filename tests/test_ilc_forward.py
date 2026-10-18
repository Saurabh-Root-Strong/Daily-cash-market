"""Forward test of the Index & Large Cap lead: the rules that keep it honest.

Offline: no DuckDB. What is pinned is the bookkeeping a forward test lives or dies
by -- first write wins, late rows never count, the frozen model is the one used,
and a missing input means no forecast rather than a guess.
"""
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from src.analytics import ilc_forward as F
from src.data.forward_log import append_once, read_log


def _row(d="2026-10-14", sym="NIFTY", score="0.1"):
    return {"trade_date": d, "symbol": sym, "logged_at": "2026-10-14T20:10:00",
            "model_version": "v", "score": score, "side": "LONG",
            "cut_low": "-0.1", "cut_high": "0.05", "inputs": "{}"}


def test_first_write_wins(tmp_path):
    p = tmp_path / "log.csv"
    assert append_once(p, _row(score="0.1")) is True
    assert append_once(p, _row(score="9.9")) is False          # a re-run cannot rewrite history
    assert append_once(p, _row(sym="BANKNIFTY")) is True
    log = read_log(p)
    assert len(log) == 2 and log.loc[log.symbol == "NIFTY", "score"].item() == "0.1"


def test_columns_cannot_drift(tmp_path):
    p = tmp_path / "log.csv"
    append_once(p, _row())
    bad = _row(d="2026-10-15"); bad["extra"] = "x"
    with pytest.raises(ValueError):
        append_once(p, bad)


@pytest.mark.parametrize("logged, nxt, want", [
    (datetime(2026, 10, 14, 21, 0), date(2026, 10, 15), True),      # same evening
    (datetime(2026, 10, 15, 9, 14), date(2026, 10, 15), True),      # just before the open
    (datetime(2026, 10, 15, 9, 15), date(2026, 10, 15), False),     # at the open = late
    (datetime(2026, 10, 16, 8, 0), date(2026, 10, 15), False),      # after the session
    (datetime(2026, 10, 17, 8, 0), date(2026, 10, 19), True),       # Friday row, Saturday write
])
def test_on_time_rule(logged, nxt, want):
    td = date(2026, 10, 16) if nxt == date(2026, 10, 19) else date(2026, 10, 14)
    assert F.on_time(td, logged, nxt) is want


def test_on_time_unknown_next_session():
    # next session not on file yet: undecided before the next weekday's open, late after it
    assert F.on_time(date(2026, 10, 14), datetime(2026, 10, 14, 22, 0), None) is None
    assert F.on_time(date(2026, 10, 14), datetime(2026, 10, 15, 10, 0), None) is False
    assert F.on_time(date(2026, 10, 16), datetime(2026, 10, 19, 9, 30), None) is False  # Fri -> Mon


def test_frozen_model_file_is_complete():
    m = F.load_model()
    assert m["version"] and m["preregistration"]["decision_at"]
    for sym in ("NIFTY", "BANKNIFTY"):
        k = m["indices"][sym]
        n = len(k["features"])
        assert n == 16 and len(k["mu"]) == n and len(k["sd"]) == n and len(k["beta"]) == n
        assert k["cuts"][0] < k["cuts"][1] and all(s > 0 for s in k["sd"])


def test_score_and_side_use_the_frozen_numbers():
    m = {"features": ["a", "b"], "mu": [0.0, 1.0], "sd": [1.0, 2.0], "beta": [0.5, -1.0],
         "intercept": 0.01, "cuts": [-0.1, 0.1]}
    s = F.score({"a": 1.0, "b": 1.0}, m)
    assert s == pytest.approx(0.51)
    assert F.side(s, m) == "LONG" and F.side(0.0, m) == "NONE" and F.side(-0.2, m) == "SHORT"
    assert F.score({"a": 999.0, "b": 1.0}, m) == pytest.approx(0.5 * 50 + 0.01)   # inputs clipped


def test_missing_input_means_no_forecast(monkeypatch):
    monkeypatch.setattr(F, "features", lambda d, s: ({"net": None}, "missing: net"))
    f = F.forecast(date(2026, 10, 14), "NIFTY")
    assert f["score"] is None and f["status"].startswith("no forecast")
    assert F.forecast(date(2026, 10, 14), "FINNIFTY")["status"] == "not in the pre-registered set"


def test_model_file_refuses_refit():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "fit_ilc_forward_model.py").read_text(encoding="utf-8")
    assert "already exists" in src and "OUT.exists()" in src
