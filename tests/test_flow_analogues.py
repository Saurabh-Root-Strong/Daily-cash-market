import datetime as _dt
import pandas as pd
import pytest
from src.analytics import index_largecap as ilc


def test_analogues_never_match_today_or_the_future():
    """The whole engine is invalid if a neighbour is not strictly in the past."""
    r = ilc.get_flow_analogues(_dt.date(2026, 9, 3), "NIFTY", 25)
    assert r["ok"], r["note"]
    d = pd.to_datetime(r["matches"]["date"])
    assert (d < pd.Timestamp("2026-09-03")).all(), "a neighbour leaked from >= today"
    assert len(d) == len(set(d)), "a session was matched twice"


def test_analogue_distances_are_sorted_and_nearest_beats_random():
    r = ilc.get_flow_analogues(_dt.date(2026, 9, 3), "NIFTY", 25)
    dist = r["matches"]["distance"].tolist()
    assert dist == sorted(dist), "matches must come back nearest-first"
    mq = r["summary"]["match_quality"]
    assert mq["ratio"] < 1.0, (
        "the k-th neighbour must be closer than a typical pair, or the matching "
        "is doing nothing at all")


def test_state_matrix_survives_the_expiry_session_gaps():
    """min_periods regression. Settlement sessions are excluded from the futures
    and options legs, so ~5% of rows are NaN; pandas' default min_periods=window
    means NO 250-row window ever qualifies and the matrix came back EMPTY."""
    Z, fwd = ilc._flow_state_matrix("NIFTY", _dt.date(2026, 9, 3))
    assert not Z.empty, "state matrix is empty — check min_periods on the z windows"
    assert Z.shape[1] == 9, f"expected 9 flow dims, got {Z.shape[1]}"
    for leg in ("delivery", "futures", "options"):
        cols = [c for c in Z.columns if c.endswith(leg)]
        assert len(cols) == 3, f"{leg} leg missing for a bucket"
        assert Z[cols].notna().all().all(), f"{leg} legs carry NaNs after dropna"


def test_analogue_docstring_keeps_both_halves_of_the_verdict():
    """The matching works AND it does not predict. Dropping either half makes the
    panel misleading in one direction or the other."""
    doc = ilc.get_flow_analogues.__doc__ or ""
    assert "1.444%" in doc and "1.462%" in doc, "nearest-vs-farthest test missing"
    assert "7.1%" in doc, "match-quality evidence missing"
    assert "RANDOM" in doc


def test_unknown_index_and_settlement_session_degrade_cleanly():
    assert not ilc.get_flow_analogues(_dt.date(2026, 9, 3), "BANKNIFTY")["ok"]
    r = ilc.get_flow_analogues(_dt.date(2026, 8, 25), "NIFTY")
    if not r["ok"]:
        assert "settlement" in r["note"] or "flow state" in r["note"]
