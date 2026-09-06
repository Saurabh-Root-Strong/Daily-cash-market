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


# ── round-2 audit regressions ────────────────────────────────────────────────

def test_purge_gap_keeps_last_weeks_sessions_out_of_the_analogue_set():
    """Measured: without a purge, 34.1% of sampled days had one of their 25
    'analogues' within 10 calendar days, the closest 1 day away. A neighbour from
    last week shares most of its forward window with today."""
    d = _dt.date(2026, 9, 3)
    r = ilc.get_flow_analogues(d, "NIFTY", 25, purge=5)
    m = pd.to_datetime(r["matches"]["date"])
    assert (pd.Timestamp(d) - m).dt.days.min() > 7, \
        "a purged lookup must not return a match from the last few sessions"
    loose = ilc.get_flow_analogues(d, "NIFTY", 25, purge=0)
    assert len(loose["matches"]) == 25, "purge=0 must still return k matches"


def test_k_sensitivity_is_published_so_an_unstable_number_cannot_hide():
    """k=25 is a choice. On 3 Sep 2026 the next-day mean ran -0.095% at k=10 and
    +0.149% at k=40 — a SIGN flip — so the sweep has to travel with the answer."""
    r = ilc.get_flow_analogues(_dt.date(2026, 9, 3), "NIFTY", 25)
    sw = r["summary"]["k_sweep"]
    assert set(sw) >= {10, 25, 60}, "the k sweep must span the plausible range"
    for h in ("d1", "d5"):
        lo, hi = r["summary"][h]["k_range"]
        assert lo <= r["summary"][h]["mean"] <= hi, "the point estimate must sit in its own range"
        assert isinstance(r["summary"][h]["k_sign_flips"], bool)
    assert r["summary"]["d1"]["k_sign_flips"] is True, (
        "3 Sep 2026 is the worked example of an unstable horizon; if this ever "
        "goes False the sweep has stopped detecting instability")


def test_net_score_falls_back_to_equal_weight_on_unknown_labels():
    """An unrecognised bucket label used to be dropped silently, rendering a blank
    score with no explanation. An approximate blend beats a mystery blank."""
    d = ilc.IndexLargeCap(trade_date=None, fno_symbol="X", display="x")
    d.rows = [ilc.BucketRow(label="Mega caps", n_members=10, n_present=10, deliv_z=2.0),
              ilc.BucketRow(label="The rest", n_members=40, n_present=40, deliv_z=-2.0)]
    assert d.net_score == pytest.approx(0.0), "unknown labels must still produce a score"


def test_flow_state_matrix_query_is_fast_enough_for_an_eager_expander():
    """Streamlit evaluates expander BODIES eagerly, and this one sits in an
    expander. The pandas-merge version cost 19.8s of a 20.2s panel render."""
    import time
    t = time.time()
    Z, _ = ilc._flow_state_matrix("NIFTY", _dt.date(2026, 9, 3))
    el = time.time() - t
    assert not Z.empty
    assert el < 8.0, f"flow state matrix took {el:.1f}s — the panel renders it eagerly"


# ── points + per-bucket matching ─────────────────────────────────────────────

def test_points_are_quoted_on_todays_level_not_the_analogue_date():
    """NIFTY ran ~17,000 in 2022 and ~23,900 now. A raw historical point move is
    not comparable across the window, so % is rescaled onto today's close."""
    d = _dt.date(2026, 9, 4)
    r = ilc.get_flow_analogues(d, "NIFTY", 25)
    spot = r["spot"]
    assert spot and 20000 < spot < 30000, f"implausible spot {spot}"
    for h in ("d1", "d5"):
        s = r["summary"][h]
        assert s["mean_pts"] == pytest.approx(s["mean"] / 100 * spot)
        assert s["best_pts"] == pytest.approx(s["best"] / 100 * spot)
    m = r["matches"]
    # compare only where the forward return exists; a match near the end of the
    # sample can have a NaN horizon and NaN != NaN breaks a naive elementwise check
    ok = m["d1"].notna()
    assert ok.any()
    import numpy as _np
    assert _np.allclose(m.loc[ok, "d1_pts"], m.loc[ok, "d1"] / 100 * spot)


def test_points_degrade_to_none_when_the_index_row_is_missing():
    """index_data lagged daily_data by a session at least once this month. A
    missing close must hide the points column, never render 0."""
    import src.analytics.index_largecap as m
    real = m._index_close
    try:
        m._index_close = lambda *a, **k: None
        r = m.get_flow_analogues(_dt.date(2026, 9, 3), "NIFTY", 25)
        assert r["ok"] and r["spot"] is None
        assert r["summary"]["d1"]["mean_pts"] is None
        assert r["summary"]["d1"]["mean"] is not None, "the % answer must survive"
    finally:
        m._index_close = real


def test_bucket_analogues_match_each_bucket_on_its_own_three_legs():
    r = ilc.get_bucket_analogues(_dt.date(2026, 9, 4), "NIFTY", 25)
    assert r["ok"]
    assert set(r["buckets"]) == {"Top 10", "Next 10", "Rest 30"}
    for bn, blk in r["buckets"].items():
        assert blk["n"] == 25
        for h in ("d1", "d5"):
            assert blk[h]["mean"] is not None
            assert "k_range" in blk[h]
    # matching on 3 legs must be LOOSER than on all 9, so the k-th neighbour is nearer
    whole = ilc.get_flow_analogues(_dt.date(2026, 9, 4), "NIFTY", 25)
    for bn, blk in r["buckets"].items():
        assert blk["kth"] < whole["summary"]["match_quality"]["kth"], (
            f"{bn}: a 3-dim match cannot be farther than the 9-dim one")


def test_bucket_analogue_docstring_records_that_narrowing_does_not_help():
    doc = ilc.get_bucket_analogues.__doc__ or ""
    assert "-0.008" in doc and "50.0%" in doc, "the TOP10-only null must stay recorded"
