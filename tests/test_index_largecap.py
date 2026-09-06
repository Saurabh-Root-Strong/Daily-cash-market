"""Index & Large Cap — the invariants that must not drift.

The panel's whole claim is that the carry spread is an EXACT identity needing no
weight data. These tests pin that, plus the guards that stop a thin bucket or a
missing index row from rendering as a confident read.
"""
import numpy as np
import pandas as pd
import pytest

from src.analytics import index_largecap as ilc


def test_bucket_membership_is_the_full_50_with_no_overlap():
    b = ilc.INDEX_BUCKETS["NIFTY"]["buckets"]
    allsyms = [s for v in b.values() for s in v]
    assert len(allsyms) == 50, f"expected 50 constituents, got {len(allsyms)}"
    assert len(set(allsyms)) == 50, "a symbol appears in more than one bucket"
    assert [len(v) for v in b.values()] == [10, 10, 30]


def test_carry_spread_is_exactly_index_minus_equalweight():
    d = ilc.IndexLargeCap(trade_date=None, fno_symbol="NIFTY", display="x")
    d.index_ret, d.equal_ret = 0.10, -0.14
    d.carry_spread = d.index_ret - d.equal_ret
    assert d.carry_spread == pytest.approx(0.24)


def test_regime_names_the_carried_case():
    d = ilc.IndexLargeCap(trade_date=None, fno_symbol="NIFTY", display="x")
    d.index_ret, d.equal_ret, d.carry_spread = 0.10, -0.14, 0.24
    d.adv_all, d.n_present = 23, 50            # index up, breadth under half
    assert d.regime == "Heavyweights carried a red market"
    d.index_ret, d.adv_all = -0.10, 27         # index down, breadth over half
    assert d.regime == "Heavyweights dragged a green market"


def test_thin_bucket_is_flagged():
    r = ilc.BucketRow(label="Top 10", n_members=10, n_present=5)
    assert r.coverage == 0.5 and r.thin
    r2 = ilc.BucketRow(label="Top 10", n_members=10, n_present=9)
    assert not r2.thin


def test_futures_lean_needs_a_quorum_and_a_clear_majority():
    """A 3-name read is not a basket verdict, and a near-tie is not a lean."""
    thin = ilc.BucketRow(label="b", n_members=10, n_present=10,
                         fut_valid=3, fut_long=3)
    assert thin.fut_lean is None
    tie = ilc.BucketRow(label="b", n_members=10, n_present=10,
                        fut_valid=10, fut_long=5, fut_short=4)
    assert tie.fut_lean == 0
    bull = ilc.BucketRow(label="b", n_members=10, n_present=10,
                         fut_valid=10, fut_long=7, fut_short=2)
    assert bull.fut_lean == 1
    bear = ilc.BucketRow(label="b", n_members=10, n_present=10,
                         fut_valid=10, fut_short=6, fut_unwind=2, fut_long=1)
    assert bear.fut_lean == -1


def test_no_index_row_means_no_carry_spread_not_a_zero():
    """A missing index_data row must hide the spread, never render it as 0.00."""
    d = ilc.IndexLargeCap(trade_date=None, fno_symbol="NIFTY", display="x")
    d.equal_ret = 0.084
    assert d.carry_spread is None
    assert d.regime == "No data"


def test_unknown_index_degrades_cleanly():
    out = ilc.get_index_largecap(__import__("datetime").date(2026, 9, 4), "BANKNIFTY")
    assert not out.data_ok and "BANKNIFTY" in out.note


# ── regressions for the three defects the self-audit found ───────────────────

def test_total_oi_change_is_roll_immune_and_same_expiry_set():
    """Near-month OI falls ~48% at 0-2 DTE because positions MIGRATE to the next
    contract. The total across every live expiry must be flat through that roll."""
    import datetime as _dt
    td, pv = _dt.date(2026, 9, 3), _dt.date(2026, 9, 2)
    fut = pd.DataFrame([
        # front contract bleeds 100 -> 20, next contract absorbs 20 -> 100
        dict(trade_date=pv, symbol="X", expiry_date=_dt.date(2026, 9, 29), open_interest=100),
        dict(trade_date=pv, symbol="X", expiry_date=_dt.date(2026, 10, 27), open_interest=20),
        dict(trade_date=td, symbol="X", expiry_date=_dt.date(2026, 9, 29), open_interest=20),
        dict(trade_date=td, symbol="X", expiry_date=_dt.date(2026, 10, 27), open_interest=100),
    ])
    out = ilc._total_oi_change(fut, td)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["oi"] == 120 and r["prev_oi"] == 120, "the roll must net to zero"


def test_total_oi_change_needs_two_sessions_and_the_target_date():
    import datetime as _dt
    td = _dt.date(2026, 9, 3)
    one = pd.DataFrame([dict(trade_date=td, symbol="X",
                             expiry_date=_dt.date(2026, 9, 29), open_interest=10)])
    assert ilc._total_oi_change(one, td).empty, "one session cannot give a change"
    stale = pd.DataFrame([
        dict(trade_date=_dt.date(2026, 9, 1), symbol="X",
             expiry_date=_dt.date(2026, 9, 29), open_interest=10),
        dict(trade_date=_dt.date(2026, 9, 2), symbol="X",
             expiry_date=_dt.date(2026, 9, 29), open_interest=12)])
    assert ilc._total_oi_change(stale, td).empty, \
        "must not silently read an older session as today"


def test_expiry_session_suppresses_the_futures_read():
    """Forward OI jumps +14.25% on a settlement session against +0.47% elsewhere,
    because the front settles and everyone rolls in. Show nothing, not a build."""
    from datetime import date as _d
    out = ilc.get_index_largecap(_d(2026, 8, 25), "NIFTY")
    assert out.is_expiry_session, "25 Aug 2026 is a monthly settlement session"
    for b in out.rows:
        assert b.fut_valid == 0 and b.fut_oi_pct is None and b.fut_lean is None
    # the cash side must survive the suppression
    assert all(b.ret_pct is not None for b in out.rows)


def test_futures_oi_pct_is_winsorised():
    """One thin contract printed +895%; the raw mean differed from a clipped one
    by up to 84.5pp on a Top-10 day."""
    import inspect
    src = inspect.getsource(ilc._bucket_row)
    assert "clip(pcts, -50, 50)" in src, "the OI mean must stay winsorised"


def test_delivery_z_is_per_symbol_not_a_z_of_the_bucket_mean():
    """The Rest-30 mean had corr +0.674 with how many names reported that day, so
    a bucket-level z was partly a coverage signal."""
    import inspect
    src = inspect.getsource(ilc._bucket_row)
    assert "pivot_table" in src and "cur[ok]" in src, \
        "delivery z must be computed per symbol, then averaged"


def test_concentration_trend_docstring_retracts_the_monotone_claim():
    """The panel originally headlined a monotone rise that does not replicate on a
    membership-independent basket. The retraction must stay in the code."""
    doc = ilc.get_concentration_trend.__doc__ or ""
    assert "NOT AS A TREND" in doc.upper() or "not as a trend" in doc
    assert "replicate" in doc and "survivorship" in doc


# ── the v2 backtest verdict, pinned so it cannot be quietly softened ─────────

def test_state_classification_is_a_partition():
    """Every (a10, a30) pair must land in exactly one of the four states."""
    from src.analytics.index_largecap import (_classify, STATE_BROAD_UP,
                                              STATE_CARRIED, STATE_LAGGED,
                                              STATE_BROAD_DOWN)
    assert _classify(60, 60) == STATE_BROAD_UP
    assert _classify(60, 40) == STATE_CARRIED
    assert _classify(40, 60) == STATE_LAGGED
    assert _classify(40, 40) == STATE_BROAD_DOWN
    assert _classify(50, 50) == STATE_BROAD_DOWN      # exactly half is not "up"
    assert _classify(None, 60) is None and _classify(60, None) is None
    seen = {_classify(a, b) for a in (40, 50, 60) for b in (40, 50, 60)}
    assert len(seen) == 4, "the four states must be reachable and disjoint"


def test_live_state_and_history_share_one_label_set():
    """If the panel's live chip and its base-rate table drifted apart, the chip
    would silently match no row and the metrics would vanish."""
    from datetime import date as _d
    from src.analytics.index_largecap import get_index_largecap, get_state_base_rates
    d = get_index_largecap(_d(2026, 9, 3), "NIFTY")
    br = get_state_base_rates("NIFTY", 5, _d(2026, 9, 3))
    assert d.state is not None
    assert d.state in set(br["state"]), "live state has no row in the base rates"


def test_state_is_none_when_a_bucket_is_too_thin_to_classify():
    d = ilc.IndexLargeCap(trade_date=None, fno_symbol="NIFTY", display="x")
    d.rows = [ilc.BucketRow(label="Top 10", n_members=10, n_present=3, adv_pct=100.0),
              ilc.BucketRow(label="Rest 30", n_members=30, n_present=30, adv_pct=60.0)]
    assert d.state is None, "a 3-of-10 bucket cannot carry a breadth verdict"


def test_base_rate_rows_sum_to_the_all_sessions_row():
    """The four states partition the sample, so their day counts must total the
    'all sessions' row. A mismatch means days are being dropped or double-counted."""
    from datetime import date as _d
    from src.analytics.index_largecap import get_state_base_rates
    br = get_state_base_rates("NIFTY", 5, _d(2026, 9, 4))
    allrow = br[br["state"].str.startswith("—")]
    parts = br[~br["state"].str.startswith("—")]
    assert len(allrow) == 1 and len(parts) == 4
    assert int(parts["days"].sum()) == int(allrow.iloc[0]["days"])


def test_base_rate_docstring_records_why_no_arrow_is_derived():
    """The panel shows forward returns. The reasons they are NOT a signal have to
    travel with the function, or the next reader will wire an arrow to them."""
    from src.analytics.index_largecap import get_state_base_rates
    doc = get_state_base_rates.__doc__ or ""
    assert "BASE RATES, NOT A FORECAST" in doc
    assert "0.65" in doc            # the F&O family reality check
    assert "t=+0.35" in doc         # breadth collapsing under the CLR control
    assert "NO SINGLE LEG" in doc   # the conjunction's failure mode


# ── the flow summary (delivery + futures + options) ──────────────────────────

def test_opt_read_needs_a_quorum_and_names_the_relative_flow():
    """CE vs PE forward-OI flow. A 3-name read is not a bucket verdict."""
    thin = ilc.BucketRow(label="b", n_members=10, n_present=10,
                         opt_valid=3, ce_oi_pct=5.0, pe_oi_pct=-5.0)
    assert thin.opt_read is None
    cw = ilc.BucketRow(label="b", n_members=10, n_present=10,
                       opt_valid=9, ce_oi_pct=4.0, pe_oi_pct=-1.0)
    assert cw.opt_read == "call writing"
    pw = ilc.BucketRow(label="b", n_members=10, n_present=10,
                       opt_valid=9, ce_oi_pct=-1.0, pe_oi_pct=4.0)
    assert pw.opt_read == "put writing"
    bal = ilc.BucketRow(label="b", n_members=10, n_present=10,
                        opt_valid=9, ce_oi_pct=3.0, pe_oi_pct=3.1)
    assert bal.opt_read == "balanced"
    # both sides building, calls faster -> leaning bearish but not pure writing
    lean = ilc.BucketRow(label="b", n_members=10, n_present=10,
                         opt_valid=9, ce_oi_pct=6.0, pe_oi_pct=2.0)
    assert lean.opt_read == "call side heavier"


def test_flow_score_averages_only_the_legs_that_exist():
    """A missing options or futures leg must not be scored as zero — that would
    drag every thin session toward 'mixed' and hide the legs that do exist."""
    r = ilc.BucketRow(label="b", n_members=10, n_present=10, deliv_z=1.0)
    assert r.flow_score == pytest.approx(1.0), "delivery alone must score alone"
    r2 = ilc.BucketRow(label="b", n_members=10, n_present=10, deliv_z=1.0,
                       fut_valid=10, fut_long=8, fut_short=1,
                       opt_valid=9, ce_oi_pct=4.0, pe_oi_pct=-1.0)
    # +1 delivery, +1 futures, -1 options (call writing) -> mean of three = 1/3
    assert r2.flow_score == pytest.approx(1 / 3)
    # and a bearish options leg must be able to flip a two-leg score negative
    r3 = ilc.BucketRow(label="b", n_members=10, n_present=10, deliv_z=0.0,
                       opt_valid=9, ce_oi_pct=4.0, pe_oi_pct=-1.0)
    assert r3.flow_score == pytest.approx(-0.5)


def test_net_score_is_bucket_weighted_and_survives_a_missing_bucket():
    d = ilc.IndexLargeCap(trade_date=None, fno_symbol="NIFTY", display="x")
    d.rows = [ilc.BucketRow(label="Top 10", n_members=10, n_present=10, deliv_z=1.0),
              ilc.BucketRow(label="Next 10", n_members=10, n_present=10, deliv_z=-1.0),
              ilc.BucketRow(label="Rest 30", n_members=30, n_present=30)]
    # Rest 30 has no legs at all, so it must drop OUT of the weighting entirely
    # rather than contribute a 0.0 that silently dilutes the other two.
    w = ilc._BUCKET_WEIGHT
    assert d.net_score == pytest.approx(
        (1.0 * w["Top 10"] - 1.0 * w["Next 10"]) / (w["Top 10"] + w["Next 10"]))


def test_net_score_docstring_carries_the_measured_hit_rate():
    """The score is displayed. The evidence that it does not forecast has to be
    attached to it in code, or someone will wire an arrow to it later."""
    doc = ilc.IndexLargeCap.net_score.__doc__ or ""
    assert "48.2%" in doc and "52.2%" in doc, "next-day hit rate vs base missing"
    assert "44.6%" in doc and "BELOW a coin flip" in doc


def test_expiry_session_suppresses_options_too():
    """Forward OI jumps mechanically on settlement, for options as for futures."""
    from datetime import date as _d
    out = ilc.get_index_largecap(_d(2026, 8, 25), "NIFTY")
    assert out.is_expiry_session
    for b in out.rows:
        assert b.opt_valid == 0 and b.ce_oi_pct is None and b.opt_read is None
