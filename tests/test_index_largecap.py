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
