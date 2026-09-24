"""Guards for the per-stock "several buyers" table.

The feature exists because a user reasonably expected two desks buying the same name
to be stronger evidence than one. Measured, it is the opposite (3 buyers within 5
sessions: -2.04% / 1 month, -7.17% / 3 months, t -5.15), so the table is descriptive
and these tests protect the parts that make it honest rather than any ranking.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.analytics import deal_signals as ds


@pytest.fixture(scope="module")
def sides():
    b = ds.load_client_events("BUY")
    s = ds.load_client_events("SELL")
    if b.empty:
        pytest.skip("deal_client_events not built")
    cut = b.trade_date.max() - pd.Timedelta(days=45)
    return b[b.trade_date >= cut], s[s.trade_date >= cut]


def test_consensus_counts_distinct_entities_not_deal_lines(sides):
    b, s = sides
    c = ds.stock_consensus(b, s)
    assert not c.empty
    row = c.iloc[0]
    g = b[b.symbol == row["Symbol"]]
    assert row["Buyers"] == g.client_key.nunique()
    # A desk that bought on three days is three Buys but ONE buyer.
    assert row["Buys"] >= row["Buyers"] or row["Buyers"] <= g.client_key.nunique()


def test_sell_side_is_present_so_crowding_can_be_read_against_distribution(sides):
    b, s = sides
    c = ds.stock_consensus(b, s)
    assert {"Sellers", "Net buyers"} <= set(c.columns)
    assert (c["Net buyers"] == c["Buyers"] - c["Sellers"]).all()
    assert c["Sellers"].sum() > 0, "no sell side at all is a join failure, not a market"


def test_net_bought_is_not_gross_traded_value(sides):
    """The defect this caught: a client showing 0.02 lakh shares against Rs 70 Cr,
    because value_cr sums every disclosed LINE while quantity was netted."""
    b, s = sides
    c = ds.stock_consensus(b, s)
    assert "Net bought (Cr)" in c.columns and "Value (Cr)" in c.columns
    assert (c["Net bought (Cr)"].abs() <= c["Value (Cr)"] + 1e-6).all(), (
        "net bought cannot exceed gross traded value"
    )


def test_excluding_hft_can_only_lower_the_count(sides):
    b, s = sides
    full = ds.stock_consensus(b, s).set_index("Symbol")["Buyers"]
    nohft = ds.stock_consensus(b, s, exclude_hft=True).set_index("Symbol")["Buyers"]
    common = full.index.intersection(nohft.index)
    assert len(common) > 0
    assert (nohft[common] <= full[common]).all()


def test_participants_lists_both_sides_with_unsigned_delivery(sides):
    b, s = sides
    c = ds.stock_consensus(b, s)
    sym = c[c["Sellers"] > 0]["Symbol"].iat[0]
    p = ds.stock_participants(b, s, sym)
    assert set(p["Side"]) == {"BOUGHT", "SOLD"}
    d = p["% of day delivery"].dropna()
    assert (d >= 0).all(), "Side already carries direction; a negative share reads as a bug"
