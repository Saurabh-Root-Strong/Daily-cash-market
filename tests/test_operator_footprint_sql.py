"""Pins the SQL moneyness bucketing against the Python one it replaced.

The 400-day option baseline used to be pulled into pandas — 4,660,068 rows on
2026-08-14 — and walked by list comprehensions. It is now aggregated in DuckDB
(~30s to ~8s per session, output verified identical over five sessions). That
moved _moneyness_bucket's logic into SQL, so the two must not drift.

The subtle case, and the reason this file exists: _moneyness_bucket guards with
`not spot or spot <= 0`, and BOTH are False for a NaN spot (`not nan` is False,
`nan <= 0` is False). So a NaN spot does NOT return 'n/a'. It falls through to
threshold comparisons that are all False against NaN and comes out 'deep OTM'.
SQL reproduces that only because a NULL spot leaves every WHEN unknown. Adding
`WHEN spot IS NULL THEN 'n/a'` reads like an obvious tidy-up and would silently
reclassify every such row into a different baseline bucket.

Runs on an in-memory DuckDB over a synthetic grid — no project database.
"""
from __future__ import annotations

import math

import duckdb
import pandas as pd
import pytest

from src.analytics.operator_footprint import _MONEYNESS_SQL, _moneyness_bucket

# Straddles every threshold (-0.10, -0.02, 0.02, 0.10) and its exact boundary,
# plus the degenerate spots.
_SPOTS = [100.0, 0.0, -5.0, None]
_STRIKES = [80.0, 89.0, 90.0, 91.0, 97.0, 98.0, 99.0, 100.0,
            101.0, 102.0, 103.0, 109.0, 110.0, 111.0, 130.0, None]
_TYPES = ["CE", "PE"]


def _grid():
    return [(s, sp, t) for sp in _SPOTS for s in _STRIKES for t in _TYPES]


@pytest.fixture(scope="module")
def sql_buckets():
    rows = _grid()
    df = pd.DataFrame(rows, columns=["strike_price", "spot", "option_type"])
    con = duckdb.connect(":memory:")
    con.register("g", df)
    out = con.execute(
        f"SELECT strike_price, spot, option_type, {_MONEYNESS_SQL} AS m FROM g"
    ).df()
    con.close()
    return out


def test_sql_matches_python_on_every_grid_point(sql_buckets):
    bad = []
    for _, r in sql_buckets.iterrows():
        spot = None if pd.isna(r["spot"]) else float(r["spot"])
        strike = float("nan") if pd.isna(r["strike_price"]) else float(r["strike_price"])
        # DuckDB NULL arrives as NaN, which is exactly what the pandas
        # implementation saw off .df(); pass NaN, not None, for a NULL spot.
        py = _moneyness_bucket(strike, float("nan") if spot is None else spot,
                               r["option_type"])
        if py != r["m"]:
            bad.append((strike, spot, r["option_type"], py, r["m"]))
    assert not bad, "python -> sql bucket mismatches:\n" + "\n".join(
        f"  strike={s} spot={sp} {t}: python={p!r} sql={q!r}" for s, sp, t, p, q in bad)


def test_null_spot_is_deep_otm_not_na():
    """The trap. If this ever flips to 'n/a' someone added a NULL guard."""
    assert _moneyness_bucket(100.0, float("nan"), "CE") == "deep OTM"
    assert _moneyness_bucket(100.0, float("nan"), "PE") == "deep OTM"


def test_zero_and_negative_spot_are_na():
    for opt in _TYPES:
        assert _moneyness_bucket(100.0, 0.0, opt) == "n/a"
        assert _moneyness_bucket(100.0, -5.0, opt) == "n/a"
        assert _moneyness_bucket(float("nan"), 100.0, opt) == "n/a"


def test_every_bucket_label_is_actually_reachable(sql_buckets):
    """A typo'd label would otherwise pass silently as 'some string matched'."""
    got = set(sql_buckets["m"])
    assert got == {"deep ITM", "ITM", "ATM", "OTM", "deep OTM", "n/a"}, got


@pytest.mark.parametrize("dte", list(range(-8, 30)))
def test_dte_bucketing_agrees_after_the_clip(dte):
    """pandas floors, DuckDB truncates toward zero; clip(0, 12) must hide it.

    They disagree only for dte in -6..-1. The port relies on the clip mapping
    both answers to 0 — assert that rather than trust it.
    """
    py = min(max(math.floor(dte / 7), 0), 12)
    con = duckdb.connect(":memory:")
    sql = con.execute(
        "SELECT least(greatest(? // 7, 0), 12)", [dte]).fetchone()[0]
    con.close()
    assert py == sql, f"dte={dte}: pandas={py} duckdb={sql}"
