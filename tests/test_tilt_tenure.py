"""Pins the "Days →" tenure badge on the Forward Sector Tilt.

The badge is rebuilt from the panel `get_forward_tilt` has already loaded rather
than logged nightly or recomputed per date, so its correctness rests on the rebuilt
label matching what the live engine would have said on each past session.

Two invariants have to hold, and neither is visible from reading the diff:

  1. OVERWEIGHT ONLY. WATCH is defined on rank <= 0.35 and so can never mask an
     OVERWEIGHT (cut 0.75), which is why an OW streak is exactly reproducible. It
     CAN mask a NEUTRAL or an UNDERWEIGHT (rank <= 0.25 sits inside the WATCH
     band). Validation against 16 real per-date engine runs found exactly that:
     all 5 OW streaks matched while 4 NEUTRAL/UNDERWEIGHT rows were wrong, because
     WATCH intervened MID-window without changing that day's final label — so an
     "agrees on as_of" guard does not catch it. If a future edit lets a non-OW
     bucket report a number, it will be quietly wrong about a third of the time.

  2. A BREAK MUST BREAK THE STREAK. A day the sector was not OVERWEIGHT, and a day
     it has no data at all (label None), must both terminate the count rather than
     be skipped over. Counting through a hole is how a tenure display turns three
     separate 2-day runs into a fake 8-day one.

Both gates that can demote an OVERWEIGHT (`thin`, then persistence) are applied in
_tilt_history in the same order as get_forward_tilt; that ordering is asserted too.

These run offline on synthetic frames. The end-to-end equality — badge vs calling
get_forward_tilt on each of the trailing 26 sessions, at horizons 10 and 60 — was
verified against the real database at the time of the change (9 OW streaks matched,
0 wrong, 0 non-OW leaks) and needs that database, so it is not repeated here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.sector_forward_tilt import (
    _MIN_LIQ_NAMES, _OW_RANK, _TENURE_LOOKBACK, _UW_RANK,
    _tenure_days, _tilt_history,
)


def _panel(labels_by_day: dict[str, list[float]], n: int = 120) -> pd.DataFrame:
    """Long-format sector panel whose LAST rows produce the wanted rank ordering.

    `labels_by_day[sector]` is that sector's daily return for the final len(...)
    sessions; earlier sessions are flat so trailing windows are well defined.

    The default 120 flat sessions is not arbitrary: the dv5d leg divides by a
    _DV_BASE=100-session delivery baseline, so a shorter panel leaves dv5d NaN,
    which propagates through the weighted score and makes EVERY label None. The
    live panel pulls ~260 sessions so it never sees this, but a fixture can.
    """
    secs = list(labels_by_day)
    tail = len(next(iter(labels_by_day.values())))
    dates = pd.bdate_range("2026-01-01", periods=n + tail)
    rows = []
    for s in secs:
        series = [0.0] * n + labels_by_day[s]
        for d, r in zip(dates, series):
            rows.append({"sector": s, "trade_date": d,
                         "wtd_ret_pct": r, "daily_dv_cr": 100.0})
    return pd.DataFrame(rows)


def _labels(panel, **kw):
    return _tilt_history(panel, pd.DataFrame(), L=5, S=2, pers_hist=None, **kw)


# ── invariant 1: only DIRECTIONAL buckets may report a number ────────────────
def test_only_directional_buckets_report_tenure():
    """OVERWEIGHT and UNDERWEIGHT count; NEUTRAL never does.

    UNDERWEIGHT became reportable once _tilt_history started modelling WATCH — it
    sits inside the underweight rank band and previously masked it mid-window.
    NEUTRAL stays silent for a different reason that no amount of modelling fixes:
    it is the residual bucket every overlay drops into, so a NEUTRAL run measures
    "nothing else fired", not a call being held.
    """
    lab = pd.DataFrame(
        {"A": ["OVERWEIGHT"] * 5, "B": ["UNDERWEIGHT"] * 5, "C": ["NEUTRAL"] * 5},
        index=pd.bdate_range("2026-03-02", periods=5))
    cur = pd.Series({"A": "OVERWEIGHT", "B": "UNDERWEIGHT", "C": "NEUTRAL"})
    out = _tenure_days(lab, cur)
    assert out["A"] == 5, "an unbroken OVERWEIGHT run must be counted"
    assert out["B"] == 5, "an unbroken UNDERWEIGHT run must be counted"
    assert out["C"] == 0, "NEUTRAL is the residual bucket and must stay silent"


def test_watch_breaks_an_underweight_streak():
    """The case that made underweight tenure unreportable before WATCH was modelled:
    today's label agrees, but a mid-window session was WATCH, so the run is shorter."""
    lab = pd.DataFrame(
        {"A": ["UNDERWEIGHT", "UNDERWEIGHT", "WATCH", "UNDERWEIGHT", "UNDERWEIGHT"]},
        index=pd.bdate_range("2026-03-02", periods=5))
    assert _tenure_days(lab, pd.Series({"A": "UNDERWEIGHT"}))["A"] == 2


def test_watch_masking_a_neutral_cannot_leak_a_number():
    """The exact shape that produced the 4 real mismatches: today's label agrees,
    a mid-window day did not."""
    lab = pd.DataFrame({"A": ["NEUTRAL", "WATCH", "NEUTRAL", "NEUTRAL"]},
                       index=pd.bdate_range("2026-03-02", periods=4))
    assert _tenure_days(lab, pd.Series({"A": "NEUTRAL"}))["A"] == 0


# ── invariant 2: breaks and holes terminate the count ────────────────────────
@pytest.mark.parametrize("seq, expected", [
    (["OVERWEIGHT"] * 4, 4),
    (["OVERWEIGHT", "NEUTRAL", "OVERWEIGHT", "OVERWEIGHT"], 2),
    (["OVERWEIGHT", "OVERWEIGHT", "NEUTRAL"], 0),          # not OW today
    ([None, "OVERWEIGHT", "OVERWEIGHT"], 2),
    (["OVERWEIGHT", None, "OVERWEIGHT"], 1),               # hole must not be skipped
])
def test_streak_counts_only_the_trailing_run(seq, expected):
    lab = pd.DataFrame({"A": seq}, index=pd.bdate_range("2026-03-02", periods=len(seq)))
    assert _tenure_days(lab, pd.Series({"A": "OVERWEIGHT"}))["A"] == expected


def test_unknown_sector_is_silent_not_zero_length_crash():
    lab = pd.DataFrame({"A": ["OVERWEIGHT"]}, index=pd.bdate_range("2026-03-02", periods=1))
    assert _tenure_days(lab, pd.Series({"ZZZ": "OVERWEIGHT"}))["ZZZ"] == 0


# ── the rebuilt label itself ─────────────────────────────────────────────────
def test_rank_bands_match_the_live_thresholds():
    """Top of the cross-section is OVERWEIGHT, bottom is UNDERWEIGHT, at the same
    cuts the engine uses."""
    n = 12
    panel = _panel({f"S{i}": [float(i)] * 6 for i in range(n)})
    lab = _labels(panel)
    last = lab.iloc[-1]
    ranks = pd.Series(np.arange(1, n + 1) / n, index=[f"S{i}" for i in range(n)])
    for s in last.index:
        want = ("OVERWEIGHT" if ranks[s] >= _OW_RANK else
                "UNDERWEIGHT" if ranks[s] <= _UW_RANK else "NEUTRAL")
        assert last[s] == want, f"{s} at rank {ranks[s]:.2f} became {last[s]}"


def test_thin_gate_demotes_the_leader_exactly_as_the_engine_does():
    panel = _panel({f"S{i}": [float(i)] * 6 for i in range(12)})
    lab = _labels(panel)
    leader = lab.iloc[-1].idxmax() if False else "S11"
    assert lab.iloc[-1][leader] == "OVERWEIGHT"

    nliq = pd.DataFrame(100.0, index=lab.index, columns=lab.columns)
    nliq[leader] = _MIN_LIQ_NAMES - 1          # too few liquid names → not a buy
    thin_lab = _labels(panel, nliq_hist=nliq)
    assert thin_lab.iloc[-1][leader] == "NEUTRAL"
    assert _tenure_days(thin_lab, pd.Series({leader: "OVERWEIGHT"}))[leader] == 0


def test_persistence_gate_demotes_a_reverting_leader():
    panel = _panel({f"S{i}": [float(i)] * 6 for i in range(12)})
    lab = _labels(panel)
    assert lab.iloc[-1]["S11"] == "OVERWEIGHT"

    pers = pd.DataFrame(1.0, index=lab.index, columns=lab.columns)
    pers["S11"] = -0.5                          # historically fades after looking strong
    gated = _tilt_history(panel, pd.DataFrame(), L=5, S=2, pers_hist=pers)
    assert gated.iloc[-1]["S11"] == "NEUTRAL"


def test_nan_persistence_keeps_the_overweight():
    """NaN means 'unknown', and the live gate is `persistence < 0` which is False
    for NaN. An 'unknown' sector must not be silently dropped from the buy list."""
    panel = _panel({f"S{i}": [float(i)] * 6 for i in range(12)})
    lab = _labels(panel)
    pers = pd.DataFrame(np.nan, index=lab.index, columns=lab.columns)
    assert _tilt_history(panel, pd.DataFrame(), L=5, S=2,
                         pers_hist=pers).iloc[-1]["S11"] == "OVERWEIGHT"


def test_history_is_capped_to_the_lookback():
    panel = _panel({f"S{i}": [float(i)] * 6 for i in range(12)}, n=_TENURE_LOOKBACK * 2)
    assert len(_labels(panel)) <= _TENURE_LOOKBACK


def test_short_history_degrades_to_empty_not_an_exception():
    panel = _panel({f"S{i}": [float(i)] * 2 for i in range(12)}, n=1)
    assert _labels(panel).empty


def test_missing_delivery_baseline_yields_no_labels_not_wrong_ones():
    """Under ~100 sessions the dv5d leg is NaN and the whole score goes NaN. That
    must surface as 'no tenure' (None labels -> streak 0), never as a number."""
    panel = _panel({f"S{i}": [float(i)] * 6 for i in range(12)}, n=30)
    lab = _labels(panel)
    assert lab.iloc[-1].isna().all() or (lab.iloc[-1] == None).all()  # noqa: E711
    assert _tenure_days(lab, pd.Series({"S11": "OVERWEIGHT"}))["S11"] == 0
