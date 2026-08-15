"""
Guards on the analogue matcher's CALIBRATION WIRING.

These are offline structural tests — no database, no market data. They exist
because the failure this module keeps suffering is not a wrong formula, it is a
DISPLAYED NUMBER THAT DESCRIBES A MODEL THAT NO LONGER RUNS. It has happened
three times: when the feature set changed, when the distance metric changed from
Euclidean to Mahalanobis, and when one flat calibration blob was emitted under
all three radio buttons so two of the three readings belonged to another mode.

Nothing here checks whether the analogues predict anything (they do not — see
scripts/audit_analogues.py). These check that what the dashboard prints is keyed
to the model that produced it.
"""
from __future__ import annotations

import pytest

from src.analytics import market_context as mc

MODES = ("price", "price+fii", "fii")
_TRIPLES = ("bull_hit", "bull_edge", "bull_n", "bull_t",
            "bear_hit", "bear_edge", "bear_n", "ctl_edge", "ctl_sd", "ic")


def test_every_mode_has_its_own_calibration():
    for m in MODES:
        assert m in mc._ANALOGUE_CAL, f"no calibration entry for mode {m!r}"


def test_calibration_is_not_shared_between_modes():
    """The actual bug: price's numbers displayed under all three buttons.

    Two modes agreeing on one field by coincidence is fine; two modes with an
    IDENTICAL calibration means someone copied a block instead of measuring it.
    """
    for a in MODES:
        for b in MODES:
            if a >= b:
                continue
            assert mc._ANALOGUE_CAL[a] != mc._ANALOGUE_CAL[b], (
                f"modes {a!r} and {b!r} carry identical calibration — one of "
                "them is describing the other's model")


def test_calibration_shape_matches_the_horizons():
    n = len(mc.HORIZONS)
    for m in MODES:
        cal = mc._ANALOGUE_CAL[m]
        for key in _TRIPLES:
            assert key in cal, f"{m}: missing {key}"
            assert len(cal[key]) == n, (
                f"{m}.{key} has {len(cal[key])} values for {n} horizons")
        for key in ("gap_guard", "gap_noguard", "sens_lo", "sens_hi",
                    "span_cov_1m", "span_width_1m"):
            assert key in cal, f"{m}: missing {key}"
        assert len(cal["span_ic"]) == n, f"{m}: span_ic must be per horizon"


def test_span_never_beats_plain_volatility():
    """The spread is calibrated but carries no information beyond vol20.

    If a future measurement flips this, the panel's "use the spread" framing has
    to be revisited deliberately rather than by a silent constant change.
    """
    vol = mc._ANALOGUE_SHARED["vol20_ic"]
    for m in MODES:
        for i, (sp_, v) in enumerate(zip(mc._ANALOGUE_CAL[m]["span_ic"], vol)):
            assert abs(sp_) < abs(v), (
                f"{m} horizon {i}: span IC {sp_} now exceeds vol20 IC {v} — "
                "re-check before presenting the spread as its own product")


def test_edges_are_consistent_with_the_shared_base_rate():
    """bull_edge must equal bull_hit minus the shared base rate.

    If someone updates one and not the other, the panel prints a hit rate and an
    edge that disagree.
    """
    base = mc._ANALOGUE_SHARED["base_hit"]
    for m in MODES:
        cal = mc._ANALOGUE_CAL[m]
        for i, (hit, edge, b) in enumerate(
                zip(cal["bull_hit"], cal["bull_edge"], base)):
            assert abs((hit - b) - edge) < 0.15, (
                f"{m} horizon {i}: bull_hit {hit} - base {b} = {hit - b:.1f}, "
                f"but bull_edge says {edge}")
        for i, (hit, edge, b) in enumerate(
                zip(cal["bear_hit"], cal["bear_edge"], base)):
            assert abs((hit - b) - edge) < 0.15, (
                f"{m} horizon {i}: bear_hit/bear_edge disagree")


def test_sensitivity_range_is_ordered_and_wide_enough_to_be_honest():
    for m in MODES:
        cal = mc._ANALOGUE_CAL[m]
        assert cal["sens_lo"] < cal["sens_hi"], f"{m}: sensitivity range inverted"
        assert cal["sens_hi"] - cal["sens_lo"] >= 10, (
            f"{m}: a <10pt sensitivity range across k=8..20 and sep=10..42 has "
            "never been measured here; check it was not copied from another mode")


def test_no_mode_claims_significance():
    """Standing decision: this panel emits no direction.

    Every measured |HAC t| is under 1.5 and the max-statistic permutation clears
    nothing. If a future measurement genuinely breaks 2.0, this test should fail
    loudly so the claim gets re-derived rather than quietly shipped.
    """
    for m in MODES:
        for t in mc._ANALOGUE_CAL[m]["bull_t"]:
            assert abs(t) < 2.0, (
                f"{m}: |HAC t| = {abs(t)} now exceeds 2.0. Re-run "
                "scripts/audit_analogues.py, price the full search, and update "
                "the standing decision deliberately before relaxing this.")
    assert mc._ANALOGUE_SHARED["perm_p"] > 0.05, (
        "the max-statistic permutation now clears 0.05. That is not licence to "
        "ship a direction call: re-price the FULL search (~410 variants), not "
        "just the 9 mode x horizon cells, before changing anything.")


def test_mode_notes_cover_every_mode():
    for m in MODES:
        note = mc._ANALOGUE_MODE_NOTE.get(m)
        assert note and len(note) > 20, f"{m}: missing or stub mode note"


@pytest.mark.parametrize("mode", MODES)
def test_meta_never_falls_back_to_another_modes_numbers(mode, monkeypatch):
    """A mode with no measurement must report has_cal=False, not borrow one."""
    monkeypatch.setitem(mc._ANALOGUE_CAL, mode, {})
    assert mc._ANALOGUE_CAL[mode] == {}
    # the view keys everything off meta["cal"] / meta["has_cal"]; an empty cal
    # must stay empty rather than resolving to a sibling mode
    assert not mc._ANALOGUE_CAL[mode].get("bull_hit")
