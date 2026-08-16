"""Pins the vectorised similarity scorer against the scalar one it replaced.

get_sector_memory_context scored its candidate history one row at a time via
iterrows — 16,368 row reads per Smart Money render, the panel's largest single
cost. It now scores the frame in one numpy pass. _similarity is still the
definition of the metric; _similarity_many must agree with it exactly, or the
memory overlay silently starts retrieving different neighbours and the
conviction column changes with no other visible cause.

Also pins the two behaviours the vectorised path depends on that are easy to
break while "tidying":

  * NaN candidates must be DROPPED. The scalar path returns 0.0 for them
    (Python's max(0.0, nan) keeps 0.0) while the array path propagates NaN.
    Both fail `>= _MIN_SIMILARITY`, which is why the caller must filter with
    >= and never with `not <` — the latter would keep every NaN row.
  * Ties must keep original order. list.sort(reverse=True) leaves equal
    elements in place rather than reversing them, and the episode-gap rule
    downstream picks the FIRST of a tie, so an unstable argsort would change
    which historical episodes are retained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.sector_memory import (
    _FEAT_WEIGHTS, _MIN_SIMILARITY, _similarity, _similarity_many,
)

COLS = list(_FEAT_WEIGHTS)


def _frame(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_weights_still_sum_to_one():
    """The metric's normalisation — max distance is sqrt(sum(w)) = 1."""
    assert abs(sum(_FEAT_WEIGHTS.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("seed", range(12))
def test_matches_the_scalar_definition_on_random_rows(seed):
    rng = np.random.default_rng(seed)
    today = {k: float(v) for k, v in zip(COLS, rng.random(len(COLS)))}
    mat = rng.random((60, len(COLS)))
    got = _similarity_many(today, _frame(mat))
    want = [_similarity(today, dict(zip(COLS, row))) for row in mat]
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_matches_at_the_extremes():
    today = {k: 0.5 for k in COLS}
    same = _frame([[0.5] * len(COLS)])
    assert _similarity_many(today, same)[0] == 1.0
    assert _similarity(today, {k: 0.5 for k in COLS}) == 1.0
    # maximally distant: every feature at the opposite end
    far = _frame([[1.0] * len(COLS)])
    assert _similarity_many(today, far)[0] == pytest.approx(
        _similarity(today, {k: 1.0 for k in COLS}))


def test_missing_today_features_default_to_half():
    """today_feat may lack keys; both paths substitute 0.5."""
    today = {COLS[0]: 0.9}          # everything else absent
    mat = np.full((3, len(COLS)), 0.5)
    got = _similarity_many(today, _frame(mat))
    want = [_similarity(today, dict(zip(COLS, row))) for row in mat]
    np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_nan_candidate_rows_are_filtered_out_by_the_ge_comparison():
    today = {k: 0.5 for k in COLS}
    bad = [0.5] * len(COLS)
    bad[3] = np.nan
    sims = _similarity_many(today, _frame([[0.5] * len(COLS), bad]))
    assert sims[0] == 1.0
    assert np.isnan(sims[1])
    keep = np.flatnonzero(sims >= _MIN_SIMILARITY)
    assert keep.tolist() == [0], "NaN row survived the >= filter"
    # the scalar path drops it too, by a different route
    assert _similarity(today, dict(zip(COLS, bad))) == 0.0


def test_the_scalar_path_really_does_return_zero_for_nan():
    """Guards the claim the comment makes about max(0.0, nan)."""
    assert max(0.0, float("nan")) == 0.0


def test_results_are_rounded_to_four_places():
    rng = np.random.default_rng(7)
    today = {k: float(v) for k, v in zip(COLS, rng.random(len(COLS)))}
    sims = _similarity_many(today, _frame(rng.random((40, len(COLS)))))
    np.testing.assert_allclose(sims, np.round(sims, 4), rtol=0, atol=0)


# ── through the SHIPPED function, not a reimplementation ─────────────────────
#
# The two properties below were first written against inline copies of the
# argsort and the >= filter. Both mutants (kind="stable" -> "quicksort", and
# `>=` -> `not <`) survived, because the tests were checking the test's own
# code. These drive get_sector_memory_context itself by handing it a synthetic
# snapshot through _log_snapshot, which is also how the overlay avoids
# re-querying per sector — so no database is involved.

_SECTOR = "TestSector"


def _snapshot(rows):
    """Build a sector_rotation_log-shaped frame. `rows` = (day_offset, feats)."""
    from datetime import date, timedelta
    base = date(2026, 1, 1)
    recs = []
    for off, feats in rows:
        r = {"trade_date": base - timedelta(days=off), "sector": _SECTOR,
             "signal": "ACCUMULATION", "regime_label": "SIDEWAYS",
             "accum_score": 1.0, "dv_ratio": 1.0, "z_pct": 0.5}
        r.update(dict(zip(COLS, feats)))
        for c in ("fwd_ret_1w", "fwd_ret_2w", "fwd_ret_1m",
                  "fwd_rs_1w", "fwd_rs_2w", "fwd_rs_1m"):
            r[c] = 1.0
        recs.append(r)
    df = pd.DataFrame(recs)
    # the real query is ORDER BY trade_date DESC
    return df.sort_values("trade_date", ascending=False).reset_index(drop=True)


def _context(snap):
    from datetime import date
    from src.analytics.sector_memory import get_sector_memory_context
    return get_sector_memory_context(
        as_of_date=date(2026, 6, 1), sector=_SECTOR, signal="ACCUMULATION",
        regime_label="SIDEWAYS", dv_ratio=1.0, z_pct=0.5, rs_1w=None,
        ema20_above=True, ema_cross_bull=None, vix=None, fii_5d_cr=None,
        hmm_state=None, pcr=None, _log_snapshot=snap)


def test_shipped_path_drops_nan_candidates():
    """Mutating `>=` to `not <` must not let NaN rows through."""
    good = [0.5] * len(COLS)
    bad = [0.5] * len(COLS)
    bad[3] = np.nan
    # spaced > _EPISODE_GAP_DAYS apart so the gap rule keeps every survivor
    snap = _snapshot([(0, bad), (40, good), (80, good), (120, good),
                      (160, good), (200, good)])
    ctx = _context(snap)
    assert ctx.n_filled == 5, f"expected the NaN row dropped, got {ctx.n_filled}"
    # the five survivors are identical rows, so they must all score the same
    sims = {s.similarity for s in ctx.similar_setups}
    assert len(sims) == 1, f"identical rows scored differently: {sims}"
    assert not any(np.isnan(s.similarity) for s in ctx.similar_setups)


def test_shipped_path_keeps_ties_in_trade_date_desc_order():
    """Mutating the stable argsort must change which episode comes first.

    All rows are identical in features, so every similarity ties. The episode
    gap rule then takes them in whatever order the sort produced, and the most
    recent must lead.
    """
    # Must be GROUPS of ties, not one big tie. numpy's quicksort returns the
    # identity permutation when every key is equal, at any size, so an
    # all-identical fixture cannot tell the two sorts apart — verified, and it
    # let the mutant through. With several distinct levels each repeated, the
    # unstable sort visibly reorders within a level.
    levels = [[0.50] * len(COLS), [0.52] * len(COLS), [0.54] * len(COLS)]
    rows = [(i * 40, levels[i % len(levels)]) for i in range(60)]
    ctx = _context(_snapshot(rows))
    got = [(s.similarity, s.trade_date) for s in ctx.similar_setups]
    assert len(got) > 1
    # stable descending: similarity non-increasing, and inside each tied level
    # the newest date first (the frame arrives trade_date DESC)
    want = sorted(got, key=lambda p: (-p[0], -p[1].toordinal()))
    assert got == want, (
        "tied similarities came back out of trade_date DESC order.\n"
        f"  got : {got[:6]}\n  want: {want[:6]}")


def test_float32_input_is_scored_in_float64():
    """Stored columns are float32; scoring in float32 would round differently."""
    rng = np.random.default_rng(3)
    mat = rng.random((30, len(COLS)))
    today = {k: float(v) for k, v in zip(COLS, rng.random(len(COLS)))}
    f32 = _frame(mat.astype(np.float32))
    got = _similarity_many(today, f32)
    want = [_similarity(today, dict(zip(COLS, row)))
            for row in mat.astype(np.float32).astype(np.float64)]
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
