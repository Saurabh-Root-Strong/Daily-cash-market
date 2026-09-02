"""Pins the horizon-scaled delivery-flow windows.

The RS legs always scaled with the horizon; the delivery factor did not — it stayed
at 5/100 at every horizon, so a 11-12wk call was ranked partly on the last five
sessions of delivery. These tests pin the fix and, above all, pin that the VALIDATED
1-2wk build is byte-identical to what shipped before it.
"""
import pytest

from src.analytics.sector_forward_tilt import (_dv_windows, _panel_lookback_cal,
                                               _DV_FLOW, _DV_BASE, TILT_HORIZONS)


def test_1_2wk_is_bit_identical_to_the_shipped_build():
    """h=10 MUST reproduce the shipped 5/100. The 1-2wk tilt is the only horizon
    that was ever validated; a change there invalidates its measured record."""
    assert _dv_windows(10) == (_DV_FLOW, _DV_BASE) == (5, 100)


def test_flow_and_base_both_scale_with_the_horizon():
    for h in (20, 30, 40, 50, 60):
        flow, base = _dv_windows(h)
        assert flow == h // 2
        assert base == 10 * h


def test_flow_in_base_overlap_is_invariant():
    """The baseline CONTAINS the flow window, so a surge inflates its own denominator.
    Scaling the flow alone drives that overlap from 5% to 30% at 11-12wk and measurably
    hurts. Both scale, so the ratio is constant — this is the whole point of the design."""
    ratios = {h: _dv_windows(h)[0] / _dv_windows(h)[1] for h, in
              [(h,) for _, h in TILT_HORIZONS]}
    assert len(set(round(r, 6) for r in ratios.values())) == 1, ratios
    assert pytest.approx(0.05, abs=1e-9) == next(iter(ratios.values()))


def test_every_offered_horizon_is_covered():
    for _label, h in TILT_HORIZONS:
        flow, base = _dv_windows(h)
        assert flow >= 2 and base > flow


def test_panel_lookback_covers_the_baseline_it_must_feed():
    """_load_sector_panel must return at least `base` TRADING sessions or the ratio
    silently computes off a short window. ~1.45 calendar days per session."""
    for _label, h in TILT_HORIZONS:
        base = _dv_windows(h)[1]
        cal = _panel_lookback_cal(h)
        assert cal / 1.45 >= base, f"h={h}: {cal} cal days cannot cover {base} sessions"


def test_default_horizon_does_not_widen_the_scan():
    """1-2wk must keep the historical 275-day window — no perf regression on the
    shipped default."""
    assert _panel_lookback_cal(10) == 275
    assert _panel_lookback_cal(60) > 275
