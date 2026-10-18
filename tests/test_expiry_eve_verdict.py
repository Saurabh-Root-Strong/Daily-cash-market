"""The session before an expiry: the verdict follows the composite, the range narrows.

The old dte <= 1 branch pointed an arrow toward max pain. Replayed 2022-2026 it was
right 40% of the time on Nifty and pointed against its own target on 23-40% of eves
(12 Oct 2026: "UP" beside a composite of -8.5 and a target under spot). These pin
the replacement so the special case cannot quietly come back.
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analytics.index_prediction import (
    IndexKeyLevels, MarketContext, _EXPIRY_PIN_MULT, _compute_expected_move,
    _compute_verdict,
)

SPOT = 22363.35
CTX = MarketContext(trade_date=date(2026, 10, 12))


def _verdict(composite, dte, max_pain=None):
    return _compute_verdict(composite, [], None, None, dte, SPOT,
                            IndexKeyLevels(max_pain=max_pain), CTX)


def test_eve_arrow_follows_composite_not_max_pain():
    # 12 Oct 2026 shape: max pain above spot, composite bearish beyond the band
    v = _verdict(-8.5, 1, max_pain=SPOT * 1.006)
    assert v[0] == "DOWN"
    assert "max pain" in v[5] and "not a pull" in v[5]


@pytest.mark.parametrize("composite", [-14.0, -8.5, -5.0, -1.0, 0.0, 2.5, 7.9, 8.0, 13.0])
@pytest.mark.parametrize("mp_gap", [None, -0.01, 0.0, 0.01])
def test_eve_direction_equals_ordinary_day_direction(composite, mp_gap):
    mp = None if mp_gap is None else SPOT * (1 + mp_gap)
    eve, ordinary = _verdict(composite, 1, mp), _verdict(composite, 10, mp)
    assert eve[:5] == ordinary[:5]
    assert eve[5].startswith("Expiry tomorrow")


def test_eve_without_max_pain_still_notes_the_pin():
    v = _verdict(0.0, 0, None)
    assert v[0] == "SIDEWAYS" and v[5].startswith("Expiry tomorrow")


def test_expected_move_pin_multiplier():
    rng = np.random.default_rng(1)
    closes = 22000 * np.cumprod(1 + rng.normal(0, 0.008, 60))
    hist = pd.DataFrame({"trade_date": pd.bdate_range("2026-07-01", periods=60).date,
                         "close_val": closes})
    lv = IndexKeyLevels()
    eve = _compute_expected_move(hist, SPOT, None, 0.0, 1, lv, True)
    far = _compute_expected_move(hist, SPOT, None, 0.0, 10, lv, True)
    assert _EXPIRY_PIN_MULT == 0.85
    half_eve = (eve["range_high"] - eve["range_low"]) / 2
    half_far = (far["range_high"] - far["range_low"]) / 2
    assert half_eve / half_far == pytest.approx(_EXPIRY_PIN_MULT, abs=0.01)
    assert "expiry pin ×0.85" in eve["move_basis"]
