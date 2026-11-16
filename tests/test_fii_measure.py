"""fii_measure + the live wiring in fii_only (added 2026-11-16)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tests.test_fii_only import fii_db  # noqa: F401  (fixture: 320 seeded sessions)

ROOT = Path(__file__).resolve().parents[1]


def test_short_history_falls_back_to_remeasured_constants(fii_db):  # noqa: F811
    from src.analytics import fii_only as F
    v = F.get_fii_only_view(fii_db[-1])
    assert v["ok"], v["error"]
    # 320 sessions < the 500 needed for walk-forward zones -> constant tables, labelled
    assert v["zones_basis"] == "re-measured constant"
    assert v["zones"] is F.FII_ZONES


def test_measure_shapes_on_seeded_data(fii_db):  # noqa: F811
    from src.analytics import fii_measure as M
    assert M.measure_zones(fii_db[-1]) is None               # not enough history: None, not a crash
    tr = M.measure_track_record(fii_db[-1])
    assert {"n", "next_day_oc_hit", "wk2_naive"} <= set(tr)
    ic = M.measure_leg_ic(fii_db[-1])
    assert set(ic["_t"]) == {"fut_long", "fut_short", "call_long", "call_short", "put_long", "put_short"}
    L = M.leg_series(fii_db[-1])
    legs = ["fut_long", "fut_short", "call_long", "call_short", "put_long", "put_short"]
    j = pd.concat([L[legs].sum(axis=1, min_count=6), L["net_1d"]], axis=1).dropna()
    assert len(j) > 200 and np.allclose(j.iloc[:, 0], j.iloc[:, 1])


def test_net_1d_series_matches_live_actions(fii_db):  # noqa: F811
    from src.analytics import fii_measure as M, fii_only as F
    a = F.get_fii_actions(fii_db[-1])
    L = M.leg_series(fii_db[-1])
    assert abs(L["net_1d"].iloc[-1] - a["net_1d"]) < 1e-9


def test_replace_block_rewrites_only_the_named_constant():
    sys.path.insert(0, str(ROOT / "scripts"))
    from remeasure_fii_only import _replace_block
    src = "A = {'x': (1, 2), 'y': {'z': 3}}\nB = [1, 2]\nC = 5\n"
    out = _replace_block(src, "A", "{'x': 9}")
    assert out == "A = {'x': 9}\nB = [1, 2]\nC = 5\n"
    assert _replace_block(out, "B", "[7]") == "A = {'x': 9}\nB = [7]\nC = 5\n"
