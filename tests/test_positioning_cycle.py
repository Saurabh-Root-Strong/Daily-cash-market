"""Positioning Cycle tests — isolated temp DuckDB, no network, no live data.

Each test pins a guard that the 2026-11-14 turning-point study showed was needed:
holiday-dated participant rows must be ignored, percentiles must not see the
future, swing turns must be confirmed before they are listed, short history must
degrade to "no zone" rather than invent one, and the zero-sum check must catch a
partial file.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


def _sessions(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _path(n: int) -> np.ndarray:
    """Nifty path with clear swings: up 12%, down 12%, up again — repeated."""
    t = np.arange(n)
    return 20_000 * (1 + 0.12 * np.sin(t / 25.0))


def _seed(days: list[date], *, drop_types: tuple[str, ...] = (), holiday: date | None = None,
          imbalance_on: date | None = None) -> np.ndarray:
    px = _path(len(days))
    rows_ix = []
    for i, d in enumerate(days):
        prev = px[i - 1] if i else px[i]
        rows_ix.append({"trade_date": d, "index_name": "Nifty 50", "open_val": prev,
                        "high_val": max(prev, px[i]), "low_val": min(prev, px[i]),
                        "close_val": px[i], "prev_close": prev, "points_chg": px[i] - prev,
                        "pct_chg": (px[i] / prev - 1) * 100, "volume": 1, "turnover_cr": 1.0,
                        "pe_ratio": 20.0, "pb_ratio": 3.0, "div_yield": 1.0})
    from src.data.repository import upsert_fao_data, upsert_index_data
    upsert_index_data(pd.DataFrame(rows_ix))

    # FII long share follows price (as measured), Client takes the other side.
    rows = []
    dates = list(days) + ([holiday] if holiday else [])
    for i, d in enumerate(dates):
        is_hol = holiday is not None and d == holiday
        share = 0.95 if is_hol else 0.2 + 0.6 * (px[min(i, len(px) - 1)] - px.min()) / (px.max() - px.min())
        tot = 100_000
        fl = int(tot * share); fs = tot - fl
        legs = {"FII": (fl, fs), "Client": (fs, fl), "Pro": (30_000, 30_000), "DII": (20_000, 20_000)}
        for ct, (l, s) in legs.items():
            if ct in drop_types and d == days[-1]:
                continue
            if imbalance_on is not None and d == imbalance_on and ct == "Pro":
                l += 50_000
            rows.append({"trade_date": d, "client_type": ct, "data_type": "OI",
                         "fut_idx_long": l, "fut_idx_short": s,
                         "fut_stk_long": 1, "fut_stk_short": 1,
                         "opt_idx_call_long": 1000 + l // 10, "opt_idx_call_short": 1000,
                         "opt_idx_put_long": 1000, "opt_idx_put_short": 1000 + l // 20,
                         "opt_stk_call_long": 1, "opt_stk_call_short": 1,
                         "opt_stk_put_long": 1, "opt_stk_put_short": 1,
                         "total_long": 1, "total_short": 1})
    upsert_fao_data(pd.DataFrame(rows))
    return px


def _fresh():
    from src.analytics import positioning_cycle as pc
    pc._frame_cached.cache_clear()
    return pc


@pytest.fixture
def days():
    return _sessions(date(2021, 1, 1), 400)   # clear of the Nov-2024 regime break


def test_basic_state_and_zero_sum(temp_db, days):
    _seed(days)
    pc = _fresh()
    v = pc.get_positioning_cycle(days[-1])
    assert not v.get("error"), v.get("error")
    assert v["data_date"] == days[-1]
    assert v["balanced"] is True
    assert {p["key"] for p in v["players"]} == {"fii", "client", "pro", "dii"}
    assert 0 <= v["fii"]["pct"] <= 100
    assert v["fii"]["zone"] in {"xs", "s", "m", "l", "xl"}
    assert sum(z["today"] for z in v["zone_rows"]) == 1
    assert sum(m["today"] for m in v["matrix"]) <= 1
    assert len(v["matrix"]) == 20


def test_holiday_rows_are_ignored(temp_db, days):
    hol = date(2021, 6, 12)          # a Saturday: never a Nifty session
    assert hol.weekday() == 5
    _seed(days, holiday=hol)
    pc = _fresh()
    v = pc.get_positioning_cycle(days[-1])
    assert v["holiday_rows_ignored"] == 1
    assert hol not in set(v["series"]["date"].dt.date)


def test_no_lookahead(temp_db, days):
    """The state on day k must not change when later data exists."""
    _seed(days)
    pc = _fresh()
    k = days[300]
    full = pc.get_positioning_cycle(k)
    # rebuild using only rows <= k by asking the same as_of: values must be identical
    pc._frame_cached.cache_clear()
    again = pc.get_positioning_cycle(k)
    assert full["fii"]["pct"] == again["fii"]["pct"]
    later = pc.get_positioning_cycle(days[-1])
    row = later["series"].set_index(later["series"]["date"].dt.date).loc[k]
    assert row["fii_pct"] == pytest.approx(full["fii"]["pct"])


def test_turns_only_listed_once_confirmed(temp_db, days):
    px = _seed(days)
    pc = _fresh()
    v = pc.get_positioning_cycle(days[-1])
    for t in v["ledger"]:
        i = days.index(t["date"])
        after = px[i:]
        if t["kind"] == "TOP":
            assert after.min() <= px[i] * 0.95 + 1e-6
        else:
            assert after.max() >= px[i] * 1.05 - 1e-6
    kinds = [t["kind"] for t in v["ledger"]]
    assert all(a != b for a, b in zip(kinds, kinds[1:])), "turns must alternate"


def test_short_history_has_no_zone(temp_db):
    d = _sessions(date(2024, 1, 1), 60)
    _seed(d)
    pc = _fresh()
    v = pc.get_positioning_cycle(d[-1])
    assert not v.get("error")
    assert v["have_pct"] is False
    assert v["fii"]["zone"] is None
    assert any("history" in w for w in v["warnings"])


def test_missing_category_warns(temp_db, days):
    _seed(days, drop_types=("DII",))
    pc = _fresh()
    v = pc.get_positioning_cycle(days[-1])
    assert any("3 of 4" in w for w in v["warnings"])


def test_imbalanced_file_warns(temp_db, days):
    _seed(days, imbalance_on=days[-1])
    pc = _fresh()
    v = pc.get_positioning_cycle(days[-1])
    assert v["balanced"] is False
    assert any("do not match" in w for w in v["warnings"])


def test_before_history_returns_error_not_exception(temp_db, days):
    _seed(days)
    pc = _fresh()
    v = pc.get_positioning_cycle(days[0] - timedelta(days=30))
    assert "error" in v


def test_zigzag_alternates_and_ignores_small_moves():
    from src.analytics.positioning_cycle import _zigzag
    c = np.array([100, 102, 101, 103, 98, 97, 99, 103, 104, 98], float)
    t = _zigzag(c, 0.05)
    assert t == [(3, "TOP"), (5, "BOTTOM"), (8, "TOP")]
