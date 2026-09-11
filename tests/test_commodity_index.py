"""Commodity vs Index: the properties that, if broken, would print a fake result.

1. TIMING. MCX closes at 23:30, NSE at 15:30. A crude close dated T must never be
   attached to NSE session T -- only to the first session AFTER it.
2. NO FUTURE. Outcomes that fall after the selected date stay blank.
3. ONE EVENT PER RUN. A 7-day rally is one event, not five overlapping ones.
4. CAUSAL FEATURES. Changing tomorrow's price must not change today's reading.
5. The frozen study numbers the page quotes add up and say what was measured.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.analytics import commodity_index as ci


# ── synthetic market ──────────────────────────────────────────────────────────
def _weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _seed(n_days: int = 320, streak_end: int = 200, streak: int = 5):
    """Crude flat-ish with a clean `streak`-day rally ending on day `streak_end`;
    Nifty falls 1% on the session AFTER the rally and 3% on the rally day itself
    (the day the crude close cannot yet be known)."""
    from src.data.repository import get_repository
    days = _weekdays(date(2018, 7, 2), n_days)
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, n_days)
    r[r > 0] *= -1                                   # no accidental up-streaks
    r[::2] = np.abs(r[::2])                          # alternate up/down
    r[streak_end - streak + 1: streak_end + 1] = 0.02
    r[streak_end + 1] = -0.005                       # ends the streak
    close = 5000 * np.cumprod(1 + r)
    cmd = pd.DataFrame({"trade_date": days, "commodity": "CRUDE OIL",
                        "symbol": "CRUDEOIL", "expiry_date": days, "close": close,
                        "ret1": r, "turnover_cr": 1000.0})
    get_repository().replace_commodity_daily(cmd)

    nret = np.full(n_days, 0.001)
    nret[streak_end] = -0.03                         # same calendar day as the MCX close
    nret[streak_end + 1] = -0.01                     # the first session that can react
    nc = 20000 * np.cumprod(1 + nret)
    rows = []
    for i, d in enumerate(days):
        for name in ("Nifty 50", "Nifty Bank"):
            rows.append(dict(trade_date=d, index_name=name, open_val=nc[i - 1] if i else nc[0],
                             high_val=nc[i], low_val=nc[i], close_val=nc[i],
                             prev_close=nc[i - 1] if i else nc[0], points_chg=0.0,
                             pct_chg=0.0, volume=0, turnover_cr=0.0, pe_ratio=0.0,
                             pb_ratio=0.0, div_yield=0.0))
    get_repository().upsert_index_data(pd.DataFrame(rows))
    return days, nc


def test_replace_commodity_daily_is_a_whole_series_replace(temp_db):
    from src.data.repository import get_repository, query_dataframe
    repo = get_repository()
    d = pd.DataFrame({"trade_date": [date(2024, 1, 1), date(2024, 1, 2)],
                      "commodity": "GOLD", "symbol": "GOLD",
                      "expiry_date": date(2024, 2, 5), "close": [1.0, 2.0],
                      "ret1": [None, 1.0], "turnover_cr": 5.0})
    assert repo.replace_commodity_daily(d) == 2
    # a rebuild with one row fewer must not leave the stale row behind
    assert repo.replace_commodity_daily(d.iloc[:1]) == 1
    got = query_dataframe("SELECT count(*) n FROM commodity_daily WHERE commodity='GOLD'")
    assert int(got.n.iloc[0]) == 1


def test_crude_close_is_attached_to_the_NEXT_nse_session_never_the_same_day(temp_db):
    days, _ = _seed()
    ev, sm = ci.get_pattern_episodes(days[-1], "CRUDE OIL", "Nifty 50",
                                     "up_streak", n=5)
    assert sm["episodes"] == 1
    row = ev.iloc[0]
    assert row.mcx_date.date() == days[200]
    assert row.session.date() == days[201], (
        "the rally's MCX close (23:30) was matched to the same NSE date -- that "
        "session closed at 15:30, before the crude close existed")
    # and the 'next day' is the -1% reaction, not the -3% same-day move
    assert row.next1 == pytest.approx(-0.01, abs=1e-9)


def test_outcomes_after_the_selected_date_stay_blank(temp_db):
    days, _ = _seed()
    as_of = days[203]                    # 2 sessions after the reaction day
    ev, _ = ci.get_pattern_episodes(as_of, "CRUDE OIL", "Nifty 50", "up_streak", n=5)
    row = ev.iloc[0]
    assert pd.notna(row.next1)
    assert pd.isna(row.next5) and pd.isna(row.next20), (
        "a 5- or 20-session outcome was filled in from beyond the selected date")


def test_a_long_rally_counts_once():
    m = pd.Series([False, True, True, True, True, False, True, False])
    s = ci.episode_starts(m)
    assert s.tolist() == [False, True, False, False, False, False, True, False]


def test_streak_mask_and_features_are_causal():
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400), index=idx)
    p = pd.DataFrame({"close": 100 * (1 + r).cumprod(), "ret1": r})
    p["lvl"] = (1 + p.ret1).cumprod()
    f1 = ci.commodity_features(p)
    p2 = p.copy()
    p2.iloc[-1, p2.columns.get_loc("ret1")] = 0.5    # wild last day
    p2["lvl"] = (1 + p2.ret1).cumprod()
    f2 = ci.commodity_features(p2)
    pd.testing.assert_frame_equal(f1.iloc[:-1].drop(columns=["close"]),
                                  f2.iloc[:-1].drop(columns=["close"]))


def test_stale_commodity_state_is_not_carried_across_a_long_gap():
    feat = pd.DataFrame({"r1": [0.01, 0.02]},
                        index=pd.DatetimeIndex(["2018-05-30", "2018-07-02"],
                                               name="trade_date"))
    days = pd.DatetimeIndex(["2018-06-15", "2018-07-03"])
    a = ci.align_to_sessions(feat, days)
    assert pd.isna(a.loc["2018-06-15", "r1"]), "a 16-day-old crude close was used"
    assert a.loc["2018-07-03", "r1"] == 0.02


def test_frozen_study_numbers_add_up_and_crude_has_no_survivor():
    g = ci.STUDY["grid"]
    assert sum(v["tests"] for v in g.values()) == ci.STUDY["cumulative_tests"]
    assert g["CRUDE OIL"]["fdr"] == 0 and g["CRUDE OIL"]["tradable"] == 0
    assert ci.STUDY["crude"]["fdr_survivors"] == 0
    # nominal hits for crude are about what chance produces -- the page says so
    assert abs(g["CRUDE OIL"]["nominal"] - g["CRUDE OIL"]["chance"]) < 25
    # the regime does not predict itself: below a coin flip
    assert ci.STUDY["crude"]["regime_sign_agree"] < 0.5
    for k, v in g.items():
        if v["fdr"] and k != "CRUDE OIL":
            assert k in ci.STUDY["survivor_note"], f"{k} survivors shown without a note"


def test_every_commodity_index_column_has_a_tooltip():
    import re
    from pathlib import Path
    body = Path("src/dashboard/views/commodity_index.py").read_text(encoding="utf-8")
    blocks = []
    for m in re.finditer(r"column_config=\{", body):
        i, depth = m.end(), 1
        while depth and i < len(body):
            depth += (body[i] == "{") - (body[i] == "}")
            i += 1
        blocks.append(body[m.end():i])
    assert len(blocks) == 3
    missing = []
    for b in blocks:
        for mm in re.finditer(r'\n\s+(f?"[^"]+"):\s*st\.column_config\.\w+\(', b):
            seg = b[mm.start():]
            nx = re.search(r'\n\s+f?"[^"]+":\s*st\.column_config', seg[1:])
            seg = seg if nx is None else seg[:nx.start() + 1]
            if "help=" not in seg:
                missing.append(mm.group(1))
    assert not missing, f"columns with no tooltip: {missing}"


def test_panel_is_on_the_sector_rotation_page():
    from src.dashboard.views import sector_rotation as sr
    assert "🛢️ Commodity vs Index" in sr._PANELS
