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
    assert len(blocks) >= 5, f'expected the 3 lookup tables + 2 matrix tables, got {len(blocks)}'
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


# ── audit 2026-09-11: one test per confirmed bug ──────────────────────────────
def test_a_return_across_a_source_hole_is_not_a_one_day_move(temp_db):
    """CFM's 2 Jul 2018 crude row carried +12.9% -- the whole 31 May -> 2 Jul
    move across the missing June -- and read as a one-session spike."""
    from src.data.repository import get_repository
    days = [date(2019, 1, 1) + timedelta(days=i) for i in range(10)]
    days += [date(2019, 3, 1) + timedelta(days=i) for i in range(10)]   # 50-day hole
    r = [0.001] * 10 + [0.129] + [0.001] * 9
    get_repository().replace_commodity_daily(pd.DataFrame({
        "trade_date": days, "commodity": "CRUDE OIL", "symbol": "X",
        "expiry_date": days, "close": 100.0, "ret1": r, "turnover_cr": 1.0}))
    p = ci.load_commodity("CRUDE OIL")
    assert p.loc["2019-03-01", "ret1"] == 0.0
    f = ci.commodity_features(p)
    after = f.loc["2019-03-01":"2019-03-04"]
    assert after["r5"].isna().all(), "a 5-session window spanning the hole was computed"
    assert f.loc["2019-03-06", "r5"] == pytest.approx(1.001 ** 5 - 1)


def test_a_rally_hovering_at_the_line_is_one_event_not_many():
    m = pd.Series([True, False, True, False, False, True, False, False, False,
                   False, False, False, True])
    s = ci.episode_starts(m, cooldown=5)
    assert s.tolist().count(True) == 2, "on/off/on within the cooldown was recounted"
    assert ci._independent(np.array([0, 3, 20, 22, 45]), 20) == 3


def test_one_episode_does_not_crash_and_says_too_few(temp_db):
    days, _ = _seed()
    ev, sm = ci.get_pattern_episodes(days[-1], "CRUDE OIL", "Nifty 50", "up_streak", n=5)
    assert sm["episodes"] == 1
    assert sm["next1"]["noise"] is None and sm["next1"]["n_eff"] == 1


def test_a_stopped_sync_is_flagged_not_explained_away(temp_db):
    days, _ = _seed()
    from src.data.repository import get_repository, query_dataframe
    # drop the last 6 MCX sessions, as if the sync had stopped a week ago
    cut = days[-7]
    df = query_dataframe("SELECT * FROM commodity_daily WHERE trade_date <= ?", [cut])
    get_repository().replace_commodity_daily(df)
    s = ci.get_commodity_state(days[-1], "CRUDE OIL", "Nifty 50")
    assert s.stale and "sync" in s.lag_note
    s2 = ci.get_commodity_state(days[-7], "CRUDE OIL", "Nifty 50")
    assert not s2.stale


def test_today_in_pattern_uses_the_latest_mcx_close(temp_db):
    days, _ = _seed()
    _, sm = ci.get_pattern_episodes(days[200], "CRUDE OIL", "Nifty 50", "up_streak", n=5)
    assert sm["today_in"] and sm["today_mcx"] == days[200]
    _, sm = ci.get_pattern_episodes(days[-1], "CRUDE OIL", "Nifty 50", "up_streak", n=5)
    assert not sm["today_in"]


def test_rail_carries_the_rupee_split_and_no_contradicting_example():
    from pathlib import Path
    body = Path("src/dashboard/views/commodity_index.py").read_text(encoding="utf-8")
    assert "usdinr_link_full" in body and "n50_up5_next_day" not in body, (
        "the rail quoted an every-day-of-the-streak average right above the "
        "once-per-episode lookup, which showed a different number for the same "
        "label")
    assert ci.STUDY["crude"]["usdinr_link_full"] < -0.3


# ── commodity x sector matrix ─────────────────────────────────────────────────
def test_liquidity_gate_is_applied_before_break_detection(temp_db):
    """Dropping a dead year AFTER loading would measure a 10-session window
    straight across the hole -- the same class of bug as the June-2018 one."""
    from src.data.repository import get_repository
    days = _weekdays(date(2019, 1, 1), 260) + _weekdays(date(2021, 1, 1), 260)
    r = [0.001] * 520
    get_repository().replace_commodity_daily(pd.DataFrame({
        "trade_date": days, "commodity": "NICKEL", "symbol": "N",
        "expiry_date": days, "close": 100.0, "ret1": r,
        # 2019 trades Rs 500 Cr, 2021 trades Rs 1 Cr -- a dead market
        "turnover_cr": [500.0] * 260 + [1.0] * 260}))
    p = ci.load_commodity("NICKEL", min_median_turnover_cr=50.0)
    assert set(p.index.year) == {2019}
    all_years = ci.load_commodity("NICKEL")
    assert set(all_years.index.year) == {2019, 2021}
    f = ci.commodity_features(p)
    assert f.r10.notna().sum() > 200


def test_sector_matrix_finds_metal_and_leaves_the_controls_alone():
    m = ci.get_sector_matrix(date(2026, 9, 11), years=6.0, sector_only=True)
    rho, band = m["rho"], m["band"]
    assert m["weeks"] > 200 and 0 < band < 0.2
    # copper is the metal sector's commodity, by a distance
    assert rho.loc["Nifty Metal", "COPPER"] > 0.35
    assert rho.loc["Nifty Metal", "COPPER"] == rho["COPPER"].max()
    # ...and the controls show nothing
    for ctrl in ("Nifty Bank", "Nifty IT", "Nifty Pharma"):
        assert abs(rho.loc[ctrl, "COPPER"]) < band * 2, (
            f"{ctrl} should not track copper; if it does, the method is "
            f"measuring the market rather than the metal")
    # the market cannot be compared with itself once it is subtracted
    assert "Nifty 50" not in rho.index
    assert "Nifty 50" in ci.get_sector_matrix(
        date(2026, 9, 11), years=6.0, sector_only=False)["rho"].index


def test_sector_matrix_never_reads_past_the_selected_date():
    early = ci.get_sector_matrix(date(2024, 6, 28), years=6.0)
    late = ci.get_sector_matrix(date(2026, 9, 11), years=6.0)
    assert early["rho"].loc["Nifty Metal", "COPPER"] != late["rho"].loc["Nifty Metal", "COPPER"]


def test_frozen_matrix_verdict_says_only_the_gap_beat_the_search():
    m = ci.STUDY["matrix"]
    assert m["rc_best"] > m["rc_null_95"], "the headline claim is that ONE thing beat it"
    assert all("gap" in x for x in m["rc_beats"]), (
        "everything that beat the search-aware null is an opening gap -- if a "
        "tradable one is ever added, the page's wording must change with it")
    for label, stat in m["rc_candidates"].items():
        assert stat < m["rc_null_95"], f"{label} is quoted as a candidate but beats the bar"
    # the overlap lesson, kept where it can be seen
    assert m["lead_overlap_adjusted"] < 2 < m["lead_hac_t"]


def test_a_dead_market_is_flagged_on_the_page():
    s = ci.get_commodity_state(date(2026, 9, 11), "NICKEL", "Nifty Metal")
    assert s.liquid_note and "stopped trading" in s.liquid_note
    assert not ci.get_commodity_state(date(2026, 9, 11), "COPPER", "Nifty Metal").liquid_note


# ── month ahead ───────────────────────────────────────────────────────────────
def test_month_ahead_separates_same_month_from_next_month():
    m = ci.get_month_ahead(date(2026, 9, 11), "COPPER")
    t = m["table"].set_index("sector")
    assert m["months"] > 60 and 1 <= m["quintile"] <= 5
    # copper's same-month partner is the metal sector, by a distance
    assert t.same_month_ic.idxmax() == "Nifty Metal"
    assert t.loc["Nifty Metal", "beta"] > 0.3
    # the implied co-move is beta x this month's commodity move, nothing else
    assert t.loc["Nifty Metal", "implied_same_month"] == pytest.approx(
        t.loc["Nifty Metal", "beta"] * m["month_move"], rel=1e-9)
    # next-month numbers must come from a handful of months and carry a band
    assert 3 <= t.next_month_n.max() <= 30
    assert t.next_month_band.notna().all()


def test_month_ahead_reads_nothing_after_the_selected_date():
    early = ci.get_month_ahead(date(2025, 9, 11), "COPPER")
    late = ci.get_month_ahead(date(2026, 9, 11), "COPPER")
    assert early["month_move"] != late["month_move"]
    assert early["mcx_date"] <= date(2025, 9, 11)


def test_frozen_monthly_verdict_is_a_null_and_says_why():
    mo = ci.STUDY["monthly"]
    # the walk-forward ranking was WRONG, not right -- the page must not sell it
    assert mo["wf_ic"] < 0 and mo["wf_pos_share"] < 0.5
    # single pairs: nothing survives, and the best is below a shuffled search
    assert mo["pair_fdr"] == 0
    assert mo["pair_rc_best"] < mo["pair_rc_null_median"]
    assert mo["pair_rc_p"] > 0.10
    # and the honest power statement: what could have been seen at all
    assert mo["ic_needed"] == pytest.approx(1.96 / (mo["months"] ** 0.5), abs=0.01)
    # the same-MONTH exposure is the part that is real
    assert mo["same_month_fdr"] >= 10
    assert mo["same_month_top"]["COPPER/Nifty Metal"][0] > 0.4


def test_per_one_percent_column_is_not_scaled_twice():
    """beta is already sector-% per commodity-%: scaling it by 100 printed
    '+13.07%' for a sector that moves 0.13% per 1% of crude."""
    from pathlib import Path
    body = Path("src/dashboard/views/commodity_index.py").read_text(encoding="utf-8")
    i = body.index('"Per 1% move"')
    assert "t.beta * 100" not in body[i:i + 200]
    m = ci.get_month_ahead(date(2026, 9, 11), "COPPER")
    assert abs(m["table"].beta).max() < 2.0, (
        "a sector moving more than 2% for each 1% of a commodity would be a "
        "unit error, not a market")
