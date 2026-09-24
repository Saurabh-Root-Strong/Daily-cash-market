"""Offline tests for the SLB build-up engine and the corporate-action parsers."""
import numpy as np

from src.analytics.slb_episodes import (START_FRACTION, episode_start, fifo_cost,
                                        phase_of, roll_adjust)
from src.ingestion.corp_actions_fetcher import classify_action, parse_share_factor

EXP = np.datetime64("2026-09-01")


def _win(n_before, n_after=0):
    return np.array([False] * 1 + [True] * n_before + [False] * n_after)


def test_roll_borrow_first_then_return_is_not_new_short():
    # near loan 100 stays open; 60 borrowed in next month; near returned on the date
    R = np.array([110, 110, 170, 170, 70.0])     # raw total
    N = np.array([100, 100, 100, 100, 0.0])      # expiring month
    win = np.array([False, True, True, True, False])
    E = roll_adjust(R, N, win, np.array([EXP] * 5)).ravel()
    assert list(E) == [110, 110, 110, 110, 70]   # flat through the roll, -40 on the day


def test_roll_repay_early_and_reborrow_has_no_overlap():
    # INFY pattern: 40 of the near loan repaid early while 40 borrowed next month
    R = np.array([110, 110, 110, 110, 50.0])
    N = np.array([100, 100, 60, 60, 0.0])
    win = np.array([False, True, True, True, False])
    E = roll_adjust(R, N, win, np.array([EXP] * 5)).ravel()
    assert list(E) == [110, 110, 110, 110, 50]   # a roll, not a cover + a new short


def test_pure_cover_is_seen_immediately():
    R = np.array([110, 110, 80, 80, 10.0])
    N = np.array([100, 100, 70, 70, 0.0])
    win = np.array([False, True, True, True, False])
    E = roll_adjust(R, N, win, np.array([EXP] * 5)).ravel()
    assert list(E) == [110, 110, 80, 80, 10]


def test_outside_window_is_raw():
    R = np.array([5, 9, 3.0])
    N = np.array([5, 9, 3.0])
    E = roll_adjust(R, N, np.array([False] * 3), np.array([EXP] * 3)).ravel()
    assert list(E) == [5, 9, 3]


def test_fifo_returns_oldest_first():
    e = np.array([100, 200, 150, 150.0])
    px = np.array([10, 20, 30, 40.0])
    c = fifo_cost(e, px)
    assert c[0] == 10 and c[1] == 15
    # 50 of the oldest (price 10) returned -> 50 @10 + 100 @20
    assert abs(c[2] - (50 * 10 + 100 * 20) / 150) < 1e-9
    assert c[3] == c[2]


def test_fifo_resets_after_full_return():
    c = fifo_cost(np.array([100, 0, 50.0]), np.array([10, 20, 30.0]))
    assert np.isnan(c[1]) and c[2] == 30


def test_episode_start_is_after_last_quarter_size_day():
    e = np.array([500, 20, 30, 80, 100.0])      # 20 < 25% of 100, 30 is not
    s, known = episode_start(e, 4)
    assert (s, known) == (2, True)
    s, known = episode_start(np.array([90, 95, 100.0]), 2)
    assert (s, known) == (0, False)
    assert START_FRACTION == 0.25


def test_phase():
    assert phase_of(60, 100, 60) == "Covering"
    assert phase_of(100, 100, 80) == "Building"
    assert phase_of(100, 100, 99) == "Holding"


def test_share_factor_and_kind():
    assert parse_share_factor("Face Value Split (Sub-Division) - From Rs 10/- Per Share "
                              "To Rs 2/- Per Share") == 5
    assert parse_share_factor("Bonus 1:2") == 1.5
    assert parse_share_factor("Bonus- 1:2") == 1.5
    assert parse_share_factor("Bonus 4:1") == 5
    assert parse_share_factor("Scheme Of Arrangement - Bonus Ncrps 4:1") == 1   # preference shares
    assert parse_share_factor("Consolidation From Re 1/- Per Share To Rs 10/- Per Share") == 0.1
    assert classify_action("Demerger") == "DEMERGER"
    assert classify_action("Bonus 1:1") == "BONUS"
    assert classify_action("Interim Dividend - Rs 5 Per Share") == "DIVIDEND"
