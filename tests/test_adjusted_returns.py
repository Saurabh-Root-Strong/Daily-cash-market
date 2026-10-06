"""Corporate-action adjusted returns: one definition, used everywhere.

daily_data.prev_close is the RAW previous close. These tests pin the shared rule
(src/analytics/adjusted_returns.py), the cases it exists for, and -- by scanning
the source -- that no panel goes back to the raw expression.
"""
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.analytics import adjusted_returns as ar
from src.data.repository import query_dataframe as q

RAW = "(b.close_price - b.prev_close) / b.prev_close * 100"


def _one(symbol, day):
    df = q(f"""SELECT {RAW} AS raw, {ar.ret_sql('b')} AS adj FROM daily_data b
               WHERE b.symbol = ? AND b.trade_date = ? AND b.series = 'EQ'""", [symbol, day])
    return float(df["raw"].iloc[0]), float(df["adj"].iloc[0])


def test_identical_to_the_raw_return_on_every_day_without_a_corporate_action():
    """The adjustment must not move a single ordinary row."""
    df = q(f"""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN ABS({ar.ret_sql('b')} - {RAW}) > 1e-9 THEN 1 ELSE 0 END) AS diff,
               SUM(CASE WHEN ABS({ar.ret_sql('b')} - {RAW}) > 1e-9
                         AND {ar.factor_sql('b')} = 1 AND NOT {ar.demerger_sql('b')}
                        THEN 1 ELSE 0 END) AS unexplained
        FROM daily_data b
        WHERE b.series = 'EQ' AND b.prev_close > 0
          AND b.trade_date BETWEEN '2025-01-01' AND '2026-09-30'
    """)
    n, diff, unexplained = (int(df[c].iloc[0]) for c in ("n", "diff", "unexplained"))
    assert n > 500_000 and 0 < diff < 400, (n, diff)
    assert unexplained == 0, "a row changed without a split, bonus or demerger"


@pytest.mark.parametrize("symbol,day,raw_below,lo,hi", [
    ("TRENT", date(2026, 6, 4), -30, -3, 3),         # bonus 1:2, slipped a +-40% guard
    ("HDFCBANK", date(2025, 8, 26), -45, -4, 2),     # bonus 1:1
    ("BAJFINANCE", date(2025, 6, 16), -85, -3, 4),   # split AND bonus, compound x10
    ("RELIANCE", date(2023, 7, 20), -6, 0, 3),       # demerger: no factor, open gap
    ("TATAMOTORS", date(2025, 10, 14), -38, -3, 1),  # demerger filed under TMPV
    ("HEG", date(2024, 10, 18), -75, -6, 6),         # split filed under HEGAM
])
def test_known_corporate_action_days(symbol, day, raw_below, lo, hi):
    raw, adj = _one(symbol, day)
    assert raw < raw_below, f"{symbol}: the raw print should look like a crash ({raw:.1f})"
    assert lo < adj < hi, f"{symbol}: adjusted return {adj:.1f} is not a normal day"


def test_a_real_crash_is_left_alone():
    """POLICYBZR 24 Sep 2026: -36% with OI +54% has the exact shape of a re-basing
    and was a real crash. Nothing is inferred from the tape."""
    raw, adj = _one("POLICYBZR", date(2026, 9, 24))
    assert raw == pytest.approx(adj) and adj < -30


def test_rename_map_comes_from_nses_list_and_a_failure_is_not_cached(monkeypatch):
    m = ar.ca_symbol_map()
    assert m["TATAMOTORS"] == "TMPV" and m["HEG"] == "HEGAM" and m["ZOMATO"] == "ETERNAL"
    assert 20 <= len(m) <= 200, "only renames that hide a corporate action belong here"
    # database locked at startup: the small fallback for THAT call, then retried
    import src.data.repository as repo
    monkeypatch.setattr(ar, "_MAP_CACHE", {})
    monkeypatch.setattr(repo, "query_dataframe", lambda *a, **k: (_ for _ in ()).throw(OSError("locked")))
    assert ar.ca_symbol_map() == ar._FALLBACK
    assert ar._MAP_CACHE == {}, "a failed build must never be cached"
    monkeypatch.undo()
    assert len(ar.ca_symbol_map()) >= 20


def test_window_return_is_free_of_a_bonus_inside_the_window():
    """HDFCBANK Aug-Sep 2025 read -52.9% (1:1 bonus on 26 Aug)."""
    from src.analytics.sector_rotation import get_sector_stocks_custom_range
    sec = q("SELECT sector FROM v_sector_master WHERE symbol = 'HDFCBANK'")["sector"].iloc[0]
    df = get_sector_stocks_custom_range(sec, date(2025, 8, 1), date(2025, 9, 30), 100.0)
    r = float(df[df["symbol"] == "HDFCBANK"]["period_ret_pct"].iloc[0])
    assert -15 < r < 15, r


def test_window_return_bridges_a_demerger_inside_a_trading_suspension():
    """India Glycols demerged 02 Sep 2026 and did not trade for two weeks: there is
    no ex-date row, and prev_close on the resumption day is already re-based. A
    level-to-level return across it read -72.7%."""
    from src.analytics.sector_rotation import get_clock_stock_detail
    sec = q("SELECT sector FROM v_sector_master WHERE symbol = 'INDIAGLYCO'")["sector"].iloc[0]
    df = get_clock_stock_detail(sec, date(2026, 10, 5), 0.0, lookback_days=20)
    row = df[df["symbol"] == "INDIAGLYCO"]
    if row.empty:
        pytest.skip("INDIAGLYCO not in the drill-down universe on this date")
    assert -40 < float(row["ret_win_pct"].iloc[0]) < 60


def test_bridge_is_one_without_an_official_action_in_the_gap():
    sql = ar.bridge_sql("'TCS'", "DATE '2026-09-01'", "DATE '2026-09-10'", "100.0", "50.0")
    assert float(q(f"SELECT {sql} AS k")["k"].iloc[0]) == 1.0, \
        "a gap alone (another series, a halt) is a real move and must be kept"
    sql = ar.bridge_sql("'TRENT'", "DATE '2026-06-03'", "DATE '2026-06-04'", "150.0", "100.0")
    assert float(q(f"SELECT {sql} AS k")["k"].iloc[0]) == pytest.approx(1.5)


def test_back_adjust_puts_an_ohlc_panel_on_one_share_basis():
    df = q("""SELECT symbol, trade_date, open_price AS o, high_price AS h, low_price AS l,
                     close_price AS c, prev_close AS pc FROM daily_data
              WHERE symbol = 'TRENT' AND series = 'EQ'
                AND trade_date BETWEEN '2026-05-20' AND '2026-06-12' ORDER BY trade_date""")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    raw_jump = df["c"].pct_change().abs().max()
    adj = ar.back_adjust(df)
    assert raw_jump > 0.25, "the raw closes should show the bonus as a jump"
    assert adj["c"].pct_change().abs().max() < 0.12, "back-adjusted closes are continuous"
    ex = adj[adj["trade_date"] == date(2026, 6, 4)].iloc[0]
    assert abs(ex["o"] / ex["pc"] - 1) < 0.05, "no fake gap on the ex-date"
    # a panel with no action in it comes back untouched
    plain = df[df["trade_date"] > date(2026, 6, 4)]
    pd.testing.assert_frame_equal(ar.back_adjust(plain), plain)


def test_no_panel_computes_a_return_from_the_raw_previous_close():
    """The guard. Any new `(close - prev_close) / prev_close` on daily_data books a
    split as a crash; it has to go through ret_sql()."""
    raw = re.compile(r"close_price\s*-\s*\w*\.?prev_close\)\s*/\s*(NULLIF\()?\s*\w*\.?prev_close")
    breadth = re.compile(r"close_price\s*>\s*prev_close")
    offenders = []
    for p in list(Path("src/analytics").glob("*.py")) + list(Path("src/dashboard").rglob("*.py")):
        if p.name == "adjusted_returns.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith(("#", "--")) or "Formula:" in line:
                continue
            if raw.search(line) or breadth.search(line):
                offenders.append(f"{p}:{i}: {line.strip()[:70]}")
    assert not offenders, "raw previous-close returns found:\n" + "\n".join(offenders)


def test_deal_outcome_builder_no_longer_claims_prev_close_is_adjusted():
    src = Path("scripts/build_deal_outcomes.py").read_text(encoding="utf-8")
    assert "NOT ex-adjusted" in src and "__APC__" in src and "prev_close_sql" in src
    view = Path("src/dashboard/views/bulk_block_deals.py").read_text(encoding="utf-8")
    assert "(NSE's prev_close is ex-adjusted)" not in view
