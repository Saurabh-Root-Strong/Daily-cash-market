"""Offline tests for the Index P/E & P/B panel's building blocks.

Each test pins one trap found while building it; none touches the network or DB.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.index_valuation import (drawdown_episodes, pit_percentile,
                                           splice_breaks, valuation_zone)
from src.ingestion.index_valuation_fetcher import _num, parse_day


def test_pit_percentile_never_sees_the_future():
    s = pd.Series(np.arange(1000, dtype=float))
    a = pit_percentile(s, min_obs=10)
    # change only the future: every earlier percentile must stay identical
    s2 = s.copy()
    s2.iloc[600:] = -1.0
    b = pit_percentile(s2, min_obs=10)
    pd.testing.assert_series_equal(a.iloc[:600], b.iloc[:600])
    # a monotone rising series is always at the top of its own past
    assert a.iloc[500] == 1.0
    assert a.iloc[:10].isna().all()


def test_pit_percentile_skips_nulls_without_filling():
    s = pd.Series([1.0, 2.0, np.nan, 3.0, 0.5])
    p = pit_percentile(s, min_obs=2)
    assert np.isnan(p.iloc[2])
    assert p.iloc[3] == 1.0            # 3 vs {1,2}
    assert p.iloc[4] == 0.0            # 0.5 vs {1,2,3}


def _frame_with_break(factor: float = 0.83) -> pd.DataFrame:
    d = pd.bdate_range("2021-03-01", "2021-04-30")
    close = pd.Series(np.linspace(100, 110, len(d)))
    pe = close / 5.0
    dy = 100.0 / close
    i = int(np.where(d == pd.Timestamp("2021-03-31"))[0][0])
    pe.iloc[i:] *= factor              # accounting change, not a price move
    dy.iloc[i:] *= 0.9
    return pd.DataFrame({"trade_date": d, "close_val": close, "pe": pe,
                         "pb": close / 20.0, "dy": dy, "tri": close})


def test_splice_finds_the_2021_break_and_leaves_pb_alone():
    out, info = splice_breaks(_frame_with_break())
    assert info["pe"]["date"] == "2021-03-31"
    assert info["pe"]["factor"] == pytest.approx(0.83, abs=1e-3)
    assert info["dy"]["factor"] == pytest.approx(0.9, abs=1e-3)
    assert info["pb"] is None
    # after splicing, P/E moves only with price: no jump on the break day
    j = np.log(out["pe_adj"]).diff() - np.log(out["close_val"]).diff()
    assert j.abs().max() < 1e-9


def test_drawdown_episodes_peak_trough_recovery():
    c = pd.Series([100, 110, 90, 80, 95, 111, 120, 100, 119, 130.0])
    d = pd.Series(pd.bdate_range("2020-01-01", periods=len(c)))
    ep = drawdown_episodes(c, d, 0.20)
    assert len(ep) == 1                           # 110 -> 80 (-27%); 120 -> 100 is -17%
    e = ep.iloc[0]
    assert (e["peak_i"], e["trough_i"], e["recover_i"]) == (1, 3, 5)
    assert e["depth"] == pytest.approx(80 / 110 - 1)


def test_parse_day_is_locale_free_and_handles_both_formats():
    s = pd.Series(["01 Jan 1999", "31-MAR-2021", "garbage"])
    p = parse_day(s)
    assert p.iloc[0] == pd.Timestamp("1999-01-01")
    assert p.iloc[1] == pd.Timestamp("2021-03-31")
    assert pd.isna(p.iloc[2])


def test_dash_pe_is_null_not_zero():
    v = _num(pd.Series(["-", "", "0", "1,234.5", "18.2"]))
    assert v.isna().tolist() == [True, True, True, False, False]
    assert v.iloc[3] == 1234.5


def test_zone_labels():
    assert valuation_zone(0.95) == "Extreme expensive"
    assert valuation_zone(0.75) == "Expensive"
    assert valuation_zone(0.5) == "Fair"
    assert valuation_zone(0.2) == "Cheap"
    assert valuation_zone(0.05) == "Extreme cheap"
    assert valuation_zone(None) == "—"


def test_forward_drawdown_excludes_today_and_covers_252_sessions():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "ivs", pathlib.Path("scripts/index_valuation_study.py"))
    ivs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ivs)
    n = 600
    c = np.full(n, 100.0)
    c[252] = 50.0                                   # exactly 252 sessions after day 0
    c[253] = 40.0                                   # 253 after day 0: out of its window
    f = pd.DataFrame({"trade_date": pd.bdate_range("2000-01-03", periods=n),
                      "close_val": c, "tri": c})
    out = ivs.outcomes(f)
    assert out["mdd_12m"].iloc[0] == pytest.approx(-0.5)
    assert out["mdd_12m"].iloc[1] == pytest.approx(-0.6)
    assert out["fwd_12m"].iloc[0] == pytest.approx(np.log(0.5))
    assert np.isnan(out["mdd_12m"].iloc[n - 252])


def test_vix_window_that_hits_the_70_row_cap_raises(monkeypatch):
    from src.ingestion import index_valuation_fetcher as m

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"EOD_TIMESTAMP": "01-JAN-2020",
                              "EOD_CLOSE_INDEX_VAL": 12.0}] * 70}

    class S:
        headers: dict = {}

        def get(self, *a, **k):
            return R()

    f = m.IndexValuationFetcher(polite_delay=0)
    monkeypatch.setattr(f, "_nse_session", lambda *_: S())
    from datetime import date
    with pytest.raises(RuntimeError, match="70-row cap"):
        f.fetch_vix(date(2020, 1, 1), date(2020, 3, 1))


def test_unbridgeable_break_drops_history_instead_of_rescaling():
    out, info = splice_breaks(_frame_with_break(factor=3.65))
    assert info["pe"]["dropped_pre_break"] is True
    pre = out["trade_date"] < "2021-03-31"
    assert out.loc[pre, "pe_adj"].isna().all()
    assert out.loc[~pre, "pe_adj"].notna().all()


def test_heatmap_month_cell_is_the_month_end_not_the_last_known_day():
    from src.dashboard.views.index_valuation import _forward_grid, _grid
    n = 300
    d = pd.bdate_range("2020-01-01", periods=n)
    c = np.linspace(100, 130, n)
    f = pd.DataFrame({"trade_date": d, "close_val": c, "tri": c, "x": np.arange(n, dtype=float)})
    f.loc[f.index[-3:], "x"] = np.nan                  # month-end value unknown
    g = _grid(f, "x", "last")
    last_y, last_m = d[-1].year, d[-1].month
    assert np.isnan(g.loc[last_y, last_m])             # blank, not an earlier day's value
    fwd, dd = _forward_grid(f)
    jan = f[(f.trade_date.dt.year == 2020) & (f.trade_date.dt.month == 1)].index[-1]
    assert fwd.loc[2020, 1] == pytest.approx(c[jan + 252] / c[jan] - 1)
    assert dd.loc[2020, 1] == pytest.approx(0.0, abs=1e-12) or dd.loc[2020, 1] > 0


def test_whole_history_rank_ignores_blank_cells():
    from src.dashboard.views.index_valuation import _whole_history_rank
    g = pd.DataFrame({1: [10.0, 20.0], 2: [30.0, np.nan], 3: [40.0, np.nan]}, index=[2025, 2026])
    r = _whole_history_rank(g)
    # 4 real values: 10, 20, 30, 40 -> 40 is above 3 of 4 (+ half its own tie)
    assert r.loc[2025, 3] == pytest.approx((3 + 0.5) / 4)
    assert r.loc[2026, 1] == pytest.approx((1 + 0.5) / 4)
    assert np.isnan(r.loc[2026, 2])


def test_return_level_never_mixes_tri_and_price_scales():
    from src.analytics.index_valuation import return_level
    # price flat at 100; TRI appears on day 3 at 106 (different scale) and has a hole
    f = pd.DataFrame({"close_val": [100.0] * 8,
                      "tri": [np.nan, np.nan, np.nan, 106.0, 106.0, np.nan, 106.0, 106.0]})
    lvl = return_level(f)
    assert lvl.max() == pytest.approx(1.0) and lvl.min() == pytest.approx(1.0)


def test_add_features_on_an_empty_frame_does_not_crash(monkeypatch):
    """Picking a date before an index launched returns zero rows."""
    from src.analytics import index_valuation as iv
    monkeypatch.setattr(iv, "_gsec_pit", lambda d: pd.Series(np.nan, index=d.index))
    empty = pd.DataFrame({"trade_date": pd.to_datetime([]), "pe": [], "pb": [], "dy": [],
                          "close_val": [], "tri": []})
    out, _ = iv.add_features(empty)
    assert out.empty and "val_score" in out


def test_index_groups_and_picker_order(monkeypatch):
    from src.analytics import index_valuation as iv
    assert iv.index_group("Nifty 50") == "Broad market"
    assert iv.index_group("Nifty Private Bank") == "Sectoral"
    assert iv.index_group("Nifty India Defence") == "Thematic"
    fake = pd.DataFrame({"index_name": ["Nifty IT", "Nifty CPSE", "Nifty 500", "Nifty 50"],
                         "first_day": ["1999-01-01"] * 4, "last_day": ["2026-09-23"] * 4,
                         "n": [1, 1, 1, 1]})
    monkeypatch.setattr(iv, "query_dataframe", lambda *a, **k: fake.copy())
    assert iv.index_catalog()["index_name"].tolist() == [
        "Nifty 50", "Nifty 500", "Nifty IT", "Nifty CPSE"]
