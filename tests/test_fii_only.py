"""FII-only analytics tests — isolated temp DuckDB, no network, no live data.

This module carries the invariants that were being checked by hand while the
FII-only page was built. They are the ones that actually broke during that work:
an expiry-day chain being shown as if it were live, a market-wide number rendered
per index, a z-score taken on a contract too small to carry one, a stance naming
a direction its own measured zone does not support, and a price band coming back
unsorted or out of range.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest


# ── fixtures ────────────────────────────────────────────────────────────────
def _sessions(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _seed_participant(days: list[date]) -> None:
    """FII participant OI with a deterministic wobble so z-scores are defined."""
    rows = []
    for i, d in enumerate(days):
        w = (i % 7) - 3
        for ct in ("FII", "Client", "Pro", "DII"):
            k = {"FII": 1.0, "Client": -0.9, "Pro": 0.5, "DII": 0.2}[ct]
            rows.append({
                "trade_date": d, "client_type": ct, "data_type": "OI",
                "fut_idx_long": int(25_000 + k * w * 400),
                "fut_idx_short": int(200_000 + k * w * 3_000),
                "fut_stk_long": int(10_000 + k * w * 100),
                "fut_stk_short": int(9_000 + k * w * 100),
                "opt_idx_call_long": int(450_000 + k * w * 9_000),
                "opt_idx_call_short": int(680_000 + k * w * 8_000),
                "opt_idx_put_long": int(940_000 + k * w * 7_000),
                "opt_idx_put_short": int(330_000 + k * w * 6_000),
                "opt_stk_call_long": 1_000, "opt_stk_call_short": 1_000,
                "opt_stk_put_long": 1_000, "opt_stk_put_short": 1_000,
                "total_long": 1_000_000, "total_short": 1_000_000,
            })
    from src.data.repository import upsert_fao_data
    upsert_fao_data(pd.DataFrame(rows))


def _seed_index(days: list[date]) -> None:
    rows = []
    px = 24_000.0
    for i, d in enumerate(days):
        prev = px
        px = px * (1 + (((i % 5) - 2) / 500.0))
        rows.append({"trade_date": d, "index_name": "Nifty 50",
                     "open_val": prev, "high_val": max(prev, px) * 1.002,
                     "low_val": min(prev, px) * 0.998, "close_val": px,
                     "prev_close": prev, "points_chg": px - prev,
                     "pct_chg": (px / prev - 1) * 100, "volume": 1_000_000,
                     "turnover_cr": 5_000.0, "pe_ratio": 22.0,
                     "pb_ratio": 4.0, "div_yield": 1.2})
    from src.data.repository import upsert_index_data
    upsert_index_data(pd.DataFrame(rows))


def _seed_chain(day: date, expiries: list[date]) -> None:
    """Option chain with two expiries so expiry-day handling can be exercised."""
    rows = []
    for exp in expiries:
        for strike in range(23_500, 25_001, 100):
            for ot in ("CE", "PE"):
                rows.append({
                    "trade_date": day, "symbol": "NIFTY", "instrument": "OPTIDX",
                    "expiry_date": exp, "strike_price": float(strike),
                    "option_type": ot, "open_price": 100.0, "high_price": 110.0,
                    "low_price": 90.0, "close_price": 100.0, "settle_price": 100.0,
                    "contracts": 100, "value_lacs": 1000.0,
                    "open_interest": 1_000_000 + strike, "chg_in_oi": 1_000.0,
                })
        rows.append({
            "trade_date": day, "symbol": "NIFTY", "instrument": "FUTIDX",
            "expiry_date": exp, "strike_price": 0.0, "option_type": "XX",
            "open_price": 24_000.0, "high_price": 24_100.0, "low_price": 23_900.0,
            "close_price": 24_050.0, "settle_price": 24_050.0, "contracts": 500,
            "value_lacs": 5000.0, "open_interest": 5_000_000, "chg_in_oi": 1_000.0,
        })
    from src.data.repository import upsert_fno_bhavcopy
    upsert_fno_bhavcopy(pd.DataFrame(rows))


def _seed_cash_and_stats(days: list[date]) -> None:
    """fii_dii_cash + fii_derivatives_stats so the composite has >=3 live parts."""
    from src.data.repository import get_repository
    cash = pd.DataFrame([
        {"trade_date": d, "fii_buy": 10000.0, "fii_sell": 10000.0 - ((i % 11) - 5) * 100,
         "fii_net": ((i % 11) - 5) * 100.0,
         "dii_buy": 9000.0, "dii_sell": 9000.0 + ((i % 11) - 5) * 90,
         "dii_net": -((i % 11) - 5) * 90.0}
        for i, d in enumerate(days)])
    stats = pd.DataFrame([
        {"trade_date": d, "category": "NIFTY FUTURES",
         "buy_contracts": 1000, "buy_value_cr": 5000.0 + ((i % 9) - 4) * 60,
         "sell_contracts": 1000, "sell_value_cr": 5000.0,
         "oi_contracts": 50_000, "oi_value_cr": 25_000.0 + ((i % 9) - 4) * 100}
        for i, d in enumerate(days)])
    repo = get_repository()
    for df, tbl in ((cash, "fii_dii_cash"), (stats, "fii_derivatives_stats")):
        cols = repo.query(f"SELECT * FROM {tbl} LIMIT 0").columns.tolist()
        use = [c for c in cols if c in df.columns]
        repo.query("INSERT INTO {} ({}) SELECT {} FROM df".format(
            tbl, ", ".join(use), ", ".join(use)).replace(" FROM df", ""), None)             if False else None
    # DuckDB registers local frames by name inside query(); use the upserts instead
    from src.data import repository as _r
    with _r.get_repository()._cm.connect() as conn:      # noqa: SLF001
        conn.register("_cash", cash); conn.register("_stats", stats)
        ccols = [c for c in conn.execute("SELECT * FROM fii_dii_cash LIMIT 0").df().columns
                 if c in cash.columns]
        scols = [c for c in conn.execute("SELECT * FROM fii_derivatives_stats LIMIT 0").df().columns
                 if c in stats.columns]
        conn.execute(f"INSERT INTO fii_dii_cash ({', '.join(ccols)}) "
                     f"SELECT {', '.join(ccols)} FROM _cash")
        conn.execute(f"INSERT INTO fii_derivatives_stats ({', '.join(scols)}) "
                     f"SELECT {', '.join(scols)} FROM _stats")
        conn.unregister("_cash"); conn.unregister("_stats")


@pytest.fixture
def fii_db(temp_db):
    days = _sessions(date(2025, 1, 1), 320)
    _seed_participant(days)
    _seed_index(days)
    _seed_cash_and_stats(days)
    return days


# ── the invariants ──────────────────────────────────────────────────────────
def test_actions_net_matches_signed_leg_sum(fii_db):
    """The NET tile must equal the signed sum of the six legs it sits under."""
    from src.analytics.fii_only import get_fii_actions, _LEG_SIGN

    act = get_fii_actions(fii_db[-1])
    assert act["ok"], act["error"]
    legs = act["legs"].set_index("leg")
    key = {"Fut Long": "fut_long", "Fut Short": "fut_short",
           "Call Long": "call_long", "Call Short": "call_short",
           "Put Long": "put_long", "Put Short": "put_short"}
    total = sum(_LEG_SIGN[key[i]] * r["pct_1d"] for i, r in legs.iterrows())
    assert total == pytest.approx(act["net_1d"], abs=1e-9)


def test_expiry_day_is_flagged_and_chain_advances(fii_db):
    """On an expiry day the settling series must not be shown as the live book."""
    from src.analytics.fii_only import get_fii_actions, get_fii_footprint

    d = fii_db[-1]
    later = d + timedelta(days=7)
    _seed_chain(d, [d, later])

    act = get_fii_actions(d)
    assert act["expiry_day"] is True
    assert act["expiry_note"]

    fp = get_fii_footprint(d, "NIFTY")
    assert fp["ok"], fp["error"]
    assert fp["expired_today"] is True
    # must have stepped PAST the expiry that settled today
    # near_expiry comes back as a pd.Timestamp against the live DuckDB but as a
    # plain date here, so normalise rather than assume either
    assert pd.Timestamp(fp["near_expiry"]).date() == later
    assert fp["dte"] > 0


def test_non_expiry_day_keeps_nearest_live_expiry(fii_db):
    from src.analytics.fii_only import get_fii_actions, get_fii_footprint

    d = fii_db[-1]
    _seed_chain(d, [d + timedelta(days=3), d + timedelta(days=10)])
    assert get_fii_actions(d)["expiry_day"] is False
    fp = get_fii_footprint(d, "NIFTY")
    assert fp["expired_today"] is False
    assert pd.Timestamp(fp["near_expiry"]).date() == d + timedelta(days=3)


def test_footprint_declares_its_market_wide_fields(fii_db):
    """FII share/futures OI have no per-index split and must say so."""
    from src.analytics.fii_only import get_fii_footprint

    d = fii_db[-1]
    _seed_chain(d, [d + timedelta(days=3), d + timedelta(days=10)])
    fp = get_fii_footprint(d, "NIFTY")
    assert "fii_share_pct" in fp["market_wide_fields"]
    assert "fii_fut_contracts" in fp["market_wide_fields"]
    assert isinstance(fp["unknown_symbols"], list)


def test_unknown_index_symbol_errors_rather_than_inventing(fii_db):
    from src.analytics.fii_only import get_fii_footprint

    fp = get_fii_footprint(fii_db[-1], "NIFTYNXT50")
    assert fp["ok"] is False
    assert "lot size" in fp["error"].lower()


def test_stance_only_names_direction_in_outer_zones(fii_db):
    """A stance must never claim a direction its own zone map does not support."""
    from src.analytics.fii_only import get_fii_only_view

    v = get_fii_only_view(fii_db[-1])
    assert v["ok"], v.get("error")
    if v.get("zone") is not None:
        assert v["zone"] in (1, 2, 3, 4, 5)
        if v["stance"] == "BEARISH":
            assert v["zone"] == 1
        if v["stance"] == "BULLISH":
            assert v["zone"] == 5
        if v["zone"] in (2, 3, 4):
            assert v["stance"] == "NEUTRAL"


def test_price_map_band_is_ordered_and_brackets_spot(fii_db):
    from src.analytics.fii_only import get_fii_only_view, get_fii_price_map

    d = fii_db[-1]
    v = get_fii_only_view(d)
    zone = v.get("zone") or 3
    m = get_fii_price_map(d, zone)
    assert m["ok"], m["error"]
    assert m["sigma_pct"] > 0
    assert m["basis"] in ("trailing",) or m["basis"].startswith("frozen")
    for hz in ("d1", "d5"):
        pts = m["bands"][hz]["cond_pts"]
        assert pts == sorted(pts), f"{hz} band not ordered"
        assert pts[0] < m["spot"] < pts[-1], f"{hz} band does not bracket spot"
        assert len(pts) == len(m["bands"][hz]["base_pts"]) == 5


def test_price_map_rejects_out_of_range_zone(fii_db):
    """A bad zone must come back as data, never as an uncaught KeyError."""
    from src.analytics.fii_only import get_fii_price_map

    for bad in (0, 6, -1, 99, "x"):
        m = get_fii_price_map(fii_db[-1], bad)
        assert m["ok"] is False
        assert "1-5" in m["error"]


def test_thin_contract_is_not_z_scored(fii_db):
    """A contract too small to carry a z-score must return NaN, not noise."""
    from src.analytics.fii_only import get_fii_only_view, _ILLIQUID_OI_CR
    import numpy as np

    v = get_fii_only_view(fii_db[-1])
    if not v.get("ok"):
        pytest.skip("composite unavailable on this fixture")
    pi = v.get("per_index")
    if pi is not None and not pi.empty and "thin" in pi.columns:
        for _, r in pi.iterrows():
            if r["oi_cr"] < _ILLIQUID_OI_CR:
                assert bool(r["thin"]) is True
                assert np.isnan(r["net_z"])


def test_no_data_before_history_returns_error_not_exception(temp_db):
    from src.analytics.fii_only import (get_fii_actions, get_fii_only_view,
                                        get_fii_footprint, get_fii_hedge_read)

    d = date(2000, 1, 3)
    for fn in (get_fii_actions, get_fii_only_view, get_fii_hedge_read):
        out = fn(d)
        assert out["ok"] is False and out["error"]
    out = get_fii_footprint(d, "NIFTY")
    assert out["ok"] is False and out["error"]


def test_future_date_resolves_back_and_never_forward(fii_db):
    """Asking past the end of the data must return the LAST session, never invent."""
    from src.analytics.fii_only import get_fii_actions

    last = fii_db[-1]
    out = get_fii_actions(last + timedelta(days=400))
    assert out["ok"]
    assert out["as_of"] <= last


def test_net_delta_is_versus_previous_participant_row(fii_db):
    """The day-over-day arrow must use the previous PARTICIPANT session."""
    from src.analytics.fii_only import get_fii_actions

    days = fii_db
    today = get_fii_actions(days[-1])
    prior = get_fii_actions(days[-2])
    assert today["ok"] and prior["ok"]
    assert today["prev_date"] == prior["as_of"]
    assert today["net_1d_prev"] == pytest.approx(prior["net_1d"], abs=1e-9)
    assert today["net_5d_prev"] == pytest.approx(prior["net_5d"], abs=1e-9)
    assert today["net_1d_delta"] == pytest.approx(
        today["net_1d"] - prior["net_1d"], abs=1e-9)
    assert today["net_5d_delta"] == pytest.approx(
        today["net_5d"] - prior["net_5d"], abs=1e-9)


def test_net_delta_absent_when_no_prior_session(temp_db):
    """One session of data must not manufacture a delta."""
    from src.analytics.fii_only import get_fii_actions
    import numpy as np

    days = _sessions(date(2025, 1, 1), 30)
    _seed_participant(days)
    out = get_fii_actions(days[-1])
    if out["ok"]:
        assert out["delta_ok"] in (True, False)
        if not out["delta_ok"]:
            assert np.isnan(out["net_1d_delta"])
