"""
FII-ONLY market read — the verdict built from FII data and nothing else.

WHAT THIS USES
--------------
  cash_net     FII net cash-market buying          (fii_dii_cash, 2024-06+)
  fut_share    FII index-futures long share of OI  (fao_participant, 2018+)
  fut_net_chg  day change in FII net futures OI    (fao_participant)
  opt_bull     calls held long + puts written
               minus calls written + puts held     (fao_participant)
  stats_net    index-futures buy minus sell value  (fii_derivatives_stats)
  per-index    NIFTY / BANKNIFTY / FINNIFTY / MIDCPNIFTY futures buy-sell value
               and OI (fii_derivatives_stats, 2023-01-31+)
Volume is carried as a CONVICTION weight, not a direction.

MEASURED TRACK RECORD — read this before using the verdict
----------------------------------------------------------
Walk-forward, 1,592 sessions (2018-03 -> 2026-08), composite threshold ±0.5σ:

  NEXT DAY, close->close  base up 53.7% | verdict hit 56.7% | IC +0.106
  NEXT DAY, open->close   base up 47.5% | verdict hit 53.2% | IC +0.090
       bullish -> up 48.5%, mean +0.013%
       bearish -> down 57.1%, mean -0.155%
  1 WEEK    verdict hit 52.7%  vs 56.5% for always saying "up"
  2 WEEKS   verdict hit 54.1%  vs 58.5% for always saying "up"

THREE THINGS THAT FOLLOW, AND THEY MATTER MORE THAN THE VERDICT:

1. MOST OF THE APPARENT EDGE IS THE OVERNIGHT GAP, AND YOU CANNOT TRADE IT.
   The base up-rate is 53.7% close-to-close but 47.5% open-to-close — a 6-point
   gap that IS the overnight move. NSE publishes the participant file AFTER the
   close, so entering at close-t is impossible; you enter at open t+1, by which
   time the gap has happened. Any close-to-close number here flatters itself.

2. WHAT SURVIVES IS THE BEARISH SIDE, AND IT IS BELOW COST.
   Tradeable open->close: a bearish reading precedes a down day 57.1% of the time
   (naive 52.5%) with a mean of -0.155%. IC +0.090 on n=1,592 is t~3.6, so this is
   statistically real, not noise. But -15.5bps against a ~50bps round trip loses
   money. Real and unprofitable are not contradictory.

3. IT IS WORSE THAN USELESS WEEKLY. At 1-2 weeks the verdict scores BELOW simply
   always saying "up" (52.7% vs 56.5%, 54.1% vs 58.5%). Do not extend it.

So this module returns a STANCE with its own measured hit-rate attached, never a
bare arrow. Consistent with the wider finding that FII positioning does not
forecast the index (see project_market_next_month_fii: 24/78/48-candidate searches
each returned a best result below their own noise floor).
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_fii_only_view", "get_fii_actions", "get_fii_footprint", "get_fii_price_map",
           "FII_TRACK_RECORD", "FII_ACTION_TRACK", "FII_ERA_SPLIT", "FII_ZONES", "FII_STRIKE_HOLD", "FII_LEG_IC", "FII_LEG_BACKTEST_2Y", "FII_HEDGE_TEST", "FII_FAMILY_SWEEP_2Y", "get_fii_hedge_read",
           "INDEX_CATEGORY", "LEG_LABELS"]

# ── What FIIs DID: the six legs NSE publishes, read directly ────────────────
# Nothing here is inferred. fao_participant carries FII long AND short OI for
# index futures, and long AND short for index calls and puts, so "added longs"
# vs "covered shorts" is READ rather than guessed from a price move (contrast the
# operator-footprint tab, where buy-vs-write had to be inferred from premium
# direction and was ~83-94% confounded with the day's move).
#   leg -> (label when RISING, label when FALLING, lean when rising)
LEG_LABELS = {
    "fut_long":   ("LONG BUILDUP", "LONG UNWINDING", "bullish"),
    "fut_short":  ("SHORT BUILDUP", "SHORT COVERING", "bearish"),
    "call_long":  ("CALL BUYING", "CALL LONGS EXITING", "bullish"),
    "call_short": ("CALL WRITING", "CALL WRITERS COVERING", "bearish"),
    "put_long":   ("PUT BUYING", "PUT LONGS EXITING", "bearish"),
    "put_short":  ("PUT WRITING", "PUT WRITERS COVERING", "bullish"),
}
_LEG_COL = {
    "fut_long": "fut_idx_long", "fut_short": "fut_idx_short",
    "call_long": "opt_idx_call_long", "call_short": "opt_idx_call_short",
    "put_long": "opt_idx_put_long", "put_short": "opt_idx_put_short",
}
_LEG_SIGN = {"fut_long": +1, "fut_short": -1, "call_long": +1,
             "call_short": -1, "put_long": -1, "put_short": +1}

# Measured 2026-08-13, walk-forward on 2,104 sessions (2018-2026). This is a
# DIFFERENT signal from the z-score composite above and gets its own numbers.
FII_ACTION_TRACK = {
    "n": 2104,
    "d1_oc_hit": 50.8, "d1_oc_naive": 52.4, "d1_oc_ic": 0.035,
    "wk1_hit": 51.2, "wk1_naive": 56.4,
    "wk2_hit": 50.5, "wk2_naive": 58.6,
    # the counter-intuitive part, and the reason the labels carry a warning
    "bear_down_1wk": 44.2, "bear_down_2wk": 41.6,
    "bull_up_2wk": 63.2, "base_up_2wk": 58.6,
    "leg_ic_range": "-0.055 to +0.009",
}

# measured 2026-08-13 — see the module docstring for how
FII_TRACK_RECORD = {
    "n": 1592, "since": "2018-03-28",
    "next_day_cc_hit": 56.7, "next_day_cc_base": 53.7, "next_day_cc_ic": 0.106,
    "next_day_oc_hit": 53.2, "next_day_oc_base": 47.5, "next_day_oc_ic": 0.090,
    "next_day_naive": 52.5,
    "bear_down_rate": 57.1, "bear_mean_pct": -0.155,
    "bull_up_rate": 48.5, "bull_mean_pct": 0.013,
    "wk1_hit": 52.7, "wk1_naive": 56.5,
    "wk2_hit": 54.1, "wk2_naive": 58.5,
    "cost_floor_bps": 50,
}

# ── COMPOSITION DRIFT, AND WHY THE HEADLINE TRACK RECORD IS SPLIT ──────────
# The composite is a plain mean of whichever components are live, and the number
# of live components CHANGES MID-HISTORY: `fii_dii_cash` only starts 2024-06, so
# 1,624 of 2,180 sessions ran on four components and 556 on five. A mean of k
# correlated z-scores does not have a stable spread, so the fixed +-0.5 gate is
# not a fixed event rate:
#     4 components -> std(score) 0.794, fires on 51.7% of days
#     5 components -> std(score) 0.643, fires on 43.5% of days
# Worse, the two eras do not behave alike. Measured next-day open->close:
#     4-component era (to 2024-06, n=1,542): directional hit 55.0%,
#         bearish readings fell 58.3% of the time, mean -0.194%
#     5-component era (2024-06 on,  n=  555): directional hit 42.1%,
#         bearish readings fell 49.6% of the time, mean -0.030%
# The blended 53.2% that this module used to advertise is therefore mostly the
# FIRST era's number, earned by a FOUR-component signal — not by the five-component
# stance the page actually shows today. In the live composition the stance is
# BELOW a coin flip. Both numbers are published; the live one governs.
FII_ERA_SPLIT = {
    "era4": {"label": "4 components (to Jun-2024)", "n": 1542, "fires_pct": 51,
             "dir_hit": 55.0, "bear_down": 58.3, "bear_mean": -0.194},
    "era5": {"label": "5 components (Jun-2024 on)", "n": 555, "fires_pct": 44,
             "dir_hit": 42.1, "bear_down": 49.6, "bear_mean": -0.030},
}

# Zone map for the FII composite, measured 2026-08-18 on the composition-stable
# score (the raw mean re-standardised on its own trailing 250 sessions, so a
# quintile means the same thing in both eras). Next-day is open->close, so it is
# the tradeable leg; the 5-day enters at the next open and exits 5 closes later.
# NOTHING HERE CLEARS MULTIPLICITY: 10 cells were measured, so Bonferroni needs
# |t| > 2.81 and the best cells are t = -2.09 (Q1, 5-day) and +2.44 (Q5, 5-day).
# Treat the gradient as descriptive.
_ZONE_CUTS = [-0.8042, -0.2626, 0.2194, 0.7934]          # quintile edges of the score
FII_ZONES = {
    "cuts": _ZONE_CUTS,
    "d1_base": {"mean": -0.0643, "up": 47.4},
    "d5_base": {"mean": +0.1067, "up": 52.9},
    "d1": {1: {"n": 396, "mean": -0.146, "up": 43.7, "t": -1.77},
           2: {"n": 396, "mean": -0.052, "up": 47.2, "t": +0.26},
           3: {"n": 394, "mean": -0.063, "up": 48.0, "t": +0.03},
           4: {"n": 397, "mean": -0.050, "up": 51.4, "t": +0.34},
           5: {"n": 396, "mean": -0.011, "up": 46.7, "t": +1.22}},
    "d5": {1: {"n": 396, "mean": -0.162, "up": 47.0, "t": -2.09},
           2: {"n": 396, "mean": +0.053, "up": 52.3, "t": -0.47},
           3: {"n": 394, "mean": +0.168, "up": 56.3, "t": +0.54},
           4: {"n": 397, "mean": +0.114, "up": 52.9, "t": +0.06},
           5: {"n": 396, "mean": +0.361, "up": 56.1, "t": +2.44}},
}
_ZONE_LABEL = {1: "most bearish", 2: "lean bearish", 3: "middle",
               4: "lean bullish", 5: "most bullish"}

# FII derivative-stats categories that are too small to z-score. FINNIFTY is a
# husk since NSE withdrew its weekly expiries: over the last 60 sessions its FII
# net futures value is under Rs 1 cr on 27% of days with a median of Rs 1.71 cr,
# against Rs 858 cr for NIFTY, and its open interest is Rs 58 cr against Rs 25,351
# cr. A z-score on that is a z-score on rounding noise -- on 2026-08-18 it printed
# +0.63 sigma off a net of Rs 0.47 cr. Flagged, not silently scored.
_ILLIQUID_OI_CR = 500.0

# the page's four indices -> their FII derivative-stats category (2023-01-31+)
INDEX_CATEGORY = {
    "Nifty 50": "NIFTY FUTURES",
    "Bank Nifty": "BANKNIFTY FUTURES",
    "Fin Nifty": "FINNIFTY FUTURES",
    "Midcap Nifty": "MIDCPNIFTY FUTURES",
}

_Z_WIN = 60
_THR = 0.5          # |composite| above this becomes a directional stance


def _z(s: pd.Series, win: int = _Z_WIN) -> pd.Series:
    """Rolling z-score that tolerates gaps.

    `rolling(win)` defaults to min_periods=win, so a SINGLE missing session in the
    window returns NaN. The components here are joined onto the participant-date
    index and the cash series is missing 3 of the last 60 of those — which made
    `cash_net` permanently NaN and silently dropped FII cash flow from the view
    entirely. min_periods lets a partly-populated window still produce a value.
    """
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    mp = max(20, win // 2)
    return ((s - s.rolling(win, min_periods=mp).mean())
            / s.rolling(win, min_periods=mp).std())


# One page render calls _components twice — once for get_fii_only_view and again
# inside _zone_series for the price map — and each call rebuilds the whole panel
# from four tables. Memoise on the as_of date; the underlying data for a past
# date never changes, and today's is refreshed by the nightly load, so a tiny
# cache keyed by date is safe. Bounded so a long replay session cannot grow it.
@lru_cache(maxsize=64)
def _components_cached(as_of: date) -> pd.DataFrame:
    return _components_uncached(as_of)


def _components(as_of: date) -> pd.DataFrame:
    return _components_cached(as_of).copy()


def _components_uncached(as_of: date) -> pd.DataFrame:
    """Market-wide FII components, z-scored. Sign convention: + = bullish."""
    p = query_dataframe(
        """SELECT * FROM fao_participant
           WHERE client_type='FII' AND data_type='OI' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if p.empty:
        return pd.DataFrame()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.drop_duplicates("trade_date").set_index("trade_date")

    F = pd.DataFrame(index=p.index)
    tot = p["fut_idx_long"] + p["fut_idx_short"]
    F["fut_share"] = np.where(tot > 0, p["fut_idx_long"] / tot * 100, np.nan)
    F["fut_net_chg"] = (p["fut_idx_long"] - p["fut_idx_short"]).diff()
    F["opt_bull"] = (p["opt_idx_call_long"] + p["opt_idx_put_short"]
                     - p["opt_idx_call_short"] - p["opt_idx_put_long"])

    c = query_dataframe(
        """SELECT trade_date, fii_net FROM fii_dii_cash
           WHERE trade_date <= ? ORDER BY trade_date""", [as_of])
    if not c.empty:
        c["trade_date"] = pd.to_datetime(c["trade_date"])
        F["cash_net"] = c.drop_duplicates("trade_date").set_index("trade_date")["fii_net"]

    ds = query_dataframe(
        """SELECT trade_date, buy_value_cr, sell_value_cr FROM fii_derivatives_stats
           WHERE category='INDEX FUTURES' AND trade_date <= ? ORDER BY trade_date""",
        [as_of])
    if not ds.empty:
        ds["trade_date"] = pd.to_datetime(ds["trade_date"])
        ds = ds.drop_duplicates("trade_date").set_index("trade_date")
        F["stats_net"] = ds["buy_value_cr"] - ds["sell_value_cr"]

    v = query_dataframe(
        """SELECT * FROM fao_participant
           WHERE client_type='FII' AND data_type='Vol' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if not v.empty:
        v["trade_date"] = pd.to_datetime(v["trade_date"])
        v = v.drop_duplicates("trade_date").set_index("trade_date")
        F["vol_intensity"] = v["total_long"] + v["total_short"]

    return pd.DataFrame({c_: _z(F[c_]) for c_ in F.columns}, index=F.index)


# Index lot sizes, DERIVED from fii_derivatives_stats (oi_value_cr / oi_contracts
# / spot) rather than assumed — verified 2026-08-12 to the exact integer.
_INDEX_LOTS = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120}


def get_fii_footprint(as_of: date, symbol: str = "NIFTY") -> dict:
    """
    How big FIIs actually are in the index book, plus where the option walls sit
    and whether positions are rolling to the next expiry.

    THE UNIT TRAP THIS FUNCTION EXISTS TO AVOID. `fao_participant` reports open
    interest in CONTRACTS; `fno_bhavcopy` reports it in UNITS (shares). Comparing
    them directly says FIIs hold 0.5% of index option OI — off by the lot factor
    and completely wrong. Verified against `fii_derivatives_stats`, which
    publishes both: FII option OI 2,376,692 in participant data vs oi_contracts
    2,376,691 — an exact match, so participant data is contracts. Converted
    properly, FIIs hold about **30%** of index option open interest.

    WHAT CANNOT BE BUILT, AND WHY. NSE does NOT publish FII positions BY STRIKE.
    The option chain gives total OI per strike across all participants; the
    participant file gives FII totals with no strike breakdown. So "the strike
    FIIs are defending" is not derivable from any available data. The walls below
    are MARKET-WIDE, with the FII share of the book shown as context — labelling
    them as FII levels would be fabrication.
    """
    out: dict = {"ok": False, "as_of": as_of, "error": ""}
    lot = _INDEX_LOTS.get(symbol.upper())
    if lot is None:
        out["error"] = f"No known lot size for {symbol}."
        return out

    dd = query_dataframe(
        """SELECT MAX(trade_date) AS d FROM fno_bhavcopy
           WHERE instrument='OPTIDX' AND trade_date <= ?""", [as_of])
    if dd.empty or dd["d"].iloc[0] is None:
        out["error"] = "No index option data on or before that date."
        return out
    d0 = pd.Timestamp(dd["d"].iloc[0]).date()

    chain = query_dataframe(
        """SELECT expiry_date, option_type, strike_price, open_interest, chg_in_oi
           FROM fno_bhavcopy
           WHERE instrument='OPTIDX' AND symbol = ? AND trade_date = ?
             AND open_interest > 0""", [symbol.upper(), d0])
    if chain.empty:
        out["error"] = f"No {symbol} option chain for {d0}."
        return out
    chain["expiry_date"] = pd.to_datetime(chain["expiry_date"])

    # ON EXPIRY DAY, THE NEAREST EXPIRY IS THE ONE THAT JUST DIED. Taking
    # `.min()` unconditionally showed the settling series as "where the walls
    # are" — on 2026-08-18 that put NIFTY walls at 24,200/24,150 for a chain
    # with zero remaining life, while the book that actually governs the next
    # session (25-Aug) sat at 25,000/24,500 CE and 24,000/23,700 PE. Step past
    # any expiry that is on or before the data date whenever a later one exists.
    expiries = sorted(chain["expiry_date"].unique())
    live = [e for e in expiries if pd.Timestamp(e).date() > d0]
    near = pd.Timestamp(live[0]) if live else pd.Timestamp(expiries[0])
    nr = chain[chain["expiry_date"] == near]
    out["expired_today"] = bool(pd.Timestamp(expiries[0]).date() <= d0)

    def _walls(df, opt, n=3):
        g = (df[df["option_type"] == opt]
             .groupby("strike_price")[["open_interest", "chg_in_oi"]].sum()
             .sort_values("open_interest", ascending=False).head(n))
        return [{"strike": float(k), "oi": float(v["open_interest"]),
                 "chg": float(v["chg_in_oi"])} for k, v in g.iterrows()]

    # FII share of the book, in comparable units
    fii = query_dataframe(
        """SELECT opt_idx_call_long + opt_idx_call_short
                + opt_idx_put_long + opt_idx_put_short AS opt_oi,
                  fut_idx_long + fut_idx_short AS fut_oi
           FROM fao_participant
           WHERE client_type='FII' AND data_type='OI' AND trade_date <= ?
           ORDER BY trade_date DESC LIMIT 1""", [as_of])
    tot_opt = query_dataframe(
        """SELECT symbol, SUM(open_interest) AS oi FROM fno_bhavcopy
           WHERE instrument='OPTIDX' AND trade_date = ? GROUP BY symbol""", [d0])
    # Any index option NSE lists but this map does not know is silently dropped
    # from the denominator, which INFLATES the FII share. Today that is NIFTYFPI
    # and NIFTYNXT50 at 0.03% of the book, immaterial — but the failure is silent
    # and grows with any new listing, so report it instead of swallowing it.
    unknown = tot_opt[~tot_opt["symbol"].isin(_INDEX_LOTS)]
    opt_contracts = float(sum(
        r["oi"] / _INDEX_LOTS[r["symbol"]] for _, r in tot_opt.iterrows()
        if r["symbol"] in _INDEX_LOTS))
    out["unknown_symbols"] = list(unknown["symbol"])
    out["unknown_oi_share_pct"] = (float(unknown["oi"].sum() / tot_opt["oi"].sum() * 100)
                                   if tot_opt["oi"].sum() else 0.0)
    fii_opt = float(fii["opt_oi"].iloc[0]) if not fii.empty else np.nan
    fii_share = (fii_opt / opt_contracts * 100) if opt_contracts else np.nan

    # rollover: how much OI has moved to later expiries
    fut = query_dataframe(
        """SELECT expiry_date, SUM(open_interest) AS oi FROM fno_bhavcopy
           WHERE instrument='FUTIDX' AND symbol = ? AND trade_date = ?
           GROUP BY expiry_date ORDER BY expiry_date""", [symbol.upper(), d0])
    roll = np.nan
    if len(fut) >= 2:
        fut["oi"] = fut["oi"].astype(float)
        roll = float(fut["oi"].iloc[1:].sum() / fut["oi"].sum() * 100)

    out.update({
        "ok": True, "as_of": d0, "symbol": symbol.upper(), "lot": lot,
        "near_expiry": near.date(),
        "dte": int((near.date() - d0).days),
        "call_walls": _walls(nr, "CE"), "put_walls": _walls(nr, "PE"),
        # THESE THREE ARE MARKET-WIDE, NOT PER-SYMBOL. The participant file has
        # no per-index split, so `fii_share_pct` and `fii_fut_contracts` are
        # IDENTICAL for every value of `symbol` (verified: 19.14% / 245,162 for
        # NIFTY, BANKNIFTY and MIDCPNIFTY alike on 2026-08-18). Only the walls,
        # the expiry and `rollover_pct` actually vary with the symbol. Rendering
        # them under a per-index selector without saying so reads as "BankNifty's
        # FII share", which is not a thing that exists.
        "fii_opt_contracts": fii_opt,
        "total_opt_contracts": opt_contracts,
        "fii_share_pct": fii_share,
        "market_wide_fields": ["fii_share_pct", "fii_opt_contracts",
                               "fii_fut_contracts"],
        "fii_fut_contracts": float(fii["fut_oi"].iloc[0]) if not fii.empty else np.nan,
        "rollover_pct": roll,
    })
    return out


# ── WHERE THE INDEX CAN GO, GIVEN FII POSITIONING ──────────────────────────
# FII data cannot produce a price band on its own: it carries no volatility
# information (FII open-interest churn scores IC +0.07 against next-day range,
# but a plain 5-day average range scores +0.49 and absorbs it). What it CAN do
# is SHIFT a band that volatility has already sized. So the construction is:
# sigma sets the scale, the FII zone sets the shape.
#
# Measured 2026-08-18 over 1,979 sessions. Forward close-to-close moves were
# divided by a causal EWMA sigma (lambda 0.94, shifted one day) and the
# quantiles taken per zone, so every row below is in units of TODAY's sigma.
#
# THE RESULT IS NOT A DIRECTION SIGNAL -- it is a DOWNSIDE-ROOM signal, and it
# is the only thing in this module that survives a permutation test:
#   Q1 p10 (next day)  -0.300 sigma vs unconditional   perm p = 0.0030
#   Q1 width minus Q4 width (next day)  +0.765 sigma    perm p < 0.0001
#   Q5 p10 (5 sessions) +0.946 sigma vs unconditional   perm p < 0.0001
#   Q1 width minus Q5 width (5 sessions) +1.593 sigma   perm p < 0.0001
# all four clearing Bonferroni over the six statistics tested. Critically the
# starting volatility is FLAT across zones (median sigma_t 0.814 to 0.829 on a
# 0.820 median), so this is not volatility clustering leaking through the
# normalisation -- identical vol, different forward tails.
#
# WHAT DOES NOT HOLD: the monotone trend across all five zones is not
# significant (perm p ~ 0.08), so read Q1 and Q5, not the ladder. Out of sample
# (2024-06 on, n=542) the Q1 fat left tail persists at both horizons (-0.24 and
# -0.38 sigma) but the Q5 tail-narrowing decays badly at five days (+1.03 -> +0.22)
# and the middle zones scramble. Q1 is the durable half of this finding.
_FII_Q = [0.10, 0.25, 0.50, 0.75, 0.90]
_FII_QUANTILES = {
    "d1": {0: (-1.26, -0.55, +0.08, +0.71, +1.26),      # 0 = unconditional
           1: (-1.56, -0.77, -0.10, +0.78, +1.34),
           2: (-1.38, -0.73, -0.00, +0.65, +1.31),
           3: (-1.22, -0.47, +0.16, +0.70, +1.13),
           4: (-0.99, -0.45, +0.16, +0.72, +1.14),
           5: (-1.00, -0.39, +0.13, +0.74, +1.30)},
    "d5": {0: (-2.91, -1.18, +0.35, +1.80, +3.00),
           1: (-3.37, -1.75, +0.01, +1.60, +3.25),
           2: (-2.87, -1.29, +0.27, +1.83, +2.89),
           3: (-2.94, -1.12, +0.57, +2.06, +3.02),
           4: (-2.71, -0.99, +0.35, +1.72, +2.89),
           5: (-1.97, -0.90, +0.42, +1.73, +3.06)},
}
# P(forward move below the stated sigma multiple), by zone. Index 0 = base rate.
# The 5-day row is monotone across all five zones -- 22.5% down to 9.8%.
_FII_TAIL = {
    "d1": {"thr": -1.0, 0: 13.8, 1: 18.9, 2: 17.4, 3: 13.5, 4: 9.3, 5: 10.1},
    "d5": {"thr": -2.0, 0: 15.8, 1: 22.5, 2: 17.2, 3: 16.2, 4: 13.1, 5: 9.8},
}


# ── DOES THE STRIKE WITH THE BIGGEST FRESH OI ADD ACTUALLY HOLD? ───────────
# The recurring intuition is that the strike where the most new open interest
# piles up is where FIIs have "placed" support or resistance, so price should
# stop there. Measured on 513 sessions of NIFTY nearest-LIVE-expiry chains:
#
#   biggest fresh PUT add, sitting BELOW spot (the claimed support)
#       price reached it the next session on 22.3% of days (102 of 457)
#       and closed back ABOVE it on only 42.2% of those — it FAILS more
#       often than it holds
#   biggest fresh CALL add, sitting ABOVE spot (the claimed resistance)
#       reached on 19.8% of days (96 of 486), capped 50.0% — a coin flip
#
# Distance-matched control (the only bucket with usable n, strikes within
# 0.5% of spot): puts held 41.4% after a BIG add versus 40.0% after a small
# one, n=29 and 55. Calls 56.5% versus 42.6%, but on n=23 the standard error
# is about 10pp. So size of the OI add buys nothing beyond where the strike
# already sits relative to spot. Consistent with the separate finding that
# max pain does not pin and that EOD walls do not bound the next day.
FII_STRIKE_HOLD = {
    "n_sessions": 513,
    "put": {"n": 457, "reached_pct": 22.3, "held_pct": 42.2,
            "ctrl_big": 41.4, "ctrl_small": 40.0},
    "call": {"n": 486, "reached_pct": 19.8, "held_pct": 50.0,
             "ctrl_big": 56.5, "ctrl_small": 42.6},
}

# Measured worth of each of the six legs on its own, next-day open->close
# (tradeable). Signed so a positive IC means the leg's printed lean is the
# right direction. Only futures-short is even marginally separated, and six
# tests need |t| > 2.64 to clear Bonferroni, so nothing here survives.
FII_LEG_IC = {
    "fut_long": +0.0089, "fut_short": +0.0557, "call_long": +0.0001,
    "call_short": +0.0324, "put_long": +0.0171, "put_short": +0.0040,
    "_t": {"fut_long": 0.41, "fut_short": 2.56, "call_long": 0.00,
           "call_short": 1.49, "put_long": 0.78, "put_short": 0.18},
    "net_ic": +0.0353, "net_t": 1.62, "n": 2110,
    # the net leg score and the composite stance are only moderately related
    # and point OPPOSITE ways on 13.3% of days, so the page must not show them
    # as if they were two confirmations of one view
    "vs_composite_spearman": +0.599, "disagree_pct": 13.3,
}


# ── WHAT THE SIX-LEG VERDICT WAS WORTH, LAST TWO YEARS ─────────────────────
# Backtest 2024-08-21 .. 2026-08-20, 496 sessions, reconstructing net_1d exactly
# as get_fii_actions builds it (each leg's change as a % of its own 60-session
# average OI, signed, summed) and classifying at the shipped +-5 threshold.
# Forward leg is next-day OPEN->CLOSE, because the participant file publishes
# after the close, so the earliest reachable entry is the next open.
#
# THE VERDICT HAS BEEN INVERTED OVER THIS WINDOW:
#   says BULLISH (n=208): next day mean -0.085%, up 41.8% [35.3, 48.6]
#   says BEARISH (n=193): next day mean +0.047%, up 52.8% [45.8, 59.8]
#   bull minus bear -0.133%, Welch t = -2.05
#   directional hit 44.4% against a 53.6% naive (in this window the naive call
#   is DOWN, not up -- open-to-close base drift is -0.032%, up only 46.4%)
#
# WHERE THE INVERSION COMES FROM: expiry days. NIFTY has weekly expiries, so
# 108 of the 496 sessions (21.8%) carry leg deltas that are settlement rather
# than positioning. Split:
#   expiry days only (n=108): bull minus bear -0.232%, t=-1.74, hit 37.8%
#   excluding expiry (n=388): bull minus bear -0.095%, t=-1.28, hit 46.3%
# So the sharpest part of the inversion is a data artefact this module already
# flags. What survives ex-expiry is weaker and not significant, and it is not a
# tradeable inversion either: bullish readings run mildly negative (-0.090%)
# while bearish readings are flat (+0.005%).
#
# STABLE ACROSS THRESHOLDS (2yr, ex-expiry, next day) -- so it is not a
# threshold artefact, it is just weak:
#   +-3  hit 46.9% | +-5  hit 46.3% | +-10 hit 46.7% | +-15 hit 45.4%
#
# NOT PRESENT IN FULL HISTORY: over all 2,110 sessions bull minus bear is
# +0.035% (t=+0.80) -- correct sign, no strength. The last two years are the
# anomaly, and with n=193/208 per arm that is exactly the sample size where a
# regime read and a fluke look identical. Do not trade the inversion either.
FII_LEG_BACKTEST_2Y = {
    "start": "2024-08-21", "end": "2026-08-20", "n": 496,
    "expiry_n": 108, "expiry_pct": 21.8,
    "base_mean": -0.032, "base_up": 46.4, "naive": 53.6, "naive_side": "down",
    "bull": {"n": 208, "mean": -0.085, "up": 41.8, "ci": (35.3, 48.6)},
    "bear": {"n": 193, "mean": +0.047, "up": 52.8, "ci": (45.8, 59.8)},
    "diff": -0.133, "welch_t": -2.05, "hit": 44.4,
    "exq": {"n": 388, "diff": -0.095, "t": -1.28, "hit": 46.3,
            "bull_mean": -0.090, "bear_mean": +0.005},
    "expiry_only": {"n": 108, "diff": -0.232, "t": -1.74, "hit": 37.8},
    "thresholds": {3: 46.9, 5: 46.3, 10: 46.7, 15: 45.4},
    "full_hist": {"n": 2110, "diff": +0.035, "t": +0.80, "hit": 51.3, "naive": 52.4},
    # the WEEK view: enter next open, hold five sessions
    "week": {"bull_mean": -0.109, "bull_up": 45.6, "bear_mean": +0.042,
             "bear_up": 45.3, "t": -0.81, "hit": 50.0, "naive": 54.1,
             "base_mean": -0.042, "base_up": 45.9},
    # the 5-session leg score (net_5d) as the signal, 2yr ex-expiry -- this one
    # at least has the RIGHT sign, though the intervals overlap heavily
    "net5": {"bull_n": 143, "bull_mean": +0.012, "bull_up": 50.3,
             "bear_n": 158, "bear_mean": -0.045, "bear_up": 41.8,
             "base_mean": -0.058, "base_up": 46.0},
}


# ── SELF-AUDIT, 2026-08-20: WHY THE TABLES ABOVE ARE NOW ONLY A FALLBACK ───
# Everything hardcoded in _ZONE_CUTS and _FII_QUANTILES was measured on the FULL
# sample and then used to classify days inside that same sample. Two things were
# checked, and one of them failed.
#
# WHAT HELD. Re-running the zone assignment walk-forward -- at every date the
# cuts taken from strictly earlier data only, n=1,481 from 2020-08 -- reproduces
# the tail finding almost exactly:
#     P(next day < -1 sigma):  base 13.6%, Q1 18.8% walk-forward vs 18.3% static
#     P(5 sessions < -2 sigma): base 14.7%, Q1 20.1% walk-forward vs 20.0% static
#     permutation on the WALK-FORWARD labels: Q1 excess +5.20pp p=0.0030 (1d)
#                                             and +5.42pp p=0.0025 (5d)
# Walk-forward and static labels agree on 96.5% of days. The finding is real and
# not an artefact of in-sample cut selection.
#
# WHAT FAILED. The static cuts no longer cut the CURRENT distribution into
# fifths. Over the last two years they produce Q1 18.2%, Q2 25.5%, Q3 21.6%,
# Q4 18.8%, Q5 16.0% -- so "each zone holds 20% by construction" stopped being
# true, and a reading is likelier to be filed in Q2 than its label implies.
# Separately the sigma-normalised quantiles are NOT stationary: since 2024-08 the
# next-day p10 is -1.41 against the -1.26 baked in here, and the median is +0.00
# against +0.08. A frozen table therefore UNDERSTATES downside in the current
# regime -- about 22 points at p10 on today's sigma, small but always in the
# dangerous direction.
#
# So the cuts and the quantiles are now computed from TRAILING data at call time.
# The frozen tables remain as a documented fallback for short histories and as
# the reference the audit numbers above were measured against.
_DIR_COMPONENTS = ("cash_net", "fut_share", "fut_net_chg",
                   "opt_bull", "stats_net")
_MIN_HIST = 500          # sessions before a trailing estimate is trustworthy
_MIN_BUCKET = 60         # per-zone minimum before its own quantiles are used


def _zone_series(as_of: date) -> pd.DataFrame:
    """Composite score, and its zone under TRAILING-ONLY quintile cuts."""
    Z = _components(as_of)
    dirs = [c for c in _DIR_COMPONENTS if c in Z.columns]
    raw = Z[dirs].mean(axis=1).where(Z[dirs].notna().sum(axis=1) >= 3)
    m = raw.rolling(250, min_periods=120).mean()
    sd = raw.rolling(250, min_periods=120).std().replace(0.0, np.nan)
    norm = (raw - m) / sd
    out = pd.DataFrame({"norm": norm}).dropna()
    if out.empty:
        return out
    nv = out["norm"].to_numpy()
    zones = np.full(len(nv), np.nan)
    for i in range(len(nv)):
        if i < _MIN_HIST:
            continue
        c = np.nanquantile(nv[:i], [0.2, 0.4, 0.6, 0.8])   # strictly earlier data
        zones[i] = 1 + int(np.sum(nv[i] > c))
    out["zone"] = zones
    return out


# ── IS THE FUTURES SHORT A HEDGE OR A BET? CASH SAYS IT CANNOT TELL YOU ────
# The natural next move after "short buildup is not bearish because they hedge"
# is to check the CASH side: if FIIs are ALSO selling cash while building
# futures shorts, that should be a genuine bearish bet rather than a hedge, and
# the two together should be a stronger signal than either alone. It was tested
# over the fii_dii_cash overlap (2024-04-01 .. 2026-08-20, 587 sessions) and it
# does not work. Next-day open->close against a -0.035% base:
#     SHORT BUILDUP + FII SELLING cash  n=245  -0.030%  up 46.1%   t=-0.67
#     SHORT BUILDUP + FII BUYING cash   n= 92  -0.005%  up 44.6%   t=-0.08
#     short COVERING + FII buying       n= 93  -0.105%  up 39.8%   t=-1.32
#     short COVERING + FII selling      n=157  -0.017%  up 49.0%   t=-0.34
# Every state sits on the base rate. Excluding expiry days changes nothing.
#
# THE "DOUBLE CONFIRMATION" ADDS NOTHING MEASURABLE:
#     short buildup ALONE        n=337  -0.023%
#     + FII also selling cash    n=245  -0.030%   adds -0.007%, Welch t=-0.12
# and at five sessions it adds -0.036% (t=-0.24) -- i.e. the confirmation makes
# the read very slightly WORSE, which is what noise looks like.
#
# DELIVERY AND TURNOVER CANNOT STAND IN FOR FII CASH. Market-wide delivery ratio
# and turnover are barely related to FII cash flow at all:
#     spearman(FII cash net, delivery-ratio z) = +0.093
#     spearman(FII cash net, turnover z)       = +0.078
# Both are dominated by domestic, retail and proprietary flow, so "delivery is
# low, so FIIs are not buying" does not follow. Adding both as filters on top of
# buildup+cash-selling moved the next day from -0.030% to -0.002% (n=82).
#
# WHY IT ALL FAILS, MECHANICALLY: spearman(FII cash net, DII cash net) = -0.652.
# Domestic institutions take the other side of most FII cash flow, so FII selling
# arrives into matching DII buying and net demand barely moves. FII cash
# direction is a statement about WHO owns the float, not about where price goes.
FII_HEDGE_TEST = {
    "start": "2024-04-01", "end": "2026-08-20", "n": 587,
    "base_oc": -0.035, "base_up": 45.7,
    "states": {
        "build_sell": {"n": 245, "oc": -0.030, "up": 46.1, "t": -0.67, "wk": +0.078},
        "build_buy":  {"n":  92, "oc": -0.005, "up": 44.6, "t": -0.08, "wk": +0.214},
        "cover_buy":  {"n":  93, "oc": -0.105, "up": 39.8, "t": -1.32, "wk": -0.218},
        "cover_sell": {"n": 157, "oc": -0.017, "up": 49.0, "t": -0.34, "wk": -0.038},
    },
    "alone_oc": -0.023, "alone_n": 337, "adds_oc": -0.007, "adds_t": -0.12,
    "adds_wk": -0.036, "adds_wk_t": -0.24,
    "rho_deliv": +0.093, "rho_turn": +0.078, "rho_dii": -0.652,
    "triple_n": 82, "triple_oc": -0.002,
}
_STATE_LABEL = {
    "build_sell": ("SHORT BUILDUP + selling cash", "reads like a genuine bearish bet"),
    "build_buy":  ("SHORT BUILDUP + buying cash", "reads like a HEDGE, not a bet"),
    "cover_buy":  ("short covering + buying cash", "reads like genuine bullishness"),
    "cover_sell": ("short covering + selling cash", "mixed - unwinding both sides"),
}


def get_fii_hedge_read(as_of: date) -> dict:
    """Classify today into the hedge-vs-bet 2x2, with what that state was worth."""
    out: dict = {"ok": False, "error": ""}
    p = query_dataframe(
        """SELECT trade_date, fut_idx_short FROM fao_participant
           WHERE client_type='FII' AND data_type='OI' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if len(p) < 25:
        out["error"] = "Not enough FII participant history."
        return out
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.drop_duplicates("trade_date").set_index("trade_date")
    sh = p["fut_idx_short"].astype(float)
    scale = sh.rolling(60, min_periods=20).mean().iloc[-1]
    if not (scale > 0):
        out["error"] = "FII short leg unusable."
        return out
    short_pct = float(sh.diff().iloc[-1] / scale * 100)

    c = query_dataframe(
        """SELECT trade_date, fii_net, dii_net FROM fii_dii_cash
           WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 5""", [as_of])
    if c.empty:
        out["error"] = "No FII cash data on or before that date."
        return out
    cash_1d = float(c["fii_net"].iloc[0])
    dii_1d = float(c["dii_net"].iloc[0]) if c["dii_net"].iloc[0] is not None else float("nan")
    cash_5d = float(c["fii_net"].head(5).sum())
    cash_date = c["trade_date"].iloc[0]
    if hasattr(cash_date, "date"):
        cash_date = cash_date.date()

    key = ("build" if short_pct > 0 else "cover") + ("_sell" if cash_1d < 0 else "_buy")
    label, reading = _STATE_LABEL[key]
    out.update({"ok": True, "as_of": p.index[-1].date(), "cash_date": cash_date,
                "short_pct": short_pct, "cash_1d": cash_1d, "cash_5d": cash_5d,
                "dii_1d": dii_1d, "state": key, "state_label": label,
                "state_reading": reading, "stats": FII_HEDGE_TEST["states"][key],
                "test": FII_HEDGE_TEST})
    return out


# ── THE WHOLE FAMILY, TESTED ONCE, LAST TWO YEARS ──────────────────────────
# Every FII-derived signal in this database was run through one protocol over
# 2024-08-19 .. 2026-08-20 (515 sessions): 22 signals x 3 horizons = 66 tests.
# Signals: the six legs individually and their signed net, futures long share,
# net futures and its change, options tilt and its change, stock-futures long,
# the Client mirror, FII cash at 1/5/20 days, DII cash, FII-minus-DII, the
# derivative-stats buy/sell net and its OI change, and the composite score.
# Horizons: next-day open->close (the only tradeable one, since every source
# publishes after the close), next-day close-to-close, and five sessions.
#
# BEST RESULT IN THE ENTIRE FAMILY: net_1d on next-day open->close, IC -0.0990,
# |t| = 2.20. Nothing else exceeds 2.06.
#   Bonferroni over 66 tests needs |t| > 3.37  -> nothing
#   Benjamini-Hochberg FDR at 5%               -> ZERO survivors
#
# AND THE TEST THAT SETTLES IT. Shuffling the forward returns and recomputing
# the BEST |t| across all 22 signals, the null distribution of "best of family"
# has median 1.95 and a 95th percentile of 2.99. Observed best is 2.20.
#   FAMILY-WISE p = 0.314
# In other words the strongest thing found is weaker than what a search this
# wide produces from pure noise. This is not "no edge was found"; it is "the
# best find is indistinguishable from the search itself".
#
# The best signal traded, for completeness. net_1d quintiles, next-day
# open->close: Q1 +0.114% (up 55.0%), Q3 -0.110%, Q5 -0.119% (up 40.4%) --
# INVERTED against its own labels, as the 2-year leg backtest also found.
# Q5 minus Q1 is -23.3 bps a day. Trading it as labelled loses 73.3 bps a round
# trip; trading it INVERTED still loses 26.7 bps against a ~50 bps cost.
#
# Base rates in this window matter for reading any of the above: next-day
# open->close drift is -0.0287% (up 45.0%) and five-session -0.0326% (up 44.3%),
# while close-to-close is +0.0006% (up 48.3%). The overnight gap carries the
# entire return; the part of the day you can actually trade on this data drifts
# down.
FII_FAMILY_SWEEP_2Y = {
    "start": "2024-08-19", "end": "2026-08-20", "n": 515,
    "n_signals": 22, "n_tests": 66,
    "best_signal": "net_1d", "best_hz": "next-day open-close",
    "best_ic": -0.0990, "best_t": 2.20,
    "bonferroni_t": 3.37, "fdr_survivors": 0,
    "null_best_t_median": 1.95, "null_best_t_p95": 2.99, "family_p": 0.314,
    "q1_mean": +0.114, "q1_up": 55.0, "q5_mean": -0.119, "q5_up": 40.4,
    "spread_bps": -23.3, "cost_bps": 50,
    "base_oc": -0.0287, "base_oc_up": 45.0,
    "base_cc": +0.0006, "base_cc_up": 48.3,
    "base_wk": -0.0326, "base_wk_up": 44.3,
}


def get_fii_price_map(as_of: date, zone: Optional[int] = None) -> dict:
    """Where the index can go, conditioned on where FII positioning sits.

    Returns both the unconditional and the FII-conditioned quantiles in POINTS,
    so the page can show what the FII read is actually worth in levels rather
    than asserting it. Sigma comes from price, the shift comes from FIIs.
    """
    out: dict = {"ok": False, "error": ""}
    px = query_dataframe(
        """SELECT trade_date, close_val FROM index_data
           WHERE index_name = 'Nifty 50' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if len(px) < 300:
        out["error"] = "Not enough Nifty history for a volatility estimate."
        return out
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.drop_duplicates("trade_date").set_index("trade_date")
    ret = px["close_val"].pct_change() * 100.0
    # Causal: today's sigma uses returns through YESTERDAY, matching how the
    # quantiles above were measured. Without the shift this reads tomorrow's
    # scale off today's move.
    sigma = float(ret.ewm(alpha=1 - 0.94).std().shift(1).iloc[-1])
    spot = float(px["close_val"].iloc[-1])
    if not (sigma > 0) or not (spot > 0):
        out["error"] = "Volatility estimate unavailable."
        return out

    if zone is None:
        v = get_fii_only_view(as_of)
        zone = v.get("zone") if v.get("ok") else None
    if zone is None:
        out["error"] = "No FII zone for that date."
        return out
    # VALIDATE THE ARGUMENT. Callers pass a zone straight through from the view,
    # and an out-of-range value used to reach _FII_QUANTILES[h][zone] and raise a
    # bare KeyError. The view wraps this call in a broad except, so the map would
    # simply vanish from the page with nothing logged. Fail as data, not as an
    # exception, and say which argument was wrong.
    try:
        zone = int(zone)
    except (TypeError, ValueError):
        out["error"] = f"Zone must be an integer 1-5, got {zone!r}."
        return out
    if zone not in (1, 2, 3, 4, 5):
        out["error"] = f"Zone must be 1-5, got {zone}."
        return out

    # Trailing-only quantiles when there is enough history, frozen table otherwise.
    live_q, basis = _live_quantiles(as_of, zone), "trailing"
    if live_q is None:
        live_q, basis = {h: {0: _FII_QUANTILES[h][0], zone: _FII_QUANTILES[h][zone]}
                         for h in ("d1", "d5")}, "frozen 2018-2026 table"

    bands = {}
    for h in ("d1", "d5"):
        cond = live_q[h][zone]
        base = live_q[h][0]
        bands[h] = {
            "q": _FII_Q,
            "cond_pts": [spot * (1 + z * sigma / 100.0) for z in cond],
            "base_pts": [spot * (1 + z * sigma / 100.0) for z in base],
            "cond_sig": list(cond), "base_sig": list(base),
            "shift_pts": [spot * (c - b) * sigma / 100.0 for c, b in zip(cond, base)],
            "tail_thr_pts": spot * (1 + _FII_TAIL[h]["thr"] * sigma / 100.0),
            "tail_pct": _FII_TAIL[h][zone],
            "tail_base_pct": _FII_TAIL[h][0],
            "tail_sig": _FII_TAIL[h]["thr"],
        }
    out.update({"ok": True, "as_of": px.index[-1].date(), "spot": spot,
                "sigma_pct": sigma, "zone": zone, "basis": basis})
    out["bands"] = bands
    return out


def _live_quantiles(as_of: date, zone: int):
    """Forward-return quantiles in sigma units, from data strictly before as_of.

    Returns None when history or the zone bucket is too thin, so the caller can
    fall back to the frozen table rather than quoting a band built on 20 days.
    """
    zs = _zone_series(as_of)
    if zs.empty or zs["zone"].notna().sum() < _MIN_HIST:
        return None
    px = query_dataframe(
        """SELECT trade_date, close_val FROM index_data
           WHERE index_name = 'Nifty 50' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if len(px) < _MIN_HIST:
        return None
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.drop_duplicates("trade_date").set_index("trade_date")
    ret = px["close_val"].pct_change() * 100.0
    sg = ret.ewm(alpha=1 - 0.94).std().shift(1)
    d = pd.DataFrame({"zone": zs["zone"], "sig": sg}).dropna()
    d["z1"] = ((px["close_val"].shift(-1) / px["close_val"] - 1) * 100.0) / d["sig"]
    d["z5"] = ((px["close_val"].shift(-5) / px["close_val"] - 1) * 100.0) / d["sig"]
    out = {}
    for h in ("d1", "d5"):
        col = "z1" if h == "d1" else "z5"
        sub = d[["zone", col]].dropna()
        if len(sub) < _MIN_HIST:
            return None
        bucket = sub[col][sub["zone"] == zone]
        if len(bucket) < _MIN_BUCKET:
            return None
        out[h] = {0: tuple(np.quantile(sub[col], _FII_Q)),
                  zone: tuple(np.quantile(bucket, _FII_Q))}
    return out


def get_fii_actions(as_of: date) -> dict:
    """
    What FIIs actually DID — per leg, today and over the last 5 sessions.

    Each leg's change is expressed as a % of that leg's OWN recent average open
    interest. Raw contract counts are NOT comparable across SEBI's Nov-2024
    lot-size change (median FII index-futures long OI fell from ~156k to ~27k),
    so an unnormalised delta would partly measure a rule change.

    Returns {"ok", "as_of", "legs", "net_1d", "net_5d", "track", "error"}.
    """
    out: dict = {"ok": False, "as_of": as_of, "error": "",
                 "track": dict(FII_ACTION_TRACK)}
    p = query_dataframe(
        """SELECT * FROM fao_participant
           WHERE client_type='FII' AND data_type='OI' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if p.empty:
        out["error"] = "No FII participant data on or before that date."
        return out
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.drop_duplicates("trade_date").set_index("trade_date")
    if len(p) < 25:
        out["error"] = "Not enough FII history to measure position changes."
        return out

    # Build each leg as a SERIES, not just today's value, so the net score's own
    # day-over-day change can be reported. net_1d is itself a one-day change in
    # open interest, so its delta is a SECOND difference: not "FIIs got more
    # bullish by 17 points" but "today's positioning shift was 17 points larger
    # than yesterday's". Noisier than the level and it means something different,
    # so the page has to say which it is showing.
    net1_series = pd.Series(0.0, index=p.index)
    net5_series = pd.Series(0.0, index=p.index)

    rows, net1, net5 = [], 0.0, 0.0
    for leg, col in _LEG_COL.items():
        raw = p[col].astype(float)
        scale = raw.rolling(60, min_periods=20).mean()
        sc = scale.iloc[-1]
        if not (sc > 0):
            continue
        net1_series += _LEG_SIGN[leg] * (raw.diff() / scale * 100)
        net5_series += _LEG_SIGN[leg] * (raw.diff(5) / scale * 100)
        d1 = float(raw.diff().iloc[-1])
        d5 = float(raw.diff(5).iloc[-1]) if len(raw) > 5 else np.nan
        pct1 = d1 / sc * 100
        pct5 = (d5 / sc * 100) if d5 == d5 else np.nan
        up_lbl, dn_lbl, rise_lean = LEG_LABELS[leg]
        action = up_lbl if d1 > 0 else dn_lbl if d1 < 0 else "UNCHANGED"
        if d1 > 0:
            lean = rise_lean
        elif d1 < 0:
            lean = "bearish" if rise_lean == "bullish" else "bullish"
        else:
            lean = "neutral"
        rows.append({
            "leg": leg.replace("_", " ").title(),
            "action": action, "lean": lean,
            "oi_now": float(raw.iloc[-1]),
            "chg_1d": d1, "pct_1d": pct1,
            "chg_5d": d5, "pct_5d": pct5,
        })
        net1 += _LEG_SIGN[leg] * pct1
        net5 += _LEG_SIGN[leg] * (pct5 if pct5 == pct5 else 0.0)

    if not rows:
        out["error"] = "FII leg data unusable for that date."
        return out

    # EXPIRY DAY IS SETTLEMENT, NOT SENTIMENT. On the 112 NIFTY index-option
    # expiries in this history the four FII option legs fall by a mean of
    # -20.8% (call long), -19.1% (call short), -15.4% (put long) and -18.6%
    # (put short) of their own 60-day average OI, against +1.0% to +1.3% on
    # every other day. Those drops are contracts ceasing to exist. Read as
    # position changes they manufacture a loud verdict from nothing: on
    # 2026-08-18 the legs printed CALL LONGS EXITING -29.5% and PUT WRITERS
    # COVERING -41.7%, and net_1d came to -47.4, essentially all of it
    # mechanical. The unwind is not symmetric across the legs, so it does not
    # cancel in the net either. Flag the day; callers must not read a stance
    # off it.
    # Previous session's net, for the day-over-day arrow. Uses the previous
    # PARTICIPANT row, not a calendar day, so a holiday or a missing file does
    # not silently produce a two-day delta labelled as one.
    def _prev(series: pd.Series):
        v = series.dropna()
        if len(v) < 2:
            return np.nan, None
        return float(v.iloc[-2]), v.index[-2].date()

    prev1, prev1_date = _prev(net1_series)
    prev5, prev5_date = _prev(net5_series)
    delta1 = (net1 - prev1) if prev1 == prev1 else np.nan
    delta5 = (net5 - prev5) if prev5 == prev5 else np.nan

    d = p.index[-1].date()
    ex = query_dataframe(
        """SELECT 1 FROM fno_bhavcopy
           WHERE instrument='OPTIDX' AND expiry_date = ? AND trade_date = ?
           LIMIT 1""", [d, d])
    is_expiry = not ex.empty

    out.update({"ok": True, "as_of": d,
                "legs": pd.DataFrame(rows),
                "net_1d": float(net1), "net_5d": float(net5),
                "net_1d_prev": prev1, "net_5d_prev": prev5,
                "net_1d_delta": delta1, "net_5d_delta": delta5,
                "prev_date": prev1_date,
                # a delta that straddles an expiry compares a settlement day with
                # a normal one, so the arrow is meaningless on those days
                "delta_ok": bool(prev1 == prev1),
                "expiry_day": is_expiry,
                "expiry_note": (
                    "Index options expired today. The 1-day leg changes below are "
                    "dominated by settlement, not by FIIs changing their minds — "
                    "option legs drop 15-21% of normal open interest on expiry days "
                    "versus about +1% otherwise. Read the 5-day column instead."
                ) if is_expiry else ""})
    return out


def get_fii_only_view(as_of: date) -> dict:
    """
    FII-only read as of `as_of`. Returns {"ok", "as_of", "stance", "score",
    "components", "per_index", "track", "error"}.

    Causal: reads only sessions on or before `as_of`.
    """
    out: dict = {"ok": False, "as_of": as_of, "error": "",
                 "track": dict(FII_TRACK_RECORD)}
    Z = _components(as_of)
    if Z.empty:
        out["error"] = "No FII participant data on or before that date."
        return out

    dirs = [c for c in ("cash_net", "fut_share", "fut_net_chg", "opt_bull",
                        "stats_net") if c in Z.columns]
    row = Z.iloc[-1]
    live = [c for c in dirs if row[c] == row[c]]
    if len(live) < 3:
        out["error"] = ("Fewer than 3 FII components available for that date — "
                        "not enough to form a view.")
        return out

    score = float(np.mean([row[c] for c in live]))

    # COMPOSITION-STABLE SCORE. `score` is a mean over whichever components are
    # live that day, and its spread moves with how many that is (see
    # FII_ERA_SPLIT). Re-standardising the composite on its own trailing 250
    # sessions makes one unit mean the same thing in every era, which is what
    # the zone map is measured on. The raw score is still returned so the two
    # can be compared on the page.
    raw_hist = Z[dirs].mean(axis=1).where(Z[dirs].notna().sum(axis=1) >= 3)
    _m = raw_hist.rolling(250, min_periods=120).mean()
    _s = raw_hist.rolling(250, min_periods=120).std()
    norm_hist = (raw_hist - _m) / _s.replace(0.0, np.nan)
    nscore = float(norm_hist.iloc[-1]) if len(norm_hist) and norm_hist.iloc[-1] == norm_hist.iloc[-1] else np.nan

    zone = None
    if nscore == nscore:
        zone = 1 + int(sum(nscore > c for c in _ZONE_CUTS))

    # STANCE IS GATED ON THE MEASURED ZONE, NOT ON A BARE THRESHOLD. Crossing
    # -0.5 sigma is a fact about the score's distribution, not evidence about
    # tomorrow. Of the five quintiles only the outer two carry any measured tilt
    # (Q1 next-day mean -0.146% vs a -0.064% base, Q5 5-day +0.361% vs +0.107%);
    # Q2-Q4 are indistinguishable from the base rate. Gating on +-0.5 of the
    # stable score would have printed BEARISH on 2026-08-18 at a score of -0.555
    # -- which lands in Q2, whose measured next-day up-rate is 47.2% against a
    # 47.4% base, i.e. nothing. A headline that says BEARISH above a table that
    # says "no effect" is the contradiction this page exists to avoid. Only the
    # outer quintiles may name a direction.
    if zone is not None:
        stance = ("BEARISH" if zone == 1 else
                  "BULLISH" if zone == 5 else "NEUTRAL")
    else:                       # early history: no stable score yet
        stance = ("BULLISH" if score >= _THR else
                  "BEARISH" if score <= -_THR else "NEUTRAL")

    labels = {
        "cash_net": ("FII cash flow", "Net FII buying in the cash market"),
        "fut_share": ("FII futures positioning", "Share of FII index-futures OI that is long"),
        "fut_net_chg": ("FII futures change", "Day change in FII net index-futures OI"),
        "opt_bull": ("FII options tilt", "Calls held long + puts written, minus the reverse"),
        "stats_net": ("FII futures flow", "Index-futures buy value minus sell value"),
    }
    comps = []
    for c_ in dirs:
        z = row.get(c_, np.nan)
        name, why = labels[c_]
        comps.append({
            "component": name, "why": why, "z": float(z) if z == z else np.nan,
            "lean": ("bullish" if z == z and z >= _THR else
                     "bearish" if z == z and z <= -_THR else
                     "neutral" if z == z else "no data"),
        })

    # conviction: how much FII volume is running vs its own normal
    vz = row.get("vol_intensity", np.nan)
    conviction = ("heavy" if vz == vz and vz >= 1 else
                  "light" if vz == vz and vz <= -1 else "normal")

    # per-index FII futures flow (2023-01-31+)
    per = []
    for disp, cat in INDEX_CATEGORY.items():
        d = query_dataframe(
            """SELECT trade_date, buy_value_cr, sell_value_cr, oi_value_cr
               FROM fii_derivatives_stats
               WHERE category = ? AND trade_date <= ? ORDER BY trade_date""",
            [cat, as_of])
        if d.empty:
            continue
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        d = d.drop_duplicates("trade_date").set_index("trade_date")
        net = d["buy_value_cr"] - d["sell_value_cr"]
        zz = _z(net, 20)
        oi_now = float(d["oi_value_cr"].iloc[-1])
        thin = oi_now < _ILLIQUID_OI_CR
        per.append({
            "index": disp, "date": d.index[-1].date(),
            "net_cr": float(net.iloc[-1]),
            # A z-score on a contract this small measures rounding, not flow.
            "net_z": (np.nan if thin else
                      (float(zz.iloc[-1]) if zz.iloc[-1] == zz.iloc[-1] else np.nan)),
            "oi_cr": oi_now,
            "oi_chg_cr": float(d["oi_value_cr"].diff().iloc[-1]),
            "net_5d_cr": float(net.tail(5).sum()),
            "thin": thin,
        })

    out.update({
        "ok": True, "as_of": Z.index[-1].date(), "stance": stance, "score": score,
        "norm_score": nscore, "zone": zone,
        "zone_label": _ZONE_LABEL.get(zone, ""),
        "zones": FII_ZONES, "era": FII_ERA_SPLIT,
        "n_live": len(live), "conviction": conviction,
        "components": pd.DataFrame(comps),
        "per_index": pd.DataFrame(per),
    })
    return out
