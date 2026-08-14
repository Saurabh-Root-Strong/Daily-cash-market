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

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_fii_only_view", "get_fii_actions", "FII_TRACK_RECORD",
           "FII_ACTION_TRACK", "INDEX_CATEGORY", "LEG_LABELS"]

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


def _components(as_of: date) -> pd.DataFrame:
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
    near = chain["expiry_date"].min()
    nr = chain[chain["expiry_date"] == near]

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
    opt_contracts = float(sum(
        r["oi"] / _INDEX_LOTS[r["symbol"]] for _, r in tot_opt.iterrows()
        if r["symbol"] in _INDEX_LOTS))
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
        "fii_opt_contracts": fii_opt,
        "total_opt_contracts": opt_contracts,
        "fii_share_pct": fii_share,
        "fii_fut_contracts": float(fii["fut_oi"].iloc[0]) if not fii.empty else np.nan,
        "rollover_pct": roll,
    })
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

    rows, net1, net5 = [], 0.0, 0.0
    for leg, col in _LEG_COL.items():
        raw = p[col].astype(float)
        scale = raw.rolling(60, min_periods=20).mean()
        sc = scale.iloc[-1]
        if not (sc > 0):
            continue
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

    out.update({"ok": True, "as_of": p.index[-1].date(),
                "legs": pd.DataFrame(rows),
                "net_1d": float(net1), "net_5d": float(net5)})
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
        per.append({
            "index": disp, "date": d.index[-1].date(),
            "net_cr": float(net.iloc[-1]),
            "net_z": float(zz.iloc[-1]) if zz.iloc[-1] == zz.iloc[-1] else np.nan,
            "oi_cr": float(d["oi_value_cr"].iloc[-1]),
            "oi_chg_cr": float(d["oi_value_cr"].diff().iloc[-1]),
            "net_5d_cr": float(net.tail(5).sum()),
        })

    out.update({
        "ok": True, "as_of": Z.index[-1].date(), "stance": stance, "score": score,
        "n_live": len(live), "conviction": conviction,
        "components": pd.DataFrame(comps),
        "per_index": pd.DataFrame(per),
    })
    return out
