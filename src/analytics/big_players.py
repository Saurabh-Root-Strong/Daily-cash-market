"""
Big Players — what each F&O participant DID, and what each scenario has historically
been followed by.

Two products, both computed at call time from rows dated <= as_of:

1. get_player_actions(as_of)
   Per participant (FII, Pro, Client, DII): change in index-futures longs and shorts
   over 1 and 5 sessions, the classic label for that change (long build / short
   build / short cover / long unwind), the four index-option legs, and the zero-sum
   "who took the other side" split of the day's futures flow.

2. get_scenario_playbook(as_of)
   Every past session classified into a named scenario:
       FII 5-session futures flow (4 states)  x  Nifty 5-session direction (2)
   plus who absorbed FII's flow (Client / Pro / DII) and FII's positioning zone.
   For each scenario: separate episodes, Nifty over the next 5/10/20 sessions
   (entry next open — the file publishes after the close), versus all days, and how
   often a 5% swing turn was close.

Guards: real sessions only (fao_participant carries junk rows dated on holidays);
5-session windows that contain a monthly futures expiry are excluded from the
statistics (expiring contracts leave both legs and fake an "unwind"); contract
counts are only ever differenced over <= 5 sessions, never compared across years.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from src.analytics.positioning_cycle import (
    TURN_PCT, NEAR_AFTER, NEAR_FROM, _episodes, _near, _pct_rank, _roll_quantile, _zigzag, _zone_of,
    data_version, expiry_calendar,
)
from src.data.repository import query_dataframe

__all__ = ["get_player_actions", "get_scenario_playbook", "FLOW_STATES", "PLAYER_ORDER"]

PLAYER_ORDER = ["FII", "Pro", "Client", "DII"]
_LEGS = ["fut_idx_long", "fut_idx_short", "opt_idx_call_long", "opt_idx_call_short",
         "opt_idx_put_long", "opt_idx_put_short"]

FLOW_STATES = {
    "LB": ("LONG BUILD", "adding fresh longs", +1),
    "SC": ("SHORT COVER", "buying back shorts", +1),
    "SB": ("SHORT BUILD", "adding fresh shorts", -1),
    "LU": ("LONG UNWIND", "selling out of longs", -1),
    "FLAT": ("NO CHANGE", "book unchanged", 0),
}
SCEN_HORIZONS = (5, 10, 20)
SCEN_COOLDOWN = 10   # = the 10-session horizon the t-stat is read on, so episodes don't overlap
SCEN_MIN_EP = 15


def _f(v):
    try:
        return None if v is None or not np.isfinite(v) else float(v)
    except TypeError:
        return None


def _flow_state(dl: float, ds: float) -> str:
    """Classic OI label from the change in longs (dl) and shorts (ds): the bigger leg wins."""
    if not np.isfinite(dl) or not np.isfinite(ds) or (dl == 0 and ds == 0):
        return "FLAT"
    if abs(dl) >= abs(ds):
        return "LB" if dl > 0 else "LU"
    return "SB" if ds > 0 else "SC"


# ───────────────────────────────────────────────────────────────────── loading
@lru_cache(maxsize=16)
def _load_cached(as_of: date, _version: tuple = ()):
    part = query_dataframe(
        f"""SELECT trade_date, client_type, {', '.join(_LEGS)}
            FROM fao_participant
            WHERE upper(data_type) = 'OI' AND trade_date <= ?""", [as_of])
    px = query_dataframe(
        """SELECT trade_date, open_val, close_val FROM index_data
           WHERE index_name = 'Nifty 50' AND trade_date <= ? ORDER BY trade_date""", [as_of])
    if part.empty or px.empty:
        return None
    part["trade_date"] = pd.to_datetime(part["trade_date"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.drop_duplicates("trade_date").set_index("trade_date").sort_index().astype(float)
    part["client_type"] = part["client_type"].astype(str).str.strip()
    part = part[part["trade_date"].isin(px.index)]          # holiday rows are junk
    wide = {}
    for ct in PLAYER_ORDER:
        g = part[part["client_type"].str.upper() == ct.upper()]
        g = g.drop(columns="client_type").drop_duplicates("trade_date").set_index("trade_date").sort_index()
        wide[ct] = g.astype(float)
    idx = sorted(set().union(*[w.index for w in wide.values()]))
    idx = pd.DatetimeIndex(idx)
    wide = {k: v.reindex(idx) for k, v in wide.items()}
    exp = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in expiry_calendar(as_of, px.index)))
    return wide, px, exp


def _load(as_of: date):
    got = _load_cached(as_of, data_version())
    if got is None:
        return None
    wide, px, exp = got
    return {k: v.copy() for k, v in wide.items()}, px.copy(), exp


# ───────────────────────────────────────────────────────────────────── actions
def _option_read(d: dict) -> tuple[str, int]:
    """Plain read of the four option-leg changes. Bullish legs: bought calls, wrote puts."""
    bull = (d["opt_idx_call_long"] - d["opt_idx_call_short"]) + (d["opt_idx_put_short"] - d["opt_idx_put_long"])
    if not np.isfinite(bull) or bull == 0:
        return "options flat", 0
    return ("options tilted bullish" if bull > 0 else "options tilted bearish"), int(np.sign(bull))


def get_player_actions(as_of: date) -> dict:
    try:
        got = _load(as_of)
    except Exception as exc:                                # noqa: BLE001
        return {"error": f"Could not read participant data: {exc}"}
    if got is None:
        return {"error": "No participant or Nifty data on or before that date."}
    wide, px, exp = got
    idx = wide["FII"].dropna(how="all").index
    if len(idx) < 2:
        return {"error": "Need at least two participant sessions."}
    d0, d1 = idx[-1], idx[-2]
    d5 = idx[-6] if len(idx) > 5 else idx[0]
    cl = px["close_val"]
    nifty_chg = _f((cl.loc[d0] / cl.loc[d1] - 1) * 100)
    nifty_chg5 = _f((cl.loc[d0] / cl.loc[d5] - 1) * 100)

    players = []
    for ct in PLAYER_ORDER:
        w = wide[ct]
        if d0 not in w.index or w.loc[d0].isna().all():
            players.append({"name": ct, "missing": True})
            continue
        now, prev, prev5 = w.loc[d0], w.loc[d1], w.loc[d5]
        dd = (now - prev).to_dict()
        dd5 = (now - prev5).to_dict()
        st1 = _flow_state(dd["fut_idx_long"], dd["fut_idx_short"])
        st5 = _flow_state(dd5["fut_idx_long"], dd5["fut_idx_short"])
        opt_txt, opt_sign = _option_read(dd)
        ls = now["fut_idx_long"] / (now["fut_idx_long"] + now["fut_idx_short"])
        ls_prev = prev["fut_idx_long"] / (prev["fut_idx_long"] + prev["fut_idx_short"])
        share_series = w["fut_idx_long"] / (w["fut_idx_long"] + w["fut_idx_short"])
        pct = _pct_rank(share_series.dropna()).iloc[-1] if share_series.notna().sum() >= 120 else np.nan
        players.append({
            "name": ct, "missing": False,
            "long": _f(now["fut_idx_long"]), "short": _f(now["fut_idx_short"]),
            "d_long": _f(dd["fut_idx_long"]), "d_short": _f(dd["fut_idx_short"]),
            "d_long5": _f(dd5["fut_idx_long"]), "d_short5": _f(dd5["fut_idx_short"]),
            "net_change": _f(dd["fut_idx_long"] - dd["fut_idx_short"]),
            "net_change5": _f(dd5["fut_idx_long"] - dd5["fut_idx_short"]),
            "net": _f(now["fut_idx_long"] - now["fut_idx_short"]),
            "state1": st1, "state5": st5,
            "long_share": _f(ls * 100), "long_share_chg": _f((ls - ls_prev) * 100),
            "pct": _f(pct), "zone": _zone_of(_f(pct)),
            "opt": {k: _f(dd[k]) for k in _LEGS[2:]},
            "opt_now": {k: _f(now[k]) for k in _LEGS[2:]},
            "opt_read": opt_txt, "opt_sign": opt_sign,
            "pcr": _f(now["opt_idx_put_long"] / now["opt_idx_call_long"])
            if now["opt_idx_call_long"] else None,
        })

    ok = [p for p in players if not p["missing"]]
    total_net = sum(p["net_change"] for p in ok) if ok else None
    fii = next((p for p in ok if p["name"] == "FII"), None)
    absorb = None
    if fii and fii["net_change"]:
        others = [p for p in ok if p["name"] != "FII"]
        opp = [p for p in others if np.sign(p["net_change"]) == -np.sign(fii["net_change"])]
        tot_opp = sum(abs(p["net_change"]) for p in opp)
        absorb = {p["name"]: (abs(p["net_change"]) / tot_opp * 100 if tot_opp else 0.0) for p in opp}
    expiry_today = d0 in exp
    expiry_in5 = bool(((exp > d5) & (exp <= d0)).any())
    lot_in5 = [d for d, _ in LOT_STEPS if d5 < d <= d0]
    warnings = []
    gap = int(((px.index > d1) & (px.index < d0)).sum())
    if gap:
        warnings.append(f"{gap} session(s) between {d1:%d %b} and {d0:%d %b} have no participant file — "
                        "the 1-day change covers more than one day.")
    if px.index[-1] > d0:
        warnings.append(f"Participant data ends {d0:%d %b %Y}; Nifty has later sessions.")
    if expiry_today:
        warnings.append("Monthly index-futures expiry today: expiring contracts leave every book, so "
                        "today's changes are largely mechanical — read the 5-session column instead.")
    elif expiry_in5:
        warnings.append("The last 5 sessions include a monthly expiry, so the 5-session changes include "
                        "contracts that simply expired.")
    if lot_in5:
        warnings.append(f"NSE cut index lot sizes on {lot_in5[0]:%d %b %Y}, inside this 5-session window: the same "
                        "exposure now shows ~15% more contracts, so the 5-session contract changes are overstated.")
    if ok and len(ok) == 4 and total_net is not None and abs(total_net) > 0.01 * max(
            1.0, sum(abs(p["net_change"]) for p in ok)):
        warnings.append(f"The four categories' futures changes do not sum to zero ({total_net:+,.0f}) — "
                        "the file is incomplete.")
    return {
        "data_date": d0.date(), "prev_date": d1.date(), "date5": d5.date(),
        "nifty_close": _f(cl.loc[d0]), "nifty_chg": nifty_chg, "nifty_chg5": nifty_chg5,
        "players": players, "absorb": absorb, "total_net": total_net,
        "expiry_today": bool(expiry_today), "expiry_in5": expiry_in5, "warnings": warnings,
    }


# ───────────────────────────────────────────────────────────────────── playbook
def _measure(mask, fwd, base, bot_i, top_i, cooldown=SCEN_COOLDOWN):
    ep = _episodes(mask, cooldown)
    ii = np.flatnonzero(ep)
    out = {"days": int(mask.sum()), "episodes": int(len(ii)),
           "bottom_near": _near(ii, bot_i), "top_near": _near(ii, top_i)}
    for h in SCEN_HORIZONS:
        x = fwd[h][ep]
        x = x[np.isfinite(x)]
        if len(x) >= 3 and x.std(ddof=1) > 0:
            out[f"fwd{h}"] = float(x.mean())
            out[f"up{h}"] = float(100 * (x > 0).mean())
            out[f"t{h}"] = float((x.mean() - base[h]) / (x.std(ddof=1) / np.sqrt(len(x))))
        else:
            out[f"fwd{h}"] = out[f"up{h}"] = out[f"t{h}"] = None
    out["judged"] = out["episodes"] >= SCEN_MIN_EP
    return out


SCENARIOS = [  # (key, flow states, price dir, name, plain meaning)
    ("rally_fii", ("LB", "SC"), +1, "FII-backed rally",
     "Nifty up over 5 sessions and FII adding longs or covering shorts — the move has FII behind it."),
    ("rally_vs_fii", ("SB", "LU"), +1, "Rally against FII",
     "Nifty up but FII adding shorts or selling longs — FII leaning against the rise."),
    ("fall_fii", ("SB", "LU"), -1, "FII-backed fall",
     "Nifty down and FII adding shorts or dumping longs — the fall has FII behind it."),
    ("fall_vs_fii", ("LB", "SC"), -1, "FII buying the dip",
     "Nifty down but FII adding longs or covering shorts — FII leaning against the fall."),
]


def get_scenario_playbook(as_of: date) -> dict:
    try:
        got = _load(as_of)
    except Exception as exc:                                # noqa: BLE001
        return {"error": f"Could not read participant data: {exc}"}
    if got is None:
        return {"error": "No participant or Nifty data on or before that date."}
    wide, px, exp = got
    fii = wide["FII"]
    idx = fii.dropna(subset=["fut_idx_long"]).index
    if len(idx) < 150:
        return {"error": f"Only {len(idx)} sessions of history — the playbook needs at least 150."}
    fii = fii.reindex(idx)
    cl, op = px["close_val"], px["open_val"]
    c = cl.reindex(idx)
    fwd = {h: ((cl.shift(-h) / op.shift(-1) - 1) * 100).reindex(idx).values for h in SCEN_HORIZONS}
    base = {h: float(np.nanmean(fwd[h])) for h in SCEN_HORIZONS}
    base_up = {h: float(100 * np.nanmean(fwd[h][np.isfinite(fwd[h])] > 0)) for h in SCEN_HORIZONS}

    dl5 = fii["fut_idx_long"].diff(5).values
    ds5 = fii["fut_idx_short"].diff(5).values
    flow = np.array([_flow_state(a, b) for a, b in zip(dl5, ds5)])
    pdir = np.sign((c / c.shift(5) - 1).values)
    # 5-session windows containing a monthly expiry are excluded from the statistics
    pos = np.searchsorted(idx.values, exp.values)
    contam = np.zeros(len(idx), bool)
    for p in pos:
        contam[p:min(p + 5, len(idx))] = True
    usable = ~contam & np.isfinite(dl5) & np.isfinite(pdir)

    share = fii["fut_idx_long"] / (fii["fut_idx_long"] + fii["fut_idx_short"])
    pct = _pct_rank(share).values
    zone = np.array([_zone_of(_f(v)) for v in pct], dtype=object)
    zgrp = np.where(np.isin(zone, ["xs"]), "short", np.where(np.isin(zone, ["xl"]), "long",
                    np.where(zone == None, "none", "mid")))  # noqa: E711

    # who absorbed FII's 5-session flow
    netchg = {ct: (wide[ct]["fut_idx_long"] - wide[ct]["fut_idx_short"]).reindex(idx).diff(5).values
              for ct in PLAYER_ORDER}
    fsign = np.sign(netchg["FII"])
    absorber = np.full(len(idx), "", dtype=object)
    for i in range(len(idx)):
        if not np.isfinite(fsign[i]) or fsign[i] == 0:
            continue
        opp = {ct: -netchg[ct][i] * fsign[i] for ct in ("Client", "Pro", "DII")}
        opp = {k: v for k, v in opp.items() if np.isfinite(v) and v > 0}
        if opp:
            absorber[i] = max(opp, key=opp.get)

    zz = _zigzag(c.ffill().values, TURN_PCT)
    top_i = np.array([i for i, k in zz if k == "TOP"], int)
    bot_i = np.array([i for i, k in zz if k == "BOTTOM"], int)
    chance_bot = _near(np.flatnonzero(usable), bot_i)
    chance_top = _near(np.flatnonzero(usable), top_i)

    today_flow, today_dir = flow[-1], pdir[-1]
    today_key = next((k for k, fs, d, *_ in SCENARIOS if today_flow in fs and today_dir == d), None)

    rows = []
    for key, fs, d, name, meaning in SCENARIOS:
        m = usable & np.isin(flow, fs) & (pdir == d)
        r = _measure(m, fwd, base, bot_i, top_i)
        r.update({"key": key, "name": name, "meaning": meaning, "today": key == today_key,
                  "share_days": float(100 * m.sum() / max(1, usable.sum()))})
        rows.append(r)

    detail = []
    for st_ in ("LB", "SC", "SB", "LU"):
        for d, dname in ((+1, "Nifty up"), (-1, "Nifty down")):
            m = usable & (flow == st_) & (pdir == d)
            r = _measure(m, fwd, base, bot_i, top_i)
            r.update({"flow": FLOW_STATES[st_][0], "price": dname,
                      "today": bool(st_ == today_flow and d == today_dir)})
            detail.append(r)

    by_zone = []
    for key, fs, d, name, _ in SCENARIOS:
        for zg, zname in (("short", "FII extreme short (≤10th pct)"), ("mid", "FII normal (10-90th)"),
                          ("long", "FII extreme long (≥90th pct)")):
            m = usable & np.isin(flow, fs) & (pdir == d) & (zgrp == zg)
            r = _measure(m, fwd, base, bot_i, top_i)
            r.update({"scenario": name, "zone": zname,
                      "today": bool(key == today_key and zgrp[-1] == zg)})
            by_zone.append(r)

    by_absorber = []
    for ab in ("Client", "Pro", "DII"):
        for sgn, sname in ((+1, "FII net buying"), (-1, "FII net selling")):
            m = usable & (absorber == ab) & (fsign == sgn)
            r = _measure(m, fwd, base, bot_i, top_i)
            r.update({"absorber": ab, "fii": sname,
                      "today": bool(absorber[-1] == ab and fsign[-1] == sgn)})
            by_absorber.append(r)

    # ---- swing legs, per FII book: index FUTURES and index OPTIONS ---------
    cw = wide["Client"].reindex(idx) if "Client" in wide else None
    fut = _book(idx, c, (share * 100).values,
                (fii["fut_idx_long"] - fii["fut_idx_short"]).values,
                fii["fut_idx_long"].astype(float), fii["fut_idx_short"].astype(float),
                (cw["fut_idx_long"] / (cw["fut_idx_long"] + cw["fut_idx_short"]) * 100).values
                if cw is not None else None, fwd, **BOOKS["fut"]["thresholds"])
    o_bull = fii["opt_idx_call_long"] + fii["opt_idx_put_short"]
    o_bear = fii["opt_idx_call_short"] + fii["opt_idx_put_long"]
    if cw is not None:
        c_bull = cw["opt_idx_call_long"] + cw["opt_idx_put_short"]
        c_tot = c_bull + cw["opt_idx_call_short"] + cw["opt_idx_put_long"]
    opt = _book(idx, c, (o_bull / (o_bull + o_bear).replace(0, np.nan) * 100).values,
                (o_bull - o_bear).values, o_bull.astype(float), o_bear.astype(float),
                (c_bull / c_tot.replace(0, np.nan) * 100).values if cw is not None else None,
                fwd, smooth_sides=True, **BOOKS["opt"]["thresholds"])
    legs, leg_stats, current = fut["legs_all"], fut["leg_stats"], fut["current"]
    share_pts, p10, p90, pct = fut["series"], fut["p10"], fut["p90"], fut["pct"]
    fii_net = fut["net"]
    raw_lab = [next((n for k, fs, d, n, _ in SCENARIOS if f in fs and dd == d), "")
               for f, dd in zip(flow, pdir)]

    # history strip for the chart: last 250 sessions
    tail = slice(-250, None)
    hist = pd.DataFrame({"date": idx[tail], "close": c.values[tail], "flow": flow[tail],
                         "dir": pdir[tail], "contaminated": contam[tail],
                         "scenario": np.array(raw_lab, dtype=object)[tail],
                         "fii_share": share_pts[tail], "fii_pct": pct[tail],
                         "p10": p10[tail], "p90": p90[tail],
                         "fii_net5": pd.Series(fii_net).diff(5).values[tail]})
    for key_, bk_ in (("fut", fut), ("opt", opt)):
        hist[f"{key_}_dl5"], hist[f"{key_}_ds5"] = bk_["dl5"][tail], bk_["ds5"][tail]
        for col in ("L", "S", "Lz", "Sz"):
            hist[f"{key_}_{col}"] = bk_[col][tail]
    hist["opt_share"] = opt["series"][tail]
    hist["opt_pct"] = opt["pct"][tail]
    hist["opt_p10"], hist["opt_p90"] = opt["p10"][tail], opt["p90"][tail]
    lo = idx[tail][0]
    win_legs = [g for g in legs if pd.Timestamp(g["end"]) >= lo]
    peaks = fut["peaks"]
    books = {}
    for key, bk in (("fut", fut), ("opt", opt)):
        books[key] = {**BOOKS[key]["meta"], "thresholds": BOOKS[key]["thresholds"],
                      "legs": [g for g in bk["legs_all"] if pd.Timestamp(g["end"]) >= lo],
                      "leg_stats": bk["leg_stats"], "current_leg": bk["current"],
                      "peaks": bk["peaks"], "legs_total": len(bk["legs_all"]),
                      "today_value": _f(bk["series"][-1]), "today_pct": _f(bk["pct"][-1]),
                      "turn_actions": bk["turn_actions"], "today_state": str(bk["states"][-1]),
                      "pos_turns": [t for t in bk["pos_turns"] if pd.Timestamp(t["date"]) >= lo],
                      "level_odds": bk["level_odds"],
                      "side_now": bk["side_now"],
                      "turn_pcts": [{"kind": r["kind"], "date": r["date"], "pct": r["pct"],
                                     "value": r["fii_long"]} for r in bk["peaks"]["at_turns"]]}

    n_tests = len(rows) + len(detail) + len(by_zone) + len(by_absorber)
    return {
        "data_date": idx[-1].date(), "sessions": int(len(idx)), "usable": int(usable.sum()),
        "history_from": idx[0].date(), "excluded_expiry": int(contam.sum()),
        "today": {"flow": today_flow, "flow_name": FLOW_STATES[today_flow][0], "dir": int(today_dir)
                  if np.isfinite(today_dir) else 0, "key": today_key,
                  "zone": zone[-1], "absorber": absorber[-1] or None,
                  "contaminated": bool(contam[-1]),
                  "dl5": _f(dl5[-1]), "ds5": _f(ds5[-1]),
                  "nifty5": _f((c.iloc[-1] / c.iloc[-6] - 1) * 100) if len(c) > 5 else None},
        "scenarios": rows, "detail": detail, "by_zone": by_zone, "by_absorber": by_absorber,
        "base": base, "base_up": base_up, "chance_bottom": chance_bot, "chance_top": chance_top,
        "history": hist, "n_tests": n_tests,
        "legs": win_legs, "leg_stats": leg_stats, "current_leg": current, "peaks": peaks,
        "books": books,
        "leg_pct": LEG_PCT * 100, "leg_flat_pts": LEG_FLAT_PTS, "legs_total": len(legs),
        "params": {"cooldown": SCEN_COOLDOWN, "min_episodes": SCEN_MIN_EP,
                   "near_from": NEAR_FROM, "near_after": NEAR_AFTER, "turn_pct": TURN_PCT * 100},
    }


def get_player_history(as_of: date, player: str, sessions: int = 250) -> dict:
    """One participant's last `sessions` sessions: long share + percentile, net futures,
    the four option legs, and the daily action label. Counts are shown only inside this
    window (lot sizes change across years, so long windows of counts mislead)."""
    if player not in PLAYER_ORDER:
        return {"error": f"Unknown participant {player!r}."}
    try:
        got = _load(as_of)
    except Exception as exc:                                # noqa: BLE001
        return {"error": f"Could not read participant data: {exc}"}
    if got is None:
        return {"error": "No participant or Nifty data on or before that date."}
    wide, px, exp = got
    w = wide[player].dropna(subset=["fut_idx_long"])
    if len(w) < 2:
        return {"error": f"No {player} history on or before that date."}
    # contract counts in today's lot size (audit: the one-year window spans the
    # 30-Dec-2025 lot cut, which drew a fake ~15% jump in every count)
    w = w.copy()
    for col in _LEGS:
        w[col] = _lot_adjust(w[col])
    share = w["fut_idx_long"] / (w["fut_idx_long"] + w["fut_idx_short"])
    pct = _pct_rank(share) if len(share) >= 120 else pd.Series(np.nan, index=share.index)
    dl, ds = w["fut_idx_long"].diff(), w["fut_idx_short"].diff()
    df = pd.DataFrame({
        "date": w.index,
        "nifty": px["close_val"].reindex(w.index).values,
        "long": w["fut_idx_long"].values, "short": w["fut_idx_short"].values,
        "net": (w["fut_idx_long"] - w["fut_idx_short"]).values,
        "d_long": dl.values, "d_short": ds.values,
        "net_change": (dl - ds).values,
        "long_share": (share * 100).values, "pct": pct.values,
        "call_net": (w["opt_idx_call_long"] - w["opt_idx_call_short"]).values,
        "put_net": (w["opt_idx_put_long"] - w["opt_idx_put_short"]).values,
        "expiry": w.index.isin(exp),
    })
    df["action"] = [FLOW_STATES[_flow_state(a, b)][0] for a, b in zip(dl.values, ds.values)]
    df["nifty_chg"] = pd.Series(df["nifty"]).pct_change().values * 100
    df = df.tail(sessions).reset_index(drop=True)
    return {"player": player, "data_date": w.index[-1].date(), "df": df,
            "lot_note": "Counts before 30 Dec 2025 are converted to today's lot size (Nifty 75 -> 65)."}


# ───────────────────────────────────────────────────────────────────── swing legs
LEG_PCT = 0.03        # a swing leg ends when price reverses 3% from its extreme
LEG_FLAT_PTS = 1.0    # FII long-share change smaller than this = "FII on the sidelines"

LEG_TYPES = {  # (direction, FII side) -> name
    (+1, +1): "FII-backed rally", (+1, -1): "Rally against FII", (+1, 0): "Rally, FII on sidelines",
    (-1, -1): "FII-backed fall", (-1, +1): "FII buying the dip", (-1, 0): "Fall, FII on sidelines",
}


def _swing_legs(idx, close, share_pts, fii_net, client_share, thr: float,
                flat: float = None, fragile: float = 3.0) -> list[dict]:
    """Cut the price history into swing legs (rally / fall between confirmed turns)
    and tag each by what FII did over the WHOLE leg.

    FII's side is the change in its index-futures LONG SHARE across the leg, not
    the change in contract counts: when a month expires, longs and shorts both
    leave the book, which moves counts a lot but the share very little.
    The final leg runs from the last confirmed turn to today and is marked open.
    Leg boundaries are only known in hindsight (a turn needs a 3% reversal), so
    this is a map of what happened, not a signal.
    """
    flat = LEG_FLAT_PTS if flat is None else flat
    n = len(close)
    turns = _zigzag(close, thr)
    # Endpoint values are the median of the 3 sessions ending there. A single day's
    # long share can jump on an expiry or a one-off book change; the audit found 35%
    # of legs flipped tag when an endpoint moved by one session with raw values.
    turn_set = {i for i, _ in turns}
    cuts = [0] + [i for i, _ in turns] + [n - 1]
    cuts = sorted(set(cuts))
    legs = []
    for k in range(len(cuts) - 1):
        a, b = cuts[k], cuts[k + 1]
        if b <= a:
            continue
        move = (close[b] / close[a] - 1) * 100
        d = 1 if move > 0 else -1
        # endpoint values = median of the 3 sessions INSIDE the leg at each end, so a leg
        # never borrows the previous leg's values (an expiry on the turn day otherwise
        # mixes the pre- and post-expiry book and can flip the tag)
        k = max(1, min(3, (b - a + 1) // 2))          # short legs: windows must not overlap
        wa, wb = share_pts[a:a + k], share_pts[b - k + 1:b + 1]
        v_a = np.nanmedian(wa) if np.isfinite(wa).any() else np.nan
        v_b = np.nanmedian(wb) if np.isfinite(wb).any() else np.nan
        ds = v_b - v_a
        seg_c = close[a:b + 1]
        side = 0 if not np.isfinite(ds) or abs(ds) < flat else (1 if ds > 0 else -1)
        seg = share_pts[a:b + 1]
        legs.append({
            "scenario": LEG_TYPES[(d, side)], "dir": d, "fii_side": side,
            "start": idx[a].date(), "end": idx[b].date(), "sessions": int(b - a),
            "nifty_from": float(close[a]), "nifty_to": float(close[b]), "move_pct": float(move),
            "fii_share_from": _f(share_pts[a]), "fii_share_to": _f(share_pts[b]), "fii_share_chg": _f(ds),
            "fii_share_max": _f(np.nanmax(seg)) if np.isfinite(seg).any() else None,
            "fii_share_min": _f(np.nanmin(seg)) if np.isfinite(seg).any() else None,
            "fii_net_chg": _f(fii_net[b] - fii_net[a]),
            "client_share_chg": _f(client_share[b] - client_share[a]) if client_share is not None else None,
            "open": b == n - 1 and (not turns or turns[-1][0] != b),
            # the first leg starts at the start of the data, not at a turn
            "partial": a == 0,
            "fragile": bool(not np.isfinite(ds) or abs(ds) < fragile),
            "extreme_close": float(seg_c.max() if d > 0 else seg_c.min()),
            "extreme_date": idx[a + int(np.argmax(seg_c) if d > 0 else np.argmin(seg_c))].date(),
        })
    return legs


def _leg_stats(legs: list[dict]) -> list[dict]:
    rows = []
    closed = [g for g in legs if not g["open"] and not g["partial"]]
    for d, side in ((+1, +1), (+1, -1), (+1, 0), (-1, -1), (-1, +1), (-1, 0)):
        name = LEG_TYPES[(d, side)]
        g = [x for x in closed if x["dir"] == d and x["fii_side"] == side]
        same_dir = [x for x in closed if x["dir"] == d]
        if not g:
            rows.append({"scenario": name, "dir": d, "side": side, "count": 0})
            continue
        mv = np.array([abs(x["move_pct"]) for x in g])
        ln = np.array([x["sessions"] for x in g])
        rows.append({
            "scenario": name, "dir": d, "side": side, "count": len(g),
            "share_of_dir": float(100 * len(g) / max(1, len(same_dir))),
            "median_move": float(np.median(mv)), "mean_move": float(mv.mean()),
            "big_share": float(100 * np.mean(mv >= 8)),
            "median_sessions": float(np.median(ln)),
            "median_fii_chg": float(np.median([x["fii_share_chg"] for x in g if x["fii_share_chg"] is not None])),
            "fragile_share": float(100 * np.mean([x["fragile"] for x in g])),
            "client_opposite": float(100 * np.mean([
                (x["client_share_chg"] or 0) * (x["fii_share_chg"] or 0) < 0 for x in g])),
        })
    return rows



# ───────────────────────────────────────────────────────────────────── peak levels
def _peak_levels(idx, share_pts, pct, legs, fwd) -> dict:
    """Is there a FII long-% level at which the market turns?

    Three views, all on the swing turns from _swing_legs (3% reversals):
      at_turns   — FII long % and its trailing-year percentile ON each top / bottom
      zone_test  — the reverse question, which is the one that matters for use:
                   when FII is heavy long (>= 90th pct) or heavy short (<= 10th pct),
                   how often is a top / bottom within -3..+10 sessions, vs any day?
      level_test — the user's absolute-level idea ("above 20% it turns down"),
                   run only from Jan-2025 on: the lot-size reset of Nov-2024 moved
                   the whole series (median ~45% before, ~11% after), so a fixed
                   level cannot mean the same thing across that break.
    """
    n = len(idx)
    turn_rows = []
    for g in legs:
        if g["open"]:
            continue
        i = idx.get_loc(pd.Timestamp(g["end"]))
        turn_rows.append({"date": g["end"], "kind": "TOP" if g["dir"] > 0 else "BOTTOM",
                          "nifty": g["nifty_to"], "fii_long": _f(share_pts[i]), "pct": _f(pct[i]),
                          "next_move": None})
    for k in range(len(turn_rows) - 1):
        turn_rows[k]["next_move"] = _f((turn_rows[k + 1]["nifty"] / turn_rows[k]["nifty"] - 1) * 100)
    top_i = np.array([idx.get_loc(pd.Timestamp(r["date"])) for r in turn_rows if r["kind"] == "TOP"], int)
    bot_i = np.array([idx.get_loc(pd.Timestamp(r["date"])) for r in turn_rows if r["kind"] == "BOTTOM"], int)

    def summ(kind):
        r = [x for x in turn_rows if x["kind"] == kind and x["pct"] is not None]
        if not r:
            return {"n": 0}
        pc = np.array([x["pct"] for x in r])
        lv = np.array([x["fii_long"] for x in r])
        return {"n": len(r), "median_pct": float(np.median(pc)), "median_long": float(np.median(lv)),
                "share_hi": float(100 * np.mean(pc >= 80)), "share_lo": float(100 * np.mean(pc <= 20)),
                "share_mid": float(100 * np.mean((pc > 20) & (pc < 80)))}

    valid = np.isfinite(pct)
    allv = np.flatnonzero(valid)
    ch_top, ch_bot = _near(allv, top_i), _near(allv, bot_i)

    def zone(mask, label):
        ep = np.flatnonzero(_episodes(mask & valid, SCEN_COOLDOWN))
        x = fwd[10][ep]
        x = x[np.isfinite(x)]
        return {"label": label, "days": int((mask & valid).sum()), "episodes": len(ep),
                "top_near": _near(ep, top_i), "bottom_near": _near(ep, bot_i),
                "fwd10": float(x.mean()) if len(x) else None,
                "up10": float(100 * (x > 0).mean()) if len(x) else None}

    zone_test = [zone(pct >= 90, "FII heavy LONG (≥ 90th percentile of its year)"),
                 zone(pct <= 10, "FII heavy SHORT (≤ 10th percentile of its year)"),
                 zone((pct > 10) & (pct < 90), "In between")]

    since = idx >= pd.Timestamp("2025-01-01")
    lv = np.where(since, share_pts, np.nan)
    level_test = []
    for lab, m in (("≥ 30%", lv >= 30), ("≥ 25%", lv >= 25), ("≥ 20%", lv >= 20), ("≥ 15%", lv >= 15),
                   ("≤ 12%", lv <= 12), ("≤ 10%", lv <= 10), ("≤ 8%", lv <= 8)):
        m = np.nan_to_num(m, nan=0).astype(bool)
        ep = np.flatnonzero(_episodes(m, SCEN_COOLDOWN))
        x = fwd[10][ep]
        x = x[np.isfinite(x)]
        level_test.append({"level": lab, "days": int(m.sum()), "episodes": len(ep),
                           "top_near": _near(ep, top_i), "bottom_near": _near(ep, bot_i),
                           "fwd10": float(x.mean()) if len(x) else None})
    sv = np.flatnonzero(since & valid)
    return {"at_turns": turn_rows, "tops": summ("TOP"), "bottoms": summ("BOTTOM"),
            "chance_top": ch_top, "chance_bottom": ch_bot, "zone_test": zone_test,
            "level_test": level_test, "level_chance_top": _near(sv, top_i),
            "level_chance_bottom": _near(sv, bot_i)}



# ───────────────────────────────────────────────────────────────────── FII books
# The same swing map, read off two different FII books. Thresholds are per book
# because the two ratios move on different scales: the options bullish share's
# median daily change is ~0.84 pts vs ~1.5 for the futures long share (2018-2026),
# so its "on the sidelines" and "fragile" cut-offs are about half the futures ones.
# Evidence (scripts/study_fii_options_swings.py, repaired data, 2026-11-16):
#   options bullish share: sides with 80% of 3% swings (shuffled-time null 39%);
#     5-day change vs PAST Nifty +0.43, vs NEXT +0.02 (follows price, like futures);
#     heavy-short -> 3% bottom next 1-10 sessions 37% vs 24%, p 0.001 (survives 20-test
#     Bonferroni); 5% bottom 17% vs 11%, p 0.011; heavy-long -> top: nothing (p 0.54).
#   put/call ratio of bought options: sides with 51% vs 42% null -- useless, not shown.
BOOKS = {
    "fut": {"thresholds": {"flat": 1.0, "fragile": 3.0},
            "meta": {"key": "fut", "name": "Index futures",
                     "actions": {"LB": "Long buildup", "SC": "Short covering",
                                 "SB": "Short buildup", "LU": "Long unwinding"},
                     "sides": ("longs", "shorts"),
                     "metric": "FII long %",
                     "definition": "FII longs ÷ (longs + shorts) in index futures",
                     "aside": "short % = 100 − this",
                     "sig": {"short_bottom": "p ≈ 0.02 for 3% bottoms, 0.003 for 5%",
                             "short_ok": True,
                             "long_top": "p ≈ 0.06 — not statistically reliable",
                             "long_ok": False}}},
    "opt": {"thresholds": {"flat": 0.5, "fragile": 1.5},
            "meta": {"key": "opt", "name": "Index options",
                     "actions": {"LB": "Bullish buildup", "SC": "Bearish covering",
                                 "SB": "Bearish buildup", "LU": "Bullish unwinding"},
                     "sides": ("bullish legs (calls bought + puts written)",
                               "bearish legs (calls written + puts bought)"),
                     "metric": "FII options bullish %",
                     "definition": "(calls bought + puts written) ÷ all four FII index-option legs",
                     "aside": "bearish % = 100 − this: calls written + puts bought",
                     "sig": {"short_bottom": "p ≈ 0.001 for 3% bottoms, 0.011 for 5% — the strongest "
                                             "bottom warning of any FII ratio tested",
                             "short_ok": True,
                             "long_top": "p ≈ 0.54 — no warning of tops at all",
                             "long_ok": False}}},
}


def _book(idx, c: pd.Series, series: np.ndarray, net: np.ndarray, L: pd.Series, S: pd.Series,
          client: np.ndarray | None, fwd: dict, flat: float, fragile: float,
          smooth_sides: bool = False) -> dict:
    """Swing legs, leg stats, the open leg, zone lines and peak tests for ONE FII book."""
    legs = _swing_legs(idx, c.ffill().values, series, net, client, LEG_PCT, flat=flat, fragile=fragile)
    sp = pd.Series(series, index=idx)
    pct = _pct_rank(sp).values
    p10, p90 = _roll_quantile(sp, 0.10).values, _roll_quantile(sp, 0.90).values
    current = dict(legs[-1]) if legs and legs[-1]["open"] else None
    if current:
        same = [g for g in legs if g["scenario"] == current["scenario"] and not g["open"] and not g["partial"]]
        if same:
            current["bigger_than_pct"] = float(100 * np.mean([abs(g["move_pct"]) < abs(current["move_pct"])
                                                             for g in same]))
            current["longer_than_pct"] = float(100 * np.mean([g["sessions"] < current["sessions"] for g in same]))
            current["peers"] = len(same)
        mv_now = abs(current["move_pct"])
        far = [abs(g["move_pct"]) for g in legs if not g["open"] and not g["partial"]
               and g["dir"] == current["dir"] and abs(g["move_pct"]) >= mv_now]
        current["reached_n"] = len(far)
        current["extended_pct"] = float(100 * np.mean([m >= mv_now + 3 for m in far])) if far else None
    Lp, Sp = _lot_adjust(L), _lot_adjust(S)
    dl5, ds5, states = _side_flows(Lp, Sp)
    Lz = Lp.rolling(5, min_periods=1).median()
    Sz = Sp.rolling(5, min_periods=1).median()
    pos_turns, side_now = _position_turns(idx, Lz.values, Sz.values)
    level_odds = _level_odds(c.ffill().values, pct)
    return {"legs_all": legs, "leg_stats": _leg_stats(legs), "current": current, "level_odds": level_odds,
            "L": Lp.values, "S": Sp.values, "Lz": Lz.values, "Sz": Sz.values,
            "pos_turns": pos_turns, "side_now": side_now,
            "series": series, "net": net, "pct": pct, "p10": p10, "p90": p90,
            "dl5": dl5, "ds5": ds5, "states": states,
            "turn_actions": _turn_actions(c.ffill().values, states),
            "peaks": _peak_levels(idx, series, pct, legs, fwd)}


def _side_flows(L: pd.Series, S: pd.Series):
    """5-session change of each side as a % of that side's own 60-session average,
    and the action it names (bigger side wins: build / unwind / cover).
    Percent, not contracts: lot sizes changed several times, a percentage of the
    book does not care."""
    dl5 = (L.diff(5) / L.rolling(60, min_periods=20).mean() * 100).values
    ds5 = (S.diff(5) / S.rolling(60, min_periods=20).mean() * 100).values
    states = np.array([_flow_state(a, b) for a, b in zip(dl5, ds5)], dtype=object)
    return dl5, ds5, states


def _turn_actions(close: np.ndarray, states: np.ndarray) -> dict:
    """Share of each action in the 5 sessions ending ON a 3% swing turn and in the
    5 sessions after it, against all days. Descriptive: the turn itself is only
    known in hindsight. (Significance from scripts/study_fii_actions_at_turns.py.)"""
    n = len(close)
    turns = [(i, k) for i, k in _zigzag(close, LEG_PCT) if 10 <= i < n - 6]
    valid = states != "FLAT"
    keys = ("LB", "SB", "LU", "SC")
    out = {"base": {k: float(100 * np.mean(states[valid] == k)) for k in keys} if valid.any() else {}}
    for kind in ("TOP", "BOTTOM"):
        ti = np.array([i for i, k in turns if k == kind], int)
        out[kind] = {"n": int(len(ti))}
        for when, off in (("before", 0), ("after", 5)):
            lab = states[ti + off] if len(ti) else np.array([])
            out[kind][when] = {k: float(100 * np.mean(lab == k)) if len(lab) else None for k in keys}
    return out



# ───────────────────────────────────────────────────────────────────── positions
# Index-derivative lot sizes were cut on 30-Dec-2025 (Nifty 75 -> 65; verified from
# fii_derivatives_stats oi_value / (oi_contracts x spot)). The same FII exposure shows
# ~15% MORE contracts after that date, so a raw line fakes a buildup. Counts before a
# step are multiplied by its factor to express them in today's lot size. (The Nov-2024
# change was phased over months and lies outside the one-year chart; not adjusted.)
LOT_STEPS = [(pd.Timestamp("2025-12-30"), 75 / 65)]
POS_TURN_PCT = 0.30      # a position peak/trough = a 30% reversal of the 5-session median
# (calibrated 2026-11-16: 12% on raw counts caught every monthly-expiry settlement, ~40 turns a
# year; 30% on the 5-session median gives ~8-12 a year per side. About 4 in 10 futures turns and
# 1 in 3 options turns fall within 5 sessions of a Nifty 3% turn, vs ~1 in 4 by chance.)


def _lot_adjust(x: pd.Series) -> pd.Series:
    out = x.astype(float).copy()
    for d, f in LOT_STEPS:
        out[out.index < d] *= f
    return out


def _position_turns(idx, L: np.ndarray, S: np.ndarray) -> tuple[list[dict], dict]:
    """Peaks and troughs of each side's position (12% reversals in contracts), and
    where each side stands now relative to its last turn.

    Measured 2018-2026 (scripts/study_fii_position_peaks.py): these turns land on Nifty's
    swing turns with a median offset of 0 sessions -- about a third before, a quarter the
    same day, the rest after -- so they CONFIRM a turn in real time rather than lead it,
    and a 'position rolled over 10% off its 20-day extreme' rule did not beat chance."""
    turns, now = [], {}
    for side, x in (("L", L), ("S", S)):
        zz = []
        fin = np.flatnonzero(np.isfinite(x))
        if len(fin) < 2:
            continue
        start = int(fin[0])
        d, hi, lo = 0, start, start
        for i in range(start + 1, len(x)):
            if not np.isfinite(x[i]):
                continue
            if x[i] > x[hi]:
                hi = i
            if x[i] < x[lo]:
                lo = i
            last = zz[-1][0] if zz else -1
            if d >= 0 and x[i] <= x[hi] * (1 - POS_TURN_PCT) and hi > last and hi > start:
                zz.append((hi, "PEAK", i)); d, lo = -1, i
            elif d <= 0 and x[i] >= x[lo] * (1 + POS_TURN_PCT) and lo > last and lo > start:
                zz.append((lo, "TROUGH", i)); d, hi = 1, i
        # A turn is only KNOWN on the day the 30% reversal completes ("confirmed"), not on the
        # day it is drawn. Measured 2018-2026: median 7-8 sessions later, by which time Nifty had
        # already moved ~2-3.5% from its own turn -- the chart looks perfect only in hindsight.
        for i, k, ci in zz:
            turns.append({"side": side, "kind": k, "date": idx[i].date(), "value": _f(x[i]),
                          "confirmed": idx[ci].date(), "lag": int(ci - i)})
        if zz and np.isfinite(x[-1]):
            i, k, _ci = zz[-1]
            seg = x[i:]
            ext = int(np.nanargmax(seg) if k == "TROUGH" else np.nanargmin(seg)) + i
            now[side] = {"last_kind": k, "last_date": idx[i].date(), "last_value": _f(x[i]),
                         "now": _f(x[-1]), "chg_pct": _f((x[-1] / x[i] - 1) * 100),
                         "sessions": int(len(x) - 1 - i),
                         "direction": "rising" if k == "TROUGH" else "falling",
                         "extreme_value": _f(x[ext]), "extreme_date": idx[ext].date(),
                         "off_extreme_pct": _f((x[-1] / x[ext] - 1) * 100),
                         # the level that would CONFIRM the current extreme as a new turn
                         "confirm_level": _f(x[ext] * (1 - POS_TURN_PCT) if k == "TROUGH"
                                             else x[ext] * (1 + POS_TURN_PCT))}
    return turns, now



def _level_odds(close: np.ndarray, pct: np.ndarray, band: float = 10.0) -> dict | None:
    """How often a 3% swing top / bottom FOLLOWED (next 1-10 sessions) from days whose
    ratio sat within +-band percentile points of today's, against all days.

    Audit 2026-11-16: the page used to count past TURNS near today's level ("5 tops,
    0 bottoms"). That ignores how long FII spends at each level -- a level visited
    often collects turns just by being common. Rates per episode fix that."""
    today = pct[-1] if len(pct) else np.nan
    valid = np.isfinite(pct)
    if not np.isfinite(today) or valid.sum() < 250:
        return None
    zz = _zigzag(close, LEG_PCT)
    tops = np.array([i for i, k in zz if k == "TOP"], int)
    bots = np.array([i for i, k in zz if k == "BOTTOM"], int)
    lo, hi = max(0.0, today - band), min(100.0, today + band)
    m = valid & (pct >= lo) & (pct <= hi)
    m[-NEAR_AFTER:] = False                     # their outcome is not known yet
    ep = np.flatnonzero(_episodes(m, SCEN_COOLDOWN))
    allv = np.flatnonzero(valid)
    allv = allv[allv < len(close) - NEAR_AFTER]
    return {"band": (round(lo, 1), round(hi, 1)), "episodes": int(len(ep)),
            "top_after": _near(ep, tops), "bottom_after": _near(ep, bots),
            "top_base": _near(allv, tops), "bottom_base": _near(allv, bots)}
