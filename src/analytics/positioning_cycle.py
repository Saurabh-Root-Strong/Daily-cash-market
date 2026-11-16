"""
Positioning Cycle — where each F&O participant stands, and what that state has
actually meant at past Nifty swing turns.

Built 2026-11-14 from scripts/study_fii_turning_points.py. The question it answers
is "which FII ratio marks the peak / trough before the market turns?", and the
measured answer shapes everything this module returns:

  * Positioning is a MIRROR of price, not a lead. The 5-session change in FII
    index-futures long share correlates +0.43 with the PAST 5 sessions of Nifty and
    +0.02 with the NEXT 5. Around 51 swing turns the ratio reaches its extreme ON
    the turn day, not before it.
  * At swing BOTTOMS FIIs are reliably at an extreme short (two thirds of 8% bottoms
    sit in the lowest fifth of the trailing year). So "FII extreme short" roughly
    doubles the odds that a bottom is close -- but most extreme-short days are still
    not a bottom. It is a ZONE, not a trigger.
  * Tops are far weaker than bottoms, and the "they turned before price" divergence
    pattern does not beat its sibling cell for any participant.

So this module reports a state and the measured odds attached to it. It never
returns an arrow.

Guards (each one is a defect found in earlier positioning work in this repo):
  * RATIOS ONLY. The Nifty lot went 50 -> 75 -> 25 -> 50 -> 65 -> 75; contract counts
    are not comparable across time. FII long share itself moved from a ~45% median
    (2018-24) to ~11% (2026), so no fixed threshold ("below 30% = bearish") can work.
    Every level is read as a percentile of its own trailing 250 sessions.
  * REAL SESSIONS ONLY. Until the 2026-11-16 repair (scripts/repair_participant_copies.py)
    fao_participant held 298 days whose file belonged to ANOTHER date with the same
    day-of-month, 82 of them dated on market holidays. The table is clean now and the
    fetchers reject wrong-date files, but rows are still kept only where Nifty has a
    session, as a second guard.
  * PUBLICATION LAG. The file publishes after the close, so a reading dated D is
    usable at the D+1 open. Forward returns are measured from that open.
  * EPISODES, not days. A state that persists for 30 days is one observation.
  * NO LOOK-AHEAD. Everything is computed from rows dated <= as_of at call time;
    nothing is a frozen constant except the audit block, which is labelled as such.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_positioning_cycle", "ZONES", "STUDY_AUDIT"]

PCT_WINDOW = 250          # sessions in the trailing percentile
PCT_MIN = 120             # fewer than this -> no percentile
TURN_PCT = 0.05           # a swing turn = close extreme followed by a 5% reversal
# "A turn is coming" = a confirmed turn from the NEXT session to 10 sessions ahead.
# Audit 2026-11-16: the earlier window started 3 sessions BEFORE the signal, so a turn
# that had already happened was counted as "came within 10 sessions" — it inflated
# the headline (heavy-short -> bottom 51% vs 32%) to roughly 35% vs 24% forward-only.
# The file publishes after the close, so +1 is the first session anyone could act in.
NEAR_FROM, NEAR_AFTER = 1, 10
NEAR_BEFORE = -NEAR_FROM           # kept for importers; negative = window starts after the signal
# NSE's lot-size reset for index derivatives (new contracts from 20-Nov-2024) moved FII
# long share from a ~45% median to ~11%. A trailing-year percentile that straddles the
# break ranks every post-break day against the old era: in 2025 FII read "heavy short"
# on 35% of days and "heavy long" on 0%. Percentiles restart at each break.
REGIME_BREAKS = (pd.Timestamp("2024-11-20"),)
EP_COOLDOWN = 10          # sessions between two episodes of the same state
MIN_EPISODES = 12         # fewer than this -> the cell is shown but not judged
HORIZONS = (10, 20)

ZONES = [  # (key, label, lower pct exclusive, upper pct inclusive)
    ("xs", "EXTREME SHORT", -1.0, 10.0),
    ("s", "SHORT", 10.0, 30.0),
    ("m", "MIDDLE", 30.0, 70.0),
    ("l", "LONG", 70.0, 90.0),
    ("xl", "EXTREME LONG", 90.0, 100.0),
]

PLAYERS = [  # (client_type in the file, display name, role)
    ("FII", "FII", "Foreign institutions — the largest directional book"),
    ("Pro", "Pro", "Proprietary desks — mostly hedging option books"),
    ("Client", "Client", "Retail and HNI — the arithmetic mirror of FII"),
    ("DII", "DII", "Domestic institutions — hedgers, short futures by mandate"),
]

# Frozen audit numbers from the 2018-2026 study (2,192 sessions, 25 tops / 26 bottoms
# at 5%). These are NOT used to classify anything; they are what the page cites for
# the tests that were run once and failed.
STUDY_AUDIT = {
    "measured_on": "2026-11-16 (after the copied-file repair)",
    "sessions": 2193,
    "lead_lag": {"fii_vs_past5": 0.479, "fii_vs_next5": 0.017,
                 "client_vs_past5": -0.386, "client_vs_next5": 0.018},
    "rule_family": {"cells": 96, "best_t": 3.13, "null_median": 2.49,
                    "null_95": 3.35, "p": 0.101},
    "divergence": {"fii_bull_n": 22, "fii_bull_fwd10": 0.58, "sibling_n": 55,
                   "sibling_fwd10": 1.03, "t": -0.39, "best_abs_t_any_player": 1.38},
    "matrix": {"cells": 20, "best_t": 1.94, "needed_t": 3.0},
    "copied_days_repaired": 298,
    "expiry_day_abs_change_pp": {"expiry": 6.91, "other": 1.52},
    "event_profile_bottoms": {"m20": 35, "m5": 31, "day": 21},
    # forward-only (1..10 sessions), regime-aware percentiles, circular-shift permutation
    "zone_significance": {"short_bottom_3pct_p": 0.018, "short_bottom_5pct_p": 0.003,
                          "long_top_3pct_p": 0.056, "long_top_5pct_p": 0.084},
    "old_engine_hit_250": 37.6,
}

# ───────────────────────────────────────────────────────────────────── helpers
def _segments(s: pd.Series):
    """Split a date-indexed series at REGIME_BREAKS (no-op for non-date indexes)."""
    if not isinstance(s.index, pd.DatetimeIndex) or s.empty:
        yield s
        return
    edges = [b for b in REGIME_BREAKS if s.index[0] < b <= s.index[-1]]
    start = s.index[0]
    for b in edges:
        yield s[(s.index >= start) & (s.index < b)]
        start = b
    yield s[s.index >= start]


def _pct_rank(s: pd.Series) -> pd.Series:
    """Causal percentile: where today's value sits within its own trailing window,
    never reaching back across a regime break (see REGIME_BREAKS)."""
    parts = [seg.rolling(PCT_WINDOW, min_periods=PCT_MIN).apply(
        lambda x: (x[-1] >= x).mean() * 100.0, raw=True) for seg in _segments(s)]
    return pd.concat(parts).reindex(s.index) if parts else s * np.nan


def _roll_quantile(s: pd.Series, q: float) -> pd.Series:
    """Trailing-year quantile of a series, restarted at regime breaks."""
    parts = [seg.rolling(PCT_WINDOW, min_periods=PCT_MIN).quantile(q) for seg in _segments(s)]
    return pd.concat(parts).reindex(s.index) if parts else s * np.nan


def _zigzag(close: np.ndarray, thr: float) -> list[tuple[int, str]]:
    """Swing turns on closes. A turn is only emitted once price has reversed by
    `thr` from it, so the list never contains a turn that as_of could not confirm."""
    turns: list[tuple[int, str]] = []
    if len(close) < 2:
        return turns
    direction, hi_i, lo_i = 0, 0, 0
    for i in range(1, len(close)):
        if close[i] > close[hi_i]:
            hi_i = i
        if close[i] < close[lo_i]:
            lo_i = i
        last = turns[-1][0] if turns else -1
        # index 0 is never a turn: nothing before it, so it cannot be a confirmed high/low
        if direction >= 0 and close[i] <= close[hi_i] * (1 - thr) and hi_i > last and hi_i > 0:
            turns.append((hi_i, "TOP"))
            direction, lo_i = -1, i
        elif direction <= 0 and close[i] >= close[lo_i] * (1 + thr) and lo_i > last and lo_i > 0:
            turns.append((lo_i, "BOTTOM"))
            direction, hi_i = 1, i
    return turns


def _episodes(mask: np.ndarray, cooldown: int = EP_COOLDOWN) -> np.ndarray:
    out = np.zeros(len(mask), bool)
    last = -10 ** 9
    for i in np.flatnonzero(mask):
        if i - last >= cooldown:
            out[i] = True
            last = i
    return out


def _near(sig_i: np.ndarray, turn_i: np.ndarray) -> float | None:
    """Share of signal positions followed by a turn NEAR_FROM..NEAR_AFTER sessions later."""
    if len(sig_i) == 0 or len(turn_i) == 0:
        return None
    d = turn_i[None, :] - sig_i[:, None]
    return float(100 * ((d >= NEAR_FROM) & (d <= NEAR_AFTER)).any(axis=1).mean())


def _zone_of(pct: float | None) -> str | None:
    if pct is None or not np.isfinite(pct):
        return None
    for key, _, lo, hi in ZONES:
        if lo < pct <= hi:
            return key
    return "xl" if pct > 100 else "xs"


def _f(v) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


# ───────────────────────────────────────────────────────────────────── loading
def data_version() -> tuple:
    """Cheap fingerprint of the participant table. Part of every cache key, so a
    long-running dashboard picks up tonight's file (or a repair) without a restart —
    a plain lru_cache keyed on the date would serve the old frame forever."""
    try:
        v = query_dataframe("SELECT MAX(trade_date) m, COUNT(*) n, SUM(fut_idx_long) s FROM fao_participant")
        return (str(v["m"].iloc[0]), int(v["n"].iloc[0]), int(v["s"].iloc[0] or 0))
    except Exception:                                       # noqa: BLE001
        return ("?",)


@lru_cache(maxsize=32)
def _frame_cached(as_of: date, _version: tuple = ()):
    part = query_dataframe(
        """SELECT trade_date, client_type,
                  fut_idx_long, fut_idx_short,
                  opt_idx_call_long, opt_idx_call_short,
                  opt_idx_put_long,  opt_idx_put_short
           FROM fao_participant
           WHERE upper(data_type) = 'OI' AND trade_date <= ?""", [as_of])
    px = query_dataframe(
        """SELECT trade_date, open_val, close_val FROM index_data
           WHERE index_name = 'Nifty 50' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if part.empty or px.empty:
        return None
    part["trade_date"] = pd.to_datetime(part["trade_date"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.drop_duplicates("trade_date").set_index("trade_date").sort_index().astype(float)

    cols: dict[str, pd.Series] = {}
    fut_long_sum = fut_short_sum = None
    for ct, g in part.groupby("client_type"):
        k = str(ct).strip().lower()
        g = (g.drop(columns="client_type").drop_duplicates("trade_date")
              .set_index("trade_date").sort_index().astype(float))
        fl, fs = g["fut_idx_long"], g["fut_idx_short"]
        cl_, cs = g["opt_idx_call_long"], g["opt_idx_call_short"]
        pl, ps = g["opt_idx_put_long"], g["opt_idx_put_short"]
        cols[f"{k}_long"], cols[f"{k}_short"] = fl, fs
        cols[f"{k}_ls"] = fl / (fl + fs).replace(0, np.nan)
        cols[f"{k}_opt"] = (cl_ + ps) / (cl_ + cs + pl + ps).replace(0, np.nan)
        fut_long_sum = fl if fut_long_sum is None else fut_long_sum.add(fl, fill_value=0)
        fut_short_sum = fs if fut_short_sum is None else fut_short_sum.add(fs, fill_value=0)
    F = pd.DataFrame(cols).sort_index()
    F["_sum_long"], F["_sum_short"] = fut_long_sum, fut_short_sum

    raw_rows = len(F)
    F = F[F.index.isin(px.index)]                 # real sessions only (holiday rows are junk)
    holiday_rows = raw_rows - len(F)
    missing = px.index[px.index >= F.index.min()].difference(F.index) if len(F) else px.index[:0]
    return F, px, holiday_rows, missing


def _frame(as_of: date):
    got = _frame_cached(as_of, data_version())
    if got is None:
        return None
    F, px, hol, missing = got
    return F.copy(), px.copy(), hol, missing


def expiry_calendar(as_of: date, sessions: pd.DatetimeIndex | None = None) -> set:
    """Monthly Nifty futures expiry dates <= as_of.

    fno_bhavcopy only starts in Jan-2022, so before its first expiry the dates are
    rebuilt by NSE's rule of that era: last Thursday of the month, moved back to the
    previous trading session when it is a holiday. (Audit 2026-11-16: without this,
    ~47 expiries in 2018-21 were neither flagged nor excluded.)"""
    out: set = set()
    first = None
    try:
        ex = query_dataframe(
            """SELECT DISTINCT expiry_date FROM fno_bhavcopy
               WHERE symbol = 'NIFTY' AND instrument = 'FUTIDX' AND expiry_date <= ?""", [as_of])
        if not ex.empty:
            e = pd.to_datetime(ex["expiry_date"])
            out |= set(e.dt.date)
            first = e.min()
    except Exception:                                       # noqa: BLE001
        pass
    if sessions is None:
        try:
            sessions = pd.DatetimeIndex(pd.to_datetime(query_dataframe(
                "SELECT trade_date FROM index_data WHERE index_name = 'Nifty 50' AND trade_date <= ?",
                [as_of])["trade_date"]))
        except Exception:                                   # noqa: BLE001
            return out
    if len(sessions) == 0:
        return out
    sess = sessions.sort_values()
    stop = first if first is not None else pd.Timestamp(as_of) + pd.Timedelta(days=1)
    for m in pd.period_range(sess[0], stop, freq="M"):
        last_day = m.to_timestamp(how="end").normalize()
        thu = last_day - pd.Timedelta(days=(last_day.weekday() - 3) % 7)
        cand = sess[sess <= thu]
        if len(cand) and cand[-1] >= m.to_timestamp() and cand[-1] < stop:
            out.add(cand[-1].date())
    return out


def _expiry_dates(as_of: date) -> set:
    return expiry_calendar(as_of)


# ───────────────────────────────────────────────────────────────────── measurement
def _measure(mask: np.ndarray, fwd: dict[int, np.ndarray], base: dict[int, float],
             bot_i: np.ndarray, top_i: np.ndarray) -> dict:
    """What followed a state, counted as episodes, against the unconditional base."""
    ep = _episodes(mask)
    ii = np.flatnonzero(ep)
    out: dict = {"days": int(mask.sum()), "episodes": int(len(ii)),
                 "bottom_near": _near(ii, bot_i), "top_near": _near(ii, top_i)}
    for h in HORIZONS:
        x = fwd[h][ep]
        x = x[~np.isnan(x)]
        if len(x) >= 3:
            sd = x.std(ddof=1)
            out[f"fwd{h}"] = float(x.mean())
            out[f"up{h}"] = float(100 * (x > 0).mean())
            out[f"excess{h}"] = float(x.mean() - base[h])
            out[f"t{h}"] = float((x.mean() - base[h]) / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0
            out[f"n{h}"] = int(len(x))
        else:
            out[f"fwd{h}"] = out[f"up{h}"] = out[f"excess{h}"] = out[f"t{h}"] = None
            out[f"n{h}"] = int(len(x))
    out["judged"] = out["episodes"] >= MIN_EPISODES
    return out


def get_positioning_cycle(as_of: date) -> dict:
    """State of all four participants on the latest session <= as_of, with the
    measured history of that state. Returns {"error": ...} instead of raising."""
    try:
        got = _frame(as_of)
    except Exception as exc:                                # noqa: BLE001
        return {"error": f"Could not read participant data: {exc}"}
    if got is None:
        return {"error": "No participant or Nifty data on or before that date."}
    F, px, holiday_rows, missing = got
    if F.empty:
        return {"error": "No participant rows fall on a Nifty session on or before that date."}

    idx = F.index
    data_date = idx[-1]
    warnings: list[str] = []

    # ---- freshness / gaps ---------------------------------------------------
    sessions_after = int((px.index > data_date).sum())
    if sessions_after:
        warnings.append(
            f"Participant data stops at {data_date:%d %b %Y}; Nifty has {sessions_after} "
            f"later session(s). Everything below describes {data_date:%d %b}, not the selected date.")
    prev_date = idx[-2] if len(idx) > 1 else None
    gap_sessions = 0
    if prev_date is not None:
        gap_sessions = int(((px.index > prev_date) & (px.index < data_date)).sum())
        if gap_sessions:
            warnings.append(
                f"{gap_sessions} session(s) between {prev_date:%d %b} and {data_date:%d %b} have no "
                "participant file, so the '1-day' change spans more than one day.")
    if holiday_rows:
        warnings.append(
            f"{holiday_rows} participant row(s) dated on market holidays were ignored — "
            "their contents do not match the sessions on either side.")

    # ---- zero-sum integrity -------------------------------------------------
    sl, ss = F["_sum_long"].iloc[-1], F["_sum_short"].iloc[-1]
    balanced = bool(np.isfinite(sl) and np.isfinite(ss) and sl > 0 and abs(sl - ss) / sl < 0.005)
    n_types = int(sum(f"{p[0].lower()}_ls" in F.columns and np.isfinite(F[f"{p[0].lower()}_ls"].iloc[-1])
                      for p in PLAYERS))
    if n_types < 4:
        warnings.append(f"Only {n_types} of 4 participant categories have a row on {data_date:%d %b %Y}.")
    elif not balanced:
        warnings.append(
            "Index-futures longs and shorts across the four categories do not match on this date "
            f"({sl:,.0f} vs {ss:,.0f}) — the file is incomplete or corrupt; treat the reading as unreliable.")

    # ---- expiry -------------------------------------------------------------
    exp = _expiry_dates(as_of)
    expiry_day = data_date.date() in exp
    post_expiry = prev_date is not None and prev_date.date() in exp
    if expiry_day or post_expiry:
        warnings.append(
            ("Monthly index-futures expiry day: " if expiry_day else "First session after monthly expiry: ")
            + "expiring contracts leave the book, so the long share typically jumps "
            f"{STUDY_AUDIT['expiry_day_abs_change_pp']['expiry']:.1f} points on its own "
            f"(against {STUDY_AUDIT['expiry_day_abs_change_pp']['other']:.1f} on a normal day). "
            "Do not read today's change as a decision.")

    # ---- players ------------------------------------------------------------
    have_pct = len(idx) >= PCT_MIN
    if not have_pct:
        warnings.append(
            f"Only {len(idx)} sessions of history — a percentile needs {PCT_MIN}. "
            "Values are shown; zones and odds are not.")
    P: dict[str, pd.Series] = {}
    players = []
    for ct, name, role in PLAYERS:
        k = ct.lower()
        if f"{k}_ls" not in F.columns:
            continue
        ls = F[f"{k}_ls"]
        P[f"{k}_ls"] = _pct_rank(ls) if have_pct else pd.Series(np.nan, index=idx)
        opt = F[f"{k}_opt"]
        P[f"{k}_opt"] = _pct_rank(opt) if have_pct else pd.Series(np.nan, index=idx)
        long_, short_ = F[f"{k}_long"].iloc[-1], F[f"{k}_short"].iloc[-1]
        thin = bool(np.isfinite(long_ + short_) and (long_ + short_) <= 0)
        players.append({
            "key": k, "name": name, "role": role,
            "long": _f(long_), "short": _f(short_),
            "long_share": _f(ls.iloc[-1] * 100),
            "ls_ratio": _f(long_ / short_) if short_ and short_ > 0 else None,
            "pct": _f(P[f"{k}_ls"].iloc[-1]),
            "zone": _zone_of(_f(P[f"{k}_ls"].iloc[-1])),
            "chg1": _f((ls.iloc[-1] - ls.iloc[-2]) * 100) if len(ls) > 1 else None,
            "chg5": _f((ls.iloc[-1] - ls.iloc[-6]) * 100) if len(ls) > 5 else None,
            "chg20": _f((ls.iloc[-1] - ls.iloc[-21]) * 100) if len(ls) > 20 else None,
            "min250": _f(ls.tail(PCT_WINDOW).min() * 100), "max250": _f(ls.tail(PCT_WINDOW).max() * 100),
            "med250": _f(ls.tail(PCT_WINDOW).median() * 100),
            "opt_bull_share": None if k == "dii" else _f(opt.iloc[-1] * 100),
            "opt_pct": None if k == "dii" else _f(P[f"{k}_opt"].iloc[-1]),
            "no_position": thin,
        })

    fii = next((p for p in players if p["key"] == "fii"), None)
    client = next((p for p in players if p["key"] == "client"), None)
    if fii is None or fii["long_share"] is None:
        return {"error": f"No FII index-futures position on {data_date:%d %b %Y}.",
                "data_date": data_date.date(), "warnings": warnings}

    # ---- price, forward returns, turns (all from data <= as_of) -------------
    cl, op = px["close_val"], px["open_val"]
    close_f = cl.reindex(idx)
    fwd = {h: ((cl.shift(-h) / op.shift(-1) - 1) * 100).reindex(idx).values for h in HORIZONS}
    base = {h: float(np.nanmean(fwd[h])) if np.isfinite(fwd[h]).any() else 0.0 for h in HORIZONS}
    base_up = {h: float(100 * np.nanmean(fwd[h][np.isfinite(fwd[h])] > 0))
               if np.isfinite(fwd[h]).any() else None for h in HORIZONS}

    zz = _zigzag(close_f.ffill().values, TURN_PCT)
    top_i = np.array([i for i, k in zz if k == "TOP"], dtype=int)
    bot_i = np.array([i for i, k in zz if k == "BOTTOM"], dtype=int)
    all_i = np.arange(len(idx))
    # chance is measured only where a percentile exists, i.e. on comparable days
    valid = np.isfinite(P["fii_ls"].values)
    chance_bot = _near(all_i[valid], bot_i) if valid.any() else None
    chance_top = _near(all_i[valid], top_i) if valid.any() else None

    # ---- zone table for FII long share -------------------------------------
    fp = P["fii_ls"].values
    zone_rows = []
    for key, label, lo, hi in ZONES:
        m = valid & (fp > lo) & (fp <= hi)
        row = _measure(m, fwd, base, bot_i, top_i)
        row.update({"key": key, "label": label, "lo": max(lo, 0.0), "hi": hi,
                    "today": key == fii["zone"]})
        zone_rows.append(row)

    # ---- full scenario matrix: zone x 5-session change x price trend --------
    d5 = F["fii_ls"].diff(5).values
    above = (close_f > close_f.rolling(20).mean()).values
    have_trend = np.isfinite(close_f.rolling(20).mean().values)
    today_cell = None
    matrix = []
    for key, label, lo, hi in ZONES:
        for dk, dlabel, dm in (("L", "adding longs", d5 > 0), ("S", "adding shorts", d5 <= 0)):
            for pk, plabel, pm in (("A", "above 20-DMA", above), ("B", "below 20-DMA", ~above)):
                m = valid & have_trend & np.isfinite(d5) & (fp > lo) & (fp <= hi) & dm & pm
                row = _measure(m, fwd, base, bot_i, top_i)
                is_today = bool(m[-1]) if len(m) else False
                row.update({"zone": label, "flow": dlabel, "price": plabel, "today": is_today})
                if is_today:
                    today_cell = row
                matrix.append(row)

    # ---- how the two readings behave around turns (live lead/lag) -----------
    def _spear(a: pd.Series, b: pd.Series) -> float | None:
        j = pd.concat([a, b], axis=1).dropna()
        if len(j) < 60:
            return None
        return _f(j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank()))
    past5 = (cl / cl.shift(5) - 1).reindex(idx)
    next5 = (cl.shift(-5) / op.shift(-1) - 1).reindex(idx)
    lead_lag = {
        "fii_vs_past5": _spear(F["fii_ls"].diff(5), past5),
        "fii_vs_next5": _spear(F["fii_ls"].diff(5), next5),
        "fii_client_corr": _spear(F["fii_ls"].diff(), F["client_ls"].diff())
        if "client_ls" in F.columns else None,
    }

    # ---- turn ledger --------------------------------------------------------
    ledger = []
    for i, kind in zz:
        d = idx[i]
        nxt = [j for j, _ in zz if j > i]
        end_i = nxt[0] if nxt else len(idx) - 1
        move = (close_f.iloc[end_i] / close_f.iloc[i] - 1) * 100
        ledger.append({
            "date": d.date(), "kind": kind, "close": _f(close_f.iloc[i]),
            "fii_ls": _f(F["fii_ls"].iloc[i] * 100), "fii_pct": _f(P["fii_ls"].iloc[i]),
            "client_pct": _f(P["client_ls"].iloc[i]) if "client_ls" in P else None,
            "pro_pct": _f(P["pro_ls"].iloc[i]) if "pro_ls" in P else None,
            "fii_opt_pct": _f(P["fii_opt"].iloc[i]),
            "next_move_pct": _f(move), "open": not nxt,
        })
    judged = [r for r in ledger if r["fii_pct"] is not None]
    bots = [r for r in judged if r["kind"] == "BOTTOM"]
    tops = [r for r in judged if r["kind"] == "TOP"]
    turn_summary = {
        "bottoms": len(bots), "tops": len(tops),
        "bottoms_fii_low20": sum(r["fii_pct"] <= 20 for r in bots),
        "tops_fii_high80": sum(r["fii_pct"] >= 80 for r in tops),
        "bottoms_mean_pct": _f(np.mean([r["fii_pct"] for r in bots])) if bots else None,
        "tops_mean_pct": _f(np.mean([r["fii_pct"] for r in tops])) if tops else None,
    }

    # ---- since the last confirmed turn -------------------------------------
    swing = None
    if zz:
        li, lk = zz[-1]
        seg = close_f.iloc[li:]
        ext_i = li + int(np.argmax(seg.values) if lk == "BOTTOM" else np.argmin(seg.values))
        swing = {
            "last_kind": lk, "last_date": idx[li].date(), "last_close": _f(close_f.iloc[li]),
            "move_pct": _f((close_f.iloc[-1] / close_f.iloc[li] - 1) * 100),
            "sessions": int(len(idx) - 1 - li),
            "extreme_close": _f(close_f.iloc[ext_i]), "extreme_date": idx[ext_i].date(),
            "off_extreme_pct": _f((close_f.iloc[-1] / close_f.iloc[ext_i] - 1) * 100),
            "reversal_needed_pct": TURN_PCT * 100,
            "fii_ls_then": _f(F["fii_ls"].iloc[li] * 100), "fii_pct_then": _f(P["fii_ls"].iloc[li]),
        }

    tail = slice(-PCT_WINDOW, None)
    series = pd.DataFrame({
        "date": idx[tail], "close": close_f.values[tail],
        "fii_ls": (F["fii_ls"].values * 100)[tail], "fii_pct": P["fii_ls"].values[tail],
        "client_ls": (F["client_ls"].values * 100)[tail] if "client_ls" in F.columns else np.nan,
        "client_pct": P["client_ls"].values[tail] if "client_ls" in P else np.nan,
    })
    lo_date = idx[tail][0]
    marks = [{"date": idx[i], "kind": k, "close": float(close_f.iloc[i])}
             for i, k in zz if idx[i] >= lo_date]

    today_zone = next((z for z in zone_rows if z["today"]), None)
    return {
        "as_of": as_of, "data_date": data_date.date(),
        "prev_date": prev_date.date() if prev_date is not None else None,
        "nifty_close": _f(close_f.iloc[-1]),
        "sessions": int(len(idx)), "history_from": idx[0].date(),
        "have_pct": bool(have_pct and fii["pct"] is not None),
        "players": players, "fii": fii, "client": client,
        "gap_fii_minus_client": _f(fii["long_share"] - client["long_share"])
        if client and client["long_share"] is not None else None,
        "zone_rows": zone_rows, "today_zone": today_zone,
        "matrix": matrix, "today_cell": today_cell,
        "base": base, "base_up": base_up,
        "chance_bottom": chance_bot, "chance_top": chance_top,
        "lead_lag": lead_lag, "ledger": ledger, "turn_summary": turn_summary,
        "swing": swing, "series": series, "marks": marks,
        "expiry_day": bool(expiry_day), "post_expiry": bool(post_expiry),
        "gap_sessions": gap_sessions, "holiday_rows_ignored": int(holiday_rows),
        "balanced": balanced, "warnings": warnings,
        "params": {"pct_window": PCT_WINDOW, "turn_pct": TURN_PCT * 100,
                   "near_from": NEAR_FROM, "near_after": NEAR_AFTER,
                   "cooldown": EP_COOLDOWN, "min_episodes": MIN_EPISODES},
        "audit": STUDY_AUDIT,
    }
