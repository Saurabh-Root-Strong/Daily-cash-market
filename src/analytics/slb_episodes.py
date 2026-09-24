"""
SLB build-up episodes: WHEN did the current borrowing start, how was it added, and
where has the price gone since?

THE ROLL TRAP (why raw outstanding cannot be used for this)
    Loans end on a fixed monthly return date. A borrower who wants to stay short
    borrows next month's contract a few days BEFORE the old one is returned, so for
    ~a week the same short sits in the total twice. Measured, 2026-08:
      INFY   X9 18.3M -> 14.1M (early repay) while XO 1.4M -> 10.9M, then X9 -> 0 on
             the 1-Sep return date: ~10M ROLLED, ~8.6M genuinely returned.
      WIPRO  X8 54.1M -> 0 on 4-Aug with nothing borrowed to replace it: returned.
    On raw totals the INFY roll reads as "fresh borrowing" followed by "covering".

EFFECTIVE (roll-adjusted) OUTSTANDING
    In the last ROLL_WINDOW sessions before each return date (window start values
    N0 = expiring month, L0 = later months):
        new_t     = max(0, later_t - L0)          borrowed in later months since
        repaid_t  = max(0, N0 - near_t)           expiring loan already returned early
        overlap_t = min(max(0, new_t - repaid_t), near_t)
        E_t       = raw_t - overlap_t
    Repay-early-and-re-borrow is a roll with no overlap; borrow-first-return-later
    overlaps until the return date. Outside the window E = raw. INFY Aug-26: E stays
    flat at ~19.5M through the roll week, then drops 8.6M on the return date (the
    unrolled part), instead of a fake +5.5M short and a fake -14M cover.
    What cannot be told apart: a cover and an unrelated new short on the same days
    in the same stock read as a roll. That is a limit of the public file.

EPISODE (a build-up)
    start = the session after the most recent day when E was below START_FRACTION
    (25%) of today's E. "The pile was under a quarter of today's size on <date>;
    everything since is this build-up." Also reported: peak and its date, how far
    E is off the peak (covering), and a FIFO estimate of the price at which the
    still-open borrowed shares were taken ("average borrow price").

SPLITS
    NCL scales SLB positions on a split while older rows stay in old shares, and
    bonus / merger / buyback / demerger FORECLOSE every loan (E collapses to ~0, a
    real reset). Quantities before a split ex-date are multiplied by the split
    factor (and prices divided) so a split never reads as a 5x borrowing spike.

WHAT THE NUMBERS ARE NOT
    Borrowing is not always a directional short: reverse cash-futures arbitrage and
    delivery-shortfall cover borrow too, and they have no "entry price" in the
    short-seller sense. The average borrow price is the close on the days the pile
    grew, not anyone's reported fill.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.analytics.slb_signals import CA_EVENTS_CTE
from src.data.repository import query_dataframe

__all__ = ["effective_oi", "episode_table", "episode_detail", "flag_history",
           "fifo_cost", "episode_start", "phase_of", "roll_adjust", "flag_streaks", "load_timeline",
           "weekly_adds",
           "ROLL_WINDOW", "START_FRACTION", "COVER_OFF_PEAK", "BUILD_5D"]

ROLL_WINDOW = 7          # sessions before a return date in which rolls are matched
START_FRACTION = 0.25    # an episode starts after E was last below 25% of today's E
MIN_SHARES = 1000        # below this a "pile" is noise (1-share prints exist)

_MONTH = {**{f"{m:02d}": m for m in range(1, 13)},
          **{f"X{m}": m for m in range(1, 10)}, "XO": 10, "XN": 11, "XD": 12}


def _expiry_calendar() -> np.ndarray:
    # A handful of one-off contracts carry stray dates (2026-12-08: 3 rows ever). Kept
    # in, they would open a fake roll window in front of themselves.
    cal = query_dataframe(
        "SELECT expiry_date FROM slb_bhavcopy WHERE series_kind <> 'ROLLOVER' "
        "GROUP BY 1 HAVING count(*) >= 10 ORDER BY 1", [])
    return pd.to_datetime(cal["expiry_date"]).values.astype("datetime64[D]")


def _series_expiry(trade_dates: np.ndarray, months: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """Return date of each (trade_date, contract month): the first calendar return
    date in that month on/after the trade date (series codes repeat every year)."""
    out = np.full(len(trade_dates), np.datetime64("NaT"), dtype="datetime64[D]")
    cal_m = pd.DatetimeIndex(cal).month.values
    for m in range(1, 13):
        c = cal[cal_m == m]
        sel = months == m
        if not sel.any() or not len(c):
            continue
        i = np.searchsorted(c, trade_dates[sel], side="left")
        ok = i < len(c)
        tmp = np.full(sel.sum(), np.datetime64("NaT"), dtype="datetime64[D]")
        tmp[ok] = c[i[ok]]
        out[sel] = tmp
    return out


def _closes(symbols: list[str], start: date, end: date) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "trade_date", "close"])
    px = query_dataframe(
        f"""
        SELECT symbol, trade_date, close_price AS close FROM daily_data
        WHERE series = 'EQ' AND close_price > 0 AND trade_date BETWEEN ? AND ?
          AND symbol IN ({', '.join('?' * len(symbols))})
        """, [start, end, *symbols])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    return px


def roll_adjust(R: np.ndarray, N: np.ndarray, win: np.ndarray, nexp: np.ndarray) -> np.ndarray:
    """Roll-adjusted outstanding E from raw R and expiring-month N (sessions x symbols).

    Inside each roll window (the last ROLL_WINDOW sessions before a return date), new
    later-month borrowing is a DUPLICATE of the expiring loan only beyond what was
    already repaid early (repay + re-borrow is a roll with no overlap), and never more
    than the expiring loan still open. Outside a window E = R."""
    R = np.asarray(R, float)
    N = np.asarray(N, float)
    if R.ndim == 1:
        R, N = R[:, None], N[:, None]
    L = R - N
    E = R.copy()
    n = len(win)
    i = 0
    while i < n:
        if not win[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and win[j + 1] and nexp[j + 1] == nexp[i]:
            j += 1
        b = max(i - 1, 0)
        base_later, base_near = L[b], N[b]
        seg = slice(i, j + 1)
        new_later = np.maximum(0.0, L[seg] - base_later)
        repaid = np.maximum(0.0, base_near - N[seg])
        E[seg] = R[seg] - np.minimum(np.maximum(0.0, new_later - repaid), N[seg])
        i = j + 1
    return E


def effective_oi(start: date, end: date, symbols: list[str] | None = None) -> pd.DataFrame:
    """Daily raw and roll-adjusted outstanding per symbol, in today's share units."""
    params: list = [start, end]
    filt = ""
    if symbols:
        filt = f" AND symbol IN ({', '.join('?' * len(symbols))})"
        params += list(symbols)
    rows = query_dataframe(
        f"SELECT trade_date, symbol, series, open_qty FROM slb_open_positions "
        f"WHERE trade_date BETWEEN ? AND ?{filt}", params)
    sess = query_dataframe(
        "SELECT DISTINCT trade_date FROM slb_open_positions WHERE trade_date BETWEEN ? AND ? "
        "ORDER BY 1", [start, end])
    if rows.empty:
        return pd.DataFrame()
    sessions = pd.to_datetime(sess["trade_date"]).values.astype("datetime64[D]")
    cal = _expiry_calendar()

    td = pd.to_datetime(rows["trade_date"]).values.astype("datetime64[D]")
    months = rows["series"].map(_MONTH).fillna(0).astype(int).values
    exp = _series_expiry(td, months, cal)
    # next return date after each session, and how many sessions away it is
    nxt_i = np.searchsorted(cal, sessions, side="right")
    nxt = np.where(nxt_i < len(cal), cal[np.minimum(nxt_i, len(cal) - 1)],
                   np.datetime64("NaT"))
    # Weekdays from each session up to the return date (1 = the session before it).
    # Business days, not positions in `sessions`: the loaded range ends at `end`, and
    # counting positions made every return date past it look 1-3 sessions away.
    has = ~np.isnat(nxt)
    k = np.full(len(sessions), 10 ** 6)
    k[has] = np.busday_count(sessions[has], nxt[has])
    sess_df = pd.DataFrame({"trade_date": sessions, "near_exp": nxt,
                            "in_window": (k >= 1) & (k <= ROLL_WINDOW)})

    rows = rows.assign(trade_date=td, exp=exp)
    rows = rows.merge(sess_df, on="trade_date", how="left")
    rows["is_near"] = rows["exp"] == rows["near_exp"]
    rows["near_q"] = np.where(rows["is_near"], rows["open_qty"], 0)
    agg = (rows.groupby(["symbol", "trade_date"], as_index=False)
               .agg(raw=("open_qty", "sum"), near=("near_q", "sum")))
    agg["trade_date"] = agg["trade_date"].values.astype("datetime64[D]")
    # ── one numpy grid (sessions x symbols) instead of a pandas frame per symbol ──
    syms = sorted(agg["symbol"].unique())
    col = {s: j for j, s in enumerate(syms)}
    row = {d: i for i, d in enumerate(sessions)}
    n, m = len(sessions), len(syms)
    R = np.full((n, m), np.nan)
    N = np.zeros((n, m))
    ii = agg["trade_date"].map(row).values
    jj = agg["symbol"].map(col).values
    R[ii, jj] = agg["raw"].values
    N[ii, jj] = agg["near"].values

    # A stock can vanish from ONE day's file while it is being adjusted (BAJFINANCE on
    # its 2025-06-16 split ex-date: 22,870 -> absent -> 73,950). Read as zero that is a
    # fake "everything returned" that restarts the episode. A single-session hole
    # between two present sessions carries the previous value (restated in today's
    # units below, after the split factor); a longer absence is a real exit.
    absent = np.isnan(R)
    prev_ab = np.vstack([np.ones((1, m), bool), absent[:-1]])
    next_ab = np.vstack([absent[1:], np.ones((1, m), bool)])
    hole = absent & ~prev_ab & ~next_ab
    R_ff = pd.DataFrame(R).ffill().values
    N_ff = pd.DataFrame(np.where(absent, np.nan, N)).ffill().values
    R = np.where(hole, R_ff, np.nan_to_num(R, nan=0.0))
    N = np.where(hole, N_ff, np.where(absent, 0.0, N))

    win = sess_df["in_window"].values
    E = roll_adjust(R, N, win, sess_df["near_exp"].values)

    # split / bonus restatement into today's share units
    F = np.ones((n, m))
    ca = query_dataframe(
        f"""
        SELECT symbol, ex_date, exp(sum(ln(share_factor))) AS f
        FROM corp_actions
        WHERE share_factor > 0 AND share_factor <> 1 AND ex_date <= ?
          AND symbol IN ({', '.join('?' * m)})
        GROUP BY 1, 2
        """, [end, *syms])
    for s_, ex, fac in zip(ca["symbol"], pd.to_datetime(ca["ex_date"]).values.astype("datetime64[D]"),
                           ca["f"]):
        F[sessions < ex, col[s_]] *= fac
    Ra, Ea = R * F, E * F
    for A in (Ra, Ea):                              # holes carry the previous ADJUSTED value
        prev = np.vstack([A[:1], A[:-1]])
        A[hole] = prev[hole]

    df = pd.DataFrame({
        "symbol": np.tile(np.array(syms, dtype=object), n),
        "trade_date": np.repeat(sessions, m),
        "raw": Ra.ravel(), "eff": Ea.ravel(), "near": N.ravel(),
        "in_window": np.repeat(win, m), "F": F.ravel(),
    })
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.merge(_closes(syms, start, end), on=["symbol", "trade_date"], how="left")
    df["close_adj"] = df["close"] / df["F"]
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return df[["symbol", "trade_date", "raw", "eff", "near", "in_window", "close", "close_adj", "F"]]


COVER_OFF_PEAK = -0.30   # "Covering": pile at least 30% below its episode peak
BUILD_5D = 0.05          # "Building": pile grew > 5% of its size in 5 sessions


def fifo_cost(e: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Average price of the borrowed shares still open, for every session.

    Each rise in E is a lot at that day's close; each fall returns the OLDEST shares
    first (loans fall due in the order they were taken). One continuous pass over the
    whole history, so an old pile that was fully returned leaves no trace in a new one.
    A missing close is carried forward rather than dropping the lot, which would make
    the queue disagree with E."""
    from collections import deque
    px = pd.Series(px, dtype=float).ffill().bfill().values
    out = np.full(len(e), np.nan)
    lots: deque = deque()
    q_tot = c_tot = 0.0            # running quantity and cost of the open lots
    prev = 0.0
    for i in range(len(e)):
        x = e[i] if np.isfinite(e[i]) else prev
        d = x - prev
        if d > 0:
            lots.append([d, px[i]])
            q_tot += d
            c_tot += d * px[i]
        elif d < 0:
            rem = -d
            while rem > 1e-9 and lots:
                take = min(rem, lots[0][0])
                lots[0][0] -= take
                q_tot -= take
                c_tot -= take * lots[0][1]
                rem -= take
                if lots[0][0] <= 1e-9:
                    lots.popleft()
        prev = x
        if q_tot > 1e-6 and lots:
            out[i] = c_tot / q_tot
        elif not lots:
            q_tot = c_tot = 0.0    # clear float drift once the pile is empty
    return out


def episode_start(e: np.ndarray, t: int) -> tuple[int, bool]:
    """Index where the build-up live at t began: the session after E was last below
    START_FRACTION of E[t]. (0, False) if it never was inside the loaded window."""
    below = np.nonzero(e[:t] < START_FRACTION * e[t])[0]
    if len(below):
        return int(below[-1]) + 1, True
    return 0, False


def phase_of(now: float, peak: float, five_ago: float) -> str:
    off = now / peak - 1 if peak > 0 else 0.0
    if off <= COVER_OFF_PEAK:
        return "Covering"
    if np.isfinite(five_ago) and now - five_ago > BUILD_5D * now:
        return "Building"
    return "Holding"


def _episode(g: pd.DataFrame) -> dict:
    """Summarise the build-up that is live on g's last session."""
    g = g.sort_values("trade_date")
    e = g["eff"].values.astype(float)
    px = g["close_adj"].values.astype(float)
    dates = g["trade_date"].values
    t = len(e) - 1
    now = e[t]
    if not np.isfinite(now) or now < MIN_SHARES:
        return {}
    s, start_known = episode_start(e, t)
    seg = e[s:t + 1]
    peak_i = s + int(np.nanargmax(seg))
    avg = fifo_cost(e, px)[t]
    p_start, p_now = px[s], px[t]
    five_ago = e[t - 5] if t >= 5 else np.nan
    return {
        "start": pd.Timestamp(dates[s]).date(), "start_known": start_known,
        "age": int(t - s + 1),
        "start_qty": float(e[s]), "now_qty": float(now),
        "peak_qty": float(e[peak_i]), "peak_date": pd.Timestamp(dates[peak_i]).date(),
        "off_peak_pct": (now / e[peak_i] - 1) * 100 if e[peak_i] > 0 else np.nan,
        "price_start": p_start, "price_now": p_now,
        "move_since_pct": (p_now / p_start - 1) * 100 if np.isfinite(p_start) and p_start else np.nan,
        "avg_borrow_px": avg,
        # positive = the price is BELOW the average borrow price: shorts are winning
        "shorts_pnl_pct": (avg / p_now - 1) * 100 if np.isfinite(avg) and p_now else np.nan,
        "phase": phase_of(now, e[peak_i], five_ago),
        "split_inside": bool((g["F"].values[s:] != g["F"].values[-1]).any()),
    }


def episode_table(d: date, lookback_days: int = 400) -> pd.DataFrame:
    """One row per stock with outstanding loans on d: its live build-up episode."""
    live = query_dataframe(
        "SELECT DISTINCT symbol FROM slb_open_positions WHERE trade_date = "
        "(SELECT max(trade_date) FROM slb_open_positions WHERE trade_date <= ?)", [d])
    if live.empty:
        return pd.DataFrame()
    eff = effective_oi(d - timedelta(days=lookback_days), d, live["symbol"].tolist())
    if eff.empty:
        return pd.DataFrame()
    rows = []
    for sym, g in eff.groupby("symbol"):
        r = _episode(g)
        if r:
            r["symbol"] = sym
            rows.append(r)
    return pd.DataFrame(rows)


def episode_detail(symbol: str, d: date, lookback_days: int = 400) -> tuple[pd.DataFrame, dict]:
    """Daily roll-adjusted series for one stock + its live-episode summary."""
    eff = effective_oi(d - timedelta(days=lookback_days), d, [symbol])
    if eff.empty:
        return eff, {}
    return eff, _episode(eff)


def load_timeline() -> dict:
    """Pre-registered timing tests (scripts/build_slb_timeline_study.py)."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "data" / "slb_study" / "timeline.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def weekly_adds(eff: pd.DataFrame, since: date) -> pd.DataFrame:
    """Roll-adjusted shares added (+) or returned (-) per week since `since`, with the
    week's average close: how the pile was built, week by week."""
    g = eff.loc[eff["trade_date"] >= pd.Timestamp(since)].sort_values("trade_date").copy()
    if g.empty:
        return g
    prev = eff.loc[eff["trade_date"] < pd.Timestamp(since), "eff"]
    base = float(prev.iloc[-1]) if len(prev) else 0.0
    g["chg"] = g["eff"].diff().fillna(g["eff"].iloc[0] - base)
    g["week"] = g["trade_date"].dt.to_period("W-FRI").dt.start_time
    w = g.groupby("week").agg(added=("chg", "sum"), avg_close=("close_adj", "mean"),
                              pile_end=("eff", "last"), sessions=("chg", "size"))
    return w.reset_index()


def flag_streaks(d: date, lookback_days: int = 150) -> pd.DataFrame:
    """Per stock flagged on d: the session its CURRENT fresh / heavy streak began (an
    unbroken run of flagged sessions; one unflagged session ends it)."""
    fh = flag_history(d - timedelta(days=lookback_days), d)
    if fh.empty:
        return pd.DataFrame(columns=["symbol", "fresh_since", "heavy_since"])
    sessions = sorted(fh["trade_date"].unique())
    last = sessions[-1]
    out = []
    for sym, g in fh.groupby("symbol"):
        g = g.set_index("trade_date").reindex(sessions)
        row = {"symbol": sym}
        for flag in ("flag_fresh", "flag_heavy"):
            v = g[flag].fillna(False).astype(bool).values
            since = None
            if v[-1]:
                i = len(v) - 1
                while i > 0 and v[i - 1]:
                    i -= 1
                since = pd.Timestamp(sessions[i]).date()
            row[flag.replace("flag_", "") + "_since"] = since
        out.append(row)
    df = pd.DataFrame(out)
    df.attrs["as_of"] = pd.Timestamp(last).date()
    return df


def flag_history(start: date, end: date) -> pd.DataFrame:
    """The live board's two flags, recomputed for every session in [start, end], with
    exactly the board's definition (raw outstanding, top decile among scored names)."""
    df = query_dataframe(
        """
        WITH sess AS (
            SELECT DISTINCT trade_date FROM slb_open_positions WHERE trade_date BETWEEN ? AND ?
        ), oi AS (
            SELECT trade_date, symbol, sum(open_qty) AS oi FROM slb_open_positions
            WHERE trade_date BETWEEN ? AND ? GROUP BY 1, 2
        ), """ + CA_EVENTS_CTE + """, px AS (
            SELECT d.trade_date, d.symbol, coalesce(d.deliv_qty, 0) AS deliv_qty, d.turnover_lacs,
                   CASE WHEN c.symbol IS NULL THEN 0 ELSE 1 END AS ca_day
            FROM daily_data d
            LEFT JOIN cae c ON c.symbol = d.symbol AND c.ex_date = d.trade_date
            WHERE d.series = 'EQ' AND d.close_price > 0 AND d.trade_date BETWEEN ? AND ?
        ), w AS (
            SELECT *,
              avg(deliv_qty) OVER (PARTITION BY symbol ORDER BY trade_date
                                   ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS dq20,
              avg(turnover_lacs) OVER (PARTITION BY symbol ORDER BY trade_date
                                       ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS adv60
            FROM px
        ), w2 AS (
            SELECT *, max(ca_day) OVER (PARTITION BY symbol ORDER BY trade_date
                                        ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS ca_recent,
                   row_number() OVER (PARTITION BY trade_date ORDER BY adv60 DESC NULLS LAST) AS liq_rank
            FROM w
        ), b AS (
            SELECT w2.trade_date, w2.symbol, w2.dq20, w2.liq_rank, w2.ca_recent,
                   coalesce(oi.oi, 0) AS oi
            FROM w2 JOIN sess USING (trade_date) LEFT JOIN oi USING (trade_date, symbol)
        )
        SELECT trade_date, symbol, liq_rank, ca_recent, oi,
               oi / nullif(dq20, 0) AS dtc,
               (oi - lag(oi, 5) OVER (PARTITION BY symbol ORDER BY trade_date)) / nullif(dq20, 0) AS doi5
        FROM b
        """,
        [start - timedelta(days=200), end] * 3,
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.loc[(df["trade_date"] >= pd.Timestamp(start)) & (df["oi"] > 0)]
    sc = df.loc[(df["liq_rank"] <= 300) & (df["ca_recent"] == 0)].copy()
    g = sc.groupby("trade_date")
    sc["flag_fresh"] = g["doi5"].rank(pct=True) > 0.9
    sc["flag_heavy"] = g["dtc"].rank(pct=True) > 0.9
    return sc[["trade_date", "symbol", "flag_fresh", "flag_heavy", "dtc", "doi5"]]
