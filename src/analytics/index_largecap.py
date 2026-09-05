"""
Index & Large Cap — who actually moved the index today.

THE QUESTION THIS ANSWERS
    Nifty is free-float cap weighted, so its 50 members do NOT contribute equally.
    The index can close green while most of its own constituents close red, because
    a handful of heavyweights carried it. This module measures that, exactly.

WHAT IS EXACT HERE (no estimation, no weights needed)
    cap-weighted return  = the index itself (index_data.pct_chg)
    equal-weight return  = mean of the constituents' returns
    CARRY SPREAD         = index - equal-weight
        > 0  heavyweights are carrying the index
        < 0  the broad basket is beating the index
    That identity needs no weight data and cannot drift out of calibration.

WHY THERE ARE NO INDEX WEIGHTS IN HERE
    The DB has no free-float or shares-outstanding, and weights are NOT recoverable
    from the index series (scripts/nifty_weight_recovery.py, nifty_weight_levels.py):
      - constrained regression on RETURNS reproduces the index (OOS R2 0.957) but
        puts ICICIBANK at 11.6% against a published 5.26% and drops HDFCBANK, the
        real rank 3, out of the top 20 entirely;
      - non-negative least squares on price LEVELS is worse: rank correlation
        -0.058 with the published order, MARUTI at 20.5% (it selects high-PRICED
        names), and only 4 of 10 top-10 members stable across refits.
    The cause is not a degenerate matrix (condition number 11, effective rank
    31.2/49). It is that there IS no constant weight vector: weights drift with
    relative price, so any window long enough to fit is fitting a moving target.
    The fix is one external snapshot propagated by w_i,t = p_i,t*ff_i / sum(p*ff),
    since ff is constant between reconstitutions -- not implemented yet, so the
    BUCKETS here are membership-based and the CONTRIBUTION column is explicitly
    an equal-weight-within-bucket approximation, never presented as index points.

NO DIRECTIONAL CALL IS MADE, AND THAT IS A MEASURED DECISION
    scripts/nifty_bucket_backtest.py, 1,154 sessions 2022-2026, 48 candidates:
    the only survivor of a date-block reality check was top-10 advance ratio ->
    next-day OVERNIGHT GAP (|t| 3.56 vs null 95th 3.44, p=0.040). It is worth
    ~5.5bps gross against a 2-6bps gap-capture cost, its quintile ladder is
    matched by the REST-30 bucket (-1.1 -> +13.8 bps, also monotone), and it is
    the close-strength/gap edge this codebase already documented -- breadth is
    another proxy for "did today close strong". The user's own framing, the
    top10-minus-rest30 DIVERGENCE, was the weakest variable tested (t +1.76).
    So this tab describes; it does not forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_index_largecap", "IndexLargeCap", "BucketRow",
    "INDEX_BUCKETS", "get_concentration_trend",
]

# ── Bucket membership ─────────────────────────────────────────────────────────
# Ranked by published index weight (Sep 2026). MEMBERSHIP is what this module
# needs -- the exact weights are not used and are not claimed. Top-10 membership
# is stable across rebalances, which is why a bucket read survives without a
# point-in-time weight table; the exact percentages do not.
_NIFTY_TOP10 = ("RELIANCE", "BHARTIARTL", "HDFCBANK", "ICICIBANK", "SBIN",
                "TCS", "BAJFINANCE", "LT", "HINDUNILVR", "INFY")
_NIFTY_NEXT10 = ("SUNPHARMA", "TITAN", "KOTAKBANK", "MARUTI", "ADANIENT",
                 "AXISBANK", "M&M", "ADANIPORTS", "HCLTECH", "ULTRACEMCO")
_NIFTY_REST30 = (
    "APOLLOHOSP", "ASIANPAINT", "BAJAJ-AUTO", "BAJAJFINSV", "BEL", "CIPLA",
    "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HDFCLIFE",
    "HINDALCO", "ITC", "INDIGO", "JSWSTEEL", "JIOFIN", "MAXHEALTH", "NTPC",
    "NESTLEIND", "ONGC", "POWERGRID", "SBILIFE", "SHRIRAMFIN", "TATACONSUM",
    "TMPV", "TATASTEEL", "TECHM", "TRENT", "WIPRO")

# fno_symbol -> (index_data name, display, {bucket label: symbols})
INDEX_BUCKETS: dict[str, dict] = {
    "NIFTY": dict(
        index_name="Nifty 50", display="Nifty 50",
        buckets={"Top 10": _NIFTY_TOP10,
                 "Next 10": _NIFTY_NEXT10,
                 "Rest 30": _NIFTY_REST30},
    ),
}

_DELIV_BASE = 100      # sessions the delivery normal is measured over
_MIN_COVER  = 0.60     # below this share of a bucket present, the row is unreliable


@dataclass
class BucketRow:
    label:        str
    n_members:    int
    n_present:    int
    ret_pct:      Optional[float] = None     # equal-weight return of the bucket
    adv:          int = 0
    dec:          int = 0
    adv_pct:      Optional[float] = None
    deliv_pct:    Optional[float] = None     # equal-weight delivery %
    deliv_z:      Optional[float] = None     # vs the bucket's own 100d normal
    fut_long:     int = 0                    # OI-price matrix counts (near month)
    fut_short:    int = 0
    fut_cover:    int = 0
    fut_unwind:   int = 0
    fut_valid:    int = 0
    fut_oi_pct:   Optional[float] = None
    movers_up:    list = field(default_factory=list)
    movers_dn:    list = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.n_present / self.n_members if self.n_members else 0.0

    @property
    def thin(self) -> bool:
        return self.coverage < _MIN_COVER

    @property
    def fut_lean(self) -> Optional[int]:
        """+1 net fresh-long / -1 net fresh-short / 0 mixed. None when too thin."""
        if self.fut_valid < 4:
            return None
        bull = self.fut_long + self.fut_cover
        bear = self.fut_short + self.fut_unwind
        if bull > bear * 1.5:  return 1
        if bear > bull * 1.5:  return -1
        return 0


@dataclass
class IndexLargeCap:
    trade_date:   date
    fno_symbol:   str
    display:      str
    index_ret:    Optional[float] = None      # the index's own % change
    equal_ret:    Optional[float] = None      # equal-weight constituent mean
    carry_spread: Optional[float] = None      # index - equal-weight
    adv_all:      int = 0
    dec_all:      int = 0
    n_present:    int = 0
    rows:         list = field(default_factory=list)
    data_ok:      bool = True
    note:         str = ""

    @property
    def regime(self) -> str:
        """Plain-language read of TODAY. Descriptive only."""
        if self.carry_spread is None or self.index_ret is None:
            return "No data"
        br = (self.adv_all / self.n_present * 100) if self.n_present else 50.0
        if self.index_ret > 0 and br < 50:
            return "Heavyweights carried a red market"
        if self.index_ret < 0 and br > 50:
            return "Heavyweights dragged a green market"
        if self.carry_spread > 0.15:
            return "Index ahead of its own basket — top-heavy day"
        if self.carry_spread < -0.15:
            return "Basket ahead of the index — broad day"
        return "Index and basket moved together"


def _load_cash(symbols: tuple, trade_date: date, lookback: int = 220) -> pd.DataFrame:
    ph = ", ".join("?" * len(symbols))
    df = query_dataframe(f"""
        SELECT trade_date, symbol, close_price, prev_close, deliv_per, turnover_lacs,
               (close_price - prev_close) / NULLIF(prev_close, 0) * 100 AS r
        FROM daily_data
        WHERE symbol IN ({ph}) AND series IN ('EQ', 'SM', 'ST')
          AND close_price > 0 AND prev_close > 0
          AND trade_date > (?::date - ?) AND trade_date <= ?
    """, [*symbols, trade_date, lookback * 2, trade_date])
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    # NSE price bands make a genuine >40% cash move impossible, so this is a
    # corporate-action filter, not a return filter (same rule as the tilt panel).
    return df[df["r"].abs() < 40]


def _load_futures(symbols: tuple, trade_date: date) -> pd.DataFrame:
    """Near-month stock futures, SAME-EXPIRY matched against the prior session.

    Settlement sessions are dropped: every contract's OI collapses on expiry, which
    would read as a basket-wide 'long unwinding' regardless of direction.
    """
    ph = ", ".join("?" * len(symbols))
    return query_dataframe(f"""
        WITH near AS (
            SELECT trade_date, symbol, MIN(expiry_date) AS exp
            FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK' AND expiry_date > trade_date
              AND open_interest > 0 AND symbol IN ({ph})
              AND trade_date <= ? AND trade_date > (?::date - 20)
            GROUP BY 1, 2
        ),
        f AS (
            SELECT b.trade_date, b.symbol, b.expiry_date, b.close_price,
                   b.open_interest
            FROM fno_bhavcopy b
            INNER JOIN near n ON n.symbol = b.symbol AND n.trade_date = b.trade_date
                             AND n.exp = b.expiry_date
            WHERE b.instrument = 'FUTSTK'
        ),
        settle AS (
            SELECT DISTINCT trade_date, symbol FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK' AND expiry_date = trade_date
        )
        SELECT f.trade_date, f.symbol, f.open_interest, f.close_price,
               LAG(f.open_interest) OVER w AS prev_oi,
               LAG(f.close_price)   OVER w AS prev_px,
               (s.symbol IS NOT NULL) AS is_settle
        FROM f LEFT JOIN settle s ON s.symbol = f.symbol AND s.trade_date = f.trade_date
        WINDOW w AS (PARTITION BY f.symbol, f.expiry_date ORDER BY f.trade_date)
    """, [*symbols, trade_date, trade_date])


def _bucket_row(label: str, members: tuple, hist: pd.DataFrame,
                today: pd.DataFrame, fut_today: pd.DataFrame) -> BucketRow:
    mem = set(members)
    t = today[today["symbol"].isin(mem)]
    row = BucketRow(label=label, n_members=len(members), n_present=len(t))
    if t.empty:
        return row

    row.ret_pct   = float(t["r"].mean())
    row.adv       = int((t["r"] > 0).sum())
    row.dec       = int((t["r"] < 0).sum())
    row.adv_pct   = row.adv / len(t) * 100
    row.deliv_pct = float(t["deliv_per"].dropna().mean()) if t["deliv_per"].notna().any() else None

    # delivery vs the bucket's OWN trailing normal — a bucket of heavyweights has a
    # structurally different delivery level than a bucket of mid-weights, so an
    # absolute % is not comparable across rows; the z-score is.
    h = hist[hist["symbol"].isin(mem)]
    if not h.empty and row.deliv_pct is not None:
        daily = h.groupby("trade_date")["deliv_per"].mean().sort_index()
        prior = daily[daily.index < t["trade_date"].iloc[0]].tail(_DELIV_BASE)
        if len(prior) >= 30 and prior.std() > 1e-9:
            row.deliv_z = float((row.deliv_pct - prior.mean()) / prior.std())

    ft = fut_today[fut_today["symbol"].isin(mem)]
    if not ft.empty:
        for _, fr in ft.iterrows():
            oi, poi = float(fr["open_interest"]), float(fr["prev_oi"])
            px, ppx = float(fr["close_price"]), float(fr["prev_px"])
            if poi <= 0:
                continue
            oi_chg = (oi - poi) / poi * 100
            row.fut_valid += 1
            if   px > ppx and oi_chg >  0.5: row.fut_long   += 1
            elif px < ppx and oi_chg >  0.5: row.fut_short  += 1
            elif px > ppx and oi_chg < -0.5: row.fut_cover  += 1
            elif px < ppx and oi_chg < -0.5: row.fut_unwind += 1
        if row.fut_valid:
            row.fut_oi_pct = float(
                ((ft["open_interest"] - ft["prev_oi"]) / ft["prev_oi"].clip(lower=1) * 100)
                .replace([np.inf, -np.inf], np.nan).dropna().mean())

    s = t.sort_values("r", ascending=False)
    row.movers_up = [(r.symbol, float(r.r)) for r in s.head(3).itertuples() if r.r > 0]
    row.movers_dn = [(r.symbol, float(r.r)) for r in s.tail(3).itertuples() if r.r < 0][::-1]
    return row


def get_index_largecap(trade_date: date, fno_symbol: str = "NIFTY") -> IndexLargeCap:
    """Weight-bucket decomposition of one index for `trade_date`. Causal."""
    meta = INDEX_BUCKETS.get(fno_symbol)
    if meta is None:
        return IndexLargeCap(trade_date=trade_date, fno_symbol=fno_symbol,
                             display=fno_symbol, data_ok=False,
                             note=f"No bucket definition for {fno_symbol}.")
    out = IndexLargeCap(trade_date=trade_date, fno_symbol=fno_symbol,
                        display=meta["display"])
    all_syms = tuple(s for b in meta["buckets"].values() for s in b)

    hist = _load_cash(all_syms, trade_date)
    if hist.empty:
        out.data_ok = False
        out.note = "No constituent data for this date."
        return out
    today = hist[hist["trade_date"] == trade_date]
    if today.empty:
        out.data_ok = False
        out.note = (f"No constituent rows on {trade_date:%d %b %Y} — "
                    f"latest available is {max(hist['trade_date']):%d %b %Y}.")
        return out

    idx = query_dataframe("""
        SELECT pct_chg FROM index_data WHERE index_name = ? AND trade_date = ?
    """, [meta["index_name"], trade_date])
    if not idx.empty and pd.notna(idx["pct_chg"].iloc[0]):
        out.index_ret = float(idx["pct_chg"].iloc[0])
    else:
        # index_data can lag daily_data by a session; say so rather than render a
        # carry spread against a missing index number.
        out.note = (f"{meta['index_name']} has no row for {trade_date:%d %b %Y} in "
                    f"index_data — the carry spread needs it, so it is hidden.")

    out.equal_ret = float(today["r"].mean())
    out.n_present = len(today)
    out.adv_all   = int((today["r"] > 0).sum())
    out.dec_all   = int((today["r"] < 0).sum())
    if out.index_ret is not None:
        out.carry_spread = out.index_ret - out.equal_ret

    fut = _load_futures(all_syms, trade_date)
    if not fut.empty:
        fut["trade_date"] = pd.to_datetime(fut["trade_date"]).dt.date
        fut = fut[(fut["trade_date"] == trade_date) & (~fut["is_settle"].astype(bool))
                  & fut["prev_oi"].notna() & fut["prev_px"].notna()]
    out.rows = [_bucket_row(lbl, mem, hist, today, fut if not fut.empty else pd.DataFrame())
                for lbl, mem in meta["buckets"].items()]
    return out


def get_concentration_trend(fno_symbol: str = "NIFTY", years: int = 5) -> pd.DataFrame:
    """Per-year: how often the index rose while most of its members fell.

    This is the tab's one genuinely measured trend, and it is pure description —
    no forecast is implied. Measured 2022-2026 it rises monotonically
    (2.4% -> 7.2% of sessions) while the carry spread itself SHRINKS, i.e. the
    index is decoupling from its own breadth.
    """
    meta = INDEX_BUCKETS.get(fno_symbol)
    if meta is None:
        return pd.DataFrame()
    all_syms = tuple(s for b in meta["buckets"].values() for s in b)
    ph = ", ".join("?" * len(all_syms))
    start = date.today() - timedelta(days=365 * years + 30)
    df = query_dataframe(f"""
        SELECT trade_date, symbol,
               (close_price - prev_close) / NULLIF(prev_close, 0) * 100 AS r
        FROM daily_data
        WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
          AND close_price > 0 AND prev_close > 0 AND trade_date >= ?
    """, [*all_syms, start])
    idx = query_dataframe("""
        SELECT trade_date, pct_chg FROM index_data
        WHERE index_name = ? AND trade_date >= ?
    """, [meta["index_name"], start])
    if df.empty or idx.empty:
        return pd.DataFrame()
    df = df[df["r"].abs() < 40]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    R = df.pivot_table("r", "trade_date", "symbol").sort_index()
    y = idx.set_index("trade_date")["pct_chg"].astype(float).reindex(R.index)
    cov = R.notna().sum(axis=1)
    keep = cov >= 40                       # need most of the basket for breadth
    R, y = R[keep], y[keep]
    ew = R.mean(axis=1)
    adv = (R > 0).sum(axis=1) / R.notna().sum(axis=1) * 100
    spread = (y - ew).dropna()
    rows = []
    for yr, g in spread.groupby(spread.index.year):
        # A partial year is not comparable: the lookback starts mid-calendar, and
        # a 100-session stub read 14.0% against a 2.4% full year beside it, which
        # looks like a trend reversal and is only a truncated denominator.
        if len(g) < 150:
            continue
        a, yy = adv.reindex(g.index), y.reindex(g.index)
        carried = ((yy > 0) & (a < 50))
        dragged = ((yy < 0) & (a > 50))
        rows.append(dict(year=int(yr), sessions=len(g),
                         abs_spread=round(float(g.abs().mean()), 3),
                         carried_days=int(carried.sum()),
                         carried_pct=round(float(carried.mean() * 100), 1),
                         dragged_days=int(dragged.sum()),
                         dragged_pct=round(float(dragged.mean() * 100), 1)))
    return pd.DataFrame(rows)
