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
    Audited: corr(spread, number of names present) = +0.010, so the spread measures
    concentration and not coverage (scripts/ilc_self_audit.py).

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

SELF-AUDIT OF THIS MODULE (scripts/ilc_self_audit.py) -- three defects found and
fixed after the first ship, one claim retracted:
    - near-month futures OI read basket-wide UNWINDING every roll week (DTE 0-2
      mean -47.9%, 99.94% of rows below -5%; 13.7% of sessions). Now TOTAL forward
      OI, which is flat at +0.45..0.67% across every DTE bucket; only the expiry
      SESSION itself still jumps (+14.25% vs +0.47%, 5.0% of days) and is suppressed.
    - the Rest-30 delivery mean had corr +0.674 with HOW MANY of its names reported
      that day. Now a per-symbol z averaged across the bucket, which is immune.
    - the raw OI mean differed from a winsorised one by up to 84.5pp on a single
      Top-10 day (one contract printed +895%). Now clipped to +-50%.
    - the concentration "trend" does not replicate on a membership-independent
      basket -- see get_concentration_trend. Claim retracted.
Checked and clean: the carry-spread identity (no coverage leak), and the
head(3)/tail(3) mover lists (buckets never fall below 9/9/25 names, so a stock can
never appear as both a top gainer and a top loser).
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
    # True on a monthly settlement session, where forward OI jumps mechanically as
    # the front contract settles and positions roll in. The futures columns are
    # suppressed rather than shown as a basket-wide build.
    is_expiry_session: bool = False

    @property
    def state(self) -> Optional[str]:
        """Which of the four top-10 / rest-30 states today falls in.

        Shares its label constants with get_state_base_rates(), so the live read
        and the historical table can never fall out of sync.
        """
        by = {r.label: r for r in self.rows}
        t, r30 = by.get("Top 10"), by.get("Rest 30")
        if t is None or r30 is None or t.thin or r30.thin:
            return None
        return _classify(t.adv_pct, r30.adv_pct)

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
    """Per-symbol TOTAL futures OI on trade_date and the prior session.

    TOTAL, NOT NEAR-MONTH -- this is the rollover trap, measured on this data
    (scripts/ilc_self_audit.py). Near-month OI bleeds into the next contract as
    expiry approaches because positions MIGRATE rather than close:
        DTE 0-2 : mean OI change -47.9%, 99.94% of rows below -5%
        DTE 3-5 : mean OI change -33.0%, 99.88% below -5%
    and 13.7% of all near-month rows sit at DTE <= 5. A near-month read therefore
    prints basket-wide "unwinding" on roughly one session in seven regardless of
    what anyone did. Summing every forward expiry is continuous through the roll.
    src/analytics/index_prediction.py solved the same trap the same way.

    Both days are summed over the SAME forward set (expiry > trade_date), so the
    comparison is apples-to-apples across an expiry boundary too.
    """
    ph = ", ".join("?" * len(symbols))
    return query_dataframe(f"""
        SELECT trade_date, symbol, expiry_date, open_interest
        FROM fno_bhavcopy
        WHERE instrument = 'FUTSTK' AND open_interest > 0
          AND symbol IN ({ph})
          AND trade_date <= ? AND trade_date > (?::date - 20)
          AND expiry_date > ?
    """, [*symbols, trade_date, trade_date, trade_date])


def _total_oi_change(fut: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """symbol -> total forward OI today vs the prior session. Roll-immune."""
    if fut.empty:
        return pd.DataFrame(columns=["symbol", "oi", "prev_oi"])
    fut = fut.copy()
    fut["trade_date"] = pd.to_datetime(fut["trade_date"]).dt.date
    dates = sorted(d for d in fut["trade_date"].unique() if d <= trade_date)
    if len(dates) < 2 or dates[-1] != trade_date:
        return pd.DataFrame(columns=["symbol", "oi", "prev_oi"])
    prev = dates[-2]
    tot = (fut[fut["trade_date"].isin((trade_date, prev))]
           .groupby(["trade_date", "symbol"])["open_interest"].sum().unstack(0))
    if trade_date not in tot.columns or prev not in tot.columns:
        return pd.DataFrame(columns=["symbol", "oi", "prev_oi"])
    out = pd.DataFrame({"oi": tot[trade_date], "prev_oi": tot[prev]}).dropna()
    return out[out["prev_oi"] > 0].reset_index()


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

    # Delivery: PER-SYMBOL z against each name's own 100-day normal, then averaged.
    # NOT the z of the bucket's mean delivery -- measured (scripts/ilc_self_audit.py)
    # the Rest-30 mean has corr +0.674 with HOW MANY of its names reported that day
    # (it ranges 25-30), so a bucket-level z is partly a coverage signal. A per-symbol
    # z is immune: a name entering or leaving changes which z's are averaged, not the
    # level. Top-10 / Next-10 were already clean (corr -0.06 / -0.03); Rest 30 was not.
    h = hist[hist["symbol"].isin(mem)]
    if not h.empty:
        wide = h.pivot_table("deliv_per", "trade_date", "symbol")
        prior = wide[wide.index < t["trade_date"].iloc[0]].tail(_DELIV_BASE)
        cur = wide.reindex([t["trade_date"].iloc[0]]).iloc[0] if \
            t["trade_date"].iloc[0] in wide.index else None
        if cur is not None and len(prior) >= 30:
            mu, sd = prior.mean(), prior.std()
            ok = (sd > 1e-9) & mu.notna() & cur.notna()
            if ok.any():
                z = ((cur[ok] - mu[ok]) / sd[ok]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(z):
                    row.deliv_z = float(z.mean())

    ft = fut_today[fut_today["symbol"].isin(mem)] if not fut_today.empty else pd.DataFrame()
    if not ft.empty:
        rets = t.set_index("symbol")["r"]
        pcts = []
        for _, fr in ft.iterrows():
            oi, poi = float(fr["oi"]), float(fr["prev_oi"])
            r = rets.get(fr["symbol"])
            if poi <= 0 or r is None or pd.isna(r):
                continue
            oi_chg = (oi - poi) / poi * 100
            # Price leg is the stock's CASH return, not a futures close: the futures
            # price series breaks across a roll (different contract), the cash one
            # does not, and the direction is the same question.
            row.fut_valid += 1
            pcts.append(oi_chg)
            if   r > 0 and oi_chg >  0.5: row.fut_long   += 1
            elif r < 0 and oi_chg >  0.5: row.fut_short  += 1
            elif r > 0 and oi_chg < -0.5: row.fut_cover  += 1
            elif r < 0 and oi_chg < -0.5: row.fut_unwind += 1
        if pcts:
            # winsorised: raw means differ from the +-50% clipped mean by up to
            # 84.5pp on the worst Top-10 day and by >2pp on ~5% of sessions, because
            # a single thin contract can print +895%.
            row.fut_oi_pct = float(np.mean(np.clip(pcts, -50, 50)))

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

    # Expiry SESSION is the one case total-forward OI still cannot read: the front
    # contract settles and everyone rolls INTO the next month, so forward OI jumps
    # mechanically. Measured on this data: +14.25% mean on expiry sessions against
    # +0.47% on every other day, and it is only 26 of 524 sessions (5.0%). Total OI
    # already fixed the far worse near-month roll bleed (-48% at DTE 0-2, 13.7% of
    # sessions); this suppresses the residual rather than printing a fake build.
    _exp = query_dataframe("""
        SELECT 1 AS x FROM fno_bhavcopy
        WHERE instrument = 'FUTSTK' AND trade_date = ? AND expiry_date = ? LIMIT 1
    """, [trade_date, trade_date])
    out.is_expiry_session = not _exp.empty
    oi = (pd.DataFrame(columns=["symbol", "oi", "prev_oi"]) if out.is_expiry_session
          else _total_oi_change(_load_futures(all_syms, trade_date), trade_date))
    out.rows = [_bucket_row(lbl, mem, hist, today, oi)
                for lbl, mem in meta["buckets"].items()]
    return out


def get_concentration_trend(fno_symbol: str = "NIFTY", years: int = 5,
                            as_of: Optional[date] = None) -> pd.DataFrame:
    """Per-year: how often the index rose while most of its members fell.

    READ THIS AS A RECORD OF TODAY'S 50, NOT AS A TREND. On the shipped basket the
    series looks monotone (2.4 -> 4.1 -> 6.0 -> 6.9 -> 7.2% of sessions, 2022-2026),
    and I originally shipped that as the panel's headline. It does NOT replicate
    (scripts/ilc_self_audit.py):
        today's 50 (shipped)          2.4  4.1  6.0  6.9  7.2   monotone
        only full-history names (46)  2.4  3.7  7.3  5.6  6.6   NOT monotone
        broad liquid universe         8.5  9.1 14.2  8.1  8.4   no trend at all
    The broad basket uses no index list at all (~186 names/day passing a lagged
    100 Cr turnover floor), and it shows a 2024 spike that reverts, not a rise.
    And extending the window BACKWARDS kills it outright — anchored at mid-2024 the
    same function returns 2020: 6.4, 2021: 11.7, 2022: 2.4, 2023: 4.1. 2021 was
    HIGHER than 2026. The apparent rise was an artifact of starting the window at
    2022, which happens to be the series minimum.
    The likely cause is survivorship: this list is TODAY's members, and Nifty
    reconstitutes twice a year, so the early years are populated by names that were
    later promoted BECAUSE they outperformed — which inflates measured breadth then
    and suppresses the "carried" count. A point-in-time membership table is what
    would settle it. Until then this is a description of one basket, not evidence
    that concentration is rising.
    """
    meta = INDEX_BUCKETS.get(fno_symbol)
    if meta is None:
        return pd.DataFrame()
    all_syms = tuple(s for b in meta["buckets"].values() for s in b)
    ph = ", ".join("?" * len(all_syms))
    # Anchored on the SELECTED date, not today: picking a 2024 session must not
    # render 2026 rows underneath it. The panel is a record as of the date on screen.
    anchor = as_of or date.today()
    start = anchor - timedelta(days=365 * years + 30)
    df = query_dataframe(f"""
        SELECT trade_date, symbol,
               (close_price - prev_close) / NULLIF(prev_close, 0) * 100 AS r
        FROM daily_data
        WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
          AND close_price > 0 AND prev_close > 0
          AND trade_date >= ? AND trade_date <= ?
    """, [*all_syms, start, anchor])
    idx = query_dataframe("""
        SELECT trade_date, pct_chg FROM index_data
        WHERE index_name = ? AND trade_date >= ? AND trade_date <= ?
    """, [meta["index_name"], start, anchor])
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


# ── state base rates ─────────────────────────────────────────────────────────
# Labels are shared by the live read and the historical table so the two can
# never drift apart. Order is the order they are rendered in.
STATE_BROAD_UP   = "Broad up"
STATE_CARRIED    = "Top-10 carried (narrow up)"
STATE_LAGGED     = "Top-10 lagged (broad up)"
STATE_BROAD_DOWN = "Broad down"


def _classify(a10: Optional[float], a30: Optional[float]) -> Optional[str]:
    """Four-way split on whether each half of the index advanced."""
    if a10 is None or a30 is None:
        return None
    if a10 > 50 and a30 > 50:   return STATE_BROAD_UP
    if a10 > 50:                return STATE_CARRIED
    if a30 > 50:                return STATE_LAGGED
    return STATE_BROAD_DOWN


def get_state_base_rates(fno_symbol: str = "NIFTY", years: int = 5,
                         as_of: Optional[date] = None) -> pd.DataFrame:
    """What NIFTY did next, historically, after each top-10 / rest-30 state.

    BASE RATES, NOT A FORECAST. Read the caveats before reading the numbers.

    The four states are a partition of every session by whether the top-10 half
    and the rest-30 half each advanced, so every day lands in exactly one and the
    table is a description of the record, not a fitted rule.

    WHY NO ARROW IS DERIVED FROM IT (scripts/nifty_bucket_backtest_v2.py and
    stages 2-4, 1,153 sessions from 2022 plus a 492-session F&O panel):
      * The gap column looks strongest, and it is: the CARRIED state precedes a
        +16.4bps overnight gap against a +6.5bps base. But put top-10 breadth in
        one regression with the index's own CLOSE-STRENGTH and breadth collapses
        to t=+0.35 while close-strength holds t=+3.43. They correlate +0.655.
        This is the close-strength/gap edge already documented in this repo,
        re-found through a different lens - not a second, independent edge.
      * The 5-day column cannot be trusted at all. Daily 5-day windows overlap
        4 of 5 days; on strictly DISJOINT weeks the breadth result is t=+0.12,
        and the five weekday phase offsets range -8.5 to +28.6 bps (t -0.71 to
        +2.40). The answer depends on which weekday you start counting from.
      * Futures and options OI add nothing. A 140-candidate reality check over
        the whole F&O family returns p=0.65, with the best |t| of 2.74 sitting
        BELOW the null's median of 2.96.
      * The three-signal conjunction the panel was designed around - heavy
        top-10 delivery AND rising futures OI AND price up - pays +76bps over
        5 days against a -2bps base, and it is NOT one lucky episode (21 distinct
        episodes, t=+2.02). It still fails, for a sharper reason: NO SINGLE LEG
        AND NO PAIR PAYS. deliv alone -5, futOI alone -11, price alone +5,
        deliv+futOI +4, futOI+price +2. Only the 3-way AND pays. Three signals
        that genuinely confirm each other leave a trace in the pairs. And a
        control drawing 21 random contiguous blocks of the same sizes has
        sd 45bps with a 95th percentile of +76.6 against the observed +78.4,
        i.e. p=0.046 before counting the 7 subsets searched.
      * Power: at 5 days this panel cannot resolve anything below ~36bps on
        non-overlapping data, so the nulls there are weak evidence, not proof of
        absence. The gap horizon (~5bps detectable) is the only one that can see
        an economically relevant effect.

    Survivorship applies here as it does to the concentration trend: the basket
    is TODAY's 50 applied to history.
    """
    meta = INDEX_BUCKETS.get(fno_symbol)
    if meta is None:
        return pd.DataFrame()
    buckets = meta["buckets"]
    top10 = tuple(buckets["Top 10"])
    rest30 = tuple(buckets["Rest 30"])
    all_syms = tuple(s for b in buckets.values() for s in b)
    ph = ", ".join("?" * len(all_syms))
    anchor = as_of or date.today()
    start = anchor - timedelta(days=365 * years + 30)
    df = query_dataframe(f"""
        SELECT trade_date, symbol,
               (close_price - prev_close) / NULLIF(prev_close, 0) * 100 AS r
        FROM daily_data
        WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
          AND close_price > 0 AND prev_close > 0
          AND trade_date >= ? AND trade_date <= ?
    """, [*all_syms, start, anchor])
    idx = query_dataframe("""
        SELECT trade_date, open_val, close_val, pct_chg FROM index_data
        WHERE index_name = ? AND trade_date >= ? AND trade_date <= ?
    """, [meta["index_name"], start, anchor])
    if df.empty or idx.empty:
        return pd.DataFrame()
    df = df[df["r"].abs() < 40]
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    R = df.pivot_table("r", "trade_date", "symbol").sort_index()
    idx = idx.set_index("trade_date").sort_index().astype(float)
    ix = R.index.intersection(idx.index)
    R = R.reindex(ix)
    y = idx.loc[ix, "pct_chg"]

    def adv_pct(cols):
        s = R[[x for x in cols if x in R.columns]]
        n = s.notna().sum(axis=1)
        return (s > 0).sum(axis=1) / n.where(n > 0) * 100

    a10, a30 = adv_pct(top10), adv_pct(rest30)
    # forward returns. Compounded in log space so a 5-day figure is a real return.
    lg = np.log1p(y / 100.0)
    fwd5 = np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100.0
    fwd1 = y.shift(-1)
    gap = (idx["open_val"] / idx["close_val"].shift(1) - 1).reindex(ix).shift(-1) * 100
    k = pd.DataFrame({"a10": a10, "a30": a30, "gap": gap,
                      "d1": fwd1, "d5": fwd5}).dropna()
    if k.empty:
        return pd.DataFrame()
    k["state"] = [_classify(x, z) for x, z in zip(k["a10"], k["a30"])]
    rows = []
    for st in (STATE_BROAD_UP, STATE_CARRIED, STATE_LAGGED, STATE_BROAD_DOWN):
        g = k[k["state"] == st]
        if g.empty:
            continue
        rows.append(dict(
            state=st, days=len(g),
            gap_bps=round(g["gap"].mean() * 100, 1),
            gap_up=round((g["gap"] > 0).mean() * 100, 1),
            d1_bps=round(g["d1"].mean() * 100, 1),
            d1_up=round((g["d1"] > 0).mean() * 100, 1),
            d5_bps=round(g["d5"].mean() * 100, 1),
            d5_up=round((g["d5"] > 0).mean() * 100, 1)))
    rows.append(dict(
        state="— all sessions —", days=len(k),
        gap_bps=round(k["gap"].mean() * 100, 1),
        gap_up=round((k["gap"] > 0).mean() * 100, 1),
        d1_bps=round(k["d1"].mean() * 100, 1),
        d1_up=round((k["d1"] > 0).mean() * 100, 1),
        d5_bps=round(k["d5"].mean() * 100, 1),
        d5_up=round((k["d5"] > 0).mean() * 100, 1)))
    return pd.DataFrame(rows)
