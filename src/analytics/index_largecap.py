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
    "get_state_base_rates", "get_flow_analogues", "get_bucket_analogues",
    "index_meta",
]

# ── Bucket membership ─────────────────────────────────────────────────────────
# NIFTY: NSE's OWN top-10 and top-20, not a hand ranking. NSE Indices runs
# "Nifty Top 10 / Top 20 Equal Weight" indices whose members are the 10 / 20
# largest Nifty 50 names by 6-MONTH AVERAGE free-float cap, re-selected every
# March and September (IndexConstituent/ind_niftytop{10,20}equalweight_list.csv).
# That is exactly the property a bucket needs: an authoritative rank that does
# not flip with a week's prices. Spot weights disagree at the margin -- the
# 31 Aug 2026 factsheet has SBIN #6 and M&M #10 while the averaged selection
# still held TCS and ITC -- and a spot cut would re-shuffle every month.
#
# The list shipped before 29 Sep 2026 had 3 of 10 Top-10 names and 5 of 10
# Next-10 names wrong against this (TCS/BAJFINANCE/HINDUNILVR/SBIN in the top;
# ADANIENT, ADANIPORTS, ULTRACEMCO in Next 10; ITC, ETERNAL, NTPC in Rest 30).
# The verdict numbers measured on it were re-run on these buckets and hold --
# scripts/ilc_multi_index_backtest.py.
# In force Mar 2026 .. 29 Sep 2026 (a dated version below takes over on 30 Sep).
_NIFTY_TOP10 = ("HDFCBANK", "ICICIBANK", "RELIANCE", "BHARTIARTL", "LT",
                "INFY", "AXISBANK", "KOTAKBANK", "TCS", "ITC")
_NIFTY_NEXT10 = ("SBIN", "BAJFINANCE", "HINDUNILVR", "M&M", "MARUTI",
                 "ETERNAL", "HCLTECH", "NTPC", "SUNPHARMA", "TITAN")
_NIFTY_REST30 = (
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "BAJAJ-AUTO",
    "BAJAJFINSV", "BEL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
    "HDFCLIFE", "HINDALCO", "INDIGO", "JIOFIN", "JSWSTEEL", "MAXHEALTH",
    "NESTLEIND", "ONGC", "POWERGRID", "SBILIFE", "SHRIRAMFIN", "TATACONSUM",
    "TATASTEEL", "TECHM", "TMPV", "TRENT", "ULTRACEMCO", "WIPRO")

# ── The sectoral / midcap F&O indices ────────────────────────────────────────
# Source: NSE Indices factsheets dated 31 Aug 2026 (top-10 weights) and the
# niftyindices.com constituent CSVs. Only the TOP 10 weights are published, so
# every split below is placed at or above rank 10 -- the "Rest" bucket is simply
# everyone else and its internal order is never needed.
#
# WHERE THE SPLITS SIT, AND WHY. A rank cut placed between two near-equal weights
# flips membership month to month as prices move, and a bucket whose members
# churn is not a bucket. So each cut sits on the widest WEIGHT GAP in the
# published top 10, not on a round number copied from Nifty's 10/10/30:
#   Bank Nifty   17.02 14.86 | 10.27  9.88  9.20 |  7.15 ...    gaps 4.6pp, 2.1pp
#   Fin Nifty    17.26 15.07 | 10.42  9.63  9.51  8.83 | 4.85   gaps 4.6pp, 4.0pp
#   Midcap Sel.   9.57 |  5.16  5.05  4.78  4.63 | 4.61 ... 4.13 (flat)
# Both bank-heavy indices now sit under the SEBI concentration cap phased in
# through 2025-26 (top name <= 20%, top three <= 45%): HDFCBANK was ~28% of Bank
# Nifty before it. So "Top 2" today is ~32% of the index, where it used to be
# ~50% -- see get_state_base_rates for what that does to the history.
#
# MIDCAP SELECT HAS NO NATURAL CUT past BSE. Ranks 2-10 span 5.16 to 4.13, so
# the Top 5 / Next 5 boundary (4.63 vs 4.61) is 0.02pp wide and WILL churn. Its
# buckets are kept for a consistent layout, and the panel says the boundary is
# arbitrary rather than presenting it as structure.
_BANK_TOP2 = ("HDFCBANK", "ICICIBANK")
_BANK_NEXT3 = ("SBIN", "KOTAKBANK", "AXISBANK")
_BANK_REST9 = ("FEDERALBNK", "INDUSINDBK", "AUBANK", "IDFCFIRSTB", "BANKBARODA",
               "CANBK", "PNB", "UNIONBANK", "YESBANK")

_FIN_TOP2 = ("HDFCBANK", "ICICIBANK")
_FIN_NEXT4 = ("SBIN", "KOTAKBANK", "AXISBANK", "BAJFINANCE")
_FIN_REST14 = ("SHRIRAMFIN", "BSE", "BAJAJFINSV", "CHOLAFIN", "HDFCLIFE",
               "ICICIGI", "JIOFIN", "LICHSGFIN", "MFSL", "MUTHOOTFIN", "PFC",
               "RECLTD", "SBICARD", "SBILIFE")

_MID_TOP5 = ("BSE", "PAYTM", "HEROMOTOCO", "INDUSINDBK", "POLICYBZR")
_MID_NEXT5 = ("BHEL", "DIXON", "AUBANK", "PERSISTENT", "SUZLON")
_MID_REST15 = ("ASHOKLEY", "AUROPHARMA", "BHARATFORG", "FORTIS", "HINDPETRO",
               "INDIANB", "INDUSTOWER", "NAUKRI", "LICI", "LUPIN", "MARICO",
               "POLYCAB", "SRF", "SWIGGY", "YESBANK")

# fno_symbol -> index_data name, display, ordered {bucket label: symbols}, and
# the per-index constants that used to be Nifty literals:
#   weights     rough published weight share per bucket, used ONLY to blend the
#               display score (never claimed as index points)
#   carry_band  |carry spread| beyond which the day is called top-heavy / broad.
#               Nifty's 0.15pp is the median of its own |spread| (p50.7). Each
#               other band is 0.15 x (that index's median / Nifty's median) over
#               the sessions since Mar 2026, i.e. the CURRENT regime: the SEBI
#               cap halved Bank Nifty's median |spread| (0.369 -> 0.184) and cut
#               Fin Nifty's (0.335 -> 0.235), so a full-history band (0.34 / 0.32)
#               would call almost no post-cap day top-heavy.
#               scripts/ilc_multi_index_backtest.py [8].
#   history_from  earliest session the base rates and analogues may use -- see
#               get_state_base_rates for why this is not always "5 years".
# The FIRST bucket is the heavyweights and the LAST is the tail; the state
# partition reads those two by POSITION, so labels are free to differ by index.
INDEX_BUCKETS: dict[str, dict] = {
    "NIFTY": dict(
        index_name="Nifty 50", display="Nifty 50",
        buckets={"Top 10": _NIFTY_TOP10,
                 "Next 10": _NIFTY_NEXT10,
                 "Rest 30": _NIFTY_REST30},
        weights={"Top 10": 0.46, "Next 10": 0.24, "Rest 30": 0.30},
        carry_band=0.15, history_from=None,
    ),
    "BANKNIFTY": dict(
        index_name="Nifty Bank", display="Bank Nifty",
        buckets={"Top 2": _BANK_TOP2, "Next 3": _BANK_NEXT3,
                 "Rest 9": _BANK_REST9},
        weights={"Top 2": 0.32, "Next 3": 0.29, "Rest 9": 0.39},
        carry_band=0.21, history_from=None,
    ),
    "FINNIFTY": dict(
        index_name="Nifty Financial Services", display="Fin Nifty",
        buckets={"Top 2": _FIN_TOP2, "Next 4": _FIN_NEXT4,
                 "Rest 14": _FIN_REST14},
        weights={"Top 2": 0.32, "Next 4": 0.38, "Rest 14": 0.30},
        carry_band=0.27, history_from=None,
    ),
    "MIDCPNIFTY": dict(
        index_name="Nifty Midcap Select", display="Midcap Select",
        buckets={"Top 5": _MID_TOP5, "Next 5": _MID_NEXT5,
                 "Rest 15": _MID_REST15},
        weights={"Top 5": 0.29, "Next 5": 0.22, "Rest 15": 0.49},
        carry_band=0.11, history_from=date(2024, 11, 29),
    ),
}

# Nifty's weights, kept under the old name: the composite was measured with
# them (scripts/nifty_bucket_composite.py) and tested under equal weighting too,
# with the same null result.
_BUCKET_WEIGHT = INDEX_BUCKETS["NIFTY"]["weights"]

# ── Membership is DATED ──────────────────────────────────────────────────────
# INDEX_BUCKETS above is the basket as of the 31 Aug 2026 factsheets. NSE
# reconstitutes every March and September, so one frozen list goes wrong on the
# next effective date. Each version below applies from `effective` onward, and
# index_meta(sym, d) returns the basket in force on d.
#
# 30 Sep 2026 (close of 29 Sep), NSE Indices press release 10 Aug 2026
# (niftyindices.com/Press_Release/ind_prs10082026.pdf):
#   Nifty 50       WIPRO out, BSE in (6-month avg free-float cap Rs 1,40,xxx Cr
#                  against Wipro's 55,930 -- far below the Next-10 cut, so Rest 30)
#   Nifty Top 10 Equal Weight: TCS out, SBIN in; Top 15 / Top 20 unchanged ->
#                  SBIN moves up to Top 10 and TCS down to Next 10
#   Nifty Bank, Nifty Financial Services: "No changes are being made"
#   Midcap Select  out BSE, FORTIS, HINDPETRO, POLYCAB
#                  in  FEDERALBNK, GVT&D, INDHOTEL, MCX
# Midcap Select loses BSE, its only real heavyweight (9.57%), so its buckets are
# re-cut. PROVISIONAL until NSE's 30 Sep factsheet: Nifty Midcap 100 is uncapped
# free-float weighted and Midcap Select's weights are exactly proportional to it
# (BSE/PAYTM = 9.57/5.16 = 1.855 in Select, 3.84/2.07 = 1.855 in Midcap 100,
# both 31 Aug 2026), so incoming names can be placed on Select's scale:
#   FEDERALBNK 2.52 -> ~6.28   MCX 2.48 -> ~6.18   | PAYTM 5.16 HERO 5.06
#   INDUSINDBK 4.78 POLICYBZR 4.63 BHEL 4.61 | DIXON 4.40 AUBANK 4.39 ...
# The only gap wider than 0.3pp is after rank 2 (1.02pp) -> Top 2 / Next 5 /
# Rest 18.
#
# CONFIRMED 05 Oct 2026 against NSE's 30 Sep 2026 factsheet. The estimate put
# FEDERALBNK ~6.28 and MCX ~6.18 on top; published: MCX 6.41, FEDERALBNK 5.96 --
# the Top 2 was right. One Next-5 name was wrong: INDHOTEL (rank unknown then)
# is rank 5 at 4.87 and POLICYBZR fell out of the top 10, so they swap. Published
# top 10: MCX 6.41 FEDERALBNK 5.96 | PAYTM 5.50 HEROMOTOCO 5.18 INDHOTEL 4.87
# BHEL 4.61 INDUSINDBK 4.52 | AUBANK 4.47 PERSISTENT 4.37 DIXON 4.15. The gap
# after rank 2 is 0.46pp; nothing after it exceeds 0.32pp, so the Next 5 / Rest
# 18 boundary (4.52 vs 4.47) is a layout choice and the panel says so.
_NIFTY_TOP10_2609 = tuple(s if s != "TCS" else "SBIN" for s in _NIFTY_TOP10)
_NIFTY_NEXT10_2609 = tuple(s if s != "SBIN" else "TCS" for s in _NIFTY_NEXT10)
_NIFTY_REST30_2609 = tuple(sorted(
    (set(_NIFTY_REST30) - {"WIPRO"}) | {"BSE"}))
_MID2609_TOP2 = ("MCX", "FEDERALBNK")
_MID2609_NEXT5 = ("PAYTM", "HEROMOTOCO", "INDHOTEL", "BHEL", "INDUSINDBK")
_MID2609_REST18 = ("AUBANK", "PERSISTENT", "DIXON", "SUZLON", "POLICYBZR",
                   "GVT&D", "ASHOKLEY", "AUROPHARMA", "BHARATFORG", "INDIANB",
                   "INDUSTOWER", "NAUKRI", "LICI", "LUPIN", "MARICO", "SRF",
                   "SWIGGY", "YESBANK")

_VERSIONS: dict[str, list[dict]] = {
    "NIFTY": [dict(
        effective=date(2026, 9, 30),
        buckets={"Top 10": _NIFTY_TOP10_2609, "Next 10": _NIFTY_NEXT10_2609,
                 "Rest 30": _NIFTY_REST30_2609},
        source=("NSE review 10 Aug 2026: WIPRO out, BSE in (Rest 30); "
                "SBIN up to Top 10, TCS down to Next 10"),
        provisional=False)],
    "MIDCPNIFTY": [dict(
        effective=date(2026, 9, 30),
        buckets={"Top 2": _MID2609_TOP2, "Next 5": _MID2609_NEXT5,
                 "Rest 18": _MID2609_REST18},
        weights={"Top 2": 0.12, "Next 5": 0.25, "Rest 18": 0.63},
        source=("NSE review 10 Aug 2026: BSE, FORTIS, HINDPETRO, POLYCAB out; "
                "FEDERALBNK, GVT&D, INDHOTEL, MCX in; buckets per the 30 Sep "
                "2026 factsheet"),
        provisional=False)],
}


def index_meta(fno_symbol: str, on: Optional[date] = None) -> Optional[dict]:
    """The basket in force on `on` (default: today). None for an unknown index.

    Returns a copy carrying `effective` (None for the base factsheet basket),
    `source` and `provisional`, so the panel can say which basket it used.
    """
    base = INDEX_BUCKETS.get(fno_symbol)
    if base is None:
        return None
    meta = dict(base, effective=None, provisional=False,
                source=("NSE Top 10 / Top 20 Equal Weight membership"
                        if fno_symbol == "NIFTY" else "NSE factsheets, 31 Aug 2026"))
    on = on or date.today()
    for v in _VERSIONS.get(fno_symbol, []):
        if on >= v["effective"]:
            meta.update(v)
            if "weights" not in v:
                # same labels, same rough shares
                meta["weights"] = base["weights"]
    return meta


def _min_valid(n_members: int) -> int:
    """F&O names a bucket needs before its futures/options read is shown.

    Four for Nifty's 10- and 30-name buckets, as before. A fixed four would
    silence a 2- or 3-name heavyweight bucket FOREVER -- Bank Nifty's Top 2 could
    never print a futures lean -- so small buckets need all-but-none of their
    names, and never fewer than two: one stock is a stock, not a bucket.
    """
    return max(2, min(4, -(-n_members * 3 // 5)))


_HEAVY_Z, _HEAVY_ACT = 0.62, 41.0    # ordinary-day p95s, see get_index_largecap


def _basket_heat(hist: pd.DataFrame, trade_date: date):
    """All-member mean delivery z (21-session, per symbol) and turnover vs norm %."""
    try:
        dl = hist.pivot_table("deliv_per", "trade_date", "symbol").sort_index()
        tv = (hist.pivot_table("turnover_lacs", "trade_date", "symbol")
              .sort_index().sum(axis=1))
        if trade_date not in dl.index:
            return None, None
        prior = dl[dl.index < trade_date].tail(_DELIV_BASE)
        if len(prior) < _DELIV_MINP:
            return None, None
        mu, sd = prior.mean(), prior.std()
        z = ((dl.loc[trade_date] - mu) / sd.where(sd > 1e-9)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        base = tv[tv.index < trade_date].tail(_DELIV_BASE).mean()
        act = float(tv[trade_date] / base * 100 - 100) if base > 0 else None
        return (float(z.mean()) if len(z) else None), act
    except Exception:                                          # noqa: BLE001
        return None, None


def _is_rebalance_close(trade_date: date) -> Optional[str]:
    """Source note if trade_date is the last session before a dated basket takes
    effect (NSE trades the rebalance at that close), else None."""
    for versions in _VERSIONS.values():
        for v in versions:
            eff = v["effective"]
            if trade_date < eff and (eff - trade_date).days <= 4:
                later = query_dataframe(
                    "SELECT 1 FROM daily_data WHERE trade_date > ? AND trade_date < ? "
                    "LIMIT 1", [trade_date, eff])
                if later.empty:
                    return f"basket change effective {eff:%d %b %Y}"
    return None
# Sessions the delivery normal is measured over. 21 = roughly one trading month.
#
# Measured against a 100-session alternative (scripts/ilc_deliv_baseline.py):
# delivery IS persistent -- bucket-level lag-21 autocorrelation runs +0.32 (Top 10),
# +0.57 (Next 10), +0.64 (Rest 30) -- so the worry is that a short baseline chases
# the very move it should be measuring and prints ~0 when the signal is strongest.
# That does not happen. The two agree at correlation +0.91, fire the +-0.3 band at
# nearly the same rate (59.3% vs 60.5% on Top 10), and on the top decile of
# SUSTAINED 10-session delivery shifts the short window is consistently LARGER,
# not smaller: ratios 1.04x to 1.28x across all three buckets in both directions.
# A tighter window also tracks regime changes faster, which is the point.
# The cost is that one unusual session moves it ~5x more (see unusual_session).
_DELIV_BASE = 21
_DELIV_MINP = max(10, _DELIV_BASE // 2)   # never exceed the window -- see below
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
    deliv_z:      Optional[float] = None     # vs the bucket's own 21-session normal
    deliv_trail:  list = field(default_factory=list)   # last 4 sessions, newest first
    # Turnover against the bucket's own 21-session norm, in %. This is the OTHER
    # factor in delivery value (value = delivery share x turnover), and it is the
    # only way to tell a genuine delivery surge from a ratio that rose because
    # volume fell. Deliberately NOT the rupee delivery value: measured over 1,153
    # sessions, a delivery-VALUE z correlates +0.94 to +0.95 with a turnover z, so
    # it is a "busy day" indicator wearing a delivery label. The delivery SHARE
    # carries the independent information (it shares only 0.39-0.56 with value).
    turnover_vs_norm: Optional[float] = None
    # Rupee delivery today and its 21-session normal, in Cr. Used ONLY to
    # prove or disprove a delivery reading in the tooltip -- deliberately
    # not a column: a delivery-value z correlates +0.94..+0.95 with a
    # turnover z, so as a standalone number it just says 'busy day'.
    deliv_value_cr: Optional[float] = None
    deliv_value_norm_cr: Optional[float] = None
    fut_long:     int = 0                    # OI-price matrix counts (near month)
    fut_short:    int = 0
    fut_cover:    int = 0
    fut_unwind:   int = 0
    fut_valid:    int = 0
    fut_oi_pct:   Optional[float] = None
    ce_oi_pct:    Optional[float] = None     # total forward CALL OI change %
    pe_oi_pct:    Optional[float] = None     # total forward PUT OI change %
    opt_valid:    int = 0
    movers_up:    list = field(default_factory=list)
    movers_dn:    list = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.n_present / self.n_members if self.n_members else 0.0

    @property
    def thin(self) -> bool:
        return self.coverage < _MIN_COVER

    @property
    def opt_read(self) -> Optional[str]:
        """CE vs PE forward-OI flow, the way the desk reads it.

        Call side building relative to the put side is call writing (bearish);
        put side building relative to the call side is put writing (bullish).
        This is a RELATIVE read and deliberately not conditioned on price -- it is
        the construction the user described, and it was tested as such.
        """
        if (self.opt_valid < _min_valid(self.n_members)
                or self.ce_oi_pct is None or self.pe_oi_pct is None):
            return None
        d = self.pe_oi_pct - self.ce_oi_pct
        if abs(d) < 0.25:
            return "balanced"
        if self.pe_oi_pct > 0 and self.ce_oi_pct <= 0:  return "put writing"
        if self.ce_oi_pct > 0 and self.pe_oi_pct <= 0:  return "call writing"
        return "put side heavier" if d > 0 else "call side heavier"

    @property
    def flow_score(self) -> Optional[float]:
        """Delivery + futures + options, each in [-1, +1], averaged.

        MEASURED WORTHLESS AS A FORECAST and kept only as a summary of what the
        flow is doing -- see IndexLargeCap.net_score for the numbers.
        """
        legs = []
        if self.deliv_z is not None:
            legs.append(1.0 if self.deliv_z > 0.3 else -1.0 if self.deliv_z < -0.3 else 0.0)
        lean = self.fut_lean
        if lean is not None:
            legs.append(float(lean))
        rd = self.opt_read
        if rd is not None:
            legs.append({"put writing": 1.0, "put side heavier": 0.5, "balanced": 0.0,
                         "call side heavier": -0.5, "call writing": -1.0}[rd])
        return float(np.mean(legs)) if legs else None

    @property
    def fut_read(self) -> Optional[str]:
        """The lean, NAMED BY WHAT DOMINATES IT. None when too thin to call.

        fut_lean lumps long build with short covering (both are price-up) and
        short build with long unwind. They are different things -- fresh money
        versus positions being closed -- and the lump was labelled "net long
        build". Measured 2022-2026: on days a bucket reads bullish, its bullish
        names are MOSTLY SHORT COVERING on 37-60% of them (Nifty Top 10: 60%), so
        that label was wrong about half the time. Bearish days are cleaner:
        mostly long unwind on 15-30%.
        """
        lean = self.fut_lean
        if lean is None:
            return None
        if lean == 0:
            return "mixed"
        if lean > 0:
            if self.fut_long > self.fut_cover:   return "long build"
            if self.fut_cover > self.fut_long:   return "short covering"
            return "long build + covering"
        if self.fut_short > self.fut_unwind:     return "short build"
        if self.fut_unwind > self.fut_short:     return "long unwind"
        return "short build + unwind"

    @property
    def fut_votes(self) -> str:
        """Per-stock quadrant counts, so the label can be checked by eye."""
        if not self.fut_valid:
            return "—"
        flat = self.fut_valid - (self.fut_long + self.fut_cover
                                 + self.fut_short + self.fut_unwind)
        return (f"▲ {self.fut_long} long · {self.fut_cover} cover  |  "
                f"▼ {self.fut_short} short · {self.fut_unwind} unwind"
                + (f"  |  {flat} flat" if flat else ""))

    @property
    def fut_lean(self) -> Optional[int]:
        """+1 net fresh-long / -1 net fresh-short / 0 mixed. None when too thin."""
        if self.fut_valid < _min_valid(self.n_members):
            return None
        bull = self.fut_long + self.fut_cover
        bear = self.fut_short + self.fut_unwind
        # A lean needs the quorum among names that actually MOVED OI (>0.5%), not
        # just names with a futures row. Otherwise one stock building while its
        # partner sat flat printed "net long build" for the whole bucket: 41.6% of
        # Bank/Fin Top-2 leans were exactly that (Nifty buckets: 0.6-0.8%).
        if bull + bear < _min_valid(self.n_members):
            return 0
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
    # Which contract month is the front one, and how far away it is. Shown so the
    # reader can see WHERE IN THE CYCLE the OI numbers were taken.
    front_expiry:  Optional[date] = None
    days_to_expiry: Optional[int] = None
    near_oi_share: Optional[float] = None   # % of forward OI in the front month
    # Set on Budget days, Muhurat and other special sessions. These are REAL
    # sessions and belong in the data, but they are short and thin, so their
    # delivery reads reflect the session TYPE rather than investor behaviour:
    # 2026-02-01 (Budget Sunday) prints delivery z of -1.75/-2.30/-2.36 across the
    # three buckets on 117 Cr of turnover against a 172 Cr weekday, and Muhurat
    # 2023-11-12 runs the other way at 64.1% delivery against ~53% on its
    # neighbours. Shown as a caveat rather than suppressed.
    unusual_session: Optional[str] = None
    # "thin" (Budget Sunday, Muhurat, weekend) or "heavy" (passive rebalance close)
    unusual_kind: Optional[str] = None
    # rough weight share per bucket label, from the basket in force on the date
    bucket_weights: dict = field(default_factory=dict)

    @property
    def net_score(self) -> Optional[float]:
        """Weight-shared blend of the three bucket flow scores, in [-1, +1].

        DISPLAYED AS POSITIONING, NEVER AS A CALL. Backtested over 1,149 sessions
        (2022-2026, scripts/nifty_bucket_composite.py) on the full F&O history:
            next day   IC -0.012   hit 48.2%   against a 52.2% base rate
            next 5 d   IC +0.005   hit 49.1%   against a 54.7% base rate
        The index's own close-strength beats it (t +2.15 vs -0.69), it adds
        nothing beside that control (t -0.17), its quintiles are not monotone, and
        the walk-forward half is worse than the in-sample half (46.3% next-day hit
        after 2024-09). The top-10-only variant hits 44.6%, BELOW a coin flip.

        RE-MEASURED 29 Sep 2026 on NSE's own Top-10 / Top-20 buckets (the list
        above had 3 Top-10 and 5 Next-10 names wrong), 1,170 sessions:
            next day   IC -0.016   hit 49.3%   against a 52.1% base rate
            next 5 d   IC +0.011   hit 49.6%   against a 54.3% base rate
        beside close-strength t -0.29, walk-forward 48.0%, top-10-only 44.1% --
        the same verdict. Other indices: scripts/ilc_multi_index_backtest.py.
        """
        scored = [r for r in self.rows if r.flow_score is not None]
        if not scored:
            return None
        # An unknown bucket label used to be dropped SILENTLY (weight None), so a
        # future index whose buckets are not named "Top 10"/"Next 10"/"Rest 30"
        # would render a blank score with no explanation. Fall back to equal
        # weighting instead: an approximate blend beats a mystery blank.
        wmap = (self.bucket_weights
                or INDEX_BUCKETS.get(self.fno_symbol, {}).get("weights", _BUCKET_WEIGHT))
        if any(r.label not in wmap for r in scored):
            return float(np.mean([r.flow_score for r in scored]))
        tw = sum(wmap[r.label] for r in scored)
        if not tw:
            return None
        return float(sum(r.flow_score * wmap[r.label] for r in scored) / tw)

    @property
    def state(self) -> Optional[str]:
        """Which of the four heavyweight / tail states today falls in.

        Reads the FIRST bucket (heavyweights) and the LAST (the tail) by position,
        so Bank Nifty's "Top 2 / Rest 9" classifies exactly as Nifty's "Top 10 /
        Rest 30" does. Shares its label constants with get_state_base_rates(), so
        the live read and the historical table can never fall out of sync.
        """
        if len(self.rows) < 2:
            return None
        t, tail = self.rows[0], self.rows[-1]
        if t.thin or tail.thin:
            return None
        return _classify(t.adv_pct, tail.adv_pct, t.ret_pct, tail.ret_pct)

    @property
    def carry_band(self) -> float:
        return INDEX_BUCKETS.get(self.fno_symbol, {}).get("carry_band", 0.15)

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
        if self.carry_spread > self.carry_band:
            return "Index ahead of its own basket — top-heavy day"
        if self.carry_spread < -self.carry_band:
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


def _load_futures(symbols: tuple, trade_date: date,
                  include_expiring: bool = False) -> pd.DataFrame:
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

    WHY NOT NEAR-MONTH-WITH-A-ROLL, which is the usual desk convention
    (scripts/ilc_expiry_choice.py, 1,155 sessions). The front month does hold most
    of the interest -- 84.1% of stock-futures OI and 90.8% of stock-options OI --
    so the instinct is right. But matching each contract to ITSELF, which removes
    the contract-switch artifact entirely, genuine roll bleed on the front
    contract runs:
        DTE 0-1  -54.99%    DTE 4-5   -31.99%    DTE  9-12  -0.65%
        DTE 2    -37.67%    DTE 6-8    -5.04%    DTE 13-20  +0.11%
    So a roll "one or two days before expiry" is far too late: at DTE 2 the front
    is already shedding 37.7% a day with 100% of rows below -5%. It does not
    settle down until DTE 9-12, a week and a half out.

    And even rolled correctly, near-month-only is far worse than the sum. With
    settlement sessions excluded, as this panel excludes them:
        TOTAL forward   sd 1.23   expiry-week +0.447% vs other +0.416%  gap 0.03pp
        near + next     sd 1.26   expiry-week +0.151% vs other +0.301%  gap 0.15pp
        near month only sd 12.79  expiry-week -22.93% vs other -2.05%   gap 20.88pp
    Near-only is ten times noisier and would print basket-wide "unwinding" every
    expiry week. Near+next tracks the total at correlation 0.97 (sign disagrees on
    4-5% of sessions) but is marginally worse on both measures, and the far months
    it drops are only 1.0% of futures OI. The SUM is the only continuous measure:
    while the front bleeds, the next fills, and only their total is flat.
    """
    ph = ", ".join("?" * len(symbols))
    if include_expiring:
        # SETTLEMENT SESSION ONLY: keep the contract that settles today, on both
        # days. See get_index_largecap for why this basis is continuous there.
        return query_dataframe(f"""
            SELECT trade_date, symbol, expiry_date, open_interest
            FROM fno_bhavcopy
            WHERE instrument = 'FUTSTK' AND open_interest > 0
              AND symbol IN ({ph})
              AND trade_date <= ? AND trade_date > (?::date - 20)
              AND expiry_date >= ?
        """, [*symbols, trade_date, trade_date, trade_date])
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


def _load_options(symbols: tuple, trade_date: date) -> pd.DataFrame:
    """Per-symbol TOTAL forward CALL and PUT OI on trade_date and the prior session.

    Same rollover discipline as the futures leg: summed over every live expiry, so
    a position migrating from the near contract to the next does not read as an
    unwind. Strikes are summed too -- this is a SIDE-level flow read, not a
    strike-level one.
    """
    ph = ", ".join("?" * len(symbols))
    return query_dataframe(f"""
        SELECT trade_date, symbol, option_type, SUM(open_interest) AS oi
        FROM fno_bhavcopy
        WHERE instrument = 'OPTSTK' AND open_interest > 0
          AND option_type IN ('CE','PE')
          AND symbol IN ({ph})
          AND trade_date <= ? AND trade_date > (?::date - 20)
          AND expiry_date > ?
        GROUP BY 1, 2, 3
    """, [*symbols, trade_date, trade_date, trade_date])


def _opt_change(opt: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """symbol -> CE and PE total forward OI % change vs the prior session."""
    cols = ["symbol", "ce_pct", "pe_pct"]
    if opt.empty:
        return pd.DataFrame(columns=cols)
    opt = opt.copy()
    opt["trade_date"] = pd.to_datetime(opt["trade_date"]).dt.date
    dates = sorted(d for d in opt["trade_date"].unique() if d <= trade_date)
    if len(dates) < 2 or dates[-1] != trade_date:
        return pd.DataFrame(columns=cols)
    prev = dates[-2]
    cur = opt[opt["trade_date"] == trade_date]
    pri = opt[opt["trade_date"] == prev]
    out = {}
    for side, key in (("CE", "ce_pct"), ("PE", "pe_pct")):
        a = cur[cur["option_type"] == side].set_index("symbol")["oi"]
        b = pri[pri["option_type"] == side].set_index("symbol")["oi"]
        j = pd.DataFrame({"a": a, "b": b}).dropna()
        j = j[j["b"] > 0]
        # winsorised for the same reason the futures leg is: one thin series can
        # print a several-hundred-percent change and carry the bucket mean.
        out[key] = ((j["a"] - j["b"]) / j["b"] * 100).clip(-50, 50)
    df = pd.DataFrame(out).dropna()
    return df.reset_index().rename(columns={"index": "symbol"}) if not df.empty         else pd.DataFrame(columns=cols)


def _bucket_row(label: str, members: tuple, hist: pd.DataFrame,
                today: pd.DataFrame, fut_today: pd.DataFrame,
                opt_today: pd.DataFrame | None = None) -> BucketRow:
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
        if cur is not None and len(prior) >= _DELIV_MINP:
            mu, sd = prior.mean(), prior.std()
            ok = (sd > 1e-9) & mu.notna() & cur.notna()
            if ok.any():
                z = ((cur[ok] - mu[ok]) / sd[ok]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(z):
                    row.deliv_z = float(z.mean())
        # last 4 sessions of the bucket's mean delivery %, newest first, so the
        # DIRECTION is visible rather than just today's level
        _recent = wide[wide.index <= t["trade_date"].iloc[0]].tail(4)
        if not _recent.empty:
            row.deliv_trail = [round(float(v), 1)
                               for v in _recent.mean(axis=1).iloc[::-1]
                               if pd.notna(v)]
        if "turnover_lacs" in h.columns:
            tw = h.pivot_table("turnover_lacs", "trade_date", "symbol").sum(axis=1)
            _today_t = t["trade_date"].iloc[0]
            _base = tw[tw.index < _today_t].tail(_DELIV_BASE)
            if _today_t in tw.index and len(_base) >= _DELIV_MINP and _base.mean() > 0:
                row.turnover_vs_norm = float(
                    tw[_today_t] / _base.mean() * 100 - 100)
            # delivered value = delivery share x traded value
            _dv = (h.assign(_d=h['deliv_per'] / 100 * h['turnover_lacs'])
                   .pivot_table('_d', 'trade_date', 'symbol').sum(axis=1))
            _dvb = _dv[_dv.index < _today_t].tail(_DELIV_BASE)
            if _today_t in _dv.index and len(_dvb) >= _DELIV_MINP:
                row.deliv_value_cr = float(_dv[_today_t] / 100)
                row.deliv_value_norm_cr = float(_dvb.mean() / 100)

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

    if opt_today is not None and not opt_today.empty:
        ot = opt_today[opt_today["symbol"].isin(mem)]
        if not ot.empty:
            row.opt_valid = len(ot)
            row.ce_oi_pct = float(ot["ce_pct"].mean())
            row.pe_oi_pct = float(ot["pe_pct"].mean())

    s = t.sort_values("r", ascending=False)
    row.movers_up = [(r.symbol, float(r.r)) for r in s.head(3).itertuples() if r.r > 0]
    row.movers_dn = [(r.symbol, float(r.r)) for r in s.tail(3).itertuples() if r.r < 0][::-1]
    return row


def get_index_largecap(trade_date: date, fno_symbol: str = "NIFTY") -> IndexLargeCap:
    """Weight-bucket decomposition of one index for `trade_date`. Causal."""
    meta = index_meta(fno_symbol, trade_date)
    if meta is None:
        return IndexLargeCap(trade_date=trade_date, fno_symbol=fno_symbol,
                             display=fno_symbol, data_ok=False,
                             note=f"No bucket definition for {fno_symbol}.")
    out = IndexLargeCap(trade_date=trade_date, fno_symbol=fno_symbol,
                        display=meta["display"],
                        bucket_weights=dict(meta["weights"]))
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

    # SETTLEMENT SESSION. The forward-only sum (expiry > today on both days) jumps
    # mechanically here: the front contract settles and everyone rolls INTO the
    # next month. Measured 2022-2026, 57 settlement sessions, Nifty Top-10 bucket
    # mean OI change: +18.5% (sd 7.0) against +0.43% (sd 1.28) on ordinary days.
    #
    # FUTURES ARE RECOVERED by counting the contract that settles today on BOTH
    # days: NSE still reports its open interest at the close, so the all-contracts
    # total is continuous through the roll -- +1.55% (sd 2.56) on settlement days
    # against +0.07% (sd 2.02) otherwise, and the bucket leans it prints match
    # ordinary days with no directional tilt (Nifty long/mixed/short 38/22/40 vs
    # 36/30/34; Bank 39/27/33 vs 36/35/29; Fin 35/36/29 vs 34/40/27; Midcap
    # 38/27/35 vs 40/26/34). The next session is safe on the usual basis (+0.69%).
    #
    # OPTIONS STAY SUPPRESSED. On the same all-contracts basis the expiring calls
    # fall away faster than the puts (CE -6.2% vs PE -2.0%), and the options read
    # comes out "put side heavier / put writing" on 72-86% of settlement bucket-
    # days against 42-46% ordinarily -- a manufactured bullish read every month.
    _exp = query_dataframe("""
        SELECT 1 AS x FROM fno_bhavcopy
        WHERE instrument = 'FUTSTK' AND trade_date = ? AND expiry_date = ? LIMIT 1
    """, [trade_date, trade_date])
    out.is_expiry_session = not _exp.empty
    oi = _total_oi_change(
        _load_futures(all_syms, trade_date, include_expiring=out.is_expiry_session),
        trade_date)
    opt = (pd.DataFrame(columns=["symbol", "ce_pct", "pe_pct"])
           if out.is_expiry_session
           else _opt_change(_load_options(all_syms, trade_date), trade_date))
    out.rows = [_bucket_row(lbl, mem, hist, today, oi, opt)
                for lbl, mem in meta["buckets"].items()]
    _exp = query_dataframe(f"""
        WITH f AS (
            SELECT expiry_date, SUM(open_interest) AS oi
            FROM fno_bhavcopy
            WHERE instrument IN ('FUTSTK','OPTSTK') AND open_interest > 0
              AND symbol IN ({", ".join("?" * len(all_syms))})
              AND trade_date = ? AND expiry_date > trade_date
            GROUP BY 1)
        SELECT expiry_date, oi, SUM(oi) OVER () AS tot FROM f ORDER BY expiry_date
    """, [*all_syms, trade_date])
    _turn = query_dataframe(f"""
        SELECT trade_date, SUM(turnover_lacs) AS t
        FROM daily_data
        WHERE series IN ('EQ','SM','ST') AND trade_date <= ?
          AND trade_date > (?::date - 45)
        GROUP BY 1 ORDER BY 1
    """, [trade_date, trade_date])
    if len(_turn) >= 6:
        _cur = float(_turn.iloc[-1]["t"])
        _norm = float(_turn.iloc[:-1]["t"].median())
        _why = []
        if trade_date.weekday() >= 5:
            _why.append(f"a {trade_date:%A} session")
        if _norm > 0 and _cur < 0.6 * _norm:
            _why.append(f"turnover {_cur / _norm * 100:.0f}% of the recent norm")
        if _why:
            out.unusual_session = " and ".join(_why)
            out.unusual_kind = "thin"
    # HEAVY, basket-wide delivery with heavy activity -- the signature of passive
    # funds trading an index rebalance at the close, which is delivery by
    # construction and says nothing about conviction. Thresholds are the ordinary-
    # day 95th percentiles of the all-member delivery z (0.62) and activity (+41%),
    # 2022-2026. Every NSE rebalance close checked sits at or beyond both: 27 Sep
    # 2024 z +0.96 / +70%, 29 Sep 2025 +0.79 / +94%, 29 Sep 2026 +0.75 / +87%.
    if out.unusual_session is None:
        z_all, act_all = _basket_heat(hist, trade_date)
        rebal = _is_rebalance_close(trade_date)
        if rebal or (z_all is not None and act_all is not None
                     and z_all > _HEAVY_Z and act_all > _HEAVY_ACT):
            bits = []
            if rebal:
                bits.append(f"NSE index rebalance at this close ({rebal})")
            if z_all is not None and act_all is not None:
                bits.append(f"delivery z {z_all:+.2f} and activity {act_all:+.0f}% "
                            "across the whole basket")
            out.unusual_session = " — ".join(bits)
            out.unusual_kind = "heavy"
    if not _exp.empty:
        _f = _exp.iloc[0]
        out.front_expiry = pd.to_datetime(_f["expiry_date"]).date()
        out.days_to_expiry = (out.front_expiry - trade_date).days
        if _f["tot"]:
            out.near_oi_share = float(_f["oi"] / _f["tot"] * 100)
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
    meta = index_meta(fno_symbol, as_of)
    if meta is None:
        return pd.DataFrame()
    all_syms = tuple(s for b in meta["buckets"].values() for s in b)
    ph = ", ".join("?" * len(all_syms))
    # Anchored on the SELECTED date, not today: picking a 2024 session must not
    # render 2026 rows underneath it. The panel is a record as of the date on screen.
    anchor = as_of or date.today()
    start = anchor - timedelta(days=365 * years + 30)
    if meta.get("history_from"):
        start = max(start, meta["history_from"])
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
    # most of the basket is needed for breadth: 40 of Nifty's 50, and the same
    # 80% share of a 14-, 20- or 25-name index (a literal 40 blanked all three)
    keep = cov >= 0.8 * len(all_syms)
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
# "Heavyweights", not "Top-10": the same partition runs on every index, and on
# Bank Nifty the heavyweight bucket is two names.
STATE_CARRIED    = "Heavyweights carried (narrow up)"
STATE_LAGGED     = "Heavyweights lagged (broad up)"
STATE_BROAD_DOWN = "Broad down"


def _up(adv: float, ret: Optional[float]) -> bool:
    """Did this bucket advance? Majority of names up; an exact tie goes to the
    bucket's equal-weight return.

    Without the tie-break an exactly-half day counted as NOT up. Harmless-looking,
    but measured (2022-2026): Bank/Fin Nifty's Top 2 split one-up/one-down on
    34.0% of sessions, and on those days the pair's return was positive 51% of
    the time -- so ~17% of ALL sessions were filed under the wrong state, piling
    into "down". Nifty's Top 10 ties on 12.0% (59% of them up). An odd-sized
    bucket cannot tie, so Midcap Select's old Top 5 was unaffected.
    """
    if adv != 50 or ret is None or pd.isna(ret):
        return adv > 50
    return ret > 0


def _classify(a10: Optional[float], a30: Optional[float],
              r10: Optional[float] = None,
              r30: Optional[float] = None) -> Optional[str]:
    """Four-way split on whether each end of the index advanced.

    r10/r30 are the buckets' equal-weight returns, used only to break an exact
    50% tie (see _up). Without them a tie counts as not up, as before.
    """
    if a10 is None or a30 is None or pd.isna(a10) or pd.isna(a30):
        return None
    u10, u30 = _up(a10, r10), _up(a30, r30)
    if u10 and u30:   return STATE_BROAD_UP
    if u10:           return STATE_CARRIED
    if u30:           return STATE_LAGGED
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

    RE-MEASURED 29 Sep 2026 ON THE CORRECTED BUCKETS. The bullets above were
    measured on a hand-ranked list with 3 Top-10 and 5 Next-10 names wrong. The
    same scripts re-run on NSE's own Top-10 / Top-20 membership (1,322 cash
    sessions, 1,173 F&O sessions):
      * the CARRIED state's gap edge vanishes: +0.2bps against a +5bps base;
      * top-10 breadth beside close-strength: t=-0.49 (CLR +3.81, corr +0.661);
        next-10 breadth is the cash panel's best (|t| 4.17, reality check
        p=0.041) and falls to t=+1.83 beside CLR;
      * F&O family reality check p=0.39;
      * the three-way gate drops to +27.6bps excess over 38 episodes, random
        blocks beat it 20% of the time, episode-t +0.50 -- it halved when three
        stocks changed bucket, which is what a subsample does;
      * ONE LEAD, NOT AN EDGE: top-10 put OI growing faster than call OI ->
        next 5 sessions +34bps (59% up) vs -7bps (50% up), every year positive,
        as large out of sample, t=+3.54 beside CLR and 1/5/20d momentum. Not
        significant across the F&O family, and at ~0.11 sd per disjoint week a
        fresh-data confirmation needs ~340 weeks. Recorded, not shipped.

    Survivorship applies here as it does to the concentration trend: the basket
    is TODAY's 50 applied to history.

    BANK NIFTY, FIN NIFTY, MIDCAP SELECT -- re-measured, not inherited
    (scripts/ilc_multi_index_backtest.py). Same null: every family survivor is
    close-strength again (Bank Top-2 futures -> gap |t| 3.48 alone, +1.72 beside
    CLR; Fin 3.68 -> +1.75; Midcap family p=0.056, and 0.038 on the 30-Sep-2026
    basket with the best falling to +1.15). Two index-specific caveats:
      * the SEBI cap (top name <= 20%, top 3 <= 45%) cut the Top-2 bucket's
        implied share of Bank Nifty moves from ~0.55 (2022-2025) to 0.39 by
        Jun-2026, Fin Nifty's to 0.33 -- older base-rate rows describe a more
        top-heavy index than today's;
      * Midcap Select starts at history_from (29 Nov 2024), when most of its
        current heavyweights got stock F&O -- ~450 sessions, so every row is thin.
    """
    meta = index_meta(fno_symbol, as_of)
    if meta is None:
        return pd.DataFrame()
    buckets = meta["buckets"]
    # first bucket = heavyweights, last = tail, by POSITION (see IndexLargeCap.state)
    top10 = tuple(list(buckets.values())[0])
    rest30 = tuple(list(buckets.values())[-1])
    all_syms = tuple(s for b in buckets.values() for s in b)
    ph = ", ".join("?" * len(all_syms))
    anchor = as_of or date.today()
    start = anchor - timedelta(days=365 * years + 30)
    if meta.get("history_from"):
        start = max(start, meta["history_from"])
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
    # equal-weight bucket returns, only to break exact-half ties (see _up)
    r10 = R[[x for x in top10 if x in R.columns]].mean(axis=1)
    r30 = R[[x for x in rest30 if x in R.columns]].mean(axis=1)
    # forward returns. Compounded in log space so a 5-day figure is a real return.
    lg = np.log1p(y / 100.0)
    fwd5 = np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100.0
    fwd1 = y.shift(-1)
    gap = (idx["open_val"] / idx["close_val"].shift(1) - 1).reindex(ix).shift(-1) * 100
    k = pd.DataFrame({"a10": a10, "a30": a30, "r10": r10, "r30": r30,
                      "gap": gap, "d1": fwd1, "d5": fwd5}).dropna()
    if k.empty:
        return pd.DataFrame()
    k["state"] = [_classify(x, z, rx, rz) for x, z, rx, rz
                  in zip(k["a10"], k["a30"], k["r10"], k["r30"])]
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


# ── flow analogues ───────────────────────────────────────────────────────────

_Z_WIN, _Z_MIN = 250, 150


def _flow_state_matrix(fno_symbol: str, as_of: date, years: int = 5,
                       with_raw: bool = False) -> tuple:
    """Causal 9-dim flow state per session, plus the index's forward returns.

    Returns (Z, fwd) indexed by trade_date. Z is standardised against each dim's
    OWN trailing 250-session mean/sd, SHIFTED, so day t never sees itself.

    min_periods on those rolling stats is load-bearing, not a default. The futures
    and options legs are absent on every settlement session (forward OI jumps
    mechanically there), which punches ~5% holes into those series. pandas
    defaults min_periods to the window, so a 250-row window holding ~12 gaps NEVER
    qualifies -- every z came back NaN and the state matrix was silently EMPTY,
    while the gap-free delivery legs looked perfectly healthy.
    """
    def _empty():
        # every early exit has to honour with_raw, or the 3-tuple caller unpacks
        # a 2-tuple and dies on an unknown index / an empty table
        e = pd.DataFrame()
        return (e, e, e) if with_raw else (e, e)

    meta = index_meta(fno_symbol, as_of)
    if meta is None:
        return _empty()
    buckets = meta["buckets"]
    all_syms = tuple(s for b in buckets.values() for s in b)
    ph = ", ".join("?" * len(all_syms))
    start = as_of - timedelta(days=365 * years + 420)

    cash = query_dataframe(f"""
        SELECT trade_date, symbol, deliv_per,
               (close_price - prev_close) / NULLIF(prev_close, 0) * 100 AS r
        FROM daily_data WHERE symbol IN ({ph}) AND series IN ('EQ','SM','ST')
          AND close_price > 0 AND prev_close > 0
          AND trade_date >= ? AND trade_date <= ?
    """, [*all_syms, start, as_of])
    # Day-over-day forward OI in SQL, NOT ~1,100 per-day pandas merges. The merge
    # version cost 19.8s of a 20.2s panel render; this is the same arithmetic in
    # about a second.
    #
    # Same-expiry matching is BY CONSTRUCTION here: LAG pairs each contract with
    # ITSELF on its prior session, so a position migrating from the near contract
    # into the next nets to zero rather than reading as an unwind. A contract with
    # no prior row (poi NULL) drops out of BOTH sums, which keeps the two days over
    # an identical contract set -- the property the merge version got by joining.
    oi = query_dataframe(f"""
        WITH f AS (
            SELECT trade_date, symbol, expiry_date, instrument, option_type,
                   SUM(open_interest) AS oi
            FROM fno_bhavcopy
            WHERE instrument IN ('FUTSTK','OPTSTK') AND open_interest > 0
              AND symbol IN ({ph}) AND expiry_date > trade_date
              AND trade_date >= ? AND trade_date <= ?
            GROUP BY 1,2,3,4,5
        ), l AS (
            SELECT *, LAG(oi) OVER (PARTITION BY symbol, expiry_date, instrument,
                                                 option_type
                                    ORDER BY trade_date) AS poi
            FROM f
        )
        SELECT trade_date, symbol, instrument, option_type,
               SUM(oi) AS oi, SUM(poi) AS poi
        FROM l WHERE poi > 0
        GROUP BY 1,2,3,4
    """, [*all_syms, start, as_of])
    idx = query_dataframe("""
        SELECT trade_date, pct_chg FROM index_data
        WHERE index_name = ? AND trade_date >= ? AND trade_date <= ?
    """, [meta["index_name"], start, as_of])
    if cash.empty or oi.empty or idx.empty:
        return _empty()

    for df in (cash, oi, idx):
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    cash = cash[cash["r"].abs() < 40]
    R = cash.pivot_table("r", "trade_date", "symbol").sort_index()
    DL = cash.pivot_table("deliv_per", "trade_date", "symbol").sort_index()

    settle = set(pd.to_datetime(query_dataframe(
        "SELECT DISTINCT trade_date FROM fno_bhavcopy WHERE instrument='FUTSTK'"
        " AND expiry_date = trade_date AND trade_date >= ?",
        [start])["trade_date"]))
    oi = oi[~oi["trade_date"].isin(settle)].copy()
    oi["pct"] = ((oi["oi"] - oi["poi"]) / oi["poi"] * 100).clip(-50, 50)
    FOI = oi[oi["instrument"] == "FUTSTK"].pivot_table("pct", "trade_date", "symbol")
    _o = oi[oi["instrument"] == "OPTSTK"]
    CE = _o[_o["option_type"] == "CE"].pivot_table("pct", "trade_date", "symbol")
    PE = _o[_o["option_type"] == "PE"].pivot_table("pct", "trade_date", "symbol")
    if FOI.empty or CE.empty or PE.empty:
        return _empty()

    raw = {}
    for bn, mem in buckets.items():
        cols = [s for s in mem if s in R.columns]
        sub = R[cols]
        # min_periods must be derived from the window, never a literal: it was
        # hardcoded at 50, so shortening _DELIV_BASE to 21 would have made
        # min_periods exceed the window and turned EVERY z into NaN, silently
        # emptying the analogue state matrix.
        mu = DL[cols].rolling(_DELIV_BASE, min_periods=_DELIV_MINP).mean().shift(1)
        sd = DL[cols].rolling(_DELIV_BASE, min_periods=_DELIV_MINP).std().shift(1)
        raw[f"{bn} delivery"] = (((DL[cols] - mu) / sd.where(sd > 1e-9))
                                 .replace([np.inf, -np.inf], np.nan).mean(axis=1))
        raw[f"{bn} futures"] = (np.sign(sub) *
                                FOI[[s for s in cols if s in FOI.columns]]
                                .reindex(sub.index)).mean(axis=1)
        raw[f"{bn} options"] = (
            PE[[s for s in cols if s in PE.columns]].reindex(sub.index)
            - CE[[s for s in cols if s in CE.columns]].reindex(sub.index)).mean(axis=1)
    X = pd.DataFrame(raw).sort_index()
    if meta.get("history_from"):
        # cut BEFORE standardising, so no pre-cutoff session leaks into the
        # 250-session mean/sd that today's z is measured against
        X = X[X.index >= pd.Timestamp(meta["history_from"])]
    Z = ((X - X.rolling(_Z_WIN, min_periods=_Z_MIN).mean().shift(1)) /
         X.rolling(_Z_WIN, min_periods=_Z_MIN).std().shift(1)
         ).replace([np.inf, -np.inf], np.nan).dropna()

    y = idx.set_index("trade_date")["pct_chg"].astype(float).sort_index()
    lg = np.log1p(y / 100.0)
    fwd = pd.DataFrame({
        "d1": y.shift(-1),
        "d5": np.expm1(lg.iloc[::-1].rolling(5).sum().iloc[::-1].shift(-1)) * 100.0,
    })
    ix = Z.index.intersection(fwd.index)
    if with_raw:
        # RAW legs as well: the panel shows the actual delivery z / OI %% numbers
        # for each matched session so the reader can SEE that it matched, rather
        # than trusting a standardised distance.
        return Z.reindex(ix), fwd.reindex(ix), X.reindex(ix)
    return Z.reindex(ix), fwd.reindex(ix)


def get_flow_analogues(trade_date: date, fno_symbol: str = "NIFTY",
                       k: int = 25, purge: int = 5) -> dict:
    """The k past sessions whose flow state most resembles `trade_date`, and what
    NIFTY did after them.

    A DESCRIPTIVE LOOKUP. The matching is real and the outcomes are real; the
    inference from one to the other is what does not hold.

    AUDITED (scripts/nifty_flow_analogues.py, 947 sessions 2022-2026, walk-forward
    with a purge gap of `horizon` sessions so a neighbour never shares forward days
    with the day being predicted):
      * The matching genuinely WORKS. The 25th-nearest neighbour sits at L1
        distance 5.40 against 9.28 for two random days, and only 7.1% of random
        pairs are closer. These are real analogues, not arbitrary days.
      * AND IT BUYS NOTHING. |mean(25 NEAREST) - actual| = 1.444% against
        |mean(25 FARTHEST) - actual| = 1.462%. The most similar days forecast the
        outcome no better than the LEAST similar ones. That one comparison is the
        whole verdict: similarity in this state space is not similarity in outcome.
      * k nearest vs k RANDOM past days: 5d hit 54.5% vs 54.3% at k=50, and at
        k=25 on 1d the RANDOM control WINS (51.2% vs 49.7%). No stability in k.
      * No subspace rescues it -- 1-dim, 3-dim and 9-dim all null, and the best
        cell (options-only, 5d, IC +0.074) flips to IC -0.013 at 1d.
      * Recency does not help (last 250 / 500 sessions ~ all history).
      * Beside the index's own close-strength the analogue dies: t +0.67 at 1d and
        +0.21 at 5d, against CLR's +2.78.
    The state space is effectively 5.6-dimensional (participation ratio 5.63 of 9),
    so this is not simply the curse of dimensionality -- the neighbours really are
    near. They just do not share a future.

    RE-MEASURED 29 Sep 2026 on NSE's own Top-10 / Top-20 buckets, 968 sessions:
    nearest 1.389% vs farthest 1.470%; 25 nearest 49.1% vs 25 random 51.2% at 1d;
    beside CLR t -0.11 (1d) and +0.27 (5d). Same verdict.
    """
    out = {"ok": False, "note": "", "k": k, "purge": purge, "spot": None,
           "matches": pd.DataFrame(), "summary": {}}
    Z, fwd, RAW = _flow_state_matrix(fno_symbol, trade_date, with_raw=True)
    if Z.empty:
        out["note"] = "Not enough history to build the flow state."
        return out
    ts = pd.Timestamp(trade_date)
    if ts not in Z.index:
        out["note"] = (f"No usable flow state for {trade_date:%d %b %Y}. The "
                       "futures and options legs need a non-settlement session "
                       "with a prior session to compare against.")
        return out
    # Strictly the PAST, and then a PURGE GAP on top of it.
    #
    # The audit backtest purged neighbours within `horizon` sessions of the target;
    # the live lookup originally did not, and the asymmetry mattered: measured over
    # 126 sampled sessions, 34.1% had at least one of their 25 "analogues" within
    # 10 CALENDAR days, and the closest match was 1 day away at the extreme. A
    # neighbour from last week shares most of its forward window with this one, so
    # it is autocorrelation wearing the costume of an analogue. Same rule here as
    # in the backtest.
    past = Z.index[Z.index < ts]
    if purge > 0 and len(past) > purge:
        past = past[:-purge]
    if len(past) < 60:
        out["note"] = "Fewer than 60 prior sessions carry a flow state."
        return out
    cur = Z.loc[ts]
    dist = (Z.loc[past] - cur).abs().sum(axis=1).sort_values()
    sel = dist.index[:k]
    out["matches"] = pd.DataFrame({
        "date": sel,
        "distance": dist.loc[sel].values,
        "d1": fwd.loc[sel, "d1"].values,
        "d5": fwd.loc[sel, "d5"].values,
    })
    # the actual numbers on each matched session, so the match is inspectable
    for c in RAW.columns:
        out["matches"][c] = RAW.loc[sel, c].values
    out["today"] = {c: float(RAW.loc[ts, c]) for c in RAW.columns}
    out["dims"] = list(RAW.columns)
    out["ok"] = True
    out["spot"] = _index_close(INDEX_BUCKETS[fno_symbol]["index_name"], trade_date)
    _sp = out["spot"]
    # Points are quoted on TODAY's level, not the level that prevailed when the
    # analogue happened: NIFTY ran ~17,000 in 2022 and ~23,900 now, so a raw
    # historical point move is not comparable across the window.
    _pts = (lambda x: (x / 100.0 * _sp) if _sp else None)
    for c in ("d1", "d5"):
        out["matches"][c + "_pts"] = (out["matches"][c] / 100.0 * _sp
                                      if _sp else np.nan)
    base = fwd.loc[past]
    for h in ("d1", "d5"):
        v = out["matches"][h].dropna()
        b = base[h].dropna()
        out["summary"][h] = dict(
            n=int(len(v)),
            mean=float(v.mean()) if len(v) else None,
            up=float((v > 0).mean() * 100) if len(v) else None,
            base_mean=float(b.mean()), base_up=float((b > 0).mean() * 100),
            best=float(v.max()) if len(v) else None,
            worst=float(v.min()) if len(v) else None,
            mean_pts=_pts(float(v.mean())) if len(v) else None,
            best_pts=_pts(float(v.max())) if len(v) else None,
            worst_pts=_pts(float(v.min())) if len(v) else None,
            base_mean_pts=_pts(float(b.mean())))
    med = float(dist.median())
    out["summary"]["match_quality"] = dict(
        kth=float(dist.loc[sel].max()), median_random=med,
        ratio=float(dist.loc[sel].max() / med) if med else None)
    # K-SENSITIVITY. k=25 is a choice, not a measurement, and the headline moves
    # with it: on 3 Sep 2026 the next-day mean ran -0.095% at k=10 and +0.149% at
    # k=40, flipping SIGN, while "up %" swung 48-60%. Publishing the sweep is the
    # only honest way to show a single number that is not stable in its own
    # parameter -- the panel renders the RANGE, not just the k=25 point.
    sweep = {}
    for kk in (10, 15, 25, 40, 60):
        if kk > len(dist):
            continue
        ss = dist.index[:kk]
        sweep[kk] = {h: dict(
            mean=float(fwd.loc[ss, h].dropna().mean()),
            up=float((fwd.loc[ss, h].dropna() > 0).mean() * 100))
            for h in ("d1", "d5")}
    out["summary"]["k_sweep"] = sweep
    for h in ("d1", "d5"):
        ms = [v[h]["mean"] for v in sweep.values()]
        out["summary"][h]["k_range"] = (min(ms), max(ms))
        out["summary"][h]["k_sign_flips"] = bool(min(ms) < 0 < max(ms))
    return out


def _match_block(Z: pd.DataFrame, fwd: pd.DataFrame, ts, k: int, purge: int,
                 cols: list, spot: Optional[float]) -> Optional[dict]:
    """One nearest-neighbour match over `cols`, priced in % AND in index POINTS.

    POINTS ARE QUOTED ON TODAY'S LEVEL, not on the level that prevailed when the
    analogue happened. Nifty ran ~17,000 in 2022 and ~23,900 now, so a raw
    historical point move is not comparable across the window and would make old
    analogues look small. `pct/100 * spot` answers the question actually being
    asked: if that same move repeated FROM HERE, how many points is it?
    """
    past = Z.index[Z.index < ts]
    if purge > 0 and len(past) > purge:
        past = past[:-purge]
    if len(past) < 60:
        return None
    sub = Z[cols]
    dist = (sub.loc[past] - sub.loc[ts]).abs().sum(axis=1).sort_values()
    sel = dist.index[:k]
    base = fwd.loc[past]
    res = {"n": int(len(sel)), "kth": float(dist.loc[sel].max()),
           "median_random": float(dist.median()), "dates": list(sel)}
    for h in ("d1", "d5"):
        v = fwd.loc[sel, h].dropna()
        b = base[h].dropna()
        if v.empty:
            res[h] = None
            continue
        pts = (lambda x: (x / 100.0 * spot) if spot else None)
        res[h] = dict(
            n=int(len(v)), mean=float(v.mean()), up=float((v > 0).mean() * 100),
            best=float(v.max()), worst=float(v.min()),
            base_mean=float(b.mean()), base_up=float((b > 0).mean() * 100),
            mean_pts=pts(float(v.mean())), best_pts=pts(float(v.max())),
            worst_pts=pts(float(v.min())),
            base_mean_pts=pts(float(b.mean())))
    # k-sensitivity: k is a choice, and on at least one worked example the
    # next-day mean flips sign between k=10 and k=40.
    sweep = {}
    for kk in (10, 15, 25, 40, 60):
        if kk > len(dist):
            continue
        ss = dist.index[:kk]
        sweep[kk] = {h: dict(mean=float(fwd.loc[ss, h].dropna().mean()),
                             up=float((fwd.loc[ss, h].dropna() > 0).mean() * 100))
                     for h in ("d1", "d5")}
    res["k_sweep"] = sweep
    for h in ("d1", "d5"):
        if res.get(h) is None or not sweep:
            continue
        ms = [v[h]["mean"] for v in sweep.values()]
        res[h]["k_range"] = (min(ms), max(ms))
        res[h]["k_sign_flips"] = bool(min(ms) < 0 < max(ms))
    return res


def get_bucket_analogues(trade_date: date, fno_symbol: str = "NIFTY",
                         k: int = 25, purge: int = 5) -> dict:
    """Per-bucket analogues: match on ONE bucket's three legs at a time.

    Answers "when the Top 10 looked like this, what did NIFTY do next?" separately
    from the same question about the Next 10 and the Rest 30, instead of requiring
    all nine dimensions to line up at once.

    STILL DESCRIPTIVE. The 3-dim-per-bucket variant is one of the subspaces the
    audit already priced (scripts/nifty_flow_analogues.py): TOP10-only matching
    returns IC -0.008 and a 50.0% hit rate at 1 day, IC -0.025 and 49.7% at 5
    days. Narrowing the state space does not rescue the null -- it was never the
    dimensionality that was wrong, since the neighbours are demonstrably near
    (25th match at distance 5.40 against 9.28 for a random pair) and still share
    no future.
    """
    out = {"ok": False, "note": "", "k": k, "purge": purge,
           "spot": None, "buckets": {}}
    meta = index_meta(fno_symbol, trade_date)
    if meta is None:
        out["note"] = f"No bucket map for {fno_symbol}."
        return out
    Z, fwd = _flow_state_matrix(fno_symbol, trade_date)
    if Z.empty:
        out["note"] = "Not enough history to build the flow state."
        return out
    ts = pd.Timestamp(trade_date)
    if ts not in Z.index:
        out["note"] = (f"No usable flow state for {trade_date:%d %b %Y}. The "
                       "futures and options legs need a non-settlement session "
                       "with a prior session to compare against.")
        return out
    out["spot"] = _index_close(meta["index_name"], trade_date)
    for bn in meta["buckets"]:
        cols = [c for c in Z.columns if c.startswith(f"{bn} ")]
        if len(cols) != 3:
            continue
        blk = _match_block(Z, fwd, ts, k, purge, cols, out["spot"])
        if blk is not None:
            out["buckets"][bn] = blk
    out["ok"] = bool(out["buckets"])
    if not out["ok"] and not out["note"]:
        out["note"] = "Fewer than 60 prior sessions carry a flow state."
    return out


def _index_close(index_name: str, trade_date: date) -> Optional[float]:
    """Closing level, used only to express historical % moves in today's points."""
    df = query_dataframe(
        "SELECT close_val FROM index_data WHERE index_name = ? AND trade_date = ?",
        [index_name, trade_date])
    if df.empty or pd.isna(df.iloc[0, 0]):
        return None
    return float(df.iloc[0, 0])
