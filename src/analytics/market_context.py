"""
Market context for the next 1-4 weeks — a CONDITIONS MONITOR, not a forecast.

WHY THIS IS NOT A PREDICTOR
---------------------------
Asked for a tab that calls the next week/month from FII positioning. Tested
2026-08-12 over 2018-2026 (2,205 FII sessions) and the forecast is not in the
data:

* Continuous signal: 12 FII features x 3 horizons, Newey-West t at lag=horizon.
  Best |t| = 2.13, and 36 tests need |t| > 3.2 to mean anything.
* Tail/extreme analysis, done properly — expanding (not in-sample) decile
  thresholds, signal lagged one session because NSE publishes the participant
  file AFTER the close, and a max-statistic block permutation so that searching
  24 candidates is priced in: **p = 0.068, does not survive**. Searching that
  many candidates on shuffled data typically produces a 1.43pp "finding", which
  is LARGER than the honest effect that was found.
* It also decays: FII index-option positioning in its top decile was worth
  +1.54pp / 2wk in 2018-21, +0.89pp in 2022-24, and +0.33pp with a 51.2% hit
  rate (base 49.4%) from Nov-2024 on.
* An 8.5-year re-test of classic FII long-share had already returned IC ~ 0.

So this module reports STATE and BASE RATES. It deliberately emits no direction,
no probability of "uptrend", and no target. See the sibling memory note
project_market_next_month_fii for the full audit trail.

THE BASE RATE IS THE POINT
--------------------------
Nifty has risen 56.6% / 58.5% / 61.8% of all 1-week / 2-week / 1-month windows
since 2018. Any claim about "next month up" has to beat 62% before it has said
anything at all. Showing that number first is most of this tab's value.

A DATA TRAP THIS MODULE AVOIDS
------------------------------
SEBI raised F&O lot sizes and cut weekly expiries from Nov-2024. Median FII
`fut_idx_long` went 156,356 (2024) -> 66,012 (2025) -> 27,460 (2026), and the
share of days with a >50% jump went from 0.4-8% (2018-24) to 27.2% (2025). Any
multi-year z-score on CONTRACT COUNTS partly measures that rule change. Ratios
(long share of OI) are immune; the count-based measures are therefore z-scored
on a short 60-session window, which is locally comparable, and never compared
across the break.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["get_market_context", "HORIZONS"]

# 1 week / 2 weeks / 1 month in trading sessions
HORIZONS: dict[str, int] = {"1 week": 5, "2 weeks": 10, "1 month": 21}

_Z_WIN = 60          # sessions; short enough to survive the Nov-2024 units break
_MIN_HIST = 250      # sessions before an expanding percentile means anything

# Feature label -> (column, plain-English reading of a HIGH value)
_FEATURES = {
    "FII index-futures long share": (
        "fut_long_share",
        "Share of FII index-futures open interest that is long. FIIs run "
        "structurally short here (they hedge cash holdings), so the level is "
        "usually low — what matters is where it sits versus its own recent range."),
    "FII index-option positioning": (
        "opt_bull",
        "Calls held long plus puts written, minus calls written and puts held "
        "long. Higher = FII index-option book leans more bullish."),
    "FII net all-derivatives": (
        "net_all",
        "Total FII long open interest minus total short, across index and stock "
        "futures and options."),
}


def _index(name: str) -> pd.Series:
    df = query_dataframe(
        """SELECT trade_date, close_val FROM index_data
           WHERE index_name = ? ORDER BY trade_date""", [name])
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.drop_duplicates("trade_date")
              .set_index("trade_date")["close_val"].astype(float))


def _nifty() -> pd.Series:
    df = query_dataframe(
        """SELECT trade_date, close_val FROM index_data
           WHERE index_name = 'Nifty 50' ORDER BY trade_date""")
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.drop_duplicates("trade_date")
              .set_index("trade_date")["close_val"].astype(float))


def _fii() -> pd.DataFrame:
    p = query_dataframe(
        """SELECT * FROM fao_participant
           WHERE client_type = 'FII' AND data_type = 'OI' ORDER BY trade_date""")
    if p.empty:
        return p
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.drop_duplicates("trade_date").set_index("trade_date")
    out = pd.DataFrame(index=p.index)
    tot = p["fut_idx_long"] + p["fut_idx_short"]
    out["fut_long_share"] = np.where(tot > 0, p["fut_idx_long"] / tot * 100, np.nan)
    out["opt_bull"] = (p["opt_idx_call_long"] + p["opt_idx_put_short"]
                       - p["opt_idx_call_short"] - p["opt_idx_put_long"])
    out["net_all"] = p["total_long"] - p["total_short"]
    return out


_ANALOGUE_K = 12
_ANALOGUE_MIN_SEP = 21     # sessions — see the docstring below

# The formation vector, defined ONCE. The calibration in _ANALOGUE_CAL was measured
# on exactly these columns; changing them invalidates it. An earlier version of this
# module quoted numbers measured on an 8-feature variant (it also carried market
# breadth), which produced entirely different neighbours — the displayed
# calibration described a model that was not running. Change this list and you must
# re-measure with `python -m scripts.audit_analogues`, which imports THIS module and
# refuses to print anything until its walk reproduces get_analogues exactly.
_ANALOGUE_FEATURES = ("ret_5d", "ret_21d", "vs_ma50", "vs_ma200",
                      "vol20", "dd_52w", "ma_slope")


def _formation(s: pd.Series) -> pd.DataFrame:
    """Build the analogue feature matrix. Single source of truth so a backtest
    cannot silently drift away from what the dashboard runs."""
    lr = np.log(s).diff()
    F = pd.DataFrame(index=s.index)
    F["ret_5d"] = s.pct_change(5) * 100
    F["ret_21d"] = s.pct_change(21) * 100
    F["vs_ma50"] = (s / s.rolling(50).mean() - 1) * 100
    F["vs_ma200"] = (s / s.rolling(200).mean() - 1) * 100
    F["vol20"] = lr.rolling(20).std() * np.sqrt(252) * 100
    F["dd_52w"] = (s / s.rolling(252).max() - 1) * 100
    F["ma_slope"] = s.rolling(50).mean().pct_change(10) * 100
    return F[list(_ANALOGUE_FEATURES)]


def _session_breaks(index_name: str = "Nifty 50") -> pd.DatetimeIndex:
    """Dates where index_data is MISSING one or more sessions immediately before.

    Detected by the exchange's own continuity: NSE publishes prev_close with every
    snapshot, so if prev_close(t) does not match close(t-1) as stored, at least one
    session between them never got ingested.

    THIS IS NOT HYPOTHETICAL. Nifty 50 has 14 such breaks over 2013-2026, and one
    of them is large: May-June 2017 holds 7 and 3 sessions against a normal 21-22,
    ~30 missing in all. That matters here because forward returns are taken with a
    POSITIONAL shift — `s.shift(-5)` means "5 stored rows", so across the 2017 hole
    2017-06-09 + 5 rows lands on 2017-07-04, and a 25-calendar-day move gets
    labelled "1 week". The values are real prices; it is the horizon label that is
    wrong, and wrong in the direction that inflates the min/max span.

    CLAUDE.md already warns never to join forward returns by positional shift
    across calendars. The case not anticipated there is that the index's OWN
    calendar has holes, so even a within-series shift is unsafe.

    Candidates whose forward window contains a break are dropped in get_analogues.
    """
    df = query_dataframe(
        """SELECT trade_date, close_val, prev_close FROM index_data
           WHERE index_name = ? AND close_val IS NOT NULL
             AND prev_close IS NOT NULL ORDER BY trade_date""", [index_name])
    if df.empty:
        return pd.DatetimeIndex([])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date").reset_index(drop=True)
    prior = df["close_val"].shift(1)
    rel = ((df["prev_close"] - prior).abs() / prior)
    return pd.DatetimeIndex(df.loc[(rel > 0.001) & prior.notna(), "trade_date"])


def _fii_formation(as_of: date, full: bool = False) -> pd.DataFrame:
    """FII positioning features for analogue matching. 2018+ only.

    Levels are z-scored on their own 60-session window rather than used raw,
    because SEBI's Nov-2024 lot-size change cut FII contract counts several-fold
    — a raw level would partly encode a rule change rather than positioning.
    FII CASH FLOW IS DELIBERATELY EXCLUDED: it starts 2024-06 and covers ~16% of
    sessions, so including it would silently discard 84% of the matchable history.
    """
    p = query_dataframe(
        """SELECT * FROM fao_participant
           WHERE client_type='FII' AND data_type='OI' AND trade_date <= ?
           ORDER BY trade_date""", [as_of])
    if p.empty:
        return pd.DataFrame()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.drop_duplicates("trade_date").set_index("trade_date")
    tot = p["fut_idx_long"] + p["fut_idx_short"]
    F = pd.DataFrame(index=p.index)
    F["fii_fut_share"] = np.where(tot > 0, p["fut_idx_long"] / tot * 100, np.nan)
    F["fii_opt_bull"] = (p["opt_idx_call_long"] + p["opt_idx_put_short"]
                         - p["opt_idx_call_short"] - p["opt_idx_put_long"])
    F["fii_net_all"] = p["total_long"] - p["total_short"]
    if full:
        # The remaining FII areas, for matching on POSITIONING ALONE. Kept out of
        # the price+fii blend on purpose: that mode's calibration was measured with
        # three, and silently widening it would make the quoted numbers describe a
        # model that no longer runs — the drift bug already caught twice here.
        F["fii_fut_net_chg"] = (p["fut_idx_long"] - p["fut_idx_short"]).diff()
        v = query_dataframe(
            """SELECT trade_date, total_long, total_short FROM fao_participant
               WHERE client_type='FII' AND data_type='Vol' AND trade_date <= ?
               ORDER BY trade_date""", [as_of])
        if not v.empty:
            v["trade_date"] = pd.to_datetime(v["trade_date"])
            v = v.drop_duplicates("trade_date").set_index("trade_date")
            F["fii_vol_intensity"] = v["total_long"] + v["total_short"]
        ds = query_dataframe(
            """SELECT trade_date, buy_value_cr, sell_value_cr
               FROM fii_derivatives_stats
               WHERE category='INDEX FUTURES' AND trade_date <= ?
               ORDER BY trade_date""", [as_of])
        if not ds.empty:
            ds["trade_date"] = pd.to_datetime(ds["trade_date"])
            ds = ds.drop_duplicates("trade_date").set_index("trade_date")
            F["fii_stats_net"] = ds["buy_value_cr"] - ds["sell_value_cr"]
    for c in list(F.columns):
        F[c] = ((F[c] - F[c].rolling(60, min_periods=30).mean())
                / F[c].rolling(60, min_periods=30).std())
    return F


_ANALOGUE_MODE_NOTE = {
    "price": "Matches on price structure alone. Reaches back to 2013.",
    "price+fii": ("Matches on price structure AND FII positioning. Only reaches "
                  "back to 2018, since that is when participant data starts."),
    "fii": ("Matches on FII POSITIONING ONLY — price is ignored entirely. Six "
            "areas: futures long share, day change in net futures, options tilt, "
            "total net position, traded volume, and index-futures rupee flow. "
            "2018+, and every area has 98.6% coverage."),
}

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION — PER MODE. Re-measured 2026-08-15.
#
# WHY THIS IS NOW A DICT. It used to be a single flat blob emitted for all three
# modes, so the dashboard quoted the PRICE model's walk-forward numbers under the
# "Price + FII" and "FII only" buttons as well. That is the third time this exact
# drift has happened in this module. The structure now makes it impossible: every
# displayed figure is keyed by the mode that produced it, and a mode with no
# measurement carries None rather than borrowing its neighbour's.
#
# HOW IT WAS MEASURED. Walk-forward over the COMMON window all three modes share,
# 1,573 sessions 2020-02-28..2026-07-15, rebuilding the matcher from scratch at
# every date (causal standardisation, same k/sep/metric as ships here).
# Base rate on that window: 57.0 / 59.2 / 64.3 % up at 1wk / 2wk / 1mo.
#
# THE RULE IS TOP-20% OF EACH MODE'S OWN VOTE, NOT AN ABSOLUTE >=70%. The vote can
# only take 13 values (k=12) and its spread differs by mode — 10 price+FII features
# disperse the matches and compress the vote toward 0.5. An absolute 70% cut
# therefore fired on 30% of days in fii mode and 15% in price+fii, and comparing
# modes at different selectivity is what produced the earlier "price+fii is worst"
# reading. It reverses at matched selectivity.
#
# `t` IS THE HAC t OF THE RETURN DIFFERENCE (return regressed on the signal dummy,
# Newey-West at lag = horizon) — NOT the t of the signal-day return level, which is
# just the equity risk premium and reads ~2.6 for buy-and-hold.
#
# NOTHING HERE IS SIGNIFICANT. Every |t| < 1.5. A max-statistic block permutation
# over the 9 mode x horizon candidates gives p = 0.430: the best real result
# (+4.40pp, fii at two weeks) sits essentially ON the null median (+4.09pp), i.e.
# it is the size of thing this search produces from shuffled data as a matter of
# course. The full audit tried ~410 variants in all (thresholds, k/sep grids,
# regime splits, anchor grids, consensus rules), which puts the required |t| near
# 3.6. See scripts/audit_analogues.py.
_ANALOGUE_CAL = {
    "price": {
        "bull_hit": (59.2, 58.3, 60.3), "bull_edge": (2.2, -0.9, -4.0),
        "bull_n": (573, 520, 335), "bull_t": (0.09, -0.09, 0.19),
        "bear_hit": (60.2, 65.8, 67.4), "bear_edge": (3.3, 6.7, 3.2),
        "bear_n": (372, 366, 473),
        "ctl_edge": (-0.2, -0.4, -1.4), "ctl_sd": (1.4, 2.6, 1.8),
        "ic": (0.037, -0.002, -0.029),
        "gap_guard": 114, "gap_noguard": 1,
        "sens_lo": 58, "sens_hi": 80,
        "span_cov_1m": 85.4, "span_width_1m": 14.34,
        "span_ic": (0.093, 0.097, -0.004),
    },
    "price+fii": {
        "bull_hit": (57.6, 60.9, 65.8), "bull_edge": (0.7, 1.8, 1.5),
        "bull_n": (517, 443, 546), "bull_t": (0.01, 0.08, 0.21),
        "bear_hit": (57.5, 58.1, 64.3), "bear_edge": (0.5, -1.0, 0.0),
        "bear_n": (381, 430, 588),
        "ctl_edge": (-0.4, -1.6, -1.6), "ctl_sd": (1.8, 2.0, 1.5),
        "ic": (0.037, 0.012, -0.010),
        "gap_guard": 66, "gap_noguard": 2,
        "sens_lo": 38, "sens_hi": 75,
        "span_cov_1m": 88.0, "span_width_1m": 15.79,
        "span_ic": (0.079, 0.055, 0.025),
    },
    "fii": {
        "bull_hit": (57.9, 63.6, 62.2), "bull_edge": (1.0, 4.4, -2.1),
        "bull_n": (504, 316, 487), "bull_t": (-0.42, 0.70, -1.14),
        "bear_hit": (61.8, 57.6, 67.3), "bear_edge": (4.8, -1.5, 3.0),
        "bear_n": (387, 576, 330),
        "ctl_edge": (-1.0, -0.6, -2.0), "ctl_sd": (1.9, 2.0, 2.0),
        "ic": (-0.036, -0.005, -0.050),
        "gap_guard": 74, "gap_noguard": 28,
        "sens_lo": 35, "sens_hi": 56,
        "span_cov_1m": 89.8, "span_width_1m": 16.45,
        "span_ic": (0.035, 0.008, 0.083),
    },
}

# Facts that genuinely are shared across the three modes.
_ANALOGUE_SHARED = {
    "window_days": 1573, "window": "2020-02-28..2026-07-15",
    "base_hit": (57.0, 59.2, 64.3),
    "rule": "top 20% of that mode's own vote",
    "perm_p": 0.430, "perm_null_median": 4.09, "perm_obs_best": 4.40,
    "n_candidates": 9, "n_candidates_full": 410,
    # THE SPAN IS CALIBRATED BUT NOT INFORMATIVE. Coverage lands at 85-90% against
    # the 84.6% that 12 independent draws imply, so the width is honest. But raced
    # against 20-day trailing volatility for predicting the realised |move|, the
    # span carries little: Spearman IC +0.093 / +0.097 / -0.004 for the span
    # against +0.243 / +0.251 / +0.328 for plain vol20, and in a joint regression
    # only vol20 survives. The vol-scaled band shown higher up this same tab uses
    # vol20 directly and is narrower at comparable coverage. Do not present the
    # span as an independent range product. (price-mode figures; the other two
    # modes are weaker still — see scripts/audit_analogues.py.)
    # vol20_ic stays shared: it is the same volatility series in every mode.
    # The span IC does NOT — it lives per mode in _ANALOGUE_CAL["span_ic"].
    "vol20_ic": (0.243, 0.251, 0.328),
}


def get_analogues(as_of: date, k: int = _ANALOGUE_K,
                  mode: str = "price") -> dict:
    """
    Past days whose market FORMATION most resembles `as_of`, and what followed.

    THE GUARD THAT MAKES THIS MEANINGFUL. Nearest neighbours in a price series are
    usually adjacent days, because markets are autocorrelated. Without a separation
    rule the 12 "most similar days in history" sit a WALK-FORWARD MEDIAN of 1
    session apart in price mode — twelve consecutive days of one episode, whose
    forward windows overlap almost entirely. That is one observation presented as
    twelve. Analogues must therefore be >= _ANALOGUE_MIN_SEP sessions apart.

    The guard's effect is mode-dependent and is now recorded per mode in
    _ANALOGUE_CAL (`gap_noguard` / `gap_guard`), because a single pair of numbers
    was being quoted under all three buttons and was wrong under two of them:

        price      median gap 1  -> 72 sessions   (guard essential)
        price+fii  median gap 2  -> 66 sessions   (guard essential)
        fii        median gap 28 -> 74 sessions   (guard barely binds)

    CALIBRATION — READ _ANALOGUE_CAL, NOT THIS DOCSTRING. Every figure the view
    displays is keyed by mode there, with the measurement window, the selection
    rule and the HAC t-statistic attached. The short version: on 1,573 common
    sessions the bull-vote edge over base runs -5.1pp to +2.4pp depending on mode
    and horizon, every |HAC t| is below 1.5, a max-statistic permutation over the
    9 mode x horizon candidates returns p = 0.550 with the observed best BELOW the
    null median, and the full audit's ~410 candidates put the required |t| near
    3.6. The vote carries no usable direction in any mode.

    THREE THINGS THAT LIMIT HOW THE OUTPUT SHOULD BE READ:

    * MATCH DISTANCE DOES NOT RANK RELIABILITY. A tighter match is not a more
      trustworthy one. The distance is also NOT comparable across modes: it is a
      Mahalanobis distance in 7 / 10 / 6 dimensions, so the same number means
      different things on different buttons. Compare it within one mode only.
    * THE OUTPUT MOVES WITH THE PARAMETERS, more than previously stated. Across
      defensible k (8-20) and separation (10-42), today's one-month up-share spans
      58-80% in price mode, 38-75% in price+fii and 35-56% in fii. Quoting a single
      figure as precise would be false.
    * THE SPREAD IS CALIBRATED BUT NOT INFORMATIVE. Coverage is 85-90% against the
      84.6% that 12 independent draws imply, so the width is honest — but raced
      against 20-day trailing volatility it adds nothing (see _ANALOGUE_SHARED).
      The vol-scaled band higher up the tab is the better range product.

    Causal: standardisation uses only prior data, matching only looks backwards, a
    candidate is skipped unless its own forward window has closed, and candidates
    whose forward window straddles a hole in index_data are skipped as well.

    NOT LAGGED. The FII participant file publishes AFTER the close, so the fii and
    price+fii formations for date D are not knowable at D's close. Forward returns
    here start at D's close regardless. Shifting entry one session moves the
    measured edges by up to 5pp with no stable sign, which is one more reason the
    numbers above are treated as descriptive rather than tradeable.
    """
    out: dict = {"ok": False, "as_of": as_of, "error": ""}
    s = _index("Nifty 50")
    if s.empty:
        out["error"] = "Nifty history unavailable."
        return out
    s = s[s.index <= pd.Timestamp(as_of)]
    if len(s) < 800:
        out["error"] = "Not enough history for analogue matching."
        return out

    F = _formation(s)
    if mode in ("price+fii", "fii"):
        fx = _fii_formation(as_of, full=(mode == "fii"))
        if fx.empty:
            out["error"] = "No FII participant data for FII matching."
            return out
        # mode "fii" drops price entirely — the match is on POSITIONING ONLY,
        # which is a different question from "what did the chart look like".
        F = fx.copy() if mode == "fii" else F.join(fx, how="left")
        F = F[F.index >= pd.Timestamp("2018-01-01")]
        # CALENDAR: the participant file and the index do not share a session list.
        # Reindexing prices onto participant dates and taking shift(-h) ROWS was a
        # real bug — measured 2026-08-14, 13 of 36 displayed forward returns were
        # wrong (2019-09-27 one-month showed +0.62% against a true +3.17%, and one
        # match, 2019-03-04, was not an index session at all). Two causes: ffill
        # invented prices for non-trading dates, and h rows stopped meaning h
        # sessions wherever the calendars diverged. Fix: keep only dates that are
        # genuine index sessions, and compute forward returns on the TRUE index
        # calendar, joined BY DATE (below) rather than by position.
        F = F[F.index.isin(s.index)]
        if len(F) < 700:
            out["error"] = "Not enough 2018+ history for FII matching."
            return out

    i = len(F) - 1
    cur = F.iloc[i]
    hist = F.iloc[:i].dropna()
    if cur.isna().any() or len(hist) < 500:
        out["error"] = "Formation features incomplete for this date."
        return out
    # MAHALANOBIS, not plain Euclidean. Audited 2026-08-14: 5 of the 7 features are
    # trend/momentum and they are heavily correlated (ret_21d~vs_ma50 0.87,
    # vs_ma200~dd_52w 0.86; mean |corr| 0.53 among the trend block). Only TWO
    # eigenvalues exceed 1 and the first component carries 60% of the variance, so
    # unweighted Euclidean distance counts momentum ~5x and volatility once —
    # "closest setup" would really mean "closest momentum". Mahalanobis divides out
    # the covariance so each independent direction counts once. It matters: the two
    # metrics agree on only 4 of 12 matches for the same day.
    mu, sd = hist.mean(), hist.std().replace(0, np.nan)
    zh = ((hist - mu) / sd).values
    zc = ((cur - mu) / sd).values
    _cov = np.cov(zh, rowvar=False)
    VI = np.linalg.pinv(_cov)               # pinv: safe if features are collinear
    _d = zh - zc
    dist = pd.Series(np.sqrt(np.clip(np.einsum("ij,jk,ik->i", _d, VI, _d), 0, None)),
                     index=hist.index).sort_values()

    # Forward returns on the TRUE index calendar, indexed BY DATE. `s` is the full
    # index series regardless of which feature calendar F uses, so shift(-h) here
    # always means h genuine trading sessions. Looked up by date below — never by
    # position into F, which is what produced wrong numbers in fii mode.
    fwd = {h: ((s.shift(-h) / s - 1) * 100) for h in HORIZONS.values()}

    # Positions whose forward window straddles a hole in index_data — see
    # _session_breaks. `shift(-h)` counts STORED rows, so across a hole "5 rows"
    # is not 5 sessions and the displayed horizon label is wrong.
    _hmax = max(HORIZONS.values())
    _breaks = _session_breaks()
    _bad_pos: set[int] = set()
    for _b in _breaks:
        if _b in s.index:
            _bp = s.index.get_loc(_b)
            _bad_pos.update(range(max(0, _bp - _hmax), _bp))
    _n_dropped = 0

    picked: list[tuple[int, float]] = []
    for dt_ in dist.index:
        # the candidate's own forward window must have completed on the INDEX
        # calendar, not merely be N rows back in the feature frame
        if dt_ not in s.index:
            continue
        _spos = s.index.get_loc(dt_)
        if _spos + _hmax >= len(s):
            continue
        if _spos in _bad_pos:
            _n_dropped += 1
            continue
        pos = F.index.get_loc(dt_)
        if any(abs(pos - p) < _ANALOGUE_MIN_SEP for p, _ in picked):
            continue
        picked.append((pos, float(dist[dt_])))
        if len(picked) >= k:
            break

    rows = []
    for pos, dd in picked:
        dt_ = F.index[pos]
        # BY DATE, not by position: F may be on the participant calendar while the
        # price series is on the exchange calendar.
        rec = {"date": dt_.date(), "distance": dd, "close": float(s.loc[dt_])}
        for name, h in HORIZONS.items():
            rec[name] = float(fwd[h].loc[dt_])
        rows.append(rec)
    df = pd.DataFrame(rows)

    summary = []
    for name in HORIZONS:
        if name in df:
            v = df[name].dropna()
            summary.append({
                "horizon": name, "n": int(len(v)),
                "up_share_pct": float((v > 0).mean() * 100),
                "median_pct": float(v.median()),
                "worst_pct": float(v.min()), "best_pct": float(v.max()),
            })

    # THE DATE THE FORMATION ACTUALLY CAME FROM, not the last index session.
    # In "fii" mode F sits on the participant calendar, which is missing 8 index
    # sessions since 2018 — every Budget special session among them (2024-01-20,
    # 2024-05-18, 2025-02-01, 2026-02-01 ...). On those dates the matcher reads the
    # PREVIOUS participant row while the panel used to caption it as today.
    # "price+fii" already errored out on the same gap; only "fii" served it
    # silently. Report the real date and let the view flag the staleness, exactly
    # as the index-level block below already does with `is_stale`.
    _form_date = F.index[i].date()
    _last_session = s.index[-1].date()
    cal = _ANALOGUE_CAL.get(mode, {})

    out.update({"ok": True, "as_of": _last_session, "analogues": df,
                "summary": pd.DataFrame(summary),
                "meta": {"k": len(df), "min_sep": _ANALOGUE_MIN_SEP,
                         "mode": mode, "mode_note": _ANALOGUE_MODE_NOTE.get(mode, ""),
                         "n_features": int(F.shape[1]),
                         "history_from": F.index.min().date(),
                         # formation provenance
                         "formation_date": _form_date,
                         "is_stale": _form_date != _last_session,
                         # candidates skipped because their forward window
                         # straddled a hole in index_data (see _session_breaks)
                         "dropped_gap_candidates": _n_dropped,
                         "n_session_breaks": int(len(_breaks)),
                         # PER-MODE calibration. Never fall back to another mode's
                         # numbers — an absent measurement must read as absent.
                         "cal": dict(cal),
                         "shared": dict(_ANALOGUE_SHARED),
                         "has_cal": bool(cal)}})
    return out


def get_market_context(as_of: date) -> dict:
    """
    Returns {"ok", "as_of", "base_rates", "state", "analogue", "meta"}.

    Causal: reads only sessions on or before `as_of`. Nothing here forecasts.
    """
    out: dict = {"ok": False, "as_of": as_of, "error": ""}
    nif, fii = _nifty(), _fii()
    if nif.empty or fii.empty:
        out["error"] = "Nifty or FII participant data unavailable."
        return out

    d = fii.join(nif.rename("nifty"), how="inner").sort_index()
    d = d[d.index <= pd.Timestamp(as_of)]
    if len(d) < _MIN_HIST:
        out["error"] = "Not enough history to establish a base rate."
        return out
    asof_ts = d.index.max()

    # ── forward returns, for base rates only (they end before as_of) ──────────
    full = fii.join(nif.rename("nifty"), how="inner").sort_index()
    for h in HORIZONS.values():
        full[f"fwd{h}"] = full["nifty"].shift(-h) / full["nifty"] - 1
    hist = full[full.index <= asof_ts]

    base_rates = []
    for name, h in HORIZONS.items():
        s = hist[f"fwd{h}"].dropna()
        base_rates.append({
            "horizon": name,
            "up_rate_pct": float((s > 0).mean() * 100) if len(s) else np.nan,
            "mean_pct": float(s.mean() * 100) if len(s) else np.nan,
            "median_pct": float(s.median() * 100) if len(s) else np.nan,
            "n": int(len(s)),
        })

    # ── current state: z-score on a SHORT window + percentile vs trailing 2y ──
    state = []
    for label, (col, why) in _FEATURES.items():
        s = d[col].dropna()
        if len(s) < _Z_WIN + 5:
            continue
        z = (s - s.rolling(_Z_WIN).mean()) / s.rolling(_Z_WIN).std()
        trail = s.tail(500)
        pct = float((trail < s.iloc[-1]).mean() * 100)
        prev = z.iloc[-6] if len(z) > 6 else np.nan
        state.append({
            "feature": label, "why": why,
            "value": float(s.iloc[-1]),
            "z60": float(z.iloc[-1]) if z.iloc[-1] == z.iloc[-1] else np.nan,
            "z60_5d_ago": float(prev) if prev == prev else np.nan,
            "pct_2y": pct,
        })

    # ── historical analogue, WITH the failed correction attached ──────────────
    analogue = []
    for label, (col, _why) in _FEATURES.items():
        s = d[col].dropna()
        if len(s) < _Z_WIN + _MIN_HIST:
            continue
        z = ((s - s.rolling(_Z_WIN).mean()) / s.rolling(_Z_WIN).std()).shift(1)
        now = z.iloc[-1]
        if now != now:
            continue
        band = 0.5
        for name, h in HORIZONS.items():
            m = pd.concat([z.rename("z"), hist[f"fwd{h}"].rename("f")],
                          axis=1).dropna()
            if len(m) < 200:
                continue
            sel = m[(m["z"] >= now - band) & (m["z"] <= now + band)]
            if len(sel) < 30:
                continue
            analogue.append({
                "feature": label, "horizon": name,
                "n_similar": int(len(sel)),
                "up_rate_pct": float((sel["f"] > 0).mean() * 100),
                "mean_pct": float(sel["f"].mean() * 100),
                "base_up_pct": float((m["f"] > 0).mean() * 100),
                "base_mean_pct": float(m["f"].mean() * 100),
                "excess_pp": float((sel["f"].mean() - m["f"].mean()) * 100),
            })

    # ── the RANGE of outcomes, for Nifty and Bank Nifty ──────────────────────
    # This is the honest answer to "what can happen next month". Direction is not
    # forecastable here, but the DISTRIBUTION is stable and is the one product in
    # this codebase that measured calibrated (range coverage ~72%, vs point
    # targets which scored NEGATIVE skill). Percentiles are computed on history up
    # to as_of only, so nothing here peeks ahead.
    # Bands are VOL-SCALED, not raw historical percentiles. Each past outcome is
    # standardised by the 20-day realised volatility known BEFORE it, quantiles are
    # taken on those standardised outcomes, then rescaled by TODAY's volatility.
    # Calibration test over 2,756-2,800 expanding-window forecasts (2013-2026),
    # where an "80% band" should contain 80% of outcomes:
    #     Nifty 1m   : raw 83.2% @ 11.07pp wide  ->  vol-scaled 81.0% @ 10.24pp
    #     BankNifty1m: raw 87.6% @ 16.32pp wide  ->  vol-scaled 82.5% @ 14.14pp
    # Vol-scaling is better calibrated AND narrower — strictly better. Raw
    # percentiles are worst on Bank Nifty, quoting an 87.6% band as "80%", because
    # they blend the COVID crash into today's regime.
    # ── where both indices actually closed, and today's move ────────────────
    # Reported per index with its OWN last traded date: an index can be missing a
    # session (holiday handling, a late feed), and silently showing a stale close
    # as "today" would be worse than showing the real date it belongs to.
    levels = []
    for idx_name in ("Nifty 50", "Nifty Bank"):
        s = _index(idx_name)
        if s.empty:
            continue
        s = s[s.index <= asof_ts]
        if len(s) < 2:
            continue
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        levels.append({
            "index": idx_name,
            "date": s.index[-1].date(),
            "close": last,
            "prev_close": prev,
            "chg_pts": last - prev,
            "chg_pct": (last / prev - 1) * 100 if prev else np.nan,
            "is_stale": s.index[-1].date() != asof_ts.date(),
        })

    ranges = []
    for idx_name in ("Nifty 50", "Nifty Bank"):
        s_full = _index(idx_name)
        if s_full.empty:
            continue
        s = s_full[s_full.index <= asof_ts]
        if len(s) < 500:
            continue
        spot = float(s.iloc[-1])
        lr = np.log(s).diff()
        vol20 = float(lr.tail(20).std())
        if not (vol20 > 0):
            continue
        for name, h in HORIZONS.items():
            r = (s.shift(-h) / s - 1).dropna() * 100
            if len(r) < 200:
                continue
            v_lag = lr.rolling(20).std().shift(1)
            std = (r / (v_lag * np.sqrt(h) * 100)).replace([np.inf, -np.inf],
                                                           np.nan).dropna()
            if len(std) < 200:
                continue
            k = vol20 * np.sqrt(h) * 100          # today's scale, in %
            qs = {p: float(std.quantile(p)) * k for p in (0.10, 0.25, 0.5, 0.75, 0.90)}
            ranges.append({
                "index": idx_name, "horizon": name, "n": int(len(std)),
                "spot": spot,
                "p10": qs[0.10], "p25": qs[0.25], "median": qs[0.5],
                "p75": qs[0.75], "p90": qs[0.90],
                # the same thing in points and levels — what actually gets traded
                "lvl_p10": spot * (1 + qs[0.10] / 100),
                "lvl_p25": spot * (1 + qs[0.25] / 100),
                "lvl_p75": spot * (1 + qs[0.75] / 100),
                "lvl_p90": spot * (1 + qs[0.90] / 100),
                "pts_p25": spot * qs[0.25] / 100,
                "pts_p75": spot * qs[0.75] / 100,
                "pts_p10": spot * qs[0.10] / 100,
                "pts_p90": spot * qs[0.90] / 100,
                "typical_swing_pts": spot * (qs[0.75] - qs[0.25]) / 200,
                "up_rate_pct": float((r > 0).mean() * 100),
                "worst": float(r.min()), "best": float(r.max()),
            })

    out.update({
        "ok": True, "as_of": asof_ts.date(),
        "base_rates": pd.DataFrame(base_rates),
        "levels": pd.DataFrame(levels),
        "ranges": pd.DataFrame(ranges),
        "state": pd.DataFrame(state),
        "analogue": pd.DataFrame(analogue),
        "meta": {
            "fii_start": fii.index.min().date(),
            "fii_sessions": int(len(d)),
            "z_window": _Z_WIN,
            # the single most important disclosure on this tab
            "reality_check_p": 0.068,
            "n_candidates_searched": 24,
            "null_max_excess_pp": 1.43,
            # Widened test (2026-08-12): FII OI + FII VOLUME + FII derivative
            # rupee stats + market breadth + delivery acceleration + sector trend
            # and dispersion = 13 features, 78 candidates. Adding features did not
            # find an edge, it raised the noise floor: the best real result
            # (1.64pp) came in BELOW the median of what shuffled data produces at
            # that search intensity (1.72pp).
            "wide_candidates": 78,
            "wide_p": 0.560,
            "wide_best_pp": 1.64,
            "wide_null_median_pp": 1.72,
        },
    })
    return out
