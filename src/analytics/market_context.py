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

# The formation vector, defined ONCE. The quoted calibration below was measured on
# exactly these columns; changing them invalidates it. An earlier version of this
# module quoted numbers measured on an 8-feature variant (it also carried market
# breadth), which produced entirely different neighbours — the displayed
# calibration described a model that was not running. Keep this list and the
# calibration in the docstring in sync, or re-measure with
# scratchpad/analogue_audit.py, which imports THIS module rather than
# reimplementing it.
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


def get_analogues(as_of: date, k: int = _ANALOGUE_K) -> dict:
    """
    Past days whose market FORMATION most resembles `as_of`, and what followed.

    THE GUARD THAT MAKES THIS MEANINGFUL. Nearest neighbours in a price series are
    usually adjacent days, because markets are autocorrelated. Measured 2026-08-13:
    without a separation rule the 12 "most similar days in history" sat a median of
    **2 sessions apart** — twelve consecutive days of one episode, whose forward
    windows overlap almost entirely. That is one observation presented as twelve,
    and it manufactures a confident consensus out of nothing. Analogues here must be
    >= _ANALOGUE_MIN_SEP sessions apart, which lifts the median gap to 70 sessions.

    CALIBRATION — READ BEFORE TRUSTING AGREEMENT. Measured on THIS feature set
    (an earlier measurement used an 8-feature variant including market breadth and
    did not describe this model — different features give different neighbours).
    Walk-forward over 2,476 days:

        base rate                    57.8% / 60.1% / 64.0%   (1wk / 2wk / 1mo)
        >=70% of analogues rose ->   59.3% / 60.0% / 59.9%
        <=30% of analogues rose ->   63.9% / 74.1% / 76.5%   (n=133 / 81 / 51)
        Spearman IC of analogue median vs actual:
                                     +0.009 / -0.027 / -0.034

    The vote carries no usable direction: it beats the base rate slightly at one
    week, matches it at two, and trails it at one month — an inconsistent sign is
    the signature of noise, not of a weak effect. The bearish tail sits above the
    base rate at every horizon, but on few independent episodes and with heavily
    overlapping windows, so it is not claimed as a signal either. What IS
    informative is the SPREAD — how differently the market resolved from setups
    that looked alike.

    Causal: standardisation uses only prior data, matching only looks backwards,
    and a candidate is skipped unless its own forward window has closed.
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

    i = len(F) - 1
    cur = F.iloc[i]
    hist = F.iloc[:i].dropna()
    if cur.isna().any() or len(hist) < 500:
        out["error"] = "Formation features incomplete for this date."
        return out
    mu, sd = hist.mean(), hist.std().replace(0, np.nan)
    dist = np.sqrt((((hist - mu) / sd - (cur - mu) / sd) ** 2).sum(axis=1)).sort_values()

    fwd = {h: (s.shift(-h) / s - 1) * 100 for h in HORIZONS.values()}
    picked: list[tuple[int, float]] = []
    for dt_ in dist.index:
        pos = F.index.get_loc(dt_)
        if pos > i - max(HORIZONS.values()):        # forward window must have closed
            continue
        if any(abs(pos - p) < _ANALOGUE_MIN_SEP for p, _ in picked):
            continue
        picked.append((pos, float(dist[dt_])))
        if len(picked) >= k:
            break

    rows = []
    for pos, dd in picked:
        rec = {"date": F.index[pos].date(), "distance": dd,
               "close": float(s.iloc[pos])}
        for name, h in HORIZONS.items():
            rec[name] = float(fwd[h].iloc[pos])
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

    out.update({"ok": True, "as_of": s.index[-1].date(), "analogues": df,
                "summary": pd.DataFrame(summary),
                # Measured on THIS feature set — see the docstring. If the features
                # change, these must be re-measured or they describe a dead model.
                "meta": {"k": len(df), "min_sep": _ANALOGUE_MIN_SEP,
                         "median_gap_no_guard": 2, "median_gap_guard": 72,
                         "walk_forward_days": 2476,
                         "ic_1w": 0.009, "ic_2w": -0.027, "ic_1m": -0.034,
                         "bull_hit": (59.3, 60.0, 59.9),
                         "bear_hit": (63.9, 74.1, 76.5),
                         "bear_n": (133, 81, 51),
                         "base_hit": (57.8, 60.1, 64.0)}})
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
