"""Index valuation (P/E, P/B, dividend yield) — history, point-in-time percentiles,
bands, drawdown episodes, and the measured record of what each valuation zone
was followed by.

Question it answers: "is Nifty (or a sector) expensive or cheap right now, and
historically what came next when it was this expensive / this cheap?"

The live numbers (today's P/E, its percentile, the band chart, the episode list)
are computed here from `index_valuation`. The measured forward-return tables come
from `scripts/index_valuation_study.py`, cached at
data/index_valuation_study/summary.json, so the page never re-runs the backtest.

Three corrections are applied everywhere before any comparison across time:

1. METHODOLOGY BREAK. On 2021-03-31 NSE Indices moved P/E (and dividend yield)
   from standalone to consolidated figures. Nifty 50 P/E printed 40.43 -> 33.20
   (-18%) on a day the index fell 1%. Comparing today's P/E with pre-2021 P/E
   without adjusting treats that accounting change as a de-rating. Pre-break values
   are rescaled by the break-day ratio, per index and per metric (`splice_breaks`).
   The ratio of consolidated to standalone earnings was not constant before 2021,
   so the adjusted history is an approximation; P/B barely moved at the break.
   Where the ratio is beyond 0.70-1.43 (Auto P/E x3.65, Midcap 100 P/E x0.46, Media
   DY x0.10) the pre-break history is DROPPED, not rescaled, and that metric's
   percentile is ranked against post-2021 data only.
2. POINT IN TIME. "Top 10% of history" is computed only from the history that
   existed on that day (expanding window). A full-sample percentile uses the
   future's distribution: Indian P/E drifted up for two decades, so a full-sample
   percentile would call 2003 "cheap" using knowledge of 2021.
3. NOT COMPUTED != ZERO. A P/E of '-' means the index's combined earnings were
   negative. It is NULL, never 0 and never forward-filled.
"""
from __future__ import annotations

import bisect
import json
import os
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "BREAK_DATE", "load_valuation", "splice_breaks", "pit_percentile",
    "add_features", "gsec_monthly", "drawdown_episodes", "study_summary",
    "current_board", "valuation_zone", "index_names", "index_catalog", "index_group",
]

BREAK_DATE = date(2021, 3, 31)
_BREAK_WINDOW = (date(2021, 3, 24), date(2021, 4, 9))
_BREAK_MIN = 0.03          # |log jump| unexplained by price, below which no splice
# Above this the old and new series measure different things (Nifty Auto P/E x3.65,
# Nifty Media DY x0.10 on the break day): one day's ratio cannot bridge 20 years,
# so pre-break history is DROPPED for that index/metric instead of rescaled.
_BREAK_MAX = float(np.log(1.43))
PIT_MIN_OBS = 750          # ~3 years before a percentile is shown at all
BAND_WINDOW = 1260         # ~5 years: the analyst "5-year average ± 1σ" band

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUDY_PATH = os.path.join(_ROOT, "data", "index_valuation_study", "summary.json")


# Grouping for the page's index picker (NSE's own sectoral/thematic split).
_BROAD = {"Nifty 50", "Nifty Next 50", "Nifty 100", "Nifty 200", "Nifty 500",
          "Nifty Midcap 100", "Nifty Midcap 150", "Nifty Smallcap 100", "Nifty Smallcap 250",
          "Nifty Midcap 50", "Nifty Smallcap 50", "Nifty MidSmallcap 400",
          "Nifty LargeMidcap 250", "Nifty Total Market", "Nifty Microcap 250",
          "Nifty Smallcap 500", "Nifty Next 100"}
_THEMATIC = {"Nifty Energy", "Nifty Infrastructure", "Nifty India Consumption",
             "Nifty Commodities", "Nifty PSE", "Nifty MNC", "Nifty Services Sector",
             "Nifty CPSE", "Nifty Capital Markets", "Nifty India Defence", "Nifty Housing",
             "Nifty Core Housing", "Nifty Mobility", "Nifty India Digital",
             "Nifty India Manufacturing", "Nifty Transportation & Logistics",
             "Nifty India Tourism", "Nifty India Railways PSU",
             "Nifty EV & New Age Automotive", "Nifty MidSmall India Consumption",
             "Nifty India New Age Consumption", "Nifty REITs & Realty"}


def index_group(name: str) -> str:
    """Broad market / Sectoral / Thematic. Anything not listed is Sectoral."""
    return "Broad market" if name in _BROAD else "Thematic" if name in _THEMATIC else "Sectoral"


def index_catalog() -> pd.DataFrame:
    """Every index with valuation history: name, group, first/last day, sessions.
    Ordered for a picker: Nifty 50 first, then Broad, Sectoral, Thematic, A-Z."""
    df = query_dataframe("""SELECT index_name, MIN(trade_date) AS first_day,
                                   MAX(trade_date) AS last_day, COUNT(*) AS n
                            FROM index_valuation GROUP BY 1""")
    if df.empty:
        return df.assign(group=[])
    df["group"] = df["index_name"].map(index_group)
    order = {"Broad market": 0, "Sectoral": 1, "Thematic": 2}
    df["_o"] = df["group"].map(order) + (df["index_name"] != "Nifty 50") * 0.5
    return (df.sort_values(["_o", "index_name"]).drop(columns="_o")
              .reset_index(drop=True))


def index_names() -> list[str]:
    df = query_dataframe("""SELECT index_name, MIN(trade_date) AS first_day, COUNT(*) AS n
                            FROM index_valuation GROUP BY 1 ORDER BY n DESC""")
    return df["index_name"].tolist()


def load_valuation(index_name: str, as_of: date | None = None) -> pd.DataFrame:
    # P/B prefers index_data (captured from each day's snapshot CSV, i.e. what was
    # PUBLISHED that day). The history endpoint serves a REVISED book value for
    # 2023-03..2024-04 (1,716 index-days differ, Nifty 3.39 vs 3.37, some sectors
    # by 0.68); using the revision would let later information into the past.
    # P/E is NOT overridden: the snapshot is the corrupt one there (2019-08-05
    # printed Nifty MNC P/E 5.30 against 25.75).
    sql = """SELECT v.trade_date, v.pe, COALESCE(i.pb_ratio, v.pb) AS pb, v.dy,
                    v.close_val, v.tri
             FROM index_valuation v
             LEFT JOIN index_data i
               ON i.trade_date = v.trade_date AND lower(i.index_name) = lower(v.index_name)
                  AND i.pb_ratio > 0
             WHERE v.index_name = ?"""
    params: list = [index_name]
    if as_of is not None:
        sql += " AND v.trade_date <= ?"
        params.append(as_of)
    df = query_dataframe(sql + " ORDER BY v.trade_date", params)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.reset_index(drop=True)


def return_level(df: pd.DataFrame) -> pd.Series:
    """A total-return level built by CHAINING daily returns, never by mixing levels.

    niftyindices' TRI starts later than the price series and on a different scale
    (Nifty 50 TRI begins 1999-06-30 at 1.06x the price; Next 50 at 1.09x in Nov-2002)
    and has holes (Nifty 50 1999-10-23). "TRI where present, else close" therefore
    put a fake +6% / +9% jump into every window that crossed the switch. Here each
    day's return comes from TRI when TRI exists on both days, else from price."""
    tri, px = df["tri"].astype(float), df["close_val"].astype(float)
    r_tri = tri / tri.shift(1) - 1
    r_px = px / px.shift(1) - 1
    r = r_tri.where(tri.notna() & tri.shift(1).notna(), r_px).fillna(0.0)
    return (1 + r).cumprod()


def splice_breaks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add pe_adj / pb_adj / dy_adj with pre-2021-03-31 values rescaled.

    The break is located as the session inside the window whose metric jump is
    least explained by the price move: log(PE_t/PE_t-1) - log(P_t/P_t-1) for P/E
    and P/B (both scale with price), log(DY_t/DY_t-1) + log(P_t/P_t-1) for yield.
    """
    out = df.copy()
    info: dict = {}
    px = np.log(out["close_val"]).diff()
    win = (out["trade_date"].dt.date >= _BREAK_WINDOW[0]) & (
        out["trade_date"].dt.date <= _BREAK_WINDOW[1])
    for col, sign in (("pe", -1.0), ("pb", -1.0), ("dy", 1.0)):
        resid = np.log(out[col]).diff() + sign * px
        r = resid.where(win).dropna()
        adj = out[col].copy()
        if not r.empty and abs(r).max() >= _BREAK_MIN:
            i = r.abs().idxmax()
            factor = float(np.exp(r.loc[i]))
            dropped = abs(r.loc[i]) > _BREAK_MAX
            adj.loc[: i - 1] = np.nan if dropped else adj.loc[: i - 1] * factor
            info[col] = {"date": str(out.loc[i, "trade_date"].date()),
                         "factor": round(factor, 4), "dropped_pre_break": bool(dropped)}
        else:
            info[col] = None
        out[col + "_adj"] = adj
    return out, info


def pit_percentile(s: pd.Series, min_obs: int = PIT_MIN_OBS) -> pd.Series:
    """Share of all EARLIER non-null values strictly below today's, plus half the
    ties. Uses only the past: no value is ranked against anything after it."""
    hist: list[float] = []
    out = np.full(len(s), np.nan)
    for k, v in enumerate(s.to_numpy(dtype=float)):
        if np.isnan(v):
            continue
        if len(hist) >= min_obs:
            lo = bisect.bisect_left(hist, v)
            hi = bisect.bisect_right(hist, v)
            out[k] = (lo + 0.5 * (hi - lo)) / len(hist)
        bisect.insort(hist, v)
    return pd.Series(out, index=s.index)


@lru_cache(maxsize=1)
def _gsec_cached() -> pd.DataFrame:
    try:
        g = query_dataframe("SELECT source, month, value FROM gsec_yield_monthly")
    except Exception:
        return pd.DataFrame(columns=["month", "gsec", "source"])
    if g.empty:
        return pd.DataFrame(columns=["month", "gsec", "source"])
    g["month"] = pd.to_datetime(g["month"])
    w = g.pivot_table(index="month", columns="source", values="value")
    # OECD's 10y benchmark where it exists (2011-12+), IMF's bond yield before.
    gsec = w.get("OECD_FINMARK", pd.Series(dtype=float)).combine_first(
        w.get("IMF_IFS", pd.Series(dtype=float)))
    src = np.where(w.get("OECD_FINMARK", pd.Series(index=w.index)).notna(),
                   "OECD", "IMF")
    return pd.DataFrame({"month": w.index, "gsec": gsec.to_numpy(), "source": src})


def gsec_monthly() -> pd.DataFrame:
    return _gsec_cached().copy()


def _gsec_pit(dates: pd.Series) -> pd.Series:
    """Yield known on each date: the PREVIOUS month's average (a month's average
    is not known until it ends). Carried at most 4 months past the last print."""
    g = gsec_monthly()
    if g.empty:
        return pd.Series(np.nan, index=dates.index)
    g = g.assign(avail=g["month"] + pd.offsets.MonthBegin(1)).sort_values("avail")
    m = pd.merge_asof(pd.DataFrame({"d": dates.values}).sort_values("d"),
                      g[["avail", "gsec"]], left_on="d", right_on="avail",
                      direction="backward")
    stale = (m["d"] - m["avail"]).dt.days > 124
    m.loc[stale, "gsec"] = np.nan
    return pd.Series(m["gsec"].to_numpy(), index=dates.index)


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Spliced metrics, point-in-time percentiles, 5y bands, earnings yield gap."""
    out, info = splice_breaks(df)
    for c in ("pe", "pb", "dy"):
        a = out[c + "_adj"]
        out[c + "_pct"] = pit_percentile(a)
        m = a.rolling(BAND_WINDOW, min_periods=PIT_MIN_OBS).mean()
        sd = a.rolling(BAND_WINDOW, min_periods=PIT_MIN_OBS).std()
        out[c + "_mean5y"], out[c + "_sd5y"] = m, sd
        out[c + "_z5y"] = (a - m) / sd
    out["ey"] = 100.0 / out["pe_adj"]
    out["gsec"] = _gsec_pit(out["trade_date"])
    out["ey_gap"] = out["ey"] - out["gsec"]          # earnings yield - bond yield
    out["ey_gap_pct"] = pit_percentile(out["ey_gap"])
    # One score, high = expensive: the three percentiles averaged (DY inverted,
    # high yield = cheap; at least two must exist), then itself ranked point in
    # time, because an average of three percentiles rarely reaches 0.9 on its own.
    # Blend only metrics that cover the index's WHOLE history. Where a metric's
    # pre-2021 history was dropped (Realty/FMCG yield, Auto P/E...), mixing it in
    # after 2021 made the blend 2 metrics before and 3 after, so today's reading was
    # ranked against a differently built past. If fewer than two metrics span the
    # full history, fall back to all of them (post-2021 only, a short sample).
    if out.empty:                      # index not launched yet on the chosen date
        out["val_raw"] = out["val_score"] = pd.Series(dtype=float)
        out["eps"] = pd.Series(dtype=float)
        return out, info
    start = out["trade_date"].iloc[0] + pd.Timedelta(days=365)
    comp = {"pe": out["pe_pct"], "pb": out["pb_pct"], "dy": 1 - out["dy_pct"]}
    full = [k for k in comp if out.loc[out[k + "_adj"].notna(), "trade_date"].min() <= start]
    # One full-history metric (usually P/B: Midcap 100/150, Smallcap 100 lost P/E and
    # yield comparability in 2021) still beats a blank: P/B was the most robust metric
    # measured, and the page labels the zone "P/B only". None at all -> post-2021 blend.
    use = full if full else list(comp)
    parts = pd.concat([comp[k] for k in use], axis=1)
    raw = parts.mean(axis=1).where(parts.notna().sum(axis=1) >= min(2, len(use)))
    out.attrs["composite_parts"] = use
    out["val_raw"] = raw
    # Same 3-year floor as the components: where pre-2021 history was dropped, a
    # 250-session floor ranked the composite against ~18 months and printed
    # "Smallcap 250: Extreme expensive" off a sample too short to mean anything.
    out["val_score"] = pit_percentile(raw, min_obs=PIT_MIN_OBS)
    out["eps"] = out["close_val"] / out["pe_adj"]
    return out, info


def valuation_zone(p: float | None) -> str:
    if p is None or pd.isna(p):
        return "—"
    if p >= 0.90:
        return "Extreme expensive"
    if p >= 0.70:
        return "Expensive"
    if p > 0.30:
        return "Fair"
    if p > 0.10:
        return "Cheap"
    return "Extreme cheap"


def drawdown_episodes(close: pd.Series, dates: pd.Series,
                      threshold: float = 0.20) -> pd.DataFrame:
    """Every peak-to-trough fall of at least `threshold` on the closing level.

    An episode starts at the running peak, its trough is the lowest close before
    the level regains that peak, and it ends on recovery (or is open)."""
    c = close.to_numpy(dtype=float)
    d = pd.to_datetime(dates).to_numpy()
    eps = []
    peak_i, i, n = 0, 0, len(c)
    while i < n:
        if c[i] >= c[peak_i]:
            peak_i = i
            i += 1
            continue
        if c[i] / c[peak_i] - 1 <= -threshold:
            j = i
            trough_i = i
            while j < n and c[j] < c[peak_i]:
                if c[j] < c[trough_i]:
                    trough_i = j
                j += 1
            eps.append({"peak_i": peak_i, "trough_i": trough_i,
                        "recover_i": j if j < n else None,
                        "peak_date": pd.Timestamp(d[peak_i]).date(),
                        "trough_date": pd.Timestamp(d[trough_i]).date(),
                        "recover_date": pd.Timestamp(d[j]).date() if j < n else None,
                        "depth": c[trough_i] / c[peak_i] - 1,
                        "fall_sessions": trough_i - peak_i})
            peak_i = j if j < n else n - 1
            i = j
            continue
        i += 1
    return pd.DataFrame(eps)


def study_summary() -> dict | None:
    try:
        with open(STUDY_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def current_board(as_of: date | None = None) -> pd.DataFrame:
    """One row per index: latest P/E, P/B, DY, their point-in-time percentiles."""
    rows = []
    for name in index_names():
        df = load_valuation(name, as_of)
        if len(df) < PIT_MIN_OBS + 10:
            continue
        f, _ = add_features(df)
        last = f.iloc[-1]
        rows.append({
            "index_name": name, "as_of": last["trade_date"].date(),
            "history_from": f["trade_date"].iloc[0].date(),
            "pe": last["pe"], "pb": last["pb"], "dy": last["dy"],
            "pe_pct": last["pe_pct"], "pb_pct": last["pb_pct"], "dy_pct": last["dy_pct"],
            "pe_z5y": last["pe_z5y"], "val_score": last["val_score"],
            # how far back each percentile's comparison really reaches
            **{f"{c}_since": (f.loc[f[c + "_adj"].notna(), "trade_date"].min().date()
                              if f[c + "_adj"].notna().any() else None)
               for c in ("pe", "pb", "dy")},
            "zone": valuation_zone(last["val_score"]),
        })
    return pd.DataFrame(rows)
