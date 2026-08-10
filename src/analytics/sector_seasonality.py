"""
Sector calendar seasonality — month-wise best/worst, with honest confidence.

WHAT THIS IS
    A DESCRIPTIVE map of how each sector has behaved in each calendar month,
    measured as excess return vs the equal-weight sector basket, plus a causal
    (walk-forward) suggestion for the month that is about to start.

WHAT THIS IS NOT
    A validated edge. A full study (scripts/study_sector_seasonality.py,
    2013-2026) found that across 548 sector-month cells:
      - ZERO survive a Benjamini-Hochberg FDR control
      - ZERO survive a permutation max-|t| control over the grid
      - the naive-significant count is LOWER than pure noise produces
      - a walk-forward monthly rotation nets -0.8% to -3.2%/yr after costs
    The one exception is financials-over-defensives in October (and its July
    mirror). Everything else here is history, not forecast. The tier column
    exists so the UI can say so on every single cell.

THREE CONTAMINATIONS CORRECTED (the shipped aggregation has all three)
    1. LISTING POPS. On a listing day bhavcopy `prev_close` is the IPO ISSUE
       PRICE, so day 1 books the listing gain as a return (WINSOL 75->383 =
       +411%). Previous close is taken from the symbol's OWN prior session and
       that session must be within 7 days, so new listings contribute nothing.
    2. SAME-DAY TURNOVER WEIGHTING. Weighting today's return by today's turnover
       is mechanically upward biased -- a stock that moves hard trades hard on
       the SAME day. Weight is the trailing 20-session mean turnover, LAGGED.
    3. UNADJUSTED CORPORATE ACTIONS. Splits/bonuses produce fake -50% bars;
       per-stock daily returns are winsorized at +/-20%.
    Together these took the mean sector return from +0.534%/day (134%/yr, absurd)
    to +0.059%/day (14.9%/yr, plausible).

CAUSALITY
    Every function takes `as_of` and uses ONLY sessions strictly before it. The
    next-month suggestion at date D is built from months that completed before D.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = [
    "get_sector_month_map",
    "get_next_month_suggestion",
    "get_seasonality_track_record",
    "MONTH_NAMES",
]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_WINSOR = 20.0          # per-stock daily return cap, % (corporate actions)
_MIN_SESSIONS = 13      # a month with fewer sessions is not a month
_MIN_OBS = 5            # minimum years before a cell is reportable
_N_PERM = 800           # permutation draws for the grid-wide critical value
_SEED = 20260807

# NSE sector index <-> DCM canonical bucket, for cross-lens confirmation.
# The NSE indices carry 13.6yr of exchange-computed history vs DCM's 8yr, so
# agreement across the two is materially stronger evidence than either alone.
_CROSS_LENS = {
    "Automobile": "Nifty Auto",
    "Banking": "Nifty Bank",
    "IT": "Nifty IT",
    "Pharma & Healthcare": "Nifty Pharma",
    "FMCG": "Nifty FMCG",
    "Metals & Mining": "Nifty Metal",
    "Realty": "Nifty Realty",
    "Media & Entertainment": "Nifty Media",
    "Financial Services": "Nifty Financial Services",
    "Oil & Gas": "Nifty Oil & Gas",
    "Infrastructure": "Nifty Infrastructure",
    "Consumer Durables": "Nifty Consumer Durables",
    "Power & Utilities": "Nifty Energy",
}


# ----------------------------------------------------------------- loading ---
def _daily_sector_returns(as_of: date) -> pd.DataFrame:
    """Corrected bottom-up daily sector returns, strictly before `as_of`."""
    sql = f"""
    with base as (
        select b.trade_date, b.symbol, b.close_price,
               lag(b.close_price) over (partition by b.symbol order by b.trade_date) pc,
               lag(b.trade_date)  over (partition by b.symbol order by b.trade_date) pd,
               avg(b.turnover_lacs) over (partition by b.symbol order by b.trade_date
                    rows between 20 preceding and 1 preceding) w_lag
        from daily_data b
        where b.series in ('EQ','SM','ST') and b.trade_date < ?
    ),
    ok as (
        select trade_date, symbol, w_lag,
               greatest(least((close_price - pc) / pc * 100.0, {_WINSOR}), -{_WINSOR}) as r
        from base
        where pc > 0 and pd is not null
          and date_diff('day', pd, trade_date) <= 7
          and w_lag > 0
    )
    select o.trade_date, s.sector,
           sum(o.r * o.w_lag) / nullif(sum(o.w_lag), 0) as r
    from ok o
    inner join v_sector_master s on o.symbol = s.symbol
    where s.sector is not null and s.sector not in ('ETF','Others')
    group by 1, 2
    """
    df = query_dataframe(sql, [as_of])
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot(index="trade_date", columns="sector", values="r").sort_index() / 100.0


_NSE_SECTOR_IDX = [
    "Nifty Auto", "Nifty Bank", "Nifty FMCG", "Nifty IT", "Nifty Media",
    "Nifty Metal", "Nifty Pharma", "Nifty PSU Bank", "Nifty Realty",
    "Nifty Financial Services", "Nifty Private Bank", "Nifty Energy",
    "Nifty Infrastructure", "Nifty Commodities", "Nifty India Consumption",
    "Nifty PSE", "Nifty CPSE", "Nifty MNC", "Nifty Services Sector",
    "Nifty Oil & Gas", "Nifty Consumer Durables", "Nifty Healthcare Index",
]


def _nse_index_returns(as_of: date, full: bool = False) -> pd.DataFrame:
    names = sorted(set(_NSE_SECTOR_IDX if full else _CROSS_LENS.values()))
    ph = ",".join("?" for _ in names)
    sql = (f"select trade_date, index_name, close_val from index_data "
           f"where index_name in ({ph}) and close_val is not null and trade_date < ?")
    df = query_dataframe(sql, [*names, as_of])
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.pivot(index="trade_date", columns="index_name", values="close_val").sort_index()


def _to_monthly(px: pd.DataFrame) -> pd.DataFrame:
    """Month-end to month-end % returns, with thin-month and gap guards."""
    if px.empty:
        return pd.DataFrame()
    per = px.index.to_period("M")
    cnt = pd.Series(1, index=px.index).groupby(per).sum()
    thin = cnt[cnt < _MIN_SESSIONS].index
    if len(thin):
        # drop the thin month AND the month after it: the latter's return is
        # measured from a stale anchor and would silently span two months.
        px = px[~per.isin(set(thin) | {p + 1 for p in thin})]
    if px.empty:
        return pd.DataFrame()
    me = px.groupby([px.index.year, px.index.month]).tail(1)
    r = me.pct_change() * 100.0
    r.index = pd.to_datetime(me.index)
    ords = r.index.to_period("M").astype("int64").to_numpy()
    return r[np.r_[False, np.diff(ords) == 1]]


# ------------------------------------------------------------------- stats ---
def _tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def _perm_critical_t(ex: pd.DataFrame, n_perm: int = _N_PERM) -> float:
    """
    95th percentile of the GRID-WIDE max |t| under the null that month labels
    carry no information. A cell must beat this to be called STRONG -- this is
    the control that stops "best of 288 cells" being reported as a discovery.
    Fully vectorised: permute every sector's own history independently, then get
    all 12 month t-stats for every sector via one matrix product per draw.
    """
    R = ex.dropna(axis=0, how="any")
    if R.shape[0] < 24 or R.shape[1] < 2:
        return float("inf")
    A = R.to_numpy(float)                       # (T, S)
    months = R.index.month.to_numpy()
    T, S = A.shape
    M = np.zeros((T, 12))
    M[np.arange(T), months - 1] = 1.0
    cnt = M.sum(axis=0)[:, None]                # (12,1)
    valid = (cnt >= 3).ravel()
    rng = np.random.default_rng(_SEED)
    out = np.empty(n_perm)
    for i in range(n_perm):
        # independent permutation per column
        P = A[np.argsort(rng.random((T, S)), axis=0), np.arange(S)]
        s1 = M.T @ P                            # (12, S)
        s2 = M.T @ (P * P)
        mean = s1 / cnt
        var = (s2 - cnt * mean ** 2) / np.maximum(cnt - 1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = mean / np.sqrt(var / cnt)
        out[i] = np.nanmax(np.abs(t[valid]))
    return float(np.percentile(out, 95))


# -------------------------------------------------------------- public API ---
def get_sector_month_map(as_of: date, min_obs: int = _MIN_OBS) -> tuple[pd.DataFrame, dict]:
    """
    Full sector x month grid using only data before `as_of`.

    Returns (grid, meta). Grid columns:
        sector, month, m, n, mean, median, hit, t, tier, cross (bool|None)
    `tier` is STRONG / WEAK / NOISE -- see _perm_critical_t. `cross` is True when
    the matching NSE sector index shows the same sign with |t| >= 1.5 over its
    longer history, None when no NSE counterpart exists.
    """
    daily = _daily_sector_returns(as_of)
    if daily.empty:
        return pd.DataFrame(), {"error": "no sector data"}

    px = (1.0 + daily.fillna(0.0)).cumprod()
    monthly = _to_monthly(px)
    if monthly.empty or len(monthly) < 24:
        return pd.DataFrame(), {"error": "insufficient month history"}

    bench = monthly.mean(axis=1)                       # equal-weight sector basket
    ex_raw = monthly.sub(bench, axis=0)
    # de-mean per sector: isolates the MONTH effect from the sector's own drift,
    # otherwise a persistently strong sector looks good in every month.
    ex = ex_raw.sub(ex_raw.mean(axis=0), axis=1)

    crit = _perm_critical_t(ex)

    # cross-lens: NSE indices, longer history, independent construction
    cross_t: dict[tuple[str, int], float] = {}
    try:
        npx = _nse_index_returns(as_of)
        if not npx.empty:
            nm = _to_monthly(npx)
            nb = nm.mean(axis=1)
            nex = nm.sub(nb, axis=0)
            nex = nex.sub(nex.mean(axis=0), axis=1)
            for dcm_s, nse_s in _CROSS_LENS.items():
                if nse_s not in nex.columns:
                    continue
                v = nex[nse_s]
                for m in range(1, 13):
                    x = v[v.index.month == m].dropna().values
                    if len(x) >= min_obs:
                        cross_t[(dcm_s, m)] = _tstat(x)
    except Exception:                                   # noqa: BLE001
        cross_t = {}

    rows: list[dict[str, Any]] = []
    for sec in ex.columns:
        v = ex[sec]
        for m in range(1, 13):
            x = v[v.index.month == m].dropna().values
            if len(x) < min_obs:
                continue
            t = _tstat(x)
            tier = ("STRONG" if abs(t) >= crit else "WEAK" if abs(t) >= 2.0 else "NOISE")
            ct = cross_t.get((sec, m))
            cross = None
            if ct is not None and not np.isnan(t):
                cross = bool(np.sign(ct) == np.sign(t) and abs(ct) >= 1.5)
            rows.append({
                "sector": sec, "month": MONTH_NAMES[m - 1], "m": m, "n": len(x),
                "mean": float(x.mean()), "median": float(np.median(x)),
                "hit": float(100.0 * (x > 0).mean()), "t": t,
                "tier": tier, "cross": cross,
            })

    grid = pd.DataFrame(rows)
    meta = {
        "crit_t": crit,
        "months": int(len(monthly)),
        "first": monthly.index.min(),
        "last": monthly.index.max(),
        "sectors": int(ex.shape[1]),
        "n_strong": int((grid.tier == "STRONG").sum()) if not grid.empty else 0,
        "n_weak": int((grid.tier == "WEAK").sum()) if not grid.empty else 0,
        "n_cells": int(len(grid)),
    }
    return grid, meta


def get_next_month_suggestion(as_of: date, top_k: int = 4,
                              min_obs: int = _MIN_OBS) -> dict:
    """
    Causal suggestion for the month that is about to start (or the current one,
    if `as_of` is in its first days). Built ONLY from months completed before
    `as_of`. Always returns the tier so the UI can qualify it.
    """
    grid, meta = get_sector_month_map(as_of, min_obs=min_obs)
    if grid.empty:
        return {"error": meta.get("error", "unavailable")}

    # If we are past the 20th, the useful question is the NEXT month.
    ts = pd.Timestamp(as_of)
    target = ts + pd.offsets.MonthBegin(1) if ts.day >= 20 else ts
    m = int(target.month)

    sub = grid[grid.m == m].copy()
    if sub.empty:
        return {"error": f"no history for {MONTH_NAMES[m - 1]}"}
    sub = sub.sort_values("mean", ascending=False)

    best = sub.head(top_k).to_dict("records")
    worst = sub.tail(top_k).iloc[::-1].to_dict("records")
    return {
        "month": MONTH_NAMES[m - 1],
        "month_num": m,
        "target_period": target.strftime("%b %Y"),
        "is_next": bool(ts.day >= 20),
        "best": best,
        "worst": worst,
        "n_strong": int((sub.tier == "STRONG").sum()),
        "n_weak": int((sub.tier == "WEAK").sum()),
        "n_cross": int(sub["cross"].fillna(False).sum()),
        "meta": meta,
    }


def get_seasonality_track_record(as_of: date, top_k: int = 4,
                                 cost_bps_per_side: float = 25.0,
                                 lens: str = "dcm",
                                 mode: str = "month",
                                 since: str | None = None) -> dict:
    """
    What this rule has ACTUALLY been worth, walked forward.

    At each month t (needing >=3 prior observations of that calendar month and
    >=36 months of history), rank sectors by their mean excess in month t using
    ONLY data before t, hold the top k, and record the realised excess. This is
    the number the UI must show next to any suggestion -- without it the tab is
    a confident-looking noise generator.

    RUN THIS FOR BOTH LENSES. Over the identical 2018-2026 window the DCM
    24-bucket lens returns about +9%/yr net while the NSE 22-index lens returns
    about -5%/yr. Same rule, same period, opposite sign -- the result is a
    property of the sector DEFINITION, not of the market. The UI must show both,
    on a MATCHED window (pass `since`), or the comparison is era vs era.

    `mode` selects what is being tested, and the control is not optional:
      "month"       rank by mean excess in THAT calendar month  (the seasonal rule)
      "demeaned"    same, minus each sector's own drift          (month effect ONLY)
      "persistence" rank by overall mean excess, IGNORING the month  (the CONTROL)

    Measured 2026-08-07 on DCM 2018+: month +9.4%/yr, persistence +9.0%/yr,
    demeaned +5.9%/yr. A rule that never looks at the calendar reproduces 96% of
    the headline, so most of "seasonality" here is sector PERSISTENCE. On NSE
    indices persistence is -10.0%/yr (t=-2.82) -- it reverses -- which points at
    DCM's constituent look-ahead (`v_sector_master` holds CURRENT tickers applied
    backwards) rather than a market effect. Always show "month" beside
    "persistence"; the difference between them is the only seasonal claim.
    """
    if lens == "nse":
        npx = _nse_index_returns(as_of, full=True)
        if npx.empty:
            return {"error": "no index data"}
        monthly = _to_monthly(npx)
    else:
        daily = _daily_sector_returns(as_of)
        if daily.empty:
            return {"error": "no data"}
        monthly = _to_monthly((1.0 + daily.fillna(0.0)).cumprod())
    if since:
        monthly = monthly[monthly.index >= pd.Timestamp(since)]
    if monthly.empty or len(monthly) < 40:
        return {"error": "insufficient history"}

    bench = monthly.mean(axis=1)
    ex = monthly.sub(bench, axis=0)

    gross, turn, hits = [], [], []
    prev: set[str] = set()
    for i, ts in enumerate(ex.index):
        hist = ex.iloc[:i]
        if len(hist) < 36:
            continue
        if mode == "persistence":
            # CONTROL: never looks at the calendar month.
            score = hist.mean(axis=0).dropna()
        else:
            hm = hist[hist.index.month == ts.month]
            if len(hm) < 3:
                continue
            score = hm.mean(axis=0)
            if mode == "demeaned":
                score = score - hist.mean(axis=0)
            score = score.dropna()
        avail = ex.iloc[i].dropna().index
        score = score[score.index.isin(avail)]
        if len(score) < top_k:
            continue
        pick = set(score.nlargest(top_k).index)
        realized = float(ex.iloc[i][list(pick)].mean())
        gross.append(realized)
        hits.append(realized > 0)
        turn.append(len(pick - prev) / top_k)
        prev = pick

    if not gross:
        return {"error": "not enough walk-forward months"}
    g = np.array(gross)
    tr = np.array(turn)
    cost = tr * 2.0 * (cost_bps_per_side / 100.0)
    net = g - cost
    return {
        "mode": mode, "lens": lens,
        "n_months": len(g),
        "gross_pm": float(g.mean()),
        "net_pm": float(net.mean()),
        "gross_t": _tstat(g),
        "net_t": _tstat(net),
        "hit": float(100.0 * np.mean(hits)),
        "turnover": float(tr.mean() * 100.0),
        "cost_pm": float(cost.mean()),
        "net_annual": float(net.mean() * 12.0),
    }
