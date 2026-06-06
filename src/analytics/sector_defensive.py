"""
Sector defensive metrics — for bear / downtrend capital protection.

WHY
---
The cross-sectional `accum_score` (RS + delivery) is a MOMENTUM/strength selector.
In a bull/trend that predicts forward returns well, but in a BEAR market the
highest-momentum sectors are usually the highest-BETA ones — they amplify the
fall. Buying the momentum leader in a downtrend is exactly how a "good signal"
still loses money. Defense is a different objective and needs different metrics:

  beta            — sensitivity to Nifty (regression slope). <1 = cushions moves.
  down_capture    — avg sector return on DOWN-Nifty days / avg Nifty down return.
                    <1 = falls LESS than the market on bad days (the key metric).
  up_capture      — same on up days (want high: participates in bounces).
  rs              — trailing relative strength vs Nifty (cumulative, robust).

DATA ROBUSTNESS
---------------
Sector daily return is the EQUAL-WEIGHT MEDIAN of constituent stock returns, NOT
the turnover-weighted mean. Turnover-weighting uses contemporaneous volume, which
correlates with the day's move (big moves trade big) → a systematic UPWARD bias
that corrupts any magnitude metric (cum return, beta, capture). Median is immune
to that bias AND to corporate-action / penny-stock return outliers.

POINT-IN-TIME
-------------
Every metric for as_of_date uses ONLY trade_date <= as_of_date (a trailing
window). No look-ahead.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.analytics.base import get_min_turnover_filter, rank01 as _rank01
from src.data.repository import query_dataframe

__all__ = ["get_sector_defensive_metrics", "DEFENSE_WEIGHTS"]

# Defense score = cross-sectional rank blend (higher = better bear protection).
DEFENSE_WEIGHTS = {
    "down_capture": 0.40,   # falls least on down days — the core defensive edge
    "rs":           0.35,   # still outperforming the index
    "beta":         0.25,   # low market sensitivity
}

_LOOKBACK_DAYS = 120        # ~80 trading days — enough for a stable beta/capture
_MIN_OBS       = 40         # need >=40 daily obs in the window for a sector
_WINSOR        = 15.0       # clip daily % returns to +/-15% (robustness belt-and-braces)


def get_sector_defensive_metrics(
    as_of_date: date,
    lookback_days: int = _LOOKBACK_DAYS,
    min_turnover_lacs: Optional[float] = None,
) -> pd.DataFrame:
    """
    Per-sector defensive metrics over a trailing window ending at as_of_date.

    Returns DataFrame[sector, beta, down_capture, up_capture, rs, n_obs,
                      defense_score]  — defense_score in 0..100 (higher = more
    defensive / better bear protection). Empty frame if data is insufficient.
    """
    if min_turnover_lacs is None:
        min_turnover_lacs = get_min_turnover_filter()
    start = as_of_date - timedelta(days=lookback_days)

    # Robust sector daily return = median of constituent stock returns (canonical
    # sectors via v_sector_master). Point-in-time: trade_date <= as_of_date.
    sec = query_dataframe("""
        SELECT sm.sector, b.trade_date,
               MEDIAN((b.close_price - b.prev_close) / b.prev_close * 100) AS ret
        FROM daily_data b
        JOIN v_sector_master sm ON b.symbol = sm.symbol
        WHERE b.series = 'EQ'
          AND b.prev_close > 0
          AND b.turnover_lacs >= ?
          AND sm.sector NOT IN ('ETF', 'Others')
          AND b.trade_date >  ?
          AND b.trade_date <= ?
        GROUP BY sm.sector, b.trade_date
    """, [min_turnover_lacs, start, as_of_date])
    if sec.empty:
        return pd.DataFrame()

    nif = query_dataframe("""
        SELECT trade_date, pct_chg AS nret
        FROM index_data
        WHERE index_name = 'Nifty 50'
          AND trade_date >  ?
          AND trade_date <= ?
        ORDER BY trade_date
    """, [start, as_of_date])
    if nif.empty:
        return pd.DataFrame()

    sec["trade_date"] = pd.to_datetime(sec["trade_date"])
    nif["trade_date"] = pd.to_datetime(nif["trade_date"])
    df = sec.merge(nif, on="trade_date", how="inner").dropna(subset=["ret", "nret"])
    if df.empty:
        return pd.DataFrame()

    # Winsorise as a second robustness belt (median already handles most of it).
    df["ret"]  = df["ret"].clip(-_WINSOR, _WINSOR)
    df["nret"] = df["nret"].clip(-_WINSOR, _WINSOR)

    rows = []
    for sector, g in df.groupby("sector"):
        s = g["ret"].to_numpy(); n = g["nret"].to_numpy()
        if len(g) < _MIN_OBS or np.var(n) <= 0:
            continue
        beta = float(np.cov(s, n)[0, 1] / np.var(n))

        dn = n < 0; up = n > 0
        dcap = (float(s[dn].mean() / n[dn].mean())
                if dn.sum() >= 5 and n[dn].mean() != 0 else np.nan)
        ucap = (float(s[up].mean() / n[up].mean())
                if up.sum() >= 5 and n[up].mean() != 0 else np.nan)

        cum_s = float((1 + pd.Series(s) / 100).prod() - 1) * 100
        cum_n = float((1 + pd.Series(n) / 100).prod() - 1) * 100
        rs = cum_s - cum_n

        rows.append({
            "sector": sector, "beta": round(beta, 2),
            "down_capture": round(dcap, 2) if not np.isnan(dcap) else None,
            "up_capture":   round(ucap, 2) if not np.isnan(ucap) else None,
            "rel_strength": round(rs, 1), "cum_ret": round(cum_s, 1), "n_obs": int(len(g)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Raw point-in-time metrics. The COMBINED defense_score (structure + flow) is
    # computed in get_sector_rotation, where delivery/breadth flow data is joined —
    # a low-beta sector that smart money is FLEEING is a value trap, not a safe buy.
    return out.sort_values("down_capture", na_position="last").reset_index(drop=True)
