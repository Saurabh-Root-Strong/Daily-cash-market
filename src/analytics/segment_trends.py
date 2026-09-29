"""Business-segment (5th classification level) trend board.

Descriptive only. Segments come from config/stock_segments.csv; whether a segment
grouping explains returns at all is recorded per sub-sector in
config/segment_validation.csv. Segment momentum has no forward power (past
60/120/250d relative return vs the next 20-60d: rank IC -0.04..+0.07, |t|<=1.2),
so these numbers describe the present -- they are not a forecast.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["SEGMENT_GROUPS", "get_segment_trends", "get_segment_verdicts"]

_ROOT = Path(__file__).resolve().parents[2]
_VALIDATION = _ROOT / "config" / "segment_validation.csv"

# NSE basic industry the segment was assigned under -> display group
SEGMENT_GROUPS = {
    "Pharmaceuticals": "Pharma & Healthcare",
    "Biotechnology": "Pharma & Healthcare",
    "Healthcare Research, Analytics & Technology": "Pharma & Healthcare",
    "Auto Components & Equipments": "Auto Components",
    "Civil Construction": "Civil Construction",
    "Specialty Chemicals": "Specialty Chemicals",
}
_VALIDATION_KEY = {             # group -> basic_industry label used in the validation file
    "Pharma & Healthcare": "Pharmaceuticals",
    "Auto Components": "Auto Components",
    "Civil Construction": "Civil Construction",
    "Specialty Chemicals": "Specialty Chemicals",
}
MIN_TURNOVER_CR = 2.0           # same liquidity floor as the validation test
_SESSIONS = {"1M": 21, "3M": 63, "6M": 126}


def get_segment_verdicts() -> dict[str, str]:
    """group -> strictest recorded verdict for its segment label (controlled test wins)."""
    if not _VALIDATION.exists():
        return {}
    v = pd.read_csv(_VALIDATION, comment="#", dtype=str)
    v = v[v["label"].str.startswith("segment")]
    out = {}
    for group, key in _VALIDATION_KEY.items():
        rows = v[v["basic_industry"].str.startswith(key)]
        if rows.empty:
            continue
        ctl = rows[rows["label"].str.contains("controlled")]
        out[group] = (ctl if not ctl.empty else rows).iloc[-1]["verdict"]
    return out


def get_segment_trends(as_of: date) -> pd.DataFrame:
    """Per segment: liquid stock count, median 1M/3M/6M return, % above 50/200-day
    average, median distance from the 52-week high. As of `as_of` (point-in-time).

    A stock is left out when a split / bonus / demerger / rights ex-date falls inside
    the last year (unadjusted closes would fake the return) or it trades under
    Rs 2 Cr/day over the last 90 sessions.
    """
    seg = query_dataframe("SELECT symbol, segment, scope_basic_industry FROM stock_segments", [])
    if seg.empty:
        return pd.DataFrame()
    seg["group"] = seg["scope_basic_industry"].map(SEGMENT_GROUPS)
    seg = seg.dropna(subset=["group"])
    syms = seg["symbol"].tolist()
    start = as_of - timedelta(days=400)
    px = query_dataframe(f"""
        SELECT trade_date, symbol, close_price, turnover_lacs FROM daily_data
        WHERE series = 'EQ' AND trade_date BETWEEN ? AND ?
          AND symbol IN ({','.join('?' * len(syms))})
    """, [start, as_of, *syms])
    if px.empty:
        return pd.DataFrame()
    ca = query_dataframe("""
        SELECT DISTINCT symbol FROM corp_actions
        WHERE kind IN ('SPLIT','BONUS','DEMERGER','RIGHTS') AND ex_date > ? AND ex_date <= ?
    """, [as_of - timedelta(days=370), as_of])
    close = px.pivot(index="trade_date", columns="symbol", values="close_price").sort_index()
    turn = px.pivot(index="trade_date", columns="symbol", values="turnover_lacs").sort_index()
    liquid = turn.iloc[-90:].mean() / 100.0 >= MIN_TURNOVER_CR
    keep = [s for s in close.columns
            if liquid.get(s, False) and s not in set(ca["symbol"]) and close[s].notna().sum() >= 200]
    if not keep:
        return pd.DataFrame()
    c = close[keep].ffill()
    last = c.iloc[-1]
    per = pd.DataFrame(index=keep)
    for label, n in _SESSIONS.items():
        per[label] = (last / c.iloc[-1 - n] - 1) * 100 if len(c) > n else float("nan")
    per["above_50"] = last > c.iloc[-50:].mean()
    per["above_200"] = last > c.iloc[-200:].mean()
    per["off_high"] = (last / c.iloc[-250:].max() - 1) * 100
    per = per.join(seg.set_index("symbol")[["group", "segment"]])
    agg = per.groupby(["group", "segment"]).agg(
        stocks=("1M", "size"), r1m=("1M", "median"), r3m=("3M", "median"),
        r6m=("6M", "median"), above_50=("above_50", "mean"),
        above_200=("above_200", "mean"), off_high=("off_high", "median"),
    ).reset_index()
    agg[["above_50", "above_200"]] *= 100
    return agg.sort_values(["group", "r3m"], ascending=[True, False]).reset_index(drop=True)
