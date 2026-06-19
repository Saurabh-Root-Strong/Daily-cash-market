"""
Walk-forward validation of Signal 6b — S/R Confluence (price-structure levels).

Tier 1  PRICE-ONLY core over the FULL index OHLC history (~350 days × 4 indices).
        No OI walls / no FII — measures whether "level tested & held/rejected/broke
        in the last 4 sessions" alone predicts the next-day return.
Tier 2  FULL CONFLUENCE over the F&O bhavcopy window (~48 days): same core plus
        OI-wall alignment (liquidity-gated) and FII 5D flow agreement — measures
        whether the confluence inputs IMPROVE the read on the same days.

Point-in-time: each day uses only OHLC ≤ t (pivots need 2 future bars to confirm,
which the detector enforces), the chain as of t, and FII stats dated ≤ t.

Output: fire%, hit% (|next ret| ≥ 0.10), Spearman IC — per index and pooled.
Same metrics as backtest_signals.py so results are comparable with the other
24 signals. Decision rule (same bar VWAP/RSI faced): no positive walk-forward
IC ⇒ the signal must be wired through _context_only().
"""
from __future__ import annotations

import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.repository import query_dataframe
from src.analytics.index_prediction import (
    _INDEX_MAP, _SYMBOL_TO_FII_FUT, _OPT_OI_MIN, _OPT_CONC_MAX,
    _sr_level_read, _compute_key_levels, _load_fno_for_date,
)

_LOOKBACK_ROWS = 82   # matches _load_index_history(lookback=60) ≈ 60×2 calendar days


def _load_full_history(index_name: str) -> pd.DataFrame:
    df = query_dataframe("""
        SELECT trade_date, open_val, high_val, low_val, close_val, prev_close,
               pct_chg, volume, turnover_cr
        FROM index_data
        WHERE index_name = ?
        ORDER BY trade_date
    """, [index_name])
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def _fii_5d(symbol: str, t: date) -> float | None:
    cat = _SYMBOL_TO_FII_FUT[symbol]
    df = query_dataframe("""
        WITH last5 AS (
            SELECT DISTINCT trade_date FROM fii_derivatives_stats
            WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 5
        )
        SELECT SUM(buy_value_cr - sell_value_cr) AS net
        FROM fii_derivatives_stats
        WHERE category = ? AND trade_date IN (SELECT trade_date FROM last5)
    """, [t, cat])
    if df.empty or df["net"].isna().all():
        return None
    return float(df["net"].iloc[0])


def _metrics(rows: list[dict], label: str) -> dict | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fired = df[df["fired"]]
    dir_rows = fired[(fired["direction"] != 0) & (fired["next_ret"].abs() >= 0.10)]
    hit = np.nan
    if len(dir_rows):
        ok = ((dir_rows["direction"] > 0) & (dir_rows["next_ret"] > 0)) | \
             ((dir_rows["direction"] < 0) & (dir_rows["next_ret"] < 0))
        hit = ok.mean() * 100
    ic = np.nan
    if len(fired) >= 5 and fired["score"].nunique() >= 3:
        ic = fired["score"].rank().corr(fired["next_ret"].rank())
    return {
        "slice": label, "days": len(df), "fires": len(fired),
        "fire%": round(100 * len(fired) / len(df), 0),
        "n_dir": len(dir_rows),
        "hit%": round(hit, 0) if hit == hit else np.nan,
        "IC": round(ic, 3) if ic == ic else np.nan,
    }


def tier1() -> None:
    print("=" * 78)
    print("TIER 1 — price-only S/R core, full OHLC history (no OI / no FII)")
    print("=" * 78, flush=True)
    all_rows: list[dict] = []
    per_index: list[dict] = []
    for sym, (_, idx_name) in _INDEX_MAP.items():
        hist = _load_full_history(idx_name)
        rows: list[dict] = []
        for i in range(60, len(hist) - 1):
            window = hist.iloc[max(0, i - _LOOKBACK_ROWS + 1): i + 1]
            spot = hist["close_val"].iloc[i]
            nxt = hist["pct_chg"].iloc[i + 1]
            if pd.isna(spot) or pd.isna(nxt):
                continue
            sr = _sr_level_read(window, float(spot))
            fired = bool(sr["bull_notes"] or sr["bear_notes"] or sr["range_read"])
            rows.append({
                "symbol": sym, "fired": fired,
                "direction": sr["direction"], "score": sr["score"],
                "next_ret": float(nxt),
            })
        all_rows.extend(rows)
        m = _metrics(rows, sym)
        if m:
            per_index.append(m)
        print(f"  .. {sym}: {len(rows)} days", flush=True)
    pooled = _metrics(all_rows, "POOLED")
    out = pd.DataFrame(per_index + ([pooled] if pooled else []))
    print(out.to_string(index=False))


def tier2() -> None:
    print("\n" + "=" * 78)
    print("TIER 2 — full confluence (OI walls + volume + FII 5D), F&O window")
    print("=" * 78, flush=True)
    fno_dates = [d.date() if hasattr(d, "date") else d for d in query_dataframe(
        "SELECT DISTINCT trade_date FROM fno_bhavcopy ORDER BY trade_date"
    )["trade_date"].tolist()]

    hists = {sym: _load_full_history(idx) for sym, (_, idx) in _INDEX_MAP.items()}
    rows_conf: list[dict] = []
    rows_price: list[dict] = []   # price-only on the SAME days, for A/B comparison

    for t in fno_dates:
        for sym, (_, idx_name) in _INDEX_MAP.items():
            hist = hists[sym]
            pos = hist.index[hist["trade_date"] == t]
            if len(pos) == 0:
                continue
            i = int(pos[0])
            if i < 60 or i + 1 >= len(hist):
                continue
            window = hist.iloc[max(0, i - _LOOKBACK_ROWS + 1): i + 1]
            spot = float(hist["close_val"].iloc[i])
            nxt = hist["pct_chg"].iloc[i + 1]
            if pd.isna(nxt):
                continue

            # Key levels from the near-expiry chain as of t (+ liquidity gate)
            levels = None
            fno = _load_fno_for_date(sym, t)
            if not fno.empty:
                opt = fno[(fno["instrument"] == "OPTIDX") & (fno["expiry_date"] > t)]
                if not opt.empty:
                    near = opt["expiry_date"].min()
                    chain = opt[opt["expiry_date"] == near]
                    total_oi = int(chain["open_interest"].sum())
                    top = int(chain.groupby(["option_type", "strike_price"])
                              ["open_interest"].sum().max() or 0)
                    conc = (top / total_oi) if total_oi > 0 else 1.0
                    if total_oi >= _OPT_OI_MIN and conc <= _OPT_CONC_MAX:
                        levels = _compute_key_levels(chain, spot)

            sr_c = _sr_level_read(window, spot, levels=levels, fii_5d=_fii_5d(sym, t))
            sr_p = _sr_level_read(window, spot)
            for sr, sink in ((sr_c, rows_conf), (sr_p, rows_price)):
                sink.append({
                    "symbol": sym,
                    "fired": bool(sr["bull_notes"] or sr["bear_notes"] or sr["range_read"]),
                    "direction": sr["direction"], "score": sr["score"],
                    "next_ret": float(nxt),
                })

    res = []
    m = _metrics(rows_price, "price-only (same days)")
    if m:
        res.append(m)
    m = _metrics(rows_conf, "with confluence")
    if m:
        res.append(m)
    for sym in _INDEX_MAP:
        m = _metrics([r for r in rows_conf if r["symbol"] == sym], f"confluence {sym}")
        if m:
            res.append(m)
    print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__":
    tier1()
    tier2()
