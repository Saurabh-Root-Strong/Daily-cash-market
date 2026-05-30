"""
READ-ONLY per-signal diagnostic for the Index Prediction engine.

Recomputes predictions across all historical FNO dates, captures every signal's
signed score, and joins with the ACTUAL next-day index return. Then scores each
signal on UP-vs-DOWN tradeability:

  fires      = times the signal appeared with a non-zero score
  dir_edge   = mean next-day return when signal is bullish (score>0)
               minus mean when bearish (score<0)   → core tradeability metric
  hit        = P(sign(score) == sign(next_day_return))  among directional days
  ic         = Spearman corr(signed score, next-day return)
  bull_ret   = avg next-day return on the signal's bullish days
  bear_ret   = avg next-day return on the signal's bearish days

A GOOD bullish-or-bearish signal has dir_edge > 0 and hit > 0.5.
A signal with dir_edge < 0 is anti-predictive (fade it or remove it).

Read-only: store_prediction is monkeypatched to a no-op so nothing is written
to prediction_log. Safe to delete this script.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Neutralize DB writes BEFORE importing the engine ─────────────────────────
import src.analytics.memory_engine as _mem
_mem.store_prediction = lambda *a, **k: None          # no-op: keep read-only

from src.analytics.index_prediction import get_index_predictions, _INDEX_MAP  # noqa: E402
from src.data.repository import query_dataframe                                # noqa: E402

_IDX_NAME = {s: idx for s, (_, idx) in _INDEX_MAP.items()}
DIR_THRESH = 0.15   # matches memory_engine _DIRECTION_THRESH


def _next_day_returns(index_name: str) -> dict:
    """trade_date → next available trading day's pct_chg, for one index."""
    df = query_dataframe(
        "SELECT trade_date, pct_chg FROM index_data "
        "WHERE index_name = ? AND pct_chg IS NOT NULL ORDER BY trade_date",
        [index_name],
    )
    if df.empty:
        return {}
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    out = {}
    rows = df.to_dict("records")
    for i in range(len(rows) - 1):
        out[rows[i]["trade_date"]] = float(rows[i + 1]["pct_chg"])
    return out


def main() -> None:
    dates_df = query_dataframe(
        "SELECT DISTINCT trade_date FROM fno_bhavcopy "
        "WHERE instrument IN ('FUTIDX','OPTIDX') ORDER BY trade_date"
    )
    if dates_df.empty:
        print("No FNO dates."); return
    dates = [d.date() if hasattr(d, "date") else d for d in dates_df["trade_date"]]
    # Drop the most recent date (no next-day outcome yet)
    dates = dates[:-1] if len(dates) > 1 else dates
    print(f"Recomputing predictions for {len(dates)} FNO dates "
          f"({dates[0]} → {dates[-1]}) × {len(_INDEX_MAP)} indices...")

    nxt = {idx: _next_day_returns(idx) for idx in _IDX_NAME.values()}

    # signal name → list of (signed_score, next_ret)
    sig_rows: dict[str, list] = defaultdict(list)
    # also track composite for sanity
    comp_rows: list = []
    n_preds = 0

    for td in dates:
        try:
            preds = get_index_predictions(td)
        except Exception as e:
            print(f"  {td}: ERROR {e}")
            continue
        for p in preds:
            if not p.data_available:
                continue
            idx_name = _IDX_NAME.get(p.fno_symbol)
            ret = nxt.get(idx_name, {}).get(td)
            if ret is None:
                continue
            n_preds += 1
            comp_rows.append((float(p.composite_score), ret))
            for s in p.signals:
                if abs(s.score) < 1e-9:
                    continue
                sig_rows[s.name].append((float(s.score), ret))

    print(f"Captured {n_preds} (prediction, next-day return) pairs.\n")

    # ── Per-signal table ──────────────────────────────────────────────────────
    results = []
    for name, pairs in sig_rows.items():
        arr = np.array(pairs)                      # (n, 2): score, ret
        scores, rets = arr[:, 0], arr[:, 1]
        n = len(arr)
        if n < 5:
            continue
        bull = rets[scores > 0]
        bear = rets[scores < 0]
        bull_ret = bull.mean() if len(bull) else np.nan
        bear_ret = bear.mean() if len(bear) else np.nan
        # directional edge: bullish-day return minus bearish-day return
        if len(bull) and len(bear):
            dir_edge = bull_ret - bear_ret
        elif len(bull):
            dir_edge = bull_ret           # only ever bullish
        else:
            dir_edge = -bear_ret          # only ever bearish → edge if bear_ret<0
        # hit rate among days the move was directional
        mask = np.abs(rets) > DIR_THRESH
        if mask.sum() > 0:
            hit = np.mean(np.sign(scores[mask]) == np.sign(rets[mask]))
        else:
            hit = np.nan
        # Spearman IC
        if np.std(scores) > 1e-9 and np.std(rets) > 1e-9:
            ic = pd.Series(scores).corr(pd.Series(rets), method="spearman")
        else:
            ic = np.nan
        results.append({
            "signal": name[:34], "fires": n,
            "dir_edge": dir_edge, "hit": hit, "ic": ic,
            "bull_ret": bull_ret, "bear_ret": bear_ret,
            "n_bull": len(bull), "n_bear": len(bear),
        })

    res = pd.DataFrame(results).sort_values("dir_edge", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)

    print("=" * 110)
    print("PER-SIGNAL TRADEABILITY  (sorted by directional edge; dir_edge>0 good, <0 anti-predictive)")
    print("=" * 110)
    fmt = res.copy()
    for c in ["dir_edge", "hit", "ic", "bull_ret", "bear_ret"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "—")
    print(fmt.to_string(index=False))

    print("\n" + "=" * 110)
    print("ANTI-PREDICTIVE SIGNALS (dir_edge < 0 — these HURT the composite):")
    print("=" * 110)
    bad = res[res["dir_edge"] < 0]
    if bad.empty:
        print("  none")
    else:
        for _, r in bad.iterrows():
            print(f"  {r['signal']:<36} dir_edge={r['dir_edge']:+.3f}  "
                  f"fires={r['fires']:>3}  bull_ret={r['bull_ret']:+.3f} bear_ret={r['bear_ret']:+.3f}")

    # ── Composite sanity ──────────────────────────────────────────────────────
    comp = np.array(comp_rows)
    if len(comp):
        cs, cr = comp[:, 0], comp[:, 1]
        ic = pd.Series(cs).corr(pd.Series(cr), method="spearman")
        print("\n" + "=" * 110)
        print(f"COMPOSITE SCORE vs next-day return:  Spearman IC = {ic:+.3f}  (n={len(comp)})")
        print(f"  avg next-day return when composite>0 (bullish): {cr[cs>0].mean():+.3f}%  (n={(cs>0).sum()})")
        print(f"  avg next-day return when composite<0 (bearish): {cr[cs<0].mean():+.3f}%  (n={(cs<0).sum()})")
        print("  → if bullish-row return is NOT > bearish-row return, the composite is mis-signed")
    print("\nDIAGNOSTIC_DONE")


if __name__ == "__main__":
    main()
