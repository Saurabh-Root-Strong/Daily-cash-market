"""
Re-measure every frozen number in src/analytics/fii_only.py and write them back.

WHY: the FII-only panel's constants were measured in Aug-2026 by scratch scripts
that were never kept, on a participant table later found to hold ~300 days of
another date's NSE file (repaired 2026-11-16). This script is the kept generator:
it measures each table with src/analytics/fii_measure.py (the SAME definitions the
live code uses), runs the family-wise sweep, prints old vs new, and rewrites the
constant blocks in fii_only.py in place (`--write`).

Usage:  python scripts/remeasure_fii_only.py [--as-of 2026-11-13] [--write]
"""
from __future__ import annotations

import argparse
import pprint
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                           # noqa: BLE001
    pass

import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from src.analytics import fii_measure as M                   # noqa: E402
from src.analytics import fii_only as F                      # noqa: E402
from src.data.repository import query_dataframe              # noqa: E402

FII_ONLY = ROOT / "src" / "analytics" / "fii_only.py"


def _plain(o):
    """numpy scalars -> python, so the literal written back is clean."""
    if isinstance(o, dict):
        return {(int(k) if isinstance(k, (np.integer,)) else k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        t = [_plain(v) for v in o]
        return tuple(t) if isinstance(o, tuple) else t
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


# ───────────────────────────────────────────────────────────── family sweep
def family_sweep(as_of: date, years: int = 2, n_perm: int = 1000, seed: int = 3) -> dict:
    """22 FII-derived signals x 3 horizons over the last `years`, one correction for
    the whole family: Bonferroni, BH-FDR and a max-|t| circular-shift permutation."""
    end = pd.Timestamp(as_of)
    start = end - pd.DateOffset(years=years)
    L = M.leg_series(as_of)
    p = query_dataframe("""SELECT * FROM fao_participant WHERE data_type='OI' AND trade_date <= ?""", [as_of])
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    fii = p[p.client_type == "FII"].drop_duplicates("trade_date").set_index("trade_date")
    cli = p[p.client_type == "Client"].drop_duplicates("trade_date").set_index("trade_date")
    S = pd.DataFrame(index=L.index)
    for leg in ("fut_long", "fut_short", "call_long", "call_short", "put_long", "put_short", "net_1d", "net_5d"):
        S[leg] = L[leg]
    S["fut_share"] = fii.fut_idx_long / (fii.fut_idx_long + fii.fut_idx_short)
    S["fut_net"] = fii.fut_idx_long - fii.fut_idx_short
    S["fut_net_chg"] = S["fut_net"].diff()
    S["opt_tilt"] = (fii.opt_idx_call_long + fii.opt_idx_put_short - fii.opt_idx_call_short - fii.opt_idx_put_long)
    S["opt_tilt_chg"] = S["opt_tilt"].diff()
    S["stk_long_share"] = fii.fut_stk_long / (fii.fut_stk_long + fii.fut_stk_short)
    S["client_share"] = -(cli.fut_idx_long / (cli.fut_idx_long + cli.fut_idx_short))   # mirror, signed bullish
    c = query_dataframe("SELECT trade_date, fii_net, dii_net FROM fii_dii_cash WHERE trade_date <= ?", [as_of])
    c["trade_date"] = pd.to_datetime(c["trade_date"])
    c = c.drop_duplicates("trade_date").set_index("trade_date").reindex(S.index)
    S["cash_1d"] = c.fii_net
    S["cash_5d"] = c.fii_net.rolling(5, min_periods=3).sum()
    S["cash_20d"] = c.fii_net.rolling(20, min_periods=10).sum()
    S["dii_cash"] = c.dii_net
    S["fii_minus_dii"] = c.fii_net - c.dii_net
    ds = query_dataframe("""SELECT trade_date, buy_value_cr, sell_value_cr, oi_value_cr FROM fii_derivatives_stats
                            WHERE category='INDEX FUTURES' AND trade_date <= ?""", [as_of])
    ds["trade_date"] = pd.to_datetime(ds["trade_date"])
    ds = ds.drop_duplicates("trade_date").set_index("trade_date").reindex(S.index)
    S["stats_net"] = ds.buy_value_cr - ds.sell_value_cr
    S["stats_oi_chg"] = ds.oi_value_cr.diff()
    comp = M._composite(as_of)["raw"]
    S["composite"] = comp
    O = M.outcomes(as_of)
    D = S.join(O[["oc1", "cc1", "w5"]], how="left")
    D = D[(D.index >= start) & (D.index <= end)]
    sigs = [s for s in S.columns]
    hz = {"oc1": "next-day open-close", "cc1": "next-day close-close", "w5": "five sessions"}

    def tstats(D_):
        res = []
        for s in sigs:
            for h in hz:
                r, t, n = M._ic(D_[s], D_[h])
                res.append((s, h, r, t, n))
        return res

    res = tstats(D)
    tab = pd.DataFrame(res, columns=["signal", "hz", "ic", "t", "n"]).dropna()
    tab["abs_t"] = tab["t"].abs()
    best = tab.sort_values("abs_t", ascending=False).iloc[0]
    from math import erf, sqrt
    pvals = np.array([2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) for t in tab["t"]])
    m = len(pvals)
    order = np.argsort(pvals)
    bh = sum(pvals[order[k]] <= 0.05 * (k + 1) / m for k in range(m))
    bonf_t = float(pd.Series([0]).pipe(lambda _: 0))  # placeholder replaced below
    from statistics import NormalDist
    bonf_t = NormalDist().inv_cdf(1 - 0.05 / (2 * m))
    rng = np.random.default_rng(seed)
    rets = D[list(hz)].to_numpy()
    X = D[sigs]
    null = []
    for _ in range(n_perm):
        k = int(rng.integers(20, len(D) - 20))
        Dp = X.copy()
        for j, h in enumerate(hz):
            Dp[h] = np.roll(rets[:, j], k)
        bt = 0.0
        for s in sigs:
            for h in hz:
                _, t, _ = M._ic(Dp[s], Dp[h])
                if t == t and abs(t) > bt:
                    bt = abs(t)
        null.append(bt)
    null = np.array(null)
    # the best signal, traded by quintile on the tradeable horizon
    sel = D[[best["signal"], "oc1"]].dropna()
    q = pd.qcut(sel[best["signal"]].rank(method="first"), 5, labels=False)
    q1, q5 = sel.loc[q == 0, "oc1"], sel.loc[q == 4, "oc1"]
    return {
        "start": str(D.index[0].date()), "end": str(D.index[-1].date()), "n": int(len(D)),
        "n_signals": len(sigs), "n_tests": int(m),
        "best_signal": str(best["signal"]), "best_hz": hz[best["hz"]],
        "best_ic": round(float(best["ic"]), 4), "best_t": round(float(best["abs_t"]), 2),
        "bonferroni_t": round(float(bonf_t), 2), "fdr_survivors": int(bh),
        "null_best_t_median": round(float(np.median(null)), 2),
        "null_best_t_p95": round(float(np.percentile(null, 95)), 2),
        "family_p": round(float((null >= best["abs_t"]).mean()), 3),
        "q1_mean": round(float(q1.mean()), 3), "q1_up": round(float(100 * (q1 > 0).mean()), 1),
        "q5_mean": round(float(q5.mean()), 3), "q5_up": round(float(100 * (q5 > 0).mean()), 1),
        "spread_bps": round(float((q5.mean() - q1.mean()) * 100), 1), "cost_bps": 50,
        "base_oc": round(float(D["oc1"].mean()), 4), "base_oc_up": round(float(100 * (D["oc1"].dropna() > 0).mean()), 1),
        "base_cc": round(float(D["cc1"].mean()), 4), "base_cc_up": round(float(100 * (D["cc1"].dropna() > 0).mean()), 1),
        "base_wk": round(float(D["w5"].mean()), 4), "base_wk_up": round(float(100 * (D["w5"].dropna() > 0).mean()), 1),
    }


# ───────────────────────────────────────────────────────────── write-back
def _replace_block(src: str, name: str, literal: str) -> str:
    m = re.search(rf"^{re.escape(name)} = ", src, flags=re.M)
    if not m:
        raise KeyError(name)
    i = m.end()
    opener = src[i]
    closer = {"{": "}", "[": "]", "(": ")"}[opener]
    depth, j = 0, i
    while True:
        ch = src[j]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                break
        j += 1
    return src[:m.start()] + f"{name} = {literal}" + src[j + 1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    as_of = date.fromisoformat(a.as_of) if a.as_of else pd.Timestamp(
        query_dataframe("SELECT MAX(trade_date) d FROM fao_participant")["d"].iloc[0]).date()
    F._components_cached.cache_clear()
    print(f"re-measuring as of {as_of}")

    zones = M.measure_zones(as_of)
    tails = M.measure_tails(as_of)
    new = {
        "FII_TRACK_RECORD": M.measure_track_record(as_of),
        "FII_ACTION_TRACK": M.measure_action_track(as_of),
        "FII_ERA_SPLIT": {k: v for k, v in M.measure_era_split(as_of).items() if k in ("era4", "era5")},
        "_ZONE_CUTS": zones["cuts"],
        "FII_ZONES": {"cuts": zones["cuts"], "d1_base": zones["d1_base"], "d5_base": zones["d5_base"],
                      "d1": zones["d1"], "d5": zones["d5"]},
        "_FII_TAIL": {h: {k: v for k, v in tails[h].items() if k == "thr" or isinstance(k, int)}
                      for h in ("d1", "d5")},
        "FII_LEG_IC": M.measure_leg_ic(as_of),
        "FII_LEG_BACKTEST_2Y": M.measure_leg_backtest(as_of),
        "FII_HEDGE_TEST": M.measure_hedge_test(as_of),
        "FII_FAMILY_SWEEP_2Y": family_sweep(as_of),
    }
    new = _plain(new)
    for k, v in new.items():
        old = getattr(F, k, None)
        print(f"\n=== {k}\n  OLD: {str(old)[:400]}\n  NEW: {str(v)[:400]}")
    print("\nTAILS (with permutation test):", tails)
    if a.write:
        src = FII_ONLY.read_text(encoding="utf-8")
        for k, v in new.items():
            lit = pprint.pformat(v, width=100, sort_dicts=False)
            if k == "FII_ZONES":
                lit = lit.replace(repr(new["_ZONE_CUTS"]), "_ZONE_CUTS", 1)
            src = _replace_block(src, k, lit)
        stamp = f"# RE-MEASURED {date.today()} by scripts/remeasure_fii_only.py on the repaired participant table, data to {as_of}."
        if "# RE-MEASURED" in src:
            src = re.sub(r"^# RE-MEASURED .*$", stamp, src, count=1, flags=re.M)
        else:
            src = src.replace("# ── What FIIs DID:", stamp + "\n# The prose comments below quote the ORIGINAL Aug-2026 numbers; the dict values are current.\n\n# ── What FIIs DID:", 1)
        FII_ONLY.write_text(src, encoding="utf-8")
        print(f"\nwrote {FII_ONLY}")


if __name__ == "__main__":
    main()
