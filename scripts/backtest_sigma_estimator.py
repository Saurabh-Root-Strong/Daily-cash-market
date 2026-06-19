"""
backtest_sigma_estimator.py  —  flat-20D σ  vs  EWMA(λ=0.94) σ  for the next-day range.

The Index-Prediction expected-move band is spot ± EM, where EM = spot · σ · cal_mult and
σ blends realized daily vol with India-VIX/√252 (NIFTY 50/50, others 70/30). The realized
leg today is a FLAT 20-day sample std — equal-weight, so a single jump inflates σ for 20
sessions then drops off a cliff, and it reacts slowly to vol regime shifts.

This replays BOTH realized estimators over the real index_data history (no engine, no
prediction_log) and measures next-day CLOSE containment of the ±1σ band, per index and
pooled. Purpose: confirm EWMA preserves the validated ~68% coverage (so _EM_CALIBRATION
stays valid) before swapping the estimator in the engine.

    flat σ_t  = std(r[t-20 : t])                       (current)
    ewma σ_t  = sqrt( EWMA_{λ=0.94}( r^2 ) )           (RiskMetrics, zero-mean)

Usage:  python scripts/backtest_sigma_estimator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = Path(r"D:\Python Projects\Daily_Cash_Market\data\market_data.duckdb")
SEED = 7
LAMBDA = 0.94                       # RiskMetrics decay → alpha = 0.06, ~32-day span
CLAIM = 0.68                        # honest ~1σ target

# fno_symbol → (index_name, cal_mult)  — must match _INDEX_MAP / _EM_CALIBRATION
INDEX = {
    "NIFTY":      ("Nifty 50",                 1.00, True),
    "BANKNIFTY":  ("Nifty Bank",               1.06, False),
    "FINNIFTY":   ("Nifty Financial Services", 1.08, False),
    "MIDCPNIFTY": ("Nifty Midcap Select",      1.18, False),
}


def _ewma_sigma(rets: pd.Series) -> pd.Series:
    """RiskMetrics zero-mean EWMA vol: σ²_t = λσ²_{t-1} + (1-λ)r²_{t-1}."""
    return np.sqrt((rets ** 2).ewm(alpha=1 - LAMBDA, adjust=False).mean())


def _boot_ci(mask: np.ndarray, rng, reps: int = 2000) -> tuple[float, float]:
    n = len(mask)
    bs = np.array([mask[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    if not DB.exists():
        print(f"DB not found: {DB}"); return
    con = duckdb.connect(str(DB), read_only=True)

    vix = con.execute(
        "select trade_date, close_val from index_data where index_name='India VIX' "
        "order by trade_date"
    ).df()
    vix["trade_date"] = pd.to_datetime(vix["trade_date"])
    vix = vix.set_index("trade_date")["close_val"]

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"\nNext-day RANGE coverage — flat-20D σ  vs  EWMA(λ={LAMBDA}) σ   (±1σ band, cal-applied)")
    print("=" * 92)
    print(f"{'index':11} {'n':>5} {'flat cover':>22} {'ewma cover':>22}   {'flat k68':>8} {'ewma k68':>8}")
    print("-" * 92)

    pooled = {"flat": [], "ewma": []}
    for sym, (idx_name, cal, is_nifty) in INDEX.items():
        df = con.execute(
            "select trade_date, close_val from index_data where index_name=? "
            "and close_val is not null order by trade_date", [idx_name]
        ).df()
        if df.empty or len(df) < 40:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        s = df["close_val"].astype(float)
        rets = s.pct_change()

        flat_sig = rets.rolling(20).std()          # flat 20D, % as fraction
        ewma_sig = _ewma_sigma(rets)
        v = vix.reindex(s.index).ffill()
        vix_sig = v / 100.0 / (252.0 ** 0.5)       # VIX is in %; → daily fraction

        w = 0.5 if is_nifty else 0.7
        out = pd.DataFrame(index=s.index)
        out["spot"] = s
        out["next"] = s.shift(-1)
        for tag, real_sig in (("flat", flat_sig), ("ewma", ewma_sig)):
            blended = np.where(vix_sig.notna(), w * real_sig + (1 - w) * vix_sig, real_sig)
            em = out["spot"] * blended * cal
            lo, hi = out["spot"] - em, out["spot"] + em
            out[f"{tag}_cont"] = ((out["next"] >= lo) & (out["next"] <= hi))
            out[f"{tag}_u"] = (out["next"] - out["spot"]).abs() / em

        d = out.dropna(subset=["next", "flat_u", "ewma_u"])
        d = d[(d["flat_u"] > 0) & (d["ewma_u"] > 0)]
        fc = d["flat_cont"].to_numpy(); ec = d["ewma_cont"].to_numpy()
        pooled["flat"].append(fc); pooled["ewma"].append(ec)
        flo, fhi = _boot_ci(fc, rng); elo, ehi = _boot_ci(ec, rng)
        fk = float(np.quantile(d["flat_u"], CLAIM)); ek = float(np.quantile(d["ewma_u"], CLAIM))
        print(f"{sym:11} {len(d):>5} "
              f"{100*fc.mean():>6.1f}% [{100*flo:>4.1f},{100*fhi:>4.1f}] "
              f"{100*ec.mean():>6.1f}% [{100*elo:>4.1f},{100*ehi:>4.1f}]   "
              f"{fk:>8.2f} {ek:>8.2f}")

    fa = np.concatenate(pooled["flat"]); ea = np.concatenate(pooled["ewma"])
    flo, fhi = _boot_ci(fa, rng); elo, ehi = _boot_ci(ea, rng)
    print("-" * 92)
    print(f"{'POOLED':11} {len(fa):>5} "
          f"{100*fa.mean():>6.1f}% [{100*flo:>4.1f},{100*fhi:>4.1f}] "
          f"{100*ea.mean():>6.1f}% [{100*elo:>4.1f},{100*ehi:>4.1f}]")
    print("\nRead: if EWMA coverage ≈ flat coverage (both near 68%, CIs overlap), the swap is")
    print("safe — _EM_CALIBRATION stays valid and EWMA buys faster vol reaction + no cliff.")


if __name__ == "__main__":
    main()
