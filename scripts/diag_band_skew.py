"""
diag_band_skew.py  —  does the next-day band need to be ASYMMETRIC (skew) or just RESIZED?

Before rewriting _compute_expected_move into a Filtered-Historical-Simulation (asymmetric
quantile) band, measure whether 1-day index returns actually carry enough skew to matter.
Equity skew is strong at monthly+ horizons but often negligible intraday/1-day, where fat
tails (kurtosis) dominate symmetrically.

Per index, on EWMA(λ=0.94)-standardized residuals z = r_t / σ_{t} (one-step-ahead, no
lookahead):
  • q16 / q84  — if |q16| ≈ |q84| the 68% band is symmetric; if |q16| > |q84| downside is fatter
  • q2.5 / q97.5 — tail asymmetry
  • sample skew
And under the CURRENT symmetric ±1σ·cal band, the realized breach SPLIT:
  • low_breach%  = P(next_close < range_low)
  • high_breach% = P(next_close > range_high)
  A symmetric band on skewed returns shows a lopsided split (e.g. 22% low vs 10% high).
  If the split is even, symmetric is fine and FHS buys nothing.

Usage:  python scripts/diag_band_skew.py
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
LAMBDA = 0.94
INDEX = {
    "NIFTY":      ("Nifty 50",                 1.00, True),
    "BANKNIFTY":  ("Nifty Bank",               1.06, False),
    "FINNIFTY":   ("Nifty Financial Services", 1.08, False),
    "MIDCPNIFTY": ("Nifty Midcap Select",      1.18, False),
}


def _ewma_sigma(rets: pd.Series) -> pd.Series:
    return np.sqrt((rets ** 2).ewm(alpha=1 - LAMBDA, adjust=False).mean())


def main() -> None:
    con = duckdb.connect(str(DB), read_only=True)
    vix = con.execute(
        "select trade_date, close_val from index_data where index_name='India VIX' order by trade_date"
    ).df()
    vix["trade_date"] = pd.to_datetime(vix["trade_date"])
    vix = vix.set_index("trade_date")["close_val"]

    print("\nStandardized-residual shape (EWMA-filtered) + symmetric-band breach split")
    print("=" * 96)
    print(f"{'index':11} {'n':>4} {'q16':>7} {'q84':>7} {'skew':>6} {'q2.5':>7} {'q97.5':>7}   "
          f"{'lowBr%':>7} {'highBr%':>7}  asym?")
    print("-" * 96)
    all_z = []
    for sym, (idx_name, cal, is_nifty) in INDEX.items():
        df = con.execute(
            "select trade_date, close_val from index_data where index_name=? and close_val is not null "
            "order by trade_date", [idx_name]
        ).df()
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        s = df.set_index("trade_date")["close_val"].astype(float)
        rets = s.pct_change()
        sig = _ewma_sigma(rets)
        z = (rets / sig.shift(1)).dropna()               # one-step-ahead standardize
        z = z[np.isfinite(z)]
        all_z.append(z.to_numpy())

        # symmetric band breach split (same blend as engine)
        v = (vix.reindex(s.index).ffill()) / 100.0 / (252.0 ** 0.5)
        w = 0.5 if is_nifty else 0.7
        blended = np.where(v.notna(), w * sig + (1 - w) * v, sig)
        em = s * blended * cal
        nxt = s.shift(-1)
        low_br = (nxt < s - em); high_br = (nxt > s + em)
        m = nxt.notna() & em.notna() & (em > 0)
        lb = 100 * low_br[m].mean(); hb = 100 * high_br[m].mean()

        q16, q84 = np.quantile(z, 0.16), np.quantile(z, 0.84)
        q025, q975 = np.quantile(z, 0.025), np.quantile(z, 0.975)
        sk = float(pd.Series(z).skew())
        asym = "YES" if abs(abs(q16) - abs(q84)) > 0.12 or abs(lb - hb) > 5 else "no"
        print(f"{sym:11} {len(z):>4} {q16:>7.2f} {q84:>7.2f} {sk:>6.2f} {q025:>7.2f} {q975:>7.2f}   "
              f"{lb:>6.1f}% {hb:>6.1f}%  {asym}")

    z = np.concatenate(all_z)
    print("-" * 96)
    print(f"{'POOLED':11} {len(z):>4} {np.quantile(z,0.16):>7.2f} {np.quantile(z,0.84):>7.2f} "
          f"{pd.Series(z).skew():>6.2f} {np.quantile(z,0.025):>7.2f} {np.quantile(z,0.975):>7.2f}")
    print("\nRead: |q16| vs |q84| and lowBr% vs highBr% — if both roughly symmetric, FHS skew")
    print("buys ~nothing at 1-day; resize (multiplier) is the real fix. If downside fatter")
    print("(|q16|>|q84|, lowBr% > highBr%), an asymmetric FHS band is justified.")


if __name__ == "__main__":
    main()
