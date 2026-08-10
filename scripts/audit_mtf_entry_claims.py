"""
Audit the Multi-timeframe trend panel's on-screen claims.

CLAIMS UNDER TEST (src/analytics/sector_forward_tilt.py::get_mtf_trend + the UI caption)
  1. "3/4-up is the BEST entry window (strongest measured forward, ~t+3.4 swing)"
  2. "4/4-up is EXTENDED, measurably weaker than 3/4, often near a local top"
  3. "all-4-down -> ~76% up over ~3mo" (offered as a bottom signal)
  4. "swing/short trend has a mild forward edge (~+1-2pp vs drift)"

THREE THINGS THE ORIGINAL AUDIT HAD TO GET RIGHT, AND WHICH ARE RE-TESTED HERE
  A. BASE RATE. In a market that drifts up, "76% up over 3 months" is only a
     signal if the UNCONDITIONAL rate is materially below 76%. Every number here
     is therefore reported as an EXCESS over the unconditional forward return,
     not as a raw hit rate.
  B. OVERLAP. The state is read daily but forward windows are 10/30/65 days, so
     consecutive observations share almost all their return. A naive t on daily
     rows is inflated by roughly sqrt(h). Reported t is Newey-West at lag=h, and
     a non-overlapping sample is shown beside it as a sanity check.
  C. MULTIPLICITY. "BEST" is the max over 5 states (0..4 up). The max of 5
     correlated statistics needs a higher bar than t=2, so a permutation test on
     the max-|t| across states is run.

ALSO SPLIT OUT: n_up==3 pools "3 up + 1 FLAT" with "3 up + 1 DOWN". The panel
scores them identically; they are reported separately here.

Usage:  python scripts/audit_mtf_entry_claims.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
BANDS = {"swing": 10, "short": 30, "long": 60, "vlong": 120}
FWD = {"swing": 10, "short": 30, "long": 65, "vlong": 130}
RNG = np.random.default_rng(20260809)


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    d = x - x.mean()
    var = (d @ d) / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1 - L / (lag + 1.0)) * ((d[L:] @ d[:-L]) / n)
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def build():
    con = duckdb.connect(DB, read_only=True)
    df = con.sql("""select trade_date, close_val from index_data
                    where index_name='Nifty 50' order by trade_date""").df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    c = df["close_val"].astype(float).reset_index(drop=True)

    st = {}
    for name, span in BANDS.items():
        e = c.ewm(span=span, adjust=False).mean()
        sl = e - e.shift(max(5, span // 4))
        s = np.where((c > e) & (sl > 0), 1, np.where((c < e) & (sl < 0), -1, 0))
        st[name] = pd.Series(s, index=c.index)
    S = pd.DataFrame(st)
    S["n_up"] = (S[list(BANDS)] == 1).sum(axis=1)
    S["n_dn"] = (S[list(BANDS)] == -1).sum(axis=1)
    S["trade_date"] = df["trade_date"]
    S["close"] = c
    for h in sorted(set(FWD.values())):
        S[f"f{h}"] = c.shift(-h) / c - 1.0
    return S.iloc[130:].reset_index(drop=True)   # drop EMA warm-up


def main():
    S = build()
    print(f"sessions: {len(S):,}  {S.trade_date.min():%Y-%m-%d} -> {S.trade_date.max():%Y-%m-%d}")

    print("\n=== 0. UNCONDITIONAL BASE RATES (the yardstick every claim must beat) ===")
    base = {}
    for h in sorted(set(FWD.values())):
        f = S[f"f{h}"].dropna()
        base[h] = (f.mean(), (f > 0).mean())
        print(f"  fwd {h:>3}d: mean {f.mean()*100:+6.2f}%   P(up) {100*(f>0).mean():5.1f}%   n={len(f):,}")

    print("\n=== 1./2. FORWARD RETURN BY ALIGNMENT STATE (n_up), EXCESS over base ===")
    print("    swing horizon = 10d, and the 65d horizon used for the 'bottom' claim")
    for h in (10, 65):
        print(f"\n  --- forward {h}d (base {base[h][0]*100:+.2f}%, P(up) {100*base[h][1]:.1f}%) ---")
        rows = []
        for k in range(5):
            m = S["n_up"] == k
            f = S.loc[m, f"f{h}"].dropna()
            if len(f) < 30:
                rows.append({"n_up": k, "n_days": int(m.sum()), "n": len(f)}); continue
            rows.append({"n_up": k, "n_days": int(m.sum()), "n": len(f),
                         "mean_%": round(f.mean() * 100, 2),
                         "excess_pp": round((f.mean() - base[h][0]) * 100, 2),
                         "P(up)_%": round(100 * (f > 0).mean(), 1),
                         "P_excess_pp": round(100 * ((f > 0).mean() - base[h][1]), 1),
                         "t_naive": round(f.mean() / f.std() * np.sqrt(len(f)), 2),
                         "t_NW": round(nw_t(f.values - base[h][0], h), 2)})
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 3. SPLIT n_up==3: is '3 up + 1 FLAT' the same as '3 up + 1 DOWN'? ===")
    for h in (10, 65):
        a = S[(S.n_up == 3) & (S.n_dn == 0)][f"f{h}"].dropna()   # 3 up + 1 flat
        b = S[(S.n_up == 3) & (S.n_dn == 1)][f"f{h}"].dropna()   # 3 up + 1 down
        print(f"  fwd {h:>2}d: 3up+FLAT n={len(a):<5} mean {a.mean()*100:+.2f}% | "
              f"3up+DOWN n={len(b):<5} mean {b.mean()*100:+.2f}%"
              + ("   <- panel scores these IDENTICALLY" if len(a) and len(b) else ""))

    print("\n=== 4. MULTIPLICITY: 'BEST' is the max over 5 states — does it survive? ===")
    for h in (10, 65):
        obs = []
        for k in range(5):
            f = S.loc[S.n_up == k, f"f{h}"].dropna()
            obs.append(abs(nw_t(f.values - base[h][0], h)) if len(f) >= 30 else np.nan)
        obs_max = np.nanmax(obs)
        lab = S["n_up"].values
        vals = S[f"f{h}"].values
        ok = ~np.isnan(vals)
        null = np.empty(2000)
        for i in range(2000):
            # circular block shift preserves the autocorrelation of BOTH series
            sh = np.roll(lab[ok], RNG.integers(1, ok.sum() - 1))
            ts = []
            for k in range(5):
                f = vals[ok][sh == k]
                ts.append(abs(nw_t(f - base[h][0], h)) if len(f) >= 30 else np.nan)
            null[i] = np.nanmax(ts)
        p = (np.sum(null >= obs_max) + 1) / 2001
        print(f"  fwd {h:>2}d: best state |t_NW| = {obs_max:.2f} | "
              f"null 95th pct {np.percentile(null,95):.2f} | p = {p:.4f}  "
              f"-> {'survives' if p < 0.05 else 'DOES NOT survive'}")

    print("\n=== 5. NON-OVERLAPPING sanity check (sample every h days) ===")
    for h in (10, 65):
        print(f"  --- fwd {h}d, every {h}th session ---")
        sub = S.iloc[::h]
        for k in (3, 4, 0):
            f = sub.loc[sub.n_up == k, f"f{h}"].dropna()
            if len(f) < 10:
                print(f"    n_up={k}: n={len(f)} too few"); continue
            print(f"    n_up={k}: n={len(f):<4} mean {f.mean()*100:+6.2f}% "
                  f"excess {(f.mean()-base[h][0])*100:+6.2f}pp  t {f.mean()/f.std()*np.sqrt(len(f)):+5.2f}")

    print("\n=== 6. STATE FREQUENCY (how often does the user see each banner?) ===")
    fr = S["n_up"].value_counts().sort_index()
    for k, v in fr.items():
        print(f"  n_up={k}: {v:>5} sessions ({100*v/len(S):5.1f}%)")


if __name__ == "__main__":
    main()
