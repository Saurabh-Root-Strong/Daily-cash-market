"""
When do FII POSITIONS peak and bottom, relative to Nifty's swing tops and bottoms?

Positions (contracts, FII, participant OI file, repaired data):
  futures  LONG  = index-futures longs           SHORT = index-futures shorts
  options  BULL  = calls bought + puts written   BEAR  = calls written + puts bought
           (options sides use a 5-session median: weekly expiry otherwise saw-tooths them)
Position peaks/troughs = zigzag on log(position) with a 12% reversal (scale-free, so a
lot-size change is a slow drift, not a turn).

Questions:
  1. Around each Nifty 3% TOP:   where is the nearest LONG peak / SHORT trough?
     Around each Nifty 3% BOTTOM: where is the nearest SHORT peak / LONG trough?
     -> offset in sessions (negative = position turned BEFORE price).
  2. As a rule a trader could use: "the position has rolled over 10% off its 20-session
     extreme" (first day). How often was the matching Nifty turn ALREADY in (previous 1-15
     sessions) or still AHEAD (next 1-10)? Versus chance (circular shift). Plus Nifty's
     next 10 sessions from the next open.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:                                          # noqa: BLE001
    pass

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from src.analytics import big_players as bp, positioning_cycle as pc   # noqa: E402

AS = date(2026, 11, 13)
RNG = np.random.default_rng(13)


def zig(x: np.ndarray, thr: float):
    """Zigzag on a positive series (relative reversal)."""
    out, d, hi, lo = [], 0, 0, 0
    for i in range(1, len(x)):
        if not np.isfinite(x[i]):
            continue
        if x[i] > x[hi]:
            hi = i
        if x[i] < x[lo]:
            lo = i
        last = out[-1][0] if out else -1
        if d >= 0 and x[i] <= x[hi] * (1 - thr) and hi > last and hi > 0:
            out.append((hi, "PEAK")); d, lo = -1, i
        elif d <= 0 and x[i] >= x[lo] * (1 + thr) and lo > last and lo > 0:
            out.append((lo, "TROUGH")); d, hi = 1, i
    return out


def main() -> None:
    wide, px, exp = bp._load(AS)
    f = wide["FII"].dropna(subset=["fut_idx_long"])
    idx = f.index
    c = px["close_val"].reindex(idx).ffill().values
    o = px["open_val"].reindex(idx).values
    fwd10 = (pd.Series(c).shift(-10).values / pd.Series(o).shift(-1).values - 1) * 100
    med = lambda s: s.rolling(5, min_periods=1).median()
    P = {
        "FUT LONG": f["fut_idx_long"].astype(float),
        "FUT SHORT": f["fut_idx_short"].astype(float),
        "OPT BULL": med((f["opt_idx_call_long"] + f["opt_idx_put_short"]).astype(float)),
        "OPT BEAR": med((f["opt_idx_call_short"] + f["opt_idx_put_long"]).astype(float)),
    }
    nz = pc._zigzag(c, 0.03)
    ntop = np.array([i for i, k in nz if k == "TOP"])
    nbot = np.array([i for i, k in nz if k == "BOTTOM"])
    base10 = np.nanmean(fwd10)

    print("1) nearest position turn to each Nifty turn (sessions; negative = position turned FIRST)")
    pairs = [("FUT LONG", "PEAK", ntop, "TOP"), ("FUT SHORT", "TROUGH", ntop, "TOP"),
             ("FUT SHORT", "PEAK", nbot, "BOTTOM"), ("FUT LONG", "TROUGH", nbot, "BOTTOM"),
             ("OPT BULL", "PEAK", ntop, "TOP"), ("OPT BEAR", "TROUGH", ntop, "TOP"),
             ("OPT BEAR", "PEAK", nbot, "BOTTOM"), ("OPT BULL", "TROUGH", nbot, "BOTTOM")]
    for name, kind, nt, nk in pairs:
        z = zig(P[name].values, 0.12)
        pi = np.array([i for i, k in z if k == kind])
        if not len(pi):
            continue
        off = np.array([pi[np.argmin(np.abs(pi - t))] - t for t in nt])
        w = np.abs(off) <= 10
        print(f"   Nifty {nk:<6} vs {name} {kind:<6}: within ±10 sessions {100*w.mean():3.0f}%  "
              f"median offset {np.median(off[w]) if w.any() else np.nan:+.0f}  "
              f"turned first {100*np.mean(off[w] < 0) if w.any() else np.nan:3.0f}% / same day "
              f"{100*np.mean(off[w] == 0) if w.any() else np.nan:3.0f}% / after "
              f"{100*np.mean(off[w] > 0) if w.any() else np.nan:3.0f}%   ({len(pi)} position turns)")

    print("\n2) rule: position rolls over 10% off its 20-session extreme (first day, 10-session cooldown)")
    print(f"   {'rule':<34}{'n':>4}{'turn already in':>17}{'chance':>8}{'p':>7}{'turn ahead':>12}{'chance':>8}"
          f"{'p':>7}{'fwd10':>8}{'base':>7}")
    rules = [("FUT LONG falls 10% off 20d high", P["FUT LONG"], "down", ntop),
             ("FUT SHORT falls 10% off 20d high", P["FUT SHORT"], "down", nbot),
             ("FUT SHORT rises 10% off 20d low", P["FUT SHORT"], "up", ntop),
             ("FUT LONG rises 10% off 20d low", P["FUT LONG"], "up", nbot),
             ("OPT BULL falls 10% off 20d high", P["OPT BULL"], "down", ntop),
             ("OPT BEAR falls 10% off 20d high", P["OPT BEAR"], "down", nbot),
             ("OPT BEAR rises 10% off 20d low", P["OPT BEAR"], "up", ntop),
             ("OPT BULL rises 10% off 20d low", P["OPT BULL"], "up", nbot)]
    vi = np.arange(len(idx))
    for lab, s, way, turns in rules:
        if way == "down":
            m = (s <= s.rolling(20, min_periods=10).max() * 0.90).values
        else:
            m = (s >= s.rolling(20, min_periods=10).min() * 1.10).values
        ep = np.flatnonzero(pc._episodes(np.nan_to_num(m).astype(bool), 10))

        def share(sig, lo, hi):
            d = turns[None, :] - sig[:, None]
            return 100 * ((d >= lo) & (d <= hi)).any(axis=1).mean() if len(sig) else np.nan
        a_obs, b_obs = share(ep, -15, 0), share(ep, 1, 10)
        a_ch, b_ch = share(vi[30:-15], -15, 0), share(vi[30:-15], 1, 10)
        na, nb_ = [], []
        for _ in range(1500):
            sh = (ep + RNG.integers(40, len(idx) - 40)) % len(idx)
            na.append(share(sh, -15, 0)); nb_.append(share(sh, 1, 10))
        pa, pb_ = (np.array(na) >= a_obs).mean(), (np.array(nb_) >= b_obs).mean()
        x = fwd10[ep]; x = x[np.isfinite(x)]
        print(f"   {lab:<34}{len(ep):>4}{a_obs:>16.0f}%{a_ch:>7.0f}%{pa:>7.3f}{b_obs:>11.0f}%{b_ch:>7.0f}%"
              f"{pb_:>7.3f}{x.mean():>+8.2f}{base10:>+7.2f}")
    print("\n   'turn already in' = matching Nifty turn in the previous 15 sessions (incl. today);"
          " 'ahead' = next 1-10. 8 rules x 2 = 16 tests; Bonferroni p < 0.003")


if __name__ == "__main__":
    main()
