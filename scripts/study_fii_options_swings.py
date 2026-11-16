"""
Does FII INDEX-OPTIONS positioning behave like its futures positioning at swing turns?

Same tests the futures swing map rests on (2026-11-16), run on the four option legs:
  bull_share   = (calls bought + puts written) / all four legs     -> + = bullish book
  call_buy     = calls bought / (calls bought + calls written)
  put_write    = puts written / (puts written + puts bought)
  pcr_long     = puts bought / calls bought   (sign-flipped so + = bullish)
For each: (1) lead or lag vs Nifty, (2) whether it sides with the swing vs a
shuffled-time null, (3) heavy-zone -> turn in the NEXT 1-10 sessions with a
circular-shift permutation, (4) how much weekly expiries move it, (5) its range,
so leg thresholds can be scaled per metric.
Repaired data only (post 2026-11-16), regime-aware percentiles.
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
from src.data.repository import query_dataframe             # noqa: E402

AS = date(2026, 11, 13)
RNG = np.random.default_rng(5)


def main() -> None:
    wide, px, exp = bp._load(AS)
    f = wide["FII"].dropna(subset=["opt_idx_call_long"])
    idx = f.index
    c = px["close_val"].reindex(idx).ffill().values
    cl_, cs, pl, ps = (f[k].astype(float) for k in ("opt_idx_call_long", "opt_idx_call_short",
                                                    "opt_idx_put_long", "opt_idx_put_short"))
    tot = cl_ + cs + pl + ps
    M = {
        "bull_share": (cl_ + ps) / tot * 100,
        "call_buy": cl_ / (cl_ + cs) * 100,
        "put_write": ps / (ps + pl) * 100,
        "neg_pcr_long": -(pl / cl_),
        "fut_share (ref)": f["fut_idx_long"] / (f["fut_idx_long"] + f["fut_idx_short"]) * 100,
    }
    oe = query_dataframe("""SELECT DISTINCT expiry_date FROM fno_bhavcopy WHERE instrument='OPTIDX'
                            AND symbol='NIFTY'""")
    opt_exp = set(pd.to_datetime(oe["expiry_date"]))
    past5 = pd.Series(c, index=idx).pct_change(5)
    nxt5 = (px["close_val"].shift(-5) / px["open_val"].shift(-1) - 1).reindex(idx)

    zz3 = pc._zigzag(c, 0.03)
    zz5 = pc._zigzag(c, 0.05)
    T = {thr: {"BOTTOM": np.array([i for i, k in zz if k == "BOTTOM"]),
               "TOP": np.array([i for i, k in zz if k == "TOP"])} for thr, zz in ((3, zz3), (5, zz5))}

    print(f"{'metric':<18}{'range p5-p95':>16}{'|d1| med':>9}{'exp-day |d1|':>13}"
          f"{'vsPAST5':>9}{'vsNEXT5':>9}{'swing agree':>13}{'null':>6}")
    rows = {}
    for name, s in M.items():
        s = s.astype(float)
        d1 = s.diff().abs()
        on = d1[d1.index.isin(opt_exp)].median()
        off = d1[~d1.index.isin(opt_exp)].median()
        j1 = pd.concat([s.diff(5), past5], axis=1).dropna().rank()
        j2 = pd.concat([s.diff(5), nxt5], axis=1).dropna().rank()
        lp, ln = j1.corr().iloc[0, 1], j2.corr().iloc[0, 1]
        # swing agreement: endpoint medians inside the leg, flat = 0.25 x median |leg change|
        sv = s.values
        legs = [g for g in bp._swing_legs(idx, c, sv, sv, None, 0.03) if not g["partial"] and not g["open"]]

        def leg_d(arr, g):
            a, b = idx.get_loc(pd.Timestamp(g["start"])), idx.get_loc(pd.Timestamp(g["end"]))
            k = max(1, min(3, (b - a + 1) // 2))
            return np.nanmedian(arr[b - k + 1:b + 1]) - np.nanmedian(arr[a:a + k])
        dl = np.array([leg_d(sv, g) for g in legs])
        flat = 0.25 * np.nanmedian(np.abs(dl))
        agree = np.mean([(np.sign(x) if abs(x) >= flat else 0) == g["dir"] for x, g in zip(dl, legs)])
        null = []
        for _ in range(300):
            sh = np.roll(sv, RNG.integers(50, len(sv) - 50))
            dd = np.array([leg_d(sh, g) for g in legs])
            null.append(np.mean([(np.sign(x) if abs(x) >= flat else 0) == g["dir"] for x, g in zip(dd, legs)]))
        print(f"{name:<18}{np.nanpercentile(s,5):>7.2f}-{np.nanpercentile(s,95):<7.2f}{off:>9.3f}{on:>13.3f}"
              f"{lp:>+9.3f}{ln:>+9.3f}{100*agree:>12.0f}%{100*np.median(null):>5.0f}%")
        rows[name] = (s, flat)

    print("\nheavy zone (regime-aware trailing-year percentile) -> turn in the NEXT 1-10 sessions")
    print(f"{'metric':<18}{'zone':<12}{'n':>4}{'3% hit':>8}{'chance':>7}{'p':>7}{'5% hit':>8}{'chance':>7}{'p':>7}")
    valid_all = None
    for name, (s, _) in rows.items():
        pr = pc._pct_rank(s).values
        valid = np.isfinite(pr)
        vi = np.flatnonzero(valid)
        for zl, m, kind in (("heavy LONG", pr >= 90, "TOP"), ("heavy SHORT", pr <= 10, "BOTTOM")):
            ep = np.flatnonzero(pc._episodes(m & valid, 10))
            out = []
            for thr in (3, 5):
                tt = T[thr][kind]
                obs = pc._near(ep, tt)
                ch = pc._near(vi, tt)
                null = []
                for _ in range(1500):
                    k = RNG.integers(60, len(vi) - 60)
                    null.append(pc._near(vi[(np.searchsorted(vi, ep) + k) % len(vi)], tt))
                out += [obs, ch, (np.array(null) >= obs).mean()]
            print(f"{name:<18}{zl:<12}{len(ep):>4}{out[0]:>7.0f}%{out[1]:>6.0f}%{out[2]:>7.3f}"
                  f"{out[3]:>7.0f}%{out[4]:>6.0f}%{out[5]:>7.3f}")
    print("\ntests: 5 metrics x 2 zones x 2 swing sizes = 20; Bonferroni p < 0.0025")


if __name__ == "__main__":
    main()
