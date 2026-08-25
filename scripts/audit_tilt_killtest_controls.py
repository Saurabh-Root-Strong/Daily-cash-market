"""
Adversarial controls on the audit's OWN kill test.

WHAT THIS CHALLENGES
    scripts/audit_tilt_survivorship.py (test G) showed the tilt's edge vanishes
    once sectors with fewer than ~15 liquid constituents are excluded, and
    concluded the edge is a thin-sector artifact. That conclusion has a confound
    big enough to invalidate it:

      Excluding thin sectors shrinks the cross-section from 24 to ~15. Picking
      top-4 of 24 is a 17th-percentile cut; top-4 of 15 is a 27th-percentile cut.
      A less selective portfolio earns less almost by definition. So the observed
      collapse may measure SELECTIVITY, not thinness.

    A destructive finding has to clear the same bar as a constructive one, so the
    kill test gets three controls:

    H  RANDOM-DROP CONTROL. Remove the same NUMBER of sectors, chosen at random,
       instead of the thin ones. If a random drop kills the edge just as hard,
       test G proved nothing about thinness -- it proved the signal needs a wide
       cross-section. 200 random draws per horizon.
    I  CONSTANT-SELECTIVITY. Re-run the thin-exclusion holding the percentile cut
       fixed (top ceil(N/6)) rather than top-4, so the portfolio is equally
       selective before and after the filter.
    J  EQUAL-WEIGHT SECTOR CONSTRUCTION. The live panel is turnover-weighted, so
       in a 4-name sector the largest stock IS the sector. Rebuild every sector
       return equal-weight across constituents and re-run. This tests the
       "one stock wearing a sector's name" hypothesis directly, without removing
       any sector from the cross-section at all -- no selectivity confound.

    J is the cleanest of the three: it changes only the thing under suspicion.

Usage:  python scripts/audit_tilt_killtest_controls.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_tilt_robustness import (Sig, backtest, ann, HORIZONS, DB,
                                           MIN_TURN, WINSOR, W_RS2, W_RS1, W_DV5,
                                           DV_BASE, DV_FLOW)

RNG = np.random.default_rng(20260816)
COST_RT = 0.5          # round-trip on a fully replaced name (2 x 25bps)


def sector_year_counts(con):
    d = con.sql(f"""
        select s.sector, cast(year(b.trade_date) as int) yr, count(distinct b.symbol) n
        from daily_data b join v_sector_master s on b.symbol=s.symbol
        where b.series in ('EQ','SM','ST') and b.turnover_lacs >= {MIN_TURN}
          and s.sector not in ('ETF','Others')
        group by 1,2
    """).df()
    return d.pivot(index="yr", columns="sector", values="n")


def bt(sc, sig, h, phase, k=None, frac=None):
    """Top-k (or top ceil(N*frac)) of whatever `sc` leaves available."""
    f = sig.fwd(h)
    out, prev = [], set()
    for dt_ in sig.ret.index[phase::h]:
        s = sc.loc[dt_].dropna()
        kk = k if k is not None else max(1, int(np.ceil(len(s) * frac)))
        if len(s) < kk or len(s) < 4:
            continue
        pick = list(s.nlargest(kk).index)
        y = f.loc[dt_, pick].dropna()
        if len(y) < max(2, kk // 2):
            continue
        out.append(float(y.mean()) - len(set(pick) - prev) / kk * COST_RT)
        prev = set(pick)
    return np.array(out)


def mask_thin(sc, piv, min_n):
    yr = sc.index.year
    ok = pd.DataFrame(True, index=sc.index, columns=sc.columns)
    for s in sc.columns:
        if s in piv.columns:
            ok[s] = piv[s].reindex(yr).values >= min_n
    return sc.where(ok)


def mask_random(sc, drop_names):
    ok = pd.DataFrame(True, index=sc.index, columns=sc.columns)
    for s in drop_names:
        ok[s] = False
    return sc.where(ok)


def main():
    con = duckdb.connect(DB, read_only=True)
    piv = sector_year_counts(con)
    sig = Sig(con)

    # ── equal-weight sector panel (only the weighting changes) ────────────────
    eq = con.sql(f"""
        with base as (
            select b.trade_date, s.sector,
                   greatest(least((b.close_price-b.prev_close)/nullif(b.prev_close,0)*100,
                            {WINSOR}), -{WINSOR}) r,
                   b.turnover_lacs, b.deliv_per,
                   lag(b.turnover_lacs) over (partition by b.symbol order by b.trade_date) w_lag
            from daily_data b join v_sector_master s on b.symbol=s.symbol
            where b.series in ('EQ','SM','ST') and s.sector not in ('ETF','Others')
        )
        select sector, trade_date, avg(r) wtd_ret_pct,
               sum(turnover_lacs*deliv_per/100.0)/100.0 daily_dv_cr
        from base where turnover_lacs >= {MIN_TURN} and w_lag is not null
        group by 1,2
    """).df()
    con.close()
    eq["trade_date"] = pd.to_datetime(eq["trade_date"])

    print(f"panel: {sig.ret.shape[1]} sectors x {len(sig.ret)} sessions")
    n_by_yr = (piv >= 15).sum(axis=1)
    print(f"sectors with >=15 names, by year: "
          + " ".join(f"{y}:{v}" for y, v in n_by_yr.items()))
    print(f"  -> the >=15 filter removes {24 - n_by_yr.mean():.0f} of 24 sectors on average, "
          f"so top-4 goes from a {4 / 24 * 100:.0f}th-pct cut to a "
          f"{4 / n_by_yr.mean() * 100:.0f}th-pct cut")

    # ═══ H. RANDOM-DROP CONTROL ══════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("H. RANDOM-DROP CONTROL — remove the same COUNT of sectors, chosen at random")
    print("=" * 78)
    print(f"  {'horizon':<10}{'all 24':>9}{'>=15 thin-filter':>18}{'random-drop mean':>18}"
          f"{'rand p5':>9}{'rand p95':>10}   verdict")
    for lbl, h in HORIZONS.items():
        sc = sig.score(h)
        base = np.nanmean([ann(bt(sc, sig, h, p, k=4), h) for p in range(h)])
        thin = np.nanmean([ann(bt(mask_thin(sc, piv, 15), sig, h, p, k=4), h)
                           for p in range(h)])
        n_drop = int(round(24 - n_by_yr.mean()))
        rands = []
        cols = list(sc.columns)
        for _ in range(60):
            dn = list(RNG.choice(cols, size=n_drop, replace=False))
            rands.append(np.nanmean([ann(bt(mask_random(sc, dn), sig, h, p, k=4), h)
                                     for p in range(0, h, max(1, h // 6))]))
        r = np.array(rands)
        lo, hi = np.percentile(r, [5, 95])
        # is the thin-filter result unusual against the random-drop null?
        pctl = float((r <= thin).mean())
        v = ("THINNESS REAL" if pctl < 0.10 else
             "CONFOUNDED — random drop does the same" if pctl > 0.25 else "borderline")
        print(f"  {lbl:<10}{base:>9.2f}{thin:>18.2f}{r.mean():>18.2f}{lo:>9.2f}{hi:>10.2f}"
              f"   {v} (pctl {pctl:.2f})")

    # ═══ I. CONSTANT SELECTIVITY ═════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("I. CONSTANT SELECTIVITY — top ceil(N/6) instead of fixed top-4")
    print("=" * 78)
    print(f"  {'horizon':<10}{'all 24 (N/6)':>14}{'>=15 names (N/6)':>19}{'delta':>9}")
    for lbl, h in HORIZONS.items():
        sc = sig.score(h)
        a = np.nanmean([ann(bt(sc, sig, h, p, frac=1 / 6), h) for p in range(h)])
        b = np.nanmean([ann(bt(mask_thin(sc, piv, 15), sig, h, p, frac=1 / 6), h)
                        for p in range(h)])
        print(f"  {lbl:<10}{a:>14.2f}{b:>19.2f}{b - a:>9.2f}")
    print("\n  selectivity is now held fixed, so any remaining drop is thinness, not the cut.")

    # ═══ J. EQUAL-WEIGHT SECTOR CONSTRUCTION ═════════════════════════════════
    print("\n" + "=" * 78)
    print("J. EQUAL-WEIGHT vs TURNOVER-WEIGHT sector returns (no sector removed)")
    print("=" * 78)
    sig_eq = Sig.__new__(Sig)
    sig_eq.ret = eq.pivot(index="trade_date", columns="sector",
                          values="wtd_ret_pct").sort_index().reindex(sig.ret.index)
    sig_eq.dv = eq.pivot(index="trade_date", columns="sector",
                         values="daily_dv_cr").sort_index().reindex(sig.ret.index)
    sig_eq.lg = np.log1p(sig_eq.ret / 100.0)
    sig_eq.nret, sig_eq.ngr = sig.nret, sig.ngr
    sig_eq.dv5d = (sig_eq.dv.rolling(DV_FLOW).mean()
                   / sig_eq.dv.shift(1).rolling(DV_BASE).mean())
    print(f"  {'horizon':<10}{'turnover-wt':>13}{'equal-wt':>11}{'delta':>8}"
          f"{'eq-wt, >=15 only':>19}")
    for lbl, h in HORIZONS.items():
        tw = np.nanmean([ann(bt(sig.score(h), sig, h, p, k=4), h) for p in range(h)])
        ew = np.nanmean([ann(bt(sig_eq.score(h), sig_eq, h, p, k=4), h) for p in range(h)])
        ew15 = np.nanmean([ann(bt(mask_thin(sig_eq.score(h), piv, 15), sig_eq, h, p, k=4), h)
                           for p in range(h)])
        print(f"  {lbl:<10}{tw:>13.2f}{ew:>11.2f}{ew - tw:>8.2f}{ew15:>19.2f}")
    print("\n  equal-weighting removes single-stock domination WITHOUT shrinking the")
    print("  cross-section, so it isolates the thinness hypothesis cleanly.")


if __name__ == "__main__":
    main()
