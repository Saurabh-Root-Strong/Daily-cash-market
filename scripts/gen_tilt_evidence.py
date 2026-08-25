"""
Emit _HORIZON_EVIDENCE for src/analytics/sector_forward_tilt.py from measured data.

WHY THIS EXISTS
    The shipped _HORIZON_EVIDENCE quoted +19.6%/yr (t 2.03) at 1-2wk and cited
    scripts/study_tilt_horizons.py -- a script that computes only IC and long/short
    spread and never produces a %/yr figure at all. The repo's own net-%/yr backtest
    (backtest_tilt_vs_clock.py) prints +0.4%/yr for the same spec, and an independent
    rebuild agrees to the decimal. The table had no reproducible provenance.

    This script is now that provenance. It regenerates every field from the CORRECTED
    panel (lagged liquidity filter, corporate actions dropped) and prints a paste-ready
    dict, so the number on screen can always be traced to a command.

METHOD (every choice here is the conservative one)
    net_yr    long-only top-4, EXCESS over the equal-weight sector basket,
              non-overlapping rebalance, net of 25bps/side -- AVERAGED OVER ALL h
              REBALANCE CALENDARS. The old table used offset 0 only; phase choice
              alone moves a horizon by 10-16pp/yr, in both directions.
    net_t     median per-phase Newey-West t (lag 1; the series is non-overlapping).
              The median, not the max, because there are h correlated phases.
    ls_ic_t   Newey-West t of the daily long/short spread at lag = h.
    pct_pos   share of rebalance calendars with a positive result. Descriptive --
              the phases overlap heavily, so this is NOT h independent tests.
    era       phase-averaged net_yr within each era.
    validated requires ALL of: >=90% of phases positive, the same sign in all three
              eras, and median per-phase |t| >= 2. Anything less is displayed as a
              lean, never as a validated edge.

Usage:  python scripts/gen_tilt_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_tilt_robustness import Sig, backtest, ann, nw_t, HORIZONS, DB

ERAS = (("2018-21", 2018, 2021), ("2022-24", 2022, 2024), ("2025-26", 2025, 2026))


def main():
    con = duckdb.connect(DB, read_only=True)
    sig = Sig(con)
    con.close()
    print(f"# panel: {sig.ret.shape[1]} sectors x {len(sig.ret)} sessions "
          f"({sig.ret.index.min():%Y-%m-%d} -> {sig.ret.index.max():%Y-%m-%d})")
    print("# corrected panel: lagged liquidity filter + corporate actions dropped\n")

    out = {}
    for lbl, h in HORIZONS.items():
        per_phase = [backtest(sig, h, phase=p) for p in range(h)]
        anns = np.array([ann(x, h) for x in per_phase], float)
        ts = np.array([nw_t(x, 1) for x in per_phase], float)

        # long/short IC t at lag = h (daily spread, overlap-corrected)
        sc, f = sig.score(h), sig.fwd(h)
        ls = []
        for dt_ in sc.index:
            s, y = sc.loc[dt_], f.loc[dt_]
            m = s.notna() & y.notna()
            if m.sum() < 8:
                continue
            k = max(3, int(m.sum() // 5))
            ls.append(y[m].loc[s[m].nlargest(k).index].mean()
                      - y[m].loc[s[m].nsmallest(k).index].mean())
        ls_t = nw_t(np.array(ls, float), h)

        era = {}
        for name, y0, y1 in ERAS:
            sel = lambda idx, y0=y0, y1=y1: (idx.year >= y0) & (idx.year <= y1)
            era[name] = round(float(np.nanmean(
                [ann(backtest(sig, h, phase=p, dates=sel), h) for p in range(h)])), 1)

        pct_pos = float(np.mean(anns > 0))
        med_t = float(np.nanmedian(ts))
        sign_stable = len({np.sign(v) for v in era.values()}) == 1
        validated = bool(pct_pos >= 0.90 and sign_stable and abs(med_t) >= 2.0)

        out[h] = dict(label=lbl, reb_yr=round(252 / h, 1),
                      net_yr=round(float(np.nanmean(anns)), 1),
                      net_t=round(med_t, 2), ls_ic_t=round(float(ls_t), 2),
                      pct_pos=round(pct_pos, 2), era=era, validated=validated)

    print("_HORIZON_EVIDENCE: dict[int, dict] = {")
    for h, v in out.items():
        print(f"    {h}: dict(label={v['label']!r:<12} reb_yr={v['reb_yr']:>5}, "
              f"net_yr={v['net_yr']:>5}, net_t={v['net_t']:>5}, ls_ic_t={v['ls_ic_t']:>5},"
              .replace("' ", "', "))
        print(f"        pct_pos={v['pct_pos']:.2f}, era={v['era']}, "
              f"validated={v['validated']}),")
    print("}")

    # ── bucket-level evidence: what a sector ON the list actually did ─────────
    # This is the granularity the data supports. The per-sector `est_rel_bps` did
    # NOT: it was a straight line through rank (corr 1.0000 with rank), calibrated
    # on a tercile spread that does not reproduce, and the true rank ladder is flat
    # then steps at the top quintile (Q1 +0.27 / Q2 +0.26 / Q3 +0.29 / Q4 +0.28 /
    # Q5 +0.67) — a linear map is the wrong SHAPE, not just the wrong scale.
    #
    # Measured as: mean forward excess over the equal-weight sector basket across
    # every sector-day in the bucket, GROSS of cost. Newey-West t at lag=h on the
    # per-date cross-sectional mean, so overlapping windows and within-date
    # correlation are both handled.
    #
    # OVERWEIGHT is exact (WATCH needs rank <= 0.35 and so can never mask it).
    # UNDERWEIGHT is the rank band only — WATCH can mask it and is not modelled.
    print("\n\n_BUCKET_EVIDENCE: dict[int, dict] = {")
    for lbl, h in HORIZONS.items():
        sc, f = sig.score(h), sig.fwd(h)
        rank = sc.rank(axis=1, pct=True)
        pers = sig.persistence(h)
        ow = (rank >= 0.75) & ~(pers < 0).fillna(False)
        uw = rank <= 0.25
        row = {}
        for nm, m in (("ow", ow), ("uw", uw)):
            v = f.values[(m & f.notna()).values]
            row[nm] = (round(float(v.mean()), 3),
                       round(float(nw_t(f.where(m).mean(axis=1).dropna().values, h)), 2),
                       int(len(v)))
        gross = float(np.nanmean([ann(backtest(sig, h, phase=p, cost_side=0.0), h)
                                  for p in range(h)]))
        net = out[h]["net_yr"]
        print(f"    {h}: dict(ow_pp={row['ow'][0]:>6}, ow_t={row['ow'][1]:>5}, "
              f"ow_n={row['ow'][2]:>5},")
        print(f"        uw_pp={row['uw'][0]:>6}, uw_t={row['uw'][1]:>5}, "
              f"uw_n={row['uw'][2]:>5},")
        print(f"        gross_yr={round(gross, 1)}, cost_drag={round(gross - net, 1)}),"
              f"   # {lbl}")
    print("}")

    print("\n# diagnostic")
    print(f"# {'horizon':<10}{'net%/yr':>9}{'med t':>8}{'ls_ic_t':>9}{'%phases+':>10}"
          f"{'sign-stable':>13}{'validated':>11}")
    for h, v in out.items():
        ss = len({np.sign(x) for x in v['era'].values()}) == 1
        print(f"# {v['label']:<10}{v['net_yr']:>9}{v['net_t']:>8}{v['ls_ic_t']:>9}"
              f"{v['pct_pos'] * 100:>9.0f}%{str(ss):>13}{str(v['validated']):>11}")


if __name__ == "__main__":
    main()
