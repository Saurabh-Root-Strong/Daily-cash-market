"""
Adversarial audit of the Forward Sector Tilt — is ANY of it real?

WHY THIS EXISTS
    scripts/study_tilt_tenure.py showed the shipped _HORIZON_EVIDENCE numbers do
    not reproduce, and that only 7-8wk (+8.8%/yr) and 11-12wk (+10.6%/yr) beat an
    equal-weight sector basket at all. Before ANY of that is acted on, the two
    survivors must be attacked, because they rest on 50 and 33 non-overlapping
    rebalances respectively -- small samples where a single phase choice, a single
    k, or a single year can manufacture the whole result.

WHAT IS TESTED (each is a way the result could be an artifact)
    1  DATA INTEGRITY  session holes, sector birth/death, corporate-action spikes
                       that survive winsorization, NaN topology, index gaps.
    2  PHASE           `ret.index[::h]` picks ONE of h possible rebalance
                       calendars. All h phases are run. A real edge is phase-
                       insensitive; an artifact lives in one or two phases.
    3  TOP-K           k = 3,4,5,6,8. The published spec says 4. If only 4 works
                       it is a fitted parameter.
    4  COST            10 / 25 / 50 / 100 bps per side. Where does it die?
    5  ERA             2018-21 / 2022-24 / 2025-26 on the SAME construction.
    6  BENCHMARK       equal-weight sector basket (not investable) vs Nifty 50
                       (investable). The tab claims excess over the former.
    7  DELIVERY LEG    drop the 0.15 dv5d weight -- is the composite doing
                       anything the raw RS momentum is not?
    8  PERSISTENCE     does the shipped gate help, hurt, or do nothing?
    9  BOOTSTRAP       block bootstrap of the rebalance series for a CI on net.

Usage:  python scripts/audit_tilt_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
WINSOR, MIN_TURN = 25.0, 100.0
W_RS2, W_RS1, W_DV5 = 0.60, 0.25, 0.15
DV_BASE, DV_FLOW = 100, 5
HORIZONS = {"1-2 wk": 10, "3-4 wk": 20, "5-6 wk": 30,
            "7-8 wk": 40, "9-10 wk": 50, "11-12 wk": 60}
PERS_LOOKBACK, PERS_MIN_OBS = 425, 30
RNG = np.random.default_rng(20260816)


def nw_t(x, lag: int) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return np.nan
    d = x - x.mean(); var = (d @ d) / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lag + 1.0)) * ((d[L:] @ d[:-L]) / n)
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


# ══ 1. DATA INTEGRITY ════════════════════════════════════════════════════════
def integrity(con):
    print("=" * 78); print("1. DATA INTEGRITY"); print("=" * 78)

    # session calendar holes: a gap of >4 calendar days that is not a known break
    d = con.sql("select distinct trade_date from daily_data where trade_date >= '2018-01-01' "
                "order by trade_date").df()
    ds = pd.to_datetime(d["trade_date"])
    gap = ds.diff().dt.days
    big = ds[gap > 5]
    print(f"  sessions 2018+: {len(ds)}  ({ds.min():%Y-%m-%d} -> {ds.max():%Y-%m-%d})")
    print(f"  gaps >5 calendar days: {len(big)}  (Diwali/long weekends expected)")
    for dt_, g in zip(big.tail(6), gap[big.tail(6).index]):
        print(f"      resumed {dt_:%Y-%m-%d} after {int(g)}d")
    # weeks with <4 sessions = possible missing days
    wk = ds.dt.to_period("W").value_counts().sort_index()
    thin_wk = wk[(wk < 4) & (wk.index.year >= 2018)]
    print(f"  weeks with <4 sessions: {len(thin_wk)} of {len(wk)} "
          f"({len(thin_wk) / len(wk) * 100:.1f}%) — holidays, but a silent DROP looks identical")

    # index_data coverage vs daily_data — a missing Nifty day makes RS wrong that day
    ix = con.sql("select distinct trade_date from index_data where index_name='Nifty 50' "
                 "and trade_date >= '2018-01-01'").df()
    ixs = set(pd.to_datetime(ix["trade_date"]).dt.date)
    miss = [x.date() for x in ds if x.date() not in ixs]
    print(f"  sessions with NO Nifty 50 row: {len(miss)}"
          + (f"  e.g. {miss[:5]}" if miss else "  — clean"))

    # corporate actions: raw |return| beyond the winsor cap
    ext = con.sql(f"""
        select count(*) n, sum(case when abs(r) >= 40 then 1 else 0 end) n40,
               sum(case when r <= -45 then 1 else 0 end) n_split
        from (select (close_price - prev_close)/nullif(prev_close,0)*100 r
              from daily_data where trade_date >= '2018-01-01'
                and series in ('EQ','SM','ST') and turnover_lacs >= {MIN_TURN})
        where abs(r) >= {WINSOR}
    """).df().iloc[0]
    print(f"  stock-days winsorized (|r|>{WINSOR:.0f}%): {int(ext['n']):,}  "
          f"of which |r|>=40%: {int(ext['n40']):,}  and r<=-45% (split/bonus-shaped): "
          f"{int(ext['n_split']):,}")
    print("     -> an UNADJUSTED 1:1 bonus prints r ~ -50%; winsor turns it into a "
          "capped -25% instead of removing it. Those days are FAKE sector losses.")

    # sector birth/death — a sector that did not exist early distorts cross-sectional rank
    sec = con.sql(f"""
        select s.sector, min(b.trade_date) first_d, max(b.trade_date) last_d,
               count(distinct b.trade_date) nd
        from daily_data b join v_sector_master s on b.symbol=s.symbol
        where b.series in ('EQ','SM','ST') and b.turnover_lacs >= {MIN_TURN}
          and s.sector not in ('ETF','Others') and b.trade_date >= '2018-01-01'
        group by 1 order by first_d desc
    """).df()
    late = sec[pd.to_datetime(sec["first_d"]) > "2018-06-01"]
    print(f"  sectors: {len(sec)}; first appearing AFTER 2018-06: {len(late)}")
    for _, r in late.iterrows():
        print(f"      {r['sector']:<28} from {r['first_d']:%Y-%m-%d}  ({r['nd']} sessions)")
    return ds


# ══ panel + signal ═══════════════════════════════════════════════════════════
def load(con):
    q = f"""
    with base as (
        select b.trade_date, b.symbol, b.turnover_lacs, b.deliv_per, s.sector,
               (b.close_price - b.prev_close)/nullif(b.prev_close,0)*100 as raw_r,
               greatest(least((b.close_price - b.prev_close)
                              / nullif(b.prev_close,0)*100, {WINSOR}), -{WINSOR}) as r,
               lag(b.turnover_lacs) over (partition by b.symbol order by b.trade_date) as w_lag
        from daily_data b join v_sector_master s on b.symbol = s.symbol
        where b.series in ('EQ','SM','ST') and s.sector is not null
          and s.sector not in ('ETF','Others')
    )
    select sector, trade_date,
           sum(turnover_lacs*deliv_per/100.0)/100.0 as daily_dv_cr,
           sum(w_lag*r)/nullif(sum(case when r is not null then w_lag end),0) as wtd_ret_pct
    from base
    -- mirrors the FIXED src/analytics/sector_forward_tilt.py::_load_sector_panel:
    -- LAGGED liquidity filter (a same-day floor admits a stock because it moved)
    -- and corporate actions DROPPED rather than winsorized into a fake -25%.
    where w_lag >= {MIN_TURN} and abs(raw_r) < 40
    group by 1,2 order by 1,2
    """
    d = con.execute(q).df(); d["trade_date"] = pd.to_datetime(d["trade_date"]); return d


class Sig:
    """Signal surface + forward returns, built once and reused by every test."""

    def __init__(self, con):
        p = load(con)
        nf = con.sql("select trade_date, close_val from index_data "
                     "where index_name='Nifty 50' order by trade_date").df()
        nf["trade_date"] = pd.to_datetime(nf["trade_date"])
        self.ret = p.pivot(index="trade_date", columns="sector",
                           values="wtd_ret_pct").sort_index().dropna(how="all")
        self.dv = p.pivot(index="trade_date", columns="sector",
                          values="daily_dv_cr").sort_index().reindex(self.ret.index)
        self.lg = np.log1p(self.ret / 100.0)
        nser = nf.set_index("trade_date")["close_val"].sort_index()
        self.nret = nser.pct_change().reindex(self.ret.index) * 100.0
        self.ngr = np.log1p(self.nret.fillna(0.0) / 100.0)
        self.dv5d = (self.dv.rolling(DV_FLOW).mean()
                     / self.dv.shift(1).rolling(DV_BASE).mean())

    def trail(self, n): return np.expm1(self.lg.rolling(n).sum()) * 100.0
    def ntrail(self, n): return np.expm1(self.ngr.rolling(n).sum()) * 100.0

    def score(self, h, use_dv=True):
        L, S = h, max(2, h // 2)
        rk = lambda d: d.rank(axis=1, pct=True)
        rsL = rk(self.trail(L).sub(self.ntrail(L), axis=0))
        rsS = rk(self.trail(S).sub(self.ntrail(S), axis=0))
        if not use_dv:
            return (W_RS2 * rsL + W_RS1 * rsS) / (W_RS2 + W_RS1)
        return W_RS2 * rsL + W_RS1 * rsS + W_DV5 * rk(self.dv5d)

    def fwd_raw(self, h):
        return np.expm1(self.lg.shift(-1).iloc[::-1]
                        .rolling(h, min_periods=h).sum().iloc[::-1]) * 100.0

    def fwd(self, h, bench="basket"):
        f = self.fwd_raw(h)
        if bench == "basket":
            return f.sub(f.mean(axis=1), axis=0)
        nf = np.expm1(self.ngr.shift(-1).iloc[::-1]
                      .rolling(h, min_periods=h).sum().iloc[::-1]) * 100.0
        return f.sub(nf, axis=0)

    def persistence(self, h):
        e = self.fwd(h).sub(self.fwd(h).median(axis=1), axis=0)
        return e.shift(h).rolling(PERS_LOOKBACK, min_periods=PERS_MIN_OBS).mean()


def backtest(sig, h, k=4, phase=0, cost_side=0.25, bench="basket",
             use_dv=True, gate=False, dates=None):
    """Non-overlapping top-k, EXCESS over `bench`, net of cost. Returns per-reb array."""
    sc = sig.score(h, use_dv=use_dv)
    if gate:
        sc = sc.where(~(sig.persistence(h) < 0))
    f = sig.fwd(h, bench=bench)
    idx = sig.ret.index[phase::h]
    if dates is not None:
        idx = idx[dates(idx)]
    out, prev = [], set()
    for dt_ in idx:
        s = sc.loc[dt_].dropna()
        if len(s) < k:
            continue
        pick = list(s.nlargest(k).index)
        y = f.loc[dt_, pick].dropna()
        if len(y) < max(2, k // 2):
            continue
        turn = len(set(pick) - prev) / k
        out.append(float(y.mean()) - turn * 2 * cost_side)
        prev = set(pick)
    return np.array(out)


def ann(net, h): return float(np.mean(net) * 252.0 / h) if len(net) else np.nan


# ══ 2-9 ══════════════════════════════════════════════════════════════════════
def test_phase(sig):
    print("\n" + "=" * 78)
    print("2. PHASE SENSITIVITY — all h rebalance calendars, not just offset 0")
    print("=" * 78)
    print(f"  {'horizon':<10}{'shipped(ph0)':>13}{'mean':>8}{'median':>8}{'min':>8}"
          f"{'max':>8}{'% phases >0':>13}{'n_reb':>7}")
    res = {}
    for lbl, h in HORIZONS.items():
        a = [ann(backtest(sig, h, phase=p), h) for p in range(h)]
        a = np.array([x for x in a if np.isfinite(x)])
        n = len(backtest(sig, h, phase=0))
        res[lbl] = a
        print(f"  {lbl:<10}{a[0]:>13.2f}{a.mean():>8.2f}{np.median(a):>8.2f}"
              f"{a.min():>8.2f}{a.max():>8.2f}{(a > 0).mean() * 100:>12.0f}%{n:>7}")
    print("\n  READ: offset 0 is an arbitrary calendar. If '% phases >0' is near 50% the")
    print("  horizon has no edge — the shipped number was the luck of one start date.")
    return res


def test_k(sig):
    print("\n" + "=" * 78)
    print("3. TOP-K SENSITIVITY (phase-averaged, so k is not confounded with phase)")
    print("=" * 78)
    ks = [3, 4, 5, 6, 8]
    print(f"  {'horizon':<10}" + "".join(f"{'k=' + str(k):>10}" for k in ks))
    for lbl, h in HORIZONS.items():
        row = []
        for k in ks:
            a = [ann(backtest(sig, h, k=k, phase=p), h) for p in range(h)]
            row.append(np.nanmean(a))
        print(f"  {lbl:<10}" + "".join(f"{v:>10.2f}" for v in row))
    print("\n  READ: a real cross-sectional edge decays smoothly with k. A spike at one k")
    print("  is a fitted parameter.")


def test_cost(sig):
    print("\n" + "=" * 78)
    print("4. COST SENSITIVITY (phase-averaged, %/yr excess vs basket)")
    print("=" * 78)
    cs = [0.0, 0.10, 0.25, 0.50, 1.00]
    print(f"  {'horizon':<10}" + "".join(f"{str(int(c * 100)) + 'bps':>10}" for c in cs)
          + "   breakeven bps/side")
    for lbl, h in HORIZONS.items():
        row = [np.nanmean([ann(backtest(sig, h, phase=p, cost_side=c), h)
                           for p in range(h)]) for c in cs]
        g, c25 = row[0], row[2]
        be = (g / ((g - c25) / 0.25) if (g - c25) > 0 else np.nan)
        print(f"  {lbl:<10}" + "".join(f"{v:>10.2f}" for v in row)
              + (f"{be:>20.0f}" if np.isfinite(be) else f"{'n/a':>20}"))


def test_era(sig):
    print("\n" + "=" * 78)
    print("5. ERA STABILITY (phase-averaged %/yr excess vs basket)")
    print("=" * 78)
    eras = {"2018-21": (2018, 2021), "2022-24": (2022, 2024), "2025-26": (2025, 2026)}
    print(f"  {'horizon':<10}" + "".join(f"{e:>12}" for e in eras)
          + f"{'all':>10}{'sign-stable':>13}")
    for lbl, h in HORIZONS.items():
        row = []
        for _, (y0, y1) in eras.items():
            sel = lambda idx, y0=y0, y1=y1: (idx.year >= y0) & (idx.year <= y1)
            a = [ann(backtest(sig, h, phase=p, dates=sel), h) for p in range(h)]
            row.append(np.nanmean(a))
        allv = np.nanmean([ann(backtest(sig, h, phase=p), h) for p in range(h)])
        stable = "YES" if all(np.sign(v) == np.sign(row[0]) for v in row) else "no"
        print(f"  {lbl:<10}" + "".join(f"{v:>12.2f}" for v in row)
              + f"{allv:>10.2f}{stable:>13}")


def test_bench_dv_gate(sig):
    print("\n" + "=" * 78)
    print("6-8. BENCHMARK · DELIVERY LEG · PERSISTENCE GATE (phase-averaged %/yr)")
    print("=" * 78)
    print(f"  {'horizon':<10}{'vs basket':>11}{'vs NIFTY':>11}{'no dv5d':>10}"
          f"{'+gate':>9}{'gate delta':>12}")
    for lbl, h in HORIZONS.items():
        f = lambda **kw: np.nanmean([ann(backtest(sig, h, phase=p, **kw), h)
                                     for p in range(h)])
        b, n_, nodv, g = f(), f(bench="nifty"), f(use_dv=False), f(gate=True)
        print(f"  {lbl:<10}{b:>11.2f}{n_:>11.2f}{nodv:>10.2f}{g:>9.2f}{g - b:>12.2f}")
    print("\n  READ: 'vs NIFTY' is the only investable comparison — the equal-weight")
    print("  24-sector basket is not a product you can buy. 'no dv5d' asks whether the")
    print("  delivery leg earns its 15% weight. 'gate delta' is the shipped persistence")
    print("  gate's actual contribution.")


def test_bootstrap(sig):
    print("\n" + "=" * 78)
    print("9. BLOCK BOOTSTRAP — 95% CI on net %/yr (all phases pooled, 2000 draws)")
    print("=" * 78)
    print(f"  {'horizon':<10}{'point':>9}{'CI low':>9}{'CI high':>9}{'P(<0)':>8}"
          f"{'n_reb':>7}   verdict")
    for lbl, h in HORIZONS.items():
        pooled = np.concatenate([backtest(sig, h, phase=p) for p in range(h)])
        n = len(backtest(sig, h, phase=0))
        # resample whole PHASES to respect within-phase serial structure
        phases = [backtest(sig, h, phase=p) for p in range(h)]
        draws = []
        for _ in range(2000):
            pick = RNG.integers(0, len(phases), len(phases))
            draws.append(ann(np.concatenate([phases[i] for i in pick]), h))
        d = np.array(draws)
        lo, hi = np.percentile(d, [2.5, 97.5])
        pneg = float((d < 0).mean())
        v = ("REAL" if lo > 0 else "NOISE" if hi > 0 > lo else "NEGATIVE")
        print(f"  {lbl:<10}{ann(pooled, h):>9.2f}{lo:>9.2f}{hi:>9.2f}{pneg:>8.2f}"
              f"{n:>7}   {v}")


def main():
    con = duckdb.connect(DB, read_only=True)
    integrity(con)
    sig = Sig(con)
    con.close()
    print(f"\n  panel: {sig.ret.shape[1]} sectors x {len(sig.ret)} sessions "
          f"({sig.ret.index.min():%Y-%m-%d} -> {sig.ret.index.max():%Y-%m-%d})")
    nan_by_yr = sig.ret.isna().mean(axis=1).groupby(sig.ret.index.year).mean() * 100
    print("  % sector-cells missing by year: "
          + " ".join(f"{y}:{v:.1f}" for y, v in nan_by_yr.items()))
    test_phase(sig)
    test_k(sig)
    test_cost(sig)
    test_era(sig)
    test_bench_dv_gate(sig)
    test_bootstrap(sig)


if __name__ == "__main__":
    main()
