"""
Head-to-head: Forward Sector Tilt vs Rotation Clock, across trade horizons.

THE TWO SIGNALS, AS SHIPPED
  TILT  = 0.60*rank(rs_long) + 0.25*rank(rs_short) + 0.15*rank(dv5d), take top-k.
          Pure cross-sectional MOMENTUM with a small delivery-flow confirm.
  CLOCK = quadrants on two axes:
            y = cross-sectional z of the DELIVERY-% linear slope over the window
            x = sector's compound return over the window MINUS the sector median
          Leading   = slope_z > +0.25 and price > median   (flow up, price up)
          Improving = slope_z > +0.25 and price < median   (flow up, price LAGGING)
          Weakening = slope_z < -0.25 and price > median
          Lagging   = slope_z < -0.25 and price < median

WHY THIS IS AN INTERESTING COMPARISON, NOT A FORMALITY
  "Leading" is momentum + flow, so it should correlate with the Tilt. "Improving"
  is the OPPOSITE of momentum — buying what is lagging on rising delivery. If the
  Clock has independent value it must come from Improving, and most plausibly at
  LONGER horizons, where accumulation has time to show up in price. Testing both
  at 1-2 .. 11-12 weeks answers "which one, for which holding period".

METHOD (identical treatment for both, or the comparison is worthless)
  - Same sector panel: EQ/SM/ST, sector NOT IN ('ETF','Others'), per-stock daily
    return winsorized +/-25%, weighted by LAGGED turnover, w_lag NULL dropped
    (so an IPO listing day, whose prev_close is the issue price, contributes 0).
  - Both signals use a lookback == the forward horizon, so neither is advantaged.
  - Forward return = compound sector return over the NEXT h sessions, EXCESS over
    the equal-weight sector basket. Strictly future; no overlap with the signal.
  - NON-OVERLAPPING rebalance (every h sessions) for the money numbers, so the
    cost model and the t-stat are honest. Overlapping daily IC also reported with
    a Newey-West t at lag=h.
  - Cost 25bps/side applied to measured basket turnover.
  - Benchmarks: equal-weight sector basket (the 0-line) and a random-k control.

Usage:  python scripts/backtest_tilt_vs_clock.py
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
COST_SIDE = 0.25                      # % per side
HORIZONS = {"1-2 wk": 10, "3-4 wk": 20, "5-6 wk": 30,
            "7-8 wk": 40, "9-10 wk": 50, "11-12 wk": 60}
RNG = np.random.default_rng(20260809)


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return np.nan
    d = x - x.mean()
    var = (d @ d) / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1 - L / (lag + 1.0)) * ((d[L:] @ d[:-L]) / n)
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def load():
    con = duckdb.connect(DB, read_only=True)
    q = f"""
    with base as (
        select b.trade_date, b.symbol, b.turnover_lacs, b.deliv_per, s.sector,
               greatest(least((b.close_price - b.prev_close)
                        / nullif(b.prev_close,0) * 100, {WINSOR}), -{WINSOR}) as r,
               lag(b.turnover_lacs) over (partition by b.symbol
                                          order by b.trade_date) as w_lag
        from daily_data b
        inner join v_sector_master s on b.symbol = s.symbol
        where b.series in ('EQ','SM','ST')
          and s.sector is not null and s.sector not in ('ETF','Others')
    )
    select sector, trade_date,
           sum(w_lag * r) / nullif(sum(case when r is not null then w_lag end),0) as ret,
           sum(turnover_lacs * deliv_per / 100.0) / 100.0                        as dv_cr,
           sum(deliv_per * turnover_lacs) / nullif(sum(turnover_lacs),0)         as deliv_pct
    from base
    where turnover_lacs >= {MIN_TURN} and w_lag is not null
    group by sector, trade_date order by sector, trade_date
    """
    d = con.execute(q).df()
    con.close()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    ret = d.pivot(index="trade_date", columns="sector", values="ret").sort_index()
    dv = d.pivot(index="trade_date", columns="sector", values="dv_cr").sort_index()
    dp = d.pivot(index="trade_date", columns="sector", values="deliv_pct").sort_index()
    return ret.dropna(how="all"), dv, dp


def rolling_slope(df: pd.DataFrame, win: int) -> pd.DataFrame:
    """OLS slope of each column over a trailing window (matches the Clock)."""
    x = np.arange(win, dtype=float)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    return df.rolling(win).apply(lambda y: float((xc * (y - y.mean())).sum() / denom), raw=True)


def main():
    ret, dv, dp = load()
    print(f"panel: {ret.shape[1]} sectors x {len(ret)} sessions "
          f"({ret.index.min():%Y-%m-%d} -> {ret.index.max():%Y-%m-%d})")
    lg = np.log1p(ret / 100.0)
    trail = lambda n: np.expm1(lg.rolling(n).sum()) * 100.0

    con = duckdb.connect(DB, read_only=True)
    nf = con.sql("""select trade_date, close_val from index_data
                    where index_name='Nifty 50' order by trade_date""").df()
    con.close()
    nf["trade_date"] = pd.to_datetime(nf["trade_date"])
    ngr = np.log1p(nf.set_index("trade_date")["close_val"].sort_index().pct_change())
    ntrail = lambda n: np.expm1(ngr.rolling(n).sum()).reindex(ret.index) * 100.0

    dv5 = dv.rolling(DV_FLOW).mean() / dv.shift(1).rolling(DV_BASE).mean()
    rk = lambda d: d.rank(axis=1, pct=True)

    def fwd(h):
        f = np.expm1(lg.shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]) * 100.0
        return f.sub(f.mean(axis=1), axis=0)

    def run(name, picker, h, k=4):
        """Non-overlapping rebalance every h sessions; returns net stats."""
        f = fwd(h)
        idx = ret.index[::h]
        rets, turns, prev = [], [], set()
        for dt_ in idx:
            pick = picker(dt_, k)
            if pick is None or len(pick) == 0:
                continue
            y = f.loc[dt_, list(pick)].dropna()
            if len(y) < max(2, len(pick) // 2):
                continue
            rets.append(float(y.mean()))
            turns.append(len(set(pick) - prev) / max(len(pick), 1))
            prev = set(pick)
        if len(rets) < 6:
            return None
        r = np.array(rets); t = np.array(turns)
        cost = t * 2 * COST_SIDE
        net = r - cost
        per_yr = 252.0 / h
        return dict(signal=name, n_reb=len(r), gross=r.mean(), cost=cost.mean(),
                    net=net.mean(), t=nw_t(net, 1), hit=100 * (net > 0).mean(),
                    net_yr=net.mean() * per_yr, turn=100 * t.mean())

    print("\n" + "=" * 108)
    print("HEAD-TO-HEAD — long-only top-4, EXCESS vs equal-weight sector basket, "
          "non-overlapping, net of 25bps/side")
    print("=" * 108)

    all_rows = []
    for lbl, h in HORIZONS.items():
        L, S = h, max(2, h // 2)
        rsL = trail(L).sub(ntrail(L), axis=0)
        rsS = trail(S).sub(ntrail(S), axis=0)
        tilt = W_RS2 * rk(rsL) + W_RS1 * rk(rsS) + W_DV5 * rk(dv5)

        # ---- Clock, lookback matched to the horizon ----
        slope = rolling_slope(dp, h)
        sz = slope.sub(slope.mean(axis=1), axis=0).div(slope.std(axis=1).replace(0, np.nan), axis=0)
        cum = trail(h)
        prel = cum.sub(cum.median(axis=1), axis=0)
        lead = (sz > 0.25) & (prel > 0.5)
        impr = (sz > 0.25) & (prel < -0.5)
        weak = (sz < -0.25) & (prel > 0.5)

        def top_tilt(dt_, k):
            s = tilt.loc[dt_].dropna()
            return list(s.nlargest(k).index) if len(s) >= k else None

        def q_pick(mask, rank_by):
            def _p(dt_, k):
                m = mask.loc[dt_]
                names = list(m[m.fillna(False)].index)
                if not names:
                    return None
                s = rank_by.loc[dt_, names].dropna()
                return list(s.nlargest(min(k, len(s))).index) if len(s) else None
            return _p

        def rand_pick(dt_, k):
            s = tilt.loc[dt_].dropna()
            if len(s) < k:
                return None
            return list(RNG.choice(list(s.index), size=k, replace=False))

        rows = [
            run("TILT top-4", top_tilt, h),
            run("CLOCK Leading", q_pick(lead, sz), h),
            run("CLOCK Improving", q_pick(impr, sz), h),
            run("CLOCK Lead+Impr", q_pick(lead | impr, sz), h),
            run("CLOCK Weakening (short-side check)", q_pick(weak, sz), h),
            run("BOTH agree (tilt top-8 AND Leading)",
                lambda dt_, k, _t=tilt, _l=lead: (
                    lambda names: (list(_t.loc[dt_, names].nlargest(min(k, len(names))).index)
                                   if len(names) else None)
                )([n for n in _l.loc[dt_][_l.loc[dt_].fillna(False)].index
                   if n in list(_t.loc[dt_].dropna().nlargest(8).index)]), h),
            run("random-4 (control)", rand_pick, h),
        ]
        rows = [r for r in rows if r]
        for r in rows:
            r["horizon"] = lbl
        all_rows += rows
        df = pd.DataFrame(rows)[["signal", "n_reb", "gross", "cost", "net", "t", "hit", "net_yr", "turn"]]
        df.columns = ["signal", "n_reb", "gross_%", "cost_%", "net_%", "net_t", "hit_%", "net_%/yr", "turn_%"]
        print(f"\n--- {lbl} (rebalance every {h} sessions) ---")
        print(df.round(2).to_string(index=False))

    A = pd.DataFrame(all_rows)
    print("\n" + "=" * 108)
    print("SUMMARY — net %/yr by signal x horizon")
    print("=" * 108)
    piv = A.pivot(index="signal", columns="horizon", values="net_yr")[list(HORIZONS)]
    print(piv.round(1).to_string())
    print("\nSUMMARY — net t by signal x horizon")
    print(A.pivot(index="signal", columns="horizon", values="t")[list(HORIZONS)].round(2).to_string())

    print("\n=== overlapping daily IC (Newey-West t at lag=h) — signal strength, cost-free ===")
    rows = []
    for lbl, h in HORIZONS.items():
        L, S = h, max(2, h // 2)
        rsL = trail(L).sub(ntrail(L), axis=0)
        rsS = trail(S).sub(ntrail(S), axis=0)
        tilt = W_RS2 * rk(rsL) + W_RS1 * rk(rsS) + W_DV5 * rk(dv5)
        slope = rolling_slope(dp, h)
        sz = slope.sub(slope.mean(axis=1), axis=0).div(slope.std(axis=1).replace(0, np.nan), axis=0)
        f = fwd(h)
        out = {"horizon": lbl}
        for nm, sig in (("tilt", tilt), ("clock_slope_z", sz)):
            ics = []
            for dt_ in sig.index:
                a, b = sig.loc[dt_], f.loc[dt_]
                m = a.notna() & b.notna()
                if m.sum() < 8:
                    continue
                ics.append(a[m].rank().corr(b[m].rank()))
            ics = np.array(ics, float)
            out[f"{nm}_IC"] = round(float(np.nanmean(ics)), 4)
            out[f"{nm}_t"] = round(nw_t(ics, h), 2)
        rows.append(out)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
