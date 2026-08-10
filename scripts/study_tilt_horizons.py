"""
Does the 1-2 week Forward Tilt still work at 3-4, 5-6, 7-8, 9-10, 11-12 weeks?

WHY THIS MUST BE MEASURED BEFORE A HORIZON SELECTOR SHIPS
    The tilt's validation (daily-IC t~9, Monte-Carlo p<0.002) is a 1-2 WEEK
    result. Its inputs are a 10-day and a 5-day relative-strength plus a 5-day
    delivery-flow ratio. Nothing about that evidence carries to a 12-week
    forward horizon. Cross-sectional equity momentum is horizon-dependent and
    routinely REVERSES between one and three months, so a dropdown that simply
    re-runs the same score against a longer horizon would imply a validation
    that does not exist.

TWO SEPARATE QUESTIONS, BOTH TESTED
    A. Same signal, longer horizon: does score(10d RS) predict 4/6/8/10/12-week
       forward excess?
    B. Matched lookback: does a horizon-matched RS (e.g. 30-day RS for a 6-week
       forward) do better? If the tilt is to be offered at longer horizons, the
       lookback probably has to scale with the horizon.

METHOD
    - Sector panel reproduces the LIVE tilt exactly: EQ/SM/ST, sector NOT IN
      ('ETF','Others'), per-stock daily return winsorized at +/-25%, weighted by
      LAGGED turnover (w_lag), rows with NULL w_lag dropped -- so a listing day,
      whose bhavcopy prev_close is the IPO issue price, contributes nothing.
    - score = 0.60*rank(rs_2w) + 0.25*rank(rs_1w) + 0.15*rank(dv5d), exactly the
      shipped weights.
    - Forward return is the sector's excess over the equal-weight sector basket
      over the NEXT h trading days, strictly after the signal date.
    - Inference is DATE-CLUSTERED. Overlapping forward windows make adjacent
      dates dependent, so a per-row t-stat is badly inflated; the IC series is
      also autocorrelated by construction at horizon h. Reported t uses a
      Newey-West correction with lag = h.

Usage:  python scripts/study_tilt_horizons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
WINSOR = 25.0
MIN_TURNOVER_LACS = 100.0
W_RS2, W_RS1, W_DV5 = 0.60, 0.25, 0.15
MOM_2W, MOM_1W = 10, 5
DV_BASE, DV_FLOW = 100, 5
HORIZONS = {"1-2 wk": 10, "3-4 wk": 20, "5-6 wk": 30,
            "7-8 wk": 40, "9-10 wk": 50, "11-12 wk": 60}


def nw_t(x: np.ndarray, lag: int) -> float:
    """Newey-West t for the mean of an autocorrelated series."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    d = x - x.mean()
    g0 = (d @ d) / n
    var = g0
    for L in range(1, min(lag, n - 1) + 1):
        gL = (d[L:] @ d[:-L]) / n
        var += 2.0 * (1.0 - L / (lag + 1.0)) * gL
    if var <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(var / n))


def load_panel(con) -> pd.DataFrame:
    """Daily sector returns + delivery value, reproducing the live tilt's SQL."""
    q = f"""
    with base as (
        select b.trade_date, b.symbol, b.turnover_lacs, b.deliv_per, s.sector,
               greatest(least((b.close_price - b.prev_close)
                              / nullif(b.prev_close, 0) * 100, {WINSOR}), -{WINSOR}) as r,
               lag(b.turnover_lacs) over (partition by b.symbol
                                          order by b.trade_date) as w_lag
        from daily_data b
        inner join v_sector_master s on b.symbol = s.symbol
        where b.series in ('EQ','SM','ST')
          and s.sector is not null and s.sector not in ('ETF','Others')
    )
    select sector, trade_date,
           sum(turnover_lacs * deliv_per / 100.0) / 100.0 as daily_dv_cr,
           sum(w_lag * r) / nullif(sum(case when r is not null then w_lag end), 0) as wtd_ret_pct
    from base
    where turnover_lacs >= {MIN_TURNOVER_LACS} and w_lag is not null
    group by sector, trade_date
    order by sector, trade_date
    """
    d = con.execute(q).df()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    return d


def main():
    con = duckdb.connect(DB, read_only=True)
    panel = load_panel(con)
    nf = con.sql("""select trade_date, close_val from index_data
                    where index_name = 'Nifty 50' order by trade_date""").df()
    con.close()
    nf["trade_date"] = pd.to_datetime(nf["trade_date"])
    nifty = nf.set_index("trade_date")["close_val"].sort_index()
    nret = nifty.pct_change() * 100.0

    ret = panel.pivot(index="trade_date", columns="sector", values="wtd_ret_pct").sort_index()
    dv = panel.pivot(index="trade_date", columns="sector", values="daily_dv_cr").sort_index()
    ret = ret.dropna(how="all")
    print(f"panel: {ret.shape[1]} sectors x {len(ret)} sessions "
          f"({ret.index.min():%Y-%m-%d} -> {ret.index.max():%Y-%m-%d})")

    gr = 1.0 + ret / 100.0
    lg = np.log(gr)

    def trail(n):     # trailing n-day compound %, causal (includes today)
        return (np.exp(lg.rolling(n).sum()) - 1.0) * 100.0

    ngr = np.log(1.0 + nret / 100.0)
    def ntrail(n):
        return (np.exp(ngr.rolling(n).sum()) - 1.0) * 100.0

    mom2, mom1 = trail(MOM_2W), trail(MOM_1W)
    n2 = ntrail(MOM_2W).reindex(ret.index)
    n1 = ntrail(MOM_1W).reindex(ret.index)
    rs2 = mom2.sub(n2, axis=0)
    rs1 = mom1.sub(n1, axis=0)

    dvflow = dv.rolling(DV_FLOW).mean()
    dvbase = dv.shift(1).rolling(DV_BASE).mean()
    dv5d = dvflow / dvbase

    r = lambda df: df.rank(axis=1, pct=True)
    score = W_RS2 * r(rs2) + W_RS1 * r(rs1) + W_DV5 * r(dv5d)

    def fwd(h):       # strictly future h-day excess vs equal-weight sector basket
        f = (np.exp(lg.shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]) - 1.0) * 100.0
        return f.sub(f.mean(axis=1), axis=0)

    print("\n=== A. SHIPPED SIGNAL (10d/5d RS) vs each forward horizon ===")
    print("    date-clustered IC, Newey-West t at lag=h (overlapping windows)")
    rows = []
    for lbl, h in HORIZONS.items():
        f = fwd(h)
        ics, ls = [], []
        for dt_ in score.index:
            s, y = score.loc[dt_], f.loc[dt_]
            m = s.notna() & y.notna()
            if m.sum() < 8:
                continue
            ics.append(s[m].rank().corr(y[m].rank()))
            k = max(3, int(m.sum() // 5))
            top = y[m].loc[s[m].nlargest(k).index].mean()
            bot = y[m].loc[s[m].nsmallest(k).index].mean()
            ls.append(top - bot)
        ics, ls = np.array(ics, float), np.array(ls, float)
        rows.append({"horizon": lbl, "days": h, "n_dates": len(ics),
                     "mean_IC": round(np.nanmean(ics), 4),
                     "IC_t(NW)": round(nw_t(ics, h), 2),
                     "%IC>0": round(100 * np.nanmean(ics > 0)),
                     "LS_%": round(np.nanmean(ls), 2),
                     "LS_t(NW)": round(nw_t(ls, h), 2)})
    a = pd.DataFrame(rows)
    print(a.to_string(index=False))

    print("\n=== B. MATCHED LOOKBACK: RS window scaled to the horizon ===")
    print("    (if a longer horizon is to be offered, does its own lookback do better?)")
    rows = []
    for lbl, h in HORIZONS.items():
        rs_h = trail(h).sub(ntrail(h).reindex(ret.index), axis=0)
        sc_h = r(rs_h)
        f = fwd(h)
        ics = []
        for dt_ in sc_h.index:
            s, y = sc_h.loc[dt_], f.loc[dt_]
            m = s.notna() & y.notna()
            if m.sum() < 8:
                continue
            ics.append(s[m].rank().corr(y[m].rank()))
        ics = np.array(ics, float)
        rows.append({"horizon": lbl, "lookback_d": h, "n_dates": len(ics),
                     "mean_IC": round(np.nanmean(ics), 4),
                     "IC_t(NW)": round(nw_t(ics, h), 2),
                     "%IC>0": round(100 * np.nanmean(ics > 0))})
    b = pd.DataFrame(rows)
    print(b.to_string(index=False))

    print("\n=== C. REVERSAL CHECK: sign of IC by horizon (shipped signal) ===")
    for _, x in a.iterrows():
        verdict = ("PREDICTIVE" if x["IC_t(NW)"] >= 2 else
                   "REVERSES" if x["IC_t(NW)"] <= -2 else "no signal")
        print(f"  {x['horizon']:<9} IC {x['mean_IC']:+.4f}  t {x['IC_t(NW)']:+5.2f}   {verdict}")

    print("\n=== D. per-era stability of the shipped signal, by horizon ===")
    yrs = score.index.year
    out = []
    for lbl, h in HORIZONS.items():
        f = fwd(h)
        row = {"horizon": lbl}
        for era, msk in (("2018-21", (yrs >= 2018) & (yrs <= 2021)),
                         ("2022-24", (yrs >= 2022) & (yrs <= 2024)),
                         ("2025-26", yrs >= 2025)):
            ics = []
            for dt_ in score.index[msk]:
                s, y = score.loc[dt_], f.loc[dt_]
                m = s.notna() & y.notna()
                if m.sum() < 8:
                    continue
                ics.append(s[m].rank().corr(y[m].rank()))
            row[era] = round(float(np.nanmean(ics)), 4) if ics else np.nan
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()


def scaled_composite_study():
    """The variant actually worth shipping: keep the 0.60/0.25/0.15 structure but
    scale the RS lookbacks with the horizon (long=h, short=h/2). At h=10 this is
    EXACTLY the shipped tilt (10d/5d), so 1-2wk stays bit-identical."""
    con = duckdb.connect(DB, read_only=True)
    panel = load_panel(con)
    nf = con.sql("""select trade_date, close_val from index_data
                    where index_name = 'Nifty 50' order by trade_date""").df()
    con.close()
    nf["trade_date"] = pd.to_datetime(nf["trade_date"])
    nret = nf.set_index("trade_date")["close_val"].sort_index().pct_change() * 100.0

    ret = panel.pivot(index="trade_date", columns="sector", values="wtd_ret_pct").sort_index().dropna(how="all")
    dv = panel.pivot(index="trade_date", columns="sector", values="daily_dv_cr").sort_index()
    lg = np.log(1.0 + ret / 100.0)
    ngr = np.log(1.0 + nret / 100.0)
    trail = lambda n: (np.exp(lg.rolling(n).sum()) - 1.0) * 100.0
    ntrail = lambda n: (np.exp(ngr.rolling(n).sum()) - 1.0) * 100.0
    dv5d = dv.rolling(DV_FLOW).mean() / dv.shift(1).rolling(DV_BASE).mean()
    r = lambda d: d.rank(axis=1, pct=True)

    def fwd(h):
        f = (np.exp(lg.shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]) - 1.0) * 100.0
        return f.sub(f.mean(axis=1), axis=0)

    print("\n=== E. SCALED COMPOSITE (long=h, short=h/2, same 0.60/0.25/0.15) ===")
    print("    h=10 is bit-identical to the shipped tilt")
    rows = []
    for lbl, h in HORIZONS.items():
        L, S = h, max(2, h // 2)
        rsL = trail(L).sub(ntrail(L).reindex(ret.index), axis=0)
        rsS = trail(S).sub(ntrail(S).reindex(ret.index), axis=0)
        sc = W_RS2 * r(rsL) + W_RS1 * r(rsS) + W_DV5 * r(dv5d)
        f = fwd(h)
        ics, ls = [], []
        for dt_ in sc.index:
            s, y = sc.loc[dt_], f.loc[dt_]
            m = s.notna() & y.notna()
            if m.sum() < 8:
                continue
            ics.append(s[m].rank().corr(y[m].rank()))
            k = max(3, int(m.sum() // 5))
            ls.append(y[m].loc[s[m].nlargest(k).index].mean() - y[m].loc[s[m].nsmallest(k).index].mean())
        ics, ls = np.array(ics, float), np.array(ls, float)
        naive = float(np.nanmean(ics) / np.nanstd(ics) * np.sqrt(len(ics)))
        rows.append({"horizon": lbl, "lookL": L, "lookS": S, "n": len(ics),
                     "mean_IC": round(float(np.nanmean(ics)), 4),
                     "t_naive": round(naive, 2),
                     "t_NW": round(nw_t(ics, h), 2),
                     "%IC>0": round(100 * float(np.nanmean(ics > 0))),
                     "LS_%": round(float(np.nanmean(ls)), 2),
                     "LS_t_NW": round(nw_t(ls, h), 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  t_naive vs t_NW: the naive column ignores that forward windows OVERLAP.")
    print("  The module docstring's 'daily-IC t~9' is a naive-style number; the")
    print("  overlap-corrected value is what should drive any UI claim.")


scaled_composite_study()
