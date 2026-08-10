"""
Sector x month calendar map: which sector usually performs well in which month.

Produces the descriptive map for BOTH taxonomies, with an honest confidence tier
on every cell:

  NSE  = 22 NSE sector indices from index_data (clean, exchange-computed, 2013+)
  DCM  = 24 canonical buckets from v_sector_master + daily_data (2018+)

THREE CONTAMINATIONS FIXED IN THE DCM LENS
------------------------------------------
The shipped aggregation (src/analytics/sector_aggregator.py:187-190) returns a
mean of +0.534%/DAY = 134%/yr, which is obviously not a return series. Causes:

1. LISTING POPS. On a listing day the bhavcopy `prev_close` is the IPO ISSUE
   PRICE, so day 1 books the listing gain as a return: WINSOL 75->383 (+411%),
   KCEIL 54->239 (+343%). Fixed by deriving the previous close from the symbol's
   OWN prior session (lag) and requiring that session to be within 7 days, so a
   newly-listed or long-suspended symbol simply has no return that day.

2. SAME-DAY TURNOVER WEIGHTING. Weighting today's return by today's turnover is
   mechanically upward biased: a stock that moves hard trades hard on the SAME
   day, so big movers get big weight in the very bar they moved. Fixed with a
   LAGGED weight (trailing 20-session mean turnover, shifted one day).

3. UNADJUSTED CORPORATE ACTIONS. Splits/bonuses are not adjusted in daily_data,
   producing fake -50% bars. Per-stock daily returns are winsorized at +/-20%
   and the affected count is reported.

CONFIDENCE TIERS (the point of the exercise -- a map without these is a lie)
   STRONG : survives the permutation max-|t| control over the whole grid
   WEAK   : nominally significant, does NOT survive multiple-testing control
   NOISE  : everything else -- report the number, do not trade it

Usage:
  python scripts/sector_month_map.py                 # both lenses, ranked map
  python scripts/sector_month_map.py --csv out.csv   # dump full grid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
RNG = np.random.default_rng(20260807)
N_PERM = 4000
WINSOR = 20.0

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SECTOR_IDX = [
    "Nifty Auto", "Nifty Bank", "Nifty FMCG", "Nifty IT", "Nifty Media",
    "Nifty Metal", "Nifty Pharma", "Nifty PSU Bank", "Nifty Realty",
    "Nifty Financial Services", "Nifty Private Bank", "Nifty Energy",
    "Nifty Infrastructure", "Nifty Commodities", "Nifty India Consumption",
    "Nifty PSE", "Nifty CPSE", "Nifty MNC", "Nifty Services Sector",
    "Nifty Oil & Gas", "Nifty Consumer Durables", "Nifty Healthcare Index",
]
BENCH = "Nifty 50"


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x))


def _month_guard(px: pd.DataFrame) -> pd.DataFrame:
    per = px.index.to_period("M")
    cnt = pd.Series(1, index=px.index).groupby(per).sum()
    thin = cnt[cnt < 13].index
    if len(thin):
        px = px[~per.isin(set(thin) | {p + 1 for p in thin})]
    return px


def _monthly_from_px(px: pd.DataFrame) -> pd.DataFrame:
    px = _month_guard(px)
    me = px.groupby([px.index.year, px.index.month]).tail(1)
    r = me.pct_change() * 100.0
    r.index = pd.to_datetime(me.index)
    ords = r.index.to_period("M").astype("int64").to_numpy()
    return r[np.r_[False, np.diff(ords) == 1]]


def load_nse(con):
    names = SECTOR_IDX + [BENCH]
    d = con.execute(
        f"select trade_date,index_name,close_val from index_data "
        f"where index_name in ({','.join('?' * len(names))}) and close_val is not null",
        names).df()
    d["trade_date"] = pd.to_datetime(d.trade_date)
    px = d.pivot(index="trade_date", columns="index_name", values="close_val").sort_index()
    r = _monthly_from_px(px)
    return r[[c for c in r.columns if c != BENCH]], r[BENCH]


def load_dcm(con):
    """Corrected bottom-up aggregation -- see module docstring."""
    q = f"""
    with base as (
        select b.trade_date, b.symbol, b.close_price, b.turnover_lacs,
               lag(b.close_price) over (partition by b.symbol order by b.trade_date) pc,
               lag(b.trade_date)  over (partition by b.symbol order by b.trade_date) pd,
               avg(b.turnover_lacs) over (partition by b.symbol order by b.trade_date
                    rows between 20 preceding and 1 preceding) w_lag
        from daily_data b
        where b.series in ('EQ','SM','ST')
    ),
    ok as (
        select trade_date, symbol, w_lag,
               greatest(least((close_price - pc) / pc * 100.0, {WINSOR}), -{WINSOR}) as r,
               case when abs((close_price - pc) / pc * 100.0) > {WINSOR} then 1 else 0 end as clipped
        from base
        where pc > 0
          and pd is not null
          and date_diff('day', pd, trade_date) <= 7   -- excludes listing day & long suspensions
          and w_lag > 0
    )
    select o.trade_date, s.sector,
           sum(o.r * o.w_lag) / nullif(sum(o.w_lag), 0) as r,
           sum(o.clipped) as clipped, count(*) as n
    from ok o
    inner join v_sector_master s on o.symbol = s.symbol
    where s.sector is not null and s.sector not in ('ETF','Others')
    group by 1, 2
    """
    d = con.execute(q).df()
    d["trade_date"] = pd.to_datetime(d.trade_date)
    tot, clip = int(d["n"].sum()), int(d["clipped"].sum())
    print(f"  [dcm] winsorized {clip:,} of {tot:,} stock-days ({100*clip/tot:.3f}%) at +/-{WINSOR}%")
    daily = d.pivot(index="trade_date", columns="sector", values="r").sort_index() / 100.0
    print(f"  [dcm] mean daily sector return now {daily.stack().mean()*100:+.4f}%/day "
          f"-> {daily.stack().mean()*100*250:+.1f}%/yr  (was +0.534%/day = +134%/yr)")
    px = (1 + daily.fillna(0)).cumprod()
    r = _monthly_from_px(px)
    return r, r.mean(axis=1)


def build_map(sect: pd.DataFrame, bench: pd.Series, label: str, min_n=6):
    ex_raw = sect.sub(bench, axis=0)
    ex = ex_raw.sub(ex_raw.mean(axis=0), axis=1)      # isolate month from drift

    cols = list(ex.columns)
    months = ex.index.month.values
    obs_t = np.full((len(cols), 12), np.nan)
    for j, c in enumerate(cols):
        v = ex[c].values
        for m in range(12):
            x = v[months == m + 1]
            if np.sum(~np.isnan(x)) >= min_n:
                obs_t[j, m] = tstat(x)

    # permutation null for the GLOBAL max |t| across the whole grid
    maxnull = np.empty(N_PERM)
    for b in range(N_PERM):
        tb = np.full_like(obs_t, np.nan)
        for j, c in enumerate(cols):
            v = ex[c].values.copy()
            mk = ~np.isnan(v)
            v[mk] = RNG.permutation(v[mk])
            for m in range(12):
                x = v[months == m + 1]
                if np.sum(~np.isnan(x)) >= min_n:
                    tb[j, m] = tstat(x)
        maxnull[b] = np.nanmax(np.abs(tb))
    crit = np.percentile(maxnull, 95)

    rows = []
    for j, c in enumerate(cols):
        v = ex[c]
        for m in range(12):
            x = v[v.index.month == m + 1].dropna().values
            if len(x) < min_n:
                continue
            t = obs_t[j, m]
            tier = ("STRONG" if abs(t) >= crit else
                    "WEAK" if abs(t) >= 2.0 else "NOISE")
            rows.append({"lens": label, "sector": c, "month": MONTHS[m], "m": m + 1,
                         "n": len(x), "mean": x.mean(), "median": float(np.median(x)),
                         "hit": 100 * (x > 0).mean(), "t": t, "tier": tier})
    return pd.DataFrame(rows), crit, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    a = ap.parse_args()
    con = duckdb.connect(DB, read_only=True)

    out = []
    for label, loader in (("NSE", load_nse), ("DCM", load_dcm)):
        print(f"\n{'='*78}\n=== {label} lens ===")
        sect, bench = loader(con)
        print(f"  months {len(sect)} ({sect.index.min():%Y-%m}..{sect.index.max():%Y-%m}), "
              f"sectors {sect.shape[1]}")
        tab, crit, ex = build_map(sect, bench, label)
        print(f"  permutation 95% critical |t| for this grid: {crit:.2f}  "
              f"(a cell must beat this to be STRONG)")
        print(f"  tiers: " + ", ".join(f"{k}={v}" for k, v in
                                       tab.tier.value_counts().to_dict().items()))

        print(f"\n  --- BEST month per sector ({label}) ---")
        best = (tab.sort_values("mean", ascending=False)
                   .groupby("sector", as_index=False).first()
                   .sort_values("mean", ascending=False))
        print(best[["sector", "month", "n", "mean", "median", "hit", "t", "tier"]]
              .round(2).to_string(index=False))

        print(f"\n  --- WORST month per sector ({label}) ---")
        worst = (tab.sort_values("mean").groupby("sector", as_index=False).first()
                    .sort_values("mean"))
        print(worst[["sector", "month", "n", "mean", "median", "hit", "t", "tier"]]
              .round(2).to_string(index=False))
        out.append(tab)

    allt = pd.concat(out, ignore_index=True)

    # cross-lens agreement: the strongest available evidence in this data
    print(f"\n{'='*78}\n=== CROSS-LENS AGREEMENT ===")
    print("Two independent taxonomies over different windows. A month effect that")
    print("is real should appear in BOTH. Matching NSE<->DCM sector pairs:")
    PAIR = {
        "Nifty Auto": "Automobile", "Nifty Bank": "Banking",
        "Nifty IT": "IT", "Nifty Pharma": "Pharma & Healthcare",
        "Nifty FMCG": "FMCG", "Nifty Metal": "Metals & Mining",
        "Nifty Realty": "Realty", "Nifty Media": "Media & Entertainment",
        "Nifty Financial Services": "Financial Services",
        "Nifty Oil & Gas": "Oil & Gas", "Nifty Infrastructure": "Infrastructure",
        "Nifty Consumer Durables": "Consumer Durables",
    }
    n_lo = allt[allt.lens == "NSE"].set_index(["sector", "m"])
    d_lo = allt[allt.lens == "DCM"].set_index(["sector", "m"])
    agree = []
    for nse, dcm in PAIR.items():
        for m in range(1, 13):
            if (nse, m) in n_lo.index and (dcm, m) in d_lo.index:
                a1, b1 = n_lo.loc[(nse, m)], d_lo.loc[(dcm, m)]
                if np.sign(a1["mean"]) == np.sign(b1["mean"]) and \
                   min(abs(a1.t), abs(b1.t)) >= 1.5:
                    agree.append({"sector": nse, "month": MONTHS[m - 1],
                                  "NSE_mean": a1["mean"], "NSE_t": a1.t,
                                  "DCM_mean": b1["mean"], "DCM_t": b1.t,
                                  "NSE_hit": a1.hit, "DCM_hit": b1.hit})
    ag = pd.DataFrame(agree)
    if ag.empty:
        print("  NONE -- no sector-month agrees across both lenses at |t|>=1.5.")
    else:
        ag["minabs_t"] = ag[["NSE_t", "DCM_t"]].abs().min(axis=1)
        print(ag.sort_values("minabs_t", ascending=False)
                .drop(columns="minabs_t").round(2).to_string(index=False))

    if a.csv:
        allt.to_csv(a.csv, index=False)
        print(f"\nfull grid -> {a.csv}")
    con.close()


if __name__ == "__main__":
    main()
