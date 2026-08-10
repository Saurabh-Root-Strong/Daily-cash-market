"""
Inside a Rotation-Clock sector, WHICH STOCK should you actually look at?

WHY THIS NEEDS MEASURING BEFORE ANY DRILL-DOWN SHIPS
    The Sector Rotation page already warns (8yr, 1.28M stock-days) that within a
    sector, high accumulation and high momentum are MILDLY ANTI-PREDICTIVE at the
    single-stock level, and the laggard tends to beat the extended leader over
    1-2 weeks. A "most probable participant stocks" list ranked the obvious way
    would therefore be ranked BACKWARDS. This re-tests that specifically inside
    the Clock's quadrants, which is a narrower and more relevant question.

WHAT IS MEASURED
    Forward return of each stock over the next h sessions, EXCESS OVER ITS OWN
    SECTOR's return over the same window. That isolates STOCK SELECTION from the
    sector call — the sector call is the Clock's job and is tested elsewhere.

STOCK FEATURES, all causal, known at the close of the signal day:
    mom_h      stock return over the trailing h sessions minus its sector's
    dacc       delivery % over last 5d divided by its own trailing 100d mean
    dvshare    stock's share of the sector's delivery value (who IS the flow)
    turnsurge  turnover over last 5d vs its own trailing 100d mean
    contrib    mom_h * dvshare — "how much did this name DRIVE the sector move"

INFERENCE
    IC is computed per (date, sector) cross-section, then averaged per date, then
    a Newey-West t at lag=h across dates (forward windows overlap).
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
H = 10                       # 2-week hold, the Clock's default lens
MIN_NAMES = 6                # need a real within-sector cross-section


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


def main():
    con = duckdb.connect(DB, read_only=True)
    q = f"""
    with b as (
        select b.trade_date, b.symbol, s.sector, b.turnover_lacs, b.deliv_per,
               greatest(least((b.close_price-b.prev_close)/nullif(b.prev_close,0)*100,
                        {WINSOR}), -{WINSOR}) as r,
               lag(b.turnover_lacs) over (partition by b.symbol order by b.trade_date) w_lag
        from daily_data b inner join v_sector_master s on b.symbol=s.symbol
        where b.series in ('EQ','SM','ST') and s.sector is not null
          and s.sector not in ('ETF','Others') and b.trade_date >= DATE '2018-01-01'
    )
    select trade_date, symbol, sector, turnover_lacs, deliv_per, r, w_lag
    from b where w_lag is not null and turnover_lacs >= {MIN_TURN}
    """
    d = con.execute(q).df()
    con.close()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    d = d.sort_values(["symbol", "trade_date"])
    print(f"stock-days: {len(d):,}  symbols {d.symbol.nunique():,}  "
          f"{d.trade_date.min():%Y-%m-%d} -> {d.trade_date.max():%Y-%m-%d}")

    g = d.groupby("symbol", sort=False)
    lg = np.log1p(d["r"] / 100.0)
    d["_lg"] = lg
    # trailing h-day stock return (causal, includes today)
    d["mom_h"] = g["_lg"].transform(lambda s: np.expm1(s.rolling(H).sum()) * 100)
    # forward h-day stock return (strictly future)
    d["fwd"] = g["_lg"].transform(
        lambda s: np.expm1(s.shift(-1)[::-1].rolling(H, min_periods=H).sum()[::-1]) * 100)
    # delivery accumulation vs own 100d
    d["dacc"] = (g["deliv_per"].transform(lambda s: s.rolling(5).mean())
                 / g["deliv_per"].transform(lambda s: s.shift(1).rolling(100).mean()))
    d["turnsurge"] = (g["turnover_lacs"].transform(lambda s: s.rolling(5).mean())
                      / g["turnover_lacs"].transform(lambda s: s.shift(1).rolling(100).mean()))
    d["dv"] = d["turnover_lacs"] * d["deliv_per"] / 100.0

    d = d.dropna(subset=["mom_h", "fwd", "dacc"])
    # within-sector-day aggregates
    k = ["trade_date", "sector"]
    d["sec_fwd"] = d.groupby(k)["fwd"].transform("mean")
    d["sec_mom"] = d.groupby(k)["mom_h"].transform("mean")
    d["dvshare"] = d["dv"] / d.groupby(k)["dv"].transform("sum")
    d["rel_mom"] = d["mom_h"] - d["sec_mom"]
    d["contrib"] = d["rel_mom"] * d["dvshare"]
    d["y"] = d["fwd"] - d["sec_fwd"]                 # WITHIN-sector excess

    n_names = d.groupby(k)["symbol"].transform("size")
    d = d[n_names >= MIN_NAMES]
    print(f"usable stock-days after filters: {len(d):,}  "
          f"sector-days {d.groupby(k).ngroups:,}")

    FEATS = ["rel_mom", "dacc", "dvshare", "turnsurge", "contrib"]
    print(f"\n=== within-sector stock IC vs forward {H}d excess-over-own-sector ===")
    print("    (positive = the HIGH-ranked stock outperforms its sector; "
          "negative = the LAGGARD does)")
    rows = []
    for f in FEATS:
        per_date = (d.dropna(subset=[f])
                     .groupby(k)
                     .apply(lambda gr: gr[f].rank().corr(gr["y"].rank()))
                     .groupby("trade_date").mean().dropna())
        rows.append({"feature": f, "n_dates": len(per_date),
                     "mean_IC": round(float(per_date.mean()), 4),
                     "t_NW": round(nw_t(per_date.values, H), 2),
                     "%>0": round(100 * float((per_date > 0).mean()))})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== decile spread on relative momentum (within sector) ===")
    dd = d.copy()
    dd["dec"] = dd.groupby(k)["rel_mom"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if len(s) >= 10 else np.nan)
    t = dd.dropna(subset=["dec"]).groupby("dec")["y"].agg(["mean", "size"])
    t.index = [f"Q{int(i)+1} ({'laggard' if i == 0 else 'leader' if i == 4 else ''})"
               for i in t.index]
    print(t.round(3).to_string())

    print("\n=== same, split by the sector's own state that day ===")
    sec_mom_rank = d.groupby("trade_date")["sec_mom"].rank(pct=True)
    d["_strong_sector"] = sec_mom_rank > 0.7
    for lbl, sub in (("STRONG sector (top-30% momentum)", d[d._strong_sector]),
                     ("rest", d[~d._strong_sector])):
        s = sub.copy()
        s["dec"] = s.groupby(k)["rel_mom"].transform(
            lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if len(x) >= 10 else np.nan)
        s = s.dropna(subset=["dec"])
        if s.empty:
            continue
        q1 = s.loc[s.dec == 0, "y"].mean(); q5 = s.loc[s.dec == 4, "y"].mean()
        pd_ = (s[s.dec.isin([0, 4])].groupby(["trade_date", "dec"])["y"].mean()
                .unstack().dropna())
        spread = (pd_[0.0] - pd_[4.0]) if 0.0 in pd_ and 4.0 in pd_ else pd.Series(dtype=float)
        print(f"  {lbl:<34} laggard {q1:+.3f}%  leader {q5:+.3f}%  "
              f"laggard-minus-leader {q1-q5:+.3f}pp  t {nw_t(spread.values, H):+.2f}")

    print("\n=== who DRIVES the sector (attribution), does it predict? ===")
    top_contrib = (d.sort_values("contrib", ascending=False)
                    .groupby(k).head(3).groupby("trade_date")["y"].mean().dropna())
    print(f"  top-3 contributors: mean within-sector excess {top_contrib.mean():+.3f}%/10d  "
          f"t {nw_t(top_contrib.values, H):+.2f}")
    print("  -> attribution answers 'who moved it', which is NOT the same as 'what to buy'.")


if __name__ == "__main__":
    main()
