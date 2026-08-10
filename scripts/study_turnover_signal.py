"""
Does sector TURNOVER predict sector returns? And does it depend on the month?

WHY THIS IS A BETTER QUESTION THAN CALENDAR SEASONALITY
    A calendar month gives 8-13 observations per sector-month cell. Turnover is
    a STATE, not a label: every sector-month is an observation, so the panel is
    ~2,400 (DCM) / ~3,400 (NSE) rows across ~100-160 monthly cross-sections.
    That is roughly two orders of magnitude more power.

THE TRAP THAT MUST BE CONTROLLED
    Turnover and return are mechanically linked -- a sector that moves hard
    trades hard IN THE SAME MONTH. Three separate guards:
      1. STRICTLY FORWARD. Features are measured through month t, the target is
         excess return in month t+1. Nothing contemporaneous is scored.
      2. RESIDUALISED ON MOMENTUM. High turnover accompanies big price moves, so
         every turnover feature is also tested after cross-sectionally
         regressing out month-t excess return. If the signal dies there, it was
         momentum wearing a volume costume.
      3. RELATIVE, NOT ABSOLUTE. Rupee turnover trends up for years (market
         growth, more listings). Absolute turnover would just be a time trend.
         Everything is a SHARE of that month's total, or a z-score vs the
         sector's own trailing 12 months.

FEATURES (all causal, known at the close of month t)
    share      sector turnover / total turnover this month
    dshare     change in that share vs month t-1
    share_z    share vs the sector's own trailing 12m mean/sd
    tgrow      log(turnover_t / trailing 12m mean turnover)
    mom        month-t excess return                      (the control)
    dlv_z      delivery-share z-score, DCM lens only      (smart-money proxy)

PRIOR FROM THIS CODEBASE: stock-level accumulation is mildly ANTI-predictive
(rank-IC t~-3.5). Expect turnover surges to be CONTRARIAN at sector level too.
Both signs are tested; the sign is not assumed.

Usage:
  python scripts/study_turnover_signal.py --lens dcm
  python scripts/study_turnover_signal.py --lens nse
  python scripts/study_turnover_signal.py --lens both --months
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
RNG = np.random.default_rng(20260807)
WINSOR = 20.0
MIN_SESSIONS = 13
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

NSE_IDX = [
    "Nifty Auto", "Nifty Bank", "Nifty FMCG", "Nifty IT", "Nifty Media",
    "Nifty Metal", "Nifty Pharma", "Nifty PSU Bank", "Nifty Realty",
    "Nifty Financial Services", "Nifty Private Bank", "Nifty Energy",
    "Nifty Infrastructure", "Nifty Commodities", "Nifty India Consumption",
    "Nifty PSE", "Nifty CPSE", "Nifty MNC", "Nifty Services Sector",
    "Nifty Oil & Gas", "Nifty Consumer Durables", "Nifty Healthcare Index",
]


def t_stat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def _month_ok(idx: pd.DatetimeIndex) -> set:
    per = idx.to_period("M")
    cnt = pd.Series(1, index=idx).groupby(per).sum()
    thin = cnt[cnt < MIN_SESSIONS].index
    return set(thin) | {p + 1 for p in thin}


# ------------------------------------------------------------------ panels ---
def panel_dcm(con) -> pd.DataFrame:
    """Monthly sector panel: corrected returns + turnover + delivery."""
    q = f"""
    with base as (
        select b.trade_date, b.symbol, b.close_price, b.turnover_lacs, b.deliv_per,
               lag(b.close_price) over (partition by b.symbol order by b.trade_date) pc,
               lag(b.trade_date)  over (partition by b.symbol order by b.trade_date) pd,
               avg(b.turnover_lacs) over (partition by b.symbol order by b.trade_date
                    rows between 20 preceding and 1 preceding) w_lag
        from daily_data b where b.series in ('EQ','SM','ST')
    ),
    ok as (
        select trade_date, symbol, w_lag, turnover_lacs, deliv_per,
               greatest(least((close_price-pc)/pc*100.0, {WINSOR}), -{WINSOR}) r
        from base
        where pc > 0 and pd is not null
          and date_diff('day', pd, trade_date) <= 7 and w_lag > 0
    )
    select o.trade_date, s.sector,
           sum(o.r * o.w_lag)/nullif(sum(o.w_lag),0) as r,
           sum(o.turnover_lacs) as turn,
           sum(o.turnover_lacs * o.deliv_per)/nullif(sum(o.turnover_lacs),0) as dlv
    from ok o inner join v_sector_master s on o.symbol = s.symbol
    where s.sector is not null and s.sector not in ('ETF','Others')
    group by 1,2
    """
    d = con.execute(q).df()
    d["trade_date"] = pd.to_datetime(d.trade_date)
    bad = _month_ok(pd.DatetimeIndex(d.trade_date.unique()))
    d = d[~d.trade_date.dt.to_period("M").isin(bad)]
    d["ym"] = d.trade_date.dt.to_period("M")

    ret = (d.assign(g=1 + d.r / 100).groupby(["ym", "sector"]).g.prod() - 1) * 100
    turn = d.groupby(["ym", "sector"]).turn.sum()
    dlv = d.groupby(["ym", "sector"]).dlv.mean()
    p = pd.concat([ret.rename("ret"), turn.rename("turn"), dlv.rename("dlv")], axis=1)
    return p.reset_index()


def panel_nse(con) -> pd.DataFrame:
    ph = ",".join("?" for _ in NSE_IDX)
    d = con.execute(
        f"select trade_date,index_name,close_val,turnover_cr from index_data "
        f"where index_name in ({ph}) and close_val is not null", NSE_IDX).df()
    d["trade_date"] = pd.to_datetime(d.trade_date)
    bad = _month_ok(pd.DatetimeIndex(d.trade_date.unique()))
    d = d[~d.trade_date.dt.to_period("M").isin(bad)]
    d["ym"] = d.trade_date.dt.to_period("M")
    d = d.sort_values("trade_date")
    last = d.groupby(["ym", "index_name"]).close_val.last().rename("px")
    turn = d.groupby(["ym", "index_name"]).turnover_cr.sum().rename("turn")
    p = pd.concat([last, turn], axis=1).reset_index().rename(columns={"index_name": "sector"})
    p = p.sort_values(["sector", "ym"])
    p["ret"] = p.groupby("sector").px.pct_change() * 100
    # only keep consecutive-month returns
    p["gap"] = p.groupby("sector").ym.diff().apply(lambda x: getattr(x, "n", np.nan))
    p = p[p.gap == 1].drop(columns=["gap", "px"])
    p["dlv"] = np.nan
    return p


# ---------------------------------------------------------------- features ---
def build_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["sector", "ym"]).copy()
    # relative, not absolute: share of the month's total turnover
    p["share"] = p.turn / p.groupby("ym").turn.transform("sum")
    g = p.groupby("sector")
    p["dshare"] = g.share.diff()
    mu = g.share.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
    sd = g.share.transform(lambda s: s.shift(1).rolling(12, min_periods=6).std())
    p["share_z"] = (p.share - mu) / sd
    tmu = g.turn.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
    p["tgrow"] = np.log(p.turn / tmu)
    dmu = g.dlv.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
    dsd = g.dlv.transform(lambda s: s.shift(1).rolling(12, min_periods=6).std())
    p["dlv_z"] = (p.dlv - dmu) / dsd

    # excess vs the equal-weight sector basket, and the FORWARD target
    p["ex"] = p.ret - p.groupby("ym").ret.transform("mean")
    p["mom"] = p.ex
    p["fwd"] = g.ex.shift(-1)
    return p


# ------------------------------------------------------------------- tests ---
def ic_series(p: pd.DataFrame, feat: str, target="fwd") -> pd.Series:
    def one(gr):
        d = gr[[feat, target]].dropna()
        if len(d) < 6:
            return np.nan
        return d[feat].rank().corr(d[target].rank())
    return p.groupby("ym").apply(one).dropna()


def resid_on_mom(p: pd.DataFrame, feat: str) -> pd.DataFrame:
    """Cross-sectionally remove month-t excess return from the feature."""
    out = p.copy()
    res = np.full(len(out), np.nan)
    for _, idx in out.groupby("ym").groups.items():
        sub = out.loc[idx, [feat, "mom"]].dropna()
        if len(sub) < 6 or sub["mom"].std() == 0:
            continue
        x = sub["mom"].rank().values
        y = sub[feat].rank().values
        b = np.polyfit(x, y, 1)
        res[out.index.get_indexer(sub.index)] = y - (b[0] * x + b[1])
    out[feat + "_r"] = res
    return out


def quintiles(p: pd.DataFrame, feat: str, target="fwd", q=5) -> pd.DataFrame:
    rows = []
    for ym, gr in p.groupby("ym"):
        d = gr[[feat, target, "sector"]].dropna()
        if len(d) < q * 2:
            continue
        d = d.assign(b=pd.qcut(d[feat].rank(method="first"), q, labels=False))
        for b, sub in d.groupby("b"):
            rows.append({"ym": ym, "b": int(b), "r": sub[target].mean()})
    if not rows:
        return pd.DataFrame()
    z = pd.DataFrame(rows).pivot(index="ym", columns="b", values="r")
    res = pd.DataFrame({
        "bucket": [f"Q{i+1}" for i in range(q)],
        "mean_fwd": [z[i].mean() for i in range(q)],
        "t": [t_stat(z[i].values) for i in range(q)],
        "hit": [100 * (z[i] > 0).mean() for i in range(q)],
    })
    spread = z[q - 1] - z[0]
    return res, spread


def run(lens: str, con, show_months: bool):
    print(f"\n{'='*80}\n=== LENS: {lens.upper()} ===")
    p = panel_dcm(con) if lens == "dcm" else panel_nse(con)
    p = build_features(p)
    n_ok = p.dropna(subset=["fwd"]).shape[0]
    print(f"panel: {p.sector.nunique()} sectors x {p.ym.nunique()} months "
          f"= {n_ok:,} usable sector-months "
          f"({p.ym.min()} -> {p.ym.max()})")

    FEATS = ["share_z", "dshare", "tgrow"] + (["dlv_z"] if lens == "dcm" else [])

    print("\n--- 1. RAW forward IC (feature at t -> excess return at t+1) ---")
    print(f"{'feature':<12}{'meanIC':>9}{'t':>8}{'%>0':>7}{'n':>6}   interpretation")
    base = {}
    for f in FEATS + ["mom"]:
        ics = ic_series(p, f)
        base[f] = ics
        d = "higher turnover -> BETTER" if ics.mean() > 0 else "higher turnover -> WORSE"
        if f == "mom":
            d = "momentum continues" if ics.mean() > 0 else "momentum reverses"
        print(f"{f:<12}{ics.mean():+9.4f}{t_stat(ics.values):+8.2f}"
              f"{100*(ics>0).mean():6.0f}%{len(ics):6d}   {d}")

    print("\n--- 2. RESIDUALISED on month-t excess return (the real test) ---")
    print("    if a turnover signal dies here, it was momentum in disguise")
    print(f"{'feature':<14}{'meanIC':>9}{'t':>8}{'%>0':>7}   verdict")
    for f in FEATS:
        pr = resid_on_mom(p, f)
        ics = ic_series(pr, f + "_r")
        keep = abs(t_stat(ics.values) or 0) >= 2
        raw_t = t_stat(base[f].values)
        v = ("SURVIVES" if keep else
             "dies (was momentum)" if abs(raw_t or 0) >= 2 else "nothing either way")
        print(f"{f+'_r':<14}{ics.mean():+9.4f}{t_stat(ics.values):+8.2f}"
              f"{100*(ics>0).mean():6.0f}%   {v}")

    print("\n--- 3. QUINTILES by turnover z-score (Q1 = lowest, Q5 = highest) ---")
    out = quintiles(p, "share_z")
    if isinstance(out, tuple):
        qt, spread = out
        print(qt.round(3).to_string(index=False))
        print(f"  Q5-Q1 spread: {spread.mean():+.3f}%/mo  t={t_stat(spread.values):+.2f}  "
              f"hit {100*(spread>0).mean():.0f}%")
        print(f"  Q1-Q5 (contrarian): {-spread.mean():+.3f}%/mo  "
              f"t={-1*(t_stat(spread.values) or 0):+.2f}")

    if show_months:
        print("\n--- 4. BY CALENDAR MONTH: does the turnover signal work better in some months? ---")
        rows = []
        pr = resid_on_mom(p, "share_z")
        for m in range(1, 13):
            sub = pr[pr.ym.dt.month == m]
            ics = ic_series(sub, "share_z")
            icr = ic_series(sub, "share_z_r")
            rows.append({"month": MONTHS[m-1], "n_months": len(ics),
                         "IC_raw": round(ics.mean(), 4) if len(ics) else np.nan,
                         "t_raw": round(t_stat(ics.values), 2) if len(ics) else np.nan,
                         "IC_resid": round(icr.mean(), 4) if len(icr) else np.nan,
                         "t_resid": round(t_stat(icr.values), 2) if len(icr) else np.nan})
        md = pd.DataFrame(rows)
        print(md.to_string(index=False))
        # snooping control over the 12 months
        obs = np.nanmax(np.abs(md.t_raw.values.astype(float)))
        allic = ic_series(p, "share_z")
        mo = allic.index.month
        null = []
        for _ in range(4000):
            sh = RNG.permutation(allic.values)
            null.append(np.nanmax([abs(t_stat(sh[mo == m]) or 0) for m in range(1, 13)]))
        null = np.array(null)
        print(f"\n  best month |t| = {obs:.2f} | null 95th pct = {np.percentile(null,95):.2f} "
              f"| p = {(np.sum(null>=obs)+1)/4001:.4f}")
        print("  ->", "a month genuinely conditions the turnover signal"
              if (np.sum(null >= obs)+1)/4001 < 0.05 else
              "NO month conditions it beyond chance")

        print("\n--- 5. WHICH SECTOR, WHICH MONTH, given turnover state ---")
        print("    mean forward excess by (month, turnover tercile), pooled across sectors")
        pp = p.dropna(subset=["share_z", "fwd"]).copy()
        pp["ter"] = pp.groupby("ym").share_z.transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"])
            if len(s) >= 6 else np.nan)
        piv = pp.pivot_table(index=pp.ym.dt.month, columns="ter",
                             values="fwd", aggfunc="mean", observed=True)
        piv.index = [MONTHS[i-1] for i in piv.index]
        piv["high-low"] = piv["high"] - piv["low"]
        print(piv.round(2).to_string())

    print("\n--- 6. NET-OF-COST long/short on the surviving direction ---")
    pr = resid_on_mom(p, "share_z")
    ics = ic_series(pr, "share_z_r")
    sign = -1.0 if (ics.mean() or 0) < 0 else 1.0
    rows, prev_l, prev_s = [], set(), set()
    for ym, gr in pr.groupby("ym"):
        d = gr[["sector", "share_z_r", "fwd"]].dropna()
        if len(d) < 10:
            continue
        d = d.assign(sc=sign * d.share_z_r)
        L = set(d.nlargest(4, "sc").sector); S = set(d.nsmallest(4, "sc").sector)
        r = d[d.sector.isin(L)].fwd.mean() - d[d.sector.isin(S)].fwd.mean()
        churn = (len(L - prev_l) + len(S - prev_s)) / 8.0
        rows.append({"ym": ym, "r": r, "c": churn * 2 * 0.25 * 2})
        prev_l, prev_s = L, S
    if rows:
        z = pd.DataFrame(rows)
        net = z.r - z.c
        d = "CONTRARIAN (short high-turnover)" if sign < 0 else "MOMENTUM (buy high-turnover)"
        print(f"  direction: {d}")
        print(f"  n={len(z)} months | gross {z.r.mean():+.3f}%/mo t={t_stat(z.r.values):+.2f} "
              f"| cost {z.c.mean():.3f}% | NET {net.mean():+.3f}%/mo t={t_stat(net.values):+.2f} "
              f"| net ann {net.mean()*12:+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", choices=["dcm", "nse", "both"], default="both")
    ap.add_argument("--months", action="store_true")
    a = ap.parse_args()
    con = duckdb.connect(DB, read_only=True)
    for lens in (["dcm", "nse"] if a.lens == "both" else [a.lens]):
        run(lens, con, a.months)
    con.close()


if __name__ == "__main__":
    main()
