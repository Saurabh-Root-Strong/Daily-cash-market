"""
Sector calendar-seasonality study.

THE QUESTION: in which month do sectors "perform well"?

THE TRAP THAT KILLS NAIVE VERSIONS OF THIS STUDY
------------------------------------------------
1. RAW MONTHLY RETURNS MOSTLY MEASURE THE MARKET. A sector's July return is
   market beta + sector-specific. If the market happened to rise in most Julys,
   every sector shows "July is good" -- that is ONE observation reported twenty
   times, not twenty pieces of evidence. Everything here is therefore run on
   EXCESS return vs Nifty 50 as the primary lens, with the absolute view kept
   only to describe the market's own seasonality separately.

2. MULTIPLE TESTING. ~20 sectors x 12 months = ~240 cells. At alpha=0.05 you
   EXPECT ~12 "significant" cells from pure noise. Reported here:
     - naive significant count (what a naive study would print)
     - Benjamini-Hochberg FDR survivors
     - a PERMUTATION max-|t| test (White's Reality Check in spirit): shuffle
       month labels within each sector, record the max |t| over ALL cells, and
       ask whether the observed best cell beats that null. This is the only
       honest test of "is the best month real?"

3. TINY n. 13 observations per cell at best. Monthly returns are fat-tailed and
   t-tests lean on assumptions that do not hold. Every p-value here is
   bootstrap/permutation based (no scipy in this env, which is fine -- these are
   more appropriate at n=13 anyway). Hit-rate uses an exact-ish sign test.

4. ONE YEAR CAN MANUFACTURE A SEASON. March 2020 alone (-23%) can create a
   "March is terrible" effect. Jackknife drops each year in turn and reports the
   WORST-CASE surviving effect.

5. PRICE INDEX, NOT TOTAL RETURN -- and Indian dividend ex-dates are themselves
   seasonal (clustered Feb-Aug). High-yield sectors (PSU Bank, CPSE, FMCG) will
   show a spurious negative drift in ex-date-heavy months. This CANNOT be fixed
   with the data available; it is quantified and flagged instead.

6. SELECTION BIAS IN "BEST MONTH". Reporting max-over-12-months is upward
   biased by construction. An out-of-sample split (fit early, test late) is the
   only claim that means anything for trading.

7. COSTS. Monthly rotation is 12 round-trips/yr. At 25bps/side that is ~6%/yr
   of drag. Gross seasonality that looks impressive routinely dies here --
   consistent with this codebase's recurring cost-floor result.

Usage:
  python scripts/study_sector_seasonality.py --source index    # NSE sector indices
  python scripts/study_sector_seasonality.py --source dcm      # DCM 26 canonical buckets
  python scripts/study_sector_seasonality.py --source index --full
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
N_BOOT = 10_000
N_PERM = 5_000

BENCH = "Nifty 50"

# NSE sector indices only -- strategy/thematic/factor indices excluded, they are
# not "sectors" and would inflate the multiple-testing burden with near-duplicates.
SECTOR_IDX = [
    "Nifty Auto", "Nifty Bank", "Nifty FMCG", "Nifty IT", "Nifty Media",
    "Nifty Metal", "Nifty Pharma", "Nifty PSU Bank", "Nifty Realty",
    "Nifty Financial Services", "Nifty Private Bank", "Nifty Energy",
    "Nifty Infrastructure", "Nifty Commodities", "Nifty India Consumption",
    "Nifty PSE", "Nifty CPSE", "Nifty MNC", "Nifty Services Sector",
    "Nifty Oil & Gas", "Nifty Consumer Durables", "Nifty Healthcare Index",
]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ----------------------------------------------------------------- loading ---
def load_index_monthly(con) -> tuple[pd.DataFrame, pd.Series]:
    """Month-end close -> monthly % return, per sector index, plus benchmark."""
    names = SECTOR_IDX + [BENCH]
    q = f"""
        select trade_date, index_name, close_val
        from index_data
        where index_name in ({','.join('?' for _ in names)})
          and close_val is not null
        order by trade_date
    """
    df = con.execute(q, names).df()
    df["trade_date"] = pd.to_datetime(df.trade_date)
    px = df.pivot(index="trade_date", columns="index_name", values="close_val").sort_index()

    # MONTH COMPLETENESS GUARD. NSE's Daily_Snapshot archive has a real hole
    # (2017-05-09 .. 2017-06-29, ~40 consecutive sessions serve no file). Without
    # this guard the "May 2017" return would anchor on 2017-05-08 and the
    # "June 2017" return would silently span 2017-05-08 -> 2017-06-30, i.e. a
    # two-month return mislabelled as one month. Any month with too few sessions
    # is dropped entirely, AND the month after it is dropped too (its return is
    # measured from a stale anchor).
    per = px.index.to_period("M")
    cnt = pd.Series(1, index=px.index).groupby(per).sum()
    thin = cnt[cnt < 13].index
    if len(thin):
        drop = set(thin) | {p + 1 for p in thin}
        print(f"  [edge] thin months dropped (<13 sessions, plus the month after "
              f"each, whose anchor is stale): {sorted(str(p) for p in thin)}")
        px = px[~per.isin(drop)]

    # Last TRADED session of each calendar month (not calendar month-end: NSE
    # holidays and the special Saturday/Sunday sessions make those differ).
    me = px.groupby([px.index.year, px.index.month]).tail(1)
    ret = me.pct_change() * 100.0
    ret.index = pd.to_datetime(me.index)

    # A month-over-month return is only valid if the PRIOR month is the
    # immediately preceding calendar month; otherwise it spans a gap.
    ords = ret.index.to_period("M").astype("int64").to_numpy()
    valid = np.r_[False, np.diff(ords) == 1]
    ret = ret[valid]

    # Drop an incomplete trailing month: if the last session in the data is not
    # near the end of its month, that month's return is a stub and must not be
    # counted as a "month".
    last = px.index.max()
    nxt = (last + pd.offsets.MonthEnd(0))
    if (nxt - last).days > 5:
        ret = ret[ret.index < pd.Timestamp(last.year, last.month, 1)]
        print(f"  [edge] dropped incomplete trailing month {last:%Y-%m} "
              f"(last session {last:%Y-%m-%d})")

    ret = ret.dropna(how="all").iloc[1:]          # first row is NaN by construction
    bench = ret[BENCH]
    sect = ret[[c for c in ret.columns if c != BENCH]]
    return sect, bench


def load_dcm_monthly(con) -> tuple[pd.DataFrame, pd.Series]:
    """DCM 26-bucket sectors: turnover-weighted daily returns -> monthly compound."""
    q = """
    select b.trade_date, s.sector,
           sum(case when b.prev_close>0
                    then ((b.close_price-b.prev_close)/b.prev_close*100.0)*b.turnover_lacs
                    else 0 end)
           / nullif(sum(case when b.prev_close>0 then b.turnover_lacs else 0 end),0) as r
    from daily_data b
    inner join v_sector_master s on b.symbol = s.symbol
    where s.sector is not null and s.sector not in ('ETF','Others')
      and b.series in ('EQ','SM','ST') and b.turnover_lacs >= 100
    group by 1,2
    """
    d = con.execute(q).df()
    d["trade_date"] = pd.to_datetime(d.trade_date)
    daily = d.pivot(index="trade_date", columns="sector", values="r").sort_index() / 100.0
    monthly = (1 + daily).groupby([daily.index.year, daily.index.month]).prod() - 1
    monthly = monthly * 100.0
    monthly.index = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in monthly.index])

    # equal-weight sector basket as the benchmark for the DCM lens
    bench = monthly.mean(axis=1)
    # drop incomplete trailing month
    last = daily.index.max()
    if (last + pd.offsets.MonthEnd(0) - last).days > 5:
        cut = pd.Timestamp(last.year, last.month, 1)
        monthly, bench = monthly[monthly.index < cut], bench[bench.index < cut]
    return monthly, bench


# ------------------------------------------------------------------- stats ---
def tstat(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x))


def boot_ci(x: np.ndarray, n=N_BOOT, lo=2.5, hi=97.5):
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return (np.percentile(means, lo), np.percentile(means, hi))


def sign_p(x: np.ndarray) -> float:
    """Two-sided exact binomial p for hit-rate != 50%, via exact enumeration."""
    x = x[~np.isnan(x)]
    n, k = len(x), int((x > 0).sum())
    if n == 0:
        return np.nan
    from math import comb
    pmf = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pmf[k]
    return float(sum(p for p in pmf if p <= obs * (1 + 1e-12)))


def bh_fdr(pvals: np.ndarray, q=0.10) -> np.ndarray:
    """Benjamini-Hochberg. Returns boolean mask of discoveries at level q."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.zeros(len(p), bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        out[order[:kmax + 1]] = True
    return out


def perm_p_and_maxnull(ex: pd.DataFrame, n_perm=N_PERM):
    """
    Permutation test. Under H0 the month LABEL carries no information, so month
    labels are exchangeable within a sector's own return history. Shuffling
    within sector preserves each sector's mean/vol and the cross-sectional
    correlation structure of the panel is retained by permuting the shared
    date-index identically per draw is NOT done deliberately: we permute per
    sector independently, which is the stricter null for a per-cell claim.

    Returns per-cell permutation p (two-sided) and the null distribution of the
    GLOBAL max |t| across all cells.
    """
    months = ex.index.month.values
    cols = list(ex.columns)
    obs_t = np.full((len(cols), 12), np.nan)
    for j, c in enumerate(cols):
        v = ex[c].values
        for m in range(1, 13):
            obs_t[j, m - 1] = tstat(v[months == m])

    ge = np.zeros_like(obs_t)
    maxnull = np.empty(n_perm)
    for b in range(n_perm):
        tb = np.full_like(obs_t, np.nan)
        for j, c in enumerate(cols):
            v = ex[c].values.copy()
            mask = ~np.isnan(v)
            vv = v.copy()
            vv[mask] = RNG.permutation(v[mask])
            for m in range(1, 13):
                tb[j, m - 1] = tstat(vv[months == m])
        with np.errstate(invalid="ignore"):
            ge += (np.abs(tb) >= np.abs(obs_t)).astype(float)
        maxnull[b] = np.nanmax(np.abs(tb))
    pperm = (ge + 1) / (n_perm + 1)
    return obs_t, pperm, maxnull


# ------------------------------------------------------------------ report ---
def cell_table(ex: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ex.columns:
        v = ex[c]
        for m in range(1, 13):
            x = v[v.index.month == m].values
            x = x[~np.isnan(x)]
            if len(x) < 4:
                continue
            lo, hi = boot_ci(x)
            rows.append({
                "sector": c, "month": MONTHS[m - 1], "m": m, "n": len(x),
                "mean": x.mean(), "median": np.median(x),
                "hit": 100 * (x > 0).mean(), "t": tstat(x),
                "ci_lo": lo, "ci_hi": hi, "sign_p": sign_p(x),
                "worst_yr_drop": _jackknife_worst(x),
            })
    return pd.DataFrame(rows)


def _jackknife_worst(x: np.ndarray) -> float:
    """Mean after removing the single most favourable observation."""
    if len(x) < 4:
        return np.nan
    if x.mean() >= 0:
        return float(np.delete(x, np.argmax(x)).mean())
    return float(np.delete(x, np.argmin(x)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["index", "dcm"], default="index")
    ap.add_argument("--full", action="store_true", help="print every cell")
    a = ap.parse_args()

    con = duckdb.connect(DB, read_only=True)
    print(f"=== source: {a.source} ===")
    if a.source == "index":
        sect, bench = load_index_monthly(con)
        blabel = BENCH
    else:
        sect, bench = load_dcm_monthly(con)
        blabel = "equal-weight sector basket"
    con.close()

    print(f"months: {len(sect)}  ({sect.index.min():%Y-%m} -> {sect.index.max():%Y-%m})")
    print(f"sectors: {sect.shape[1]}   benchmark: {blabel}")
    yrs = sect.index.year
    print("obs per month:", dict(pd.Series(sect.index.month).value_counts().sort_index()))

    # ---------- market's own seasonality (asked separately, on purpose) ----------
    print(f"\n=== A. MARKET seasonality ({blabel}, absolute) ===")
    mk = []
    for m in range(1, 13):
        x = bench[bench.index.month == m].values
        x = x[~np.isnan(x)]
        lo, hi = boot_ci(x)
        mk.append({"month": MONTHS[m - 1], "n": len(x), "mean": round(x.mean(), 2),
                   "median": round(float(np.median(x)), 2),
                   "hit%": round(100 * (x > 0).mean(), 1), "t": round(tstat(x), 2),
                   "ci95": f"[{lo:+.2f},{hi:+.2f}]"})
    mkdf = pd.DataFrame(mk)
    print(mkdf.to_string(index=False))
    naive_m = [r["month"] for r in mk
               if not (float(r["ci95"].split(",")[0][1:]) < 0 <
                       float(r["ci95"].split(",")[1][:-1]))]
    print("  naive: months whose 95% CI excludes 0:", naive_m or "NONE")

    # Even 12 tests need a snooping control: the BEST of 12 months is biased.
    bv = bench.values
    bm = bench.index.month.values
    obs_mt = np.array([abs(tstat(bv[bm == m])) for m in range(1, 13)])
    nullmax = np.empty(N_PERM)
    for b in range(N_PERM):
        sh = RNG.permutation(bv)
        nullmax[b] = np.nanmax([abs(tstat(sh[bm == m])) for m in range(1, 13)])
    pg = (np.sum(nullmax >= np.nanmax(obs_mt)) + 1) / (N_PERM + 1)
    print(f"  permutation over the 12 months: best |t| = {np.nanmax(obs_mt):.2f}, "
          f"null 95th pct = {np.percentile(nullmax,95):.2f}, global p = {pg:.4f}")
    print("  ->", "market month effect survives snooping control"
          if pg < 0.05 else "market month effect does NOT survive the snooping control")

    # ---------- sector EXCESS seasonality: the real question ----------
    #
    # SPECIFICATION. The model is  r[s,t] = alpha[s] + season[s, month(t)] + eps.
    # A sector that persistently beats the benchmark (alpha>0) would otherwise
    # show a "good month" in EVERY month, and its argmax month would be picked
    # up as seasonality when it is really just drift. Demeaning each sector's
    # own excess series isolates season[] from alpha[]. This is the primary lens;
    # the raw (non-demeaned) excess is reported alongside for reference only.
    ex_raw = sect.sub(bench, axis=0)
    ex = ex_raw.sub(ex_raw.mean(axis=0), axis=1)
    print(f"\n=== B. SECTOR EXCESS vs {blabel}, demeaned per sector ===")
    print("    (each sector's own average excess removed, so this measures the")
    print("     MONTH effect only, not the sector's persistent drift)")
    alph = ex_raw.mean(axis=0).sort_values(ascending=False)
    print(f"    sector alphas removed: max {alph.iloc[0]:+.2f}%/mo ({alph.index[0]}), "
          f"min {alph.iloc[-1]:+.2f}%/mo ({alph.index[-1]})")
    tab = cell_table(ex)
    if tab.empty:
        print("no cells"); return

    naive = (tab.sign_p < 0.05).sum()
    print(f"cells tested: {len(tab)}   naive sign-p<0.05: {naive} "
          f"(pure-noise expectation ~{0.05*len(tab):.0f})")

    tab["fdr10"] = bh_fdr(tab.sign_p.values, q=0.10)
    print(f"BH-FDR q=0.10 survivors: {int(tab.fdr10.sum())}")

    print("\n--- top 15 cells by |t| (EXCESS) ---")
    show = tab.reindex(tab.t.abs().sort_values(ascending=False).index).head(15)
    print(show[["sector", "month", "n", "mean", "median", "hit", "t",
                "ci_lo", "ci_hi", "sign_p", "worst_yr_drop", "fdr10"]]
          .round(2).to_string(index=False))

    # ---------- permutation: is the BEST cell better than noise's best? ----------
    print("\n=== C. PERMUTATION max-|t| test (data-snooping control) ===")
    obs_t, pperm, maxnull = perm_p_and_maxnull(ex)
    obs_max = np.nanmax(np.abs(obs_t))
    p_global = (np.sum(maxnull >= obs_max) + 1) / (len(maxnull) + 1)
    j, k = np.unravel_index(np.nanargmax(np.abs(obs_t)), obs_t.shape)
    print(f"  best cell observed : {ex.columns[j]} / {MONTHS[k]}  |t| = {obs_max:.2f}")
    print(f"  null max-|t|       : median {np.median(maxnull):.2f}, "
          f"95th pct {np.percentile(maxnull,95):.2f}, max {maxnull.max():.2f}")
    print(f"  GLOBAL p           : {p_global:.4f}")
    print("  ->", "REJECT the null: at least one real month effect"
          if p_global < 0.05 else
          "CANNOT reject the null: the best cell is within what noise produces")

    # ---------- out-of-sample ----------
    print("\n=== D. OUT-OF-SAMPLE: pick best month in-sample, trade it later ===")
    split = ex.index[len(ex) // 2]
    ins, oos = ex[ex.index < split], ex[ex.index >= split]
    print(f"  in-sample {ins.index.min():%Y-%m}..{ins.index.max():%Y-%m}  "
          f"oos {oos.index.min():%Y-%m}..{oos.index.max():%Y-%m}")
    # CONTROL: the picked month must be compared with that sector's OWN average
    # month in the SAME out-of-sample window. Without this baseline a period in
    # which every sector happened to beat the benchmark (e.g. broad mid/small-cap
    # outperformance) reads as "seasonality works" when nothing seasonal happened.
    rows = []
    skipped = []
    for c in ex.columns:
        vi = ins[c].dropna()
        # Sectors launched mid-sample (Oil & Gas, Healthcare, Consumer Durables
        # start 2020) have no in-sample history to pick a month from. Including
        # them would silently pick month 1 by argmax-of-all-NaN.
        means = pd.Series({m: vi[vi.index.month == m].mean() for m in range(1, 13)})
        if len(vi) < 24 or means.notna().sum() < 12:
            skipped.append(c)
            continue
        best_m = int(means.idxmax())
        vo = oos[c]
        xo = vo[vo.index.month == best_m].values
        xo = xo[~np.isnan(xo)]
        xi = vi[vi.index.month == best_m].values
        base = float(np.nanmean(vo.values))          # all months, same window
        rows.append({"sector": c, "best_month_IS": MONTHS[best_m - 1],
                     "IS_mean": round(np.nanmean(xi), 2),
                     "OOS_n": len(xo),
                     "OOS_mean": round(xo.mean(), 2) if len(xo) else np.nan,
                     "OOS_base": round(base, 2),
                     "OOS_lift": round(xo.mean() - base, 2) if len(xo) else np.nan,
                     "OOS_hit%": round(100 * (xo > 0).mean(), 1) if len(xo) else np.nan})
    od = pd.DataFrame(rows)
    if skipped:
        print(f"  skipped (insufficient in-sample history): {', '.join(skipped)}")
    print(od.to_string(index=False))
    good = od.OOS_mean.dropna()
    lift = od.OOS_lift.dropna()
    print(f"\n  IS mean of picked cells   : {od.IS_mean.mean():+.2f}%/month")
    print(f"  OOS mean of same cells    : {good.mean():+.2f}%/month  "
          f"({(good>0).sum()}/{len(good)} sectors positive)")
    print(f"  OOS baseline (all months) : {od.OOS_base.mean():+.2f}%/month  <-- the control")
    print(f"  OOS LIFT over baseline    : {lift.mean():+.2f}%/month  "
          f"({(lift>0).sum()}/{len(lift)} sectors positive)")
    tl = tstat(lift.values)
    print(f"  lift t across sectors     : {tl:+.2f}   (cross-sector, not independent)")
    print("  ->", "seasonal lift survives OOS" if lift.mean() > 0 and tl > 2 else
          "NO seasonal lift once the baseline is removed")

    # ---------- jackknife ----------
    print("\n=== E. JACKKNIFE: does any top cell survive dropping its best year? ===")
    top = tab.reindex(tab.t.abs().sort_values(ascending=False).index).head(10)
    jk = top[["sector", "month", "n", "mean", "worst_yr_drop"]].copy()
    jk["survives_sign"] = np.sign(jk["mean"]) == np.sign(jk["worst_yr_drop"])
    jk["retained%"] = (jk.worst_yr_drop / jk["mean"] * 100).round(0)
    print(jk.round(2).to_string(index=False))

    # ---------- F. tradable walk-forward rotation, net of cost ----------
    print("\n=== F. WALK-FORWARD seasonal rotation, net of cost ===")
    print("  rule: at each month t, rank sectors by their mean excess in THAT")
    print("  calendar month using ONLY data before t (expanding, causal);")
    print("  hold the top k. Requires >=3 prior observations of that month.")
    idx = ex_raw.index
    for k in (1, 3, 5):
        gross, turn, dates = [], [], []
        prev = set()
        for i, ts in enumerate(idx):
            hist = ex_raw.iloc[:i]
            if len(hist) < 24:
                continue
            hm = hist[hist.index.month == ts.month]
            if len(hm) < 3:
                continue
            score = hm.mean(axis=0).dropna()
            avail = ex_raw.iloc[i].dropna().index
            score = score[score.index.isin(avail)]
            if len(score) < k:
                continue
            pick = set(score.nlargest(k).index)
            gross.append(float(ex_raw.iloc[i][list(pick)].mean()))
            turn.append(len(pick - prev) / k)
            prev = pick
            dates.append(ts)
        if not gross:
            print(f"  k={k}: insufficient history"); continue
        g = np.array(gross)
        tr = np.array(turn)
        # 25 bps per side; a replaced name costs a sell + a buy
        cost = tr * 2 * 0.25
        net = g - cost
        print(f"  k={k}: n={len(g):3d} months | gross {g.mean():+.3f}%/mo "
              f"t={tstat(g):+.2f} | turnover {tr.mean()*100:.0f}% "
              f"| cost {cost.mean():.3f}% | NET {net.mean():+.3f}%/mo "
              f"t={tstat(net):+.2f} | net ann {net.mean()*12:+.2f}%")

    if a.full:
        print("\n=== FULL CELL TABLE (excess) ===")
        piv = tab.pivot(index="sector", columns="m", values="mean")
        piv.columns = [MONTHS[c - 1] for c in piv.columns]
        print(piv.round(2).to_string())
        print("\n--- hit% ---")
        piv2 = tab.pivot(index="sector", columns="m", values="hit")
        piv2.columns = [MONTHS[c - 1] for c in piv2.columns]
        print(piv2.round(0).to_string())


if __name__ == "__main__":
    main()
