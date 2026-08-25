"""
Which stocks to show under an OVERWEIGHT sector — measured, not assumed.

WHY THIS CANNOT BE GUESSED
    The obvious build ("top 4 by return / turnover") is measured BACKWARDS at the
    stock level. scripts/audit_stock_pick_in_clock.py (1.19M stock-days) found, on
    forward-10d return EXCESS OVER OWN SECTOR:

        dacc      (deliv 5d / own 100d normal)   IC +0.0323  t +9.56   <- only +
        rel_mom   (momentum vs sector)           IC -0.0226  t -4.79
        dvshare   (share of sector delivery)     IC -0.0351  t -4.79
        turnsurge (turnover vs own normal)       IC -0.0164  t -3.52
        contrib   (drove the sector move)        IC -0.0212  t -4.85

    So a "top 4 by strength" list would be ranked by four signals that are all
    anti-predictive, and would omit the one that works.

FOUR THINGS THAT PRIOR RESULT DOES NOT SETTLE, ALL TESTED HERE
    1. PANEL. It predates the same-day-turnover-FILTER fix (a stock entered the
       universe BECAUSE it moved that day: +0.223%/day vs +0.072% lagged). Every
       stock-level number has to be re-measured on the corrected panel.
    2. CONDITIONING. It pooled across ALL sectors. This feature only ever renders
       under an OVERWEIGHT sector, and the same audit found the leader/laggard
       ordering FLIPS there (in a top-30% sector the leader beat the laggard by
       +0.249pp, t -2.70 for laggard-minus-leader). Conditional is what matters.
    3. HORIZON. Measured at 10 days only. The tab now offers 10-60. A 5-day
       delivery window predicting a 60-day forward return is not implied.
    4. THIN NAMES. Extreme dacc favours small names (one Rs48 Cr name against a
       Rs167 Cr sector median). Does the edge survive a tradeable size floor?

METHOD
    Target = stock forward h-day return MINUS its own sector's forward h-day
    return, so the sector call is stripped out and this is pure selection.
    Universe = stocks in sectors that were OVERWEIGHT on that date, per the live
    rank + persistence gate. Panel mirrors the fixed engine: lagged liquidity
    filter, corporate actions dropped, returns winsorized. Newey-West t at lag=h.

Usage:  python scripts/audit_within_sector_pick.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_tilt_robustness import Sig, nw_t, DB, WINSOR, MIN_TURN

HORIZONS = {"1-2 wk": 10, "3-4 wk": 20, "5-6 wk": 30,
            "7-8 wk": 40, "9-10 wk": 50, "11-12 wk": 60}
DACC_FLOW, DACC_BASE = 5, 100
OW_RANK = 0.75
TOPK = 4
ERAS = (("2018-20", 2018, 2020), ("2021-23", 2021, 2023), ("2024-26", 2024, 2026))


def load_stock_panel(con, floor: float = MIN_TURN) -> pd.DataFrame:
    """Per-stock daily panel, same universe rule as the fixed sector engine."""
    q = f"""
    with base as (
        select b.trade_date, b.symbol, s.sector, b.turnover_lacs, b.deliv_per,
               (b.close_price - b.prev_close)/nullif(b.prev_close,0)*100 as raw_r,
               greatest(least((b.close_price - b.prev_close)
                        /nullif(b.prev_close,0)*100, {WINSOR}), -{WINSOR}) as r,
               lag(b.turnover_lacs) over (partition by b.symbol
                                          order by b.trade_date) as w_lag
        from daily_data b join v_sector_master s on b.symbol = s.symbol
        where b.series in ('EQ','SM','ST') and s.sector not in ('ETF','Others')
    )
    select trade_date, symbol, sector, r, deliv_per, turnover_lacs
    from base
    where w_lag >= {floor} and abs(raw_r) < 40
    """
    d = con.execute(q).df()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    return d


def build_features(sp: pd.DataFrame, sig: Sig, h: int):
    """Stock features + forward EXCESS-OVER-OWN-SECTOR target, all causal."""
    r = sp.pivot_table("r", "trade_date", "symbol")
    dp = sp.pivot_table("deliv_per", "trade_date", "symbol")
    to = sp.pivot_table("turnover_lacs", "trade_date", "symbol")
    sec = sp.drop_duplicates("symbol").set_index("symbol")["sector"]
    idx = r.index

    lg = np.log1p(r / 100.0)
    # trailing h-day stock momentum (causal, includes today)
    mom = np.expm1(lg.rolling(h).sum()) * 100.0
    # forward h-day stock return, strictly after today
    fwd = np.expm1(lg.shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]) * 100.0

    # sector series aligned onto the stock columns
    slg = np.log1p(sig.ret.reindex(idx) / 100.0)
    smom = np.expm1(slg.rolling(h).sum()) * 100.0
    sfwd = np.expm1(slg.shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]) * 100.0
    cmap = sec.reindex(r.columns)
    smom_b = smom.reindex(columns=cmap.values); smom_b.columns = r.columns
    sfwd_b = sfwd.reindex(columns=cmap.values); sfwd_b.columns = r.columns

    feats = {
        # the stock's recent delivery vs ITS OWN long-run normal (not a percentile)
        "dacc": dp.rolling(DACC_FLOW).mean() / dp.shift(1).rolling(DACC_BASE).mean(),
        "rel_mom": mom - smom_b,
        "turnsurge": to.rolling(DACC_FLOW).mean() / to.shift(1).rolling(DACC_BASE).mean(),
        "deliv_cr": (dp / 100.0 * to).rolling(DACC_FLOW).mean() / 100.0,
        "abs_mom": mom,
    }
    feats["dvshare"] = feats["deliv_cr"].div(
        feats["deliv_cr"].T.groupby(cmap.values).transform("sum").T.replace(0, np.nan))
    target = fwd - sfwd_b
    return feats, target, cmap


def ow_stock_mask(sig: Sig, h: int, cmap: pd.Series, idx) -> pd.DataFrame:
    """True where the stock's sector was OVERWEIGHT that day (rank + persistence)."""
    rank = sig.score(h).rank(axis=1, pct=True)
    ow = (rank >= OW_RANK) & ~(sig.persistence(h) < 0).fillna(False)
    ow = ow.reindex(idx).fillna(False)
    out = ow.reindex(columns=cmap.values)
    out.columns = cmap.index
    return out


def ic_stats(f: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame, h: int):
    """Per-date cross-sectional rank IC inside the masked universe + NW t."""
    fm, ym = f.where(mask), y.where(mask)
    ok = fm.notna() & ym.notna()
    n_ok = ok.sum(axis=1)
    use = n_ok >= 8
    if use.sum() < 30:
        return np.nan, np.nan, 0
    ics = fm[use].rank(axis=1).corrwith(ym[use].rank(axis=1), axis=1)
    ics = ics.dropna()
    return float(ics.mean()), nw_t(ics.values, h), int(ok.sum().sum())


def topk_excess(f: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame,
                h: int, k: int = TOPK, bottom: bool = False):
    """Mean forward excess of the top-k (or bottom-k) names per sector-day."""
    fm, ym = f.where(mask), y.where(mask)
    vals = []
    per_date = []
    for dt_ in fm.index:
        row, yy = fm.loc[dt_].dropna(), ym.loc[dt_]
        row = row[yy.reindex(row.index).notna()]
        if len(row) < k + 2:
            continue
        pick = (row.nsmallest(k) if bottom else row.nlargest(k)).index
        v = yy.reindex(pick).dropna()
        if len(v):
            vals.extend(v.tolist()); per_date.append(float(v.mean()))
    if len(per_date) < 30:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), nw_t(np.array(per_date), h), len(vals)


def main():
    con = duckdb.connect(DB, read_only=True)
    sig = Sig(con)
    sp = load_stock_panel(con)
    con.close()
    print(f"stock panel: {sp.symbol.nunique():,} symbols x {sp.trade_date.nunique():,} "
          f"sessions ({sp.trade_date.min():%Y-%m-%d} -> {sp.trade_date.max():%Y-%m-%d}), "
          f"{len(sp):,} stock-days")

    for lbl, h in HORIZONS.items():
        feats, y, cmap = build_features(sp, sig, h)
        mask = ow_stock_mask(sig, h, cmap, y.index)
        print("\n" + "=" * 84)
        print(f"HORIZON {lbl} (h={h}) — target = stock fwd{h}d MINUS its own sector's fwd{h}d")
        print(f"universe = stocks whose sector was OVERWEIGHT that day "
              f"({int(mask.sum().sum()):,} stock-days)")
        print("=" * 84)
        print(f"  {'feature':<12}{'IC':>9}{'NW-t':>8}{'n':>10}   "
              f"{'top4 exc':>10}{'t':>7}   {'bot4 exc':>10}{'t':>7}   spread")
        for name in ("dacc", "rel_mom", "abs_mom", "turnsurge", "dvshare"):
            f = feats[name]
            ic, t, n = ic_stats(f, y, mask, h)
            tp, tpt, _ = topk_excess(f, y, mask, h)
            bt, btt, _ = topk_excess(f, y, mask, h, bottom=True)
            sp_ = tp - bt if np.isfinite(tp) and np.isfinite(bt) else np.nan
            print(f"  {name:<12}{ic:>9.4f}{t:>8.2f}{n:>10,}   "
                  f"{tp:>10.3f}{tpt:>7.2f}   {bt:>10.3f}{btt:>7.2f}   {sp_:>+6.3f}pp")

        # quintile ladder for the winner
        f = feats["dacc"]
        fm, ym = f.where(mask), y.where(mask)
        q = fm.rank(axis=1, pct=True)
        print(f"\n  dacc quintile ladder (mean fwd{h}d excess vs own sector):")
        cells = []
        for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
            m = (q > lo) & (q <= hi)
            v = ym.values[(m & ym.notna()).values]
            cells.append(float(v.mean()) if len(v) > 100 else np.nan)
        print("    " + " · ".join(f"Q{i+1} {c:+.3f}" for i, c in enumerate(cells))
              + f"   Q5-Q1 {cells[-1] - cells[0]:+.3f}pp")

        # size guard: does it survive a tradeable delivery floor?
        print("  dacc top-4 excess by delivery-size floor:")
        for floor_cr in (0, 1, 5, 25):
            big = feats["deliv_cr"] >= floor_cr
            tp, tpt, n = topk_excess(f, y, mask & big, h)
            print(f"    >= Rs {floor_cr:>2} Cr delivery: {tp:>+7.3f}pp (t {tpt:+.2f}, n {n:,})")

        # era stability
        print("  dacc top-4 excess by era:", end="")
        for nm, y0, y1 in ERAS:
            sel = (y.index.year >= y0) & (y.index.year <= y1)
            tp, tpt, _ = topk_excess(f.loc[sel], y.loc[sel], mask.loc[sel], h)
            print(f"  {nm} {tp:+.3f} (t {tpt:+.1f})", end="")
        print()

        # the conditional the prior audit flagged: leader vs laggard INSIDE a strong sector
        rm = feats["rel_mom"]
        lead, lt, _ = topk_excess(rm, y, mask, h)
        lag, gt, _ = topk_excess(rm, y, mask, h, bottom=True)
        print(f"  leader-vs-laggard inside OVERWEIGHT sectors: "
              f"leader {lead:+.3f} (t {lt:+.2f}) · laggard {lag:+.3f} (t {gt:+.2f}) "
              f"· laggard-minus-leader {lag - lead:+.3f}pp")


if __name__ == "__main__":
    main()
