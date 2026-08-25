"""
How long does a sector STAY on the Forward-Tilt buy list, and does it matter?

THE QUESTION THIS ANSWERS
    The tab recomputes the OVERWEIGHT list every day but was validated on a
    NON-OVERLAPPING rebalance (25.2x/yr at 1-2wk). So a user who buys on Monday
    and sees the sector drop off on Tuesday has no rule to follow. Three things
    must be measured before a "freshness / day N" badge can carry any claim:

      1. CHURN.    How often does the list actually change, and what is the
                   distribution of tenure? If mean tenure is ~2 sessions the
                   badge is noise; if it is ~8 the badge is informative.
      2. TENURE.   Does forward excess depend on how long a sector has been on
                   the list? (day-1 fresh entry vs day-10 established leader)
      3. EXIT.     A sector leaves the list. Is selling then better than holding
                   to the horizon the call was actually making -- NET OF COST?

METHOD
    - Panel reproduces the live tilt SQL exactly (EQ/SM/ST, ex-ETF/Others,
      +/-25% winsorized daily return, LAGGED-turnover weights, w_lag NOT NULL).
    - score = 0.60*rank(rs_L) + 0.25*rank(rs_S) + 0.15*rank(dv5d) with
      rs lookbacks scaled to the horizon (L=h, S=h//2), matching get_forward_tilt.
    - OVERWEIGHT = rank(score) >= 0.75, then the PERSISTENCE GATE demotes any
      sector whose trailing mean forward excess (vs the cross-sectional MEDIAN,
      as the live code does) is < 0 over a ~620-calendar-day window, min 30 obs.
    - Two OW definitions are tracked separately, because they churn differently:
        ow_raw   rank only          (what the ranking says)
        ow_net   rank + persistence (what the tab actually prints)
      THIN and WATCH demotions are NOT modelled: WATCH requires rank <= 0.35 so
      it can never touch an OVERWEIGHT, and THIN (<5 liquid names) is rare for a
      real sector at the 1 Cr floor. Both are stated, not silently dropped.
    - Forward excess = sector compound return over the next k sessions minus the
      equal-weight all-sector basket, strictly after the signal date.
    - Inference is Newey-West at lag = the forward window: overlapping windows
      make adjacent dates dependent and a raw t is badly inflated.
    - Multiplicity: tenure is bucketed 6 ways, so the best bucket is a max over
      6 cells. A permutation test on max-|t| is run against shuffled tenure.

Usage:  python scripts/study_tilt_tenure.py
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
MIN_TURNOVER_LACS = 100.0          # the dashboard's default 1.00 Cr slider
W_RS2, W_RS1, W_DV5 = 0.60, 0.25, 0.15
DV_BASE, DV_FLOW = 100, 5
OW_RANK = 0.75
PERS_LOOKBACK_SESS = 425           # ~620 calendar days
PERS_MIN_OBS = 30
TOPK = 4                           # the basket size the published record used
COST_BPS_SIDE = 25.0
HORIZONS = {"1-2 wk": 10, "3-4 wk": 20, "5-6 wk": 30,
            "7-8 wk": 40, "9-10 wk": 50, "11-12 wk": 60}
ERAS = [("2018-21", "2018-01-01", "2021-12-31"),
        ("2022-24", "2022-01-01", "2024-12-31"),
        ("2025-26", "2025-01-01", "2026-12-31")]
RNG = np.random.default_rng(20260816)


# ── inference helpers ────────────────────────────────────────────────────────
def nw_t(x, lag: int) -> float:
    """Newey-West t for the mean of an autocorrelated series."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    d = x - x.mean()
    var = (d @ d) / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lag + 1.0)) * ((d[L:] @ d[:-L]) / n)
    if var <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(var / n))


def load_panel(con) -> pd.DataFrame:
    q = f"""
    with base as (
        select b.trade_date, b.symbol, b.turnover_lacs, b.deliv_per, s.sector,
               (b.close_price - b.prev_close)/nullif(b.prev_close,0)*100 as raw_r,
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
    -- mirrors the FIXED _load_sector_panel: lagged liquidity filter, corporate
    -- actions dropped rather than winsorized.
    where w_lag >= {MIN_TURNOVER_LACS} and abs(raw_r) < 40
    group by sector, trade_date
    order by sector, trade_date
    """
    d = con.execute(q).df()
    d["trade_date"] = pd.to_datetime(d["trade_date"])
    return d


def build(h: int, lg: pd.DataFrame, dv: pd.DataFrame, ngr: pd.Series):
    """Reproduce the live tilt for horizon h: score, raw OW, persistence-gated OW."""
    L, S = h, max(2, h // 2)

    def trail(n):
        return (np.exp(lg.rolling(n).sum()) - 1.0) * 100.0

    def ntrail(n):
        return (np.exp(ngr.rolling(n).sum()) - 1.0) * 100.0

    rsL = trail(L).sub(ntrail(L).reindex(lg.index), axis=0)
    rsS = trail(S).sub(ntrail(S).reindex(lg.index), axis=0)
    dv5d = dv.rolling(DV_FLOW).mean() / dv.shift(1).rolling(DV_BASE).mean()

    rk = lambda d: d.rank(axis=1, pct=True)
    score = W_RS2 * rk(rsL) + W_RS1 * rk(rsS) + W_DV5 * rk(dv5d)
    rank = rk(score)
    ow_raw = rank >= OW_RANK

    # forward h-session excess vs the cross-sectional MEDIAN (the live gate's
    # definition), then the trailing-window mean of it, lagged h so no realized
    # window overlaps the signal date. This is the persistence gate, causally.
    fwd_h = fwd(lg, h)
    edge_med = fwd_h.sub(fwd_h.median(axis=1), axis=0)
    lagged = edge_med.shift(h)                       # only windows that have CLOSED
    pers = lagged.rolling(PERS_LOOKBACK_SESS, min_periods=PERS_MIN_OBS).mean()
    revert = pers < 0                                # NaN -> False, as the live code
    ow_net = ow_raw & ~revert.fillna(False)
    return score, rank, ow_raw, ow_net, pers


def fwd(lg: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compound % return over the k sessions STRICTLY AFTER each date."""
    f = lg.shift(-1).iloc[::-1].rolling(k, min_periods=k).sum().iloc[::-1]
    return (np.exp(f) - 1.0) * 100.0


def fwd_excess(lg: pd.DataFrame, k: int) -> pd.DataFrame:
    f = fwd(lg, k)
    return f.sub(f.mean(axis=1), axis=0)


def run_length(mask: pd.DataFrame) -> pd.DataFrame:
    """Vectorised consecutive-True counter, column-wise."""
    out = pd.DataFrame(0, index=mask.index, columns=mask.columns, dtype=int)
    prev = np.zeros(mask.shape[1], dtype=int)
    vals = mask.values
    for i in range(len(mask)):
        prev = np.where(vals[i], prev + 1, 0)
        out.iloc[i] = prev
    return out


# ── the four studies ─────────────────────────────────────────────────────────
def _runs(ten: pd.DataFrame) -> np.ndarray:
    """Completed run lengths: a run ends on the last session it was True."""
    ended = (ten > 0) & (ten.shift(-1).fillna(0).astype(int) == 0)
    return ten.values[ended.values]


def study_churn(ow_raw, ow_net, ten_raw, ten_net, pers, h, lbl):
    n_ow = ow_net.sum(axis=1)
    prev = ow_net.shift(1).fillna(False).astype(bool)
    entries = (ow_net & ~prev).sum(axis=1)
    exits = (~ow_net & prev).sum(axis=1)
    exit_mask = (~ow_net) & prev
    gate_exit = int((exit_mask & ow_raw).sum().sum())      # rank held, gate demoted
    rank_exit = int((exit_mask & ~ow_raw).sum().sum())     # rank actually fell
    print(f"\n  [{lbl}]  OW slots/day  raw {ow_raw.sum(axis=1).mean():.1f} · "
          f"net {n_ow.mean():.1f} (min {n_ow.min():.0f}, max {n_ow.max():.0f})")
    print(f"           list changes: {entries.mean():.2f} in / {exits.mean():.2f} out per session; "
          f"{(entries + exits > 0).mean() * 100:.0f}% of sessions see at least one change")
    for nm, ten in (("rank-only", ten_raw), ("as printed", ten_net)):
        r = _runs(ten)
        if len(r) == 0:
            continue
        q = np.percentile(r, [50, 75, 90])
        print(f"           tenure {nm:>10}: mean {r.mean():5.1f} · median {q[0]:.0f} · "
              f"p75 {q[1]:.0f} · p90 {q[2]:.0f} · max {r.max():.0f} sessions "
              f"(n={len(r)} runs) · {(r == 1).mean() * 100:.0f}% last ONE session")
    tot = gate_exit + rank_exit
    if tot:
        print(f"           exit cause: rank fell {rank_exit / tot * 100:.0f}% ({rank_exit}) · "
              f"persistence gate {gate_exit / tot * 100:.0f}% ({gate_exit})")
    # how stable is the gate itself? a sign flip demotes/promotes with momentum unchanged
    sgn = (pers < 0)
    valid = pers.notna() & pers.shift(1).notna()
    flips = int((sgn.ne(sgn.shift(1)) & valid).sum().sum())
    sec_days = int(valid.sum().sum())
    print(f"           persistence-gate sign flips: {flips} in {sec_days} sector-days "
          f"({flips / max(sec_days, 1) * 100:.3f}%) — "
          f"{'STABLE, not a churn source' if flips / max(sec_days, 1) < 0.005 else 'a real churn source'}")


def study_tenure_buckets(ow_net, ten_net, lg, h, lbl):
    ex = fwd_excess(lg, h)
    base_v = ex.values[(ow_net & ex.notna()).values]
    print(f"\n  [{lbl}]  forward {h}-session excess vs equal-weight basket, by tenure")
    print(f"           ALL overweight sector-days: {base_v.mean():+.3f}pp  "
          f"(n={len(base_v)}, NW-t {nw_t(pooled_daily(ex.where(ow_net)), h):+.2f})")
    buckets = [(1, 1), (2, 3), (4, 5), (6, 10), (11, 20), (21, 10_000)]
    rows = []
    for lo, hi in buckets:
        m = ow_net & (ten_net >= lo) & (ten_net <= hi)
        v = ex.values[(m & ex.notna()).values]
        nm = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 1000 else f"{lo}+")
        if len(v) < 30:
            rows.append((nm, np.nan, 0, np.nan)); continue
        rows.append((nm, float(v.mean()), len(v), nw_t(pooled_daily(ex.where(m)), h)))
    base = pd.Series(base_v)
    for name, mu, n, t in rows:
        if n == 0:
            print(f"             day {name:<6} — insufficient")
            continue
        print(f"             day {name:<6} {mu:+7.3f}pp   n={n:<6d}  NW-t {t:+5.2f}"
              f"   excess-over-OW-base {mu - base.mean():+.3f}pp")
    # multiplicity: is the best bucket better than the best of a shuffled labelling?
    ts = np.array([r[3] for r in rows if np.isfinite(r[3])])
    if len(ts) >= 3:
        obs = np.nanmax(np.abs(ts))
        null = []
        flat = ten_net.where(ow_net)
        for _ in range(200):
            perm = pd.DataFrame(RNG.permutation(flat.values.ravel()).reshape(flat.shape),
                                index=flat.index, columns=flat.columns)
            tt = []
            for lo, hi in buckets:
                m = ow_net & (perm >= lo) & (perm <= hi)
                if m.sum().sum() < 30:
                    continue
                tt.append(nw_t(pooled_daily(ex.where(m)), h))
            if tt:
                null.append(np.nanmax(np.abs(tt)))
        if null:
            p = float(np.mean(np.asarray(null) >= obs))
            print(f"           multiplicity: best |t| = {obs:.2f}; permutation p = {p:.3f} "
                  f"({'PASSES' if p < 0.05 else 'FAILS — no tenure effect'})")


def pooled_daily(df: pd.DataFrame) -> np.ndarray:
    """Per-date cross-sectional mean → one observation per date (kills within-date
    cross-correlation before the Newey-West lag handles the time dependence)."""
    return df.mean(axis=1).values


def study_exit(ow_net, ten_net, lg, h, lbl):
    """A sector drops off the list. Sell, or hold to the horizon it promised?"""
    exit_day = ((~ow_net) & ow_net.shift(1).fillna(False).astype(bool))
    print(f"\n  [{lbl}]  EXIT EVENT — sector left the buy list on day t "
          f"(n={int(exit_day.sum().sum())} events)")
    print("             k    left-list    still-on-list      diff      NW-t   "
          "round-trip cost to beat")
    cost = 2 * COST_BPS_SIDE / 100.0
    for k in sorted({1, 2, 3, 5, 10, h}):
        ex = fwd_excess(lg, k)
        va = ex.values[(exit_day & ex.notna()).values]
        vb = ex.values[(ow_net & ex.notna()).values]
        if len(va) < 30:
            continue
        # PAIRED daily difference: on each date, mean excess of the sectors that
        # left minus mean excess of the sectors that stayed. One obs per date, so
        # cross-sectional correlation cannot inflate n; NW handles the overlap.
        da = ex.where(exit_day).mean(axis=1)
        db = ex.where(ow_net).mean(axis=1)
        d = (da - db).dropna()
        t = nw_t(d.values, k)
        mu_a, mu_b = float(va.mean()), float(vb.mean())
        verdict = "SELL PAYS" if (mu_b - mu_a) > cost else "hold"
        print(f"           {k:>4}  {mu_a:+8.3f}pp   {mu_b:+8.3f}pp   "
              f"{mu_a - mu_b:+8.3f}pp   {t:+5.2f}   need >{cost:.2f}pp → {verdict}")


def study_cost(score, lg, h, lbl):
    """Hold-to-horizon vs exit-on-drop, both long-only top-K, net of cost."""
    ret = (np.exp(lg) - 1.0) * 100.0
    dates = lg.index
    basket = ret.mean(axis=1)

    def run(rebalance_every: int):
        w = pd.Series(0.0, index=lg.columns)
        eq, turn, nreb, reb_turn = [], 0.0, 0, []
        for i, d in enumerate(dates):
            if i and w.abs().sum() > 0:
                eq.append(float((w * ret.loc[d].fillna(0)).sum()))
            else:
                eq.append(0.0)
            if i % rebalance_every == 0:
                s = score.loc[d].dropna()
                if len(s) >= TOPK:
                    tgt = pd.Series(0.0, index=lg.columns)
                    tgt[s.nlargest(TOPK).index] = 1.0 / TOPK
                    tv = float((tgt - w).abs().sum())
                    turn += tv
                    reb_turn.append((d, tv))
                    nreb += 1
                    w = tgt
        r = pd.Series(eq, index=dates)
        yrs = (dates[-1] - dates[0]).days / 365.25
        # cost is charged on the rebalance day itself, so the NET series compounds
        # correctly instead of subtracting an arithmetic total at the end.
        cost_d = pd.Series(0.0, index=dates)
        for d, tv in reb_turn:
            cost_d.loc[d] += tv * COST_BPS_SIDE / 100.0
        cagr = lambda s: (np.exp(np.log1p(s / 100.0).sum() / yrs) - 1.0) * 100.0
        g, n, b = cagr(r), cagr(r - cost_d), cagr(basket)
        return dict(gross_yr=g, net_yr=n, excess_yr=n - b, bench_yr=b,
                    cost_drag=g - n, reb_yr=nreb / yrs, turn_yr=turn / yrs)

    print(f"\n  [{lbl}]  TURNOVER COST — long-only top-{TOPK}, {COST_BPS_SIDE:.0f}bps/side, CAGR")
    print("             rebalance                       reb/yr  gross  net   cost drag  "
          "excess vs basket")
    for every, nm in ((h, f"every {h} sessions (as validated)"), (1, "DAILY (act on flips)")):
        r = run(every)
        print(f"           {nm:<32}{r['reb_yr']:6.1f} {r['gross_yr']:+6.2f} {r['net_yr']:+6.2f}"
              f"   {r['cost_drag']:6.2f}      {r['excess_yr']:+6.2f}")
    print(f"           equal-weight sector basket CAGR: {run(h)['bench_yr']:+.2f}%/yr")


def main():
    con = duckdb.connect(DB, read_only=True)
    panel = load_panel(con)
    nf = con.sql("select trade_date, close_val from index_data "
                 "where index_name='Nifty 50' order by trade_date").df()
    con.close()
    nf["trade_date"] = pd.to_datetime(nf["trade_date"])
    nifty = nf.set_index("trade_date")["close_val"].sort_index()

    ret = panel.pivot(index="trade_date", columns="sector", values="wtd_ret_pct").sort_index()
    dv = panel.pivot(index="trade_date", columns="sector", values="daily_dv_cr").sort_index()
    ret = ret.dropna(how="all")
    dv = dv.reindex(ret.index)
    lg = np.log(1.0 + ret / 100.0)
    ngr = np.log(1.0 + nifty.pct_change().reindex(ret.index).fillna(0.0))

    print(f"panel: {ret.shape[1]} sectors x {len(ret)} sessions "
          f"({ret.index.min():%Y-%m-%d} -> {ret.index.max():%Y-%m-%d})")
    print(f"OW quota by construction: rank>=0.75 of {ret.shape[1]} sectors "
          f"= top {int(np.ceil(0.25 * ret.shape[1]))} slots, every single day")

    for lbl, h in HORIZONS.items():
        score, rank, ow_raw, ow_net, pers = build(h, lg, dv, ngr)
        ten_raw, ten_net = run_length(ow_raw), run_length(ow_net)
        print("\n" + "=" * 78)
        print(f"HORIZON {lbl}  (h={h}, rs lookbacks {h}/{max(2, h // 2)})")
        print("=" * 78)
        study_churn(ow_raw, ow_net, ten_raw, ten_net, pers, h, lbl)
        study_tenure_buckets(ow_net, ten_net, lg, h, lbl)
        study_exit(ow_net, ten_net, lg, h, lbl)
        study_cost(score, lg, h, lbl)


if __name__ == "__main__":
    main()
