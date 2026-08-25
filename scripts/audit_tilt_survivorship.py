"""
Quantify the survivorship / selection bias in the sector panel.

THE PROBLEM
    Every sector analytic joins daily_data to v_sector_master with an INNER JOIN.
    v_sector_master carries ~1,045 symbols against ~2,800 liquid names per year,
    and it has NO as-of column -- it is TODAY's membership applied to 8.6 years of
    history. Anything delisted, merged or renamed since 2018 is absent from the
    past entirely, so every historical sector return is computed on companies that
    survived to 2026.

    scripts/audit_tilt_robustness.py showed the only surviving tilt horizons are
    7-12wk, and that essentially all of their edge sits in 2022-24. If that edge is
    survivorship rather than signal, it should DECAY as the amount of future
    deletion shrinks -- i.e. it should be largest in the oldest years and vanish in
    the newest. That is a directly testable prediction and it is test D below.

TESTS
    A  Is v_sector_master a today-only snapshot? (last-trade-date of its members)
    B  How much of the market does the INNER JOIN discard, by count and by turnover?
    C  Selection premium: mapped-universe basket vs FULL liquid-universe basket.
       daily_data holds the dead names even though the master does not, so the full
       universe is the honest (near-)point-in-time benchmark.
    D  Tilt edge by YEAR against years-of-future-deletion-exposure. A real signal is
       flat in that variable; a survivorship artifact slopes down toward the present.
    E  Per-sector exposure: which sectors' constituent counts have grown most, i.e.
       whose history is thinnest and most survivor-selected.

Usage:  python scripts/audit_tilt_survivorship.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_tilt_robustness import Sig, backtest, ann, HORIZONS, DB, MIN_TURN, WINSOR

pd.set_option("display.width", 200)


def A_snapshot(con):
    print("=" * 78); print("A. IS v_sector_master A TODAY-ONLY SNAPSHOT?"); print("=" * 78)
    d = con.sql(f"""
        select m.symbol, max(b.trade_date) last_seen
        from v_sector_master m
        left join daily_data b on b.symbol = m.symbol
             and b.series in ('EQ','SM','ST')
        group by 1
    """).df()
    d["last_seen"] = pd.to_datetime(d["last_seen"])
    latest = d["last_seen"].max()
    dead = d[d["last_seen"] < latest - pd.Timedelta(days=60)]
    print(f"  members: {len(d)};  archive ends {latest:%Y-%m-%d}")
    print(f"  members whose last trade is >60d before the archive end (i.e. DEAD): "
          f"{len(dead)} ({len(dead) / len(d) * 100:.1f}%)")
    if len(dead) <= 12:
        for _, r in dead.iterrows():
            print(f"      {r['symbol']:<16} last {r['last_seen']:%Y-%m-%d}")
    print("  -> a genuine point-in-time master would carry HUNDREDS of dead names "
          "over 8.6 years.")
    print("     Near-zero dead names proves the mapping is a present-day snapshot.")


def B_discard(con):
    print("\n" + "=" * 78)
    print("B. WHAT THE INNER JOIN DISCARDS — by name count AND by turnover")
    print("=" * 78)
    d = con.sql(f"""
        select cast(year(b.trade_date) as int) yr,
               count(distinct b.symbol) syms,
               count(distinct case when m.symbol is null then b.symbol end) syms_drop,
               sum(b.turnover_lacs) turn,
               sum(case when m.symbol is null then b.turnover_lacs else 0 end) turn_drop
        from daily_data b
        left join v_sector_master m on b.symbol = m.symbol
        where b.series in ('EQ','SM','ST') and b.turnover_lacs >= {MIN_TURN}
          and b.trade_date >= '2018-01-01'
        group by 1 order by 1
    """).df()
    d["%names_dropped"] = d["syms_drop"] / d["syms"] * 100
    d["%turnover_dropped"] = d["turn_drop"] / d["turn"] * 100
    print(d[["yr", "syms", "syms_drop", "%names_dropped", "%turnover_dropped"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("  -> if %turnover_dropped is small the discarded names are minnows and the")
    print("     bias is bounded; if it is large the panel is not the market.")


def C_selection(con, sig):
    print("\n" + "=" * 78)
    print("C. SELECTION PREMIUM — mapped universe vs FULL liquid universe")
    print("=" * 78)
    # full liquid universe, SAME construction (lagged-turnover weights, winsorized)
    full = con.sql(f"""
        with base as (
            select b.trade_date, b.symbol, b.turnover_lacs,
                   greatest(least((b.close_price-b.prev_close)/nullif(b.prev_close,0)*100,
                            {WINSOR}), -{WINSOR}) r,
                   lag(b.turnover_lacs) over (partition by b.symbol order by b.trade_date) w_lag
            from daily_data b where b.series in ('EQ','SM','ST')
        )
        select trade_date,
               sum(w_lag*r)/nullif(sum(case when r is not null then w_lag end),0) ret
        from base where turnover_lacs >= {MIN_TURN} and w_lag is not null
        group by 1 order by 1
    """).df()
    full["trade_date"] = pd.to_datetime(full["trade_date"])
    fs = full.set_index("trade_date")["ret"].reindex(sig.ret.index)
    mapped = sig.ret.mean(axis=1)
    yrs = (sig.ret.index[-1] - sig.ret.index[0]).days / 365.25
    cagr = lambda s: (np.exp(np.log1p(s.fillna(0) / 100.0).sum() / yrs) - 1.0) * 100.0
    cm, cf, cn = cagr(mapped), cagr(fs), cagr(sig.nret)
    print(f"  mapped 24-sector equal-weight basket : {cm:+7.2f}%/yr   <- what the tab benchmarks against")
    print(f"  FULL liquid universe (incl dead names): {cf:+7.2f}%/yr   <- the honest market")
    print(f"  Nifty 50                             : {cn:+7.2f}%/yr")
    print(f"\n  SELECTION PREMIUM (mapped - full)    : {cm - cf:+7.2f}%/yr")
    print(f"  'basket premium vs Nifty' previously reported: {cm - cn:+7.2f}%/yr")
    print(f"  ... of which selection/survivorship   : {cm - cf:+7.2f}%/yr "
          f"({(cm - cf) / max(cm - cn, 1e-9) * 100:.0f}% of it)")
    # per-era, since deletion exposure shrinks toward the present
    print(f"\n  {'era':<10}{'mapped':>10}{'full':>10}{'premium':>10}")
    for lbl, y0, y1 in (("2018-21", 2018, 2021), ("2022-24", 2022, 2024),
                        ("2025-26", 2025, 2026)):
        m = (sig.ret.index.year >= y0) & (sig.ret.index.year <= y1)
        yy = (sig.ret.index[m][-1] - sig.ret.index[m][0]).days / 365.25
        c = lambda s: (np.exp(np.log1p(s[m].fillna(0) / 100.0).sum() / yy) - 1.0) * 100.0
        print(f"  {lbl:<10}{c(mapped):>10.2f}{c(fs):>10.2f}{c(mapped) - c(fs):>10.2f}")
    print("  -> the premium should SHRINK toward the present as there is less future")
    print("     deletion left to bias the sample. If it does, it is survivorship.")
    return fs


def D_edge_by_year(sig):
    print("\n" + "=" * 78)
    print("D. TILT EDGE BY YEAR vs YEARS-OF-FUTURE-DELETION EXPOSURE")
    print("=" * 78)
    print("   prediction: a real signal is FLAT in exposure; survivorship slopes DOWN")
    last_yr = int(sig.ret.index.year.max())
    years = list(range(2018, last_yr + 1))
    print(f"  {'horizon':<10}" + "".join(f"{y:>8}" for y in years) + f"{'slope/yr':>11}{'corr':>8}")
    for lbl, h in HORIZONS.items():
        row = []
        for y in years:
            sel = lambda idx, y=y: idx.year == y
            a = [ann(backtest(sig, h, phase=p, dates=sel), h) for p in range(h)]
            a = [x for x in a if np.isfinite(x)]
            row.append(np.mean(a) if a else np.nan)
        r = np.array(row, float)
        ok = np.isfinite(r)
        expo = np.array([last_yr - y for y in years], float)   # years of future deletion
        if ok.sum() >= 4:
            sl = np.polyfit(expo[ok], r[ok], 1)[0]
            cc = float(np.corrcoef(expo[ok], r[ok])[0, 1])
        else:
            sl = cc = np.nan
        print(f"  {lbl:<10}" + "".join(f"{v:>8.1f}" if np.isfinite(v) else f"{'—':>8}"
                                       for v in r) + f"{sl:>11.2f}{cc:>8.2f}")
    print("\n  slope = %/yr of edge gained per extra YEAR of future-deletion exposure.")
    print("  positive slope + positive corr = the edge lives where the bias lives.")


def E_sector_thinness(con):
    print("\n" + "=" * 78)
    print("E. PER-SECTOR CONSTITUENT GROWTH — whose history is thinnest")
    print("=" * 78)
    d = con.sql(f"""
        select s.sector,
               count(distinct case when year(b.trade_date)=2018 then b.symbol end) n2018,
               count(distinct case when year(b.trade_date)=2026 then b.symbol end) n2026
        from daily_data b join v_sector_master s on b.symbol=s.symbol
        where b.series in ('EQ','SM','ST') and b.turnover_lacs >= {MIN_TURN}
          and s.sector not in ('ETF','Others')
        group by 1
    """).df()
    d["growth_x"] = d["n2026"] / d["n2018"].replace(0, np.nan)
    d = d.sort_values("growth_x", ascending=False)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  -> a sector at 4 names in 2018 and 30 today had its 2018 'sector return'")
    print("     computed from 4 survivors. Its early history is close to meaningless,")
    print("     and it competes in the SAME cross-sectional rank as a stable sector.")


def main():
    con = duckdb.connect(DB, read_only=True)
    A_snapshot(con)
    B_discard(con)
    sig = Sig(con)
    C_selection(con, sig)
    E_sector_thinness(con)
    con.close()
    D_edge_by_year(sig)


if __name__ == "__main__":
    main()
