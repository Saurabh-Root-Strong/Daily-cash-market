"""
Is "TRENDING_UP" one state, or two?

THE PROBLEM ON SCREEN
    The backdrop label is built from EMA20/EMA50 only — it has NO 200-day
    awareness. So it prints TRENDING_UP (verdict ACT, size 100%) during a bear
    rally, while the same dashboard simultaneously reports "0.8% below the
    200-day line", "death cross", "RECOVERING — not confirmed bull" and
    "choppy/grinding, ER 0.16". Those are not contradictions in the data; they
    are two engines reading different windows, with only one of them driving the
    size dial.

THE TEST
    Split TRENDING_UP days by whether Nifty is ABOVE or BELOW its 200-day SMA and
    compare what the tilt actually earned. If the two halves pay the same, the
    label is fine and the size dial is right. If the below-200 half is materially
    worse, the state is overconfident exactly when the market is most fragile,
    and it should be split.

MEASURE
    OW-UW = mean forward excess of the top-4 minus the bottom-4 sectors by the
    SHIPPED tilt score (0.60*rank(rs_2w) + 0.25*rank(rs_1w) + 0.15*rank(dv5d)),
    forward 10 trading days, excess over the equal-weight sector basket.
    Long-only top-4 is also reported because the live book is long-only.
    Inference is Newey-West at lag=10 (forward windows overlap).
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv = [sys.argv[0]]

from scripts.study_tilt_horizons import (DB, DV_BASE, DV_FLOW, W_DV5, W_RS1,  # noqa: E402
                                         W_RS2, load_panel, nw_t)


def main():
    con = duckdb.connect(DB, read_only=True)
    panel = load_panel(con)
    nf = con.sql("""select trade_date, close_val from index_data
                    where index_name='Nifty 50' order by trade_date""").df()
    con.close()
    nf["trade_date"] = pd.to_datetime(nf["trade_date"])
    nfs = nf.set_index("trade_date")["close_val"].astype(float).sort_index()

    ret = panel.pivot(index="trade_date", columns="sector",
                      values="wtd_ret_pct").sort_index().dropna(how="all")
    dv = panel.pivot(index="trade_date", columns="sector", values="daily_dv_cr").sort_index()
    lg = np.log1p(ret / 100.0)
    nret = nfs.pct_change() * 100.0
    ngr = np.log1p(nret / 100.0)
    trail = lambda n: (np.expm1(lg.rolling(n).sum())) * 100.0
    ntrail = lambda n: (np.expm1(ngr.rolling(n).sum())) * 100.0

    rs2 = trail(10).sub(ntrail(10).reindex(ret.index), axis=0)
    rs1 = trail(5).sub(ntrail(5).reindex(ret.index), axis=0)
    dv5 = dv.rolling(DV_FLOW).mean() / dv.shift(1).rolling(DV_BASE).mean()
    r = lambda d: d.rank(axis=1, pct=True)
    score = W_RS2 * r(rs2) + W_RS1 * r(rs1) + W_DV5 * r(dv5)

    H = 10
    f = (np.expm1(lg.shift(-1).iloc[::-1].rolling(H, min_periods=H).sum().iloc[::-1])) * 100.0
    f = f.sub(f.mean(axis=1), axis=0)

    # regime flags on the Nifty, aligned to the sector panel's dates
    e20 = nfs.ewm(span=20, adjust=False).mean()
    e50 = nfs.ewm(span=50, adjust=False).mean()
    sma200 = nfs.rolling(200).mean()
    up = ((nfs > e20) & (e20 > e50)).reindex(ret.index)
    above200 = (nfs > sma200).reindex(ret.index)
    dc = (e50 < sma200).reindex(ret.index)          # long-term lines crossed down

    rows = []
    for dt_ in score.index:
        s, y = score.loc[dt_], f.loc[dt_]
        m = s.notna() & y.notna()
        if m.sum() < 10:
            continue
        k = 4
        ow = y[m].loc[s[m].nlargest(k).index].mean()
        uw = y[m].loc[s[m].nsmallest(k).index].mean()
        rows.append({"date": dt_, "ow": ow, "uw": uw, "ls": ow - uw,
                     "up": bool(up.get(dt_, False)),
                     "a200": bool(above200.get(dt_, False)),
                     "dc": bool(dc.get(dt_, False))})
    d = pd.DataFrame(rows).dropna()
    print(f"panel {len(d):,} sessions {d.date.min():%Y-%m-%d} -> {d.date.max():%Y-%m-%d}")

    def rep(label, sub):
        if len(sub) < 40:
            print(f"  {label:<34} n={len(sub):<5} (too few)"); return
        print(f"  {label:<34} n={len(sub):<5} "
              f"OW-UW {sub.ls.mean():+6.2f}%  t {nw_t(sub.ls.values, H):+5.2f}   |   "
              f"long-only OW {sub.ow.mean():+6.2f}%  t {nw_t(sub.ow.values, H):+5.2f}   "
              f"{100*len(sub)/len(d):4.1f}% of days")

    print(f"\n=== forward {H}d, excess vs equal-weight sector basket ===")
    rep("ALL sessions", d)
    print()
    rep("TRENDING_UP (as shipped)", d[d.up])
    rep("  ... and ABOVE the 200-DMA", d[d.up & d.a200])
    rep("  ... and BELOW the 200-DMA", d[d.up & ~d.a200])
    print()
    rep("NOT trending up", d[~d.up])
    rep("  ... above 200-DMA", d[~d.up & d.a200])
    rep("  ... below 200-DMA", d[~d.up & ~d.a200])

    print("\n=== death-cross overlay (EMA50 < SMA200), inside TRENDING_UP ===")
    rep("TRENDING_UP, no death cross", d[d.up & ~d.dc])
    rep("TRENDING_UP, death cross ON", d[d.up & d.dc])

    print("\n=== paired difference: does 'below 200' cost the tilt? ===")
    a = d[d.up & d.a200]; b = d[d.up & ~d.a200]
    if len(a) > 40 and len(b) > 40:
        print(f"  long-only OW: above {a.ow.mean():+.2f}%  vs below {b.ow.mean():+.2f}%  "
              f"-> gap {a.ow.mean()-b.ow.mean():+.2f}pp")
        print(f"  OW-UW      : above {a.ls.mean():+.2f}%  vs below {b.ls.mean():+.2f}%  "
              f"-> gap {a.ls.mean()-b.ls.mean():+.2f}pp")

    print("\n=== how often does the board show a full-size call in a broken tape? ===")
    n_up = int(d.up.sum())
    n_bad = int((d.up & ~d.a200).sum())
    print(f"  TRENDING_UP days: {n_up:,} ({100*n_up/len(d):.1f}% of all sessions)")
    print(f"  of those, BELOW the 200-DMA: {n_bad:,} ({100*n_bad/max(n_up,1):.1f}% of TRENDING_UP)")
    print(f"  i.e. {100*n_bad/len(d):.1f}% of all sessions show 'ACT / size 100%' "
          f"while the long-term trend is broken.")


if __name__ == "__main__":
    main()
