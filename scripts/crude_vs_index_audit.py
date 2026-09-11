"""Stage 4: adversarial audit of the Commodity vs Index claims.

A. RUPEE CONFOUND. MCX crude is priced in rupees = USD crude x USD/INR. A weak
   rupee is itself bad news for Nifty (FII outflows), so part of the 2026
   'crude up, Nifty down' link could be the currency, not the oil. Split it:
   same-week link of Nifty with INR crude, USD crude, and USD/INR alone, plus a
   two-factor regression. USD/INR = ECB reference (CFM fx_spot, 2000+).
B. REGIME PERSISTENCE at a finer step. The 26-week/26-week test had only 14
   disjoint half-years. Re-test with 13-week steps (trailing 26w link vs the
   NEXT 13 weeks' link) -- more samples, closer to how a trader would use it.
C. NATGAS SURVIVOR vs gap reversion. Indian index open-to-close is partly a
   reversal of the opening gap. If natgas rallies coincide with gap-downs, the
   'edge' is gap reversion. The gap is known at 09:15, so controlling for it is
   legitimate for a tradable-at-open claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analytics.commodity_index import (START, align_to_sessions,  # noqa: E402
                                           commodity_features, index_outcomes,
                                           load_commodity, load_index_ohlc)
from crude_vs_index_study import ols_hac, spear  # noqa: E402

CFM_DB = r"D:\Python Projects\Commodity_Forex_Market\data\cfm.duckdb"


def usdinr() -> pd.Series:
    con = duckdb.connect(CFM_DB, read_only=True)
    s = con.execute("SELECT quote_date d, rate FROM fx_spot WHERE pair='USDINR' "
                    "AND source='ecb_reference' ORDER BY 1").fetchdf()
    con.close()
    return s.assign(d=pd.to_datetime(s.d)).set_index("d").rate


def weekly(series: dict[str, pd.Series], days: pd.DatetimeIndex) -> pd.DataFrame:
    """5-session log-ish returns on NSE days, non-overlapping, anchored at end."""
    out = {}
    for k, s in series.items():
        v = s.reindex(days, method="ffill")
        out[k] = v / v.shift(5) - 1
    w = pd.DataFrame(out, index=days)
    pos = np.arange(len(w))[::-1][::5][::-1]
    return w.iloc[pos].dropna()


def part_a():
    print("=== A. rupee confound: same-week link with Nifty 50 ===")
    p = load_commodity("CRUDE OIL")
    fx = usdinr()
    _, c = load_index_ohlc(["Nifty 50"])
    n = c["Nifty 50"]
    inr = p["lvl"]
    fx_d = fx.reindex(inr.index, method="ffill")
    usd = inr / fx_d                       # chained INR crude / USDINR
    w = weekly({"inr_crude": inr, "usd_crude": usd, "usdinr": fx, "nifty": n},
               c.index[c.index >= START])
    rows = []
    for lab, g in [("2018-2026", w), ("2022", w[w.index.year == 2022]),
                   ("2024-2025", w[(w.index.year >= 2024) & (w.index.year <= 2025)]),
                   ("2026", w[w.index.year == 2026])]:
        X = np.column_stack([np.ones(len(g)), g.usd_crude, g.usdinr])
        b, t = ols_hac(g.nifty.values, X, 1)
        rows.append(dict(period=lab, weeks=len(g),
                         inr_crude=spear(g.inr_crude, g.nifty),
                         usd_crude=spear(g.usd_crude, g.nifty),
                         usdinr=spear(g.usdinr, g.nifty),
                         crude_vs_fx=spear(g.usd_crude, g.usdinr),
                         t_usd_crude=t[1], t_usdinr=t[2]))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"2026 move: USDINR {fx.loc['2025-12-31']:.2f} -> {fx.iloc[-1]:.2f} "
          f"({fx.iloc[-1] / fx.loc['2025-12-31'] - 1:+.1%}); INR crude "
          f"{inr.iloc[-1] / inr.loc[:'2025-12-31'].iloc[-1] - 1:+.1%}; USD crude "
          f"{usd.iloc[-1] / usd.loc[:'2025-12-31'].iloc[-1] - 1:+.1%}")


def part_b():
    print("\n=== B. regime persistence, 13-week steps ===")
    p = load_commodity("CRUDE OIL")
    _, c = load_index_ohlc(["Nifty 50"])
    w = weekly({"cr": p["lvl"], "n": c["Nifty 50"]}, c.index[c.index >= START])
    past = w.cr.rolling(26).corr(w.n)
    nxt = w.cr[::-1].rolling(13).corr(w.n[::-1])[::-1].shift(-1)   # next 13 weeks
    d = pd.DataFrame({"past": past, "next": nxt}).dropna().iloc[::13]
    agree = (np.sign(d.past) == np.sign(d.next)).mean()
    print(f"{len(d)} disjoint 13-week steps: corr(past26, next13) {d.past.corr(d.next):+.2f}, "
          f"sign agrees {agree:.0%}")
    strong = d[d.past.abs() > 0.4]
    print(f"when |past| > 0.4 ({len(strong)} cases): sign agrees "
          f"{(np.sign(strong.past) == np.sign(strong.next)).mean():.0%}; "
          f"cases: " + ", ".join(f"{i:%b%y} {r.past:+.2f}->{r.next:+.2f}"
                                 for i, r in strong.iterrows()))


def part_c():
    print("\n=== C. natgas survivor vs gap reversion ===")
    p = load_commodity("NATURALGAS")
    f = commodity_features(p)
    idx = ["Nifty 50", "Nifty Energy", "Nifty PSE", "Nifty Metal"]
    o, c = load_index_ohlc(idx)
    a = align_to_sessions(f, c.index)
    oc = index_outcomes(o, c)
    valid = (a.r1.notna() & (a.index >= START)).values
    m = (a.pct10 >= 0.90).fillna(False).to_numpy(bool) & valid
    for ix in idx:
        y, g = oc["oc1"][ix].values, oc["gap"][ix].values
        ok = valid & ~np.isnan(y) & ~np.isnan(g)
        b0, t0 = ols_hac(y[ok], np.column_stack([np.ones(ok.sum()), m[ok]]), 5)
        b1, t1 = ols_hac(y[ok], np.column_stack([np.ones(ok.sum()), m[ok], g[ok]]), 5)
        gm = np.nanmean(g[ok & m]) - np.nanmean(g[ok & ~m])
        print(f"  {ix:14s} oc1 excess {b0[1]*100:+.3f}% t {t0[1]:+.2f} | +gap control "
              f"{b1[1]*100:+.3f}% t {t1[1]:+.2f} (gap beta {b1[2]:+.2f} t {t1[2]:+.1f}) | "
              f"signal-day gap vs others {gm*100:+.3f}%")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
