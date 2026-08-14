"""
backtest_banknifty_constituent.py — does WEIGHT-WEIGHTED constituent F&O carry a
next-day edge for Bank Nifty that the (pruned) EQUAL-COUNT breadth signal didn't?

Thesis (user, 2026-06-20): Bank Nifty is 71% private / 24% PSU by weight. A few
heavyweights (HDFC 17.9%, ICICI 13.6%, AXIS 10.3%) dominate. So whether the index
move is BACKED by the heavy constituents' stock-futures positioning — weighted by
index weight, not a flat stock count — should add an edge / move-quality read:
  • CONFIRMATION: index down AND heavy banks fresh-short  → move has fuel (continue)
  • DIVERGENCE:   index down BUT heavy banks long/short-cover → move unsupported (fade)

Walk-forward, point-in-time. For each F&O day D:
  per constituent: stock price%chg (daily_data) × total FUTSTK OI chg vs prev day →
    OI-price matrix score: FreshLong +1 · ShortCov +0.5 · LongUnwind −0.5 · FreshShort −1
  weighted_score = Σ w_i · score_i   (index weights below; renormalised over covered)
  measured vs Bank Nifty NEXT-day return.

Reports: IC of (all / private-only / PSU-only) weighted score; the equal-count net for
comparison; and the confirmation-vs-divergence conditional next-day means.

Usage: python scripts/backtest_banknifty_constituent.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from src.data.repository import query_dataframe

# Bank Nifty weights (user treemap snapshot 2026-06; static — refresh periodically).
_WEIGHTS = {
    "HDFCBANK": 17.93, "ICICIBANK": 13.63, "AXISBANK": 10.28, "KOTAKBANK": 9.81,
    "FEDERALBNK": 6.38, "INDUSINDBK": 5.40, "IDFCFIRSTB": 4.27, "YESBANK": 3.67,   # private
    "SBIN": 9.07, "BANKBARODA": 4.47, "CANBK": 3.98, "PNB": 3.30, "UNIONBANK": 2.93,  # PSU
    "AUBANK": 4.87,                                                                    # other
}
_PRIVATE = {"HDFCBANK","ICICIBANK","AXISBANK","KOTAKBANK","FEDERALBNK","INDUSINDBK","IDFCFIRSTB","YESBANK","AUBANK"}
_PSU     = {"SBIN","BANKBARODA","CANBK","PNB","UNIONBANK"}


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10: return np.nan, int(m.sum())
    ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]), int(m.sum())


def _oi_price_score(price_chg, oi_chg):
    if price_chg > 0 and oi_chg > 0:  return 1.0    # fresh long
    if price_chg > 0 and oi_chg < 0:  return 0.5    # short covering
    if price_chg < 0 and oi_chg < 0:  return -0.5   # long unwinding
    if price_chg < 0 and oi_chg > 0:  return -1.0   # fresh short
    return 0.0


def main():
    syms = tuple(_WEIGHTS)
    ph = ",".join("?" * len(syms))
    # Per-symbol per-day total FUTSTK OI (rollover-immune: sum across expiries).
    fut = query_dataframe(f"""
        SELECT trade_date, symbol, SUM(open_interest) AS oi
        FROM fno_bhavcopy WHERE instrument='FUTSTK' AND symbol IN ({ph})
        GROUP BY trade_date, symbol ORDER BY trade_date
    """, list(syms))
    # Per-symbol daily stock price %chg from daily_data (EQ series).
    px = query_dataframe(f"""
        SELECT trade_date, symbol,
               CASE WHEN prev_close>0 THEN (close_price-prev_close)/prev_close*100 END AS pchg
        FROM daily_data WHERE series='EQ' AND symbol IN ({ph})
    """, list(syms))
    bn = query_dataframe("""
        SELECT trade_date, pct_chg FROM index_data
        WHERE index_name='Nifty Bank' AND pct_chg IS NOT NULL ORDER BY trade_date
    """)
    for d in (fut, px, bn):
        d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date

    fut = fut.sort_values(["symbol","trade_date"])
    fut["oi_prev"] = fut.groupby("symbol")["oi"].shift(1)
    fut["oi_chg"] = fut["oi"] - fut["oi_prev"]
    m = fut.merge(px, on=["trade_date","symbol"], how="inner").dropna(subset=["oi_chg","pchg"])
    m["s"] = [ _oi_price_score(p, o) for p, o in zip(m["pchg"], m["oi_chg"]) ]
    m["w"] = m["symbol"].map(_WEIGHTS)

    # Bank Nifty next-day return map + same-day return.
    bn = bn.sort_values("trade_date").reset_index(drop=True)
    bn["next"] = bn["pct_chg"].shift(-1)
    bn_same = dict(zip(bn["trade_date"], bn["pct_chg"]))
    bn_next = dict(zip(bn["trade_date"], bn["next"]))

    rows = []
    for td, g in m.groupby("trade_date"):
        def wscore(sub):
            wsum = sub["w"].sum()
            return float((sub["s"] * sub["w"]).sum() / wsum) if wsum > 0 else np.nan
        allw = wscore(g)
        prv  = wscore(g[g["symbol"].isin(_PRIVATE)])
        psu  = wscore(g[g["symbol"].isin(_PSU)])
        eqct = float(g["s"].mean())  # equal-count (the pruned style)
        rows.append({"date": td, "wall": allw, "wpriv": prv, "wpsu": psu, "eqct": eqct,
                     "same": bn_same.get(td, np.nan), "next": bn_next.get(td, np.nan),
                     "ncov": len(g)})
    df = pd.DataFrame(rows).dropna(subset=["next"])
    print(f"[n={len(df)} days, {df.date.min()}..{df.date.max()}, avg {df.ncov.mean():.1f} banks/day]\n")

    print("NEXT-DAY IC (Spearman) of weighted constituent F&O score vs Bank Nifty next return:")
    for col, lab in [("wall","weighted ALL"),("wpriv","weighted PRIVATE"),("wpsu","weighted PSU"),
                     ("eqct","equal-count (pruned style)")]:
        ic, n = _spearman(df[col], df["next"])
        print(f"  {lab:28} IC={ic:+.3f}  n={n}")

    # Confirmation vs divergence (user's exact thesis): split on whether the weighted
    # constituent score AGREES with Bank Nifty's SAME-day move.
    print("\nCONFIRMATION vs DIVERGENCE (weighted-ALL score vs same-day index move):")
    sign_idx = np.sign(df["same"]); sign_con = np.sign(df["wall"])
    confirm = df[sign_idx == sign_con]
    diverge = df[(sign_idx != sign_con) & (sign_idx != 0) & (sign_con != 0)]
    print(f"  confirm  (constituents agree w/ move): n={len(confirm):3d}  "
          f"next mean={confirm['next'].mean():+.3f}%  |next| persist (same sign next)="
          f"{(np.sign(confirm['next'])==sign_idx[confirm.index]).mean():.0%}")
    print(f"  diverge  (constituents oppose move):   n={len(diverge):3d}  "
          f"next mean={diverge['next'].mean():+.3f}%  fade (next opposes today)="
          f"{(np.sign(diverge['next'])!=np.sign(diverge['same'])).mean():.0%}")

    # Does divergence specifically dampen the move magnitude next day?
    print(f"\n  avg |next-day move|:  confirm={confirm['next'].abs().mean():.3f}%  "
          f"diverge={diverge['next'].abs().mean():.3f}%")
    print("\nDONE")


if __name__ == "__main__":
    main()
