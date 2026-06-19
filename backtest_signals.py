"""
Walk-forward signal-attribution backtest.

For every F&O trading day in history it re-runs the live prediction engine
(as-of that day, no look-ahead — same path cmd_daily uses), captures EVERY
individual signal's (name, category, direction, score), and joins it to the
REALIZED next-trading-day return of that index. Then it reports, per signal
and per signal-family:

  fire%   how often the signal is emitted
  hit%    directional accuracy (sign(direction) == sign(next-day return))
  IC      Spearman rank corr( signed score , next-day return )  -> the edge
  |score| average magnitude (how loud the signal is)

Goal: separate signals that carry edge from dead weight / wrong-sign signals,
so the composite can cap or cut the latter. Pure measurement — changes nothing.
"""
from __future__ import annotations
import sys, math
from datetime import date
import numpy as np, pandas as pd
from src.data.repository import get_repository, query_dataframe
from src.analytics.index_prediction import get_index_predictions

_IDX = {"NIFTY":"Nifty 50","BANKNIFTY":"Nifty Bank",
        "FINNIFTY":"Nifty Financial Services","MIDCPNIFTY":"Nifty Midcap Select"}

def next_day_return(idx_name: str, d: date) -> float | None:
    df = query_dataframe(
        "SELECT pct_chg FROM index_data WHERE index_name=? AND trade_date>? "
        "ORDER BY trade_date ASC LIMIT 1", [idx_name, d])
    if df.empty or df["pct_chg"].isna().all():
        return None
    return float(df["pct_chg"].iloc[0])

def main():
    dates = [d.date() if hasattr(d,"date") else d for d in query_dataframe(
        "SELECT DISTINCT trade_date FROM fno_bhavcopy ORDER BY trade_date"
    )["trade_date"].tolist()]
    print(f"[harness] {len(dates)} F&O days: {dates[0]} .. {dates[-1]}", flush=True)

    rows = []
    for i, td in enumerate(dates):
        try:
            preds = get_index_predictions(td, persist=False)
        except Exception as e:
            print(f"  ! {td} compute failed: {e}", flush=True); continue
        for p in preds:
            if not getattr(p, "data_available", True):
                continue
            r = next_day_return(_IDX[p.fno_symbol], td)
            if r is None:
                continue  # no settled outcome yet (last day)
            for s in p.signals:
                rows.append({
                    "date": td, "symbol": p.fno_symbol,
                    "name": s.name, "category": s.category,
                    "direction": int(s.direction), "score": float(s.score),
                    "next_ret": r,
                    "composite": float(p.composite_score),
                })
        if (i+1) % 5 == 0:
            print(f"  .. {i+1}/{len(dates)} days", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet("signal_attribution.parquet")
    print(f"[harness] captured {len(df)} signal-rows across "
          f"{df['date'].nunique()} days x {df['symbol'].nunique()} indices\n", flush=True)

    def ic(g):  # Spearman rank IC of signed score vs next-day return
        if g["score"].nunique() < 3 or g["next_ret"].nunique() < 3: return np.nan
        return g["score"].rank().corr(g["next_ret"].rank())
    def hit(g):  # directional hit-rate on non-neutral fires (ignore tiny moves)
        m = (g["direction"] != 0) & (g["next_ret"].abs() >= 0.10)
        if m.sum() == 0: return np.nan
        gg = g[m]
        ok = ((gg["direction"] > 0) & (gg["next_ret"] > 0)) | \
             ((gg["direction"] < 0) & (gg["next_ret"] < 0))
        return ok.mean()

    total_days = df["date"].nunique() * df["symbol"].nunique()

    # ---- per-signal ----
    recs = []
    for name, g in df.groupby("name"):
        recs.append({"signal": name, "cat": g["category"].iloc[0],
                     "fires": len(g), "fire%": round(100*len(g)/total_days,0),
                     "hit%": round(100*hit(g),0) if not math.isnan(hit(g)) else np.nan,
                     "IC": round(ic(g),3), "|score|": round(g["score"].abs().mean(),2),
                     "mean_dir": round(g["direction"].mean(),2)})
    sig = pd.DataFrame(recs).sort_values("IC", na_position="last")
    pd.set_option("display.width",200,"display.max_rows",80,"display.max_columns",20)
    print("="*70, "\nPER-SIGNAL EDGE (sorted by IC; negative IC = wrong-sign / fade)\n", "="*70)
    print(sig.to_string(index=False))

    # ---- per-family (category) ----
    fam = []
    for cat, g in df.groupby("category"):
        # build per-(date,symbol) family net score vs next_ret
        agg = g.groupby(["date","symbol"]).agg(fscore=("score","sum"),
                                               r=("next_ret","first")).reset_index()
        fic = agg["fscore"].rank().corr(agg["r"].rank()) if len(agg) >= 5 else np.nan
        fhit = (((agg["fscore"]>0)&(agg["r"]>0))|((agg["fscore"]<0)&(agg["r"]<0)))[
                agg["r"].abs()>=0.10].mean() if len(agg) else np.nan
        fam.append({"family": cat, "signals": g["name"].nunique(),
                    "rows": len(g), "fam_IC": round(fic,3) if fic==fic else np.nan,
                    "fam_hit%": round(100*fhit,0) if fhit==fhit else np.nan,
                    "mean_net": round(agg["fscore"].mean(),2)})
    print("\n", "="*70, "\nPER-FAMILY EDGE (net family score vs next-day return)\n", "="*70)
    print(pd.DataFrame(fam).sort_values("fam_IC", na_position="last").to_string(index=False))

    # ---- composite baseline per symbol ----
    comp = df.groupby(["date","symbol"]).agg(c=("composite","first"),
                                             r=("next_ret","first")).reset_index()
    print("\n", "="*70, "\nCOMPOSITE BASELINE (per index)\n", "="*70)
    cr=[]
    for sym,g in comp.groupby("symbol"):
        cic = g["c"].rank().corr(g["r"].rank())
        chit = (((g["c"]>0)&(g["r"]>0))|((g["c"]<0)&(g["r"]<0)))[g["r"].abs()>=0.10].mean()
        cr.append({"symbol":sym,"n":len(g),"composite_IC":round(cic,3),
                   "dir_hit%":round(100*chit,0)})
    print(pd.DataFrame(cr).to_string(index=False))
    print("\n[harness] done -> signal_attribution.parquet", flush=True)

if __name__ == "__main__":
    main()
