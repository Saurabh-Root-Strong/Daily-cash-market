"""
backtest_banknifty_constituent_v2.py — exhaust the constituent formulations for a
Bank Nifty next-day edge (futures already null in v1; here: DELIVERY, OPTIONS,
conviction = OI×delivery, and dispersion→move-size). Honest go/no-go before wiring.

Walk-forward, point-in-time, weight-weighted by Bank Nifty index weight. Outcome =
Bank Nifty NEXT-day % return (direction) and |return| (magnitude). All features use
only data up to and including day D.
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

_WEIGHTS = {
    "HDFCBANK":17.93,"ICICIBANK":13.63,"AXISBANK":10.28,"KOTAKBANK":9.81,"FEDERALBNK":6.38,
    "INDUSINDBK":5.40,"IDFCFIRSTB":4.27,"YESBANK":3.67,"SBIN":9.07,"BANKBARODA":4.47,
    "CANBK":3.98,"PNB":3.30,"UNIONBANK":2.93,"AUBANK":4.87,
}
SYMS = tuple(_WEIGHTS)


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10: return np.nan, int(m.sum())
    ra = pd.Series(a[m]).rank().to_numpy(); rb = pd.Series(b[m]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]), int(m.sum())


def main():
    ph = ",".join("?" * len(SYMS))
    # daily_data: price chg + delivery% + 20d delivery baseline (point-in-time, excl today)
    dd = query_dataframe(f"""
        SELECT trade_date, symbol,
               CASE WHEN prev_close>0 THEN (close_price-prev_close)/prev_close*100 END AS pchg,
               deliv_per,
               AVG(deliv_per) OVER (PARTITION BY symbol ORDER BY trade_date
                                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS deliv_avg20
        FROM daily_data WHERE series='EQ' AND symbol IN ({ph})
    """, list(SYMS))
    # OPTSTK near-month PCR per symbol (put OI / call OI), nearest expiry on/after D
    opt = query_dataframe(f"""
        WITH ne AS (SELECT symbol, trade_date, MIN(expiry_date) e FROM fno_bhavcopy
                    WHERE instrument='OPTSTK' AND symbol IN ({ph}) AND expiry_date>=trade_date
                    GROUP BY symbol, trade_date)
        SELECT o.trade_date, o.symbol,
               SUM(CASE WHEN o.option_type='PE' THEN o.open_interest ELSE 0 END) AS poi,
               SUM(CASE WHEN o.option_type='CE' THEN o.open_interest ELSE 0 END) AS coi
        FROM fno_bhavcopy o JOIN ne ON o.symbol=ne.symbol AND o.trade_date=ne.trade_date
                                   AND o.expiry_date=ne.e
        WHERE o.instrument='OPTSTK' GROUP BY o.trade_date, o.symbol
    """, list(SYMS))
    fut = query_dataframe(f"""
        SELECT trade_date, symbol, SUM(open_interest) AS oi FROM fno_bhavcopy
        WHERE instrument='FUTSTK' AND symbol IN ({ph}) GROUP BY trade_date, symbol
    """, list(SYMS))
    bn = query_dataframe("""
        SELECT trade_date, pct_chg FROM index_data
        WHERE index_name='Nifty Bank' AND pct_chg IS NOT NULL ORDER BY trade_date
    """)
    for d in (dd, opt, fut, bn): d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date

    fut = fut.sort_values(["symbol","trade_date"]); fut["oichg"] = fut.groupby("symbol")["oi"].diff()
    m = dd.merge(opt, on=["trade_date","symbol"], how="left").merge(
        fut[["trade_date","symbol","oichg"]], on=["trade_date","symbol"], how="left")
    m["w"] = m["symbol"].map(_WEIGHTS)
    m = m.dropna(subset=["pchg"])
    m["deliv_ratio"] = m["deliv_per"] / m["deliv_avg20"].replace(0, np.nan)
    m["pcr"] = m["poi"] / m["coi"].replace(0, np.nan)
    # per-stock conviction = OI-price matrix score × delivery conviction
    def ops(p, o):
        if pd.isna(o): return 0.0
        return (1.0 if p>0 and o>0 else 0.5 if p>0 and o<0 else -0.5 if p<0 and o<0 else -1.0 if p<0 and o>0 else 0.0)
    m["fscore"] = [ops(p, o) for p, o in zip(m["pchg"], m["oichg"])]

    bn = bn.sort_values("trade_date").reset_index(drop=True)
    bn["next"] = bn["pct_chg"].shift(-1)
    nxt = dict(zip(bn["trade_date"], bn["next"])); same = dict(zip(bn["trade_date"], bn["pct_chg"]))

    rows = []
    for td, g in m.groupby("trade_date"):
        w = g["w"]; wsum = w.sum()
        def wm(x):
            v = (x * w); ok = v.notna() & x.notna()
            return float(v[ok].sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan
        rows.append({
            "date": td,
            "wdeliv_dir": wm(np.sign(g["pchg"]) * g["deliv_ratio"]),  # delivery-confirmed direction
            "wdeliv":     wm(g["deliv_ratio"]),                        # pure delivery intensity
            "wpcr":       wm(g["pcr"]),                                # weighted constituent PCR
            "wconv":      wm(g["fscore"] * g["deliv_ratio"].clip(0.5, 2)),  # OI-price × delivery conviction
            "disp":       float(np.sqrt((w*(g["pchg"]-wm(g["pchg"]))**2).sum()/wsum)) if wsum>0 else np.nan,
            "same": same.get(td, np.nan), "next": nxt.get(td, np.nan),
        })
    df = pd.DataFrame(rows).dropna(subset=["next"])
    print(f"[n={len(df)} days {df.date.min()}..{df.date.max()}]\n")
    print("DIRECTION — next-day IC vs Bank Nifty return:")
    for c, l in [("wdeliv_dir","delivery-confirmed direction"),("wdeliv","delivery intensity"),
                 ("wpcr","weighted constituent PCR"),("wconv","OI-price×delivery conviction")]:
        ic, n = _spear(df[c], df["next"]); print(f"  {l:32} IC={ic:+.3f} n={n}")
    print("\nMAGNITUDE — does constituent DISPERSION predict next-day |move|?")
    ic, n = _spear(df["disp"], df["next"].abs()); print(f"  dispersion → |next move|        IC={ic:+.3f} n={n}")
    ic2, _ = _spear(df["wdeliv"], df["next"].abs()); print(f"  delivery intensity → |next move| IC={ic2:+.3f}")

    # ── OOS gate (fit 70% / test 30%) — the codebase's overfit guard ──────────
    dfo = df.sort_values("date").reset_index(drop=True)
    cut = int(len(dfo) * 0.7); tr, te = dfo.iloc[:cut], dfo.iloc[cut:]
    print(f"\nOOS GATE (train n={len(tr)} / test n={len(te)}):")
    for c, lab, tgt in [("wpcr", "wPCR → direction", te["next"]),
                        ("disp", "dispersion → |next move|", te["next"].abs())]:
        is_ic = _spear(tr[c], tr["next"].abs() if c == "disp" else tr["next"])[0]
        oos_ic, on = _spear(te[c], tgt)
        print(f"  {lab:26} IS={is_ic:+.3f} | OOS={oos_ic:+.3f} (n={on})")
    print("\nDONE")


if __name__ == "__main__":
    main()
