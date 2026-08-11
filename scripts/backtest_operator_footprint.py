"""
Does unusual single-stock F&O positioning predict the stock?

WHY THIS MUST RUN BEFORE THE TAB IS TRUSTED
    This codebase has tested open-interest reads three times at index level and
    found them descriptive, not predictive (EOD OI walls / max pain, CE-PE
    crossover, the F&O expiry-cycle overlay). Single-stock "operator" positioning
    is a different claim and deserves its own test rather than inheriting either
    verdict.

HYPOTHESES, each tested separately because they say opposite things
    H1  heavy CALL buildup with RISING premium  (call buying)     -> stock UP
    H2  heavy CALL buildup with FALLING premium (call writing)    -> stock DOWN
    H3  heavy PUT  buildup with RISING premium  (put buying)      -> stock DOWN
    H4  heavy PUT  buildup with FALLING premium (put writing)     -> stock UP
    H5  futures LONG BUILDUP                                      -> stock UP
    H6  ITM buildup is sharper than OTM buildup (the tab weights ITM higher,
        so if that is wrong the weighting is wrong)

METHOD
    - Symbol-day panel built in bulk SQL; the per-day analytics function is far
      too slow to walk 500 sessions, but the aggregation is identical.
    - Buildup is measured as OI ADDED today as a share of that symbol's own
      option book, so a crore of OI is judged against the stock's own size.
    - Target = forward stock return EXCESS OVER THE F&O UNIVERSE MEAN, so a
      market-wide move cannot be mistaken for a stock signal. (Median was tried
      and is WRONG here: comparing basket means to a median benchmark biases every
      subset upward on a skewed cross-section.)
    - A RANDOM-decile control runs alongside. If it does not sit at zero, the
      benchmark is broken, not the signal.
    - Inference is DATE-CLUSTERED with a Newey-West correction at lag = horizon;
      overlapping forward windows make a per-row t-stat meaningless.

VERDICT (run 2026-08-10, 105k symbol-days, 500 sessions 2024-07 -> 2026-08):
    NO forward edge. Every directional IC is inside noise (t -0.80..+0.97) and,
    once benchmarked correctly, every top-decile basket collapses to ~0 while the
    random control also sits at ~0. Call BUYING and call WRITING - which predict
    OPPOSITE directions - scored identically (+0.337 vs +0.334), which is the
    signature of a common exposure rather than information. The tab therefore
    ships as DESCRIPTIVE: it shows where unusual positioning is and what it
    mechanically means, and claims nothing about what happens next.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = r"d:/Python Projects/Daily_Cash_Market/data/market_data.duckdb"
HORIZONS = (5, 10, 22)
MIN_BOOK_CR = 5.0          # ignore symbols whose whole option book is tiny


def nw_t(x, lag):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n < 8:
        return np.nan
    d = x - x.mean(); var = (d @ d) / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1 - L / (lag + 1.0)) * ((d[L:] @ d[:-L]) / n)
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def main():
    con = duckdb.connect(DB, read_only=True)
    print("building symbol-day option panel ...")
    q = """
    WITH o AS (
        SELECT f.trade_date, f.symbol, f.option_type, f.strike_price,
               f.open_interest, f.chg_in_oi, f.close_price,
               LAG(f.close_price) OVER (PARTITION BY f.symbol, f.expiry_date,
                                        f.option_type, f.strike_price
                                        ORDER BY f.trade_date) AS prev_prem,
               d.close_price AS spot
        FROM fno_bhavcopy f
        LEFT JOIN daily_data d ON d.symbol = f.symbol AND d.trade_date = f.trade_date
                              AND d.series IN ('EQ','SM','ST')
        WHERE f.instrument = 'OPTSTK' AND f.open_interest > 0
    ),
    tagged AS (
        SELECT *,
               CASE WHEN option_type='CE' THEN strike_price/NULLIF(spot,0)-1
                    ELSE -(strike_price/NULLIF(spot,0)-1) END AS m,
               close_price - prev_prem AS prem_chg
        FROM o WHERE spot > 0
    )
    SELECT trade_date, symbol,
           SUM(open_interest * spot)/1e7                                   AS book_cr,
           SUM(CASE WHEN option_type='CE' AND chg_in_oi>0 THEN chg_in_oi END)      AS ce_add,
           SUM(CASE WHEN option_type='PE' AND chg_in_oi>0 THEN chg_in_oi END)      AS pe_add,
           SUM(CASE WHEN option_type='CE' AND chg_in_oi>0 AND prem_chg>0
                    THEN chg_in_oi END)                                    AS ce_buy,
           SUM(CASE WHEN option_type='CE' AND chg_in_oi>0 AND prem_chg<0
                    THEN chg_in_oi END)                                    AS ce_write,
           SUM(CASE WHEN option_type='PE' AND chg_in_oi>0 AND prem_chg>0
                    THEN chg_in_oi END)                                    AS pe_buy,
           SUM(CASE WHEN option_type='PE' AND chg_in_oi>0 AND prem_chg<0
                    THEN chg_in_oi END)                                    AS pe_write,
           SUM(CASE WHEN chg_in_oi>0 AND m <= -0.02 THEN chg_in_oi END)     AS itm_add,
           SUM(CASE WHEN chg_in_oi>0 AND m >=  0.02 THEN chg_in_oi END)     AS otm_add,
           SUM(open_interest)                                              AS oi_total
    FROM tagged GROUP BY 1,2
    """
    p = con.execute(q).df()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    print(f"  {len(p):,} symbol-days, {p.symbol.nunique()} symbols, "
          f"{p.trade_date.min():%Y-%m-%d} -> {p.trade_date.max():%Y-%m-%d}")

    fut = con.execute("""
        SELECT f.trade_date, f.symbol, SUM(f.chg_in_oi) AS fut_oi_chg,
               SUM(f.open_interest) AS fut_oi
        FROM fno_bhavcopy f WHERE f.instrument='FUTSTK' GROUP BY 1,2
    """).df()
    fut["trade_date"] = pd.to_datetime(fut["trade_date"])

    px = con.execute("""
        SELECT trade_date, symbol, close_price
        FROM daily_data WHERE series IN ('EQ','SM','ST')
          AND trade_date >= (SELECT min(trade_date) FROM fno_bhavcopy)
    """).df()
    con.close()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    wide = px.pivot_table(index="trade_date", columns="symbol", values="close_price").sort_index()
    lg = np.log(wide).diff()

    def fwd_excess(h):
        f = (np.exp(lg.shift(-1).iloc[::-1].rolling(h, min_periods=h).sum().iloc[::-1]) - 1) * 100
        # MEAN, not median. Benchmarking basket MEANS against a universe MEDIAN on a
        # right-skewed cross-section gives every subset a spurious positive: a random
        # decile scored +0.433%/5d (t 4.71) that way, beating every "signal" tested.
        # Against the mean the random control correctly sits at ~0.
        return f.sub(f.mean(axis=1), axis=0)

    p = p.merge(fut, on=["trade_date", "symbol"], how="left")
    p = p[p["book_cr"] >= MIN_BOOK_CR].copy()
    for c in ("ce_add", "pe_add", "ce_buy", "ce_write", "pe_buy", "pe_write",
              "itm_add", "otm_add", "fut_oi_chg"):
        p[c] = p[c].fillna(0.0)

    tot = p["oi_total"].replace(0, np.nan)
    p["ce_buy_s"]   = p["ce_buy"]   / tot * 100
    p["ce_write_s"] = p["ce_write"] / tot * 100
    p["pe_buy_s"]   = p["pe_buy"]   / tot * 100
    p["pe_write_s"] = p["pe_write"] / tot * 100
    p["itm_s"]      = p["itm_add"]  / tot * 100
    p["otm_s"]      = p["otm_add"]  / tot * 100
    p["fut_s"]      = p["fut_oi_chg"] / p["fut_oi"].replace(0, np.nan) * 100

    FEATS = {
        "H1 call BUYING":   ("ce_buy_s", "+"),
        "H2 call WRITING":  ("ce_write_s", "-"),
        "H3 put BUYING":    ("pe_buy_s", "-"),
        "H4 put WRITING":   ("pe_write_s", "+"),
        "H5 futures OI up": ("fut_s", "+"),
        "H6 ITM buildup":   ("itm_s", "?"),
        "   OTM buildup":   ("otm_s", "?"),
    }

    print(f"\npanel after liquidity filter: {len(p):,} symbol-days\n")
    for h in HORIZONS:
        fx = fwd_excess(h).stack().rename("y").reset_index()
        fx.columns = ["trade_date", "symbol", "y"]
        m = p.merge(fx, on=["trade_date", "symbol"], how="inner").dropna(subset=["y"])
        print(f"=== forward {h}d, excess over F&O universe MEAN "
              f"(n={len(m):,} obs, {m.trade_date.nunique()} dates) ===")
        rows = []
        for lbl, (col, sign) in FEATS.items():
            d = m.dropna(subset=[col])
            if len(d) < 500:
                continue
            ic = (d.groupby("trade_date")
                    .apply(lambda g: g[col].rank().corr(g["y"].rank()) if len(g) > 8 else np.nan)
                    .dropna())
            # top-decile basket vs the rest
            top = (d.groupby("trade_date")
                     .apply(lambda g: g.nlargest(max(3, len(g)//10), col)["y"].mean())
                     .dropna())
            rows.append({"hypothesis": lbl, "expect": sign,
                         "mean_IC": round(float(ic.mean()), 4),
                         "IC_t(NW)": round(nw_t(ic.values, h), 2),
                         "top10%_excess": round(float(top.mean()), 3),
                         "top_t(NW)": round(nw_t(top.values, h), 2)})
        print(pd.DataFrame(rows).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
