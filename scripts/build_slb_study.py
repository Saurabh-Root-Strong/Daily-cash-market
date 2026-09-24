"""Does SLB borrowing (India's only overnight short channel) forecast returns?

    python scripts/build_slb_study.py

Writes data/slb_study/panel.parquet (symbol-day features + forward returns) and
data/slb_study/summary.json (every test run, so the dashboard shows the grade the
code actually measured).

PRE-REGISTERED (fixed before any result was seen)
    signals   dtc      SLB outstanding / 20d avg delivery qty   (short-interest LEVEL)
              doi5     5-session change in outstanding / same base (fresh borrowing)
              fee_ann  annualised borrow fee, notional-weighted over 20 sessions
    horizons  5 and 21 sessions
    entry     close of t+1. SLB files publish after the 17:00 SLB session, so a
              close-t entry would trade on information not yet out.
    universe  top-300 by lagged 60d turnover (point in time); and the subset with
              SLB outstanding > 0
    target    forward return minus the same-day universe median
    controls  past 5d and 21d return, 20d volatility, log liquidity
    => 3 signals x 2 horizons x 2 universes = 12 candidates; Bonferroni at 12.

GUARDS
    - prev_close contradicted by its own open (439 rows, see CLAUDE.md) -> ratio_clean.
    - A split scales outstanding SLB quantity (NCL adjusts positions) while the 20d
      delivery base is still pre-split: symbol-days within 25 sessions after any
      prev_close adjustment are dropped from the signal.
    - A bonus / merger / buyback FORECLOSES every SLB position: outstanding collapses
      to ~0 for a mechanical reason, not a short cover. Same 25-session exclusion.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "slb_study")
DB = os.path.join(ROOT, "data", "market_data.duckdb")
H = (5, 21)
SIGNALS = ("dtc", "doi5", "fee_ann")
CONTROLS = ("r5", "r21", "vol20", "lturn")

PANEL_SQL = """
CREATE OR REPLACE TEMP TABLE sess AS
SELECT DISTINCT trade_date FROM slb_open_positions;

CREATE OR REPLACE TEMP TABLE px AS
SELECT trade_date, symbol, close_price, prev_close, open_price,
       coalesce(deliv_qty, 0) AS deliv_qty, turnover_lacs,
       CASE WHEN close_price / prev_close BETWEEN 0.5 AND 2.0 THEN close_price / prev_close
            WHEN open_price / prev_close NOT BETWEEN 0.5 AND 2.0
                 AND close_price / open_price BETWEEN 0.5 AND 2.0
              THEN close_price / open_price
            ELSE greatest(least(close_price / prev_close, 2.0), 0.5) END
       -- NSE does NOT adjust prev_close for dividends: add the payout back on the
       -- ex-date or every ex-div drop is booked as a loss.
       * (1 + CASE WHEN coalesce(v.dividend, 0) / close_price < 0.5
                   THEN coalesce(v.dividend, 0) / close_price ELSE 0 END) AS ratio_clean,
       CASE WHEN v.dividend > 0 THEN 1 ELSE 0 END AS is_ex
FROM daily_data
LEFT JOIN divs v USING (symbol, trade_date)
WHERE series = 'EQ' AND close_price > 0 AND open_price > 0 AND prev_close > 0
  AND trade_date >= DATE '2017-06-01';

CREATE OR REPLACE TEMP TABLE cum AS
SELECT *,
  exp(sum(ln(ratio_clean)) OVER w) AS cf,
  -- prev_close that disagrees with yesterday's close = NSE adjusted it for a
  -- corporate action (split / bonus / demerger / rights) on this ex-date.
  CASE WHEN abs(prev_close / nullif(lag(close_price) OVER w, 0) - 1) > 0.02 THEN 1 ELSE 0 END AS ca_day,
  avg(turnover_lacs) OVER (PARTITION BY symbol ORDER BY trade_date
                           ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING) AS adv60,
  avg(deliv_qty) OVER (PARTITION BY symbol ORDER BY trade_date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS dq20,
  stddev_samp(ln(ratio_clean)) OVER (PARTITION BY symbol ORDER BY trade_date
                                     ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol20
FROM px WINDOW w AS (PARTITION BY symbol ORDER BY trade_date);

-- returns on the FULL trading calendar, before the join to SLB sessions
CREATE OR REPLACE TEMP TABLE cum2 AS
SELECT *,
  cf / lag(cf, 5) OVER w - 1 AS r5,
  cf / lag(cf, 21) OVER w - 1 AS r21,
  lead(cf, 1) OVER w AS cf_e,
  lead(cf, 6) OVER w AS cf_5,
  lead(cf, 22) OVER w AS cf_21,
  -- ex-dates falling inside each holding window (entry close t+1 .. exit)
  lead(exn, 6) OVER w - lead(exn, 1) OVER w AS ex_in_5,
  lead(exn, 22) OVER w - lead(exn, 1) OVER w AS ex_in_21
FROM (SELECT *, sum(is_ex) OVER (PARTITION BY symbol ORDER BY trade_date) AS exn FROM cum) cum
WINDOW w AS (PARTITION BY symbol ORDER BY trade_date);

CREATE OR REPLACE TEMP TABLE oi AS
SELECT trade_date, symbol, sum(open_qty) AS oi_qty
FROM slb_open_positions GROUP BY 1, 2;

CREATE OR REPLACE TEMP TABLE fee AS
SELECT b.trade_date, b.symbol,
       sum(b.value) AS fee_value,                        -- rupees of fee
       sum(b.quantity * p.close_price) AS notional,      -- rupees of stock lent
       sum((b.value / b.quantity) / p.close_price * 365.0
           / greatest(date_diff('day', b.trade_date, b.expiry_date), 1)
           * b.quantity * p.close_price) AS ann_x_notional
FROM slb_bhavcopy b
JOIN px p ON p.symbol = b.symbol AND p.trade_date = b.trade_date
WHERE b.series_kind <> 'ROLLOVER' AND b.quantity > 0
  AND date_diff('day', b.trade_date, b.expiry_date) >= 10
  AND b.quantity * p.close_price >= 100000               -- ignore 1-share prints
GROUP BY 1, 2;

CREATE OR REPLACE TEMP TABLE base AS
SELECT c.trade_date, c.symbol, c.close_price, c.cf, c.dq20, c.vol20, c.adv60,
       c.deliv_qty, c.turnover_lacs, c.r5, c.r21, c.cf_e, c.cf_5, c.cf_21,
       c.ex_in_5, c.ex_in_21,
       max(c.ca_day) OVER (PARTITION BY c.symbol ORDER BY c.trade_date
                           ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS ca_recent,
       coalesce(o.oi_qty, 0) AS oi_qty,
       f.notional, f.ann_x_notional
FROM cum2 c
JOIN sess s USING (trade_date)
LEFT JOIN oi o USING (trade_date, symbol)
LEFT JOIN fee f USING (trade_date, symbol);

CREATE OR REPLACE TEMP TABLE feat AS
SELECT *,
  oi_qty / nullif(dq20, 0) AS dtc,
  (oi_qty - lag(oi_qty, 5) OVER w) / nullif(dq20, 0) AS doi5,
  sum(ann_x_notional) OVER r20 / nullif(sum(notional) OVER r20, 0) AS fee_ann,
  ln(adv60 + 1) AS lturn
FROM base
WINDOW w AS (PARTITION BY symbol ORDER BY trade_date),
       r20 AS (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW);
"""


def nw_t(x: np.ndarray, lags: int) -> float:
    """Newey-West t-stat of the mean of a (overlapping) time series."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def rank_z(s: pd.Series) -> pd.Series:
    r = s.rank(pct=True)
    return (r - r.mean()) / r.std(ddof=0)


def build_panel(con) -> pd.DataFrame:
    ca = pd.read_parquet(os.path.join(OUT, "corp_actions.parquet"))
    ca = ca.loc[(ca["dividend"] > 0) & ca["ex_date"].notna()]
    con.register("_ca", ca.rename(columns={"ex_date": "trade_date"})[["symbol", "trade_date", "dividend"]])
    con.execute("CREATE OR REPLACE TEMP TABLE divs AS SELECT symbol, trade_date, sum(dividend) AS dividend "
                "FROM _ca GROUP BY 1, 2")
    for stmt in PANEL_SQL.split(";\n"):
        if stmt.strip():
            con.execute(stmt)
    df = con.execute("""
        SELECT trade_date, symbol, close_price, oi_qty, oi_qty * close_price / 1e7 AS oi_cr,
               dq20, dtc, doi5, fee_ann, r5, r21, vol20, lturn, adv60, ca_recent,
               cf_5 / cf_e - 1 AS f5, cf_21 / cf_e - 1 AS f21, ex_in_5, ex_in_21,
               coalesce(m.sector, 'UNKNOWN') AS sector
        FROM feat LEFT JOIN v_sector_master m USING (symbol) WHERE trade_date >= DATE '2018-03-01'
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # point-in-time liquidity universe
    df["liq_rank"] = df.groupby("trade_date")["adv60"].rank(ascending=False, method="first")
    df = df.loc[df["liq_rank"] <= 300].copy()
    for h in H:
        df[f"x{h}"] = df[f"f{h}"] - df.groupby("trade_date")[f"f{h}"].transform("median")
        # sector-neutral variant: vs the same-day sector median where the sector has
        # >= 5 names in the universe, else vs the universe median
        g = df.groupby(["trade_date", "sector"])[f"f{h}"]
        sec_med = g.transform("median").where(g.transform("count") >= 5)
        df[f"s{h}"] = df[f"f{h}"] - sec_med.fillna(df[f"f{h}"] - df[f"x{h}"])
    return df


def test(df: pd.DataFrame, sig: str, h: int, target: str = "x") -> dict:
    d = df.loc[(df["ca_recent"] == 0) & df[sig].notna() & df[f"x{h}"].notna()
               & df[list(CONTROLS)].notna().all(axis=1)].copy()
    d[f"x{h}"] = d[f"{target}{h}"]
    ics, fm, spread, dates = [], [], [], []
    for dt, g in d.groupby("trade_date"):
        if len(g) < 30 or g[sig].nunique() < 5:
            continue
        ics.append(g[sig].rank().corr(g[f"x{h}"].rank()))
        X = np.column_stack([np.ones(len(g)), rank_z(g[sig])] + [rank_z(g[c]) for c in CONTROLS])
        y = g[f"x{h}"].clip(g[f"x{h}"].quantile(.01), g[f"x{h}"].quantile(.99)).values
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fm.append(beta[1])
        q = pd.qcut(g[sig].rank(method="first"), 5, labels=False)
        spread.append(g.loc[q == 4, f"x{h}"].mean() - g.loc[q == 0, f"x{h}"].mean())
        dates.append(dt)
    ics, fm, spread = map(np.array, (ics, fm, spread))
    yrs = pd.Series(ics, index=pd.to_datetime(dates)).groupby(lambda x: x.year).mean()
    return {
        "signal": sig, "h": h, "n_dates": int(len(ics)), "n_obs": int(len(d)),
        "ic": float(np.nanmean(ics)), "ic_t": float(nw_t(ics, h)),
        "fm_beta_bps": float(np.nanmean(fm) * 1e4), "fm_t": float(nw_t(fm, h)),
        "q5_q1_bps": float(np.nanmean(spread) * 1e4), "q5_q1_t": float(nw_t(spread, h)),
        "ic_by_year": {int(k): round(float(v), 4) for k, v in yrs.items()},
        "years_same_sign": int((np.sign(yrs) == np.sign(np.nanmean(ics))).sum()),
        "n_years": int(len(yrs)),
    }


def diagnostics(df: pd.DataFrame, con) -> dict:
    """Checks that decide whether the surviving tests are usable: where in the
    distribution the effect lives, whether a few collapses carry it, whether it is
    just reverse cash-futures arbitrage, and whether a short could pay its costs."""
    d = df.loc[(df["oi_qty"] > 0) & (df["ca_recent"] == 0) & df["r21"].notna()].copy()
    out: dict = {"ladders": {}, "haircut": {}, "per_name": {}, "basis_control": {}}
    for sig in ("doi5", "dtc"):
        x = d.loc[d[sig].notna()].copy()
        x["q"] = x.groupby("trade_date")[sig].transform(
            lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
        for h in H:
            y = x.loc[x[f"s{h}"].notna()]
            lad = y.groupby(["trade_date", "q"])[f"s{h}"].mean().unstack()
            top = lad[9] - lad[[3, 4, 5, 6]].mean(axis=1)
            yr = (lad[9] - lad[[3, 4, 5, 6]].mean(axis=1)).groupby(lad.index.year).mean()
            out["ladders"][f"{sig}_{h}"] = {
                "decile_bps": [round(float(v) * 1e4, 1) for v in lad.mean()],
                "d10_vs_mid_bps": round(float(top.mean()) * 1e4, 1),
                "d10_vs_mid_t": round(float(nw_t(top.values, h)), 2),
                "d10_vs_mid_by_year_bps": {int(k): round(float(v) * 1e4, 0) for k, v in yr.items()},
            }
        # a few collapses (DHFL, YESBANK, JETAIRWAYS) could carry the top decile:
        # remove the worst-k names from EVERY decile, not only the top one.
        y = x.loc[x["s21"].notna()]
        hc = {}
        for k in (0, 5, 10, 20):
            row = {}
            for q in (4, 5, 6, 9):
                t = y.loc[y["q"] == q]
                worst = t.groupby("symbol")["s21"].sum().nsmallest(k).index
                row[f"D{q + 1}"] = round(float(t.loc[~t["symbol"].isin(worst), "s21"].mean()) * 1e4, 1)
            hc[k] = row
        out["haircut"][sig] = hc
        pn = y.groupby(["symbol", "q"])["s21"].mean().unstack()
        g = (pn[9] - pn[list(range(1, 9))].mean(axis=1)).dropna()
        out["per_name"][sig] = {"n": int(len(g)), "mean_bps": round(float(g.mean()) * 1e4, 1),
                                "median_bps": round(float(g.median()) * 1e4, 1),
                                "share_negative": round(float((g < 0).mean()), 3),
                                "t": round(float(g.mean() / g.std() * np.sqrt(len(g))), 2)}
        top10 = x.loc[x["q"] == 9, "fee_ann"]
        out.setdefault("d10_fee_ann_median_pct", {})[sig] = round(float(top10.median()) * 100, 2)

    # reverse cash-futures arbitrage: borrow + sell cash, buy the cheap future.
    fut = con.execute("""
        WITH f AS (SELECT trade_date, symbol, close_price AS fc, expiry_date,
                          row_number() OVER (PARTITION BY trade_date, symbol ORDER BY expiry_date) AS rn
                   FROM fno_bhavcopy WHERE instrument = 'FUTSTK' AND close_price > 0)
        SELECT trade_date, symbol, fc, expiry_date FROM f WHERE rn = 1""").df()
    fut["trade_date"] = pd.to_datetime(fut["trade_date"])
    b = d.merge(fut, on=["trade_date", "symbol"])
    dte = (pd.to_datetime(b["expiry_date"]) - b["trade_date"]).dt.days.clip(lower=1)
    b["basis_ann"] = (b["fc"] / b["close_price"] - 1) * 365 / dte
    b = b.loc[b["basis_ann"].between(-2, 2)]
    out["basis_control"]["from"] = str(b["trade_date"].min().date())
    for sig in ("doi5", "dtc"):
        out["basis_control"][f"corr_{sig}_basis"] = round(float(b.groupby("trade_date").apply(
            lambda g: g[sig].rank().corr(g["basis_ann"].rank())).mean()), 3)
        for h in H:
            res = {}
            for tag, ctrl in (("without", CONTROLS), ("with", CONTROLS + ("basis_ann",))):
                betas = []
                for _, g in b.loc[b[sig].notna() & b[f"s{h}"].notna()].groupby("trade_date"):
                    if len(g) < 30:
                        continue
                    X = np.column_stack([np.ones(len(g)), rank_z(g[sig])] + [rank_z(g[c]) for c in ctrl])
                    y = g[f"s{h}"].clip(g[f"s{h}"].quantile(.01), g[f"s{h}"].quantile(.99)).values
                    betas.append(np.linalg.lstsq(X, y, rcond=None)[0][1])
                betas = np.array(betas)
                res[tag] = {"bps": round(float(betas.mean()) * 1e4, 1), "t": round(float(nw_t(betas, h)), 2)}
            out["basis_control"][f"{sig}_{h}"] = res
    # The page shows three flag groups (both / fresh only / heavy only). Grade each
    # against UNFLAGGED names, so the red "both" label is measured, not assumed.
    x = d.loc[d["doi5"].notna() & d["dtc"].notna() & d["s21"].notna()].copy()
    g = x.groupby("trade_date")
    fr, hv = g["doi5"].rank(pct=True) > 0.9, g["dtc"].rank(pct=True) > 0.9
    x["grp"] = np.select([fr & hv, fr, hv], ["both", "fresh_only", "heavy_only"], "none")
    base = x.loc[x["grp"] == "none"].groupby("trade_date")["s21"].mean()
    out["flag_groups"] = {}
    for k in ("both", "fresh_only", "heavy_only"):
        m = x.loc[x["grp"] == k].groupby("trade_date")["s21"].mean()
        diff = (m - base.reindex(m.index)).dropna()
        yr = diff.groupby(diff.index.year).mean()
        out["flag_groups"][k] = {
            "share_pct": round(float((x["grp"] == k).mean()) * 100, 1),
            "vs_unflagged_bps": round(float(diff.mean()) * 1e4, 1),
            "t": round(float(nw_t(diff.values, 21)), 2),
            "years_negative": int((yr < 0).sum()), "n_years": int(len(yr)),
        }

    out["fee_ann_quantiles_pct"] = {str(k): round(float(v) * 100, 2)
                                    for k, v in d["fee_ann"].quantile([.25, .5, .75, .9]).items()}
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect(DB, read_only=True)
    df = build_panel(con)
    df.to_parquet(os.path.join(OUT, "panel.parquet"), index=False)
    print(f"panel {len(df):,} symbol-days, {df['trade_date'].nunique()} sessions, "
          f"{int((df['oi_qty'] > 0).sum()):,} with SLB outstanding", flush=True)

    results = []
    variants = (("div_adj", "x", None), ("no_exdate", "x", "ex"), ("sector_neutral", "s", None))
    for var, target, drop in variants:
      for uni, sub in (("top300", df), ("slb_active", df.loc[df["oi_qty"] > 0])):
        for sig in SIGNALS:
            for h in H:
                s = sub if drop is None else sub.loc[sub[f"ex_in_{h}"].fillna(0) == 0]
                r = test(s, sig, h, target)
                r["universe"], r["variant"] = uni, var
                results.append(r)
                print(f"{var:14s} {uni:10s} {sig:8s} h={h:2d}  IC {r['ic']:+.4f} (t {r['ic_t']:+.2f})  "
                      f"FM {r['fm_beta_bps']:+.1f}bps (t {r['fm_t']:+.2f})  "
                      f"Q5-Q1 {r['q5_q1_bps']:+.1f}bps (t {r['q5_q1_t']:+.2f})  "
                      f"yrs {r['years_same_sign']}/{r['n_years']}  n={r['n_dates']}", flush=True)
    diag = diagnostics(df, con)
    print(json.dumps(diag, indent=1), flush=True)
    summary = {"built_through": str(df["trade_date"].max().date()),
               "n_candidates": len(SIGNALS) * len(H) * 2, "results": results,
               "diagnostics": diag}
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
