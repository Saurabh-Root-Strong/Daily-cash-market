"""
READ-ONLY factor diagnostic for Sector Performance / Sector Rotation.

Question it answers: for each signal factor, how well does it predict FUTURE
cross-sectional sector returns, and at what horizon? Pure walk-forward — every
factor at day D uses only data up to D; forward returns look D+1..D+h.

Metrics per (factor, horizon):
  IC    = mean daily Spearman rank-corr between factor and forward return
  ICIR  = mean(IC) / std(IC)         (consistency; >0.3 is decent, >0.5 strong)
  t     = ICIR * sqrt(n_days)        (significance)
  hit   = fraction of days IC > 0
  spr   = avg (top-tercile fwd ret − bottom-tercile fwd ret), in %

Two return targets:
  ABSOLUTE        — sector's own forward return
  vs NIFTY        — sector forward return − Nifty50 forward return (true rotation)

Nothing is written. Safe to delete.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:                                  # Windows console/file defaults to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.data.repository import query_dataframe  # noqa: E402

MIN_TURNOVER_LACS = 100.0          # ~1 Cr/stock/day liquidity floor (matches UI default)
HORIZONS          = [1, 3, 5, 10, 20]
BASELINE_WIN      = 100             # trailing window for DV baseline (excludes today)
MIN_BASELINE      = 20             # need >=20 prior obs before a factor is valid
MIN_SECTORS_DAY   = 6              # need >=6 sectors cross-section to compute daily IC


# ── Load the per-sector daily panel ──────────────────────────────────────────
def load_panel() -> pd.DataFrame:
    sql = """
        SELECT s.sector, b.trade_date,
            SUM(b.turnover_lacs * b.deliv_per / 100.0) / 100.0          AS daily_dv_cr,
            SUM(b.deliv_per * b.turnover_lacs)
                / NULLIF(SUM(b.turnover_lacs), 0)                        AS wtd_deliv_pct,
            SUM(b.turnover_lacs * (b.close_price - b.prev_close)
                    / NULLIF(b.prev_close, 0) * 100)
                / NULLIF(SUM(CASE WHEN b.prev_close > 0
                             THEN b.turnover_lacs END), 0)              AS wtd_ret_pct
        FROM daily_data b
        INNER JOIN sector_master s ON b.symbol = s.symbol
        WHERE b.series IN ('EQ', 'SM', 'ST')
          AND s.sector NOT IN ('ETF', 'Others')
          AND b.turnover_lacs >= ?
        GROUP BY s.sector, b.trade_date
        ORDER BY s.sector, b.trade_date
    """
    df = query_dataframe(sql, [MIN_TURNOVER_LACS])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def load_nifty() -> pd.DataFrame:
    try:
        df = query_dataframe(
            "SELECT trade_date, close_val, pct_chg FROM index_data "
            "WHERE index_name = 'Nifty 50' ORDER BY trade_date",
            [],
        )
    except Exception as exc:
        print(f"[warn] nifty load failed ({exc}); market-relative analysis skipped")
        return pd.DataFrame()
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if df["pct_chg"].notna().any():
        df["nret"] = df["pct_chg"].astype(float)
    else:
        df["nret"] = df["close_val"].astype(float).pct_change() * 100
    return df[["trade_date", "nret"]]


# ── Factor + forward-return engineering (per sector, walk-forward) ────────────
def _slope_norm(arr: np.ndarray) -> float:
    y = arr[~np.isnan(arr)]
    if len(y) < 8:
        return np.nan
    x = np.arange(len(y), dtype=float)
    m = np.polyfit(x, y, 1)[0]
    mu = y.mean() if y.mean() != 0 else 1.0
    return float(m / mu * 100)


def _trailing_pct(arr: np.ndarray) -> np.ndarray:
    """Percentile rank of each point vs its prior BASELINE_WIN values (exclusive)."""
    out = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        lo = max(0, i - BASELINE_WIN)
        prior = arr[lo:i]
        prior = prior[~np.isnan(prior)]
        if len(prior) >= MIN_BASELINE and not np.isnan(arr[i]):
            out[i] = (prior < arr[i]).mean()
    return out


def build_factors(panel: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sector, g in panel.groupby("sector", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)
        dv = g["daily_dv_cr"]

        mean100 = dv.shift(1).rolling(BASELINE_WIN, min_periods=MIN_BASELINE).mean()
        std100  = dv.shift(1).rolling(BASELINE_WIN, min_periods=MIN_BASELINE).std()
        logdv   = np.log(dv.clip(lower=1e-6))
        lmean   = logdv.shift(1).rolling(BASELINE_WIN, min_periods=MIN_BASELINE).mean()
        lstd    = logdv.shift(1).rolling(BASELINE_WIN, min_periods=MIN_BASELINE).std()

        g["dv_ratio"]   = dv / mean100
        g["z_score"]    = (dv - mean100) / std100                 # current parametric z
        g["z_log"]      = (logdv - lmean) / lstd                  # log-transformed z
        g["rank_pct"]   = _trailing_pct(dv.values)               # robust percentile
        g["dv5d"]       = dv.rolling(5).mean() / mean100
        g["deliv_slope20"] = (
            g["wtd_deliv_pct"].rolling(20).apply(_slope_norm, raw=True)
        )

        lr = np.log1p(g["wtd_ret_pct"] / 100.0)
        cr = lr.cumsum()
        g["price_mom_1w"] = np.expm1(cr - cr.shift(5)) * 100      # trailing 5d
        g["price_mom_2w"] = np.expm1(cr - cr.shift(10)) * 100     # trailing 10d
        g["_cr"] = cr
        for h in HORIZONS:
            g[f"fwd_{h}"] = np.expm1(cr.shift(-h) - cr) * 100     # D+1..D+h
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)

    # Nifty forward returns + trailing momentum for RS
    if not nifty.empty:
        nf = nifty.sort_values("trade_date").reset_index(drop=True)
        nlr = np.log1p(nf["nret"] / 100.0)
        ncr = nlr.cumsum()
        nf["n_mom_1w"] = np.expm1(ncr - ncr.shift(5)) * 100
        nf["n_mom_2w"] = np.expm1(ncr - ncr.shift(10)) * 100
        for h in HORIZONS:
            nf[f"n_fwd_{h}"] = np.expm1(ncr.shift(-h) - ncr) * 100
        out = out.merge(
            nf[["trade_date", "n_mom_1w", "n_mom_2w"] +
               [f"n_fwd_{h}" for h in HORIZONS]],
            on="trade_date", how="left",
        )
        out["rs_1w"] = out["price_mom_1w"] - out["n_mom_1w"]     # relative strength
        out["rs_2w"] = out["price_mom_2w"] - out["n_mom_2w"]
    else:
        out["rs_1w"] = out["price_mom_1w"]
        out["rs_2w"] = out["price_mom_2w"]
    return out


# ── IC + spread evaluation ───────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, factor: str, fwd: str) -> dict:
    sub = df[["trade_date", factor, fwd]].dropna()
    if sub.empty:
        return {}
    daily_ic, daily_spr = [], []
    for _, day in sub.groupby("trade_date"):
        if len(day) < MIN_SECTORS_DAY:
            continue
        ic = day[factor].corr(day[fwd], method="spearman")
        if not np.isnan(ic):
            daily_ic.append(ic)
        n3 = max(1, len(day) // 3)
        ranked = day.sort_values(factor)
        bot = ranked.head(n3)[fwd].mean()
        top = ranked.tail(n3)[fwd].mean()
        daily_spr.append(top - bot)
    if len(daily_ic) < 10:
        return {}
    ic = np.array(daily_ic)
    icir = ic.mean() / ic.std() if ic.std() > 0 else 0.0
    return {
        "ic":   ic.mean(),
        "icir": icir,
        "t":    icir * np.sqrt(len(ic)),
        "hit":  (ic > 0).mean(),
        "spr":  np.nanmean(daily_spr),
        "n":    len(ic),
    }


FACTORS = [
    "dv_ratio", "dv5d", "z_score", "z_log", "rank_pct",
    "deliv_slope20", "price_mom_1w", "rs_1w", "price_mom_2w", "rs_2w",
]


def print_block(df: pd.DataFrame, target_prefix: str, title: str) -> None:
    print(f"\n{'='*96}\n{title}\n{'='*96}")
    head = f"{'factor':<15}"
    for h in HORIZONS:
        head += f"  h={h:<2d} IC/ICIR/spr%   "
    print(head)
    print("-" * 96)
    for f in FACTORS:
        line = f"{f:<15}"
        for h in HORIZONS:
            r = evaluate(df, f, f"{target_prefix}{h}")
            if r:
                line += f"  {r['ic']:+.3f}/{r['icir']:+.2f}/{r['spr']:+.2f}".ljust(19)
            else:
                line += f"  {'—':<17}"
        print(line)


def main() -> None:
    panel = load_panel()
    nifty = load_nifty()
    sectors = panel["sector"].nunique()
    d0, d1 = panel["trade_date"].min(), panel["trade_date"].max()
    days = panel["trade_date"].nunique()
    print(f"\nDB panel: {sectors} sectors | {days} trading days "
          f"| {d0.date()} → {d1.date()} | min_turnover={MIN_TURNOVER_LACS:.0f} lacs "
          f"| nifty_rows={len(nifty)}")

    df = build_factors(panel, nifty)

    # ABSOLUTE forward returns
    print_block(df, "fwd_", "ABSOLUTE forward sector returns  (IC = rank-corr factor→future return)")

    # MARKET-RELATIVE forward returns (sector fwd − nifty fwd)
    if not nifty.empty:
        for h in HORIZONS:
            df[f"rel_{h}"] = df[f"fwd_{h}"] - df[f"n_fwd_{h}"]
        print_block(df, "rel_", "MARKET-RELATIVE forward returns  (sector fwd − Nifty50 fwd = true rotation)")

    # Headline comparisons
    print(f"\n{'='*96}\nKEY COMPARISONS (absolute, horizon=5d unless noted)\n{'='*96}")
    def g(f, h, pre="fwd_"):
        r = evaluate(df, f, f"{pre}{h}")
        return r.get("ic", float("nan")), r.get("icir", float("nan"))
    for label, a, b, h in [
        ("rank_pct vs z_score   @5d", "rank_pct", "z_score", 5),
        ("z_log    vs z_score   @5d", "z_log",    "z_score", 5),
        ("rs_1w    vs price_1w  @5d", "rs_1w",    "price_mom_1w", 5),
    ]:
        ia, ra = g(a, h); ib, rb = g(b, h)
        win = "→ " + (a if abs(ia) > abs(ib) else b) + " wins"
        print(f"  {label}:  {a}={ia:+.3f}(IR{ra:+.2f})  vs  {b}={ib:+.3f}(IR{rb:+.2f})   {win}")

    print("\n  Best horizon for delivery flow (dv5d, by |IC|):")
    best = max(HORIZONS, key=lambda h: abs(evaluate(df, "dv5d", f"fwd_{h}").get("ic", 0)))
    for h in HORIZONS:
        r = evaluate(df, "dv5d", f"fwd_{h}")
        star = "  <== peak" if h == best else ""
        if r:
            print(f"    h={h:<2d}: IC={r['ic']:+.3f}  ICIR={r['icir']:+.2f}  "
                  f"t={r['t']:+.1f}  spread={r['spr']:+.2f}%{star}")
    print()


if __name__ == "__main__":
    main()
