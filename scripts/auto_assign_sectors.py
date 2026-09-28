"""
Auto-assign rotation sectors to stocks that have no manual classification.

Why: sector_master is seeded from 15 NSE sectoral index lists, so every new listing,
SME→main-board migration and rename arrived WITHOUT a sector until someone added it
by hand (178 such EQ names listed Jul–Sep 2026 alone).

How:
  1. Official 4-level labels come from config/stock_classification.csv
     (scripts/fetch_stock_classification.py --universe all).
  2. basic_industry → rotation sector via config/industry_crosswalk.csv (reviewed).
  3. Only LIQUID names join the rotation universe. The rotation engine aggregates
     equal-weight medians across constituents, so adding ~1,400 micro-caps would turn
     every sector score into a small-cap signal. The bar matches the existing members
     (their 5th percentile is ~₹6 Cr/day): avg turnover ≥ ₹5 Cr/day over the last 60
     sessions AND ≥ 20 sessions traded (keeps IPO listing-day frenzy out for a month).
  4. Writes config/sector_auto.csv. Precedence when applied (orchestrator):
     _SYMBOL_OVERRIDES > config/sector_overrides.csv (manual) > sector_auto.csv > rules.
     Manual symbols are never written here.

Membership is sticky: once auto-added, a name stays (the rotation's own ₹1 Cr/day
filter already drops it on days it is illiquid) — avoids membership churn.

Run:  python scripts/auto_assign_sectors.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASSIFIED = PROJECT_ROOT / "config" / "stock_classification.csv"
CROSSWALK  = PROJECT_ROOT / "config" / "industry_crosswalk.csv"
MANUAL     = PROJECT_ROOT / "config" / "sector_overrides.csv"
AUTO       = PROJECT_ROOT / "config" / "sector_auto.csv"

MIN_AVG_CR      = 5.0   # ₹ Cr/day, 60-session average
MIN_SESSIONS    = 20
LOOKBACK        = 60


def _liquidity() -> pd.DataFrame:
    from src.data.repository import query_dataframe
    return query_dataframe(f"""
        WITH l AS (SELECT DISTINCT trade_date FROM daily_data WHERE series = 'EQ'
                   ORDER BY trade_date DESC LIMIT {LOOKBACK})
        SELECT symbol, AVG(turnover_lacs) / 100.0 AS avg_cr, COUNT(*) AS sessions
        FROM daily_data WHERE series = 'EQ' AND trade_date IN (SELECT * FROM l)
        GROUP BY symbol
    """, [])


def _seeded_symbols() -> set[str]:
    """sector_master rows that did NOT come from this script (index seed + manual)."""
    from src.data.repository import query_dataframe
    master = set(query_dataframe("SELECT symbol FROM sector_master", [])["symbol"])
    prev_auto = set(pd.read_csv(AUTO, comment="#")["symbol"]) if AUTO.exists() else set()
    return master - prev_auto


def build() -> pd.DataFrame:
    cls = pd.read_csv(CLASSIFIED, dtype=str).fillna("")
    xw  = pd.read_csv(CROSSWALK, comment="#", dtype=str).fillna("")
    manual = set(pd.read_csv(MANUAL, comment="#", dtype=str)["symbol"])
    seeded = _seeded_symbols()
    prev_auto = (pd.read_csv(AUTO, comment="#", dtype=str).fillna("")
                 if AUTO.exists() else pd.DataFrame(columns=["symbol"]))

    liq = _liquidity()
    cand = cls[~cls["symbol"].isin(seeded | manual)].merge(liq, on="symbol", how="left")
    cand["avg_cr"] = cand["avg_cr"].fillna(0.0)
    cand["sessions"] = cand["sessions"].fillna(0).astype(int)

    xmap = xw.set_index("basic_industry")
    rows = []
    for r in cand.itertuples():
        sticky = r.symbol in set(prev_auto["symbol"])
        liquid = r.avg_cr >= MIN_AVG_CR and r.sessions >= MIN_SESSIONS
        if r.status == "etf":
            canon, review, note = "ETF", False, "fund page"
        elif r.status != "ok":
            continue
        elif r.basic_industry in xmap.index:
            x = xmap.loc[r.basic_industry]
            canon, review, note = x["canonical_sector"], x["review"] == "True", x["note"]
        else:
            # A basic industry never seen among classified stocks: hold it out of
            # rotation and surface it for a crosswalk decision.
            canon, review, note = "", True, "basic industry not in crosswalk — add a row"
        if not canon or not (liquid or sticky or canon == "ETF"):
            continue
        rows.append({
            "symbol": r.symbol,
            "sector": r.sector or "ETF",            # raw sector = NSE sector (provenance)
            "industry": r.basic_industry or "ETF / Index Fund",
            "category": r.industry,
            "canonical_sector": canon,
            "review": review,
            "avg_cr_60d": round(r.avg_cr, 1),
            "sessions_60d": r.sessions,
            "notes": note,
            "assigned_at": (prev_auto.set_index("symbol")["assigned_at"].get(r.symbol)
                            if sticky and "assigned_at" in prev_auto else date.today().isoformat()),
        })
    out = pd.DataFrame(rows)
    # Sticky carry-over: a previously auto-assigned symbol that this run did not
    # re-derive (ETFs are skipped by the fetcher once in sector_master; a page may
    # fail to fetch) keeps its old row instead of silently dropping out.
    if len(prev_auto):
        keep = prev_auto[~prev_auto["symbol"].isin(out["symbol"]) & ~prev_auto["symbol"].isin(manual)]
        out = pd.concat([out, keep.reindex(columns=out.columns)], ignore_index=True)
    return out.sort_values(["canonical_sector", "symbol"]).reset_index(drop=True)


def main() -> None:
    df = build()
    hdr = ("# AUTO-GENERATED by scripts/auto_assign_sectors.py — do not hand-edit.\n"
           "# To change a stock, add it to config/sector_overrides.csv (manual rows win).\n")
    AUTO.write_text(hdr + df.to_csv(index=False), encoding="utf-8")
    stocks = df[df["canonical_sector"] != "ETF"]
    print(f"[auto] {len(stocks)} stocks + {len(df) - len(stocks)} ETFs → {AUTO.name}")
    print(stocks["canonical_sector"].value_counts().to_string())
    rev = stocks[stocks["review"]]
    if len(rev):
        print(f"\n[auto] {len(rev)} need a human look (review=True):")
        print(rev[["symbol", "industry", "canonical_sector", "avg_cr_60d", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
