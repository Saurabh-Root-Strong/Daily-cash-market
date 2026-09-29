"""
Load the 4-level stock classification into DuckDB.

Inputs
  config/stock_classification.csv            — fetched (scripts/fetch_stock_classification.py)
  config/stock_classification_overrides.csv  — hand-verified rows; WIN over fetched ones.
      Used for (a) stocks the source has not classified yet (fresh IPOs, renames) and
      (b) official labels that lag a corporate event (e.g. HEGAM kept its pre-demerger
      "Electrodes & Refractories" label). Every row carries its own source/notes.

ETFs are not companies: they get an asset-class label derived from the symbol.

Output
  table stock_classification (symbol PK, 4 levels, source, status, notes, fetched_at)
  view  v_stock_classification = stock_classification ⨝ v_sector_master, adding
        canonical_sector  — the rotation engine's sector (NULL if not a member)
        suggested_sector  — canonical_sector, else the crosswalk sector for the
                            stock's basic industry (config/industry_crosswalk.csv)
        in_rotation       — member of the rotation universe (non-ETF sector_master row)

This is additive: sector_master / canonical_sector / the memory engine are untouched.

Run:  python scripts/load_stock_classification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FETCHED   = PROJECT_ROOT / "config" / "stock_classification.csv"
OVERRIDES = PROJECT_ROOT / "config" / "stock_classification_overrides.csv"
CROSSWALK = PROJECT_ROOT / "config" / "industry_crosswalk.csv"
SEGMENTS  = PROJECT_ROOT / "config" / "stock_segments.csv"   # 5th level (hand-assigned)
SEG_COLS  = ["symbol", "segment", "secondary_segment", "market_focus",
             "mnc_subsidiary", "confidence", "evidence", "scope_basic_industry", "source", "as_of"]
# market_focus is a different dimension per segmented sub-sector; focus_type names it
# so a cross-sector query never mixes geography with end market or ownership.
FOCUS_TYPE = {
    "Pharmaceuticals": "geography", "Biotechnology": "geography",
    "Healthcare Research, Analytics & Technology": "geography",
    "Auto Components & Equipments": "end market",
    "Civil Construction": "ownership / business model",
    "Specialty Chemicals": "sales channel",
}
COLS = ["symbol", "company_name", "macro_sector", "sector", "industry",
        "basic_industry", "crosswalk_sector", "source", "status", "notes", "fetched_at"]


def _etf_label(sym: str) -> str:
    s = sym.upper()
    if "GOLD" in s:
        return "Gold ETF"
    if "SILVER" in s or "SILVR" in s:
        return "Silver ETF"
    if "LIQ" in s or "CASH" in s:
        return "Liquid / Money-market ETF"
    if any(k in s for k in ("MON100", "MONQ", "NASDAQ", "HNGSNG", "MAFANG", "MAHKTECH", "SP500")):
        return "International Equity ETF"
    if any(k in s for k in ("GILT", "BOND", "SDL", "BBETF", "EBBETF")):
        return "Debt ETF"
    return "Domestic Equity ETF"


def build() -> pd.DataFrame:
    from src.data.repository import query_dataframe

    fetched = pd.read_csv(FETCHED, dtype=str).fillna("")
    fetched["notes"] = ""
    df = fetched.reindex(columns=COLS, fill_value="")

    if OVERRIDES.exists():
        ov = pd.read_csv(OVERRIDES, dtype=str, comment="#").fillna("")
        ov["status"] = "manual"
        ov = ov.reindex(columns=COLS, fill_value="")
        keep_name = df.set_index("symbol")["company_name"]
        ov["company_name"] = [
            c or keep_name.get(s, "") for s, c in zip(ov["symbol"], ov["company_name"])
        ]
        df = pd.concat([df[~df["symbol"].isin(ov["symbol"])], ov], ignore_index=True)

    etfs = set(query_dataframe(
        "SELECT symbol FROM v_sector_master WHERE sector = 'ETF'", []
    )["symbol"]) | set(df.loc[df["status"] == "etf", "symbol"])
    names = df.set_index("symbol")["company_name"]
    etf_rows = pd.DataFrame({
        "symbol": sorted(etfs), "macro_sector": "ETF / Fund",
        "sector": "ETF / Fund", "industry": "Exchange Traded Fund",
        "basic_industry": [_etf_label(s) for s in sorted(etfs)],
        "source": "derived from symbol", "status": "etf", "notes": "", "fetched_at": "",
    })
    etf_rows["company_name"] = [names.get(s, "") for s in etf_rows["symbol"]]
    df = pd.concat([df[~df["symbol"].isin(etfs)], etf_rows], ignore_index=True)

    xw = pd.read_csv(CROSSWALK, comment="#", dtype=str).set_index("basic_industry")["canonical_sector"]
    df["crosswalk_sector"] = df["basic_industry"].map(xw).fillna("")
    df.loc[df["status"] == "etf", "crosswalk_sector"] = "ETF"
    return df.reindex(columns=COLS, fill_value="").fillna("").sort_values("symbol").reset_index(drop=True)


def load(df: pd.DataFrame) -> None:
    from src.data.repository import get_repository
    repo = get_repository()
    with repo._cm.connect() as conn:
        # Fully derived from the CSVs each run, so rebuild (schema can evolve).
        conn.execute("""
            CREATE OR REPLACE TABLE stock_classification (
                symbol VARCHAR PRIMARY KEY, company_name VARCHAR,
                macro_sector VARCHAR, sector VARCHAR, industry VARCHAR,
                basic_industry VARCHAR, crosswalk_sector VARCHAR, source VARCHAR,
                status VARCHAR, notes VARCHAR, fetched_at VARCHAR)
        """)
        conn.register("_sc_df", df[COLS])
        conn.execute("INSERT INTO stock_classification SELECT * FROM _sc_df")
        # 5th level: business segment inside a basic industry (Pharma so far).
        seg = (pd.read_csv(SEGMENTS, comment="#", dtype=str).fillna("")
               .reindex(columns=SEG_COLS, fill_value="")) if SEGMENTS.exists()             else pd.DataFrame(columns=SEG_COLS)
        seg["focus_type"] = seg["scope_basic_industry"].map(FOCUS_TYPE).fillna("")
        conn.execute("""
            CREATE OR REPLACE TABLE stock_segments (
                symbol VARCHAR PRIMARY KEY, segment VARCHAR, secondary_segment VARCHAR,
                market_focus VARCHAR, mnc_subsidiary VARCHAR, confidence VARCHAR,
                evidence VARCHAR, scope_basic_industry VARCHAR, source VARCHAR,
                as_of VARCHAR, focus_type VARCHAR)
        """)
        conn.register("_seg_df", seg)
        conn.execute("INSERT INTO stock_segments SELECT * FROM _seg_df")
        conn.execute("""
            CREATE OR REPLACE VIEW v_stock_classification AS
            SELECT c.symbol,
                   COALESCE(NULLIF(c.company_name, ''), m.company_name) AS company_name,
                   c.macro_sector, c.sector, c.industry, c.basic_industry,
                   m.sector AS canonical_sector,
                   COALESCE(m.sector, NULLIF(c.crosswalk_sector, '')) AS suggested_sector,
                   (m.symbol IS NOT NULL AND m.sector NOT IN ('ETF', 'Others')) AS in_rotation,
                   g.segment, g.secondary_segment, g.market_focus, g.focus_type,
                   (g.mnc_subsidiary = 'True') AS mnc_subsidiary,
                   g.confidence AS segment_confidence,
                   c.source, c.status, c.notes, c.fetched_at
            FROM stock_classification c
            LEFT JOIN v_sector_master m USING (symbol)
            LEFT JOIN stock_segments g USING (symbol)
        """)


def segment_health(df: pd.DataFrame) -> list[str]:
    """Warnings for the hand-assigned 5th level, which nothing else refreshes:
    drift  — NSE moved a segmented stock to a different basic industry
    gaps   — a stock in a segmented basic industry has no segment (new listing)"""
    if not SEGMENTS.exists():
        return []
    seg = pd.read_csv(SEGMENTS, comment="#", dtype=str).fillna("")
    now = df.set_index("symbol")["basic_industry"]
    msgs = []
    drift = seg[seg["symbol"].isin(now.index)]
    drift = drift[drift["scope_basic_industry"] != drift["symbol"].map(now)]
    for r in drift.itertuples():
        msgs.append(f"DRIFT {r.symbol}: segmented under '{r.scope_basic_industry}', "
                    f"now '{now[r.symbol]}' — review its segment")
    scoped = set(seg["scope_basic_industry"])
    gaps = df[df["basic_industry"].isin(scoped) & ~df["symbol"].isin(seg["symbol"])
              & (df["status"] != "etf")]
    for bi, g in gaps.groupby("basic_industry"):
        msgs.append(f"GAP {bi}: {len(g)} stock(s) without a segment: "
                    f"{', '.join(sorted(g['symbol'])[:12])}")
    return msgs


def main() -> None:
    df = build()
    load(df)
    # v_sector_master takes its sub-sector from stock_classification; rebuild it so a
    # fresh database switches from the raw vendor text to the official label.
    from src.analytics.sector_taxonomy import ensure_canonical_view
    ensure_canonical_view()
    for m in segment_health(df):
        print(f"[segments] {m}")
    print(df["status"].value_counts().to_string())
    print(f"Loaded {len(df)} rows into stock_classification "
          f"({df['macro_sector'].nunique()} macro sectors, {df['sector'].nunique()} sectors, "
          f"{df['industry'].nunique()} industries, {df['basic_industry'].nunique()} basic industries)")


if __name__ == "__main__":
    main()
