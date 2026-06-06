"""
Canonical sector taxonomy.

PROBLEM
-------
The raw `sector_master.sector` column is a fragmented merge of ≥2 vendor schemes:
the oil majors (RELIANCE/ONGC/BPCL) sit in a mixed "Energy" bucket (power + solar +
oil) while small explorers get a separate thin "Oil & Gas"; IT is split across
"Information Technology" and "IT Industry"; autos across "Automobile" and
"Automobile & Ancillaries"; etc. Sector-rotation ranking across these inconsistent
buckets is economically meaningless.

SOLUTION
--------
Map at the FINE `industry` level (211 clean values) to ~20 canonical sectors aligned
with the NSE sectoral indices plus the practical Indian rotation themes (Capital
Goods, Defence, Infrastructure, Chemicals, Telecom, Logistics). The industry column
disambiguates the mixed raw sectors — e.g. "Energy" splits into Power Generation →
Power & Utilities and Refineries/Gas → Oil & Gas.

`canonical_sector(industry, raw_sector)` is a pure function: ordered substring rules
(most specific first), with `raw_sector` only as a last-resort fallback when industry
is missing/unknown. Run `audit()` to print the full industry→canonical mapping.
"""
from __future__ import annotations

from typing import Optional

__all__ = ["canonical_sector", "CANONICAL_SECTORS", "audit", "populate_canonical_sectors"]

# The canonical universe (stable list — used for validation / UI ordering).
CANONICAL_SECTORS = [
    "IT", "Banking", "Financial Services", "Automobile", "Pharma & Healthcare",
    "FMCG", "Oil & Gas", "Power & Utilities", "Metals & Mining", "Chemicals",
    "Cement & Building Materials", "Capital Goods", "Defence", "Infrastructure",
    "Realty", "Consumer Durables", "Gems & Jewellery", "Telecom",
    "Media & Entertainment", "Consumer Services", "Textiles", "Logistics",
    "Renewables", "Diversified",
]

# Per-symbol corrections that OVERRIDE industry/sector mapping — for data errors
# in the vendor `industry` column. Each is web-verified against the actual business.
_SYMBOL_OVERRIDES: dict[str, str] = {
    # "Amir Chand Jagdish Kumar (Exports)" is a BASMATI RICE exporter (FMCG), but
    # the vendor mislabels its industry as "Jewellery". Verified via NSE/Screener.
    "AMIRCHAND": "FMCG",

    # ── Renewables mislabelled "Power Generation/Distribution" by the vendor ──
    # (pure solar/wind/green plays — verified by business). Routed to Renewables.
    "ADANIGREEN": "Renewables",   # Adani Green Energy — renewable IPP
    "ACMESOLAR":  "Renewables",   # ACME Solar Holdings — solar generation
    "SUZLON":     "Renewables",   # Suzlon Energy — wind turbines
    "INOXWIND":   "Renewables",   # Inox Wind — wind turbines
    "KPIGREEN":   "Renewables",   # KPI Green Energy — solar
    "WAAREEENER": "Renewables",   # Waaree Energies — solar module manufacturing
    "PREMIERENE": "Renewables",   # Premier Energies — solar cell/module manufacturing

    # ── Conventional power generator mis-bucketed as fuels ───────────────────
    "NLCINDIA":   "Power & Utilities",   # NLC India — lignite mining + power generation

    # ── Power/T&D EPC contractors — projects, not utilities → Infrastructure ──
    "KEC":        "Infrastructure",      # KEC International — T&D EPC
    "KALPATARU":  "Infrastructure",      # Kalpataru Projects — diversified EPC
    "POWERMECH":  "Infrastructure",      # Power Mech Projects — power plant erection/O&M

    # ── Refrigerant gases / coal-logistics conglomerate, not a power generator ─
    "REFEX":      "Diversified",         # Refex Industries

    # ── Power value-chain (user choice): fold Equipment + Financing INTO Power ─
    # Equipment makers (capex-cycle) and power-sector NBFCs (lending) are grouped
    # with the utilities so the Power sector matches the full value chain. POWERINDIA
    # (Hitachi Energy) / GVT&D (GE Vernova) / JYOTISTRUC enter via the "Power T&D
    # Equipment" industry rule above; the rest need symbol overrides:
    "BHEL":       "Power & Utilities",   # Bharat Heavy Electricals — boilers/turbines
    "CGPOWER":    "Power & Utilities",   # CG Power — transformers / switchgear
    "PFC":        "Power & Utilities",   # Power Finance Corporation — power lender
    "RECLTD":     "Power & Utilities",   # REC Ltd — rural electrification lender
}

# Ordered (substring, canonical) rules. FIRST match wins, so the order encodes
# precedence: narrow exceptions are placed before the broad keyword that would
# otherwise swallow them (e.g. "Power T&D Equipment" before "Power Generation";
# "Pharmacy Retail" before "Retail"; "Investment Banking" before "Bank ").
# All comparisons are case-insensitive substring tests on the `industry` string.
_RULES: list[tuple[str, str]] = [
    # ── Non-tradable buckets preserved so rotation's NOT IN ('ETF','Others') holds ─
    ("etf",                      "ETF"),
    ("index fund",               "ETF"),
    # ── Gems & Jewellery — its own sector (gold-linked, distinct demand cycle) ─
    # Pre-empts Consumer Durables. Covers retail jewellers, diamond/studded
    # jewellery makers/exporters, and gemological certification. Sub-industries
    # (Jewellery / Diamond & Jewellery / Jewellery & Accessories) are retained.
    ("jewell",                   "Gems & Jewellery"),
    ("gemolog",                  "Gems & Jewellery"),
    # ── Narrow exceptions that must pre-empt broader rules below ───────────────
    ("defence electronics",      "Defence"),
    ("defence components",       "Defence"),
    ("aerospace components",     "Defence"),
    ("munitions",                "Defence"),
    ("shipbuilding",             "Defence"),
    ("defence",                  "Defence"),
    ("pharmacy retail",          "Pharma & Healthcare"),
    ("investment banking",       "Financial Services"),
    ("power t&d equipment",      "Power & Utilities"),  # value-chain: power equip in Power
    ("power epc",                "Infrastructure"),
    ("water epc",                "Infrastructure"),
    ("petroleum specialties",    "Chemicals"),
    ("petrochem",                "Chemicals"),

    # ── Information Technology ─────────────────────────────────────────────────
    ("it - ", "IT"), ("it services", "IT"), ("it hardware", "IT"),
    ("it distribution", "IT"), ("bpo", "IT"), ("software", "IT"),
    ("cloud", "IT"), (" ems", "IT"), ("ems", "IT"), ("hardware", "IT"),
    ("adtech", "IT"), ("edtech", "IT"), ("online services", "IT"),
    ("financial technology", "Financial Services"),
    ("payment gateway & fintech", "Financial Services"),

    # ── Telecom ────────────────────────────────────────────────────────────────
    ("telecom", "Telecom"),

    # ── Banking vs broader financials ─────────────────────────────────────────
    ("bank - ", "Banking"), ("banks", "Banking"),
    ("nbfc", "Financial Services"), ("finance", "Financial Services"),
    ("insurance", "Financial Services"), ("brokerage", "Financial Services"),
    ("broking", "Financial Services"), ("asset management", "Financial Services"),
    ("wealth management", "Financial Services"), ("payments", "Financial Services"),
    ("housing finance", "Financial Services"), ("exchange", "Financial Services"),
    ("ratings", "Financial Services"), ("diversified financial", "Financial Services"),

    # ── Automobile ─────────────────────────────────────────────────────────────
    ("auto", "Automobile"), ("tyres", "Automobile"), ("oem", "Automobile"),

    # ── Pharma & Healthcare ────────────────────────────────────────────────────
    ("pharma", "Pharma & Healthcare"), ("hospital", "Pharma & Healthcare"),
    ("healthcare", "Pharma & Healthcare"), ("medical", "Pharma & Healthcare"),
    ("wellness & ayurveda", "Pharma & Healthcare"),

    # ── Oil & Gas / Consumable fuels ──────────────────────────────────────────
    ("refineries", "Oil & Gas"), ("oil exploration", "Oil & Gas"),
    ("exploration & production", "Oil & Gas"), ("oilfield", "Oil & Gas"),
    ("gas transmission", "Oil & Gas"), ("gas distribution", "Oil & Gas"),
    ("lpg", "Oil & Gas"), ("lubricants", "Oil & Gas"),
    ("industrial gases & fuels", "Oil & Gas"), ("biofuels", "Oil & Gas"),
    ("coal", "Oil & Gas"),

    # ── Renewables — solar (mfg + generation + EPC) + wind + green IPP ─────────
    # Its OWN sector, distinct from conventional power: different demand drivers
    # (energy-transition capex / PLI / module prices) vs electricity tariffs.
    # NOTE: many pure renewables are MISLABELLED "Power Generation/Distribution"
    # in the vendor data (Adani Green, Suzlon, Waaree, Premier Energies…); those
    # are routed here by _SYMBOL_OVERRIDES, which is checked before these rules.
    ("renewable", "Renewables"), ("wind", "Renewables"), ("solar", "Renewables"),
    ("ev / solar", "Renewables"), ("biogas", "Renewables"),
    # ── Power & Utilities — conventional electricity (thermal/hydro/lignite) ───
    # generation + transmission/distribution UTILITIES + power trading.
    ("power generation", "Power & Utilities"), ("power distribution", "Power & Utilities"),
    ("power trading", "Power & Utilities"),

    # ── Metals & Mining ────────────────────────────────────────────────────────
    ("steel", "Metals & Mining"), ("metal", "Metals & Mining"),
    ("iron ore", "Metals & Mining"), ("mining", "Metals & Mining"),
    ("aluminium", "Metals & Mining"), ("ferro alloys", "Metals & Mining"),
    ("graphite", "Metals & Mining"), ("recycling", "Metals & Mining"),

    # ── Chemicals (incl. fertilisers / agrochem) ──────────────────────────────
    ("specialty chemicals", "Chemicals"), ("chemical", "Chemicals"),
    ("fertiliser", "Chemicals"), ("fertilizer", "Chemicals"),
    ("pesticides", "Chemicals"), ("agrochemical", "Chemicals"),
    ("synthetic materials", "Chemicals"), ("industrial gases", "Chemicals"),
    ("paints", "Chemicals"), ("starch", "Chemicals"), ("solvent extraction", "Chemicals"),

    # ── Cement & Building Materials ───────────────────────────────────────────
    ("cement", "Cement & Building Materials"),
    ("building materials", "Cement & Building Materials"),
    ("construction materials", "Cement & Building Materials"),
    ("tiles", "Cement & Building Materials"), ("ceramics", "Cement & Building Materials"),
    ("glass", "Cement & Building Materials"), ("refractories", "Cement & Building Materials"),
    ("wood", "Cement & Building Materials"),

    # ── Infrastructure & Construction ─────────────────────────────────────────
    ("epc", "Infrastructure"), ("engineering - construction", "Infrastructure"),
    ("roads", "Infrastructure"), ("infrastructure", "Infrastructure"),
    ("port", "Infrastructure"),

    # ── Realty ─────────────────────────────────────────────────────────────────
    ("real estate", "Realty"), ("developers", "Realty"), ("land bank", "Realty"),
    ("co-working", "Realty"),

    # ── Capital Goods / Industrials ───────────────────────────────────────────
    ("engineering", "Capital Goods"), ("industrial machinery", "Capital Goods"),
    ("electric equipment", "Capital Goods"), ("electrical equipment", "Capital Goods"),
    ("cables & wires", "Capital Goods"), ("compressors", "Capital Goods"),
    ("pumps", "Capital Goods"), ("bearings", "Capital Goods"),
    ("precision engineering", "Capital Goods"), ("railways wagons", "Capital Goods"),
    ("industrial products", "Capital Goods"), ("industrial packaging", "Capital Goods"),
    ("abrasives", "Capital Goods"), ("industrial gases", "Capital Goods"),
    ("pipes", "Capital Goods"), ("plastic products", "Capital Goods"),
    ("packaging", "Capital Goods"), ("electronics", "Capital Goods"),

    # ── Media & Entertainment ─────────────────────────────────────────────────
    ("film production", "Media & Entertainment"), ("broadcasting", "Media & Entertainment"),
    ("entertainment", "Media & Entertainment"), ("gaming", "Media & Entertainment"),
    ("online classifieds", "Media & Entertainment"),

    # ── Consumer Durables ──────────────────────────────────────────────────────
    ("consumer durables", "Consumer Durables"), ("home appliances", "Consumer Durables"),
    ("air conditioners", "Consumer Durables"), ("home furnishings", "Consumer Durables"),
    ("footwear", "Consumer Durables"), ("luggage", "Consumer Durables"),
    ("stationery", "Consumer Durables"),

    # ── FMCG ───────────────────────────────────────────────────────────────────
    ("consumer food", "FMCG"), ("household & personal", "FMCG"), ("sugar", "FMCG"),
    ("foods", "FMCG"), ("beverages", "FMCG"), ("breweries", "FMCG"),
    ("cigarettes", "FMCG"), ("tobacco", "FMCG"), ("personal care", "FMCG"),
    ("tea/coffee", "FMCG"), ("aquaculture", "FMCG"), ("agriculture", "FMCG"),
    ("dairy", "FMCG"),

    # ── Textiles ───────────────────────────────────────────────────────────────
    ("textile", "Textiles"), ("apparel & fashion", "Textiles"),
    ("apparel & accessories", "Textiles"),

    # ── Logistics ──────────────────────────────────────────────────────────────
    ("logistics", "Logistics"), ("shipping", "Logistics"), ("courier", "Logistics"),
    ("marine services", "Logistics"), ("storage terminals", "Logistics"),

    # ── Consumer Services / Retail / Travel ───────────────────────────────────
    ("retail", "Consumer Services"), ("hypermarket", "Consumer Services"),
    ("e-commerce", "Consumer Services"), ("ecommerce", "Consumer Services"),
    ("hotel", "Consumer Services"), ("restaurants", "Consumer Services"),
    ("qsr", "Consumer Services"), ("amusement", "Consumer Services"),
    ("travel", "Consumer Services"), ("airlines", "Consumer Services"),
    ("facility services", "Consumer Services"), ("staffing", "Consumer Services"),
    ("distribution", "Consumer Services"), ("trading", "Consumer Services"),

    # ── Diversified / holding (last-resort buckets) ───────────────────────────
    ("diversified", "Diversified"), ("multi-business", "Diversified"),
    ("holding company", "Diversified"), ("paper", "Diversified"),
]

# Fallback when industry is missing/unmapped: collapse the worst raw-sector
# fragments onto a canonical name; otherwise keep the raw sector as-is.
_RAW_SECTOR_FALLBACK: dict[str, str] = {
    "Information Technology": "IT", "IT Industry": "IT",
    "Automobile & Ancillaries": "Automobile", "Automobile": "Automobile",
    "Banking & Financial Services": "Financial Services", "Financial": "Financial Services",
    "Banking": "Banking",
    "Tele-Communication": "Telecom", "Telecom": "Telecom",
    "Retailing": "Consumer Services", "Retail": "Consumer Services", "E-Commerce": "Consumer Services",
    "Energy": "Power & Utilities", "Oil & Gas": "Oil & Gas",
    "Raw Material": "Metals & Mining", "Derived Materials": "Chemicals",
    "Healthcare": "Pharma & Healthcare", "FMCG": "FMCG",
    "Consumer Goods": "Consumer Durables", "Consumer Durables": "Consumer Durables",
    "Capital Goods": "Capital Goods", "Industrial Products": "Capital Goods",
    "Defence & Aerospace": "Defence", "Real Estate": "Realty",
    "Cement & Building Materials": "Cement & Building Materials",
    "Chemicals": "Chemicals", "Metals & Mining": "Metals & Mining",
    "Power & Utilities": "Power & Utilities", "Textile Industry": "Textiles",
    "Logistics & Freight": "Logistics", "Media & Entertainment": "Media & Entertainment",
    "Infrastructure & Construction": "Infrastructure",
    "Hospitality & Travel": "Consumer Services", "Hospitality": "Consumer Services",
    "Agricultural": "FMCG", "Apparel & Accessories": "Textiles",
    "Services": "Diversified", "Diversified": "Diversified", "Industries": "Diversified",
    "ETF": "ETF", "Others": "Others",
}


def canonical_sector(
    industry: Optional[str], raw_sector: Optional[str] = None,
    symbol: Optional[str] = None,
) -> str:
    """Map a stock's (industry, raw_sector, symbol) to its canonical rotation sector.

    A per-symbol override (for vendor industry data errors) takes precedence.
    """
    if symbol and symbol.strip().upper() in _SYMBOL_OVERRIDES:
        return _SYMBOL_OVERRIDES[symbol.strip().upper()]
    ind = (industry or "").strip().lower()
    if ind:
        for needle, canon in _RULES:
            if needle in ind:
                return canon
    # Industry missing or unmatched → fall back to the raw sector map.
    if raw_sector:
        rs = raw_sector.strip()
        if rs in _RAW_SECTOR_FALLBACK:
            return _RAW_SECTOR_FALLBACK[rs]
    return "Diversified"


def populate_canonical_sectors() -> int:
    """
    Add (if absent) and (re)populate the sector_master.canonical_sector column.

    Non-destructive: the raw `sector` column is preserved for provenance. Safe to
    re-run any time (e.g. after seed_sectors adds new symbols). Returns row count.
    """
    from src.data.repository import get_repository, query_dataframe
    import pandas as pd

    rows = query_dataframe("SELECT symbol, sector, industry FROM sector_master", [])
    if rows.empty:
        return 0
    rows["canonical_sector"] = rows.apply(
        lambda r: canonical_sector(r["industry"], r["sector"], r["symbol"]), axis=1
    )
    repo = get_repository()
    with repo._cm.connect() as conn:
        conn.execute("ALTER TABLE sector_master ADD COLUMN IF NOT EXISTS canonical_sector VARCHAR")
        conn.register("_canon_df", rows[["symbol", "canonical_sector"]])
        conn.execute("""
            UPDATE sector_master
            SET canonical_sector = _canon_df.canonical_sector
            FROM _canon_df
            WHERE sector_master.symbol = _canon_df.symbol
        """)
        conn.execute(CANONICAL_VIEW_DDL)
    return len(rows)


# The canonical view exposes canonical_sector AS sector so rotation / memory
# queries select it transparently while raw_sector preserves provenance.
CANONICAL_VIEW_DDL = """
CREATE OR REPLACE VIEW v_sector_master AS
SELECT symbol, company_name,
       COALESCE(canonical_sector, sector) AS sector,
       sector AS raw_sector,
       industry, market_cap_category, category
FROM sector_master
"""


def ensure_canonical_view() -> None:
    """Create the canonical column + view if absent (safe on an empty DB)."""
    from src.data.repository import get_repository
    repo = get_repository()
    with repo._cm.connect() as conn:
        conn.execute("ALTER TABLE sector_master ADD COLUMN IF NOT EXISTS canonical_sector VARCHAR")
        conn.execute(CANONICAL_VIEW_DDL)


def audit() -> None:
    """Print the industry→canonical mapping and per-canonical symbol counts.

    Run as: python -m src.analytics.sector_taxonomy
    """
    from src.data.repository import query_dataframe
    df = query_dataframe(
        "SELECT industry, sector AS raw_sector, COUNT(*) AS n "
        "FROM sector_master GROUP BY industry, sector ORDER BY industry", []
    )
    by_canon: dict[str, int] = {}
    unmapped: list[tuple] = []
    print(f"{'INDUSTRY':<42} {'RAW SECTOR':<30} {'→ CANONICAL':<28} N")
    print("-" * 110)
    for _, r in df.iterrows():
        ind = r["industry"]; raw = r["raw_sector"]; n = int(r["n"])
        canon = canonical_sector(ind, raw)
        by_canon[canon] = by_canon.get(canon, 0) + n
        if canon == "Diversified" and (ind or "").strip().lower() not in (
            "diversified", "multi-business", "holding company", "paper", "others", ""
        ):
            unmapped.append((ind, raw, n))
        print(f"{str(ind)[:41]:<42} {str(raw)[:29]:<30} {canon:<28} {n}")
    print("\n── Canonical sector symbol counts ──")
    for c, n in sorted(by_canon.items(), key=lambda x: -x[1]):
        print(f"  {c:<30} {n}")
    if unmapped:
        print("\n⚠ Industries that fell through to Diversified (review):")
        for ind, raw, n in unmapped:
            print(f"  {ind}  ({raw})  n={n}")


if __name__ == "__main__":
    audit()
