"""Regression tests for canonical_sector() rule matching.

The rules are substring needles; before 2026-09-28 they matched anywhere in the
industry string, so short needles fired inside unrelated words.
"""
import pytest

from src.analytics.sector_taxonomy import canonical_sector


@pytest.mark.parametrize("industry, expected", [
    # Mid-word traps that used to misroute
    ("Textiles", "Textiles"),                          # "tiles" → Cement
    ("TV Broadcasting & Software Production", "Media & Entertainment"),  # "software" → IT
    ("Industrial Automation", "Capital Goods"),        # "auto" → Automobile
    ("Transport Services", "Logistics"),
    # Intended word-stem matches must keep working
    ("Pharmaceuticals & Drugs", "Pharma & Healthcare"),
    ("Automobiles - Passenger Cars", "Automobile"),
    ("Pesticides & Agrochemicals", "Chemicals"),
    ("Telecommunication - Equipment", "Telecom"),
    ("IT - Software", "IT"),
    ("EMS", "IT"),
    ("Ports", "Infrastructure"),
    ("Petrochemicals", "Chemicals"),
])
def test_rule_matching(industry, expected):
    assert canonical_sector(industry, None) == expected


@pytest.mark.parametrize("industry", ["Security Systems", "Sports Goods", "Power Systems"])
def test_no_mid_word_false_positive(industry):
    # "ems" must not hit "Systems" (IT); "port" must not hit "Sports" (Infrastructure)
    assert canonical_sector(industry, None) not in ("IT", "Infrastructure")


def test_kalpataru_not_forced_to_infrastructure():
    # KALPATARU is now Kalpataru Ltd (developer); the EPC company is KPIL
    assert canonical_sector("Developers", "Real Estate", "KALPATARU") == "Realty"


def test_no_module_reads_the_raw_sector_table():
    """Every consumer must read v_sector_master (canonical sector + official
    sub-sector). The raw table mixes 52 vendor sector labels ("Energy" = Reliance
    + NTPC). Only the code that builds the canonical column may read it."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "src"
    allowed = {"sector_taxonomy.py", "orchestrator.py"}
    raw = re.compile(r"\b(FROM|JOIN)\s+sector_master\b", re.I)
    offenders = [f"{p.relative_to(root)}:{i}"
                 for p in root.rglob("*.py") if p.name not in allowed
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if raw.search(line)]
    assert not offenders, offenders


def test_view_falls_back_then_uses_official_sub_sector():
    import duckdb
    from src.analytics.sector_taxonomy import _create_view
    con = duckdb.connect(":memory:")
    con.execute("""CREATE TABLE sector_master (symbol VARCHAR, company_name VARCHAR,
        sector VARCHAR, industry VARCHAR, market_cap_category VARCHAR,
        category VARCHAR, canonical_sector VARCHAR)""")
    con.execute("INSERT INTO sector_master VALUES ('X','X Ltd','Healthcare',"
                "'Pharmaceuticals & Drugs','','','Pharma & Healthcare')")
    _create_view(con)                                   # fresh DB: no classification yet
    assert con.execute("select industry, sector from v_sector_master").fetchone() == \
        ("Pharmaceuticals & Drugs", "Pharma & Healthcare")
    con.execute("CREATE TABLE stock_classification (symbol VARCHAR, basic_industry VARCHAR)")
    con.execute("INSERT INTO stock_classification VALUES ('X','Pharmaceuticals')")
    _create_view(con)
    assert con.execute("select industry, raw_industry from v_sector_master").fetchone() == \
        ("Pharmaceuticals", "Pharmaceuticals & Drugs")
