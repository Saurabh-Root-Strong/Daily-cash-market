"""Integrity of config/stock_segments.csv (5th classification level)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEGMENTS = {
    # Pharma / Biotech / Healthcare research
    "Integrated Global Pharma", "Branded Generics – India & EM",
    "Generics & Specialty – US/Regulated", "Formulations Export – Emerging Markets",
    "API / Bulk Drugs & Intermediates", "CDMO / CRDMO",
    "Formulation Contract Manufacturing – India", "Biologics, Biosimilars & Vaccines",
    "Injectables & Critical Care", "Consumer Health / OTC & Nutraceuticals", "Animal Health",
    "CRO, Testing & Life-sciences Services", "Drug Discovery (R&D stage)",
    "Pharma Inputs (Excipients, Capsules, Enzymes, Packaging)", "Pharma Retail & Distribution",
    # Auto components
    "Engine & ICE Powertrain Parts", "Drivetrain, Transmission & Bearings",
    "Auto Electricals, Electronics & Lighting", "Chassis, Suspension, Brakes, Steering & Wheels",
    "Body, Interiors, Glass & Sheet Metal", "Forgings, Castings & Precision Machining",
    "Batteries & Energy Storage", "Diversified Auto Systems", "Auto Accessories & Aftermarket",
    "Tyre Inputs",
    # Civil construction
    "Roads & Highways EPC", "Road Concessions & Toll Assets", "Railways & Metro",
    "Water, Irrigation, Hydro & Sanitation", "Power T&D and Energy EPC", "Solar / Renewable EPC",
    "Buildings & Urban Construction", "Diversified Infrastructure EPC",
    "Engineering Consultancy & PMC", "Pre-engineered Buildings & Structures",
    "Other / Non-construction",
    # Specialty chemicals
    "Fluorochemicals", "Pharma & Agro Intermediates / Custom Synthesis",
    "Amines, Acetyls & Solvent Derivatives", "Performance, Polymer & Rubber Additives",
    "Aroma, Surfactants & Personal-care Ingredients", "Construction Chemicals & Adhesives",
    "Textile & Leather Chemicals", "Inorganic, Chlor-alkali & Marine Chemicals",
    "Diversified Chemicals", "Dyes, Pigments & Food Colours",
    "Industrial Process Chemicals & Distribution",
}


def _seg():
    return pd.read_csv(ROOT / "config" / "stock_segments.csv", comment="#", dtype=str).fillna("")


def test_one_row_per_symbol():
    assert not _seg()["symbol"].duplicated().any()


def test_segments_from_the_fixed_vocabulary():
    s = _seg()
    assert set(s["segment"]) <= SEGMENTS
    assert set(s["secondary_segment"]) - {""} <= SEGMENTS
    assert (s["segment"] != s["secondary_segment"]).all()


def test_every_segmented_symbol_is_classified():
    cls = pd.read_csv(ROOT / "config" / "stock_classification.csv", dtype=str)
    missing = set(_seg()["symbol"]) - set(cls["symbol"])
    assert not missing, missing


def test_confidence_and_mnc_values():
    s = _seg()
    assert set(s["confidence"]) <= {"high", "medium", "low"}
    assert set(s["mnc_subsidiary"]) <= {"True", "False"}


def test_every_segmented_industry_has_a_validation_record():
    val = pd.read_csv(ROOT / "config" / "segment_validation.csv", comment="#", dtype=str)
    assert set(val["verdict"]) <= {"structural", "weak", "unconfirmed", "descriptive"}
    assert len(val) >= 4


def test_segment_health_flags_drift_and_gaps():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import load_stock_classification as L
    df = pd.read_csv(ROOT / "config" / "stock_classification.csv", dtype=str).fillna("")
    assert L.segment_health(df) == []                      # clean today
    seg = _seg()
    sym, bi = seg.iloc[0]["symbol"], seg.iloc[0]["scope_basic_industry"]
    other = "Biotechnology" if bi != "Biotechnology" else "Pharmaceuticals"
    df.loc[df["symbol"] == sym, "basic_industry"] = other   # NSE reclassifies it
    df = pd.concat([df, pd.DataFrame([{"symbol": "NEWLISTING", "basic_industry": bi,
                                       "status": "ok"}])], ignore_index=True)
    msgs = L.segment_health(df)
    assert any(m.startswith(f"DRIFT {sym}") for m in msgs)
    assert any(m.startswith("GAP") and "NEWLISTING" in m for m in msgs)


def test_scope_recorded_for_every_row():
    assert (_seg()["scope_basic_industry"] != "").all()


def test_dashboard_verdicts_follow_the_validation_record():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.analytics.segment_trends import get_segment_verdicts, SEGMENT_GROUPS
    v = get_segment_verdicts()
    assert set(v) == set(SEGMENT_GROUPS.values())
    assert v["Pharma & Healthcare"] == "structural"      # survives size+momentum control
    assert v["Auto Components"] == "unconfirmed"         # controlled test p=0.14 wins over raw
