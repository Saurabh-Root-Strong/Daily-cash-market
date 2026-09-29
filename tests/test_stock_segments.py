"""Integrity of config/stock_segments.csv (5th classification level)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEGMENTS = {
    "Integrated Global Pharma", "Branded Generics – India & EM",
    "Generics & Specialty – US/Regulated", "Formulations Export – Emerging Markets",
    "API / Bulk Drugs & Intermediates", "CDMO / CRDMO",
    "Formulation Contract Manufacturing – India", "Biologics, Biosimilars & Vaccines",
    "Injectables & Critical Care", "Consumer Health / OTC & Nutraceuticals", "Animal Health",
    "CRO, Testing & Life-sciences Services", "Drug Discovery (R&D stage)",
    "Pharma Inputs (Excipients, Capsules, Enzymes, Packaging)", "Pharma Retail & Distribution",
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
