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
