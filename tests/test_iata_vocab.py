"""
tests/test_iata_vocab.py
─────────────────────────
Unit tests for IATA vocab lookups and NER patterns.
No API keys required — pure in-memory logic.
"""

import pytest
from app.nlp.iata_vocab import (
    AIRPORTS,
    build_spacy_patterns,
    get_iata_keywords,
    resolve_airline,
    resolve_location,
)


# ── AIRPORTS dict ─────────────────────────────────────────────────────────────


def test_airports_loaded():
    assert len(AIRPORTS) >= 100


def test_manila_present():
    assert "MNL" in AIRPORTS
    assert AIRPORTS["MNL"]["city"] == "Manila"
    assert AIRPORTS["MNL"]["name"] == "Ninoy Aquino Intl"


def test_african_airports_present():
    for code in ["NBO", "ADD", "LOS", "CMN", "CPT", "EBB", "KGL", "DAR", "DKR", "LAD"]:
        assert code in AIRPORTS, f"{code} missing from AIRPORTS"


def test_south_american_airports_present():
    for code in ["SCL", "LIM", "UIO", "MVD", "CCS", "GIG", "BSB", "CUZ", "VVI"]:
        assert code in AIRPORTS, f"{code} missing from AIRPORTS"


# ── resolve_location ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        # Exact IATA code
        ("CDG", "CDG"),
        ("MNL", "MNL"),
        ("NBO", "NBO"),
        # City name
        ("Paris", "CDG"),
        ("Manila", "MNL"),
        ("Nairobi", "NBO"),
        ("Lima", "LIM"),
        ("Lagos", "LOS"),
        # Full airport name
        ("Ninoy Aquino Intl", "MNL"),
        ("Jomo Kenyatta Intl", "NBO"),
        ("Charles de Gaulle", "CDG"),
        ("Heathrow", "LHR"),
        ("Schiphol", "AMS"),
        ("Guarulhos", "GRU"),
        # Short / partial
        ("de Gaulle", "CDG"),
        ("Kenyatta", "NBO"),
        ("Changi", "SIN"),
    ],
)
def test_resolve_location(text, expected):
    assert resolve_location(text) == expected, (
        f"resolve_location({text!r}) should be {expected}"
    )


def test_resolve_location_unknown():
    assert resolve_location("XYZ unknown city") is None


# ── resolve_airline ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("AF", "AF"),
        ("Air France", "AF"),
        ("Emirates", "EK"),
        ("Ethiopian Airlines", "ET"),
        ("LATAM", "LA"),
    ],
)
def test_resolve_airline(text, expected):
    assert resolve_airline(text) == expected


# ── get_iata_keywords ──────────────────────────────────────────────────────────


def test_keywords_include_airport_names():
    keywords = get_iata_keywords()
    assert "Ninoy Aquino Intl" in keywords
    assert "Charles de Gaulle" in keywords
    assert "Jomo Kenyatta Intl" in keywords
    assert "Schiphol" in keywords


def test_keywords_include_codes():
    keywords = get_iata_keywords()
    for code in ["CDG", "MNL", "NBO", "LOS", "SCL"]:
        assert code in keywords


def test_keywords_include_cities():
    keywords = get_iata_keywords()
    for city in ["Manila", "Nairobi", "Santiago", "Lima"]:
        assert city in keywords


# ── build_spacy_patterns ───────────────────────────────────────────────────────


def test_spacy_patterns_airport_code():
    patterns = build_spacy_patterns()
    labels = {
        p["pattern"]: p["label"] for p in patterns if isinstance(p["pattern"], str)
    }
    assert labels.get("MNL") == "AIRPORT"
    assert labels.get("NBO") == "AIRPORT"
    assert labels.get("CDG") == "AIRPORT"


def test_spacy_patterns_airport_name():
    patterns = build_spacy_patterns()
    texts = {p["pattern"] for p in patterns if isinstance(p["pattern"], str)}
    assert "Ninoy Aquino Intl" in texts
    assert "Charles de Gaulle" in texts
    assert "Heathrow" in texts
    assert "Schiphol" in texts


def test_spacy_patterns_city():
    patterns = build_spacy_patterns()
    texts = {p["pattern"] for p in patterns if isinstance(p["pattern"], str)}
    assert "Manila" in texts
    assert "Nairobi" in texts
    assert "Paris" in texts
