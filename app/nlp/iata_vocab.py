"""
app/nlp/iata_vocab.py
──────────────────────
Loads IATA airport + airline codes from data/iata_codes.json and exposes:

  • AIRPORTS / AIRLINES          — raw dicts for lookup
  • get_iata_keywords()          — flat list of codes + names for ASR boosting
  • IATA_PROMPT_HINT             — short sentence to prime Whisper's decoder
  • build_spacy_patterns()       — EntityRuler patterns for NER pipeline
  • resolve_location(text)       — fuzzy-match text → IATA code
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "iata_codes.json"

with _DATA_PATH.open(encoding="utf-8") as _f:
    _raw = json.load(_f)

AIRPORTS: dict[str, dict] = _raw["airports"]  # {"CDG": {name, city, country}}
AIRLINES: dict[str, dict] = _raw["airlines"]  # {"AF":  {name, country}}

# City → IATA reverse index (lowercase city → code)
# First entry wins so that "Paris" → CDG (not ORY), "London" → LHR (not LGW), etc.
_CITY_INDEX: dict[str, str] = {}
for _code, _info in AIRPORTS.items():
    _key = _info["city"].lower()
    if _key not in _CITY_INDEX:
        _CITY_INDEX[_key] = _code

# Airport full name → IATA code (e.g. "charles de gaulle" → "CDG")
# First entry wins for any shared name fragments.
_AIRPORT_NAME_INDEX: dict[str, str] = {}
for _code, _info in AIRPORTS.items():
    _key = _info["name"].lower()
    if _key not in _AIRPORT_NAME_INDEX:
        _AIRPORT_NAME_INDEX[_key] = _code
# Airline name → code
_AIRLINE_NAME_INDEX: dict[str, str] = {
    v["name"].lower(): k for k, v in AIRLINES.items()
}


# ── ASR keyword boost list ─────────────────────────────────────────────────────


def get_iata_keywords() -> list[str]:
    """
    Returns a flat list of strings to inject into Deepgram keyword boosts
    and Whisper prompt hints:  codes + city names + airline names.
    """
    terms: list[str] = []
    terms += list(AIRPORTS.keys())  # CDG, LHR, …
    terms += [v["city"] for v in AIRPORTS.values()]  # Paris, London, …
    terms += [v["name"] for v in AIRPORTS.values()]  # Charles de Gaulle, Heathrow, …
    terms += list(AIRLINES.keys())  # AF, BA, …
    terms += [v["name"] for v in AIRLINES.values()]  # Air France, …
    return terms


# Short prompt that primes Whisper's BPE decoder toward aviation tokens
IATA_PROMPT_HINT: str = (
    "Flight booking assistant. "
    + "IATA codes: "
    + ", ".join(list(AIRPORTS.keys())[:20])
    + ". "
    + "Airlines: "
    + ", ".join(list(AIRLINES.keys())[:15])
    + "."
)


# ── spaCy EntityRuler patterns ─────────────────────────────────────────────────


def build_spacy_patterns() -> list[dict]:
    """
    Returns patterns for spaCy EntityRuler.
    Labels: AIRPORT, AIRLINE, FLIGHT_NO
    """
    patterns: list[dict] = []

    # Airport IATA codes (e.g. "CDG", "LHR")
    for code in AIRPORTS:
        patterns.append({"label": "AIRPORT", "pattern": code})
        patterns.append({"label": "AIRPORT", "pattern": code.lower()})

    # Airport city names  (e.g. "Paris", "London")
    for code, info in AIRPORTS.items():
        patterns.append({"label": "AIRPORT", "pattern": info["city"], "id": code})

    # Airport full names  (e.g. "Charles de Gaulle", "Heathrow", "Schiphol")
    _STRIP_SUFFIXES = (
        " international airport",
        " international",
        " intl airport",
        " intl",
        " airport",
        " aéroport",
    )
    for code, info in AIRPORTS.items():
        name = info["name"]
        patterns.append({"label": "AIRPORT", "pattern": name, "id": code})

        # Short form 1: drop trailing "International Airport" etc.
        # e.g. "Ninoy Aquino International Airport" → "Ninoy Aquino"
        short_suffix = name.lower()
        for suffix in _STRIP_SUFFIXES:
            if short_suffix.endswith(suffix):
                short_suffix = short_suffix[: -len(suffix)].strip()
                break
        # Reconstruct with original casing
        core = name[: len(short_suffix)].strip()
        if core and core.lower() != name.lower():
            patterns.append({"label": "AIRPORT", "pattern": core, "id": code})

        # Short form 2: drop the first word
        # e.g. "de Gaulle" from "Charles de Gaulle"
        words = name.split()
        if len(words) >= 2:
            short = " ".join(words[1:])
            if short.lower() != core.lower():
                patterns.append({"label": "AIRPORT", "pattern": short, "id": code})

    # Airline codes
    for code in AIRLINES:
        patterns.append({"label": "AIRLINE", "pattern": code})

    # Airline full names (multi-token)
    for code, info in AIRLINES.items():
        name = info["name"]
        patterns.append({"label": "AIRLINE", "pattern": name, "id": code})

    # Flight number pattern: 2-letter code + 1-4 digits (e.g. "AF 447", "BA2490")
    patterns.append(
        {
            "label": "FLIGHT_NO",
            "pattern": [
                {"TEXT": {"REGEX": r"^[A-Z]{2}$"}},
                {"TEXT": {"REGEX": r"^\d{1,4}$"}},
            ],
        }
    )
    patterns.append(
        {
            "label": "FLIGHT_NO",
            "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2}\d{1,4}$"}}],
        }
    )

    return patterns


# ── Fuzzy resolver ────────────────────────────────────────────────────────────


def resolve_location(text: str) -> str | None:
    """
    Map a free-text location mention to an IATA airport code.
    Resolution order:
      1. Exact IATA code          ("CDG")
      2. Exact city name          ("Paris")
      3. Exact airport full name  ("Charles de Gaulle")
      4. Partial city match       ("New" → New York)
      5. Partial airport name     ("Gaulle" → CDG)
    """
    t = text.strip().upper()

    if t in AIRPORTS:
        return t

    lower = t.lower()

    if lower in _CITY_INDEX:
        return _CITY_INDEX[lower]

    if lower in _AIRPORT_NAME_INDEX:
        return _AIRPORT_NAME_INDEX[lower]

    for city, code in _CITY_INDEX.items():
        if lower in city or city in lower:
            return code

    for name, code in _AIRPORT_NAME_INDEX.items():
        if lower in name or name in lower:
            return code

    return None


def resolve_airline(text: str) -> str | None:
    """Map an airline mention to its IATA code."""
    t = text.strip()
    if t.upper() in AIRLINES:
        return t.upper()
    lower = t.lower()
    if lower in _AIRLINE_NAME_INDEX:
        return _AIRLINE_NAME_INDEX[lower]
    for name, code in _AIRLINE_NAME_INDEX.items():
        if lower in name:
            return code
    return None
