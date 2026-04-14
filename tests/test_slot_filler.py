"""
tests/test_slot_filler.py
──────────────────────────
Unit tests for NER + slot filling.
Requires: pip install spacy && python -m spacy download en_core_web_sm
No API keys required.
"""

import pytest
from app.nlp.slot_filler import Intent, get_slot_filler


@pytest.fixture(scope="module")
def filler():
    return get_slot_filler()


# ── Intent detection ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected_intent",
    [
        ("I want to book a flight from Paris to Manila", Intent.search_flight),
        ("What is the status of flight AF 447", Intent.flight_status),
        ("How much baggage can I bring on economy?", Intent.baggage_info),
        ("I need to check in for my flight", Intent.check_in),
        ("Is there a lounge near gate F32?", Intent.lounge_access),
        ("I want to cancel my booking", Intent.cancel_booking),
        ("Can I change my flight to next Monday?", Intent.change_booking),
        ("Where is the nearest restaurant in terminal 2?", Intent.airport_info),
    ],
)
def test_intent_detection(filler, text, expected_intent):
    result = filler.parse(text)
    assert result.intent == expected_intent, (
        f"Text: {text!r}\n  Expected: {expected_intent}, Got: {result.intent}"
    )


# ── Location slot filling ──────────────────────────────────────────────────────


def test_origin_destination_iata(filler):
    r = filler.parse("I want to fly from CDG to MNL")
    assert r.slots.origin == "CDG"
    assert r.slots.destination == "MNL"


def test_origin_destination_city(filler):
    r = filler.parse("Book a flight from Paris to Manila")
    assert r.slots.origin == "CDG"
    assert r.slots.destination == "MNL"


def test_origin_destination_airport_name(filler):
    r = filler.parse("Fly from Charles de Gaulle to Ninoy Aquino")
    assert r.slots.origin == "CDG"
    assert r.slots.destination == "MNL"


def test_african_route(filler):
    r = filler.parse("I need a flight from Nairobi to Lagos")
    assert r.slots.origin == "NBO"
    assert r.slots.destination == "LOS"


def test_south_american_route(filler):
    r = filler.parse("Book me from Lima to Santiago please")
    assert r.slots.origin == "LIM"
    assert r.slots.destination == "SCL"


# ── Cabin class ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected_cabin",
    [
        ("I want business class from Paris to London", "business"),
        ("economy ticket to Manila", "economy"),
        ("first class to Dubai", "first"),
    ],
)
def test_cabin_class(filler, text, expected_cabin):
    r = filler.parse(text)
    assert r.slots.cabin_class == expected_cabin


# ── Passenger count ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected_pax",
    [
        ("2 adults flying from Paris to Rome", 2),
        ("book for one passenger", 1),
        ("three passengers to Nairobi", 3),
    ],
)
def test_passenger_count(filler, text, expected_pax):
    r = filler.parse(text)
    assert r.slots.passenger_count == expected_pax


# ── Flight number ──────────────────────────────────────────────────────────────


def test_flight_number_spaced(filler):
    r = filler.parse("What is the status of AF 447")
    assert r.slots.flight_number == "AF 447"


def test_flight_number_compact(filler):
    r = filler.parse("Check BA2490 please")
    assert r.slots.flight_number == "BA2490"


# ── Missing slots ──────────────────────────────────────────────────────────────


def test_missing_destination(filler):
    r = filler.parse("I want to fly from Paris next Friday")
    assert "destination" in r.missing_slots


def test_no_missing_slots_with_full_info(filler):
    r = filler.parse("Book a flight from Paris CDG to Manila MNL on April 20")
    assert r.slots.origin is not None
    assert r.slots.destination is not None
    assert r.slots.departure_date is not None
    assert "origin" not in r.missing_slots
    assert "destination" not in r.missing_slots


# ── French language ────────────────────────────────────────────────────────────


def test_french_intent(filler):
    r = filler.parse("Je veux réserver un vol de Paris à Nairobi", language="fr")
    assert r.intent == Intent.search_flight


def test_french_cabin(filler):
    r = filler.parse("Je voudrais un billet en classe affaires", language="fr")
    assert r.slots.cabin_class == "business"
