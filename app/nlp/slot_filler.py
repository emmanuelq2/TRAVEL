"""
app/nlp/slot_filler.py
───────────────────────
Slot filling: maps NER output → structured Pydantic flight intent.

Slots
─────
  origin            IATA airport code
  destination       IATA airport code
  departure_date    ISO date
  return_date       ISO date (None = one-way)
  cabin_class       economy | business | first
  passenger_count   int ≥ 1
  airline_pref      IATA airline code
  flight_number     designator for status queries (e.g. "AF447")

Intents
───────
  SEARCH_FLIGHT     book or search for a flight
  FLIGHT_STATUS     check status of a specific flight
  BAGGAGE_INFO      baggage allowance questions
  CHECK_IN          check-in queries
  LOUNGE_ACCESS     lounge information
  CHANGE_BOOKING    modify existing booking
  CANCEL_BOOKING    cancel a booking
  AIRPORT_INFO      general airport / gate info
  UNKNOWN           catch-all
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.nlp.date_normalizer import normalize_date
from app.nlp.iata_vocab import resolve_airline, resolve_location
from app.nlp.ner import AviationNER, Entity, get_ner

logger = get_logger(__name__)


class CabinClass(StrEnum):
    economy = "economy"
    premium_economy = "premium_economy"
    business = "business"
    first = "first"


class Intent(StrEnum):
    search_flight = "SEARCH_FLIGHT"
    flight_status = "FLIGHT_STATUS"
    baggage_info = "BAGGAGE_INFO"
    check_in = "CHECK_IN"
    lounge_access = "LOUNGE_ACCESS"
    change_booking = "CHANGE_BOOKING"
    cancel_booking = "CANCEL_BOOKING"
    airport_info = "AIRPORT_INFO"
    unknown = "UNKNOWN"


class FlightSlots(BaseModel):
    origin: str | None = None
    destination: str | None = None
    departure_date: date | None = None
    return_date: date | None = None
    cabin_class: CabinClass = CabinClass.economy
    passenger_count: int = Field(default=1, ge=1, le=9)
    airline_pref: str | None = None
    flight_number: str | None = None


class ParsedUtterance(BaseModel):
    intent: Intent
    slots: FlightSlots
    raw_entities: list[dict]
    missing_slots: list[str]    # slots required for the intent but not yet filled
    language: str = "en"


# ── Intent detection ──────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (Intent.flight_status,  ["status", "delayed", "on time", "arrived", "departed"]),
    (Intent.baggage_info,   ["baggage", "luggage", "bag", "suitcase", "carry-on", "allowance"]),
    (Intent.check_in,       ["check in", "check-in", "boarding pass", "seat"]),
    (Intent.lounge_access,  ["lounge", "vip", "priority"]),
    (Intent.change_booking, ["change", "modify", "reschedule", "rebook"]),
    (Intent.cancel_booking, ["cancel", "refund", "void"]),
    (Intent.airport_info,   ["terminal", "gate", "map", "wifi", "restaurant", "pharmacy"]),
    (Intent.search_flight,  ["book", "fly", "flight", "ticket", "search", "find"]),
]

_FR_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (Intent.flight_status,  ["statut", "retard", "à l'heure", "porte", "arrivé"]),
    (Intent.baggage_info,   ["bagage", "valise", "cabine", "franchise"]),
    (Intent.check_in,       ["enregistrement", "carte d'embarquement", "siège"]),
    (Intent.lounge_access,  ["salon", "priorité"]),
    (Intent.change_booking, ["modifier", "changer", "reprogrammer"]),
    (Intent.cancel_booking, ["annuler", "remboursement"]),
    (Intent.airport_info,   ["terminal", "porte", "plan", "wifi"]),
    (Intent.search_flight,  ["réserver", "voler", "vol", "billet", "chercher"]),
]


def _detect_intent(text: str, language: str = "en") -> Intent:
    lower = text.lower()
    patterns = _FR_INTENT_PATTERNS if language == "fr" else _INTENT_PATTERNS
    for intent, keywords in patterns:
        if any(kw in lower for kw in keywords):
            return intent
    return Intent.unknown


# ── Cabin mapping ──────────────────────────────────────────────────────────────

_CABIN_MAP: dict[str, CabinClass] = {
    "economy": CabinClass.economy,
    "eco": CabinClass.economy,
    "économique": CabinClass.economy,
    "premium": CabinClass.premium_economy,
    "premium economy": CabinClass.premium_economy,
    "business": CabinClass.business,
    "affaires": CabinClass.business,
    "first": CabinClass.first,
    "première": CabinClass.first,
}

# ── PAX extraction ─────────────────────────────────────────────────────────────

_PAX_RE = re.compile(
    r"(\d+|one|two|three|four|five|six|sept|deux|trois|quatre|cinq|six)\s+"
    r"(adult|child|infant|passenger|pax|personne|adulte|enfant)s?",
    re.IGNORECASE,
)
_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
}


def _extract_pax(text: str) -> int:
    m = _PAX_RE.search(text)
    if not m:
        return 1
    raw = m.group(1).lower()
    return _WORD_NUMS.get(raw, int(raw) if raw.isdigit() else 1)


# ── Required slots per intent ─────────────────────────────────────────────────

_REQUIRED: dict[Intent, list[str]] = {
    Intent.search_flight:  ["origin", "destination", "departure_date"],
    Intent.flight_status:  ["flight_number"],
    Intent.baggage_info:   [],
    Intent.check_in:       [],
    Intent.lounge_access:  [],
    Intent.change_booking: [],
    Intent.cancel_booking: [],
    Intent.airport_info:   [],
    Intent.unknown:        [],
}


# ── Main entry point ──────────────────────────────────────────────────────────

class SlotFiller:
    def __init__(self, ner: AviationNER | None = None) -> None:
        self._ner = ner or get_ner()

    def parse(self, text: str, language: str = "en") -> ParsedUtterance:
        entities: list[Entity] = self._ner.extract(text)
        intent = _detect_intent(text, language)

        airports = [e for e in entities if e.label == "AIRPORT"]
        airlines = [e for e in entities if e.label == "AIRLINE"]
        dates    = [e for e in entities if e.label == "DATE"]
        cabins   = [e for e in entities if e.label == "CABIN"]
        flight_nos = [e for e in entities if e.label == "FLIGHT_NO"]

        origin = resolve_location(airports[0].text) if len(airports) >= 1 else None
        dest   = resolve_location(airports[1].text) if len(airports) >= 2 else None
        dep_date = normalize_date(dates[0].text) if len(dates) >= 1 else None
        ret_date = normalize_date(dates[1].text) if len(dates) >= 2 else None
        cabin = _CABIN_MAP.get(cabins[0].text.lower()) if cabins else CabinClass.economy
        airline = resolve_airline(airlines[0].text) if airlines else None
        flight_no = flight_nos[0].text if flight_nos else None

        slots = FlightSlots(
            origin=origin,
            destination=dest,
            departure_date=dep_date,
            return_date=ret_date,
            cabin_class=cabin,
            passenger_count=_extract_pax(text),
            airline_pref=airline,
            flight_number=flight_no,
        )

        missing = [
            s for s in _REQUIRED.get(intent, [])
            if getattr(slots, s) is None
        ]

        logger.info(
            "slot_fill_result",
            intent=intent,
            missing=missing,
            origin=origin,
            destination=dest,
        )

        return ParsedUtterance(
            intent=intent,
            slots=slots,
            raw_entities=[{"label": e.label, "text": e.text} for e in entities],
            missing_slots=missing,
            language=language,
        )


_filler: SlotFiller | None = None


def get_slot_filler() -> SlotFiller:
    global _filler
    if _filler is None:
        _filler = SlotFiller()
    return _filler
