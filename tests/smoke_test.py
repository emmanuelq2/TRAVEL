"""
tests/smoke_test.py
────────────────────
Quick offline smoke test — runs without a server or API keys.
Tests: IATA vocab, NER resolve, slot filling, date normalizer.

Run:
    python tests/smoke_test.py
"""

import io
import os
import sys

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def section(title: str) -> None:
    print(f"\n{'-'*50}")
    print(f"  {title}")
    print('-'*50)


def check(label: str, got, expected=None) -> None:
    if expected is not None:
        ok = got == expected
        status = "OK" if ok else "FAIL"
        print(f"  [{status}]  {label}: {got!r}  (expected {expected!r})")
        if not ok:
            sys.exit(1)
    else:
        print(f"  [--]  {label}: {got!r}")


# ── 1. IATA vocab ─────────────────────────────────────────────────────────────

section("1. IATA vocab — airport count & spot checks")

from app.nlp.iata_vocab import AIRPORTS, AIRLINES, resolve_location, resolve_airline, get_iata_keywords

check("Total airports >= 100", len(AIRPORTS) >= 100, True)
check("MNL present", "MNL" in AIRPORTS, True)
check("NBO present", "NBO" in AIRPORTS, True)
check("SCL present", "SCL" in AIRPORTS, True)

check("resolve_location('Manila')",            resolve_location("Manila"),            "MNL")
check("resolve_location('Nairobi')",           resolve_location("Nairobi"),           "NBO")
check("resolve_location('Ninoy Aquino Intl')", resolve_location("Ninoy Aquino Intl"), "MNL")
check("resolve_location('Charles de Gaulle')", resolve_location("Charles de Gaulle"), "CDG")
check("resolve_location('de Gaulle')",         resolve_location("de Gaulle"),         "CDG")
check("resolve_location('Kenyatta')",          resolve_location("Kenyatta"),          "NBO")
check("resolve_location('Heathrow')",          resolve_location("Heathrow"),          "LHR")
check("resolve_location('Schiphol')",          resolve_location("Schiphol"),          "AMS")
check("resolve_location('Lima')",              resolve_location("Lima"),              "LIM")
check("resolve_location('Lagos')",             resolve_location("Lagos"),             "LOS")
check("resolve_location('CDG')",               resolve_location("CDG"),               "CDG")
check("resolve_location('unknown')",           resolve_location("xyz unknown"),       None)

check("resolve_airline('Air France')",         resolve_airline("Air France"),         "AF")
check("resolve_airline('Ethiopian Airlines')", resolve_airline("Ethiopian Airlines"), "ET")
check("resolve_airline('LATAM')",              resolve_airline("LATAM"),              "LA")

keywords = get_iata_keywords()
check("Keywords include 'Ninoy Aquino Intl'",  "Ninoy Aquino Intl" in keywords, True)
check("Keywords include 'Charles de Gaulle'",  "Charles de Gaulle" in keywords, True)
check("Keywords include 'MNL'",                "MNL" in keywords, True)

# ── 2. Date normalizer ────────────────────────────────────────────────────────

section("2. Date normalizer  (needs: pip install dateparser)")

try:
    from app.nlp.date_normalizer import normalize_date, normalize_time
    from datetime import date

    check("ISO date",           normalize_date("2026-07-15"),      date(2026, 7, 15))
    check("Natural date",       normalize_date("July 15 2026"),    date(2026, 7, 15))
    check("French date",        normalize_date("le 20 avril 2026"), date(2026, 4, 20))
    tomorrow = normalize_date("tomorrow")
    check("Tomorrow >= today",  tomorrow >= date.today(),          True)
    check("Invalid -> None",    normalize_date("not a date"),       None)
    check("Time 14:30",         normalize_time("14:30"),           "14:30")
    check("Time 8am",           normalize_time("8am"),             "08:00")
except ImportError as e:
    print(f"  [SKIP]  dateparser not installed ({e})")
    print("          Run: pip install dateparser")

# ── 3. Slot filler (requires spaCy model) ────────────────────────────────────

section("3. Slot filler  (needs: pip install spacy && python -m spacy download en_core_web_sm)")

try:
    from app.nlp.slot_filler import get_slot_filler, Intent

    filler = get_slot_filler()

    r = filler.parse("I want to fly from Paris to Manila on April 20 business class")
    check("origin",             r.slots.origin,           "CDG")
    check("dest",               r.slots.destination,      "MNL")
    check("cabin",              str(r.slots.cabin_class), "business")
    check("intent",             r.intent,                 Intent.search_flight)

    r2 = filler.parse("I need a flight from Nairobi to Lagos")
    check("Africa origin",      r2.slots.origin,          "NBO")
    check("Africa dest",        r2.slots.destination,     "LOS")

    r3 = filler.parse("What is the status of AF 447")
    check("flight status",      r3.intent,                Intent.flight_status)
    check("flight number",      r3.slots.flight_number,   "AF 447")

    r4 = filler.parse("Book me from Lima to Santiago please")
    check("Lima->SCL origin",   r4.slots.origin,          "LIM")
    check("Lima->SCL dest",     r4.slots.destination,     "SCL")

    r5 = filler.parse("Je veux reserver un vol de Paris a Nairobi", language="fr")
    check("French intent",      r5.intent,                Intent.search_flight)

except ImportError as e:
    print(f"  [SKIP]  spaCy not installed ({e})")
    print("          Run: pip install spacy && python -m spacy download en_core_web_sm")
except OSError as e:
    print(f"  [SKIP]  spaCy model missing ({e})")
    print("          Run: python -m spacy download en_core_web_sm")

# ── Done ──────────────────────────────────────────────────────────────────────

print("\n[PASS] All smoke tests passed\n")
