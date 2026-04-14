"""
app/agent/prompts.py
─────────────────────
System prompts and tool definitions for the Claude dialogue manager.
Kept separate so they can be cached at the prompt level (Anthropic prompt caching).
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are an airport voice assistant helping travelers with flight bookings,
flight status, baggage information, check-in, and general airport queries.
You support English and French — always respond in the same language the traveler used.

Guidelines
──────────
• Be concise — the traveler may be in a noisy airport terminal.
• When slots are missing, ask for ONE missing slot at a time.
• Always confirm dates in the format "Day, Month DD" (e.g. "Friday, April 17").
• Use IATA codes internally but speak city names aloud (CDG → "Charles de Gaulle", LHR → "Heathrow").
• For flight status, report: gate, departure time, status (on time / delayed / cancelled).
• Never invent flight data — if you lack real-time data, say so clearly.
• If the intent is ambiguous, ask a clarifying question rather than guessing.

When you call a tool, do NOT narrate the action to the user. Wait for the tool result,
then respond naturally."""

# Tool definitions passed to Claude's `tools` parameter
TOOLS: list[dict] = [
    {
        "name": "search_flights",
        "description": (
            "Search for available flights between two airports on a given date. "
            "Returns a list of flight options with price, duration, and stops."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Origin IATA code (e.g. CDG)",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination IATA code (e.g. JFK)",
                },
                "departure_date": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD)",
                },
                "return_date": {
                    "type": "string",
                    "description": "ISO date for return, or null for one-way",
                },
                "cabin_class": {
                    "type": "string",
                    "enum": ["economy", "premium_economy", "business", "first"],
                },
                "passenger_count": {"type": "integer", "minimum": 1, "maximum": 9},
                "airline_pref": {
                    "type": "string",
                    "description": "Preferred airline IATA code, or null",
                },
            },
            "required": ["origin", "destination", "departure_date"],
        },
    },
    {
        "name": "get_flight_status",
        "description": "Get real-time status for a specific flight number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "flight_number": {
                    "type": "string",
                    "description": "Flight designator (e.g. AF447, BA2490)",
                },
                "date": {
                    "type": "string",
                    "description": "Flight date ISO (YYYY-MM-DD), or null for today",
                },
            },
            "required": ["flight_number"],
        },
    },
    {
        "name": "get_baggage_rules",
        "description": "Return baggage allowance rules for an airline and cabin class.",
        "input_schema": {
            "type": "object",
            "properties": {
                "airline": {"type": "string", "description": "Airline IATA code"},
                "cabin_class": {
                    "type": "string",
                    "enum": ["economy", "premium_economy", "business", "first"],
                },
            },
            "required": ["airline", "cabin_class"],
        },
    },
    {
        "name": "get_airport_info",
        "description": "Get general airport information: terminals, lounges, amenities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "airport": {"type": "string", "description": "Airport IATA code"},
                "info_type": {
                    "type": "string",
                    "enum": [
                        "terminals",
                        "lounges",
                        "restaurants",
                        "wifi",
                        "transport",
                    ],
                },
            },
            "required": ["airport"],
        },
    },
]
