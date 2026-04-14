"""
app/agent/dialogue_manager.py
──────────────────────────────
Claude-powered multi-turn dialogue manager.

Architecture
────────────
  Each session keeps a message history.
  The system prompt is marked for prompt caching (Anthropic cache_control)
  so it is only billed on cache miss (every ~5 min of inactivity).

  Per turn:
    1. Receive parsed utterance (intent + slots from slot filler)
    2. Build messages list
    3. Call Claude with tools defined in prompts.py
    4. If Claude calls a tool → dispatch to tool handler → re-enter
    5. Return final text response

Tool dispatch
─────────────
  Tool results here are stubs — wire them to real APIs (Amadeus, OAG, etc.)
  in production.  The stubs return realistic-looking JSON so the dialogue
  can be tested end-to-end without live data.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import anthropic

from app.agent.prompts import SYSTEM_PROMPT, TOOLS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.nlp.slot_filler import FlightSlots, Intent, ParsedUtterance

logger = get_logger(__name__)

Message = dict[str, Any]


class DialogueSession:
    """One conversation session (one traveler)."""

    def __init__(self) -> None:
        self.history: list[Message] = []
        self.slots = FlightSlots()
        self.language: str = "en"


class DialogueManager:
    def __init__(self) -> None:
        cfg = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.claude_model
        self._max_tokens = cfg.claude_max_tokens

    # ── Public API ────────────────────────────────────────────────────────────

    async def respond(
        self,
        session: DialogueSession,
        utterance: ParsedUtterance,
        raw_text: str,
    ) -> str:
        """
        Process one user turn.
        Returns the assistant's natural-language reply.
        """
        session.language = utterance.language
        self._merge_slots(session, utterance.slots)

        # Build user message with structured context injected as hidden XML
        user_content = self._build_user_message(raw_text, utterance)
        session.history.append({"role": "user", "content": user_content})

        reply = await self._call_claude(session)
        session.history.append({"role": "assistant", "content": reply})
        return reply

    # ── Claude call (with agentic tool loop) ─────────────────────────────────

    async def _call_claude(self, session: DialogueSession) -> str:
        # System prompt with prompt caching — only re-billed on cache miss
        system: list[dict] = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        messages = list(session.history)

        for _ in range(5):   # max tool-call iterations
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            logger.debug(
                "claude_response",
                stop_reason=response.stop_reason,
                usage=response.usage.model_dump(),
            )

            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            if response.stop_reason == "tool_use":
                # Append assistant turn with tool_use blocks
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool and collect results
                tool_results: list[dict] = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = await self._dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })

                messages.append({"role": "user", "content": tool_results})
                continue

            break  # unexpected stop reason

        return "I'm sorry, I encountered an issue processing your request."

    # ── Tool dispatcher (stub implementations) ────────────────────────────────

    async def _dispatch_tool(self, name: str, inputs: dict) -> dict:
        logger.info("tool_call", tool=name, inputs=inputs)

        if name == "search_flights":
            return self._stub_search_flights(inputs)
        if name == "get_flight_status":
            return self._stub_flight_status(inputs)
        if name == "get_baggage_rules":
            return self._stub_baggage_rules(inputs)
        if name == "get_airport_info":
            return self._stub_airport_info(inputs)

        return {"error": f"Unknown tool: {name}"}

    # ── Stub tool results (replace with real API calls in production) ─────────

    def _stub_search_flights(self, inp: dict) -> dict:
        return {
            "flights": [
                {
                    "flight_number": "AF 380",
                    "origin": inp.get("origin"),
                    "destination": inp.get("destination"),
                    "departure": f"{inp.get('departure_date')}T08:15",
                    "arrival": f"{inp.get('departure_date')}T10:30",
                    "duration_min": 135,
                    "stops": 0,
                    "cabin": inp.get("cabin_class", "economy"),
                    "price_eur": 249,
                    "seats_left": 7,
                },
                {
                    "flight_number": "AF 384",
                    "origin": inp.get("origin"),
                    "destination": inp.get("destination"),
                    "departure": f"{inp.get('departure_date')}T14:45",
                    "arrival": f"{inp.get('departure_date')}T17:00",
                    "duration_min": 135,
                    "stops": 0,
                    "cabin": inp.get("cabin_class", "economy"),
                    "price_eur": 199,
                    "seats_left": 3,
                },
            ]
        }

    def _stub_flight_status(self, inp: dict) -> dict:
        return {
            "flight_number": inp.get("flight_number"),
            "status": "on_time",
            "gate": "F32",
            "scheduled_departure": "2026-04-12T10:15",
            "estimated_departure": "2026-04-12T10:15",
            "terminal": "2F",
        }

    def _stub_baggage_rules(self, inp: dict) -> dict:
        cabin = inp.get("cabin_class", "economy")
        return {
            "airline": inp.get("airline"),
            "cabin_class": cabin,
            "carry_on_kg": 12,
            "checked_bags": 1 if cabin == "economy" else 2,
            "checked_kg": 23 if cabin == "economy" else 32,
        }

    def _stub_airport_info(self, inp: dict) -> dict:
        return {
            "airport": inp.get("airport"),
            "info_type": inp.get("info_type", "terminals"),
            "data": "Terminal information is available at information desks on level 2.",
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_user_message(self, raw_text: str, utterance: ParsedUtterance) -> str:
        """
        Inject structured NLP output as hidden XML context so Claude has
        all slot data without the user seeing it.
        """
        slots = utterance.slots
        context = (
            f"<nlu_context>"
            f"<intent>{utterance.intent}</intent>"
            f"<language>{utterance.language}</language>"
            f"<slots>"
            f"<origin>{slots.origin or ''}</origin>"
            f"<destination>{slots.destination or ''}</destination>"
            f"<departure_date>{slots.departure_date or ''}</departure_date>"
            f"<return_date>{slots.return_date or ''}</return_date>"
            f"<cabin>{slots.cabin_class}</cabin>"
            f"<pax>{slots.passenger_count}</pax>"
            f"<airline>{slots.airline_pref or ''}</airline>"
            f"<flight_number>{slots.flight_number or ''}</flight_number>"
            f"</slots>"
            f"<missing_slots>{', '.join(utterance.missing_slots)}</missing_slots>"
            f"</nlu_context>"
        )
        return f"{context}\n\n{raw_text}"

    @staticmethod
    def _merge_slots(session: DialogueSession, new_slots: FlightSlots) -> None:
        """Update session slots — only overwrite with non-None values."""
        for field in FlightSlots.model_fields:
            val = getattr(new_slots, field)
            if val is not None:
                setattr(session.slots, field, val)

    @staticmethod
    def _extract_text(response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""


# ── Session store (in-memory, keyed by session_id) ────────────────────────────
_sessions: dict[str, DialogueSession] = {}


def get_or_create_session(session_id: str) -> DialogueSession:
    if session_id not in _sessions:
        _sessions[session_id] = DialogueSession()
    return _sessions[session_id]


# ── Singleton manager ─────────────────────────────────────────────────────────
_manager: DialogueManager | None = None


def get_dialogue_manager() -> DialogueManager:
    global _manager
    if _manager is None:
        _manager = DialogueManager()
    return _manager
