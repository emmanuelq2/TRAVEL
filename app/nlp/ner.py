"""
app/nlp/ner.py
──────────────
Aviation-domain Named Entity Recognition.

Entity labels produced
──────────────────────
  AIRPORT   — IATA code, city name, or airport full name  ("CDG", "Paris", "Charles de Gaulle", "Heathrow")
  AIRLINE   — IATA code or carrier name     ("AF", "Air France")
  FLIGHT_NO — Flight designator             ("AF447", "BA 2490")
  DATE      — Travel date expression        ("next Friday", "March 15")
  TIME      — Departure/arrival time        ("8am", "14:30")
  CABIN     — Cabin class keyword           ("business", "economy", "first")
  PAX       — Passenger count phrase        ("2 adults", "one child")

Pipeline
────────
  spaCy en_core_web_sm
    └── EntityRuler (IATA codes, flight numbers)  ← added first (highest priority)
    └── standard NER component (DATE, TIME, CARDINAL from spaCy)
    └── post-processing: map spaCy DATE/CARDINAL → our DATE/PAX labels

Usage
─────
    ner = AviationNER()
    entities = ner.extract("I want to fly from Paris to CDG on next Friday business class")
    # → [Entity(label='AIRPORT', text='Paris', ...), ...]
"""

from __future__ import annotations

from dataclasses import dataclass

import spacy
from spacy.language import Language

from app.core.logging import get_logger
from app.nlp.iata_vocab import build_spacy_patterns

logger = get_logger(__name__)

_CABIN_KEYWORDS = {
    "economy",
    "business",
    "first",
    "premium",
    "eco",
    "affaires",
    "économique",
    "première",
}
_PAX_PATTERNS = (
    r"(\d+|one|two|three|four|five|six)\s+(adult|child|infant|passenger|pax)s?"
)


@dataclass
class Entity:
    label: str
    text: str
    start: int
    end: int
    iata_code: str | None = None  # resolved IATA code if available


class AviationNER:
    def __init__(self) -> None:
        logger.info("ner_loading_spacy_model")
        self._nlp = self._build_pipeline()
        logger.info("ner_ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, text: str) -> list[Entity]:
        doc = self._nlp(text)
        entities: list[Entity] = []
        seen_spans: set[tuple[int, int]] = set()

        for ent in doc.ents:
            span = (ent.start_char, ent.end_char)
            if span in seen_spans:
                continue

            label = self._map_label(ent.label_, ent.text)
            if label is None:
                continue
            seen_spans.add(span)

            iata_code = ent.kb_id_ or None
            entities.append(
                Entity(
                    label=label,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    iata_code=iata_code if iata_code else None,
                )
            )

        # Cabin class (not covered by spaCy NER — rule-based)
        lower = text.lower()
        for cabin in _CABIN_KEYWORDS:
            idx = lower.find(cabin)
            if idx != -1:
                span = (idx, idx + len(cabin))
                if span not in seen_spans:
                    seen_spans.add(span)
                    entities.append(
                        Entity(
                            label="CABIN",
                            text=text[idx : idx + len(cabin)],
                            start=idx,
                            end=idx + len(cabin),
                        )
                    )

        return sorted(entities, key=lambda e: e.start)

    def extract_dict(self, text: str) -> dict[str, list[str]]:
        """Convenience: returns {label: [text, ...]} grouped dict."""
        result: dict[str, list[str]] = {}
        for ent in self.extract(text):
            result.setdefault(ent.label, []).append(ent.text)
        return result

    # ── Pipeline builder ──────────────────────────────────────────────────────

    def _build_pipeline(self) -> Language:
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])

        # EntityRuler runs BEFORE the standard NER so aviation codes take priority
        ruler = nlp.add_pipe(
            "entity_ruler", before="ner", config={"overwrite_ents": True}
        )
        ruler.add_patterns(build_spacy_patterns())

        return nlp

    # ── Label mapping ─────────────────────────────────────────────────────────

    @staticmethod
    def _map_label(spacy_label: str, text: str) -> str | None:
        """
        Translate spaCy's built-in labels to our aviation-domain labels.
        Returns None to drop the entity.
        """
        if spacy_label in ("AIRPORT", "AIRLINE", "FLIGHT_NO", "CABIN", "PAX"):
            return spacy_label  # already our label from EntityRuler
        if spacy_label == "DATE":
            return "DATE"
        if spacy_label == "TIME":
            return "TIME"
        if spacy_label == "CARDINAL":
            # Only keep if it looks like a pax count context
            # (handled in slot filler from surrounding tokens)
            return None
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────
_ner: AviationNER | None = None


def get_ner() -> AviationNER:
    global _ner
    if _ner is None:
        _ner = AviationNER()
    return _ner
