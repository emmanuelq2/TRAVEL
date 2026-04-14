"""
app/asr/language_detector.py
─────────────────────────────
Two-stage language identification:
  1. langdetect  — fast, probabilistic (good for ≥ 20 words)
  2. lingua      — ML-backed, better on short/accented utterances

Returns a BCP-47 language code, e.g. "en" or "fr".
Defaults to "en" if detection fails.
"""

from __future__ import annotations

from langdetect import LangDetectException, detect
from lingua import Language, LanguageDetectorBuilder

from app.core.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED = {"en", "fr"}
_DEFAULT = "en"

# Build lingua detector once (loads ~15 MB models)
_lingua = (
    LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.FRENCH)
    .with_minimum_relative_distance(0.15)
    .build()
)


def detect_language(text: str) -> str:
    """
    Return BCP-47 language code for the given text.
    Falls back to lingua for short utterances (< 6 tokens).
    """
    text = text.strip()
    if not text:
        return _DEFAULT

    word_count = len(text.split())

    if word_count >= 6:
        try:
            lang = detect(text)
            if lang in _SUPPORTED:
                logger.debug("lid_langdetect", lang=lang, words=word_count)
                return lang
        except LangDetectException:
            pass

    # Lingua fallback (handles short / accented utterances)
    result = _lingua.detect_language_of(text)
    if result is None:
        logger.warning("lid_lingua_no_result", text_snippet=text[:40])
        return _DEFAULT

    lang = result.iso_code_639_1.name.lower()
    lang = lang if lang in _SUPPORTED else _DEFAULT
    logger.debug("lid_lingua", lang=lang, words=word_count)
    return lang
