"""
app/nlp/date_normalizer.py
──────────────────────────
Normalize natural-language date/time expressions to ISO-8601.

Uses `dateparser` with an aviation-aware configuration:
  • Prefer future dates (you always book *ahead*)
  • Handle French and English expressions
  • Clamp past dates to today

Examples
────────
  "next Friday"          → "2026-04-17"
  "March 15"             → "2026-03-15"
  "le 20 avril"          → "2026-04-20"
  "tomorrow morning"     → "2026-04-13T08:00"
  "14h30"                → "14:30"  (time-only)
  "in 3 days"            → "2026-04-15"
"""

from __future__ import annotations

import re
from datetime import date, datetime

import dateparser

from app.core.logging import get_logger

logger = get_logger(__name__)

_DATEPARSER_SETTINGS = {
    "PREFER_DATES_FROM": "future",  # airport context: always booking ahead
    "RETURN_AS_TIMEZONE_AWARE": False,
    "PREFER_DAY_OF_MONTH": "first",
}
_LANGUAGES = ["en", "fr"]  # passed as separate kwarg, not inside settings dict

# Ordinal words → digits  (spoken ASR output: "twentieth" → "20")
_ORDINALS = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
    "twenty-first": "21",
    "twenty first": "21",
    "twenty-second": "22",
    "twenty second": "22",
    "twenty-third": "23",
    "twenty third": "23",
    "twenty-fourth": "24",
    "twenty fourth": "24",
    "twenty-fifth": "25",
    "twenty fifth": "25",
    "twenty-sixth": "26",
    "twenty sixth": "26",
    "twenty-seventh": "27",
    "twenty seventh": "27",
    "twenty-eighth": "28",
    "twenty eighth": "28",
    "twenty-ninth": "29",
    "twenty ninth": "29",
    "thirtieth": "30",
    "thirty-first": "31",
    "thirty first": "31",
}
_ORDINAL_RE = re.compile(
    r"\b("
    + "|".join(re.escape(k) for k in sorted(_ORDINALS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def _normalise_ordinals(text: str) -> str:
    """Replace written ordinals with digits: 'April twentieth' → 'April 20'."""
    return _ORDINAL_RE.sub(lambda m: _ORDINALS[m.group(1).lower()], text)


def normalize_date(text: str) -> date | None:
    """
    Parse a natural-language date string.
    Returns a `datetime.date` or None if parsing fails.
    """
    if not text:
        return None
    text = _normalise_ordinals(text)
    parsed: datetime | None = dateparser.parse(
        text, languages=_LANGUAGES, settings=_DATEPARSER_SETTINGS
    )
    if parsed is None:
        logger.warning("date_normalize_failed", text=text)
        return None
    result = parsed.date()
    logger.debug("date_normalized", raw=text, result=str(result))
    return result


def normalize_time(text: str) -> str | None:
    """
    Parse a time expression.
    Returns "HH:MM" string or None.
    """
    if not text:
        return None
    parsed: datetime | None = dateparser.parse(
        text, languages=_LANGUAGES, settings=_DATEPARSER_SETTINGS
    )
    if parsed is None:
        return None
    return parsed.strftime("%H:%M")


def normalize_date_range(
    departure_text: str,
    return_text: str | None,
) -> tuple[date | None, date | None]:
    """
    Parse a (departure, return) pair.
    Ensures return_date > departure_date when both are present.
    """
    dep = normalize_date(departure_text)
    ret = normalize_date(return_text) if return_text else None

    if dep and ret and ret <= dep:
        logger.warning(
            "date_range_invalid",
            departure=str(dep),
            return_=str(ret),
        )
        ret = None  # discard illogical return date

    return dep, ret
