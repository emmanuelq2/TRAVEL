"""
tests/test_date_normalizer.py
──────────────────────────────
Unit tests for date/time normalization.
No API keys required.
"""

from datetime import date, timedelta

from app.nlp.date_normalizer import normalize_date, normalize_date_range, normalize_time

TODAY = date.today()


def test_iso_date():
    assert normalize_date("2026-07-15") == date(2026, 7, 15)


def test_natural_date():
    d = normalize_date("July 15 2026")
    assert d == date(2026, 7, 15)


def test_tomorrow():
    d = normalize_date("tomorrow")
    assert d == TODAY + timedelta(days=1)


def test_prefers_future():
    # "March 1" should resolve to a future date, not past
    d = normalize_date("March 1")
    assert d is not None
    assert d >= TODAY


def test_french_date():
    d = normalize_date("le 20 avril 2026")
    assert d == date(2026, 4, 20)


def test_invalid_returns_none():
    assert normalize_date("not a date at all xyz") is None


def test_empty_returns_none():
    assert normalize_date("") is None


# ── normalize_time ─────────────────────────────────────────────────────────────


def test_time_hhmm():
    assert normalize_time("14:30") == "14:30"


def test_time_ampm():
    t = normalize_time("8am")
    assert t == "08:00"


def test_time_invalid():
    assert normalize_time("not a time") is None


# ── normalize_date_range ───────────────────────────────────────────────────────


def test_valid_range():
    dep, ret = normalize_date_range("April 20 2026", "April 27 2026")
    assert dep == date(2026, 4, 20)
    assert ret == date(2026, 4, 27)


def test_invalid_range_discards_return():
    # return before departure → return should be discarded
    dep, ret = normalize_date_range("April 27 2026", "April 20 2026")
    assert dep == date(2026, 4, 27)
    assert ret is None


def test_one_way():
    dep, ret = normalize_date_range("May 5 2026", None)
    assert dep == date(2026, 5, 5)
    assert ret is None
