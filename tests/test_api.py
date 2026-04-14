"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI endpoints.
Requires: server running OR uses httpx TestClient (no real server needed).

ASR providers are mocked so no API keys are needed.
The Claude dialogue manager is also mocked for /query.

Run with:
    pytest tests/test_api.py -v
"""

import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.asr.base import TranscriptionResult
from app.main import app
from app.nlp.slot_filler import FlightSlots, Intent, ParsedUtterance


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_wav(duration_s: float = 1.0, sample_rate: int = 16_000) -> bytes:
    """Generate a minimal valid WAV file (silence) for upload tests."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


_MOCK_TRANSCRIPTION = TranscriptionResult(
    text="I want to fly from Paris to Manila on April 20 business class",
    language="en",
    confidence=0.97,
    provider="cohere",
    words=[],
)

_MOCK_PARSED = ParsedUtterance(
    intent=Intent.search_flight,
    slots=FlightSlots(origin="CDG", destination="MNL"),
    raw_entities=[
        {"label": "AIRPORT", "text": "Paris"},
        {"label": "AIRPORT", "text": "Manila"},
        {"label": "DATE",    "text": "April 20"},
        {"label": "CABIN",   "text": "business"},
    ],
    missing_slots=["departure_date"],
    language="en",
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    # Patch heavy singletons that the lifespan tries to pre-load at startup.
    # SileroVAD needs torchaudio; spaCy NER and ASR router need API credentials.
    # All real calls are mocked at the endpoint level in each test.
    with (
        patch("app.main.get_vad"),
        patch("app.main.get_ner"),
        patch("app.main.get_slot_filler"),
        patch("app.main.get_asr_router"),
    ):
        with TestClient(app) as c:
            yield c


# ── Health endpoint ────────────────────────────────────────────────────────────

def test_health(client):
    mock_router = MagicMock()
    mock_router.health = AsyncMock(return_value={"cohere": True, "deepgram": True, "whisper": True})
    with patch("app.main.get_asr_router", return_value=mock_router):
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "asr" in data


# ── POST /v1/audio/transcribe ──────────────────────────────────────────────────

@patch("app.api.v1.audio.get_vad")
@patch("app.api.v1.audio.get_asr_router")
def test_transcribe_endpoint(mock_router_factory, mock_vad_factory, client):
    wav = _make_wav()

    # Mock VAD: speech detected, return audio unchanged
    mock_vad = mock_vad_factory.return_value
    mock_vad.extract_speech.return_value = wav
    mock_vad.has_speech.return_value = True

    # Mock ASR: return canned transcription
    mock_router = mock_router_factory.return_value
    mock_router.transcribe = AsyncMock(return_value=_MOCK_TRANSCRIPTION)

    resp = client.post(
        "/v1/audio/transcribe",
        files={"file": ("test.wav", wav, "audio/wav")},
        data={"language": "en", "diarize": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == _MOCK_TRANSCRIPTION.text
    assert body["language"] == "en"
    assert body["provider"] == "cohere"
    assert "slots" in body
    assert "entities" in body


# ── POST /v1/audio/query ───────────────────────────────────────────────────────

@patch("app.api.v1.audio.get_vad")
@patch("app.api.v1.audio.get_asr_router")
@patch("app.api.v1.audio.get_slot_filler")
@patch("app.api.v1.audio.get_dialogue_manager")
def test_query_endpoint(
    mock_dm_factory, mock_filler_factory, mock_router_factory, mock_vad_factory, client
):
    wav = _make_wav()

    mock_vad = mock_vad_factory.return_value
    mock_vad.extract_speech.return_value = wav
    mock_vad.has_speech.return_value = True

    mock_router = mock_router_factory.return_value
    mock_router.transcribe = AsyncMock(return_value=_MOCK_TRANSCRIPTION)

    mock_filler = mock_filler_factory.return_value
    mock_filler.parse.return_value = _MOCK_PARSED

    mock_dm = mock_dm_factory.return_value
    mock_dm.respond = AsyncMock(
        return_value="Sure! What date would you like to fly from Paris to Manila?"
    )

    resp = client.post(
        "/v1/audio/query",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == _MOCK_TRANSCRIPTION.text
    assert body["intent"] == "SEARCH_FLIGHT"
    assert "session_id" in body
    assert "reply" in body
    assert "Manila" in body["reply"] or "date" in body["reply"].lower()


def test_query_no_speech(client):
    """VAD detecting no speech should return 422."""
    wav = _make_wav(duration_s=0.1)

    with (
        patch("app.api.v1.audio.get_vad") as mock_vad_factory,
        patch("app.api.v1.audio.get_asr_router"),
    ):
        mock_vad = mock_vad_factory.return_value
        mock_vad.has_speech.return_value = False

        resp = client.post(
            "/v1/audio/query",
            files={"file": ("silent.wav", wav, "audio/wav")},
        )
    assert resp.status_code == 422
