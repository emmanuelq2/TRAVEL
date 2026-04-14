"""
app/api/v1/audio.py
────────────────────
POST /v1/audio/transcribe  — batch file upload → full pipeline response
POST /v1/audio/query       — convenience: transcribe + NLU + agent reply

Full pipeline per request
──────────────────────────
  1. VAD      — strip silence from uploaded audio
  2. LID      — detect language from first-pass transcript
  3. ASR      — Cohere Transcribe (batch primary) → Whisper (fallback)
  4. NER      — extract aviation entities
  5. Slots    — fill flight-intent slots
  6. Diarize  — speaker attribution (async, non-blocking for /transcribe)
  7. Agent    — Claude dialogue turn (only on /query)
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent.dialogue_manager import get_dialogue_manager, get_or_create_session
from app.asr.language_detector import detect_language
from app.asr.router import get_asr_router
from app.asr.vad import get_vad
from app.core.config import get_settings
from app.core.logging import get_logger
from app.diarization.speaker import SpeakerSegment, get_diarizer
from app.nlp.slot_filler import ParsedUtterance, get_slot_filler

router = APIRouter(prefix="/audio", tags=["audio"])
logger = get_logger(__name__)


# ── Response models ────────────────────────────────────────────────────────────


class TranscribeResponse(BaseModel):
    transcript: str
    language: str
    confidence: float
    provider: str
    entities: list[dict]  # [{label, text}, ...]
    slots: dict
    speakers: list[dict]  # [{speaker, start_s, end_s, text}]


class QueryResponse(BaseModel):
    transcript: str
    language: str
    intent: str
    missing_slots: list[str]
    reply: str
    session_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (wav/mp3/ogg/webm)"),
    language: str | None = Form(None, description="BCP-47 hint: 'en' or 'fr'"),
    diarize: bool = Form(False, description="Run speaker diarization"),
):
    """
    Full pipeline without agent: returns transcript, NER, slots, and
    optional speaker diarization.
    """
    audio_bytes = await file.read()

    # Sanitise language hint — reject Swagger placeholder "string"
    lang_hint = (
        language
        if language and language.lower() not in ("string", "none", "")
        else None
    )

    # 1. VAD
    vad = get_vad()
    clean_audio = vad.extract_speech(audio_bytes)  # always outputs 16-bit WAV

    # 2. ASR
    asr = get_asr_router()
    result = await asr.transcribe(
        clean_audio, hint_language=lang_hint, audio_format="wav"
    )

    # 3. LID (use detected language from ASR, or run detector on transcript)
    lang = result.language or detect_language(result.text)

    # 4. NER + slot filling
    filler = get_slot_filler()
    parsed: ParsedUtterance = filler.parse(result.text, language=lang)

    # 5. Diarization (optional)
    speakers: list[SpeakerSegment] = []
    if diarize and get_settings().diarization_enabled:
        diarizer = get_diarizer()
        segs = await diarizer.diarize(clean_audio)
        speakers = diarizer.merge_with_transcript(segs, result.words)

    return TranscribeResponse(
        transcript=result.text,
        language=lang,
        confidence=result.confidence,
        provider=result.provider,
        entities=parsed.raw_entities,  # type: ignore[arg-type]
        slots=parsed.slots.model_dump(mode="json"),
        speakers=[
            {
                "speaker": s.speaker,
                "start_s": s.start_s,
                "end_s": s.end_s,
                "text": s.text,
            }
            for s in speakers
        ],
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    language: str | None = Form(None),
):
    """
    Full pipeline with agent: transcribe + NLU + Claude dialogue turn.
    Maintains multi-turn conversation via session_id.
    """
    audio_bytes = await file.read()
    sid = session_id or str(uuid.uuid4())

    # Sanitise language hint — reject Swagger placeholder "string"
    lang_hint = (
        language
        if language and language.lower() not in ("string", "none", "")
        else None
    )

    # VAD → ASR
    vad = get_vad()
    if not vad.has_speech(audio_bytes):
        raise HTTPException(status_code=422, detail="No speech detected in audio.")

    clean = vad.extract_speech(audio_bytes)  # always outputs 16-bit WAV
    asr = get_asr_router()
    result = await asr.transcribe(clean, hint_language=lang_hint, audio_format="wav")

    lang = result.language or detect_language(result.text)

    # NLU
    filler = get_slot_filler()
    parsed = filler.parse(result.text, language=lang)

    # Agent
    manager = get_dialogue_manager()
    session = get_or_create_session(sid)
    reply = await manager.respond(session, parsed, result.text)

    return QueryResponse(
        transcript=result.text,
        language=lang,
        intent=parsed.intent,
        missing_slots=parsed.missing_slots,
        reply=reply,
        session_id=sid,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _guess_format(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    return {"mp3": "mp3", "ogg": "ogg", "webm": "webm", "m4a": "mp4"}.get(ext, "wav")
