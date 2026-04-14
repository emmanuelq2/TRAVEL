"""
app/main.py
────────────
FastAPI application entry point for the Aviation Voice Assistant.

Pipeline overview
─────────────────
  Audio input
      │
      ▼
  VAD  (Silero)          ─── silence stripped, speech windows extracted
      │
      ▼
  LID  (langdetect/lingua) ── language: "en" | "fr"
      │
      ▼
  ASR  ┌─ Batch:    Cohere Transcribe (primary) → Whisper (fallback)
       └─ Streaming: Deepgram Nova-3 (primary)  → Whisper (fallback)
      │
      ▼
  NER  (spaCy + IATA EntityRuler)
       AIRPORT · AIRLINE · FLIGHT_NO · DATE · TIME · CABIN · PAX
      │
      ▼
  Slot Filler  → ParsedUtterance(intent, slots, missing_slots)
      │
      ▼
  Speaker Diarization  (pyannote.audio 3.x) ─ optional, multi-speaker audio
      │
      ▼
  Claude Dialogue Manager  (claude-sonnet-4-6, tool use, prompt caching)
      │
      ▼
  JSON / WebSocket response

Endpoints
─────────
  POST /v1/audio/transcribe  — batch upload, full pipeline, no agent
  POST /v1/audio/query       — batch upload, full pipeline + agent reply
  WS   /v1/chat/stream       — real-time streaming voice chat
  GET  /health               — liveness + provider health
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.audio import router as audio_router
from app.api.v1.chat import router as chat_router
from app.asr.router import get_asr_router
from app.asr.vad import get_vad
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.nlp.ner import get_ner
from app.nlp.slot_filler import get_slot_filler

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    cfg = get_settings()
    logger.info("startup", model=cfg.claude_model, batch_asr=cfg.asr_batch_provider)

    # Pre-load heavy singletons so the first request isn't slow
    get_vad()
    get_ner()
    get_slot_filler()
    # ASR router instantiates providers (connection pools)
    get_asr_router()

    yield

    logger.info("shutdown")


app = FastAPI(
    title="Aviation Voice Assistant",
    description=(
        "Multilingual (EN/FR) airport chatbot. "
        "Pipeline: VAD → LID → ASR → NER → Slot Fill → Diarization → Claude agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(audio_router, prefix="/v1")
app.include_router(chat_router, prefix="/v1")


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
async def health():
    asr_health = await get_asr_router().health()
    return {"status": "ok", "asr": asr_health}
