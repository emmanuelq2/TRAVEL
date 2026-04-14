"""
app/asr/router.py
──────────────────
ASR Router — selects provider per request type with failover.

Routing table
─────────────
┌──────────────────┬───────────────────────┬────────────────────────┐
│  Request type    │  Primary              │  Fallback              │
├──────────────────┼───────────────────────┼────────────────────────┤
│  Batch (file)    │  CohereProvider       │  WhisperProvider       │
│  WebSocket (RT)  │  DeepgramProvider     │  WhisperProvider       │
└──────────────────┴───────────────────────┴────────────────────────┘

The VAD stage runs before any provider is called; callers pass the
already-trimmed audio so providers never see leading/trailing silence.
"""

from __future__ import annotations

from typing import AsyncGenerator

from app.asr.base import BaseASRProvider, StreamChunk, TranscriptionResult
from app.asr.cohere_provider import CohereProvider
from app.asr.deepgram_provider import DeepgramProvider
from app.asr.whisper_provider import WhisperProvider
from app.core.config import ASRProvider, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ASRRouter:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._providers: dict[str, BaseASRProvider] = {
            ASRProvider.cohere: CohereProvider(),
            ASRProvider.deepgram: DeepgramProvider(),
            ASRProvider.whisper: WhisperProvider(),
        }

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        hint_language: str | None = None,
        audio_format: str = "wav",
        force_provider: str | None = None,
    ) -> TranscriptionResult:
        if force_provider:
            return await self._providers[force_provider].transcribe(
                audio_bytes, hint_language=hint_language, audio_format=audio_format
            )

        primary = self._providers[self._settings.asr_batch_provider]
        try:
            result = await primary.transcribe(
                audio_bytes, hint_language=hint_language, audio_format=audio_format
            )
            return result
        except Exception as exc:
            logger.warning(
                "asr_batch_primary_failed",
                provider=primary.name,
                error=str(exc),
                fallback_enabled=self._settings.asr_fallback_enabled,
            )
            if not self._settings.asr_fallback_enabled:
                raise

        fallback = self._providers[self._settings.asr_fallback_provider]
        logger.info("asr_batch_using_fallback", provider=fallback.name)
        return await fallback.transcribe(
            audio_bytes, hint_language=hint_language, audio_format=audio_format
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        hint_language: str | None = None,
        sample_rate: int = 16_000,
    ) -> AsyncGenerator[StreamChunk, None]:
        primary = self._providers[self._settings.asr_stream_provider]
        try:
            async for chunk in primary.transcribe_stream(
                audio_chunks, hint_language=hint_language, sample_rate=sample_rate
            ):
                yield chunk
        except Exception as exc:
            logger.warning(
                "asr_stream_primary_failed",
                provider=primary.name,
                error=str(exc),
            )
            if not self._settings.asr_fallback_enabled:
                raise
            fallback = self._providers[self._settings.asr_fallback_provider]
            logger.info("asr_stream_using_fallback", provider=fallback.name)
            async for chunk in fallback.transcribe_stream(
                audio_chunks, hint_language=hint_language, sample_rate=sample_rate
            ):
                yield chunk

    # ── Health ────────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, bool]:
        results = {}
        for name, provider in self._providers.items():
            results[name] = await provider.health_check()
        return results


_router: ASRRouter | None = None


def get_asr_router() -> ASRRouter:
    global _router
    if _router is None:
        _router = ASRRouter()
    return _router
