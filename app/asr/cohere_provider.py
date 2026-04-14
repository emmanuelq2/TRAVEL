"""
app/asr/cohere_provider.py
──────────────────────────
Batch transcription via Cohere Transcribe API.

Cohere Transcribe is optimised for accuracy over latency — ideal for
uploaded audio files (boarding pass scans, recorded queries, IVR clips).

API notes
─────────
  • SDK: cohere>=5.0   (cohere.Client / cohere.AsyncClient)
  • Endpoint: co.transcribe(file=<bytes|BinaryIO>, model="transcribe-v1")
  • Response: response.text  (full transcript)
             response.segments  (word-level timings if requested)
  • No streaming support — use DeepgramProvider for real-time.
  • Language is auto-detected; pass `language` hint to improve accuracy.
"""

from __future__ import annotations

import io
from typing import AsyncGenerator

import cohere

from app.asr.base import BaseASRProvider, StreamChunk, TranscriptionResult
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CohereProvider(BaseASRProvider):
    name = "cohere"

    def __init__(self) -> None:
        cfg = get_settings()
        self._client = cohere.AsyncClientV2(api_key=cfg.cohere_api_key)
        self._model = cfg.cohere_transcribe_model

    # ── Batch transcription ───────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        hint_language: str | None = None,
        audio_format: str = "wav",
    ) -> TranscriptionResult:
        logger.info(
            "cohere_transcribe_start",
            bytes=len(audio_bytes),
            hint_language=hint_language,
        )

        kwargs: dict = {
            "model": self._model,
            "file": (f"audio.{audio_format}", io.BytesIO(audio_bytes), f"audio/{audio_format}"),
            "language": hint_language or "en",
        }

        response = await self._client.audio.transcriptions.create(**kwargs)

        logger.debug("cohere_raw_response", fields=list(response.__fields_set__) if hasattr(response, '__fields_set__') else dir(response))

        text = response.text
        if text is None:
            raise ValueError(f"Cohere returned null transcript. Response: {response}")
        text = text.strip()
        detected_lang = getattr(response, "language", None) or hint_language or "en"

        logger.info(
            "cohere_transcribe_done",
            chars=len(text),
            language=detected_lang,
        )
        return TranscriptionResult(
            text=text,
            language=detected_lang,
            confidence=1.0,          # Cohere does not expose per-token confidence
            provider=self.name,
        )

    # ── Streaming (not supported — raises to trigger failover) ────────────────

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        hint_language: str | None = None,
        sample_rate: int = 16_000,
    ) -> AsyncGenerator[StreamChunk, None]:
        raise NotImplementedError(
            "Cohere Transcribe does not support streaming. "
            "Configure asr_stream_provider=deepgram."
        )
        # make the type-checker happy
        yield  # type: ignore[misc]

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            # Lightweight check: list models (no audio needed)
            await self._client.models.list()
            return True
        except Exception as exc:
            logger.warning("cohere_health_failed", error=str(exc))
            return False
