"""
app/asr/whisper_provider.py
────────────────────────────
Whisper via OpenAI API — used as universal fallback.

"whisper-1" on the OpenAI API maps to Whisper Large v3 under the hood.
It is the most accurate option for heavily accented or low-resource speech
and handles rare proper nouns (city names, airline codes) better than
smaller models when paired with a prompt hint.
"""

from __future__ import annotations

import io
from typing import AsyncGenerator

from openai import AsyncOpenAI

from app.asr.base import BaseASRProvider, StreamChunk, TranscriptionResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.nlp.iata_vocab import IATA_PROMPT_HINT

logger = get_logger(__name__)


class WhisperProvider(BaseASRProvider):
    name = "whisper"

    def __init__(self) -> None:
        cfg = get_settings()
        self._client = AsyncOpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.whisper_model

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        hint_language: str | None = None,
        audio_format: str = "wav",
    ) -> TranscriptionResult:
        logger.info("whisper_transcribe_start", bytes=len(audio_bytes))

        audio_file = (f"audio.{audio_format}", io.BytesIO(audio_bytes), f"audio/{audio_format}")

        response = await self._client.audio.transcriptions.create(
            model=self._model,
            file=audio_file,
            language=hint_language,        # None → auto-detect
            response_format="verbose_json",
            # Domain hint: primes Whisper decoder toward aviation vocabulary
            prompt=IATA_PROMPT_HINT,
            timestamp_granularities=["word"],
        )

        words = []
        if hasattr(response, "words") and response.words:
            words = [
                {"word": w.word, "start": w.start, "end": w.end, "confidence": 1.0}
                for w in response.words
            ]

        return TranscriptionResult(
            text=(response.text or "").strip(),
            language=getattr(response, "language", None) or hint_language or "en",
            confidence=1.0,
            words=words,
            provider=self.name,
        )

    # ── Streaming (emulated — buffers and transcribes) ────────────────────────

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        hint_language: str | None = None,
        sample_rate: int = 16_000,
    ) -> AsyncGenerator[StreamChunk, None]:
        buf = bytearray()
        async for chunk in audio_chunks:
            buf.extend(chunk)

        result = await self.transcribe(
            bytes(buf),
            hint_language=hint_language,
            audio_format="wav",
        )
        yield StreamChunk(
            text=result.text,
            is_final=True,
            confidence=result.confidence,
            provider=self.name,
        )

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            await self._client.models.retrieve(self._model)
            return True
        except Exception as exc:
            logger.warning("whisper_health_failed", error=str(exc))
            return False
