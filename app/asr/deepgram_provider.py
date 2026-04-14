"""
app/asr/deepgram_provider.py
─────────────────────────────
Real-time streaming transcription via Deepgram Nova-3.

SDK v6.1.1 API (Fern-generated — incompatible with v3/v5 docs)
───────────────────────────────────────────────────────────────
  Client init : AsyncDeepgramClient(api_key=...)
                No DeepgramClientOptions — removed in v6.

  Batch       : await client.listen.v1.media.transcribe_file(
                    request=audio_bytes,   # NOT a FileSource object
                    model="nova-3",
                    language="en",
                    ...kwargs...
                )
                Response: response.body.results.channels[0].alternatives[0]

  Streaming   : async with client.listen.v1.connect(model=..., ...) as ws:
                    await ws.send_media(chunk)
                    async for msg in ws:
                        if isinstance(msg, ListenV1Results): ...

Aviation keyword boosting
──────────────────────────
Deepgram `keyterm` param accepts "word:boost" strings.
We inject IATA codes + city/airport names at boost 0.8.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from deepgram import AsyncDeepgramClient
from deepgram.listen.v1 import (
    ListenV1Metadata,
    ListenV1Results,
    ListenV1SpeechStarted,
    ListenV1UtteranceEnd,
)

from app.asr.base import BaseASRProvider, StreamChunk, TranscriptionResult
from app.core.config import get_settings
from app.core.logging import get_logger
from app.nlp.iata_vocab import get_iata_keywords

logger = get_logger(__name__)


class DeepgramProvider(BaseASRProvider):
    name = "deepgram"

    def __init__(self) -> None:
        cfg = get_settings()
        self._client = AsyncDeepgramClient(api_key=cfg.deepgram_api_key)
        self._model = cfg.deepgram_model
        self._hotwords_enabled = cfg.deepgram_hotwords_enabled

    # ── Batch ─────────────────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        hint_language: str | None = None,
        audio_format: str = "wav",
    ) -> TranscriptionResult:
        kwargs: dict = dict(
            request=audio_bytes,
            model=self._model,
            smart_format=True,
            punctuate=True,
            utterances=True,
        )
        if hint_language:
            kwargs["language"] = hint_language
        if self._hotwords_enabled:
            kwargs["keyterm"] = self._build_keyterm()

        response = await self._client.listen.v1.media.transcribe_file(**kwargs)

        body = response.body
        channel = body.results.channels[0]
        alt = channel.alternatives[0]

        detected_lang = getattr(body.metadata, "language", None) or hint_language or "en"
        words = [
            {
                "word":       w.word,
                "start":      w.start,
                "end":        w.end,
                "confidence": w.confidence,
            }
            for w in (alt.words or [])
        ]

        return TranscriptionResult(
            text=alt.transcript.strip(),
            language=detected_lang,
            confidence=alt.confidence or 1.0,
            words=words,
            provider=self.name,
        )

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        hint_language: str | None = None,
        sample_rate: int = 16_000,
    ) -> AsyncGenerator[StreamChunk, None]:
        connect_kwargs: dict = dict(
            model=self._model,
            sample_rate=sample_rate,
            channels=1,
            encoding="linear16",
            smart_format=True,
            punctuate=True,
            interim_results=True,
        )
        if hint_language:
            connect_kwargs["language"] = hint_language
        if self._hotwords_enabled:
            connect_kwargs["keyterm"] = self._build_keyterm()

        queue: asyncio.Queue[StreamChunk | None] = asyncio.Queue()

        async with self._client.listen.v1.connect(**connect_kwargs) as ws:
            # Feed audio chunks to Deepgram in a background task
            async def _feeder() -> None:
                async for chunk in audio_chunks:
                    await ws.send_media(chunk)
                await ws.send_finalize()
                await ws.send_close_stream()

            feeder_task = asyncio.create_task(_feeder())

            # Consume messages from the socket
            async for msg in ws:
                if isinstance(msg, ListenV1Results):
                    alt = msg.channel.alternatives[0]
                    if alt.transcript:
                        await queue.put(
                            StreamChunk(
                                text=alt.transcript,
                                is_final=bool(msg.is_final),
                                confidence=alt.confidence or 1.0,
                                provider=self.name,
                            )
                        )
                elif isinstance(msg, (ListenV1Metadata, ListenV1SpeechStarted, ListenV1UtteranceEnd)):
                    pass  # informational — ignore

            await feeder_task

        # Drain any buffered chunks
        while not queue.empty():
            item = queue.get_nowait()
            if item is not None:
                yield item

        # Yield buffered items accumulated during socket iteration
        # (queue is populated in the feeder; drain here)
        # The above loop already drains — this generator returns naturally.

    # ── Keywords ──────────────────────────────────────────────────────────────

    def _build_keyterm(self) -> list[str]:
        return list(get_iata_keywords())

    # ── Health check ──────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        try:
            await self._client.manage.v1.projects.list()
            return True
        except Exception as exc:
            logger.warning("deepgram_health_failed", error=str(exc))
            return False
