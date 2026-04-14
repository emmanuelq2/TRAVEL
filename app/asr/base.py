"""
app/asr/base.py
───────────────
Abstract base class shared by all ASR providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator


@dataclass
class TranscriptionResult:
    text: str
    language: str                       # detected BCP-47 code, e.g. "en", "fr"
    confidence: float = 1.0
    words: list[dict] = field(default_factory=list)   # [{word, start, end, conf}]
    provider: str = ""


@dataclass
class StreamChunk:
    text: str
    is_final: bool
    confidence: float = 1.0
    provider: str = ""


class BaseASRProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        hint_language: str | None = None,
        audio_format: str = "wav",
    ) -> TranscriptionResult: ...

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        *,
        hint_language: str | None = None,
        sample_rate: int = 16_000,
    ) -> AsyncGenerator[StreamChunk, None]: ...

    async def health_check(self) -> bool:
        return True
