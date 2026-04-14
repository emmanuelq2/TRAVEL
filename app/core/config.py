"""
app/core/config.py
──────────────────
Central settings loaded from environment / .env file.

Provider roles
──────────────
  Batch (file upload)  → Cohere Transcribe (primary)  | Whisper (fallback)
  Streaming (WebSocket)→ Deepgram Nova-3   (primary)  | Whisper (fallback)
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ASRProvider(StrEnum):
    cohere = "cohere"
    deepgram = "deepgram"
    whisper = "whisper"         # kept as universal fallback


class Settings(BaseSettings):
    # ── API keys ─────────────────────────────────────────────────────────────
    cohere_api_key: str = ""
    deepgram_api_key: str = ""
    openai_api_key: str = ""          # Whisper fallback
    anthropic_api_key: str = ""
    huggingface_token: str = ""       # pyannote speaker diarization

    # ── ASR routing ──────────────────────────────────────────────────────────
    asr_batch_provider: ASRProvider = ASRProvider.cohere
    asr_stream_provider: ASRProvider = ASRProvider.deepgram
    asr_fallback_provider: ASRProvider = ASRProvider.whisper
    asr_fallback_enabled: bool = True

    # ── Cohere Transcribe ─────────────────────────────────────────────────────
    cohere_transcribe_model: str = "cohere-transcribe-03-2026"

    # ── Deepgram ──────────────────────────────────────────────────────────────
    deepgram_model: str = "nova-3"
    deepgram_hotwords_enabled: bool = True   # inject IATA vocab as keyword boosts

    # ── Whisper (fallback) ────────────────────────────────────────────────────
    whisper_model: str = "whisper-1"         # OpenAI API alias for large-v3

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500

    # ── Diarization ───────────────────────────────────────────────────────────
    diarization_enabled: bool = True
    max_speakers: int = 4

    # ── Claude dialogue manager ────────────────────────────────────────────────
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 1024

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
