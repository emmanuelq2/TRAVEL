"""
app/diarization/speaker.py
───────────────────────────
Speaker diarization using pyannote.audio 3.x.

This module answers "who spoke when?" for multi-speaker audio (e.g. a
passenger speaking to a check-in agent, or a family at a kiosk).

Setup
─────
  1. Accept the model license on HuggingFace:
     https://huggingface.co/pyannote/speaker-diarization-3.1
  2. Set HUGGINGFACE_TOKEN in .env

Output
──────
  List of SpeakerSegment:
    speaker  — "SPEAKER_00", "SPEAKER_01", …
    start_s  — segment start in seconds
    end_s    — segment end in seconds
    text     — transcript fragment (populated by merge_with_transcript)

Usage
─────
    diar = SpeakerDiarizer()

    # From file
    segments = await diar.diarize(audio_bytes)

    # Align with ASR word-level timestamps
    aligned = diar.merge_with_transcript(segments, asr_result)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SpeakerSegment:
    speaker: str
    start_s: float
    end_s: float
    text: str = ""


class SpeakerDiarizer:
    def __init__(self) -> None:
        cfg = get_settings()
        if not cfg.huggingface_token:
            logger.warning("diarization_no_hf_token")
        self._max_speakers = cfg.max_speakers
        self._pipeline = self._load_pipeline(cfg.huggingface_token)

    # ── Public API ────────────────────────────────────────────────────────────

    async def diarize(
        self,
        audio_bytes: bytes,
        num_speakers: int | None = None,
    ) -> list[SpeakerSegment]:
        """
        Run speaker diarization on audio bytes.
        `num_speakers` pins the count; None = auto-detect (up to max_speakers).
        """
        if self._pipeline is None:
            logger.warning("diarization_skipped_no_pipeline")
            return [SpeakerSegment(speaker="SPEAKER_00", start_s=0.0, end_s=9999.0)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        kwargs: dict = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        else:
            kwargs["max_speakers"] = self._max_speakers

        diarization = self._pipeline(tmp_path, **kwargs)

        segments: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    speaker=speaker,
                    start_s=round(turn.start, 3),
                    end_s=round(turn.end, 3),
                )
            )

        logger.info(
            "diarization_done",
            segments=len(segments),
            speakers=len({s.speaker for s in segments}),
        )
        return segments

    def merge_with_transcript(
        self,
        segments: list[SpeakerSegment],
        words: list[dict],
    ) -> list[SpeakerSegment]:
        """
        Assign ASR word-level timestamps to speaker segments.
        `words` is the list of {word, start, end} dicts from TranscriptionResult.
        """
        if not words:
            return segments

        for seg in segments:
            seg.text = " ".join(
                w["word"] for w in words if seg.start_s <= w.get("start", 0) < seg.end_s
            )
        return segments

    # ── Pipeline loader ────────────────────────────────────────────────────────

    @staticmethod
    def _load_pipeline(hf_token: str):
        try:
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token or True,
            )
            # Move to GPU if available
            try:
                import torch

                if torch.cuda.is_available():
                    pipeline = pipeline.to(torch.device("cuda"))
            except Exception:
                pass
            logger.info("diarization_pipeline_loaded")
            return pipeline
        except Exception as exc:
            logger.warning("diarization_pipeline_unavailable", error=str(exc))
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
_diarizer: SpeakerDiarizer | None = None


def get_diarizer() -> SpeakerDiarizer:
    global _diarizer
    if _diarizer is None:
        _diarizer = SpeakerDiarizer()
    return _diarizer
