"""
app/asr/vad.py
──────────────
Voice Activity Detection using Silero VAD (PyPI package, not torch.hub).

Uses the `silero-vad` PyPI package with ONNX runtime — no torchaudio required,
no hub download, works on CPU-only machines.

Audio loading uses soundfile (WAV/OGG/FLAC) with pydub as fallback for MP3/WebM
so that ffprobe is never required.

Usage
─────
    vad = SileroVAD()
    clean_bytes = vad.extract_speech(raw_audio_bytes)   # silence stripped
    segments     = vad.get_segments(raw_audio_bytes)    # [{start_ms, end_ms}]
"""

from __future__ import annotations

import audioop
import io
import subprocess

import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TARGET_SR = 16_000


class SileroVAD:
    def __init__(self) -> None:
        cfg = get_settings()
        self.threshold = cfg.vad_threshold
        self.min_speech_ms = cfg.vad_min_speech_ms
        self.min_silence_ms = cfg.vad_min_silence_ms

        logger.info("vad_loading_model", backend="silero-vad-pypi/onnx")
        self._model = load_silero_vad(onnx=True)
        logger.info("vad_model_ready")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_segments(
        self, audio_bytes: bytes, sample_rate: int = _TARGET_SR
    ) -> list[dict[str, int]]:
        """Return [{start_ms, end_ms}] speech windows."""
        wav = self._load_mono_16k(audio_bytes, sample_rate)
        raw = get_speech_timestamps(
            wav,
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            return_seconds=False,
        )
        segments = [
            {
                "start_ms": int(t["start"] / sample_rate * 1_000),
                "end_ms":   int(t["end"]   / sample_rate * 1_000),
            }
            for t in raw
        ]
        logger.debug("vad_segments", count=len(segments))
        return segments

    def extract_speech(
        self, audio_bytes: bytes, sample_rate: int = _TARGET_SR
    ) -> bytes:
        """Strip silence — return only speech-containing audio as 16-bit WAV."""
        segments = self.get_segments(audio_bytes, sample_rate)
        if not segments:
            logger.warning("vad_no_speech_detected")
            return audio_bytes

        data = self._load_mono_16k(audio_bytes, sample_rate).numpy()

        chunks = [
            data[int(s["start_ms"] / 1_000 * sample_rate) :
                 int(s["end_ms"]   / 1_000 * sample_rate)]
            for s in segments
        ]
        speech = np.concatenate(chunks)

        buf = io.BytesIO()
        sf.write(buf, speech, sample_rate, format="WAV", subtype="PCM_16")
        logger.debug(
            "vad_speech_extracted",
            original_ms=len(data) * 1_000 // sample_rate,
            speech_ms=len(speech) * 1_000 // sample_rate,
        )
        return buf.getvalue()

    def has_speech(self, audio_bytes: bytes, sample_rate: int = _TARGET_SR) -> bool:
        return bool(self.get_segments(audio_bytes, sample_rate))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_mono_16k(self, audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
        """
        Load audio bytes → float32 mono tensor at `sample_rate`.

        Primary:  soundfile      (WAV / OGG / FLAC — fast, no subprocess)
        Fallback: imageio_ffmpeg (M4A / MP3 / WebM — bundled binary, no ffprobe)
        """
        try:
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        except Exception:
            data, sr = self._decode_via_ffmpeg(audio_bytes, sample_rate)
            return torch.from_numpy(data)   # already mono 16k from ffmpeg

        # Mix down to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample if needed (audioop-lts, no scipy required)
        if sr != sample_rate:
            pcm = (np.clip(data, -1.0, 1.0) * 32_767).astype(np.int16).tobytes()
            pcm, _ = audioop.ratecv(pcm, 2, 1, sr, sample_rate, None)
            data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32_768.0

        return torch.from_numpy(data)

    @staticmethod
    def _decode_via_ffmpeg(audio_bytes: bytes, sample_rate: int) -> tuple[np.ndarray, int]:
        """
        Decode any audio format to float32 mono PCM using the bundled ffmpeg binary.
        Handles M4A, MP3, WebM, AAC — no ffprobe, no pydub needed.

        Uses a temp file for input so ffmpeg can seek (required by MP4/M4A containers
        whose metadata sits at the end of the file).
        """
        import subprocess
        import tempfile
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            cmd = [
                ffmpeg_exe,
                "-hide_banner", "-loglevel", "error",
                "-i", tmp_path,
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ar", str(sample_rate),
                "-ac", "1",
                "pipe:1",
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()}")
            data = np.frombuffer(result.stdout, dtype=np.float32).copy()
        finally:
            import os
            os.unlink(tmp_path)

        return data, sample_rate


# ── Singleton ─────────────────────────────────────────────────────────────────
_vad: SileroVAD | None = None


def get_vad() -> SileroVAD:
    global _vad
    if _vad is None:
        _vad = SileroVAD()
    return _vad
