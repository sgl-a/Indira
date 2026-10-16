from __future__ import annotations

"""
MLX Whisper STT Provider.

Uses mlx-whisper for Apple Silicon optimized speech-to-text transcription.
Drop-in replacement for openai-whisper with significantly better performance
on M-series Macs via the MLX framework.

Requirements:
    brew install ffmpeg
    pip install mlx-whisper
"""

import asyncio
import functools
import logging
import re
import time
import unicodedata

import numpy as np

from src.core.interfaces.stt import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)

# Whisper hallucinates on non-speech: a clap or a lighter click comes back as
# "¡Suscríbete!", a line from the subtitled videos it was trained on. Two
# filters, both here because they're Whisper's quirks, not the conversation's:
#
# 1. Whisper's own no-speech probability, per segment. Measured 2026-10-16:
#    claps / lighter clicks 0.79-0.85; real short answers 0.09-0.18 (short
#    answers in the TTS voice up to 0.56). Its avg_logprob does NOT separate
#    them — a real "¿Sí?" scored lower than the hallucinations — which is why
#    Whisper's built-in skip (no_speech AND low logprob) lets them through.
# 2. Its stock phrases, matched against the WHOLE transcription. A substring
#    match (the old filter in voice mode) dropped real lines such as
#    "me encanta esta música".
_DEFAULT_NO_SPEECH_THRESHOLD = 0.7


def _normalize(text: str) -> str:
    """Lowercase, no accents or punctuation: '¡Suscríbete!' → 'suscribete'."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


_STOCK_PHRASES = {_normalize(p) for p in (
    "¡Suscríbete!", "Suscríbete al canal", "Subscribe",
    "Gracias por ver", "Gracias por ver el video",
    "Thank you.", "Thanks for watching!", "Thank you for watching!",
    "Subtítulos realizados por la comunidad de Amara.org",
    "Subtítulos por la comunidad de Amara.org",
    "Música", "Aplausos", "Risas",
)}
# Never part of real dialogue, so safe to match anywhere in the text
_STOCK_MARKERS = ("amara org",)

# Model size → HuggingFace repo mapping
_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


class WhisperSTTProvider(STTProvider):
    """
    Speech-to-Text Provider using mlx-whisper (Apple Silicon optimized).

    Uses MLX framework for native GPU acceleration on M-series Macs.
    Significantly faster than standard openai-whisper.

    Available models (speed vs accuracy trade-off):
        tiny   — ~39MB,  fastest, lower accuracy
        base   — ~74MB,  good balance for real-time    ← default
        small  — ~244MB, better accuracy
        medium — ~769MB, high accuracy
        large  — ~1.5GB, best accuracy, slowest
        turbo  — ~809MB, large-v3 speed-optimized
    """

    def __init__(self):
        self._model_repo: str = "mlx-community/whisper-base"
        self._language: str | None = None
        self._no_speech_threshold: float = _DEFAULT_NO_SPEECH_THRESHOLD
        self._mlx_whisper = None

    async def initialize(self, config: dict) -> None:
        stt_config = config.get("stt", config)
        model_name = stt_config.get("model", "base")
        self._language = stt_config.get("language", None)
        self._no_speech_threshold = stt_config.get(
            "no_speech_threshold", _DEFAULT_NO_SPEECH_THRESHOLD
        )

        # Resolve model name to HF repo
        if model_name in _MODEL_REPOS:
            self._model_repo = _MODEL_REPOS[model_name]
        elif "/" in model_name:
            # Allow direct HF repo paths
            self._model_repo = model_name
        else:
            logger.warning(
                f"Unknown model '{model_name}', falling back to 'base'. "
                f"Available: {', '.join(_MODEL_REPOS.keys())}"
            )
            self._model_repo = _MODEL_REPOS["base"]

        try:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper
            logger.info(f"mlx-whisper ready | model: {self._model_repo}")
        except ImportError:
            raise ImportError(
                "mlx-whisper is required for WhisperSTTProvider. "
                "Install with: pip install mlx-whisper"
            )

    async def transcribe(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> TranscriptionResult:
        if self._mlx_whisper is None:
            raise RuntimeError("mlx-whisper not initialized")

        start_time = time.time()

        # Convert bytes to numpy float32 array
        audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # mlx-whisper accepts numpy arrays directly — no ffmpeg needed
        options = {
            "path_or_hf_repo": self._model_repo,
            "fp16": True,  # MLX handles fp16 natively on Apple Silicon
            "verbose": False,
        }
        if self._language:
            options["language"] = self._language

        # Run in executor — transcription takes ~0.5-2s and would freeze
        # the event loop (and any background tasks) if called directly
        result = await asyncio.get_event_loop().run_in_executor(
            None, functools.partial(self._mlx_whisper.transcribe, audio_np, **options)
        )

        elapsed = (time.time() - start_time) * 1000
        text = self._speech_text(result)
        language = result.get("language", "en")

        if text:
            logger.debug(
                f"🎤 STT ({elapsed:.0f}ms, lang={language}): {text[:100]}"
            )

        return TranscriptionResult(
            text=text,
            confidence=1.0,
            language=language,
            is_final=True,
        )

    def _speech_text(self, result: dict) -> str:
        """The transcription minus what Whisper hallucinated (see top of module)."""
        segments = result.get("segments")
        if segments is None:
            text = result.get("text", "").strip()
        else:
            kept = []
            for seg in segments:
                if seg.get("no_speech_prob", 0.0) > self._no_speech_threshold:
                    logger.debug(
                        f"🎤 STT dropped non-speech (no_speech_prob "
                        f"{seg['no_speech_prob']:.2f}): {seg.get('text', '').strip()!r}"
                    )
                else:
                    kept.append(seg.get("text", ""))
            text = "".join(kept).strip()

        normalized = _normalize(text)
        if normalized in _STOCK_PHRASES or any(m in normalized for m in _STOCK_MARKERS):
            logger.debug(f"🎤 STT dropped Whisper stock phrase: {text!r}")
            return ""
        return text

    async def shutdown(self) -> None:
        self._mlx_whisper = None
        logger.info("mlx-whisper unloaded")
