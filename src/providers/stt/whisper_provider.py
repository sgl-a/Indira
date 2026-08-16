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
import time
from typing import AsyncIterator

import numpy as np

from src.core.interfaces.stt import STTProvider, TranscriptionResult

logger = logging.getLogger(__name__)

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
        self._mlx_whisper = None

    async def initialize(self, config: dict) -> None:
        stt_config = config.get("stt", config)
        model_name = stt_config.get("model", "base")
        self._language = stt_config.get("language", None)

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
        text = result.get("text", "").strip()
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

    async def stream_transcribe(
        self, audio_stream: AsyncIterator[bytes], sample_rate: int = 16000
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Stream transcription by processing audio chunks.

        Accumulates audio until a pause is detected, then transcribes.
        """
        buffer = bytearray()
        # Process in chunks of ~2 seconds
        chunk_size = sample_rate * 2 * 4  # 2 seconds * 4 bytes per float32 sample

        async for chunk in audio_stream:
            buffer.extend(chunk)

            if len(buffer) >= chunk_size:
                result = await self.transcribe(bytes(buffer), sample_rate)
                buffer.clear()

                if result.text:
                    yield result

        # Process remaining buffer
        if buffer:
            result = await self.transcribe(bytes(buffer), sample_rate)
            if result.text:
                yield result

    def get_supported_languages(self) -> list[str]:
        return [
            "en", "es", "de", "fr", "it", "pt", "nl", "ja", "zh", "ko",
            "ar", "hi", "ru", "pl", "tr", "vi", "th", "uk", "cs", "da",
        ]

    async def shutdown(self) -> None:
        self._mlx_whisper = None
        logger.info("mlx-whisper unloaded")
