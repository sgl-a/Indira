from __future__ import annotations

"""
Speech-to-Text Provider Interface.

Any STT implementation (Whisper, Parakeet, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""

    text: str
    confidence: float = 1.0
    language: str = "en"
    is_final: bool = True
    # Who is speaking (if speaker identification is available)
    speaker_id: str | None = None
    # Timestamps for word-level alignment
    word_timestamps: list[dict] | None = None


class STTProvider(ABC):
    """
    Abstract interface for Speech-to-Text providers.

    Implementations:
        - WhisperProvider (whisper.cpp / openai-whisper)
        - ParakeetProvider (NVIDIA Parakeet)
    """

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize the provider with configuration."""
        pass

    @abstractmethod
    async def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe an audio chunk.

        Args:
            audio_data: Raw audio bytes (PCM 16-bit)
            sample_rate: Audio sample rate in Hz

        Returns:
            TranscriptionResult with transcribed text
        """
        pass

    @abstractmethod
    async def stream_transcribe(
        self, audio_stream: AsyncIterator[bytes], sample_rate: int = 16000
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Stream transcription for real-time use.

        Yields partial results as speech is detected.
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""
        pass

    async def shutdown(self) -> None:
        """Clean up resources."""
        pass
