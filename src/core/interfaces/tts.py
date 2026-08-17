from __future__ import annotations

"""
Text-to-Speech Provider Interface.

Any TTS implementation (Chatterbox, F5-TTS, Bark, etc.) must implement this interface.
Supports voice cloning, emotion control, and age-based voice profiles.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator


@dataclass
class VoiceProfile:
    """
    A voice identity for a specific age stage.

    Each age stage has its own voice profile with reference audio
    that defines how the AI should sound at that age.
    """

    id: str
    name: str
    age_stage: str
    reference_audio_path: str
    # Voice modification parameters
    pitch_shift: float = 0.0  # semitones
    speed: float = 1.0
    breathiness: float = 0.0
    tremor: float = 0.0  # for elderly voice


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""

    audio_data: bytes
    sample_rate: int = 22050
    duration_seconds: float = 0.0
    generation_time_ms: float = 0.0


class TTSProvider(ABC):
    """
    Abstract interface for Text-to-Speech providers.

    Implementations:
        - SystemTTSProvider (macOS 'say' command — fallback)
        - ChatterboxProvider (Resemble AI Chatterbox)
        - F5TTSProvider (F5-TTS)
        - BarkProvider (Suno Bark)
    """

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize the provider with configuration."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_profile: VoiceProfile | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        """
        Generate audio from text.

        Args:
            text: The text to speak
            voice_profile: Voice identity and age characteristics
            emotion: Emotional intent (e.g., "warm", "angry", "sad")

        Returns:
            TTSResult with audio data
        """
        pass

    async def shutdown(self) -> None:
        """Clean up resources."""
        pass
