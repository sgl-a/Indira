from __future__ import annotations

"""
Kokoro TTS Provider (via kokoro-onnx).

Ultra-fast text-to-speech using Kokoro 82M with ONNX runtime.
Runs near real-time on Apple Silicon, no GPU required.
Supports English and Spanish with 48+ voice options.

Models: models/kokoro/kokoro-v1.0.onnx + voices-v1.0.bin
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import soundfile as sf

from src.core.interfaces.tts import TTSProvider, TTSResult, VoiceProfile

logger = logging.getLogger(__name__)

# Default voice mappings for age stages
_AGE_VOICES = {
    "en": {
        "10-15": "af_heart",     # Young female
        "15-20": "af_heart",
        "20-25": "af_bella",
        "25-30": "af_bella",
        "30-40": "af_nicole",
        "40-50": "af_nicole",
        "50-60": "af_sarah",
        "60-70": "af_sarah",
    },
    "es": {
        "10-15": "ef_dora",      # Spanish female
        "15-20": "ef_dora",
        "20-25": "ef_dora",
        "25-30": "ef_dora",
        "30-40": "ef_dora",
        "40-50": "ef_dora",
        "50-60": "ef_dora",
        "60-70": "ef_dora",
    },
}

# Emotion → speed modulation (simulates expressiveness)
_EMOTION_SPEED = {
    "excited": 1.15,
    "cheerful": 1.10,
    "happy": 1.08,
    "animated": 1.12,
    "enthusiastic": 1.12,
    "playful": 1.10,
    "nervous": 1.08,
    "angry": 1.12,
    "frustrated": 1.05,
    "neutral": 1.0,
    "calm": 0.95,
    "thoughtful": 0.95,
    "attentive": 0.98,
    "concerned": 0.93,
    "worried": 0.92,
    "sad": 0.88,
    "melancholic": 0.85,
    "tired": 0.88,
    "warm": 1.02,
    "nostalgic": 0.93,
}


class KokoroTTSProvider(TTSProvider):
    """
    Ultra-fast TTS using Kokoro 82M (ONNX runtime).

    - 82M parameters, ~300MB model
    - Near real-time on Apple Silicon
    - Supports streaming via create_stream()
    - English + Spanish + more languages
    - 48+ voice options
    """

    def __init__(self):
        self.kokoro = None
        self.default_voice: str = "af_heart"
        self.default_lang: str = "en-us"
        self.model_dir: str = "models/kokoro"
        self.speed: float = 1.0

    async def initialize(self, config: dict) -> None:
        """Load Kokoro ONNX model."""
        self.default_voice = config.get("voice", "af_heart")
        self.default_lang = config.get("lang", "en-us")
        self.model_dir = config.get("model_dir", "models/kokoro")
        self.speed = config.get("default_voice", {}).get("speed", 1.0)

        model_path = str(Path(self.model_dir) / "kokoro-v1.0.onnx")
        voices_path = str(Path(self.model_dir) / "voices-v1.0.bin")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Kokoro model not found at {model_path}. "
                "Download from: https://github.com/thewh1teagle/kokoro-onnx/releases"
            )

        logger.info(f"Loading Kokoro TTS model from {self.model_dir}")

        # Load in executor to avoid blocking
        loop = asyncio.get_event_loop()
        self.kokoro = await loop.run_in_executor(
            None, self._load_model, model_path, voices_path
        )

        # Log available voices
        try:
            voices = self.kokoro.get_voices()
            logger.info(
                f"Kokoro TTS initialized | voice: {self.default_voice} | "
                f"lang: {self.default_lang} | {len(voices)} voices available"
            )
        except Exception:
            logger.info(f"Kokoro TTS initialized | voice: {self.default_voice}")

    @staticmethod
    def _load_model(model_path: str, voices_path: str):
        """Synchronous model loading."""
        from kokoro_onnx import Kokoro
        return Kokoro(model_path, voices_path)

    async def synthesize(
        self,
        text: str,
        voice_profile: VoiceProfile | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        """Generate audio from text."""
        start_time = time.time()

        voice = self._select_voice(voice_profile)
        lang = self.default_lang
        speed = self._emotion_speed(emotion)

        # Run synthesis in executor
        loop = asyncio.get_event_loop()
        samples, sr = await loop.run_in_executor(
            None,
            self.kokoro.create,
            text, voice, speed, lang,
        )

        # Convert to WAV bytes
        import io
        buf = io.BytesIO()
        sf.write(buf, samples, sr, format="WAV")
        audio_data = buf.getvalue()

        generation_time = (time.time() - start_time) * 1000
        duration = len(samples) / sr if len(samples) > 0 else 0.0

        logger.debug(
            f"TTS: {len(text)} chars → {duration:.1f}s audio "
            f"in {generation_time:.0f}ms"
        )

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sr,
            duration_seconds=duration,
            generation_time_ms=generation_time,
        )

    async def stream_synthesize(
        self,
        text: str,
        voice_profile: VoiceProfile | None = None,
        emotion: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream audio chunks as they're generated."""
        voice = self._select_voice(voice_profile)

        async for samples, sr in self.kokoro.create_stream(
            text, voice=voice, speed=self.speed, lang=self.default_lang
        ):
            import io
            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="WAV")
            yield buf.getvalue()

    async def speak_directly(
        self,
        text: str,
        voice_profile: VoiceProfile | None = None,
        emotion: str | None = None,
    ) -> None:
        """Generate audio and play directly through speakers."""
        result = await self.synthesize(text, voice_profile, emotion)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(result.audio_data)
            temp_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                "afplay", temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.wait()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _select_voice(self, voice_profile: VoiceProfile | None) -> str:
        """Select voice based on age stage or use default."""
        if voice_profile:
            age_stage = voice_profile.age_stage
            lang_key = "es" if "es" in self.default_lang else "en"
            return _AGE_VOICES.get(lang_key, {}).get(
                age_stage, self.default_voice
            )
        return self.default_voice

    def _emotion_speed(self, emotion: str | None) -> float:
        """Map LLM emotion tag to speed modifier. Defaults to base speed."""
        if not emotion:
            return self.speed

        # Parse multi-word emotions like "cheerful, excited"
        words = [w.strip().lower() for w in emotion.replace(",", " ").split()]
        matched = [_EMOTION_SPEED[w] for w in words if w in _EMOTION_SPEED]

        if not matched:
            return self.speed  # Unknown emotion → default speed

        # Average of matched emotion speeds, scaled by base speed
        avg = sum(matched) / len(matched)
        return self.speed * avg

    async def shutdown(self) -> None:
        """Release model."""
        if self.kokoro is not None:
            del self.kokoro
            self.kokoro = None
            logger.info("Kokoro TTS model unloaded")
