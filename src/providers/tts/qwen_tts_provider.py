from __future__ import annotations

"""
Qwen3-TTS Provider (via mlx-audio on Apple Silicon).

High-quality multilingual TTS with instruction-driven emotion control
and Voice Design for age-based voice changes. Runs locally via MLX.

Features:
- Instruction-driven emotion: "Speak with sadness" maps to LLM emotion tags
- Voice Design: describe a voice per age range (no reference audio needed)
- Spanish as first-class language
- ~700MB on Apple Silicon (0.6B 8-bit model)
"""

import asyncio
import io
import logging
import time

import numpy as np
import soundfile as sf

from src.core.interfaces.tts import TTSProvider, TTSResult, VoiceProfile

logger = logging.getLogger(__name__)

# Language code → full name mapping (Qwen3-TTS uses full language names)
_LANG_MAP = {
    "es": "spanish",
    "en": "english",
    "en-us": "english",
    "en-gb": "english",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ru": "russian",
}

# Emotion tag → TTS instruction (maps LLM [emoción] output to Qwen3-TTS instruct)
_EMOTION_INSTRUCTIONS = {
    # Spanish emotion tags (from LLM output)
    "contenta": "Speak cheerfully and warmly",
    "feliz": "Speak with happiness and joy",
    "triste": "Speak with sadness, softly",
    "enojada": "Speak with frustration and intensity",
    "frustrada": "Speak with mild frustration",
    "entusiasmada": "Speak with excitement and energy",
    "curiosa": "Speak with curiosity and interest",
    "nostálgica": "Speak softly with a hint of longing",
    "cálida": "Speak warmly and gently",
    "reflexiva": "Speak thoughtfully, with pauses",
    "sonriente": "Speak with a smile in your voice",
    "nerviosa": "Speak with slight nervousness",
    "preocupada": "Speak with concern, gently",
    "seria": "Speak seriously, with gravitas",
    "risas": "Speak with laughter in your voice",
    "risas leves": "Speak with a slight chuckle",
    "cansada": "Speak tiredly, slowly",
    "sorprendida": "Speak with surprise",
    "tierna": "Speak tenderly and softly",
    "irónica": "Speak with dry irony",
    "en shock": "Speak with shock, confusion, and disbelief",
    "asustada": "Speak with fear, panic, and urgency",
    "confundida": "Speak with confusion and uncertainty",
    "divertida": "Speak with amusement and lightheartedness",

    # English fallbacks
    "neutral": "Speak naturally and conversationally",
    "warm": "Speak warmly and gently",
    "happy": "Speak with happiness",
    "sad": "Speak with sadness",
    "angry": "Speak with anger",
    "excited": "Speak with excitement",
    "calm": "Speak calmly and peacefully",
}


class QwenTTSProvider(TTSProvider):
    """
    TTS using Qwen3-TTS via mlx-audio on Apple Silicon.

    - 0.6B parameters, ~700MB (8-bit quantized)
    - Instruction-driven emotion control via 'instruct' parameter
    - Voice Design for age-based voice changes
    - Spanish as first-class language
    - Real-time on Apple Silicon via MLX
    """

    def __init__(self):
        self._model = None
        self._model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
        self._voice: str = "serena"  # Built-in: serena, vivian, aura, sohee, ono_anna (F) | ryan, aiden, eric, dylan, uncle_fu (M)
        self._speed: float = 1.0
        self._lang_code: str = "es"

    async def initialize(self, config: dict) -> None:
        """Load Qwen3-TTS model via mlx-audio."""
        self._model_name = config.get(
            "model", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
        )
        self._voice = config.get("voice", "serena")
        self._speed = config.get("default_voice", {}).get("speed", 1.0)
        lang_code = config.get("lang", "es")
        # Map language code to full name for Qwen3-TTS
        self._lang_code = _LANG_MAP.get(lang_code, lang_code)

        logger.info(f"Loading Qwen3-TTS model: {self._model_name}")

        # Load in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)

        logger.info(
            f"Qwen3-TTS initialized | model: {self._model_name} | "
            f"voice: {self._voice} | lang: {self._lang_code}"
        )

    def _load_model(self):
        """Synchronous model loading."""
        try:
            from mlx_audio.tts import load
            return load(self._model_name)
        except ImportError:
            raise ImportError(
                "mlx-audio is required for Qwen3-TTS. "
                "Install with: pip install mlx-audio"
            )

    async def synthesize(
        self,
        text: str,
        voice_profile: VoiceProfile | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        """Generate audio from text with emotion and age control."""
        if not self._model:
            raise RuntimeError("Qwen3-TTS not initialized")

        start_time = time.time()

        # Build emotion instruction for CustomVoice instruct
        instruct = self._get_emotion_instruction(emotion)

        # Run synthesis in executor (MLX operations can block)
        loop = asyncio.get_event_loop()
        audio_array, sample_rate = await loop.run_in_executor(
            None,
            self._generate_audio,
            text,
            instruct,
        )

        # Convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio_array, sample_rate, format="WAV")
        audio_data = buf.getvalue()

        generation_time = (time.time() - start_time) * 1000
        duration = len(audio_array) / sample_rate if len(audio_array) > 0 else 0.0

        logger.debug(
            f"Qwen3-TTS: {len(text)} chars → {duration:.1f}s audio "
            f"in {generation_time:.0f}ms (emotion: {emotion})"
        )

        return TTSResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_seconds=duration,
            generation_time_ms=generation_time,
        )

    def _generate_audio(self, text: str, instruct: str) -> tuple[np.ndarray, int]:
        """Synchronous audio generation via Qwen3-TTS CustomVoice."""
        import mlx.core as mx

        if not self._model:
            raise RuntimeError("Qwen3-TTS not initialized")

        # Use generate_custom_voice() for consistent named speaker + emotion
        logger.debug(
            f"Qwen3-TTS generate: speaker={self._voice}, lang={self._lang_code}, "
            f"instruct='{instruct}'"
        )
        results = self._model.generate_custom_voice(
            text=text,
            speaker=self._voice,
            language=self._lang_code,
            instruct=instruct,
            verbose=False,
            stream=False,
        )

        # Collect all audio chunks
        audio_chunks = []
        sample_rate = getattr(self._model, "sample_rate", 24000)

        for result in results:
            audio = result.audio
            if hasattr(audio, "tolist"):
                # Convert mlx array to numpy
                audio = np.array(audio, dtype=np.float32)
            elif isinstance(audio, mx.array):
                audio = np.array(audio.tolist(), dtype=np.float32)
            audio_chunks.append(audio)
            if hasattr(result, "sample_rate"):
                sample_rate = result.sample_rate

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
        else:
            full_audio = np.array([], dtype=np.float32)

        return full_audio.flatten(), sample_rate

    def _get_emotion_instruction(self, emotion: str | None) -> str:
        """Map LLM emotion tag to TTS instruction."""
        if not emotion:
            return "Speak naturally and conversationally"

        emotion_lower = emotion.strip().lower()
        if emotion_lower in _EMOTION_INSTRUCTIONS:
            return _EMOTION_INSTRUCTIONS[emotion_lower]

        # Try matching individual words from compound emotions
        words = [w.strip() for w in emotion_lower.replace(",", " ").split()]
        for word in words:
            if word in _EMOTION_INSTRUCTIONS:
                return _EMOTION_INSTRUCTIONS[word]

        logger.debug(f"Unknown emotion tag for TTS: '{emotion}', using neutral")
        return "Speak naturally and conversationally"

    async def shutdown(self) -> None:
        """Release model."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Qwen3-TTS model unloaded")
