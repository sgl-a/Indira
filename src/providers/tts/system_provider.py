from __future__ import annotations

"""
System TTS Provider (macOS).

Uses the macOS 'say' command as a lightweight fallback TTS.
Great for initial development and testing without heavy dependencies.

To replace with a better TTS, change config to use chatterbox/f5tts/bark provider.
"""

import asyncio
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from src.core.interfaces.tts import TTSProvider, TTSResult, VoiceProfile

logger = logging.getLogger(__name__)

# macOS voice mappings for approximate age simulation
_MACOS_VOICES = {
    "en": {
        "10-15": "Samantha",   # Younger sounding
        "15-20": "Samantha",
        "20-25": "Karen",
        "25-30": "Karen",
        "30-40": "Moira",
        "40-50": "Moira",
        "50-60": "Fiona",
        "60-70": "Fiona",
    },
    "es": {
        "10-15": "Paulina",
        "15-20": "Paulina",
        "20-25": "Paulina",
        "25-30": "Paulina",
        "30-40": "Paulina",
        "40-50": "Paulina",
        "50-60": "Paulina",
        "60-70": "Paulina",
    },
}


class SystemTTSProvider(TTSProvider):
    """
    Lightweight TTS using macOS 'say' command.

    This is a FALLBACK provider for quick development.
    For production, use Chatterbox, F5-TTS, or Bark.

    Limitations:
        - No voice cloning
        - No emotion control
        - Limited voice variety
        - macOS only
    """

    def __init__(self):
        self.default_voice: str = "Samantha"
        self.default_rate: int = 175  # words per minute

    async def initialize(self, config: dict) -> None:
        # Verify 'say' command exists (macOS only)
        try:
            result = subprocess.run(
                ["which", "say"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("'say' command not found. This provider is macOS only.")
        except FileNotFoundError:
            raise RuntimeError("'say' command not found. This provider is macOS only.")

        voice_config = config.get("default_voice", {})
        speed = voice_config.get("speed", 1.0)
        self.default_rate = int(175 * speed)

        logger.info("System TTS (macOS 'say') initialized")

    async def synthesize(
        self,
        text: str,
        voice_profile: VoiceProfile | None = None,
        emotion: str | None = None,
    ) -> TTSResult:
        start_time = time.time()

        # Select voice based on age stage
        voice = self.default_voice
        rate = self.default_rate

        if voice_profile:
            age_stage = voice_profile.age_stage
            voice = _MACOS_VOICES.get("en", {}).get(age_stage, self.default_voice)
            rate = int(self.default_rate * voice_profile.speed)

        # Generate audio to file
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            output_path = f.name

        cmd = [
            "say",
            "-v", voice,
            "-r", str(rate),
            "-o", output_path,
            text,
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.wait()

        # Read the generated audio
        audio_data = Path(output_path).read_bytes()
        Path(output_path).unlink(missing_ok=True)

        generation_time = (time.time() - start_time) * 1000

        return TTSResult(
            audio_data=audio_data,
            sample_rate=22050,
            generation_time_ms=generation_time,
        )

    async def shutdown(self) -> None:
        pass
