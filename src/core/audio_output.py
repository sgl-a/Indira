from __future__ import annotations

"""
Audio output: play in-memory WAV bytes through the system speaker.

This is the ONLY module that talks to macOS `afplay` — a future port to
another platform (or honoring the audio.output_device config knob via
sounddevice) touches this file alone.
"""

import asyncio
import tempfile
from pathlib import Path


class WavPlayback:
    """
    One playback of in-memory WAV bytes.

    Lifecycle: `await WavPlayback.start(data)` writes a temp file and spawns
    the player WITHOUT blocking (so synthesis of the next chunk can overlap
    playback); `await .wait()` blocks until the audio ends and removes the
    temp file. A caller abandoning a playback early (task cancellation)
    should call `cleanup()`.
    """

    def __init__(self, process: asyncio.subprocess.Process, temp_path: Path):
        self._process = process
        self._temp_path = temp_path

    @classmethod
    async def start(cls, audio_data: bytes) -> WavPlayback:
        """Write `audio_data` to a temp file and start playing it."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = Path(f.name)
        try:
            process = await asyncio.create_subprocess_exec(
                "afplay", str(temp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return cls(process, temp_path)

    async def wait(self) -> None:
        """Block until playback finishes, then remove the temp file."""
        try:
            await self._process.wait()
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Remove the temp file (idempotent; safe after wait())."""
        self._temp_path.unlink(missing_ok=True)
