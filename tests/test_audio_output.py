"""
Tests for WavPlayback — the single audio-playback lifecycle
(write temp file → spawn player → wait → delete). No real audio
is played; the subprocess is faked.
"""

import asyncio
from pathlib import Path

import pytest

from src.core.audio_output import WavPlayback


class FakeProcess:
    def __init__(self):
        self.waited = False

    async def wait(self):
        self.waited = True
        return 0


@pytest.fixture
def fake_player(monkeypatch):
    """Replace afplay spawning with a fake; records the command used."""
    captured = {}

    async def fake_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        captured["process"] = FakeProcess()
        return captured["process"]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return captured


async def test_start_writes_file_and_wait_removes_it(fake_player):
    playback = await WavPlayback.start(b"RIFF-fake-wav-bytes")

    assert fake_player["cmd"][0] == "afplay"
    temp_path = Path(fake_player["cmd"][1])
    # While "playing": file exists with the exact audio bytes
    assert temp_path.read_bytes() == b"RIFF-fake-wav-bytes"

    await playback.wait()

    assert fake_player["process"].waited
    assert not temp_path.exists()


async def test_cleanup_is_idempotent_and_covers_abandonment(fake_player):
    playback = await WavPlayback.start(b"bytes")
    temp_path = Path(fake_player["cmd"][1])

    playback.cleanup()  # abandoned without wait() (e.g. task cancelled)
    assert not temp_path.exists()
    playback.cleanup()  # safe to call again


async def test_spawn_failure_removes_temp_file(monkeypatch):
    import src.core.audio_output as audio_output

    created: list[Path] = []
    real_ntf = audio_output.tempfile.NamedTemporaryFile

    def spying_ntf(*args, **kwargs):
        f = real_ntf(*args, **kwargs)
        created.append(Path(f.name))
        return f

    async def failing_exec(*cmd, **kwargs):
        raise FileNotFoundError("afplay not found")

    monkeypatch.setattr(audio_output.tempfile, "NamedTemporaryFile", spying_ntf)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_exec)

    with pytest.raises(FileNotFoundError):
        await WavPlayback.start(b"bytes")

    # The temp file written before the failed spawn must not leak
    assert created and not created[0].exists()
