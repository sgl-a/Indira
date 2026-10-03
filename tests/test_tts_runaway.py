"""
Tests for the Qwen3-TTS runaway guard.

Sampling occasionally misses the end-of-speech token and the model babbles
to its frame cap — seen live as 5.5 minutes of noise for a one-line reply.
The provider caps generation by text length, resamples once when a run hits
the cap, and raises if every attempt does (callers skip the line: silence
beats noise mid-performance). A fake model stands in for the real one.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.providers.tts.qwen_tts_provider import QwenTTSProvider, TTSRunawayError


class _FakeModel:
    """Emits one result per call; `runaways` says which calls never stop."""

    sample_rate = 24000

    def __init__(self, runaways: list[bool]):
        self._runaways = list(runaways)
        self.max_tokens_seen: list[int] = []

    def generate_custom_voice(self, *, max_tokens, **_):
        self.max_tokens_seen.append(max_tokens)
        frames = max_tokens if self._runaways.pop(0) else 10
        yield SimpleNamespace(
            audio=np.zeros(frames * 1920, dtype=np.float32),
            token_count=frames,
            sample_rate=24000,
        )


def _provider(model: _FakeModel) -> QwenTTSProvider:
    provider = QwenTTSProvider()
    provider._model = model
    return provider


async def test_clean_run_is_capped_by_text_length():
    model = _FakeModel([False])
    result = await _provider(model).synthesize("No entiendo por qué decís eso.")
    assert result.duration_seconds == pytest.approx(10 / 12.5)
    # Well under mlx-audio's 4096 default, which is what let the noise run 5.5 min
    assert model.max_tokens_seen == [90]


async def test_runaway_is_discarded_and_resampled():
    model = _FakeModel([True, False])
    result = await _provider(model).synthesize("No entiendo por qué decís eso.")
    assert len(model.max_tokens_seen) == 2
    assert result.duration_seconds == pytest.approx(10 / 12.5)


async def test_repeated_runaway_raises_instead_of_playing_noise():
    model = _FakeModel([True, True])
    with pytest.raises(TTSRunawayError):
        await _provider(model).synthesize("No entiendo por qué decís eso.")


async def test_very_short_line_still_has_room():
    model = _FakeModel([False])
    await _provider(model).synthesize("Sí.")
    assert model.max_tokens_seen == [50]
