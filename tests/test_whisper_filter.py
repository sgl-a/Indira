"""
Tests for the Whisper provider's hallucination filter.

On non-speech, Whisper invents lines from the subtitled videos it was
trained on — a clap comes back as "¡Suscríbete!". The provider drops
segments Whisper itself rates as non-speech (no_speech_prob), and its
stock phrases only when they are the whole transcription, so real
dialogue that merely contains one ("me encanta esta música") survives.
A fake mlx_whisper stands in for the real one.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.providers.stt.whisper_provider import WhisperSTTProvider


def seg(text: str, no_speech: float, logprob: float = -0.3) -> dict:
    return {"text": text, "no_speech_prob": no_speech, "avg_logprob": logprob}


async def transcribe(monkeypatch, segments, **stt_config) -> str:
    result = {
        "text": "".join(s["text"] for s in segments),
        "segments": segments,
        "language": "es",
    }
    fake = SimpleNamespace(transcribe=lambda audio, **options: result)
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake)
    provider = WhisperSTTProvider()
    await provider.initialize({"model": "small", "language": "es", **stt_config})
    audio = np.zeros(1600, dtype=np.float32).tobytes()
    return (await provider.transcribe(audio)).text


async def test_clap_is_dropped_by_no_speech_prob(monkeypatch):
    # Measured: a clap → "¡Suscríbete!" at no_speech 0.84, logprob -0.85
    assert await transcribe(monkeypatch, [seg(" ¡Suscríbete!", 0.84, -0.85)]) == ""


async def test_real_short_answer_is_kept(monkeypatch):
    # Measured: a real "¿Sí?" at no_speech 0.18 — and logprob -0.94, LOWER
    # than the hallucinations, which is why logprob isn't used
    assert await transcribe(monkeypatch, [seg(" ¿Sí?", 0.18, -0.94)]) == "¿Sí?"


async def test_only_the_non_speech_segment_is_dropped(monkeypatch):
    text = await transcribe(monkeypatch, [
        seg(" Hola hija, ¿cómo estás?", 0.06),
        seg(" Gracias.", 0.9),
    ])
    assert text == "Hola hija, ¿cómo estás?"


async def test_threshold_comes_from_config(monkeypatch):
    segments = [seg(" No.", 0.65)]
    assert await transcribe(monkeypatch, segments) == "No."
    assert await transcribe(monkeypatch, segments, no_speech_threshold=0.6) == ""


@pytest.mark.parametrize("phrase", [
    "¡Suscríbete!", "Suscribete al canal.", "Subscribe!", "Thank you.",
    "Gracias por ver.", "[Música]", "(Aplausos)",
    "Subtítulos realizados por la comunidad de Amara.org",
])
async def test_stock_phrase_is_dropped_even_when_whisper_is_sure(monkeypatch, phrase):
    assert await transcribe(monkeypatch, [seg(f" {phrase}", 0.2)]) == ""


@pytest.mark.parametrize("line", [
    "Me encanta esta música.",
    "Gracias por verme, má.",
    "Thank you, mom.",
    "No me hagas reír, que me dan risas nerviosas.",
])
async def test_dialogue_containing_a_stock_word_is_kept(monkeypatch, line):
    # The old voice-mode filter matched substrings and dropped all of these
    assert await transcribe(monkeypatch, [seg(f" {line}", 0.1)]) == line
