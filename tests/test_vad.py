"""
Tests for EndpointDetector — when voice mode decides a turn started and
ended. Two thresholds: a block louder than `start_threshold` starts a turn;
once speaking, only blocks quieter than `keep_threshold` count as silence.
With a single threshold, soft speech counted as silence and the speaker
was cut off mid-sentence.
"""

import numpy as np
import pytest

from src.core.vad import EndpointDetector

RATE = 16000
BLOCK = 1600  # 100 ms


def block(rms: float) -> np.ndarray:
    """A 100 ms mic block (shaped like sounddevice's) with exactly this RMS."""
    return np.full((BLOCK, 1), rms, dtype=np.float32)


def detector(**overrides) -> EndpointDetector:
    params = dict(
        sample_rate=RATE, block_ms=100, start_threshold=0.01, keep_threshold=0.005,
        silence_ms=800, preroll_ms=300, max_turn_ms=60000,
    )
    params.update(overrides)
    return EndpointDetector(**params)


def feed(d: EndpointDetector, levels: list[float]) -> list:
    """Feed one block per level; return [(block_index, Utterance), ...]."""
    return [(i, u) for i, lvl in enumerate(levels) if (u := d.feed(block(lvl))) is not None]


LOUD, SOFT, QUIET = 0.02, 0.008, 0.001  # SOFT: real speech between keep and start


def test_soft_speech_keeps_the_turn_open():
    levels = [LOUD] * 3 + [SOFT] * 10 + [LOUD] * 2 + [QUIET] * 8
    ends = feed(detector(), levels)
    assert [i for i, _ in ends] == [len(levels) - 1]  # one turn, ended by real silence


def test_single_threshold_cuts_soft_speech():
    # The old behavior, kept reachable by omitting keep_threshold in config
    levels = [LOUD] * 3 + [SOFT] * 10 + [LOUD] * 2 + [QUIET] * 8
    ends = feed(detector(keep_threshold=0.01), levels)
    assert ends[0][0] == 3 + 8 - 1  # cut 800 ms into the soft stretch


def test_soft_sound_alone_does_not_start_a_turn():
    d = detector()
    assert feed(d, [SOFT] * 20) == []
    assert not d.is_speaking


def test_turn_ends_exactly_at_silence_ms():
    ends = feed(detector(), [LOUD] * 3 + [QUIET] * 10)
    assert [i for i, _ in ends] == [3 + 8 - 1]


def test_speech_excludes_preroll_and_trailing_silence():
    [(_, u)] = feed(detector(), [QUIET] * 5 + [LOUD] * 4 + [QUIET] * 8)
    assert len(u.audio) == (3 + 4 + 8) * BLOCK  # 300 ms pre-roll + speech + silence
    assert u.speech_seconds == pytest.approx(0.4)
    assert u.speech_rms == pytest.approx(LOUD)


def test_short_soft_answer_is_judged_on_its_speech():
    # A quiet "sí": averaged together with the trailing silence (the old
    # gate) it fell under 0.7 × start and was silently dropped
    [(_, u)] = feed(detector(silence_ms=1200), [0.006, 0.015, 0.012] + [0.004] + [QUIET] * 11)
    assert u.speech_seconds == pytest.approx(0.2)
    assert u.speech_rms > 0.01 * 0.7
    assert np.sqrt(np.mean(u.audio[3 * BLOCK:] ** 2)) < 0.01 * 0.7


def test_max_turn_ends_a_turn_without_silence():
    # A room noisier than keep_threshold would otherwise never end the turn
    ends = feed(detector(max_turn_ms=1000), [LOUD] + [SOFT] * 20)
    assert [i for i, _ in ends][0] == 9


def test_back_to_back_turns_each_get_their_own_preroll():
    ends = feed(detector(), [LOUD] * 2 + [QUIET] * 8 + [QUIET] * 3 + [LOUD] * 2 + [QUIET] * 8)
    assert len(ends) == 2
    second = ends[1][1]
    assert len(second.audio) == (3 + 2 + 8) * BLOCK


def test_reset_drops_preroll():
    d = detector()
    feed(d, [QUIET] * 3)
    d.reset()
    [(_, u)] = feed(d, [LOUD] + [QUIET] * 8)
    assert len(u.audio) == 9 * BLOCK


def test_keep_threshold_above_start_is_rejected():
    with pytest.raises(ValueError):
        detector(keep_threshold=0.02)
