from __future__ import annotations

"""
Voice activity detection: when does the speaker start, and when are they done?

Energy-based, on the fixed-size blocks the mic delivers, with two thresholds
(hysteresis): a turn STARTS when a block is louder than `start_threshold`,
and ENDS only after `silence_ms` of blocks quieter than the lower
`keep_threshold`. With one threshold for both (the original design), soft
syllables and a dropping voice counted as silence and the turn ended while
the speaker was still talking — measured 2026-10-15 on the MacBook's
built-in mic: 72% of a natural turn sat below the old 0.03 threshold.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Utterance:
    """One finished turn."""

    # Pre-roll + speech + trailing silence: what STT gets
    audio: np.ndarray
    # First loud block → last voiced block: what the noise gates judge
    # (pre-roll and trailing silence would pad a blip's duration and
    # drag the average energy down)
    speech: np.ndarray
    sample_rate: int

    @property
    def speech_seconds(self) -> float:
        return len(self.speech) / self.sample_rate

    @property
    def speech_rms(self) -> float:
        return float(np.sqrt(np.mean(self.speech ** 2))) if len(self.speech) else 0.0


class EndpointDetector:
    """
    Feed it mic blocks; it returns an Utterance when a turn ends.

    Keeps a short pre-roll of the blocks before speech starts (onsets fade
    in below the threshold; without it the first syllable is clipped), and
    ends a turn after `max_turn_ms` even without silence — a safety valve
    for a room whose noise sits above `keep_threshold`, where the turn
    would otherwise never end.
    """

    def __init__(
        self,
        sample_rate: int,
        block_ms: int,
        start_threshold: float,
        keep_threshold: float,
        silence_ms: int,
        preroll_ms: int = 300,
        max_turn_ms: int = 60000,
    ) -> None:
        if keep_threshold > start_threshold:
            raise ValueError(
                f"VAD keep_threshold ({keep_threshold}) must not exceed "
                f"energy_threshold ({start_threshold})"
            )
        self._rate = sample_rate
        self._start = start_threshold
        self._keep = keep_threshold
        self._silence_limit = int(silence_ms / 1000 * sample_rate)
        self._max_samples = int(max_turn_ms / 1000 * sample_rate) if max_turn_ms else 0
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, round(preroll_ms / block_ms)))
        self.reset()

    def reset(self) -> None:
        """Drop everything, pre-roll included (e.g. after the mic was paused)."""
        self._preroll.clear()
        self._start_turn_state()

    def _start_turn_state(self) -> None:
        self.is_speaking = False
        self._blocks: list[np.ndarray] = []
        self._preroll_len = 0   # samples of pre-roll at the head of _blocks
        self._length = 0        # samples captured so far, pre-roll included
        self._voiced_end = 0    # end of the last block above keep_threshold
        self._silence = 0       # samples of silence since then

    def feed(self, block: np.ndarray) -> Utterance | None:
        """Feed one mic block; returns the Utterance when the turn ends."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        energy = float(np.sqrt(np.mean(block ** 2)))

        if not self.is_speaking:
            if energy <= self._start:
                self._preroll.append(block)
                return None
            self.is_speaking = True
            self._blocks = list(self._preroll)
            self._preroll_len = self._length = sum(len(b) for b in self._preroll)
            self._preroll.clear()

        self._blocks.append(block)
        self._length += len(block)
        if energy > self._keep:
            self._silence = 0
            self._voiced_end = self._length
        else:
            self._silence += len(block)

        if self._silence >= self._silence_limit or (
            self._max_samples and self._length - self._preroll_len >= self._max_samples
        ):
            return self._finish()
        return None

    def _finish(self) -> Utterance:
        audio = np.concatenate(self._blocks)
        utterance = Utterance(
            audio=audio,
            speech=audio[self._preroll_len:self._voiced_end],
            sample_rate=self._rate,
        )
        self._start_turn_state()
        return utterance
