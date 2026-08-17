from __future__ import annotations

"""
State management for the AI Actor system.

Tracks the current state of the entire system: age, emotion,
conversation context, and performance timing.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PerformancePhase(Enum):
    """Current phase of the performance."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" (actress), "assistant" (AI), "audience"
    content: str
    emotion: str | None = None
    timestamp: float = field(default_factory=time.time)
    speaker_id: str | None = None


@dataclass
class ActorState:
    """
    Complete state of the AI Actor at any point in time.

    This is the single source of truth for the system's current state.
    """

    # ─── Performance Timing ───
    performance_start_time: float | None = None
    performance_phase: PerformancePhase = PerformancePhase.NOT_STARTED

    # ─── Age & Identity ───
    current_age_stage: str = "10-15"
    current_age_stage_index: int = 0

    # ─── Emotional State ───
    current_emotion: str = "neutral"

    # ─── Conversation ───
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    # Index of the first turn still in the short-term window. Turns before
    # this have been dropped from the prompt (and consolidated into
    # long-term memories); the full transcript stays in the list.
    history_window_start: int = 0
    last_interaction_time: float = 0.0
    is_speaking: bool = False

    @property
    def hours_elapsed(self) -> float:
        """Hours since performance started."""
        if self.performance_start_time is None:
            return 0.0
        return (time.time() - self.performance_start_time) / 3600

    @property
    def is_running(self) -> bool:
        return self.performance_phase == PerformancePhase.RUNNING

    def get_recent_messages(self, limit: int | None = None) -> list[dict]:
        """
        Get the short-term window formatted for LLM input.

        Returns every turn from `history_window_start` onward (the window
        only moves in blocks — see trim_history — so the replayed prefix
        stays byte-stable for the prompt cache between trims).

        Assistant turns get their parsed [emoción] tag re-prefixed, so the
        model reads back exactly what it originally emitted: it sees its own
        emotional arc (continuity without injecting a state instruction) and
        keeps seeing the output format it must produce.

        Returns list of {role, content} dicts.
        """
        window = self.conversation_history[self.history_window_start:]
        if limit is not None:
            window = window[-limit:]
        messages = []
        for turn in window:
            content = turn.content
            if turn.role == "assistant" and turn.emotion:
                content = f"[{turn.emotion}] {content}"
            messages.append({"role": turn.role, "content": content})
        return messages

    def trim_history(self, max_turns: int, trim_to: int) -> list[ConversationTurn]:
        """
        Block-trim the short-term window: once it exceeds `max_turns`,
        advance the window start so `trim_to` turns remain, and return the
        dropped turns (for consolidation into long-term memory).

        Trimming in blocks instead of sliding one turn at a time keeps the
        replayed history prefix-stable for the LLM KV cache between trims.
        Returns [] when no trim happened.
        """
        window_len = len(self.conversation_history) - self.history_window_start
        if window_len <= max_turns:
            return []
        new_start = len(self.conversation_history) - trim_to
        dropped = self.conversation_history[self.history_window_start:new_start]
        self.history_window_start = new_start
        return dropped

    def add_turn(
        self,
        role: str,
        content: str,
        emotion: str | None = None,
        speaker_id: str | None = None,
    ) -> ConversationTurn:
        """Add a conversation turn and update timing."""
        turn = ConversationTurn(
            role=role,
            content=content,
            emotion=emotion,
            speaker_id=speaker_id,
        )
        self.conversation_history.append(turn)
        self.last_interaction_time = time.time()
        return turn

    def start_performance(self) -> None:
        """Mark the performance as started."""
        self.performance_start_time = time.time()
        self.performance_phase = PerformancePhase.RUNNING
        self.last_interaction_time = time.time()

    def end_performance(self) -> None:
        """Mark the performance as ended."""
        self.performance_phase = PerformancePhase.ENDED
