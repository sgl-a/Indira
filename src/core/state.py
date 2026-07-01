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
    emotion_intensity: float = 0.5  # 0.0 - 1.0

    # ─── Conversation ───
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    last_interaction_time: float = 0.0
    is_speaking: bool = False
    is_listening: bool = False

    # ─── Proactive Behavior ───
    silence_start_time: float | None = None
    last_proactive_time: float = 0.0

    @property
    def hours_elapsed(self) -> float:
        """Hours since performance started."""
        if self.performance_start_time is None:
            return 0.0
        return (time.time() - self.performance_start_time) / 3600

    @property
    def is_running(self) -> bool:
        return self.performance_phase == PerformancePhase.RUNNING

    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """
        Get recent conversation history formatted for LLM input.

        Returns list of {role, content} dicts.
        """
        recent = self.conversation_history[-limit:]
        return [
            {"role": turn.role, "content": turn.content}
            for turn in recent
        ]

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
        self.silence_start_time = None  # Reset silence timer on any interaction
        return turn

    def start_performance(self) -> None:
        """Mark the performance as started."""
        self.performance_start_time = time.time()
        self.performance_phase = PerformancePhase.RUNNING
        self.last_interaction_time = time.time()

    def end_performance(self) -> None:
        """Mark the performance as ended."""
        self.performance_phase = PerformancePhase.ENDED
