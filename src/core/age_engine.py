from __future__ import annotations

"""
Age Engine.

Manages the AI Actor's aging progression over 72 hours.
Determines current age stage and loads age-specific personality traits,
voice profiles, and vocabulary constraints.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.state import ActorState

logger = logging.getLogger(__name__)


@dataclass
class AgeStage:
    """Configuration for a specific age stage."""

    range: str  # e.g., "10-15"
    start_hour: float
    end_hour: float
    # Loaded from profile YAML
    personality_traits: list[str] | None = None
    vocabulary_level: str | None = None
    speaking_style: str | None = None
    emotional_tendencies: list[str] | None = None
    relationship_with_parent: str | None = None
    voice_profile_path: str | None = None
    face_profile_path: str | None = None


class AgeEngine:
    """
    Manages age progression and loads age-specific configurations.

    The AI ages through 8 stages over 72 hours, each with distinct
    personality traits, vocabulary, and emotional tendencies.
    """

    def __init__(self, config: dict, profiles_dir: str = "profiles"):
        self.stages: list[AgeStage] = []
        self.profiles_dir = Path(profiles_dir)
        self._load_stages(config)

    def _load_stages(self, config: dict) -> None:
        """Load age stages from config and profile directories."""
        age_config = config.get("age", {})
        stages_config = age_config.get("stages", [])

        for stage_data in stages_config:
            stage = AgeStage(
                range=stage_data["range"],
                start_hour=stage_data["start_hour"],
                end_hour=stage_data["end_hour"],
            )

            # Try to load personality profile
            profile_dir = self.profiles_dir / f"age_{stage.range.replace('-', '_')}"
            personality_path = profile_dir / "personality.yaml"

            if personality_path.exists():
                with open(personality_path) as f:
                    profile = yaml.safe_load(f) or {}

                stage.personality_traits = profile.get("traits", [])
                stage.vocabulary_level = profile.get("vocabulary_level")
                stage.speaking_style = profile.get("speaking_style")
                stage.emotional_tendencies = profile.get("emotional_tendencies", [])
                stage.relationship_with_parent = profile.get("relationship_with_parent")

                # Voice and face files
                voice_path = profile_dir / "voice_reference.wav"
                if voice_path.exists():
                    stage.voice_profile_path = str(voice_path)

                face_path = profile_dir / "face_reference.png"
                if face_path.exists():
                    stage.face_profile_path = str(face_path)

            self.stages.append(stage)

        if not self.stages:
            logger.warning("No age stages configured, using defaults")
            self._create_default_stages()

    def _create_default_stages(self) -> None:
        """Create default age stages if none are configured."""
        defaults = [
            ("10-15", 0, 9),
            ("15-20", 9, 18),
            ("20-25", 18, 27),
            ("25-30", 27, 36),
            ("30-40", 36, 45),
            ("40-50", 45, 54),
            ("50-60", 54, 63),
            ("60-70", 63, 72),
        ]
        for range_str, start, end in defaults:
            self.stages.append(AgeStage(range=range_str, start_hour=start, end_hour=end))

    def get_current_stage(self, state: ActorState) -> AgeStage:
        """Determine the current age stage based on elapsed time."""
        return self.stage_for_hours(state.hours_elapsed)

    def stage_for_hours(self, hours: float) -> AgeStage:
        """Map an elapsed-hours value to its age stage (last stage if past the end)."""
        for stage in self.stages:
            if stage.start_hour <= hours < stage.end_hour:
                return stage
        return self.stages[-1]

    def get_stage_index(self, state: ActorState) -> int:
        """Get the index of the current stage (0-7)."""
        hours = state.hours_elapsed
        for i, stage in enumerate(self.stages):
            if stage.start_hour <= hours < stage.end_hour:
                return i
        return len(self.stages) - 1

    def update_state(self, state: ActorState) -> bool:
        """
        Update the state with current age information.

        Returns True if the age stage changed (transition occurred).
        """
        new_index = self.get_stage_index(state)
        stage_changed = new_index != state.current_age_stage_index

        if stage_changed:
            old_stage = state.current_age_stage
            state.current_age_stage_index = new_index
            state.current_age_stage = self.stages[new_index].range
            logger.info(
                f"🎂 Age transition: {old_stage} → {state.current_age_stage} "
                f"(hour {state.hours_elapsed:.1f})"
            )

        return stage_changed

    def build_personality_prompt(self, state: ActorState, config: dict) -> str:
        """
        Build the system prompt for the LLM based on current age stage.

        Combines permanent identity with age-specific traits.
        All in Spanish to match the performance language.

        Deliberately contains ONLY per-stage-stable content: Ollama reuses
        its KV cache for the byte-identical prompt prefix across requests,
        so anything that changes per turn (emotion, memories) goes in the
        outgoing user message instead (see Orchestrator._wrap_with_context).
        """
        stage = self.get_current_stage(state)
        personality = config.get("llm", {}).get("personality", {})

        name = personality.get("name", "Entity")
        identity = personality.get("identity", [])

        # Build the prompt
        prompt_parts = [
            f"Sos {name}, tenés {stage.range} años.",
        ]

        # Permanent identity rules (from config)
        if identity:
            prompt_parts.append("")
            for rule in identity:
                prompt_parts.append(f"- {rule}")

        # Age-specific traits (from profile YAML)
        if stage.personality_traits:
            prompt_parts.extend([
                "",
                f"## Tu edad actual ({stage.range} años)",
            ])
            for trait in stage.personality_traits:
                prompt_parts.append(f"- {trait}")

        if stage.speaking_style:
            prompt_parts.extend(["", f"**Cómo hablás:** {stage.speaking_style}"])

        if stage.vocabulary_level:
            prompt_parts.extend(["", f"**Vocabulario:** {stage.vocabulary_level}"])

        if stage.emotional_tendencies:
            prompt_parts.extend(["", "**Cómo sentís:**"])
            for tendency in stage.emotional_tendencies:
                prompt_parts.append(f"- {tendency}")

        if stage.relationship_with_parent:
            prompt_parts.extend([
                "",
                "## Tu relación con tu mamá",
                stage.relationship_with_parent,
            ])

        # Emotion tag — kept for TTS pipeline, presented as brief formatting note
        prompt_parts.extend([
            "",
            "## Formato",
            "Respondé solo con lo que dirías en voz alta.",
            "Sin emojis, sin acciones entre asteriscos, sin acotaciones escénicas.",
            "Antes de tu línea, poné una etiqueta de emoción entre corchetes.",
            "Ejemplo: [cálida, nostálgica] Me acuerdo cuando me enseñaste esa palabra.",
            "",
            "Los mensajes pueden empezar con un bloque [Contexto ...]: son",
            "tus recuerdos, información interna tuya, no algo que te dijeron.",
            "No lo menciones ni lo leas en voz alta.",
        ])

        return "\n".join(prompt_parts)
