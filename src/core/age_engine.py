from __future__ import annotations

"""
Age Engine.

Manages the Indira's aging progression over 72 hours.
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


def _flat(text: str) -> str:
    """Collapse a YAML block scalar into one line for the prompt."""
    return " ".join(text.split())


@dataclass
class AgeStage:
    """Configuration for a specific age stage."""

    range: str  # e.g., "10-15"
    start_hour: float
    end_hour: float
    # Loaded from profile YAML (keys are Spanish, matching the performance language)
    motor_dramatico: dict | None = None  # quiere / obstaculo / tacticas
    cree_de_ximena: str | None = None
    estado_base: str | None = None
    ritmo: str | None = None
    vocabulario: str | None = None
    mirada: dict | None = None  # nota / lee / rompe_el_silencio
    ya_no: list[str] | None = None
    circunstancias_dadas: list[str] | None = None
    muestras: list[dict] | None = None  # situacion / ximena|otro / indira
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

                stage.motor_dramatico = profile.get("motor_dramatico")
                stage.cree_de_ximena = profile.get("cree_de_ximena")
                stage.estado_base = profile.get("estado_base")
                stage.ritmo = profile.get("ritmo")
                stage.vocabulario = profile.get("vocabulario")
                stage.mirada = profile.get("mirada")
                stage.ya_no = profile.get("ya_no", [])
                stage.circunstancias_dadas = profile.get("circunstancias_dadas", [])
                stage.muestras = profile.get("muestras", [])

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

        Combines permanent identity with the age-specific profile.
        All in Spanish to match the performance language.

        Section order is deliberate: objective (what she wants) first so it
        colours everything read after it, then who she is, then how she
        perceives, and dialogue exemplars last — closest to generation, where
        few-shot anchoring does the most work against persona drift.

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

        # ── Objective first: what she is after right now drives everything else.
        # (Hagen 7-9. Also the only place a plot-less, improvised piece can
        #  carry its arc, and what lets an aligned model play friction as
        #  pursuit of a legitimate goal rather than as a "be difficult" trait.)
        motor = stage.motor_dramatico or {}
        if motor.get("quiere"):
            prompt_parts.extend(["", "## Lo que querés ahora"])
            prompt_parts.append(f"Querés: {motor['quiere']}")
            if motor.get("obstaculo"):
                prompt_parts.append(f"Lo que se interpone: {motor['obstaculo']}")
            if motor.get("tacticas"):
                prompt_parts.append("Lo que hacés para conseguirlo:")
                for tactic in motor["tacticas"]:
                    prompt_parts.append(f"- {tactic}")

        if stage.cree_de_ximena:
            prompt_parts.extend([
                "",
                f"**Lo que creés de tu mamá:** {stage.cree_de_ximena}",
                "(Podés estar equivocada.)",
            ])

        # No abstract trait list here by design: a line like "encontrás alegría
        # en cosas mínimas" never tells her what to do when something happens,
        # and it is the field that flattens into the model's default warm-
        # wisdom register. Character is carried by the objective, the tactics,
        # ritmo/vocabulario and the exemplars instead.
        if stage.ritmo:
            prompt_parts.extend(["", f"**Tu ritmo al hablar:** {_flat(stage.ritmo)}"])

        if stage.vocabulario:
            prompt_parts.extend(["", f"**Tu vocabulario:** {_flat(stage.vocabulario)}"])

        if stage.estado_base:
            prompt_parts.extend(["", f"**Tu ánimo de base:** {_flat(stage.estado_base)}"])

        # Perception: the vision layer supplies raw facts per turn; this says
        # how someone of this age reads them.
        mirada = stage.mirada or {}
        if mirada:
            prompt_parts.extend(["", "## Qué mirás"])
            if mirada.get("nota"):
                prompt_parts.append(f"Registrás: {_flat(mirada['nota'])}")
            if mirada.get("lee"):
                prompt_parts.append(f"Cómo lo interpretás: {_flat(mirada['lee'])}")
            if mirada.get("rompe_el_silencio"):
                prompt_parts.append(
                    f"Qué te hace hablar primero: {_flat(mirada['rompe_el_silencio'])}"
                )

        if stage.ya_no:
            prompt_parts.extend(["", "## Cosas que ya no hacés"])
            for gone in stage.ya_no:
                prompt_parts.append(f"- {gone}")

        # Given circumstances (Hagen 5): 72 real hours hold 60 fictional years,
        # so most of her life happens in the gaps between stages. This names
        # what was already true when the stage opened. Empty until written.
        if stage.circunstancias_dadas:
            prompt_parts.extend(["", "## Lo que ya pasó"])
            for fact in stage.circunstancias_dadas:
                prompt_parts.append(f"- {fact}")

        # Exemplars last before the format contract: few-shot dialogue anchors
        # voice far harder than trait lists, and sitting near the end keeps it
        # close to generation. Marked hard as examples — anything echoed
        # verbatim goes straight out the speaker.
        if stage.muestras:
            prompt_parts.extend([
                "",
                "## Ejemplos de cómo sonás",
                "Muestran tu registro a esta edad. No son líneas para repetir:",
                "nunca las digas textual.",
            ])
            for sample in stage.muestras:
                if not sample.get("indira"):
                    continue
                prompt_parts.append("")
                if sample.get("situacion"):
                    prompt_parts.append(f"({sample['situacion']})")
                if sample.get("ximena"):
                    prompt_parts.append(f"Mamá: {sample['ximena']}")
                elif sample.get("otro"):
                    prompt_parts.append(f"Otra persona: {sample['otro']}")
                prompt_parts.append(f"Vos: {sample['indira']}")

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
