"""
Age profile schema + prompt assembly.

The eight profiles carry the whole arc of a plot-less, improvised piece, and
nothing else in the codebase reads them — so a typo in a key name silently
produces a character with no personality. These tests guard that.
"""

from __future__ import annotations

import time

import pytest
import yaml

from src.core.age_engine import AgeEngine
from src.core.config import get_config
from src.core.state import ActorState

PROFILE_DIRS = [
    "age_10_15", "age_15_20", "age_20_25", "age_25_30",
    "age_30_40", "age_40_50", "age_50_60", "age_60_70",
]

REQUIRED = [
    "motor_dramatico", "cree_de_ximena", "estado_base",
    "ritmo", "vocabulario", "mirada", "rasgos", "muestras",
]


def _profile(name: str) -> dict:
    with open(f"profiles/{name}/personality.yaml") as f:
        return yaml.safe_load(f)


def _state_at(hours: float) -> ActorState:
    state = ActorState()
    state.performance_start_time = time.time() - hours * 3600
    return state


@pytest.mark.parametrize("name", PROFILE_DIRS)
def test_profile_has_required_fields(name):
    profile = _profile(name)
    missing = [key for key in REQUIRED if key not in profile]
    assert not missing, f"{name} is missing {missing}"


@pytest.mark.parametrize("name", PROFILE_DIRS)
def test_motor_dramatico_is_complete(name):
    """The objective is the arc — a stage without one has nothing driving it."""
    motor = _profile(name)["motor_dramatico"]
    for key in ("quiere", "obstaculo", "tacticas"):
        assert motor.get(key), f"{name}: motor_dramatico.{key} is empty"
    assert len(motor["tacticas"]) >= 3, f"{name}: needs at least 3 tactics"


@pytest.mark.parametrize("name", PROFILE_DIRS)
def test_mirada_is_complete(name):
    mirada = _profile(name)["mirada"]
    for key in ("nota", "lee", "rompe_el_silencio"):
        assert mirada.get(key), f"{name}: mirada.{key} is empty"


@pytest.mark.parametrize("name", PROFILE_DIRS)
def test_muestras_are_well_formed(name):
    """Each exemplar needs a situation, an opening line, and Indira's reply."""
    muestras = _profile(name)["muestras"]
    assert len(muestras) >= 4, f"{name}: exemplars anchor voice; want 4+"
    for i, sample in enumerate(muestras):
        assert sample.get("situacion"), f"{name}[{i}]: no situacion"
        assert sample.get("ximena") or sample.get("otro"), f"{name}[{i}]: no opening line"
        assert sample.get("indira"), f"{name}[{i}]: no reply"
        assert sample["indira"].lstrip().startswith("["), (
            f"{name}[{i}]: reply must carry an [emoción] tag — the exemplars are "
            "what keep the tag protocol reinforced"
        )


def test_objectives_are_all_different():
    """Description saturates across a lifetime; desire is the axis that doesn't."""
    wants = [_profile(n)["motor_dramatico"]["quiere"] for n in PROFILE_DIRS]
    assert len(set(wants)) == len(wants), "two stages share an objective"


def test_prompt_is_byte_stable_within_a_stage():
    """Ollama reuses its KV cache only for a byte-identical prefix."""
    config = get_config()
    engine = AgeEngine(config)
    for stage in config["age"]["stages"]:
        early = engine.build_personality_prompt(
            _state_at(stage["start_hour"] + 0.1), config
        )
        late = engine.build_personality_prompt(
            _state_at(stage["end_hour"] - 0.1), config
        )
        assert early == late, f"stage {stage['range']} prompt drifts within the stage"


def test_prompt_differs_between_stages():
    config = get_config()
    engine = AgeEngine(config)
    prompts = {
        stage["range"]: engine.build_personality_prompt(
            _state_at((stage["start_hour"] + stage["end_hour"]) / 2), config
        )
        for stage in config["age"]["stages"]
    }
    assert len(set(prompts.values())) == len(prompts), "two stages render identically"


def test_prompt_carries_the_profile_content():
    config = get_config()
    engine = AgeEngine(config)
    prompt = engine.build_personality_prompt(_state_at(12.0), config)  # 15-20
    profile = _profile("age_15_20")

    assert profile["motor_dramatico"]["quiere"] in prompt
    assert profile["cree_de_ximena"] in prompt
    assert profile["rasgos"][0] in prompt
    assert profile["ya_no"][0] in prompt
    assert profile["muestras"][0]["indira"] in prompt
    # Exemplars must be labelled, or the model echoes them and they hit the speaker
    assert "no son líneas para repetir" in prompt.lower()


def test_prompt_renders_non_ximena_interlocutors():
    """Visitors may speak to her; those exemplars must not be labelled 'Mamá'."""
    config = get_config()
    engine = AgeEngine(config)
    prompt = engine.build_personality_prompt(_state_at(12.0), config)
    assert "Otra persona:" in prompt
