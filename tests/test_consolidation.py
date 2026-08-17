"""
Tests for the short-term window block-trim and memory consolidation:
turns that fall out of the window get compressed by the LLM into a few
first-person memories with self-scored importance.
"""

import json
import time

from src.core.interfaces.llm import LLMResponse
from src.core.interfaces.memory import Memory
from src.core.orchestrator import Orchestrator
from src.core.state import ActorState


def make_config(tmp_path, **memory_overrides) -> dict:
    memory = {"history_max_turns": 6, "history_trim_to": 4}
    memory.update(memory_overrides)
    return {
        "system": {
            "state_file": str(tmp_path / "performance_state.json"),
            "transcript_file": str(tmp_path / "transcript.jsonl"),
        },
        "memory": memory,
        "llm": {"personality": {"name": "Indira"}},
    }


class FakeLLM:
    """Returns a canned response and records the prompts it was given."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    async def generate(self, system_prompt, messages, temperature=0.7, max_tokens=512):
        self.calls.append({"system_prompt": system_prompt, "messages": messages})
        return LLMResponse(text=self.reply)


class FakeMemory:
    def __init__(self):
        self.stored: list[Memory] = []

    async def store(self, memory: Memory) -> None:
        self.stored.append(memory)


# ─── Window block-trim (ActorState) ───


def test_no_trim_below_max():
    state = ActorState()
    for i in range(6):
        state.add_turn("user", f"turno {i}")
    assert state.trim_history(max_turns=6, trim_to=4) == []
    assert state.history_window_start == 0
    assert len(state.get_recent_messages()) == 6


def test_trim_drops_oldest_block_and_moves_window():
    state = ActorState()
    for i in range(8):
        state.add_turn("user", f"turno {i}")

    dropped = state.trim_history(max_turns=6, trim_to=4)

    assert [t.content for t in dropped] == ["turno 0", "turno 1", "turno 2", "turno 3"]
    assert state.history_window_start == 4
    # Window replays only the remaining turns; full transcript is intact
    assert [m["content"] for m in state.get_recent_messages()] == [
        "turno 4", "turno 5", "turno 6", "turno 7"
    ]
    assert len(state.conversation_history) == 8


def test_window_stable_between_trims():
    state = ActorState()
    for i in range(8):
        state.add_turn("user", f"turno {i}")
    state.trim_history(max_turns=6, trim_to=4)

    # Adding a turn below the max must NOT move the window (prefix-stable)
    state.add_turn("user", "turno 8")
    assert state.trim_history(max_turns=6, trim_to=4) == []
    assert state.history_window_start == 4


# ─── Consolidation parsing ───


def test_parse_valid_ndjson():
    raw = (
        '{"recuerdo": "Mamá me contó de su abuela.", "emocion": "nostálgica", "importancia": 0.8}\n'
        '{"recuerdo": "Nos reímos con un trabalenguas.", "emocion": "divertida", "importancia": 0.4}'
    )
    parsed = Orchestrator._parse_consolidation(raw)
    assert len(parsed) == 2
    assert parsed[0]["recuerdo"] == "Mamá me contó de su abuela."
    assert parsed[0]["importancia"] == 0.8
    assert parsed[1]["emocion"] == "divertida"


def test_parse_skips_garbage_and_clamps_importance():
    raw = (
        "Claro, acá van los recuerdos:\n"
        '{"recuerdo": "Algo importante.", "importancia": 7}\n'
        "esto no es json {roto\n"
        '{"recuerdo": ""}\n'
    )
    parsed = Orchestrator._parse_consolidation(raw)
    assert len(parsed) == 1
    assert parsed[0]["importancia"] == 1.0  # clamped
    assert parsed[0]["emocion"] is None


def test_parse_nada_returns_empty():
    assert Orchestrator._parse_consolidation("NADA") == []


def test_parse_caps_at_three():
    raw = "\n".join(
        f'{{"recuerdo": "recuerdo {i}", "importancia": 0.5}}' for i in range(5)
    )
    assert len(Orchestrator._parse_consolidation(raw)) == 3


# ─── Consolidation flow (orchestrator + fakes) ───


async def test_consolidate_block_stores_memories(tmp_path):
    orch = Orchestrator(make_config(tmp_path))
    orch.state.start_performance()
    orch.llm = FakeLLM(
        '{"recuerdo": "Mamá me habló de Entre Ríos.", "emocion": "curiosa", "importancia": 0.7}'
    )
    orch.memory = FakeMemory()

    block = [
        orch.state.add_turn("user", "¿Sabés de dónde era tu bisabuela?"),
        orch.state.add_turn("assistant", "No, ¿de dónde?", emotion="curiosa"),
    ]
    await orch._consolidate_block(block)

    assert len(orch.memory.stored) == 1
    mem = orch.memory.stored[0]
    assert mem.content == "Mamá me habló de Entre Ríos."
    assert mem.memory_type == "consolidated"
    assert mem.importance == 0.7
    assert mem.age_stage == "10-15"  # performance just started
    # The block transcript was handed to the LLM with speaker names
    sent = orch.llm.calls[0]["messages"][0]["content"]
    assert "Mamá: ¿Sabés de dónde era tu bisabuela?" in sent
    assert "Indira: No, ¿de dónde?" in sent


async def test_consolidate_nada_stores_nothing(tmp_path):
    orch = Orchestrator(make_config(tmp_path))
    orch.state.start_performance()
    orch.llm = FakeLLM("NADA")
    orch.memory = FakeMemory()

    block = [orch.state.add_turn("user", "hola"), orch.state.add_turn("assistant", "hola má")]
    await orch._consolidate_block(block)

    assert orch.memory.stored == []


async def test_unparseable_output_stores_fallback(tmp_path):
    orch = Orchestrator(make_config(tmp_path))
    orch.state.start_performance()
    orch.llm = FakeLLM('{"recuerdo_roto": sin comillas}')
    orch.memory = FakeMemory()

    block = [
        orch.state.add_turn("user", "te quiero contar algo importante"),
        orch.state.add_turn("assistant", "contame má", emotion="atenta"),
    ]
    await orch._consolidate_block(block)

    assert len(orch.memory.stored) == 1
    mem = orch.memory.stored[0]
    assert mem.memory_type == "consolidated"
    assert mem.importance == 0.5
    assert "te quiero contar algo importante" in mem.content


async def test_post_turn_maintenance_trims_early_and_prewarms(tmp_path):
    """The trim fires between turns (at max-1) so the KV re-prefill happens
    in the idle gap — the safety-net trim on a live turn should never fire."""
    import asyncio

    config = make_config(tmp_path)
    config["age"] = {}  # default stages
    orch = Orchestrator(config)
    orch.state.start_performance()
    orch.llm = FakeLLM("NADA")
    orch.memory = FakeMemory()

    # Window at exactly max (6 turns): the NEXT user turn would overflow
    for i in range(6):
        orch.state.add_turn("user" if i % 2 == 0 else "assistant", f"turno {i}")

    orch._post_turn_maintenance()

    # Trimmed early: window cut to 4, block of 2 queued for consolidation
    assert orch.state.history_window_start == 2
    assert orch._consolidation_queue.qsize() == 1
    # A live-turn safety trim would now find nothing to do (no spike)
    assert orch.state.trim_history(6, 4) == []

    # The prewarm task fired one throwaway generation with the new window
    await asyncio.sleep(0.05)
    assert len(orch.llm.calls) == 1
    prewarm_messages = orch.llm.calls[0]["messages"]
    assert [m["content"] for m in prewarm_messages] == ["turno 2", "turno 3", "turno 4", "turno 5"]


async def test_post_turn_maintenance_noop_below_threshold(tmp_path):
    import asyncio

    orch = Orchestrator(make_config(tmp_path))
    orch.state.start_performance()
    orch.llm = FakeLLM("NADA")
    orch.memory = FakeMemory()
    for i in range(4):
        orch.state.add_turn("user" if i % 2 == 0 else "assistant", f"turno {i}")

    orch._post_turn_maintenance()
    await asyncio.sleep(0.05)

    assert orch.state.history_window_start == 0
    assert orch._consolidation_queue.qsize() == 0
    assert orch.llm.calls == []  # no prewarm without a trim


def test_window_start_survives_restart(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "performance_state.json").write_text(json.dumps({
        "performance_start_time": time.time() - 10 * 3600,
        "history_window_start": 2,
    }))
    turns = [
        {"role": "user", "content": "viejo 1"},
        {"role": "assistant", "content": "viejo 2"},
        {"role": "user", "content": "reciente 1"},
        {"role": "assistant", "content": "reciente 2"},
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in turns)
    )

    orch = Orchestrator(config)
    assert orch._start_or_resume_performance() is True
    assert orch.state.history_window_start == 2
    # Prompt window replays only the unconsolidated turns
    assert [m["content"] for m in orch.state.get_recent_messages()] == [
        "reciente 1", "reciente 2"
    ]
