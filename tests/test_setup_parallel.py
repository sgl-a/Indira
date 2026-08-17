"""
Tests for concurrent provider initialization (Orchestrator.setup):
startup must overlap the four provider loads, tolerate STT failure
(text-only mode), and abort loudly — but with partial state assigned
for cleanup — when a required provider fails.
"""

import asyncio

import pytest

import src.core.orchestrator as orchestrator_module
from src.core.orchestrator import Orchestrator


def make_orchestrator(tmp_path) -> Orchestrator:
    return Orchestrator({
        "system": {
            "state_file": str(tmp_path / "performance_state.json"),
            "transcript_file": str(tmp_path / "transcript.jsonl"),
        },
        "memory": {"consolidation": {"enabled": False}},
    })


class FakeProvider:
    async def shutdown(self):
        pass


def make_creator(name, delay, active, overlaps, fail=False):
    """A fake create_*_provider that records how many creators ran concurrently."""
    async def create(config):
        active.append(name)
        overlaps.append(len(active))
        await asyncio.sleep(delay)
        active.remove(name)
        if fail:
            raise RuntimeError(f"{name} exploded")
        return FakeProvider()
    return create


async def test_providers_initialize_concurrently(tmp_path, monkeypatch):
    active, overlaps = [], []
    for kind in ("llm", "tts", "memory", "stt"):
        monkeypatch.setattr(
            orchestrator_module, f"create_{kind}_provider",
            make_creator(kind, 0.05, active, overlaps),
        )

    orch = make_orchestrator(tmp_path)
    await orch.setup()

    assert orch.llm and orch.tts and orch.memory and orch.stt
    # All four creators were in flight at the same time at some point
    assert max(overlaps) == 4


async def test_stt_failure_is_tolerated(tmp_path, monkeypatch):
    active, overlaps = [], []
    for kind, fail in (("llm", False), ("tts", False), ("memory", False), ("stt", True)):
        monkeypatch.setattr(
            orchestrator_module, f"create_{kind}_provider",
            make_creator(kind, 0.01, active, overlaps, fail=fail),
        )

    orch = make_orchestrator(tmp_path)
    await orch.setup()  # must not raise

    assert orch.stt is None
    assert orch.llm and orch.tts and orch.memory


async def test_required_provider_failure_aborts_with_partial_state(tmp_path, monkeypatch):
    active, overlaps = [], []
    for kind, fail in (("llm", False), ("tts", True), ("memory", False), ("stt", False)):
        monkeypatch.setattr(
            orchestrator_module, f"create_{kind}_provider",
            make_creator(kind, 0.01, active, overlaps, fail=fail),
        )

    orch = make_orchestrator(tmp_path)
    with pytest.raises(RuntimeError, match="TTS provider failed"):
        await orch.setup()

    # Successful providers were assigned before the raise → shutdown() can clean them
    assert orch.llm is not None
    assert orch.memory is not None
    assert orch.tts is None
