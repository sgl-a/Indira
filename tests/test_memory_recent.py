"""Tests for memory recency — get_recent must return the NEWEST memories.

Regression: ChromaDB's get(limit=N) returns an arbitrary slice, so the old
implementation returned the oldest memories once the collection grew past
~30 entries — "recent context" was stuck at the start of the show forever.
"""

import time
import uuid

import pytest

from src.core.interfaces.memory import Memory
from src.providers.memory.simple_provider import SimpleMemoryProvider


def make_memory(i: int, base_time: float) -> Memory:
    return Memory(
        id=str(uuid.uuid4()),
        content=f"Recuerdo número {i}",
        age_stage="10-15",
        importance=0.5,
        timestamp=base_time + i,
        memory_type="interaction",
    )


async def test_simple_provider_get_recent_returns_newest():
    provider = SimpleMemoryProvider()
    await provider.initialize({})
    base = time.time()
    for i in range(40):
        await provider.store(make_memory(i, base))

    recent = await provider.get_recent(limit=3)
    assert [m.content for m in recent] == [
        "Recuerdo número 39", "Recuerdo número 38", "Recuerdo número 37",
    ]


async def test_chroma_get_recent_returns_newest_not_oldest(tmp_path):
    chromadb = pytest.importorskip("chromadb")  # noqa: F841

    from src.providers.memory.chroma_provider import ChromaMemoryProvider

    provider = ChromaMemoryProvider()
    await provider.initialize({"memory": {
        "persist_directory": str(tmp_path / "mem"),
        "collection_name": "test_recent",
    }})

    base = time.time()
    # Well past the old limit*3 window where the bug appeared
    for i in range(40):
        await provider.store(make_memory(i, base))

    recent = await provider.get_recent(limit=3)
    assert [m.content for m in recent] == [
        "Recuerdo número 39", "Recuerdo número 38", "Recuerdo número 37",
    ]

    # A restart (new provider over the same persist dir) must re-seed
    # the recency buffer from disk, newest first
    provider2 = ChromaMemoryProvider()
    await provider2.initialize({"memory": {
        "persist_directory": str(tmp_path / "mem"),
        "collection_name": "test_recent",
    }})
    recent2 = await provider2.get_recent(limit=2)
    assert [m.content for m in recent2] == [
        "Recuerdo número 39", "Recuerdo número 38",
    ]
