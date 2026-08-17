"""Tests for the provider registry (config-driven creation, one generic creator)."""

import pytest

from src.core.registry import (
    create_memory_provider,
    list_providers,
    register_provider,
)
from src.providers.memory.simple_provider import SimpleMemoryProvider


def test_list_providers_covers_all_categories():
    providers = list_providers()
    assert set(providers) == {"stt", "llm", "tts", "memory"}
    assert "ollama" in providers["llm"]
    assert "qwen" in providers["tts"]
    assert "chroma" in providers["memory"]


async def test_unknown_provider_raises_with_category():
    with pytest.raises(ValueError, match="Unknown memory provider: 'nope'"):
        await create_memory_provider({"memory": {"provider": "nope"}})


async def test_register_then_create_roundtrip():
    register_provider(
        "memory", "test_fake",
        "src.providers.memory.simple_provider.SimpleMemoryProvider",
    )
    provider = await create_memory_provider({"memory": {"provider": "test_fake"}})
    assert isinstance(provider, SimpleMemoryProvider)
    assert "test_fake" in list_providers()["memory"]


def test_register_unknown_category_raises():
    with pytest.raises(ValueError, match="Unknown provider category"):
        register_provider("avatar", "x", "some.path.Cls")
