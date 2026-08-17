from __future__ import annotations

"""
Provider Registry.

Discovers and loads providers based on configuration.
This is the core of the model-agnostic design — change config, not code.
"""

import logging
from typing import Any, Type

from src.core.interfaces.stt import STTProvider
from src.core.interfaces.llm import LLMProvider
from src.core.interfaces.tts import TTSProvider
from src.core.interfaces.memory import MemoryProvider

logger = logging.getLogger(__name__)


# Provider registries — add new providers here
_STT_PROVIDERS: dict[str, str] = {
    "whisper": "src.providers.stt.whisper_provider.WhisperSTTProvider",
}

_LLM_PROVIDERS: dict[str, str] = {
    "ollama": "src.providers.llm.ollama_provider.OllamaLLMProvider",
}

_TTS_PROVIDERS: dict[str, str] = {
    "system": "src.providers.tts.system_provider.SystemTTSProvider",
    "kokoro": "src.providers.tts.kokoro_provider.KokoroTTSProvider",
    "qwen": "src.providers.tts.qwen_tts_provider.QwenTTSProvider",
}

_MEMORY_PROVIDERS: dict[str, str] = {
    "chroma": "src.providers.memory.chroma_provider.ChromaMemoryProvider",
    "simple": "src.providers.memory.simple_provider.SimpleMemoryProvider",
}

# Category → (registry, default provider name). The single source of truth
# for what exists; the create_* functions below are typed wrappers over it.
_CATEGORIES: dict[str, tuple[dict[str, str], str]] = {
    "stt": (_STT_PROVIDERS, "whisper"),
    "llm": (_LLM_PROVIDERS, "ollama"),
    "tts": (_TTS_PROVIDERS, "system"),
    "memory": (_MEMORY_PROVIDERS, "simple"),
}


def _import_class(dotted_path: str) -> Type:
    """Import a class from a dotted path string."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def _create_provider(category: str, config: dict):
    """Create and initialize a provider of `category` from config."""
    registry, default_name = _CATEGORIES[category]
    section = config.get(category, {})
    provider_name = section.get("provider", default_name)

    if provider_name not in registry:
        raise ValueError(
            f"Unknown {category} provider: '{provider_name}'. "
            f"Available: {list(registry.keys())}"
        )

    cls = _import_class(registry[provider_name])
    provider = cls()
    await provider.initialize(section)
    detail = f" ({section.get('model', 'default')})" if "model" in section else ""
    logger.info(f"{category.upper()} provider initialized: {provider_name}{detail}")
    return provider


async def create_stt_provider(config: dict) -> STTProvider:
    """Create and initialize an STT provider from config."""
    return await _create_provider("stt", config)


async def create_llm_provider(config: dict) -> LLMProvider:
    """Create and initialize an LLM provider from config."""
    return await _create_provider("llm", config)


async def create_tts_provider(config: dict) -> TTSProvider:
    """Create and initialize a TTS provider from config."""
    return await _create_provider("tts", config)


async def create_memory_provider(config: dict) -> MemoryProvider:
    """Create and initialize a memory provider from config."""
    return await _create_provider("memory", config)


def register_provider(
    category: str, name: str, dotted_path: str
) -> None:
    """
    Register a new provider at runtime.

    Args:
        category: "stt", "llm", "tts", or "memory"
        name: Provider name (used in config)
        dotted_path: Full dotted import path to the provider class

    Example:
        register_provider("tts", "chatterbox", "src.providers.tts.chatterbox_provider.ChatterboxTTSProvider")
    """
    if category not in _CATEGORIES:
        raise ValueError(f"Unknown provider category: {category}")

    _CATEGORIES[category][0][name] = dotted_path
    logger.info(f"Registered {category} provider: {name} → {dotted_path}")


def list_providers() -> dict[str, list[str]]:
    """List all registered providers by category."""
    return {
        category: list(registry.keys())
        for category, (registry, _default) in _CATEGORIES.items()
    }
