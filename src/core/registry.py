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


def _import_class(dotted_path: str) -> Type:
    """Import a class from a dotted path string."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def create_stt_provider(config: dict) -> STTProvider:
    """Create and initialize an STT provider from config."""
    stt_config = config.get("stt", {})
    provider_name = stt_config.get("provider", "whisper")

    if provider_name not in _STT_PROVIDERS:
        raise ValueError(
            f"Unknown STT provider: '{provider_name}'. "
            f"Available: {list(_STT_PROVIDERS.keys())}"
        )

    cls = _import_class(_STT_PROVIDERS[provider_name])
    provider = cls()
    await provider.initialize(stt_config)
    logger.info(f"STT provider initialized: {provider_name}")
    return provider


async def create_llm_provider(config: dict) -> LLMProvider:
    """Create and initialize an LLM provider from config."""
    llm_config = config.get("llm", {})
    provider_name = llm_config.get("provider", "ollama")

    if provider_name not in _LLM_PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Available: {list(_LLM_PROVIDERS.keys())}"
        )

    cls = _import_class(_LLM_PROVIDERS[provider_name])
    provider = cls()
    await provider.initialize(llm_config)
    logger.info(f"LLM provider initialized: {provider_name} ({llm_config.get('model', 'default')})")
    return provider


async def create_tts_provider(config: dict) -> TTSProvider:
    """Create and initialize a TTS provider from config."""
    tts_config = config.get("tts", {})
    provider_name = tts_config.get("provider", "system")

    if provider_name not in _TTS_PROVIDERS:
        raise ValueError(
            f"Unknown TTS provider: '{provider_name}'. "
            f"Available: {list(_TTS_PROVIDERS.keys())}"
        )

    cls = _import_class(_TTS_PROVIDERS[provider_name])
    provider = cls()
    await provider.initialize(tts_config)
    logger.info(f"TTS provider initialized: {provider_name}")
    return provider


async def create_memory_provider(config: dict) -> MemoryProvider:
    """Create and initialize a memory provider from config."""
    memory_config = config.get("memory", {})
    provider_name = memory_config.get("provider", "simple")

    if provider_name not in _MEMORY_PROVIDERS:
        raise ValueError(
            f"Unknown memory provider: '{provider_name}'. "
            f"Available: {list(_MEMORY_PROVIDERS.keys())}"
        )

    cls = _import_class(_MEMORY_PROVIDERS[provider_name])
    provider = cls()
    await provider.initialize(memory_config)
    logger.info(f"Memory provider initialized: {provider_name}")
    return provider


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
    registries = {
        "stt": _STT_PROVIDERS,
        "llm": _LLM_PROVIDERS,
        "tts": _TTS_PROVIDERS,
        "memory": _MEMORY_PROVIDERS,
    }

    if category not in registries:
        raise ValueError(f"Unknown provider category: {category}")

    registries[category][name] = dotted_path
    logger.info(f"Registered {category} provider: {name} → {dotted_path}")


def list_providers() -> dict[str, list[str]]:
    """List all registered providers by category."""
    return {
        "stt": list(_STT_PROVIDERS.keys()),
        "llm": list(_LLM_PROVIDERS.keys()),
        "tts": list(_TTS_PROVIDERS.keys()),
        "memory": list(_MEMORY_PROVIDERS.keys()),
    }
