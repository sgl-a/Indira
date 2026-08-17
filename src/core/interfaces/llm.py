from __future__ import annotations

"""
Language Model Provider Interface.

Any LLM implementation (Ollama, MLX, llama.cpp, etc.) must implement this interface.
The LLM serves as the "brain" of the AI Actor — generating dialogue,
expressing emotion, and staying in character.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class LLMResponse:
    """
    Structured response from the LLM.

    The LLM outputs not just text, but also metadata about how
    to deliver the line (emotion, internal state, etc.)
    """

    # The spoken line (what the audience hears)
    text: str
    # Emotional intent for TTS and avatar
    emotion: str | None = None
    # Raw generation metadata
    generation_time_ms: float = 0
    first_token_time_ms: float = 0  # Time to first token (streaming latency)
    tokens_generated: int = 0


class LLMProvider(ABC):
    """
    Abstract interface for Language Model providers.

    Implementations:
        - OllamaProvider (via Ollama API)
        - MLXProvider (Apple MLX framework)
        - LlamaCppProvider (llama.cpp server)
    """

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize the provider with configuration."""
        pass

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """
        Generate a complete response.

        Args:
            system_prompt: System-level instructions (personality, age, rules)
            messages: Conversation history [{role: "user"/"assistant", content: "..."}]
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with text, emotion, and metadata
        """
        pass

    async def stream_generate_with_metadata(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str | LLMResponse]:
        """
        Stream text chunks, then yield a final LLMResponse with metadata.

        Default implementation falls back to non-streaming generate() —
        providers with real token streaming should override this.
        """
        response = await self.generate(
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield response.text
        yield response

    @abstractmethod
    async def get_model_info(self) -> dict:
        """
        Return model metadata (name, size, parameters).
        Useful for benchmarking.
        """
        pass

    async def shutdown(self) -> None:
        """Clean up resources."""
        pass
