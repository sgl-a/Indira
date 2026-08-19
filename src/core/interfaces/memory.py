from __future__ import annotations

"""
Memory Provider Interface.

Any memory/vector DB implementation (ChromaDB, LanceDB, etc.) must implement this.
Handles short-term, long-term, and emotional memory for the Indira.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator
import time


@dataclass
class Memory:
    """A single memory entry."""

    id: str
    content: str
    # What age stage this memory was formed in
    age_stage: str
    # Emotional context when memory was formed
    emotional_tag: str | None = None
    # How important this memory is (0.0 - 1.0)
    importance: float = 0.5
    # When this memory was created (unix timestamp)
    timestamp: float = field(default_factory=time.time)
    # Memory type classification
    memory_type: str = "interaction"  # "interaction", "observation", "milestone", "emotional"
    # Metadata
    metadata: dict = field(default_factory=dict)


class MemoryProvider(ABC):
    """
    Abstract interface for memory storage and retrieval.

    Implementations:
        - ChromaMemoryProvider (ChromaDB)
        - LanceDBMemoryProvider (LanceDB)
    """

    @abstractmethod
    async def initialize(self, config: dict) -> None:
        """Initialize the provider with configuration."""
        pass

    @abstractmethod
    async def store(self, memory: Memory) -> None:
        """
        Store a memory.

        The provider handles embedding generation internally.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 5,
        age_stage: str | None = None,
        memory_type: str | None = None,
    ) -> list[Memory]:
        """
        Semantic search for relevant memories.

        Args:
            query: Search query (will be embedded)
            limit: Maximum results to return
            age_stage: Filter by age stage (optional)
            memory_type: Filter by memory type (optional)

        Returns:
            List of relevant memories, ordered by relevance
        """
        pass

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[Memory]:
        """
        Get most recent memories (short-term memory).

        Returns chronologically ordered recent memories.
        """
        pass

    async def get_milestones(self) -> list[Memory]:
        """
        Get milestone memories (age transitions, significant events).

        Default implementation searches by type.
        """
        return await self.search("milestone", memory_type="milestone", limit=20)

    @abstractmethod
    async def count(self) -> int:
        """Return total number of stored memories."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored memories."""
        pass

    async def shutdown(self) -> None:
        """Clean up resources."""
        pass
