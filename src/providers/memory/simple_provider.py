from __future__ import annotations

"""
Simple In-Memory Provider.

A lightweight memory implementation using in-memory storage with
basic keyword-based search. Perfect for prototyping without
requiring ChromaDB or other vector databases.

For production, switch to ChromaMemoryProvider for semantic search.
"""

import logging
import time
import uuid
from typing import AsyncIterator

from src.core.interfaces.memory import Memory, MemoryProvider

logger = logging.getLogger(__name__)


class SimpleMemoryProvider(MemoryProvider):
    """
    Simple in-memory storage with keyword-based search.

    Fast and dependency-free. No vector embeddings — uses
    basic text matching for retrieval.

    For production, use ChromaMemoryProvider for semantic search.
    """

    def __init__(self):
        self.memories: list[Memory] = []

    async def initialize(self, config: dict) -> None:
        logger.info("Simple memory provider initialized (in-memory)")

    async def store(self, memory: Memory) -> None:
        if not memory.id:
            memory.id = str(uuid.uuid4())
        self.memories.append(memory)
        logger.debug(
            f"Stored memory [{memory.memory_type}]: {memory.content[:60]}..."
        )

    async def search(
        self,
        query: str,
        limit: int = 5,
        age_stage: str | None = None,
        memory_type: str | None = None,
    ) -> list[Memory]:
        """Simple keyword-based search (not semantic)."""
        query_words = set(query.lower().split())
        scored: list[tuple[float, Memory]] = []

        for mem in self.memories:
            # Apply filters
            if age_stage and mem.age_stage != age_stage:
                continue
            if memory_type and mem.memory_type != memory_type:
                continue

            # Score by keyword overlap
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words) * mem.importance
                scored.append((score, mem))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]

    async def get_recent(self, limit: int = 10) -> list[Memory]:
        """Get most recent memories."""
        sorted_mems = sorted(
            self.memories, key=lambda m: m.timestamp, reverse=True
        )
        return sorted_mems[:limit]

    async def count(self) -> int:
        return len(self.memories)

    async def clear(self) -> None:
        """Clear all stored memories from the array."""
        self.memories.clear()
        logger.info("Simple memory cleared.")

    async def shutdown(self) -> None:
        logger.info(f"Simple memory shutting down. Total memories: {len(self.memories)}")
