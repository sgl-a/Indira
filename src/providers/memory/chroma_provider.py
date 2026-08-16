from __future__ import annotations

"""
ChromaDB Memory Provider.

Production-ready memory implementation using ChromaDB for semantic
vector search. Stores memories as embeddings and retrieves by
meaning rather than keyword match.

Requirements:
    pip install chromadb

Switch from SimpleMemoryProvider by setting config:
    memory:
      provider: "chroma"
"""

import logging
import time
import uuid
from collections import deque
from pathlib import Path

from src.core.interfaces.memory import Memory, MemoryProvider

logger = logging.getLogger(__name__)

# How many of the newest memories to keep in the recency buffer.
# Chroma's get(limit=N) returns an arbitrary slice (no ORDER BY), so
# recency is tracked here instead of queried from the collection.
_RECENT_BUFFER_SIZE = 100


class ChromaMemoryProvider(MemoryProvider):
    """
    ChromaDB-backed memory with semantic search.

    Uses ChromaDB's built-in embedding model for vector similarity
    search, enabling retrieval by meaning rather than keywords.

    Features:
        - Semantic search for relevant memories
        - Persistent storage across sessions
        - Metadata filtering (age stage, memory type, emotion)
        - Importance-weighted retrieval
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self._persist_dir: str | None = None
        # Newest-first buffer of recent memories (see _RECENT_BUFFER_SIZE)
        self._recent: deque[Memory] = deque(maxlen=_RECENT_BUFFER_SIZE)

    async def initialize(self, config: dict) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB is required for ChromaMemoryProvider. "
                "Install with: pip install chromadb"
            )

        memory_config = config.get("memory", {})
        self._persist_dir = memory_config.get("persist_directory", "data/memory")
        collection_name = memory_config.get("collection_name", "ai_actor_memories")

        # Ensure persist directory exists
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection
        # ChromaDB uses its own default embedding function (all-MiniLM-L6-v2)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        existing_count = self.collection.count()
        self._seed_recent_buffer()
        logger.info(
            f"ChromaDB initialized. Collection: {collection_name}, "
            f"Existing memories: {existing_count}, "
            f"Persist: {self._persist_dir}"
        )

    def _seed_recent_buffer(self) -> None:
        """Load the newest memories from the collection into the recency buffer."""
        self._recent.clear()
        if not self.collection or self.collection.count() == 0:
            return
        try:
            all_results = self.collection.get()  # full scan, once at startup
        except Exception as e:
            logger.warning(f"Could not seed recent-memory buffer: {e}")
            return
        memories = [
            self._to_memory(
                all_results["ids"][i],
                all_results["metadatas"][i] if all_results["metadatas"] else {},
                all_results["documents"][i] if all_results["documents"] else "",
            )
            for i in range(len(all_results["ids"]))
        ]
        memories.sort(key=lambda m: m.timestamp)  # oldest → newest
        for mem in memories[-_RECENT_BUFFER_SIZE:]:
            self._recent.appendleft(mem)

    @staticmethod
    def _to_memory(mem_id: str, meta: dict, content: str) -> Memory:
        """Build a Memory from ChromaDB id/metadata/document."""
        return Memory(
            id=mem_id,
            content=content,
            age_stage=meta.get("age_stage", "unknown"),
            emotional_tag=meta.get("emotional_tag"),
            importance=meta.get("importance", 0.5),
            timestamp=meta.get("timestamp", 0.0),
            memory_type=meta.get("memory_type", "interaction"),
        )

    async def store(self, memory: Memory) -> None:
        """Store a memory with its embedding."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")

        if not memory.id:
            memory.id = str(uuid.uuid4())

        # Build metadata for filtering
        metadata = {
            "age_stage": memory.age_stage,
            "importance": memory.importance,
            "memory_type": memory.memory_type,
            "timestamp": memory.timestamp,
        }
        if memory.emotional_tag:
            metadata["emotional_tag"] = memory.emotional_tag

        # Add any extra metadata
        if memory.metadata:
            for k, v in memory.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v

        self.collection.add(
            ids=[memory.id],
            documents=[memory.content],
            metadatas=[metadata],
        )
        self._recent.appendleft(memory)

        logger.debug(
            f"Stored memory [{memory.memory_type}] "
            f"(age={memory.age_stage}): {memory.content[:60]}..."
        )

    async def search(
        self,
        query: str,
        limit: int = 5,
        age_stage: str | None = None,
        memory_type: str | None = None,
    ) -> list[Memory]:
        """Semantic search for relevant memories."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")

        if self.collection.count() == 0:
            return []

        # Build where filter
        where_conditions = []
        if age_stage:
            where_conditions.append({"age_stage": age_stage})
        if memory_type:
            where_conditions.append({"memory_type": memory_type})

        where = None
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(limit, self.collection.count()),
                where=where if where else None,
            )
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
            return []

        # Convert results to Memory objects
        memories = []
        if results and results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""
                memories.append(self._to_memory(mem_id, meta, content))

        return memories

    async def get_recent(self, limit: int = 10) -> list[Memory]:
        """
        Get most recent memories, newest first.

        Served from the in-process recency buffer: ChromaDB's get(limit=N)
        returns an arbitrary slice, so querying it for "recent" silently
        returns the oldest memories once the collection grows.
        """
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")

        return list(self._recent)[:limit]

    async def count(self) -> int:
        if not self.collection:
            return 0
        return self.collection.count()

    async def clear(self) -> None:
        """Clear all stored memories."""
        if not self.collection or not self.client:
            return
            
        name = self.collection.name
        try:
            self.client.delete_collection(name)
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            self._recent.clear()
            logger.info(f"ChromaDB memory cleared. Collection {name} reset.")
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB memory: {e}")

    async def shutdown(self) -> None:
        count = await self.count()
        logger.info(
            f"ChromaDB shutting down. "
            f"Total memories: {count}, "
            f"Persisted to: {self._persist_dir}"
        )
