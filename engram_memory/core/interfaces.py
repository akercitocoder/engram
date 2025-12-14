"""
Core interfaces and data models for Engram.

This module defines the abstract base classes and data structures
that all memory implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar, Optional, Protocol


T = TypeVar("T")


class MemoryType(Enum):
    """Types of memory in the Engram system."""

    WORKING = "working"  # Current context, conversation buffer
    EPISODIC = "episodic"  # Past experiences, conversations
    SEMANTIC = "semantic"  # Facts, entities, relationships
    PROCEDURAL = "procedural"  # Learned preferences, patterns


@dataclass
class MemoryEntry:
    """Base class for all memory entries."""

    id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalQuery:
    """Query for retrieving memories."""

    text: str  # Natural language query
    memory_types: list[MemoryType] = field(default_factory=lambda: list(MemoryType))
    limit: int = 10
    min_relevance: float = 0.0  # 0-1, filter by relevance score
    time_range: Optional[tuple[datetime, datetime]] = None
    metadata_filters: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""

    entry: Any  # The memory entry (type varies by memory type)
    relevance_score: float  # 0-1, how relevant to query
    memory_type: MemoryType
    reasoning: Optional[str] = None  # Why this was retrieved


@dataclass
class MemoryContext:
    """
    Context assembled from all memory types for the LLM.

    This is what gets injected into the prompt to give the agent
    access to its memories.
    """

    working: dict  # Current state
    episodic: list[RetrievalResult]  # Relevant past experiences
    semantic: list[RetrievalResult]  # Relevant facts/entities
    procedural: dict  # Relevant preferences/patterns

    def to_prompt_section(self) -> str:
        """Format memories for inclusion in LLM prompt."""
        sections = []

        # Episodic memories
        if self.episodic:
            sections.append("## Relevant Past Experiences")
            for r in self.episodic:
                entry = r.entry
                summary = getattr(entry, "summary", str(entry))
                sections.append(f"- [{r.relevance_score:.0%}] {summary}")

        # Semantic memories
        if self.semantic:
            sections.append("\n## Relevant Knowledge")
            for r in self.semantic:
                entry = r.entry
                name = getattr(entry, "name", str(entry))
                desc = getattr(entry, "description", "")
                sections.append(f"- {name}: {desc}" if desc else f"- {name}")

        # Procedural memories (preferences)
        if self.procedural:
            sections.append("\n## User Preferences & Patterns")
            for category, prefs in self.procedural.items():
                if category.startswith("_"):
                    continue  # Skip internal keys
                if isinstance(prefs, dict):
                    for key, value in prefs.items():
                        if isinstance(value, dict):
                            sections.append(f"- {key}: {value.get('value', value)}")
                        else:
                            sections.append(f"- {key}: {value}")

            # Handle corrections specially
            corrections = self.procedural.get("_corrections", [])
            if corrections:
                sections.append("\n### Learned Corrections (avoid these mistakes)")
                for c in corrections:
                    sections.append(f"- Don't: {c['wrong']}")
                    sections.append(f"  Do: {c['correct']}")

        return "\n".join(sections) if sections else ""

    def is_empty(self) -> bool:
        """Check if there's any relevant context."""
        return (
            not self.episodic
            and not self.semantic
            and not self.procedural
        )


class BaseMemory(ABC, Generic[T]):
    """
    Abstract base class for all memory types.

    Each memory type (working, episodic, semantic, procedural) must
    implement this interface.
    """

    @abstractmethod
    async def store(self, entry: T) -> str:
        """
        Store a memory entry.

        Args:
            entry: The memory entry to store

        Returns:
            The ID of the stored entry
        """
        pass

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """
        Retrieve relevant memories based on query.

        Args:
            query: The retrieval query with filters

        Returns:
            List of retrieval results sorted by relevance
        """
        pass

    @abstractmethod
    async def update(self, entry_id: str, updates: dict) -> bool:
        """
        Update an existing memory entry.

        Args:
            entry_id: ID of the entry to update
            updates: Dictionary of field updates

        Returns:
            True if successful, False if entry not found
        """
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if successful, False if entry not found
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memories of this type."""
        pass

    @property
    @abstractmethod
    def memory_type(self) -> MemoryType:
        """Return the type of this memory."""
        pass


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...


class VectorStore(Protocol):
    """Protocol for vector storage backend."""

    async def add(self, id: str, embedding: list[float], metadata: dict) -> None:
        """Add a vector with metadata."""
        ...

    async def search(
        self, embedding: list[float], limit: int, filters: dict = None
    ) -> list[tuple[str, float, dict]]:
        """
        Search for similar vectors.

        Returns list of (id, similarity_score, metadata) tuples.
        """
        ...

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        ...

    async def clear(self) -> None:
        """Clear all vectors."""
        ...


class GraphStore(Protocol):
    """Protocol for graph storage backend."""

    async def add_node(self, id: str, labels: list[str], properties: dict) -> None:
        """Add a node to the graph."""
        ...

    async def add_edge(
        self, source_id: str, target_id: str, relation: str, properties: dict = None
    ) -> None:
        """Add an edge between nodes."""
        ...

    async def get_node(self, id: str) -> Optional[dict]:
        """Get a node by ID."""
        ...

    async def get_neighbors(
        self, id: str, relation: str = None, direction: str = "outgoing"
    ) -> list[tuple[dict, str, dict]]:
        """
        Get neighboring nodes.

        Returns list of (node, relation_type, edge_properties) tuples.
        """
        ...

    async def delete_node(self, id: str) -> bool:
        """Delete a node and its edges."""
        ...

    async def clear(self) -> None:
        """Clear the entire graph."""
        ...


class LLMClient(Protocol):
    """Protocol for LLM client."""

    async def complete(self, prompt: str) -> str:
        """Generate a completion for a prompt."""
        ...

    async def chat(self, messages: list[dict]) -> str:
        """Generate a chat completion."""
        ...
