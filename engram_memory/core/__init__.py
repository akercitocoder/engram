"""Core interfaces and coordinator for Engram Memory."""

from engram_memory.core.interfaces import (
    MemoryType,
    MemoryEntry,
    RetrievalQuery,
    RetrievalResult,
    ConversationTurn,
    MemoryContext,
    BaseMemory,
)

__all__ = [
    "MemoryType",
    "MemoryEntry",
    "RetrievalQuery",
    "RetrievalResult",
    "ConversationTurn",
    "MemoryContext",
    "BaseMemory",
]
