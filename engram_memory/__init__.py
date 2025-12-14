"""
Engram - Memory architecture for AI agents.

An engram is the physical trace a memory leaves in the brain.
This package gives AI agents the same capability - persistent,
human-like memory across sessions.

Memory Types:
- Working Memory: Current context and conversation state
- Episodic Memory: Past experiences and conversations (vector DB)
- Semantic Memory: Facts, entities, and relationships (graph DB)
- Procedural Memory: Learned preferences and patterns

Quick Start:
    from engram_memory import WorkingMemory, EpisodicMemory
    from engram_memory.episodic import ChromaDBStore
    from engram_memory.utils import LocalEmbeddings

    # Working memory (current conversation)
    working = WorkingMemory()
    working.add_turn("user", "Help me with authentication")

    # Episodic memory (past experiences)
    episodic = EpisodicMemory(
        store=ChromaDBStore(persist_directory="./engram_data"),
        embeddings=LocalEmbeddings()
    )
"""

from engram_memory.core.interfaces import (
    MemoryType,
    MemoryEntry,
    RetrievalQuery,
    RetrievalResult,
    ConversationTurn,
    MemoryContext,
)
from engram_memory.working.memory import WorkingMemory
from engram_memory.episodic.memory import EpisodicMemory, Episode, EpisodeType
from engram_memory.episodic.storage import ChromaDBStore, QdrantStore
from engram_memory.semantic.memory import SemanticMemory, Entity, EntityType, Relationship, Fact

__version__ = "0.1.0"

__all__ = [
    # Core types
    "MemoryType",
    "MemoryEntry",
    "RetrievalQuery",
    "RetrievalResult",
    "ConversationTurn",
    "MemoryContext",
    # Working Memory
    "WorkingMemory",
    # Episodic Memory
    "EpisodicMemory",
    "Episode",
    "EpisodeType",
    # Semantic Memory
    "SemanticMemory",
    "Entity",
    "EntityType",
    "Relationship",
    "Fact",
    # Storage backends
    "ChromaDBStore",
    "QdrantStore",
]
