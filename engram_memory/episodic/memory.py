"""
Episodic Memory implementation for Engram.

Episodic memory stores past experiences and conversations using
vector embeddings for semantic similarity search.

Key concepts:
- Episode: A single memory unit (conversation, task, debugging session)
- Embedding: Numerical representation for similarity search
- Retrieval: Find relevant past experiences based on current context
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union
import uuid
import json
from pathlib import Path

from engram_memory.core.interfaces import (
    BaseMemory,
    MemoryType,
    MemoryEntry,
    RetrievalQuery,
    RetrievalResult,
    ConversationTurn,
    EmbeddingProvider,
)


class EpisodeType(Enum):
    """Types of episodic memories."""

    CONVERSATION = "conversation"  # A conversation session
    TASK = "task"  # Completed a task
    DEBUGGING = "debugging"  # Debugging session
    DECISION = "decision"  # Made a decision
    LEARNING = "learning"  # Learned something new
    ERROR = "error"  # Encountered an error
    CODE_REVIEW = "code_review"  # Code review session
    RESEARCH = "research"  # Research/exploration


@dataclass
class Episode(MemoryEntry):
    """
    A single episodic memory - a past experience.

    Episodes are the core unit of episodic memory. Each episode
    captures a meaningful interaction or event that might be
    useful to recall later.
    """

    # Core content
    type: EpisodeType = EpisodeType.CONVERSATION
    summary: str = ""  # Brief summary (used for embedding)
    content: str = ""  # Full content/details

    # Context
    project: Optional[str] = None  # Which project this was about
    files: list[str] = field(default_factory=list)  # Files involved
    tags: list[str] = field(default_factory=list)  # Searchable tags

    # Outcome
    outcome: Optional[str] = None  # How did it end?
    key_points: list[str] = field(default_factory=list)  # Important takeaways

    # Importance and access tracking
    importance: float = 0.5  # 0-1, how important is this memory
    access_count: int = 0  # Times retrieved
    last_accessed: Optional[datetime] = None

    def to_embed_text(self) -> str:
        """
        Create text for embedding.

        Combines summary, key points, and tags into a single
        string optimized for semantic search.
        """
        parts = [self.summary]

        if self.key_points:
            parts.append("Key points: " + "; ".join(self.key_points))

        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))

        if self.project:
            parts.append(f"Project: {self.project}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "summary": self.summary,
            "content": self.content,
            "project": self.project,
            "files": self.files,
            "tags": self.tags,
            "outcome": self.outcome,
            "key_points": self.key_points,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=EpisodeType(data["type"]),
            summary=data["summary"],
            content=data.get("content", ""),
            project=data.get("project"),
            files=data.get("files", []),
            tags=data.get("tags", []),
            outcome=data.get("outcome"),
            key_points=data.get("key_points", []),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


class EpisodicMemory(BaseMemory[Episode]):
    """
    Episodic memory implementation.

    Stores past experiences with vector embeddings for semantic search.
    Think of it as your agent's autobiography - it remembers what happened,
    when, and can recall relevant experiences when needed.

    Usage:
        # Initialize with ChromaDB (default)
        from engram_memory import EpisodicMemory
        from engram_memory.episodic import ChromaDBStore
        from engram_memory.utils import LocalEmbeddings

        memory = EpisodicMemory(
            store=ChromaDBStore(persist_directory="./engram_data"),
            embeddings=LocalEmbeddings()
        )

        # Store an episode
        episode = Episode(
            type=EpisodeType.CONVERSATION,
            summary="Helped user implement JWT authentication",
            key_points=["Used httpOnly cookies", "Added refresh token rotation"],
            project="myapp",
            tags=["auth", "jwt", "security"]
        )
        await memory.store(episode)

        # Retrieve relevant episodes
        results = await memory.retrieve(RetrievalQuery(
            text="How did we handle token refresh?",
            limit=5
        ))
    """

    def __init__(
        self,
        store: "VectorStore",
        embeddings: EmbeddingProvider,
        data_path: Optional[str] = None,
    ):
        """
        Initialize episodic memory.

        Args:
            store: Vector store backend (ChromaDBStore, QdrantStore, etc.)
            embeddings: Embedding provider (OpenAIEmbeddings, LocalEmbeddings, etc.)
            data_path: Optional path to store episode metadata (for full episode data)
        """
        self._store = store
        self._embeddings = embeddings
        self._data_path = Path(data_path) if data_path else None

        # In-memory cache of full episode data
        # (Vector store only holds embeddings + minimal metadata)
        self._episodes: dict[str, Episode] = {}

        # Load existing episodes if data path exists
        if self._data_path:
            self._load_episodes()

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.EPISODIC

    async def store(self, entry: Episode) -> str:
        """
        Store an episode in memory.

        Args:
            entry: The episode to store

        Returns:
            The episode ID
        """
        # Generate ID if not set
        if not entry.id:
            entry.id = str(uuid.uuid4())

        # Generate embedding
        embed_text = entry.to_embed_text()
        embedding = await self._embeddings.embed(embed_text)

        # Prepare metadata for vector store
        # (Keep it minimal - full data stored separately)
        metadata = {
            "type": entry.type.value,
            "summary": entry.summary[:500],  # Truncate for storage
            "project": entry.project or "",
            "tags": entry.tags,
            "importance": entry.importance,
            "created_at": entry.created_at.isoformat(),
        }

        # Store in vector database
        await self._store.add(
            id=entry.id,
            embedding=embedding,
            metadata=metadata,
            document=embed_text,
        )

        # Store full episode data
        self._episodes[entry.id] = entry
        self._save_episodes()

        return entry.id

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """
        Retrieve relevant episodes based on semantic similarity.

        Args:
            query: The retrieval query

        Returns:
            List of retrieval results sorted by relevance
        """
        # Generate query embedding
        query_embedding = await self._embeddings.embed(query.text)

        # Build filters for vector store
        filters = {}
        if query.metadata_filters:
            filters.update(query.metadata_filters)

        # Search vector store
        results = await self._store.search(
            embedding=query_embedding,
            limit=query.limit,
            filters=filters if filters else None,
        )

        # Build retrieval results
        retrieval_results = []
        for doc_id, similarity, metadata in results:
            # Skip if below minimum relevance
            if similarity < query.min_relevance:
                continue

            # Get full episode data
            episode = self._episodes.get(doc_id)
            if not episode:
                # Try to reconstruct from metadata if not in cache
                episode = Episode(
                    id=doc_id,
                    type=EpisodeType(metadata.get("type", "conversation")),
                    summary=metadata.get("summary", ""),
                    project=metadata.get("project"),
                    tags=metadata.get("tags", []),
                    importance=metadata.get("importance", 0.5),
                )

            # Update access stats
            episode.access_count += 1
            episode.last_accessed = datetime.utcnow()

            retrieval_results.append(
                RetrievalResult(
                    entry=episode,
                    relevance_score=similarity,
                    memory_type=MemoryType.EPISODIC,
                )
            )

        # Save updated access stats
        self._save_episodes()

        return retrieval_results

    async def update(self, entry_id: str, updates: dict) -> bool:
        """
        Update an existing episode.

        Args:
            entry_id: ID of the episode to update
            updates: Dictionary of field updates

        Returns:
            True if successful
        """
        if entry_id not in self._episodes:
            return False

        episode = self._episodes[entry_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(episode, key):
                setattr(episode, key, value)

        episode.updated_at = datetime.utcnow()

        # Re-embed if content changed
        content_fields = {"summary", "key_points", "tags", "project"}
        if content_fields & set(updates.keys()):
            embed_text = episode.to_embed_text()
            embedding = await self._embeddings.embed(embed_text)

            metadata = {
                "type": episode.type.value,
                "summary": episode.summary[:500],
                "project": episode.project or "",
                "tags": episode.tags,
                "importance": episode.importance,
                "created_at": episode.created_at.isoformat(),
            }

            # Update in vector store
            await self._store.delete(entry_id)
            await self._store.add(
                id=entry_id,
                embedding=embedding,
                metadata=metadata,
                document=embed_text,
            )

        self._save_episodes()
        return True

    async def delete(self, entry_id: str) -> bool:
        """
        Delete an episode.

        Args:
            entry_id: ID of the episode to delete

        Returns:
            True if successful
        """
        if entry_id in self._episodes:
            del self._episodes[entry_id]
            self._save_episodes()

        return await self._store.delete(entry_id)

    async def clear(self) -> None:
        """Clear all episodic memories."""
        self._episodes.clear()
        await self._store.clear()

        if self._data_path and self._data_path.exists():
            self._data_path.unlink()

    # ========== Episodic Memory Specific Methods ==========

    async def store_conversation(
        self,
        turns: list[Union[ConversationTurn, dict]],
        summary: Optional[str] = None,
        project: Optional[str] = None,
        files: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> Episode:
        """
        Create and store an episode from a conversation.

        This is a convenience method for the common case of
        storing a conversation as an episode.

        Args:
            turns: List of conversation turns
            summary: Optional summary (auto-generated if not provided)
            project: Project name
            files: Files involved in the conversation
            tags: Searchable tags

        Returns:
            The created episode
        """
        # Convert dicts to ConversationTurn if needed
        normalized_turns = []
        for turn in turns:
            if isinstance(turn, dict):
                normalized_turns.append(turn)
            else:
                normalized_turns.append(turn.to_dict())

        # Generate summary if not provided
        if not summary:
            # Use first user message as summary
            for turn in normalized_turns:
                if turn.get("role") == "user":
                    summary = turn.get("content", "")[:200]
                    break
            if not summary:
                summary = "Conversation"

        # Extract key points from assistant responses
        key_points = []
        for turn in normalized_turns:
            if turn.get("role") == "assistant":
                content = turn.get("content", "")
                # Take first sentence as a key point
                first_sentence = content.split(".")[0]
                if len(first_sentence) < 200:
                    key_points.append(first_sentence)
                if len(key_points) >= 3:
                    break

        # Create episode
        episode = Episode(
            type=EpisodeType.CONVERSATION,
            summary=summary,
            content=json.dumps(normalized_turns),
            project=project,
            files=files or [],
            tags=tags or [],
            key_points=key_points,
            importance=self._calculate_importance(normalized_turns, key_points),
        )

        await self.store(episode)
        return episode

    async def get_recent(self, limit: int = 10) -> list[Episode]:
        """
        Get most recent episodes.

        Args:
            limit: Maximum number of episodes to return

        Returns:
            List of recent episodes
        """
        sorted_episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.created_at,
            reverse=True,
        )
        return sorted_episodes[:limit]

    async def get_by_project(
        self,
        project: str,
        limit: int = 20,
    ) -> list[Episode]:
        """
        Get episodes for a specific project.

        Args:
            project: Project name
            limit: Maximum number of episodes

        Returns:
            List of episodes for the project
        """
        return [
            e for e in self._episodes.values()
            if e.project == project
        ][:limit]

    async def get_by_type(
        self,
        episode_type: EpisodeType,
        limit: int = 20,
    ) -> list[Episode]:
        """
        Get episodes of a specific type.

        Args:
            episode_type: Type of episodes to retrieve
            limit: Maximum number of episodes

        Returns:
            List of episodes of that type
        """
        return [
            e for e in self._episodes.values()
            if e.type == episode_type
        ][:limit]

    async def search_by_tags(
        self,
        tags: list[str],
        limit: int = 20,
    ) -> list[Episode]:
        """
        Get episodes matching any of the given tags.

        Args:
            tags: Tags to search for
            limit: Maximum number of episodes

        Returns:
            List of matching episodes
        """
        tags_set = set(tags)
        return [
            e for e in self._episodes.values()
            if tags_set & set(e.tags)
        ][:limit]

    async def count(self) -> int:
        """Return the number of stored episodes."""
        return await self._store.count()

    def _calculate_importance(
        self,
        turns: list[dict],
        key_points: list[str],
    ) -> float:
        """
        Calculate importance score for an episode.

        Factors:
        - Conversation length
        - Number of key points
        - Presence of code
        - Presence of decisions
        """
        score = 0.5

        # Longer conversations might be more important
        if len(turns) > 10:
            score += 0.1
        elif len(turns) > 20:
            score += 0.2

        # More key points = more important
        score += min(len(key_points) * 0.05, 0.2)

        # Check for code blocks (likely technical discussion)
        full_text = " ".join(t.get("content", "") for t in turns)
        if "```" in full_text:
            score += 0.1

        # Check for decision language
        decision_words = ["decided", "chose", "will use", "going with", "selected"]
        if any(word in full_text.lower() for word in decision_words):
            score += 0.1

        return min(score, 1.0)

    # ========== Persistence ==========

    def _save_episodes(self) -> None:
        """Save episode metadata to disk."""
        if not self._data_path:
            return

        self._data_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            episode_id: episode.to_dict()
            for episode_id, episode in self._episodes.items()
        }

        with open(self._data_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_episodes(self) -> None:
        """Load episode metadata from disk."""
        if not self._data_path or not self._data_path.exists():
            return

        try:
            with open(self._data_path, "r") as f:
                data = json.load(f)

            for episode_id, episode_data in data.items():
                self._episodes[episode_id] = Episode.from_dict(episode_data)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not load episodes: {e}")
