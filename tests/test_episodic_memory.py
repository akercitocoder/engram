"""
Tests for EpisodicMemory implementation.
"""

import pytest
from datetime import datetime

from engram_memory import (
    EpisodicMemory,
    Episode,
    EpisodeType,
    RetrievalQuery,
)
from engram_memory.core.interfaces import MemoryType


class TestEpisodeDataclass:
    """Test Episode dataclass functionality."""

    def test_episode_creation(self, sample_episode):
        """Test basic episode creation."""
        assert sample_episode.type == EpisodeType.CONVERSATION
        assert sample_episode.summary == "Implemented JWT authentication with httpOnly cookies"
        assert sample_episode.project == "test-project"

    def test_episode_default_values(self):
        """Test episode default values."""
        episode = Episode(summary="Test")

        assert episode.type == EpisodeType.CONVERSATION
        assert episode.content == ""
        assert episode.project is None
        assert episode.files == []
        assert episode.tags == []
        assert episode.importance == 0.5
        assert episode.access_count == 0
        assert episode.id is not None

    def test_episode_to_embed_text(self, sample_episode):
        """Test episode embedding text generation."""
        embed_text = sample_episode.to_embed_text()

        assert "Implemented JWT authentication" in embed_text
        assert "Key points:" in embed_text
        assert "Tags:" in embed_text
        assert "Project: test-project" in embed_text

    def test_episode_to_dict(self, sample_episode):
        """Test episode serialization."""
        data = sample_episode.to_dict()

        assert data["type"] == "conversation"
        assert data["summary"] == sample_episode.summary
        assert data["project"] == "test-project"
        assert "created_at" in data
        assert "updated_at" in data

    def test_episode_from_dict(self, sample_episode):
        """Test episode deserialization."""
        data = sample_episode.to_dict()
        restored = Episode.from_dict(data)

        assert restored.id == sample_episode.id
        assert restored.type == sample_episode.type
        assert restored.summary == sample_episode.summary
        assert restored.tags == sample_episode.tags


class TestEpisodeTypes:
    """Test different episode types."""

    def test_conversation_type(self):
        """Test conversation episode type."""
        episode = Episode(
            type=EpisodeType.CONVERSATION,
            summary="Had a conversation about databases"
        )
        assert episode.type == EpisodeType.CONVERSATION
        assert episode.type.value == "conversation"

    def test_debugging_type(self):
        """Test debugging episode type."""
        episode = Episode(
            type=EpisodeType.DEBUGGING,
            summary="Fixed memory leak",
            outcome="resolved"
        )
        assert episode.type == EpisodeType.DEBUGGING
        assert episode.outcome == "resolved"

    def test_all_episode_types(self):
        """Test all episode types can be created."""
        types = [
            EpisodeType.CONVERSATION,
            EpisodeType.TASK,
            EpisodeType.DEBUGGING,
            EpisodeType.DECISION,
            EpisodeType.LEARNING,
            EpisodeType.ERROR,
            EpisodeType.CODE_REVIEW,
            EpisodeType.RESEARCH,
        ]

        for ep_type in types:
            episode = Episode(type=ep_type, summary=f"Test {ep_type.value}")
            assert episode.type == ep_type


class TestEpisodicMemoryStorage:
    """Test EpisodicMemory storage operations."""

    @pytest.mark.asyncio
    async def test_store_episode(self, episodic_memory, sample_episode):
        """Test storing an episode."""
        episode_id = await episodic_memory.store(sample_episode)

        assert episode_id is not None
        assert episode_id == sample_episode.id

    @pytest.mark.asyncio
    async def test_store_multiple_episodes(self, episodic_memory, sample_episodes):
        """Test storing multiple episodes."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        count = await episodic_memory.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_store_generates_id(self, episodic_memory):
        """Test that storing generates ID if not set."""
        episode = Episode(
            summary="Test episode without ID",
            tags=["test"]
        )
        episode.id = ""  # Clear the default ID

        episode_id = await episodic_memory.store(episode)

        assert episode_id is not None
        assert len(episode_id) > 0


class TestEpisodicMemoryRetrieval:
    """Test EpisodicMemory retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_by_text(self, episodic_memory, sample_episodes):
        """Test retrieving episodes by text similarity."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        # Use exact text from one of the episodes for reliable matching
        # with deterministic mock embeddings
        first_episode = sample_episodes[0]
        query_text = first_episode.to_embed_text()

        results = await episodic_memory.retrieve(
            RetrievalQuery(text=query_text, limit=5, min_relevance=0.0)
        )

        assert len(results) > 0
        # Results should be RetrievalResult objects
        assert results[0].memory_type == MemoryType.EPISODIC
        assert hasattr(results[0], "relevance_score")

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self, episodic_memory, sample_episodes):
        """Test retrieve respects limit parameter."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        results = await episodic_memory.retrieve(
            RetrievalQuery(text="test", limit=1)
        )

        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_retrieve_empty_store(self, episodic_memory):
        """Test retrieving from empty store."""
        results = await episodic_memory.retrieve(
            RetrievalQuery(text="anything")
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_updates_access_stats(self, episodic_memory, sample_episode):
        """Test that retrieval updates access statistics."""
        await episodic_memory.store(sample_episode)

        # Use exact embed text for retrieval to ensure match with mock embeddings
        query_text = sample_episode.to_embed_text()
        results = await episodic_memory.retrieve(RetrievalQuery(text=query_text, min_relevance=0.0))

        # Verify we got results
        assert len(results) >= 1, "Should find stored episode"

        episode = episodic_memory._episodes.get(sample_episode.id)
        assert episode.access_count >= 1
        assert episode.last_accessed is not None


class TestEpisodicMemoryUpdate:
    """Test EpisodicMemory update operations."""

    @pytest.mark.asyncio
    async def test_update_episode(self, episodic_memory, sample_episode):
        """Test updating an episode."""
        await episodic_memory.store(sample_episode)

        result = await episodic_memory.update(
            sample_episode.id,
            {"importance": 0.95}
        )

        assert result is True
        episode = episodic_memory._episodes[sample_episode.id]
        assert episode.importance == 0.95

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, episodic_memory):
        """Test updating non-existent episode."""
        result = await episodic_memory.update(
            "nonexistent-id",
            {"importance": 0.9}
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_content_reembeds(self, episodic_memory, sample_episode):
        """Test that updating content triggers re-embedding."""
        await episodic_memory.store(sample_episode)
        original_updated = sample_episode.updated_at

        await episodic_memory.update(
            sample_episode.id,
            {"summary": "New summary about different topic"}
        )

        episode = episodic_memory._episodes[sample_episode.id]
        assert episode.summary == "New summary about different topic"
        assert episode.updated_at > original_updated


class TestEpisodicMemoryDelete:
    """Test EpisodicMemory delete operations."""

    @pytest.mark.asyncio
    async def test_delete_episode(self, episodic_memory, sample_episode):
        """Test deleting an episode."""
        await episodic_memory.store(sample_episode)

        result = await episodic_memory.delete(sample_episode.id)

        assert result is True
        assert sample_episode.id not in episodic_memory._episodes

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, episodic_memory):
        """Test deleting non-existent episode."""
        result = await episodic_memory.delete("nonexistent-id")

        # Should still return True from store (no error)
        assert result is True

    @pytest.mark.asyncio
    async def test_clear(self, episodic_memory, sample_episodes):
        """Test clearing all episodes."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        await episodic_memory.clear()

        count = await episodic_memory.count()
        assert count == 0
        assert len(episodic_memory._episodes) == 0


class TestEpisodicMemoryConvenience:
    """Test EpisodicMemory convenience methods."""

    @pytest.mark.asyncio
    async def test_store_conversation(self, episodic_memory, sample_conversation):
        """Test storing a conversation as an episode."""
        episode = await episodic_memory.store_conversation(
            turns=sample_conversation,
            summary="Authentication implementation discussion",
            project="myapp",
            tags=["auth", "jwt"]
        )

        assert episode.type == EpisodeType.CONVERSATION
        assert episode.project == "myapp"
        assert "auth" in episode.tags

    @pytest.mark.asyncio
    async def test_store_conversation_auto_summary(self, episodic_memory):
        """Test that conversation generates summary if not provided."""
        turns = [
            {"role": "user", "content": "How do I implement caching?"},
            {"role": "assistant", "content": "You can use Redis for caching."}
        ]

        episode = await episodic_memory.store_conversation(turns=turns)

        assert episode.summary == "How do I implement caching?"

    @pytest.mark.asyncio
    async def test_get_recent(self, episodic_memory, sample_episodes):
        """Test getting recent episodes."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        recent = await episodic_memory.get_recent(limit=2)

        assert len(recent) <= 2

    @pytest.mark.asyncio
    async def test_get_by_project(self, episodic_memory, sample_episodes):
        """Test getting episodes by project."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        results = await episodic_memory.get_by_project("test-project")

        # Two episodes have project="test-project"
        assert len(results) == 2
        for ep in results:
            assert ep.project == "test-project"

    @pytest.mark.asyncio
    async def test_get_by_type(self, episodic_memory, sample_episodes):
        """Test getting episodes by type."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        results = await episodic_memory.get_by_type(EpisodeType.DEBUGGING)

        assert len(results) == 1
        assert results[0].type == EpisodeType.DEBUGGING

    @pytest.mark.asyncio
    async def test_search_by_tags(self, episodic_memory, sample_episodes):
        """Test searching episodes by tags."""
        for episode in sample_episodes:
            await episodic_memory.store(episode)

        results = await episodic_memory.search_by_tags(["auth", "tokens"])

        # Episode 2 has "auth" and "tokens" tags
        assert len(results) >= 1
        for ep in results:
            assert any(tag in ep.tags for tag in ["auth", "tokens"])


class TestEpisodicMemoryImportance:
    """Test importance calculation."""

    @pytest.mark.asyncio
    async def test_importance_base(self, episodic_memory):
        """Test base importance score."""
        turns = [
            {"role": "user", "content": "Quick question"},
            {"role": "assistant", "content": "Quick answer"}
        ]

        episode = await episodic_memory.store_conversation(turns=turns)

        assert 0.5 <= episode.importance <= 1.0

    @pytest.mark.asyncio
    async def test_importance_with_code(self, episodic_memory):
        """Test importance increases with code blocks."""
        turns = [
            {"role": "user", "content": "Show me the code"},
            {"role": "assistant", "content": "Here is the code:\n```python\nprint('hello')\n```"}
        ]

        episode = await episodic_memory.store_conversation(turns=turns)

        assert episode.importance >= 0.6  # Should be boosted

    @pytest.mark.asyncio
    async def test_importance_with_decisions(self, episodic_memory):
        """Test importance increases with decision language."""
        turns = [
            {"role": "user", "content": "What should we use?"},
            {"role": "assistant", "content": "I decided to use PostgreSQL because it's better for this use case."}
        ]

        episode = await episodic_memory.store_conversation(turns=turns)

        assert episode.importance >= 0.6  # Should be boosted


class TestEpisodicMemoryPersistence:
    """Test persistence functionality."""

    @pytest.mark.asyncio
    async def test_episode_persists_to_file(self, temp_file, chroma_store_memory, mock_embeddings):
        """Test episodes are persisted to file."""
        memory = EpisodicMemory(
            store=chroma_store_memory,
            embeddings=mock_embeddings,
            data_path=temp_file,
        )

        episode = Episode(
            summary="Test persistence",
            tags=["test"]
        )
        await memory.store(episode)

        # Check file exists and contains data
        from pathlib import Path
        import json

        assert Path(temp_file).exists()
        with open(temp_file) as f:
            data = json.load(f)
        assert episode.id in data

    @pytest.mark.asyncio
    async def test_no_persistence_without_path(self, chroma_store_memory, mock_embeddings, temp_dir):
        """Test no persistence when data_path not set."""
        from pathlib import Path

        memory = EpisodicMemory(
            store=chroma_store_memory,
            embeddings=mock_embeddings,
            data_path=None,
        )

        episode = Episode(summary="Test no persistence")
        await memory.store(episode)

        # Check no JSON file created in temp dir
        json_files = list(Path(temp_dir).glob("*.json"))
        assert len(json_files) == 0
