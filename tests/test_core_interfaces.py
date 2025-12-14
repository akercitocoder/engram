"""
Tests for core interfaces and data models.
"""

import pytest
from datetime import datetime, timedelta

from engram_memory.core.interfaces import (
    MemoryType,
    MemoryEntry,
    ConversationTurn,
    RetrievalQuery,
    RetrievalResult,
    MemoryContext,
)


# ============================================================
# MemoryType Tests
# ============================================================

class TestMemoryType:
    """Test MemoryType enum."""

    def test_all_types_exist(self):
        """Test all memory types are defined."""
        assert MemoryType.WORKING.value == "working"
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_memory_type_count(self):
        """Test we have exactly 4 memory types."""
        assert len(MemoryType) == 4


# ============================================================
# MemoryEntry Tests
# ============================================================

class TestMemoryEntry:
    """Test MemoryEntry base dataclass."""

    def test_default_id_generation(self):
        """Test that ID is auto-generated if not provided."""
        entry = MemoryEntry()

        assert entry.id is not None
        assert len(entry.id) > 0

    def test_custom_id(self):
        """Test that custom ID is preserved."""
        entry = MemoryEntry(id="custom-id")

        assert entry.id == "custom-id"

    def test_default_timestamps(self):
        """Test that timestamps are auto-set."""
        before = datetime.utcnow()
        entry = MemoryEntry()
        after = datetime.utcnow()

        assert before <= entry.created_at <= after
        assert before <= entry.updated_at <= after

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        entry = MemoryEntry()

        assert entry.metadata == {}

    def test_custom_metadata(self):
        """Test custom metadata is preserved."""
        entry = MemoryEntry(metadata={"key": "value"})

        assert entry.metadata == {"key": "value"}


# ============================================================
# ConversationTurn Tests
# ============================================================

class TestConversationTurn:
    """Test ConversationTurn dataclass."""

    def test_creation(self):
        """Test basic creation."""
        turn = ConversationTurn(role="user", content="Hello!")

        assert turn.role == "user"
        assert turn.content == "Hello!"

    def test_timestamp_auto_set(self):
        """Test timestamp is auto-set."""
        before = datetime.utcnow()
        turn = ConversationTurn(role="assistant", content="Hi there!")
        after = datetime.utcnow()

        assert before <= turn.timestamp <= after

    def test_to_dict(self):
        """Test serialization to dict."""
        turn = ConversationTurn(
            role="user",
            content="Test message",
            metadata={"intent": "question"}
        )

        data = turn.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert "timestamp" in data
        assert data["metadata"] == {"intent": "question"}

    def test_from_dict(self):
        """Test deserialization from dict."""
        original = ConversationTurn(
            role="assistant",
            content="Response",
            metadata={"tokens": 50}
        )
        data = original.to_dict()

        restored = ConversationTurn.from_dict(data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata

    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = ConversationTurn(
            role="user",
            content="Complex message with special chars: éàü",
            metadata={"nested": {"key": "value"}}
        )

        restored = ConversationTurn.from_dict(original.to_dict())

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.metadata == original.metadata


# ============================================================
# RetrievalQuery Tests
# ============================================================

class TestRetrievalQuery:
    """Test RetrievalQuery dataclass."""

    def test_basic_query(self):
        """Test basic query creation."""
        query = RetrievalQuery(text="authentication")

        assert query.text == "authentication"

    def test_default_values(self):
        """Test default values."""
        query = RetrievalQuery(text="test")

        assert query.limit == 10
        assert query.min_relevance == 0.0
        assert query.time_range is None
        assert query.metadata_filters == {}
        assert len(query.memory_types) == 4  # All types by default

    def test_custom_limit(self):
        """Test custom limit."""
        query = RetrievalQuery(text="test", limit=5)

        assert query.limit == 5

    def test_time_range(self):
        """Test time range filter."""
        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow()

        query = RetrievalQuery(
            text="test",
            time_range=(start, end)
        )

        assert query.time_range == (start, end)

    def test_metadata_filters(self):
        """Test metadata filters."""
        query = RetrievalQuery(
            text="test",
            metadata_filters={"project": "myapp", "tag": "auth"}
        )

        assert query.metadata_filters["project"] == "myapp"
        assert query.metadata_filters["tag"] == "auth"

    def test_specific_memory_types(self):
        """Test filtering to specific memory types."""
        query = RetrievalQuery(
            text="test",
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC]
        )

        assert len(query.memory_types) == 2
        assert MemoryType.EPISODIC in query.memory_types
        assert MemoryType.SEMANTIC in query.memory_types


# ============================================================
# RetrievalResult Tests
# ============================================================

class TestRetrievalResult:
    """Test RetrievalResult dataclass."""

    def test_creation(self):
        """Test basic creation."""
        entry = MemoryEntry()
        result = RetrievalResult(
            entry=entry,
            relevance_score=0.85,
            memory_type=MemoryType.EPISODIC
        )

        assert result.entry == entry
        assert result.relevance_score == 0.85
        assert result.memory_type == MemoryType.EPISODIC

    def test_reasoning_optional(self):
        """Test reasoning is optional."""
        entry = MemoryEntry()
        result = RetrievalResult(
            entry=entry,
            relevance_score=0.9,
            memory_type=MemoryType.SEMANTIC
        )

        assert result.reasoning is None

    def test_reasoning_provided(self):
        """Test reasoning can be provided."""
        entry = MemoryEntry()
        result = RetrievalResult(
            entry=entry,
            relevance_score=0.9,
            memory_type=MemoryType.SEMANTIC,
            reasoning="High similarity to query terms"
        )

        assert result.reasoning == "High similarity to query terms"


# ============================================================
# MemoryContext Tests
# ============================================================

class TestMemoryContext:
    """Test MemoryContext dataclass."""

    def test_creation(self):
        """Test basic creation."""
        context = MemoryContext(
            working={"task": "coding"},
            episodic=[],
            semantic=[],
            procedural={}
        )

        assert context.working == {"task": "coding"}
        assert context.episodic == []
        assert context.semantic == []
        assert context.procedural == {}

    def test_is_empty_true(self):
        """Test is_empty returns True when no memories."""
        context = MemoryContext(
            working={},
            episodic=[],
            semantic=[],
            procedural={}
        )

        assert context.is_empty() is True

    def test_is_empty_false_with_episodic(self):
        """Test is_empty returns False with episodic memories."""
        entry = MemoryEntry()
        result = RetrievalResult(
            entry=entry,
            relevance_score=0.8,
            memory_type=MemoryType.EPISODIC
        )

        context = MemoryContext(
            working={},
            episodic=[result],
            semantic=[],
            procedural={}
        )

        assert context.is_empty() is False

    def test_is_empty_false_with_semantic(self):
        """Test is_empty returns False with semantic memories."""
        entry = MemoryEntry()
        result = RetrievalResult(
            entry=entry,
            relevance_score=0.7,
            memory_type=MemoryType.SEMANTIC
        )

        context = MemoryContext(
            working={},
            episodic=[],
            semantic=[result],
            procedural={}
        )

        assert context.is_empty() is False

    def test_is_empty_false_with_procedural(self):
        """Test is_empty returns False with procedural memories."""
        context = MemoryContext(
            working={},
            episodic=[],
            semantic=[],
            procedural={"preferences": {"language": "Python"}}
        )

        assert context.is_empty() is False

    def test_to_prompt_section_empty(self):
        """Test to_prompt_section with empty context."""
        context = MemoryContext(
            working={},
            episodic=[],
            semantic=[],
            procedural={}
        )

        prompt = context.to_prompt_section()

        assert prompt == ""

    def test_to_prompt_section_with_episodic(self):
        """Test to_prompt_section includes episodic memories."""
        # Create a mock entry with summary attribute
        class MockEpisode:
            summary = "Implemented JWT authentication"

        result = RetrievalResult(
            entry=MockEpisode(),
            relevance_score=0.85,
            memory_type=MemoryType.EPISODIC
        )

        context = MemoryContext(
            working={},
            episodic=[result],
            semantic=[],
            procedural={}
        )

        prompt = context.to_prompt_section()

        assert "Relevant Past Experiences" in prompt
        assert "85%" in prompt
        assert "JWT authentication" in prompt

    def test_to_prompt_section_with_semantic(self):
        """Test to_prompt_section includes semantic memories."""
        # Create a mock entity with name and description
        class MockEntity:
            name = "PostgreSQL"
            description = "Relational database"

        result = RetrievalResult(
            entry=MockEntity(),
            relevance_score=0.9,
            memory_type=MemoryType.SEMANTIC
        )

        context = MemoryContext(
            working={},
            episodic=[],
            semantic=[result],
            procedural={}
        )

        prompt = context.to_prompt_section()

        assert "Relevant Knowledge" in prompt
        assert "PostgreSQL" in prompt
        assert "Relational database" in prompt

    def test_to_prompt_section_with_procedural(self):
        """Test to_prompt_section includes procedural memories."""
        context = MemoryContext(
            working={},
            episodic=[],
            semantic=[],
            procedural={
                "coding": {
                    "language": "Python",
                    "style": "functional"
                }
            }
        )

        prompt = context.to_prompt_section()

        assert "User Preferences" in prompt
        assert "language" in prompt
        assert "Python" in prompt

    def test_to_prompt_section_with_corrections(self):
        """Test to_prompt_section includes corrections."""
        context = MemoryContext(
            working={},
            episodic=[],
            semantic=[],
            procedural={
                "_corrections": [
                    {"wrong": "use var", "correct": "use const or let"}
                ]
            }
        )

        prompt = context.to_prompt_section()

        assert "Learned Corrections" in prompt
        assert "use var" in prompt
        assert "const or let" in prompt

    def test_to_prompt_section_full(self):
        """Test to_prompt_section with all memory types."""
        class MockEpisode:
            summary = "Fixed authentication bug"

        class MockEntity:
            name = "Redis"
            description = "Cache database"

        context = MemoryContext(
            working={"task": "coding"},
            episodic=[RetrievalResult(
                entry=MockEpisode(),
                relevance_score=0.8,
                memory_type=MemoryType.EPISODIC
            )],
            semantic=[RetrievalResult(
                entry=MockEntity(),
                relevance_score=0.9,
                memory_type=MemoryType.SEMANTIC
            )],
            procedural={"prefs": {"language": "TypeScript"}}
        )

        prompt = context.to_prompt_section()

        assert "Relevant Past Experiences" in prompt
        assert "Relevant Knowledge" in prompt
        assert "User Preferences" in prompt
