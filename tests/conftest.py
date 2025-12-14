"""
Pytest configuration and shared fixtures for Engram tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator
import asyncio


# ============================================================
# Pytest Configuration
# ============================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================
# Temporary Directory Fixtures
# ============================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    dirpath = tempfile.mkdtemp(prefix="engram_test_")
    yield dirpath
    # Cleanup after test
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file path."""
    return str(Path(temp_dir) / "test_data.json")


# ============================================================
# Working Memory Fixtures
# ============================================================

@pytest.fixture
def working_memory():
    """Create a fresh WorkingMemory instance (no persistence)."""
    from engram_memory import WorkingMemory
    return WorkingMemory(max_conversation_turns=10)


@pytest.fixture
def working_memory_persistent(temp_file):
    """Create a WorkingMemory instance with persistence."""
    from engram_memory import WorkingMemory
    return WorkingMemory(
        max_conversation_turns=10,
        persist_path=temp_file,
    )


# ============================================================
# Mock Embedding Provider
# ============================================================

class MockEmbeddings:
    """
    Mock embedding provider for testing.

    Generates deterministic fake embeddings based on text hash.
    This allows tests to run without external dependencies.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic fake embedding from text."""
        import hashlib

        # Create deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Expand hash to fill embedding dimension
        embedding = []
        for i in range(self._dimension):
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1] range
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]


@pytest.fixture
def mock_embeddings():
    """Create a mock embedding provider."""
    return MockEmbeddings(dimension=384)


# ============================================================
# Storage Fixtures
# ============================================================

@pytest.fixture
def chroma_store(temp_dir):
    """Create a ChromaDB store for testing."""
    import uuid
    from engram_memory.episodic.storage import ChromaDBStore

    # Use unique collection name to avoid test isolation issues
    collection_name = f"test_persist_{uuid.uuid4().hex[:8]}"
    return ChromaDBStore(
        collection_name=collection_name,
        persist_directory=temp_dir,
    )


@pytest.fixture
def chroma_store_memory():
    """Create an in-memory ChromaDB store for testing."""
    import uuid
    from engram_memory.episodic.storage import ChromaDBStore

    # Use unique collection name to avoid test isolation issues
    collection_name = f"test_memory_{uuid.uuid4().hex[:8]}"
    return ChromaDBStore(
        collection_name=collection_name,
        persist_directory=None,  # In-memory
    )


# ============================================================
# Episodic Memory Fixtures
# ============================================================

@pytest.fixture
def episodic_memory(chroma_store_memory, mock_embeddings, temp_file):
    """Create an EpisodicMemory instance for testing."""
    from engram_memory import EpisodicMemory
    return EpisodicMemory(
        store=chroma_store_memory,
        embeddings=mock_embeddings,
        data_path=temp_file,
    )


@pytest.fixture
def episodic_memory_no_persist(chroma_store_memory, mock_embeddings):
    """Create an EpisodicMemory instance without persistence."""
    from engram_memory import EpisodicMemory
    return EpisodicMemory(
        store=chroma_store_memory,
        embeddings=mock_embeddings,
        data_path=None,
    )


# ============================================================
# Sample Data Fixtures
# ============================================================

@pytest.fixture
def sample_conversation():
    """Sample conversation turns for testing."""
    return [
        {"role": "user", "content": "Help me implement authentication"},
        {"role": "assistant", "content": "I'll help you with JWT authentication using httpOnly cookies."},
        {"role": "user", "content": "Sounds good, let's do it"},
        {"role": "assistant", "content": "First, let's install the required packages..."},
    ]


@pytest.fixture
def sample_episode():
    """Create a sample episode for testing."""
    from engram_memory import Episode, EpisodeType
    return Episode(
        type=EpisodeType.CONVERSATION,
        summary="Implemented JWT authentication with httpOnly cookies",
        content="Full conversation content here...",
        project="test-project",
        files=["auth.py", "middleware.py"],
        tags=["auth", "jwt", "security"],
        key_points=[
            "Used httpOnly cookies for security",
            "Added refresh token rotation",
        ],
        importance=0.8,
    )


@pytest.fixture
def sample_episodes():
    """Create multiple sample episodes for testing."""
    from engram_memory import Episode, EpisodeType
    return [
        Episode(
            type=EpisodeType.CONVERSATION,
            summary="Set up PostgreSQL database",
            project="test-project",
            tags=["database", "postgresql"],
            key_points=["Used SQLAlchemy ORM"],
        ),
        Episode(
            type=EpisodeType.DEBUGGING,
            summary="Fixed token expiration bug",
            project="test-project",
            tags=["auth", "bug", "tokens"],
            key_points=["Added clock skew buffer"],
            outcome="resolved",
        ),
        Episode(
            type=EpisodeType.RESEARCH,
            summary="Researched caching strategies",
            project="other-project",
            tags=["caching", "redis"],
            key_points=["Chose Redis over Memcached"],
        ),
    ]
