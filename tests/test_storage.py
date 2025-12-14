"""
Tests for vector storage backends.
"""

import pytest
import json

from engram_memory.episodic.storage import ChromaDBStore


class TestChromaDBStoreBasics:
    """Test basic ChromaDBStore functionality."""

    def test_init_in_memory(self, chroma_store_memory):
        """Test in-memory initialization."""
        assert chroma_store_memory is not None
        assert chroma_store_memory._collection is not None

    def test_init_persistent(self, chroma_store):
        """Test persistent initialization."""
        assert chroma_store is not None
        assert chroma_store._collection is not None

    @pytest.mark.asyncio
    async def test_count_empty(self, chroma_store_memory):
        """Test count on empty store."""
        count = await chroma_store_memory.count()
        assert count == 0


class TestChromaDBStoreAdd:
    """Test ChromaDBStore add operations."""

    @pytest.mark.asyncio
    async def test_add_vector(self, chroma_store_memory, mock_embeddings):
        """Test adding a vector."""
        embedding = await mock_embeddings.embed("test text")

        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata={"summary": "Test summary"},
            document="test text"
        )

        count = await chroma_store_memory.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_add_multiple_vectors(self, chroma_store_memory, mock_embeddings):
        """Test adding multiple vectors."""
        for i in range(5):
            embedding = await mock_embeddings.embed(f"test text {i}")
            await chroma_store_memory.add(
                id=f"test-{i}",
                embedding=embedding,
                metadata={"index": i},
            )

        count = await chroma_store_memory.count()
        assert count == 5

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, chroma_store_memory, mock_embeddings):
        """Test that adding same ID overwrites."""
        embedding = await mock_embeddings.embed("original")

        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata={"version": 1},
        )

        embedding2 = await mock_embeddings.embed("updated")
        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding2,
            metadata={"version": 2},
        )

        count = await chroma_store_memory.count()
        assert count == 1

        result = await chroma_store_memory.get("test-1")
        assert result[1]["version"] == 2


class TestChromaDBStoreSearch:
    """Test ChromaDBStore search operations."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, chroma_store_memory, mock_embeddings):
        """Test basic search functionality."""
        # Add some vectors
        for i in range(3):
            embedding = await mock_embeddings.embed(f"document {i}")
            await chroma_store_memory.add(
                id=f"doc-{i}",
                embedding=embedding,
                metadata={"index": i},
            )

        # Search
        query_embedding = await mock_embeddings.embed("document 1")
        results = await chroma_store_memory.search(
            embedding=query_embedding,
            limit=5
        )

        assert len(results) == 3
        # Results are tuples of (id, similarity, metadata)
        assert all(len(r) == 3 for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, chroma_store_memory, mock_embeddings):
        """Test search limit parameter."""
        for i in range(10):
            embedding = await mock_embeddings.embed(f"doc {i}")
            await chroma_store_memory.add(
                id=f"doc-{i}",
                embedding=embedding,
                metadata={},
            )

        query_embedding = await mock_embeddings.embed("query")
        results = await chroma_store_memory.search(
            embedding=query_embedding,
            limit=3
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_returns_similarity(self, chroma_store_memory, mock_embeddings):
        """Test that search returns similarity scores."""
        embedding = await mock_embeddings.embed("exact match text")
        await chroma_store_memory.add(
            id="doc-1",
            embedding=embedding,
            metadata={},
        )

        results = await chroma_store_memory.search(
            embedding=embedding,
            limit=1
        )

        assert len(results) == 1
        doc_id, similarity, metadata = results[0]
        # Same embedding should have high similarity (close to 1)
        assert similarity > 0.99

    @pytest.mark.asyncio
    async def test_search_empty_store(self, chroma_store_memory, mock_embeddings):
        """Test searching empty store."""
        query_embedding = await mock_embeddings.embed("anything")
        results = await chroma_store_memory.search(
            embedding=query_embedding,
            limit=5
        )

        assert results == []


class TestChromaDBStoreGet:
    """Test ChromaDBStore get operations."""

    @pytest.mark.asyncio
    async def test_get_existing(self, chroma_store_memory, mock_embeddings):
        """Test getting an existing vector."""
        embedding = await mock_embeddings.embed("test")
        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata={"key": "value"},
        )

        result = await chroma_store_memory.get("test-1")

        assert result is not None
        stored_embedding, metadata = result
        assert metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, chroma_store_memory):
        """Test getting a non-existent vector."""
        result = await chroma_store_memory.get("nonexistent")

        assert result is None


class TestChromaDBStoreDelete:
    """Test ChromaDBStore delete operations."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, chroma_store_memory, mock_embeddings):
        """Test deleting an existing vector."""
        embedding = await mock_embeddings.embed("test")
        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata={},
        )

        result = await chroma_store_memory.delete("test-1")

        assert result is True
        assert await chroma_store_memory.get("test-1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, chroma_store_memory):
        """Test deleting non-existent vector."""
        result = await chroma_store_memory.delete("nonexistent")

        # ChromaDB doesn't error on non-existent deletes
        assert result is True


class TestChromaDBStoreClear:
    """Test ChromaDBStore clear operations."""

    @pytest.mark.asyncio
    async def test_clear(self, chroma_store_memory, mock_embeddings):
        """Test clearing all vectors."""
        for i in range(5):
            embedding = await mock_embeddings.embed(f"doc {i}")
            await chroma_store_memory.add(
                id=f"doc-{i}",
                embedding=embedding,
                metadata={},
            )

        await chroma_store_memory.clear()

        count = await chroma_store_memory.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_empty(self, chroma_store_memory):
        """Test clearing empty store."""
        await chroma_store_memory.clear()

        count = await chroma_store_memory.count()
        assert count == 0


class TestChromaDBStoreMetadata:
    """Test metadata handling in ChromaDBStore."""

    @pytest.mark.asyncio
    async def test_clean_metadata_primitives(self, chroma_store_memory, mock_embeddings):
        """Test that primitive types are stored correctly."""
        embedding = await mock_embeddings.embed("test")
        metadata = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
        }

        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata=metadata,
        )

        result = await chroma_store_memory.get("test-1")
        stored_metadata = result[1]

        assert stored_metadata["string"] == "hello"
        assert stored_metadata["integer"] == 42
        assert stored_metadata["float"] == 3.14
        assert stored_metadata["boolean"] is True

    @pytest.mark.asyncio
    async def test_clean_metadata_list(self, chroma_store_memory, mock_embeddings):
        """Test that lists are JSON-serialized and restored."""
        embedding = await mock_embeddings.embed("test")
        metadata = {
            "tags": ["auth", "jwt", "security"],
        }

        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata=metadata,
        )

        result = await chroma_store_memory.get("test-1")
        stored_metadata = result[1]

        assert stored_metadata["tags"] == ["auth", "jwt", "security"]

    @pytest.mark.asyncio
    async def test_clean_metadata_dict(self, chroma_store_memory, mock_embeddings):
        """Test that dicts are JSON-serialized and restored."""
        embedding = await mock_embeddings.embed("test")
        metadata = {
            "context": {"project": "myapp", "version": 1},
        }

        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata=metadata,
        )

        result = await chroma_store_memory.get("test-1")
        stored_metadata = result[1]

        assert stored_metadata["context"] == {"project": "myapp", "version": 1}

    @pytest.mark.asyncio
    async def test_clean_metadata_none_values(self, chroma_store_memory, mock_embeddings):
        """Test that None values are skipped."""
        embedding = await mock_embeddings.embed("test")
        metadata = {
            "present": "value",
            "absent": None,
        }

        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata=metadata,
        )

        result = await chroma_store_memory.get("test-1")
        stored_metadata = result[1]

        assert "present" in stored_metadata
        assert "absent" not in stored_metadata

    @pytest.mark.asyncio
    async def test_search_restores_metadata(self, chroma_store_memory, mock_embeddings):
        """Test that search results have restored metadata."""
        embedding = await mock_embeddings.embed("test")
        metadata = {
            "tags": ["a", "b", "c"],
            "info": {"key": "value"},
        }

        await chroma_store_memory.add(
            id="test-1",
            embedding=embedding,
            metadata=metadata,
        )

        results = await chroma_store_memory.search(
            embedding=embedding,
            limit=1
        )

        _, _, result_metadata = results[0]
        assert result_metadata["tags"] == ["a", "b", "c"]
        assert result_metadata["info"] == {"key": "value"}


class TestChromaDBStorePersistence:
    """Test persistence functionality."""

    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir, mock_embeddings):
        """Test that data persists across instances."""
        # Create first instance and add data
        store1 = ChromaDBStore(
            collection_name="test_persist",
            persist_directory=temp_dir,
        )

        embedding = await mock_embeddings.embed("persistent data")
        await store1.add(
            id="persist-1",
            embedding=embedding,
            metadata={"value": "stored"},
        )

        # Create second instance pointing to same directory
        store2 = ChromaDBStore(
            collection_name="test_persist",
            persist_directory=temp_dir,
        )

        count = await store2.count()
        assert count == 1

        result = await store2.get("persist-1")
        assert result is not None
        assert result[1]["value"] == "stored"
