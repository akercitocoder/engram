"""
Tests for embedding providers.
"""

import pytest
import os


class TestMockEmbeddings:
    """Test the MockEmbeddings fixture."""

    @pytest.mark.asyncio
    async def test_embed_returns_list(self, mock_embeddings):
        """Test that embed returns a list of floats."""
        result = await mock_embeddings.embed("test text")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_dimension(self, mock_embeddings):
        """Test that embed returns correct dimension."""
        result = await mock_embeddings.embed("test text")

        assert len(result) == mock_embeddings.dimension
        assert mock_embeddings.dimension == 384

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, mock_embeddings):
        """Test that same text produces same embedding."""
        result1 = await mock_embeddings.embed("same text")
        result2 = await mock_embeddings.embed("same text")

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_embed_different_texts(self, mock_embeddings):
        """Test that different texts produce different embeddings."""
        result1 = await mock_embeddings.embed("text one")
        result2 = await mock_embeddings.embed("text two")

        assert result1 != result2

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_embeddings):
        """Test batch embedding."""
        texts = ["first", "second", "third"]
        results = await mock_embeddings.embed_batch(texts)

        assert len(results) == 3
        assert all(len(e) == mock_embeddings.dimension for e in results)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, mock_embeddings):
        """Test batch embedding with empty list."""
        results = await mock_embeddings.embed_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_embedding_range(self, mock_embeddings):
        """Test that embeddings are in valid range."""
        result = await mock_embeddings.embed("test text")

        # Mock embeddings normalize to [-1, 1]
        assert all(-1 <= v <= 1 for v in result)


class TestChromaEmbeddings:
    """Test ChromaDB's built-in embeddings."""

    @pytest.fixture
    def chroma_embeddings(self):
        """Create ChromaEmbeddings instance."""
        try:
            from engram_memory.utils.embeddings import ChromaEmbeddings
            return ChromaEmbeddings()
        except ImportError:
            pytest.skip("ChromaDB not installed")

    @pytest.mark.asyncio
    async def test_embed_returns_list(self, chroma_embeddings):
        """Test that embed returns a list or array-like."""
        result = await chroma_embeddings.embed("test text")

        # ChromaDB may return list or numpy array
        assert hasattr(result, "__len__")
        assert len(result) == chroma_embeddings.dimension

    @pytest.mark.asyncio
    async def test_embed_dimension(self, chroma_embeddings):
        """Test embedding dimension."""
        result = await chroma_embeddings.embed("test text")

        assert len(result) == chroma_embeddings.dimension
        assert chroma_embeddings.dimension == 384

    @pytest.mark.asyncio
    async def test_embed_batch(self, chroma_embeddings):
        """Test batch embedding."""
        texts = ["hello", "world"]
        results = await chroma_embeddings.embed_batch(texts)

        assert len(results) == 2
        assert all(len(e) == 384 for e in results)

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, chroma_embeddings):
        """Test batch embedding with empty list."""
        results = await chroma_embeddings.embed_batch([])

        assert results == []


class TestLocalEmbeddings:
    """Test local sentence-transformers embeddings."""

    @pytest.fixture
    def local_embeddings(self):
        """Create LocalEmbeddings instance."""
        try:
            from engram_memory.utils.embeddings import LocalEmbeddings
            return LocalEmbeddings()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.asyncio
    async def test_embed_returns_list(self, local_embeddings):
        """Test that embed returns a list."""
        result = await local_embeddings.embed("test text")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_dimension(self, local_embeddings):
        """Test embedding dimension matches model."""
        result = await local_embeddings.embed("test text")

        assert len(result) == local_embeddings.dimension

    @pytest.mark.asyncio
    async def test_similar_texts_closer(self, local_embeddings):
        """Test that similar texts have closer embeddings."""
        import math

        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        # Get embeddings
        dog_embed = await local_embeddings.embed("dog")
        cat_embed = await local_embeddings.embed("cat")
        car_embed = await local_embeddings.embed("automobile")

        # Dog and cat (animals) should be closer than dog and car
        dog_cat_sim = cosine_similarity(dog_embed, cat_embed)
        dog_car_sim = cosine_similarity(dog_embed, car_embed)

        assert dog_cat_sim > dog_car_sim

    @pytest.mark.asyncio
    async def test_embed_batch(self, local_embeddings):
        """Test batch embedding."""
        texts = ["hello", "world", "test"]
        results = await local_embeddings.embed_batch(texts)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, local_embeddings):
        """Test batch embedding with empty list."""
        results = await local_embeddings.embed_batch([])

        assert results == []


class TestOpenAIEmbeddings:
    """Test OpenAI embeddings (requires API key)."""

    @pytest.fixture
    def openai_embeddings(self):
        """Create OpenAIEmbeddings instance."""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            from engram_memory.utils.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings()
        except ImportError:
            pytest.skip("openai package not installed")

    def test_dimension_small(self):
        """Test dimension for small model."""
        try:
            from engram_memory.utils.embeddings import OpenAIEmbeddings
        except ImportError:
            pytest.skip("openai package not installed")

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        assert embeddings.dimension == 1536

    def test_dimension_large(self):
        """Test dimension for large model."""
        try:
            from engram_memory.utils.embeddings import OpenAIEmbeddings
        except ImportError:
            pytest.skip("openai package not installed")

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        assert embeddings.dimension == 3072

    @pytest.mark.asyncio
    async def test_embed_returns_list(self, openai_embeddings):
        """Test that embed returns a list."""
        result = await openai_embeddings.embed("test text")

        assert isinstance(result, list)
        assert len(result) == openai_embeddings.dimension

    @pytest.mark.asyncio
    async def test_embed_batch(self, openai_embeddings):
        """Test batch embedding."""
        texts = ["hello", "world"]
        results = await openai_embeddings.embed_batch(texts)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self, openai_embeddings):
        """Test batch embedding with empty list."""
        results = await openai_embeddings.embed_batch([])

        assert results == []


class TestGetDefaultEmbeddings:
    """Test get_default_embeddings helper."""

    def test_returns_provider(self):
        """Test that function returns a provider."""
        from engram_memory.utils.embeddings import get_default_embeddings

        provider = get_default_embeddings()

        # Should have the required interface
        assert hasattr(provider, "embed")
        assert hasattr(provider, "embed_batch")
        assert hasattr(provider, "dimension")

    def test_prefers_openai_if_key_set(self):
        """Test that OpenAI is preferred when key is available."""
        original_key = os.environ.get("OPENAI_API_KEY")

        try:
            from engram_memory.utils.embeddings import get_default_embeddings

            # Skip if openai not installed
            try:
                import openai
            except ImportError:
                pytest.skip("openai not installed")

            os.environ["OPENAI_API_KEY"] = "sk-test-key"
            provider = get_default_embeddings()

            # Should be OpenAI provider
            assert "OpenAI" in type(provider).__name__

        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_falls_back_without_openai_key(self):
        """Test fallback when no OpenAI key is set."""
        original_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            from engram_memory.utils.embeddings import get_default_embeddings

            provider = get_default_embeddings()

            # Should be either Local or Chroma embeddings
            assert "OpenAI" not in type(provider).__name__

        finally:
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
