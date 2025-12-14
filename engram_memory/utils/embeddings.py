"""
Embedding providers for Engram.

This module provides different embedding implementations:
- OpenAIEmbeddings: Uses OpenAI's API (requires API key)
- LocalEmbeddings: Uses sentence-transformers (runs locally, free)

Usage:
    # OpenAI (best quality, requires API key)
    embeddings = OpenAIEmbeddings(api_key="sk-...")

    # Local (free, runs on your machine)
    embeddings = LocalEmbeddings()

    # Auto-detect best available
    embeddings = get_default_embeddings()
"""

from typing import Protocol, Optional
import os


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class OpenAIEmbeddings:
    """
    OpenAI embeddings using the API.

    Requires an OpenAI API key (set OPENAI_API_KEY env var or pass directly).

    Models:
    - text-embedding-3-small: 1536 dims, cheapest, good quality
    - text-embedding-3-large: 3072 dims, best quality
    - text-embedding-ada-002: 1536 dims, legacy
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI embeddings.

        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI client is required. "
                "Install it with: pip install openai"
            )

        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self._model = model

        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimensions.get(self._model, 1536)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [d.embedding for d in response.data]


class LocalEmbeddings:
    """
    Local embeddings using sentence-transformers.

    Runs entirely on your machine - no API key needed!
    First run will download the model (~100MB).

    Recommended models:
    - all-MiniLM-L6-v2: 384 dims, fast, good quality (default)
    - all-mpnet-base-v2: 768 dims, slower, better quality
    - multi-qa-MiniLM-L6-cos-v1: 384 dims, optimized for Q&A
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embeddings.

        Args:
            model_name: sentence-transformers model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        import asyncio

        # sentence-transformers is sync, run in executor
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self._model.encode(text, convert_to_numpy=True).tolist()
        )
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        import asyncio

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings


class ChromaEmbeddings:
    """
    Use ChromaDB's built-in embedding function.

    This is the simplest option - ChromaDB handles embeddings automatically.
    Uses 'all-MiniLM-L6-v2' by default (same as LocalEmbeddings).

    Note: Only works with ChromaDB storage backend.
    """

    def __init__(self):
        """Initialize ChromaDB's default embedding function."""
        try:
            from chromadb.utils import embedding_functions
        except ImportError:
            raise ImportError(
                "ChromaDB is required. "
                "Install it with: pip install chromadb"
            )

        self._ef = embedding_functions.DefaultEmbeddingFunction()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (MiniLM = 384)."""
        return 384

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        result = self._ef([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        return self._ef(texts)


def get_default_embeddings() -> EmbeddingProvider:
    """
    Get the best available embedding provider.

    Priority:
    1. OpenAI (if OPENAI_API_KEY is set)
    2. Local sentence-transformers (if installed)
    3. ChromaDB default (fallback)

    Returns:
        An embedding provider instance
    """
    # Try OpenAI first
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddings()
        except ImportError:
            pass

    # Try local embeddings
    try:
        return LocalEmbeddings()
    except ImportError:
        pass

    # Fallback to ChromaDB's built-in
    try:
        return ChromaEmbeddings()
    except ImportError:
        pass

    raise ImportError(
        "No embedding provider available. Install one of:\n"
        "  pip install openai          # For OpenAI embeddings\n"
        "  pip install sentence-transformers  # For local embeddings\n"
        "  pip install chromadb        # For ChromaDB default embeddings"
    )
