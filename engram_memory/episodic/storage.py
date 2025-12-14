"""
Vector storage backends for Episodic Memory.

This module provides storage implementations for vector similarity search.
The default is ChromaDB (embedded, no server needed), with an abstraction
layer to support other backends like Qdrant in the future.
"""

from typing import Optional, Protocol
import json


class VectorStore(Protocol):
    """
    Protocol for vector storage backends.

    Implement this to add support for other vector databases
    like Qdrant, Pinecone, Weaviate, etc.
    """

    async def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict,
        document: str = "",
    ) -> None:
        """Add a vector with metadata."""
        ...

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        """
        Search for similar vectors.

        Returns list of (id, similarity_score, metadata) tuples,
        sorted by similarity (highest first).
        """
        ...

    async def get(self, id: str) -> Optional[tuple[list[float], dict]]:
        """Get a vector and its metadata by ID."""
        ...

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        ...

    async def clear(self) -> None:
        """Clear all vectors."""
        ...

    async def count(self) -> int:
        """Return the number of vectors stored."""
        ...


class ChromaDBStore:
    """
    ChromaDB vector store implementation.

    ChromaDB is embedded (no server needed) and perfect for:
    - Local development
    - Small to medium datasets (< 1M vectors)
    - Quick prototyping

    Usage:
        store = ChromaDBStore(
            collection_name="episodic_memory",
            persist_directory="./engram_data"
        )
    """

    def __init__(
        self,
        collection_name: str = "episodic_memory",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB store.

        Args:
            collection_name: Name of the collection to use
            persist_directory: Path to persist data (None for in-memory)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB is required for EpisodicMemory. "
                "Install it with: pip install chromadb"
            )

        # Configure client
        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        self._collection_name = collection_name

    async def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict,
        document: str = "",
    ) -> None:
        """
        Add a vector with metadata.

        Args:
            id: Unique identifier for this vector
            embedding: The vector embedding
            metadata: Associated metadata (must be JSON-serializable)
            document: Optional text document for full-text search
        """
        # ChromaDB doesn't like None values or complex types in metadata
        clean_metadata = self._clean_metadata(metadata)

        self._collection.upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[clean_metadata],
            documents=[document] if document else None,
        )

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        """
        Search for similar vectors.

        Args:
            embedding: Query vector
            limit: Maximum results to return
            filters: Optional metadata filters (ChromaDB where clause)

        Returns:
            List of (id, similarity_score, metadata) tuples
        """
        # Build query
        query_params = {
            "query_embeddings": [embedding],
            "n_results": limit,
            "include": ["metadatas", "distances", "documents"],
        }

        if filters:
            query_params["where"] = filters

        results = self._collection.query(**query_params)

        # Convert to standard format
        output = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}

                # Restore complex types from JSON strings
                metadata = self._restore_metadata(metadata)

                output.append((doc_id, similarity, metadata))

        return output

    async def get(self, id: str) -> Optional[tuple[list[float], dict]]:
        """Get a vector and its metadata by ID."""
        results = self._collection.get(
            ids=[id],
            include=["embeddings", "metadatas"],
        )

        if results["ids"]:
            # Handle numpy arrays from ChromaDB
            embeddings = results.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                embedding = list(embeddings[0])
            else:
                embedding = []

            metadatas = results.get("metadatas")
            if metadatas is not None and len(metadatas) > 0:
                metadata = metadatas[0] or {}
            else:
                metadata = {}

            metadata = self._restore_metadata(metadata)
            return (embedding, metadata)

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        try:
            self._collection.delete(ids=[id])
            return True
        except Exception:
            return False

    async def clear(self) -> None:
        """Clear all vectors by recreating the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def count(self) -> int:
        """Return the number of vectors stored."""
        return self._collection.count()

    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Clean metadata for ChromaDB storage.

        ChromaDB only supports: str, int, float, bool
        Complex types are JSON-serialized.
        ChromaDB requires at least one metadata attribute.
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue  # Skip None values
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Serialize lists to JSON
                clean[key] = json.dumps(value)
                clean[f"__{key}_type"] = "list"
            elif isinstance(value, dict):
                # Serialize dicts to JSON
                clean[key] = json.dumps(value)
                clean[f"__{key}_type"] = "dict"
            else:
                # Convert other types to string
                clean[key] = str(value)
                clean[f"__{key}_type"] = "str"

        # ChromaDB requires at least one metadata attribute
        if not clean:
            clean["_placeholder"] = True

        return clean

    def _restore_metadata(self, metadata: dict) -> dict:
        """Restore complex types from JSON strings."""
        restored = {}
        type_markers = [k for k in metadata.keys() if k.startswith("__") and k.endswith("_type")]

        for key, value in metadata.items():
            # Skip type markers and placeholder
            if key.startswith("__") and key.endswith("_type"):
                continue
            if key == "_placeholder":
                continue

            type_key = f"__{key}_type"
            if type_key in metadata:
                type_hint = metadata[type_key]
                if type_hint in ("list", "dict"):
                    try:
                        restored[key] = json.loads(value)
                    except json.JSONDecodeError:
                        restored[key] = value
                else:
                    restored[key] = value
            else:
                restored[key] = value

        return restored


class QdrantStore:
    """
    Qdrant vector store implementation.

    Qdrant is production-ready with advanced filtering.
    Use this when you need:
    - Scale (millions of vectors)
    - Complex metadata filtering
    - Cloud deployment

    Usage:
        # Local
        store = QdrantStore(url="localhost", port=6333)

        # Cloud (free tier available)
        store = QdrantStore(
            url="https://xxx.cloud.qdrant.io",
            api_key="your-api-key"
        )
    """

    def __init__(
        self,
        collection_name: str = "episodic_memory",
        url: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        embedding_dim: int = 1536,  # OpenAI ada-002 default
    ):
        """
        Initialize Qdrant store.

        Args:
            collection_name: Name of the collection
            url: Qdrant server URL
            port: Qdrant server port (ignored for cloud URLs)
            api_key: API key for Qdrant Cloud
            embedding_dim: Dimension of embeddings
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "Qdrant client is required. "
                "Install it with: pip install qdrant-client"
            )

        # Connect to Qdrant
        if api_key:
            # Cloud connection
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            # Local connection
            self._client = QdrantClient(host=url, port=port)

        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        # Create collection if not exists
        collections = self._client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    async def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict,
        document: str = "",
    ) -> None:
        """Add a vector with metadata."""
        from qdrant_client.models import PointStruct

        # Add document to metadata if provided
        if document:
            metadata["_document"] = document

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=embedding,
                    payload=metadata,
                )
            ],
        )

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=embedding,
            limit=limit,
            query_filter=query_filter,
        )

        return [(str(r.id), r.score, r.payload or {}) for r in results]

    async def get(self, id: str) -> Optional[tuple[list[float], dict]]:
        """Get a vector and its metadata by ID."""
        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[id],
            with_vectors=True,
        )

        if results:
            point = results[0]
            return (point.vector, point.payload or {})

        return None

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        from qdrant_client.models import PointIdsList

        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=[id]),
            )
            return True
        except Exception:
            return False

    async def clear(self) -> None:
        """Clear all vectors."""
        from qdrant_client.models import Distance, VectorParams

        self._client.delete_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    async def count(self) -> int:
        """Return the number of vectors stored."""
        info = self._client.get_collection(self._collection_name)
        return info.points_count
