"""
Semantic Memory implementation for Engram.

Semantic memory stores facts, entities, and relationships as a knowledge graph.
Think of it as your agent's "world knowledge" - facts it knows to be true.

Examples of what semantic memory stores:
- "User prefers TypeScript over JavaScript"
- "Project myapp uses PostgreSQL"
- "The authenticate function is in auth.py"
- "FastAPI is a Python web framework"
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
import uuid
import json
from pathlib import Path

from engram_memory.core.interfaces import (
    BaseMemory,
    MemoryType,
    MemoryEntry,
    RetrievalQuery,
    RetrievalResult,
)


class EntityType(Enum):
    """Types of entities in the knowledge graph."""

    # Technical entities
    PROJECT = "project"
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    VARIABLE = "variable"
    PACKAGE = "package"
    API = "api"
    DATABASE = "database"
    SERVICE = "service"

    # Conceptual entities
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    PATTERN = "pattern"
    PREFERENCE = "preference"

    # People/agents
    USER = "user"
    AGENT = "agent"

    # Generic
    OTHER = "other"


@dataclass
class Entity(MemoryEntry):
    """
    An entity in the knowledge graph - a "thing" we know about.

    Entities are the nodes in our graph. They represent concrete
    or abstract things that can have relationships.
    """

    name: str = ""
    entity_type: EntityType = EntityType.OTHER
    description: str = ""
    properties: dict = field(default_factory=dict)

    # Confidence and tracking
    confidence: float = 1.0  # How confident are we this entity exists
    source: str = ""  # Where did we learn about this entity
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def __post_init__(self):
        # Generate ID from type and name for deduplication BEFORE calling super
        if not self.id:
            self.id = f"{self.entity_type.value}:{self.name}".lower().replace(" ", "_")
        # Now call parent to set timestamps
        if not self.created_at:
            self.created_at = datetime.utcnow()
        if not self.updated_at:
            self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Relationship:
    """
    A relationship between two entities.

    Relationships are the edges in our graph. They describe
    how entities relate to each other.
    """

    id: str = ""
    source_id: str = ""  # Entity ID
    target_id: str = ""  # Entity ID
    relation_type: str = ""  # e.g., "uses", "prefers", "contains", "depends_on"
    properties: dict = field(default_factory=dict)

    # Confidence and tracking
    confidence: float = 1.0
    source: str = ""  # Where did we learn about this relationship
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.source_id}-{self.relation_type}-{self.target_id}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            properties=data.get("properties", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class Fact:
    """
    A fact is a triple: (subject, predicate, object).

    This is a convenience wrapper for creating entities and relationships
    from natural language-like statements.

    Examples:
        Fact("user", "prefers", "TypeScript")
        Fact("myapp", "uses", "PostgreSQL")
        Fact("auth.py", "contains", "authenticate")
    """

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: str = ""
    properties: dict = field(default_factory=dict)


class SemanticMemory(BaseMemory[Entity]):
    """
    Semantic memory implementation using a knowledge graph.

    Stores facts, entities, and relationships using NetworkX.
    Supports graph traversal, pattern matching, and inference.

    Usage:
        from engram_memory import SemanticMemory

        memory = SemanticMemory(persist_path="./engram_data/knowledge.json")

        # Add facts
        await memory.add_fact("user", "prefers", "TypeScript")
        await memory.add_fact("myapp", "uses", "PostgreSQL")
        await memory.add_fact("auth.py", "contains", "authenticate")

        # Query
        results = await memory.get_related("user", "prefers")
        # Returns: ["TypeScript"]

        techs = await memory.query("What does myapp use?")
        # Returns relevant entities and relationships
    """

    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize semantic memory.

        Args:
            persist_path: Optional path to persist the knowledge graph
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for SemanticMemory. "
                "Install it with: pip install networkx"
            )

        self._graph = nx.DiGraph()
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path:
            self._load()

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.SEMANTIC

    # ========== Core Memory Interface ==========

    async def store(self, entry: Entity) -> str:
        """Store an entity in the knowledge graph."""
        self._entities[entry.id] = entry
        self._graph.add_node(
            entry.id,
            name=entry.name,
            entity_type=entry.entity_type.value,
            description=entry.description,
            **entry.properties,
        )
        self._save()
        return entry.id

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """
        Retrieve entities matching the query.

        For semantic memory, this does keyword matching on entity
        names and descriptions.
        """
        results = []
        query_lower = query.text.lower()

        for entity in self._entities.values():
            score = 0.0

            # Check name match
            if query_lower in entity.name.lower():
                score = 0.9
            # Check description match
            elif entity.description and query_lower in entity.description.lower():
                score = 0.7
            # Check properties
            else:
                for key, value in entity.properties.items():
                    if query_lower in str(value).lower():
                        score = 0.5
                        break

            if score >= query.min_relevance:
                entity.access_count += 1
                entity.last_accessed = datetime.utcnow()

                results.append(
                    RetrievalResult(
                        entry=entity,
                        relevance_score=score,
                        memory_type=MemoryType.SEMANTIC,
                    )
                )

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        self._save()

        return results[: query.limit]

    async def update(self, entry_id: str, updates: dict) -> bool:
        """Update an entity."""
        if entry_id not in self._entities:
            return False

        entity = self._entities[entry_id]
        for key, value in updates.items():
            if hasattr(entity, key):
                setattr(entity, key, value)

        entity.updated_at = datetime.utcnow()

        # Update graph node
        if entry_id in self._graph:
            for key, value in updates.items():
                if key in ["name", "description", "entity_type"]:
                    self._graph.nodes[entry_id][key] = value

        self._save()
        return True

    async def delete(self, entry_id: str) -> bool:
        """Delete an entity and its relationships."""
        if entry_id in self._entities:
            del self._entities[entry_id]

        if entry_id in self._graph:
            # Remove all edges connected to this node
            edges_to_remove = list(self._graph.in_edges(entry_id)) + list(
                self._graph.out_edges(entry_id)
            )
            self._graph.remove_edges_from(edges_to_remove)
            self._graph.remove_node(entry_id)

            # Remove relationships
            rels_to_remove = [
                r_id
                for r_id, r in self._relationships.items()
                if r.source_id == entry_id or r.target_id == entry_id
            ]
            for r_id in rels_to_remove:
                del self._relationships[r_id]

        self._save()
        return True

    async def clear(self) -> None:
        """Clear all semantic memory."""
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()

        if self._persist_path and self._persist_path.exists():
            self._persist_path.unlink()

    # ========== Semantic Memory Specific Methods ==========

    async def add_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.OTHER,
        description: str = "",
        properties: Optional[dict] = None,
        confidence: float = 1.0,
        source: str = "",
    ) -> Entity:
        """
        Add an entity to the knowledge graph.

        Args:
            name: Entity name
            entity_type: Type of entity
            description: Description of the entity
            properties: Additional properties
            confidence: Confidence score (0-1)
            source: Where we learned about this

        Returns:
            The created entity
        """
        entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
            confidence=confidence,
            source=source,
        )

        # Check if entity already exists
        if entity.id in self._entities:
            # Update existing entity
            existing = self._entities[entity.id]
            existing.confidence = max(existing.confidence, confidence)
            if description:
                existing.description = description
            if properties:
                existing.properties.update(properties)
            existing.updated_at = datetime.utcnow()
            self._save()
            return existing

        await self.store(entity)
        return entity

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[dict] = None,
        confidence: float = 1.0,
        source: str = "",
    ) -> Relationship:
        """
        Add a relationship between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship (e.g., "uses", "prefers")
            properties: Additional properties
            confidence: Confidence score
            source: Where we learned about this

        Returns:
            The created relationship
        """
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence,
            source=source,
        )

        self._relationships[relationship.id] = relationship
        self._graph.add_edge(
            source_id,
            target_id,
            relation_type=relation_type,
            confidence=confidence,
            **relationship.properties,
        )

        self._save()
        return relationship

    async def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        subject_type: EntityType = EntityType.OTHER,
        object_type: EntityType = EntityType.OTHER,
        confidence: float = 1.0,
        source: str = "",
    ) -> tuple[Entity, Relationship, Entity]:
        """
        Add a fact as a triple (subject, predicate, object).

        This is a convenience method that creates entities and
        relationships from a simple triple.

        Args:
            subject: Subject of the fact (e.g., "user", "myapp")
            predicate: Relationship type (e.g., "prefers", "uses")
            obj: Object of the fact (e.g., "TypeScript", "PostgreSQL")
            subject_type: Type of subject entity
            object_type: Type of object entity
            confidence: Confidence score
            source: Where we learned this fact

        Returns:
            Tuple of (subject_entity, relationship, object_entity)

        Example:
            await memory.add_fact("user", "prefers", "TypeScript",
                                  subject_type=EntityType.USER,
                                  object_type=EntityType.TECHNOLOGY)
        """
        # Create or get subject entity
        subject_entity = await self.add_entity(
            name=subject,
            entity_type=subject_type,
            confidence=confidence,
            source=source,
        )

        # Create or get object entity
        object_entity = await self.add_entity(
            name=obj,
            entity_type=object_type,
            confidence=confidence,
            source=source,
        )

        # Create relationship
        relationship = await self.add_relationship(
            source_id=subject_entity.id,
            target_id=object_entity.id,
            relation_type=predicate,
            confidence=confidence,
            source=source,
        )

        return subject_entity, relationship, object_entity

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    async def get_entity_by_name(
        self, name: str, entity_type: Optional[EntityType] = None
    ) -> Optional[Entity]:
        """Get an entity by name (and optionally type)."""
        for entity in self._entities.values():
            if entity.name.lower() == name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    return entity
        return None

    async def get_related(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> list[Entity]:
        """
        Get entities related to a given entity.

        Args:
            entity_id: The entity to find relations for
            relation_type: Filter by relationship type (optional)
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of related entities
        """
        if entity_id not in self._graph:
            return []

        related_ids = set()

        if direction in ("outgoing", "both"):
            for _, target, data in self._graph.out_edges(entity_id, data=True):
                if relation_type is None or data.get("relation_type") == relation_type:
                    related_ids.add(target)

        if direction in ("incoming", "both"):
            for source, _, data in self._graph.in_edges(entity_id, data=True):
                if relation_type is None or data.get("relation_type") == relation_type:
                    related_ids.add(source)

        return [self._entities[eid] for eid in related_ids if eid in self._entities]

    async def get_relationships(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
    ) -> list[Relationship]:
        """Get all relationships for an entity."""
        return [
            r
            for r in self._relationships.values()
            if (r.source_id == entity_id or r.target_id == entity_id)
            and (relation_type is None or r.relation_type == relation_type)
        ]

    async def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> Optional[list[str]]:
        """
        Find a path between two entities.

        Args:
            source_id: Starting entity ID
            target_id: Target entity ID
            max_depth: Maximum path length

        Returns:
            List of entity IDs forming the path, or None if no path exists
        """
        import networkx as nx

        # Check if both nodes exist
        if source_id not in self._graph or target_id not in self._graph:
            return None

        try:
            path = nx.shortest_path(
                self._graph, source_id, target_id, weight=None
            )
            if len(path) <= max_depth + 1:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return None

    async def get_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    async def search_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> list[tuple[Entity, Relationship, Entity]]:
        """
        Search for facts matching the pattern.

        Any parameter can be None to match anything.

        Args:
            subject: Subject name pattern (optional)
            predicate: Relationship type (optional)
            obj: Object name pattern (optional)

        Returns:
            List of matching (subject, relationship, object) tuples
        """
        results = []

        for rel in self._relationships.values():
            source = self._entities.get(rel.source_id)
            target = self._entities.get(rel.target_id)

            if not source or not target:
                continue

            # Check subject match
            if subject and subject.lower() not in source.name.lower():
                continue

            # Check predicate match
            if predicate and predicate.lower() != rel.relation_type.lower():
                continue

            # Check object match
            if obj and obj.lower() not in target.name.lower():
                continue

            results.append((source, rel, target))

        return results

    async def count_entities(self) -> int:
        """Return the number of entities."""
        return len(self._entities)

    async def count_relationships(self) -> int:
        """Return the number of relationships."""
        return len(self._relationships)

    def get_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        import networkx as nx

        return {
            "entities": len(self._entities),
            "relationships": len(self._relationships),
            "entity_types": dict(
                (t.value, sum(1 for e in self._entities.values() if e.entity_type == t))
                for t in EntityType
                if any(e.entity_type == t for e in self._entities.values())
            ),
            "relationship_types": dict(
                (rt, sum(1 for r in self._relationships.values() if r.relation_type == rt))
                for rt in set(r.relation_type for r in self._relationships.values())
            ),
            "connected_components": nx.number_weakly_connected_components(self._graph)
            if self._graph.number_of_nodes() > 0
            else 0,
        }

    # ========== Persistence ==========

    def _save(self) -> None:
        """Save knowledge graph to disk."""
        if not self._persist_path:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "entities": {eid: e.to_dict() for eid, e in self._entities.items()},
            "relationships": {rid: r.to_dict() for rid, r in self._relationships.items()},
        }

        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load knowledge graph from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)

            # Load entities
            for eid, edata in data.get("entities", {}).items():
                entity = Entity.from_dict(edata)
                self._entities[eid] = entity
                self._graph.add_node(
                    eid,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    description=entity.description,
                    **entity.properties,
                )

            # Load relationships
            for rid, rdata in data.get("relationships", {}).items():
                rel = Relationship.from_dict(rdata)
                self._relationships[rid] = rel
                self._graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    relation_type=rel.relation_type,
                    confidence=rel.confidence,
                    **rel.properties,
                )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not load semantic memory: {e}")
