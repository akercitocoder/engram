"""
Tests for SemanticMemory implementation.
"""

import pytest
from pathlib import Path

from engram_memory import (
    SemanticMemory,
    Entity,
    EntityType,
    Relationship,
    Fact,
    RetrievalQuery,
)
from engram_memory.core.interfaces import MemoryType


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def semantic_memory():
    """Create an in-memory SemanticMemory instance."""
    return SemanticMemory(persist_path=None)


@pytest.fixture
def semantic_memory_persistent(temp_file):
    """Create a SemanticMemory instance with persistence."""
    return SemanticMemory(persist_path=temp_file)


# ============================================================
# Entity Tests
# ============================================================

class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(
            name="PostgreSQL",
            entity_type=EntityType.DATABASE,
            description="Relational database",
        )

        assert entity.name == "PostgreSQL"
        assert entity.entity_type == EntityType.DATABASE
        assert entity.id == "database:postgresql"

    def test_entity_auto_id(self):
        """Test that ID is auto-generated from type and name."""
        entity = Entity(name="My Project", entity_type=EntityType.PROJECT)

        assert entity.id == "project:my_project"

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity(
            name="FastAPI",
            entity_type=EntityType.TECHNOLOGY,
            properties={"language": "Python"},
        )

        data = entity.to_dict()

        assert data["name"] == "FastAPI"
        assert data["entity_type"] == "technology"
        assert data["properties"] == {"language": "Python"}

    def test_entity_from_dict(self):
        """Test entity deserialization."""
        entity = Entity(name="Test", entity_type=EntityType.CONCEPT)
        data = entity.to_dict()

        restored = Entity.from_dict(data)

        assert restored.id == entity.id
        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type


class TestRelationship:
    """Test Relationship dataclass."""

    def test_relationship_creation(self):
        """Test basic relationship creation."""
        rel = Relationship(
            source_id="user:alice",
            target_id="technology:python",
            relation_type="prefers",
        )

        assert rel.source_id == "user:alice"
        assert rel.target_id == "technology:python"
        assert rel.relation_type == "prefers"
        assert rel.id == "user:alice-prefers-technology:python"

    def test_relationship_to_dict(self):
        """Test relationship serialization."""
        rel = Relationship(
            source_id="a",
            target_id="b",
            relation_type="uses",
            properties={"since": "2024"},
        )

        data = rel.to_dict()

        assert data["source_id"] == "a"
        assert data["target_id"] == "b"
        assert data["relation_type"] == "uses"

    def test_relationship_from_dict(self):
        """Test relationship deserialization."""
        rel = Relationship(source_id="a", target_id="b", relation_type="uses")
        data = rel.to_dict()

        restored = Relationship.from_dict(data)

        assert restored.id == rel.id
        assert restored.relation_type == rel.relation_type


# ============================================================
# SemanticMemory Basic Tests
# ============================================================

class TestSemanticMemoryBasics:
    """Test basic SemanticMemory functionality."""

    def test_init(self, semantic_memory):
        """Test initialization."""
        assert semantic_memory.memory_type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_add_entity(self, semantic_memory):
        """Test adding an entity."""
        entity = await semantic_memory.add_entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            description="Programming language",
        )

        assert entity.name == "Python"
        assert entity.id == "technology:python"

    @pytest.mark.asyncio
    async def test_add_entity_deduplication(self, semantic_memory):
        """Test that adding same entity updates instead of duplicating."""
        entity1 = await semantic_memory.add_entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            confidence=0.5,
        )

        entity2 = await semantic_memory.add_entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            confidence=0.9,
            description="A great language",
        )

        # Should be same entity, updated
        assert entity1.id == entity2.id
        assert entity2.confidence == 0.9
        assert entity2.description == "A great language"

        # Should only have one entity
        count = await semantic_memory.count_entities()
        assert count == 1

    @pytest.mark.asyncio
    async def test_get_entity(self, semantic_memory):
        """Test getting an entity by ID."""
        await semantic_memory.add_entity(
            name="FastAPI",
            entity_type=EntityType.TECHNOLOGY,
        )

        entity = await semantic_memory.get_entity("technology:fastapi")

        assert entity is not None
        assert entity.name == "FastAPI"

    @pytest.mark.asyncio
    async def test_get_entity_by_name(self, semantic_memory):
        """Test getting an entity by name."""
        await semantic_memory.add_entity(
            name="PostgreSQL",
            entity_type=EntityType.DATABASE,
        )

        entity = await semantic_memory.get_entity_by_name("PostgreSQL")

        assert entity is not None
        assert entity.entity_type == EntityType.DATABASE


# ============================================================
# Fact Tests
# ============================================================

class TestSemanticMemoryFacts:
    """Test fact operations."""

    @pytest.mark.asyncio
    async def test_add_fact(self, semantic_memory):
        """Test adding a fact."""
        subject, rel, obj = await semantic_memory.add_fact(
            "user",
            "prefers",
            "TypeScript",
            subject_type=EntityType.USER,
            object_type=EntityType.TECHNOLOGY,
        )

        assert subject.name == "user"
        assert obj.name == "TypeScript"
        assert rel.relation_type == "prefers"

    @pytest.mark.asyncio
    async def test_add_multiple_facts(self, semantic_memory):
        """Test adding multiple facts."""
        await semantic_memory.add_fact("myapp", "uses", "PostgreSQL")
        await semantic_memory.add_fact("myapp", "uses", "Redis")
        await semantic_memory.add_fact("myapp", "written_in", "Python")

        # myapp should have 3 relationships
        entity = await semantic_memory.get_entity_by_name("myapp")
        related = await semantic_memory.get_related(entity.id)

        assert len(related) == 3

    @pytest.mark.asyncio
    async def test_search_facts(self, semantic_memory):
        """Test searching for facts."""
        await semantic_memory.add_fact("user", "prefers", "Python")
        await semantic_memory.add_fact("user", "prefers", "TypeScript")
        await semantic_memory.add_fact("user", "dislikes", "Java")

        # Search for preferences
        results = await semantic_memory.search_facts(
            subject="user",
            predicate="prefers",
        )

        assert len(results) == 2
        objects = [r[2].name for r in results]
        assert "Python" in objects
        assert "TypeScript" in objects


# ============================================================
# Relationship Tests
# ============================================================

class TestSemanticMemoryRelationships:
    """Test relationship operations."""

    @pytest.mark.asyncio
    async def test_add_relationship(self, semantic_memory):
        """Test adding a relationship."""
        await semantic_memory.add_entity("auth.py", EntityType.FILE)
        await semantic_memory.add_entity("authenticate", EntityType.FUNCTION)

        rel = await semantic_memory.add_relationship(
            source_id="file:auth.py",
            target_id="function:authenticate",
            relation_type="contains",
        )

        assert rel.relation_type == "contains"

    @pytest.mark.asyncio
    async def test_get_related_outgoing(self, semantic_memory):
        """Test getting related entities (outgoing)."""
        await semantic_memory.add_fact("project", "uses", "FastAPI")
        await semantic_memory.add_fact("project", "uses", "SQLAlchemy")

        project = await semantic_memory.get_entity_by_name("project")
        related = await semantic_memory.get_related(
            project.id,
            relation_type="uses",
            direction="outgoing",
        )

        assert len(related) == 2
        names = [e.name for e in related]
        assert "FastAPI" in names
        assert "SQLAlchemy" in names

    @pytest.mark.asyncio
    async def test_get_related_incoming(self, semantic_memory):
        """Test getting related entities (incoming)."""
        await semantic_memory.add_fact("project_a", "uses", "PostgreSQL")
        await semantic_memory.add_fact("project_b", "uses", "PostgreSQL")

        pg = await semantic_memory.get_entity_by_name("PostgreSQL")
        related = await semantic_memory.get_related(
            pg.id,
            direction="incoming",
        )

        assert len(related) == 2

    @pytest.mark.asyncio
    async def test_get_relationships(self, semantic_memory):
        """Test getting relationships for an entity."""
        await semantic_memory.add_fact("user", "prefers", "Python")
        await semantic_memory.add_fact("user", "knows", "JavaScript")

        user = await semantic_memory.get_entity_by_name("user")
        rels = await semantic_memory.get_relationships(user.id)

        assert len(rels) == 2


# ============================================================
# Graph Traversal Tests
# ============================================================

class TestSemanticMemoryTraversal:
    """Test graph traversal operations."""

    @pytest.mark.asyncio
    async def test_find_path(self, semantic_memory):
        """Test finding path between entities."""
        # Create a chain: A -> B -> C
        await semantic_memory.add_fact("A", "connects_to", "B")
        await semantic_memory.add_fact("B", "connects_to", "C")

        a = await semantic_memory.get_entity_by_name("A")
        c = await semantic_memory.get_entity_by_name("C")

        path = await semantic_memory.find_path(a.id, c.id)

        assert path is not None
        assert len(path) == 3

    @pytest.mark.asyncio
    async def test_find_path_no_connection(self, semantic_memory):
        """Test finding path when none exists."""
        await semantic_memory.add_entity("isolated1", EntityType.OTHER)
        await semantic_memory.add_entity("isolated2", EntityType.OTHER)

        path = await semantic_memory.find_path("other:isolated1", "other:isolated2")

        assert path is None

    @pytest.mark.asyncio
    async def test_get_by_type(self, semantic_memory):
        """Test getting entities by type."""
        await semantic_memory.add_entity("Python", EntityType.TECHNOLOGY)
        await semantic_memory.add_entity("JavaScript", EntityType.TECHNOLOGY)
        await semantic_memory.add_entity("myapp", EntityType.PROJECT)

        techs = await semantic_memory.get_by_type(EntityType.TECHNOLOGY)

        assert len(techs) == 2
        names = [e.name for e in techs]
        assert "Python" in names
        assert "JavaScript" in names


# ============================================================
# Retrieval Tests
# ============================================================

class TestSemanticMemoryRetrieval:
    """Test retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_by_name(self, semantic_memory):
        """Test retrieving by entity name."""
        await semantic_memory.add_entity(
            "PostgreSQL",
            EntityType.DATABASE,
            description="Open source relational database",
        )

        results = await semantic_memory.retrieve(
            RetrievalQuery(text="PostgreSQL")
        )

        assert len(results) > 0
        assert results[0].entry.name == "PostgreSQL"

    @pytest.mark.asyncio
    async def test_retrieve_by_description(self, semantic_memory):
        """Test retrieving by description match."""
        await semantic_memory.add_entity(
            "Redis",
            EntityType.DATABASE,
            description="In-memory data structure store",
        )

        results = await semantic_memory.retrieve(
            RetrievalQuery(text="in-memory")
        )

        assert len(results) > 0
        assert results[0].entry.name == "Redis"

    @pytest.mark.asyncio
    async def test_retrieve_empty(self, semantic_memory):
        """Test retrieving from empty memory."""
        results = await semantic_memory.retrieve(
            RetrievalQuery(text="anything")
        )

        assert results == []


# ============================================================
# Update and Delete Tests
# ============================================================

class TestSemanticMemoryUpdateDelete:
    """Test update and delete operations."""

    @pytest.mark.asyncio
    async def test_update_entity(self, semantic_memory):
        """Test updating an entity."""
        entity = await semantic_memory.add_entity("test", EntityType.CONCEPT)

        result = await semantic_memory.update(
            entity.id,
            {"description": "Updated description"},
        )

        assert result is True
        updated = await semantic_memory.get_entity(entity.id)
        assert updated.description == "Updated description"

    @pytest.mark.asyncio
    async def test_delete_entity(self, semantic_memory):
        """Test deleting an entity."""
        await semantic_memory.add_fact("a", "relates_to", "b")

        a = await semantic_memory.get_entity_by_name("a")
        await semantic_memory.delete(a.id)

        # Entity should be gone
        assert await semantic_memory.get_entity(a.id) is None

        # Relationships should also be gone
        rels = await semantic_memory.get_relationships(a.id)
        assert len(rels) == 0

    @pytest.mark.asyncio
    async def test_clear(self, semantic_memory):
        """Test clearing all memory."""
        await semantic_memory.add_fact("a", "uses", "b")
        await semantic_memory.add_fact("c", "uses", "d")

        await semantic_memory.clear()

        assert await semantic_memory.count_entities() == 0
        assert await semantic_memory.count_relationships() == 0


# ============================================================
# Statistics Tests
# ============================================================

class TestSemanticMemoryStats:
    """Test statistics methods."""

    @pytest.mark.asyncio
    async def test_counts(self, semantic_memory):
        """Test entity and relationship counts."""
        await semantic_memory.add_fact("user", "prefers", "Python")
        await semantic_memory.add_fact("user", "knows", "JavaScript")

        entities = await semantic_memory.count_entities()
        relationships = await semantic_memory.count_relationships()

        # user, Python, JavaScript = 3 entities
        assert entities == 3
        assert relationships == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, semantic_memory):
        """Test getting statistics."""
        await semantic_memory.add_entity("Python", EntityType.TECHNOLOGY)
        await semantic_memory.add_entity("FastAPI", EntityType.TECHNOLOGY)
        await semantic_memory.add_entity("myapp", EntityType.PROJECT)

        # Add relationship between existing entities
        myapp = await semantic_memory.get_entity_by_name("myapp")
        fastapi = await semantic_memory.get_entity_by_name("FastAPI")
        await semantic_memory.add_relationship(
            myapp.id, fastapi.id, "uses"
        )

        stats = semantic_memory.get_stats()

        assert stats["entities"] == 3
        assert stats["relationships"] == 1
        assert "technology" in stats["entity_types"]
        assert stats["entity_types"]["technology"] == 2


# ============================================================
# Persistence Tests
# ============================================================

class TestSemanticMemoryPersistence:
    """Test persistence functionality."""

    @pytest.mark.asyncio
    async def test_persistence(self, temp_file):
        """Test saving and loading."""
        # Create and populate
        memory1 = SemanticMemory(persist_path=temp_file)
        await memory1.add_fact("user", "prefers", "Python")
        await memory1.add_entity(
            "FastAPI",
            EntityType.TECHNOLOGY,
            description="Web framework",
        )

        # Create new instance from same file
        memory2 = SemanticMemory(persist_path=temp_file)

        # Should have loaded data
        assert await memory2.count_entities() == 3
        entity = await memory2.get_entity_by_name("FastAPI")
        assert entity is not None
        assert entity.description == "Web framework"

    @pytest.mark.asyncio
    async def test_no_persistence(self, semantic_memory, temp_dir):
        """Test no files created without persist path."""
        await semantic_memory.add_fact("a", "b", "c")

        # No files should be created
        files = list(Path(temp_dir).glob("*.json"))
        assert len(files) == 0
