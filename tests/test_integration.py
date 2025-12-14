"""
Integration tests for Engram memory system.

These tests verify that different memory types work correctly together
in realistic usage scenarios.
"""

import pytest
from datetime import datetime

from engram_memory import (
    WorkingMemory,
    EpisodicMemory,
    Episode,
    EpisodeType,
    SemanticMemory,
    EntityType,
    RetrievalQuery,
    MemoryContext,
)
from engram_memory.core.interfaces import MemoryType


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def all_memories(temp_dir, chroma_store_memory, mock_embeddings):
    """Create all memory types for integration testing."""
    return {
        "working": WorkingMemory(max_conversation_turns=50),
        "episodic": EpisodicMemory(
            store=chroma_store_memory,
            embeddings=mock_embeddings,
        ),
        "semantic": SemanticMemory(),
    }


# ============================================================
# Scenario: Development Session
# ============================================================

class TestDevelopmentSessionScenario:
    """
    Test a realistic development session where a user asks
    for help implementing a feature.
    """

    @pytest.mark.asyncio
    async def test_full_development_session(self, all_memories):
        """Test a complete development session flow."""
        working = all_memories["working"]
        episodic = all_memories["episodic"]
        semantic = all_memories["semantic"]

        # === Phase 1: User starts a conversation ===
        working.add_turn("user", "Help me implement user authentication")
        working.add_turn("assistant", "I'll help you with authentication. Do you want JWT or session-based?")
        working.add_turn("user", "Let's use JWT with refresh tokens")

        # Set task context
        working.set_task(
            "Implement JWT authentication",
            context={"project": "myapp", "framework": "FastAPI"}
        )

        assert working.state.current_task == "Implement JWT authentication"
        assert len(working.state.conversation) == 3

        # === Phase 2: Store knowledge about decisions ===
        await semantic.add_fact(
            "myapp", "uses", "JWT",
            subject_type=EntityType.PROJECT,
            object_type=EntityType.TECHNOLOGY
        )
        await semantic.add_fact(
            "myapp", "uses", "FastAPI",
            subject_type=EntityType.PROJECT,
            object_type=EntityType.TECHNOLOGY
        )
        await semantic.add_fact(
            "user", "prefers", "refresh tokens",
            subject_type=EntityType.USER,
            object_type=EntityType.CONCEPT
        )

        # Verify semantic memory
        myapp = await semantic.get_entity_by_name("myapp")
        technologies = await semantic.get_related(myapp.id, relation_type="uses")
        assert len(technologies) == 2

        # === Phase 3: Complete the conversation and store as episode ===
        working.add_turn("assistant", "Here's the JWT implementation with refresh token rotation...")
        working.add_turn("user", "Perfect, that works great!")

        # Store the conversation as an episode
        episode = await episodic.store_conversation(
            turns=working.get_conversation_for_llm(),
            summary="Implemented JWT authentication with refresh tokens for myapp",
            project="myapp",
            tags=["auth", "jwt", "fastapi"]
        )

        assert episode.type == EpisodeType.CONVERSATION
        assert "jwt" in episode.tags

        # === Phase 4: Later, recall relevant memories ===
        # Simulate a new session where user asks about auth
        working_new = WorkingMemory()
        working_new.add_turn("user", "How did we handle token refresh?")

        # Query episodic memory using the exact embed text for reliable matching
        results = await episodic.retrieve(
            RetrievalQuery(
                text=episode.to_embed_text(),
                limit=5,
                min_relevance=0.0
            )
        )

        assert len(results) > 0
        assert results[0].memory_type == MemoryType.EPISODIC

        # Query semantic memory
        user_prefs = await semantic.search_facts(subject="user", predicate="prefers")
        assert len(user_prefs) > 0

    @pytest.mark.asyncio
    async def test_context_building(self, all_memories):
        """Test building a MemoryContext from all memory types."""
        working = all_memories["working"]
        episodic = all_memories["episodic"]
        semantic = all_memories["semantic"]

        # Set up some data
        working.add_turn("user", "Let's work on the database schema")
        working.set_task("Design database schema")

        await semantic.add_entity(
            "PostgreSQL",
            EntityType.DATABASE,
            description="Primary database"
        )

        episode = Episode(
            type=EpisodeType.TASK,
            summary="Set up database migrations",
            tags=["database"]
        )
        await episodic.store(episode)

        # Build context (simulating what a coordinator would do)
        episodic_results = await episodic.retrieve(
            RetrievalQuery(text=episode.to_embed_text(), limit=3, min_relevance=0.0)
        )
        semantic_results = await semantic.retrieve(
            RetrievalQuery(text="PostgreSQL")
        )

        context = MemoryContext(
            working=working.to_dict(),
            episodic=episodic_results,
            semantic=semantic_results,
            procedural={}
        )

        # Verify context
        assert not context.is_empty()
        assert context.working["current_task"] == "Design database schema"

        # Generate prompt section
        prompt = context.to_prompt_section()
        assert len(prompt) > 0


# ============================================================
# Scenario: Knowledge Building Over Time
# ============================================================

class TestKnowledgeBuildingScenario:
    """Test building up knowledge across multiple interactions."""

    @pytest.mark.asyncio
    async def test_accumulating_knowledge(self, all_memories):
        """Test that knowledge accumulates correctly."""
        semantic = all_memories["semantic"]

        # Session 1: User talks about project setup
        await semantic.add_fact("myapp", "uses", "Python")
        await semantic.add_fact("myapp", "uses", "FastAPI")

        # Session 2: User adds more details
        await semantic.add_fact("myapp", "uses", "PostgreSQL")
        await semantic.add_fact("myapp", "uses", "Redis")

        # Session 3: User specifies preferences
        await semantic.add_fact("user", "prefers", "type hints")
        await semantic.add_fact("user", "prefers", "async/await")

        # Verify accumulated knowledge
        myapp = await semantic.get_entity_by_name("myapp")
        all_tech = await semantic.get_related(myapp.id, relation_type="uses")
        assert len(all_tech) == 4

        user = await semantic.get_entity_by_name("user")
        prefs = await semantic.get_related(user.id, relation_type="prefers")
        assert len(prefs) == 2

    @pytest.mark.asyncio
    async def test_episode_accumulation(self, all_memories):
        """Test that episodes accumulate and can be queried."""
        episodic = all_memories["episodic"]

        # Store multiple episodes over time
        episodes_data = [
            ("Set up project structure", ["setup", "structure"]),
            ("Implemented user model", ["database", "models"]),
            ("Added authentication", ["auth", "security"]),
            ("Fixed login bug", ["auth", "bugfix"]),
            ("Deployed to staging", ["deployment", "staging"]),
        ]

        for summary, tags in episodes_data:
            episode = Episode(
                type=EpisodeType.TASK,
                summary=summary,
                tags=tags
            )
            await episodic.store(episode)

        # Verify count
        count = await episodic.count()
        assert count == 5

        # Query by tags
        auth_episodes = await episodic.search_by_tags(["auth"])
        assert len(auth_episodes) == 2


# ============================================================
# Scenario: Error Recovery
# ============================================================

class TestErrorRecoveryScenario:
    """Test error handling and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_query_nonexistent_data(self, all_memories):
        """Test querying when no relevant data exists."""
        episodic = all_memories["episodic"]
        semantic = all_memories["semantic"]

        # Query empty memories
        episodic_results = await episodic.retrieve(
            RetrievalQuery(text="something that doesn't exist")
        )
        semantic_results = await semantic.retrieve(
            RetrievalQuery(text="nonexistent entity")
        )

        # Should return empty lists, not errors
        assert episodic_results == []
        assert semantic_results == []

    @pytest.mark.asyncio
    async def test_clear_and_rebuild(self, all_memories):
        """Test clearing and rebuilding memories."""
        episodic = all_memories["episodic"]
        semantic = all_memories["semantic"]

        # Add some data
        await semantic.add_fact("test", "has", "data")
        await episodic.store(Episode(summary="Test episode"))

        # Clear everything
        await episodic.clear()
        await semantic.clear()

        # Verify cleared
        assert await episodic.count() == 0
        assert await semantic.count_entities() == 0

        # Rebuild
        await semantic.add_fact("new", "has", "data")
        await episodic.store(Episode(summary="New episode"))

        assert await episodic.count() == 1
        assert await semantic.count_entities() == 2


# ============================================================
# Scenario: Working Memory Consolidation
# ============================================================

class TestWorkingMemoryConsolidation:
    """Test consolidating working memory to episodic."""

    @pytest.mark.asyncio
    async def test_consolidation_workflow(self, all_memories):
        """Test the workflow of consolidating working to episodic memory."""
        working = all_memories["working"]
        episodic = all_memories["episodic"]

        # Fill up working memory
        for i in range(8):  # 80% of 10 turns
            role = "user" if i % 2 == 0 else "assistant"
            working.add_turn(role, f"Message {i}")

        # Check if should consolidate
        # Note: Default fixture has max_turns=10
        working_custom = WorkingMemory(max_conversation_turns=10)
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            working_custom.add_turn(role, f"Message {i}")

        assert working_custom.should_consolidate() is True

        # Consolidate to episodic
        episode = await episodic.store_conversation(
            turns=working_custom.get_conversation_for_llm(),
            summary="Consolidated conversation",
            project="test"
        )

        # Clear working memory
        await working_custom.clear()

        # Verify
        assert len(working_custom.state.conversation) == 0
        assert await episodic.count() >= 1


# ============================================================
# Scenario: Cross-Memory Queries
# ============================================================

class TestCrossMemoryQueries:
    """Test queries that span multiple memory types."""

    @pytest.mark.asyncio
    async def test_related_information_across_memories(self, all_memories):
        """Test finding related info across memory types."""
        episodic = all_memories["episodic"]
        semantic = all_memories["semantic"]

        # Store related information in different memories
        # Semantic: Facts about the project
        await semantic.add_fact("myapp", "uses", "GraphQL")
        await semantic.add_fact("GraphQL", "requires", "schema definition")

        # Episodic: Experience implementing it
        episode = Episode(
            type=EpisodeType.TASK,
            summary="Implemented GraphQL API with schema-first approach",
            project="myapp",
            tags=["graphql", "api"]
        )
        await episodic.store(episode)

        # Query both for GraphQL-related info
        semantic_results = await semantic.retrieve(
            RetrievalQuery(text="GraphQL")
        )
        episodic_results = await episodic.search_by_tags(["graphql"])

        # Should find relevant info in both
        assert len(semantic_results) > 0
        assert len(episodic_results) > 0

        # The semantic knowledge and episodic experience are complementary
        graphql_entity = semantic_results[0].entry
        graphql_episode = episodic_results[0]

        assert "GraphQL" in graphql_entity.name
        assert "GraphQL" in graphql_episode.summary
