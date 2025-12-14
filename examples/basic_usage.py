"""
Basic usage example for Engram memory system.

This example demonstrates:
1. Working Memory - tracking current conversation
2. Episodic Memory - storing and retrieving past experiences

Run this example:
    cd engram
    pip install -e ".[all]"
    python examples/basic_usage.py
"""

import asyncio
from pathlib import Path

# Engram imports
from engram_memory import (
    WorkingMemory,
    EpisodicMemory,
    Episode,
    EpisodeType,
    RetrievalQuery,
    ChromaDBStore,
)
from engram_memory.utils import LocalEmbeddings, get_default_embeddings


async def demo_working_memory():
    """Demonstrate working memory capabilities."""
    print("\n" + "=" * 60)
    print("WORKING MEMORY DEMO")
    print("=" * 60)

    # Initialize working memory
    memory = WorkingMemory(
        max_conversation_turns=50,
        persist_path="./engram_data/working_memory.json",
    )

    # Add conversation turns
    memory.add_turn("user", "Help me implement JWT authentication")
    memory.add_turn(
        "assistant",
        "I'll help you with JWT auth. Do you want to use httpOnly cookies or localStorage?"
    )
    memory.add_turn("user", "Let's use httpOnly cookies for better security")
    memory.add_turn(
        "assistant",
        "Great choice! httpOnly cookies prevent XSS attacks. Let me show you the implementation."
    )

    # Set current task
    memory.set_task(
        "Implement JWT authentication",
        context={"project": "myapp", "framework": "FastAPI"}
    )

    # Mark active files
    memory.add_active_file("src/auth.py")
    memory.add_active_file("src/middleware.py")

    # Add attention markers
    memory.add_attention("User prefers httpOnly cookies")

    # Display state
    print(f"\n{memory.summary()}")
    print(f"\nToken estimate: ~{memory.get_token_estimate()} tokens")

    # Get conversation for LLM
    messages = memory.get_conversation_for_llm(n_turns=10)
    print(f"\nConversation ({len(messages)} turns):")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content'][:50]}...")

    # Check if should consolidate
    print(f"\nShould consolidate: {memory.should_consolidate()}")

    return memory


async def demo_episodic_memory():
    """Demonstrate episodic memory capabilities."""
    print("\n" + "=" * 60)
    print("EPISODIC MEMORY DEMO")
    print("=" * 60)

    # Initialize storage and embeddings
    store = ChromaDBStore(
        collection_name="engram_demo",
        persist_directory="./engram_data/chroma",
    )

    # Use local embeddings (no API key needed)
    try:
        embeddings = LocalEmbeddings()
        print("\nUsing local embeddings (sentence-transformers)")
    except ImportError:
        print("\nsentence-transformers not installed, using ChromaDB default")
        from engram_memory.utils.embeddings import ChromaEmbeddings
        embeddings = ChromaEmbeddings()

    # Initialize episodic memory
    memory = EpisodicMemory(
        store=store,
        embeddings=embeddings,
        data_path="./engram_data/episodes.json",
    )

    # Store some episodes
    print("\nStoring episodes...")

    # Episode 1: JWT implementation
    episode1 = Episode(
        type=EpisodeType.CONVERSATION,
        summary="Implemented JWT authentication with httpOnly cookies",
        key_points=[
            "Used httpOnly cookies instead of localStorage",
            "Added refresh token rotation",
            "Set 15min access token expiry",
        ],
        project="myapp",
        tags=["auth", "jwt", "security", "cookies"],
    )
    await memory.store(episode1)
    print(f"  Stored: {episode1.summary[:50]}...")

    # Episode 2: Database setup
    episode2 = Episode(
        type=EpisodeType.TASK,
        summary="Set up PostgreSQL database with SQLAlchemy",
        key_points=[
            "Chose PostgreSQL over MongoDB for ACID compliance",
            "Used SQLAlchemy async for ORM",
            "Configured connection pooling",
        ],
        project="myapp",
        tags=["database", "postgresql", "sqlalchemy"],
    )
    await memory.store(episode2)
    print(f"  Stored: {episode2.summary[:50]}...")

    # Episode 3: Bug fix
    episode3 = Episode(
        type=EpisodeType.DEBUGGING,
        summary="Fixed token expiration bug causing logout loops",
        key_points=[
            "Bug was due to clock skew between servers",
            "Added 30 second buffer to expiration check",
            "Implemented automatic token refresh",
        ],
        project="myapp",
        tags=["bug", "auth", "tokens", "debugging"],
        outcome="resolved",
    )
    await memory.store(episode3)
    print(f"  Stored: {episode3.summary[:50]}...")

    # Episode 4: Different project
    episode4 = Episode(
        type=EpisodeType.RESEARCH,
        summary="Researched caching strategies for API optimization",
        key_points=[
            "Evaluated Redis vs Memcached",
            "Chose Redis for data persistence",
            "Implemented write-through caching pattern",
        ],
        project="api-gateway",
        tags=["caching", "redis", "performance"],
    )
    await memory.store(episode4)
    print(f"  Stored: {episode4.summary[:50]}...")

    print(f"\nTotal episodes stored: {await memory.count()}")

    # Demonstrate retrieval
    print("\n" + "-" * 40)
    print("RETRIEVAL DEMO")
    print("-" * 40)

    # Query 1: Find auth-related experiences
    print("\nQuery: 'How did we handle authentication?'")
    results = await memory.retrieve(
        RetrievalQuery(text="How did we handle authentication?", limit=3)
    )
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.relevance_score:.0%}] {result.entry.summary[:60]}...")

    # Query 2: Find debugging experiences
    print("\nQuery: 'token expiration issues'")
    results = await memory.retrieve(
        RetrievalQuery(text="token expiration issues", limit=3)
    )
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.relevance_score:.0%}] {result.entry.summary[:60]}...")

    # Query 3: Find caching-related
    print("\nQuery: 'caching implementation'")
    results = await memory.retrieve(
        RetrievalQuery(text="caching implementation", limit=3)
    )
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result.relevance_score:.0%}] {result.entry.summary[:60]}...")

    # Get by project
    print("\n" + "-" * 40)
    print("FILTER BY PROJECT")
    print("-" * 40)

    myapp_episodes = await memory.get_by_project("myapp")
    print(f"\nEpisodes for 'myapp' project ({len(myapp_episodes)}):")
    for ep in myapp_episodes:
        print(f"  - [{ep.type.value}] {ep.summary[:50]}...")

    # Get by tags
    print("\n" + "-" * 40)
    print("FILTER BY TAGS")
    print("-" * 40)

    auth_episodes = await memory.search_by_tags(["auth", "security"])
    print(f"\nEpisodes tagged 'auth' or 'security' ({len(auth_episodes)}):")
    for ep in auth_episodes:
        print(f"  - {ep.summary[:50]}... (tags: {', '.join(ep.tags)})")

    return memory


async def demo_conversation_to_episode():
    """Demonstrate storing a conversation as an episode."""
    print("\n" + "=" * 60)
    print("CONVERSATION TO EPISODE DEMO")
    print("=" * 60)

    # Initialize
    store = ChromaDBStore(
        collection_name="engram_conversations",
        persist_directory="./engram_data/chroma",
    )

    try:
        embeddings = LocalEmbeddings()
    except ImportError:
        from engram_memory.utils.embeddings import ChromaEmbeddings
        embeddings = ChromaEmbeddings()

    memory = EpisodicMemory(
        store=store,
        embeddings=embeddings,
    )

    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "How do I implement rate limiting?"},
        {"role": "assistant", "content": "I recommend using a token bucket algorithm. Here's how..."},
        {"role": "user", "content": "Should I use Redis for this?"},
        {"role": "assistant", "content": "Yes, Redis is great for rate limiting because it's fast and supports atomic operations."},
        {"role": "user", "content": "Perfect, let's implement it with Redis"},
    ]

    # Store the conversation as an episode
    episode = await memory.store_conversation(
        turns=conversation,
        summary="Implemented rate limiting with Redis token bucket",
        project="api-gateway",
        tags=["rate-limiting", "redis", "api"],
    )

    print(f"\nStored conversation as episode:")
    print(f"  ID: {episode.id}")
    print(f"  Summary: {episode.summary}")
    print(f"  Key points: {episode.key_points}")
    print(f"  Importance: {episode.importance:.2f}")

    # Later, retrieve it
    print("\nQuery: 'rate limiting implementation'")
    results = await memory.retrieve(
        RetrievalQuery(text="rate limiting implementation", limit=1)
    )
    if results:
        result = results[0]
        print(f"  Found: [{result.relevance_score:.0%}] {result.entry.summary}")


async def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  ENGRAM MEMORY SYSTEM DEMO")
    print("#" * 60)

    # Create data directory
    Path("./engram_data").mkdir(exist_ok=True)

    # Run demos
    await demo_working_memory()
    await demo_episodic_memory()
    await demo_conversation_to_episode()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nData stored in ./engram_data/")
    print("Run again to see persistence working!")


if __name__ == "__main__":
    asyncio.run(main())
