# Engram

**Memory architecture for AI agents** - giving AI the ability to remember.

> *An engram is the physical trace a memory leaves in the brain. This library gives AI agents the same capability.*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-136%20passed-brightgreen.svg)]()

## What is Engram?

Engram is a Python framework that provides persistent, human-like memory for AI agents. Unlike stateless assistants that forget everything between sessions, agents built with Engram can:

- **Remember** past conversations and experiences
- **Recall** relevant context using semantic search
- **Learn** preferences and adapt over time
- **Build** knowledge graphs of facts and relationships

## Installation

```bash
pip install engram-memory
```

Or with optional dependencies:

```bash
# With local embeddings (no API key needed)
pip install engram-memory[local]

# With OpenAI embeddings
pip install engram-memory[openai]

# Everything
pip install engram-memory[all]
```

## Quick Start

### Working Memory (Current Context)

```python
from engram_memory import WorkingMemory

# Track current conversation
memory = WorkingMemory(max_conversation_turns=50)

memory.add_turn("user", "Help me with authentication")
memory.add_turn("assistant", "I'll help you implement JWT auth...")

# Set current task
memory.set_task("Implement JWT authentication", context={"project": "myapp"})

# Track active files
memory.add_active_file("src/auth.py")

# Get conversation for LLM
messages = memory.get_conversation_for_llm(n_turns=10)
```

### Episodic Memory (Past Experiences)

```python
import asyncio
from engram_memory import EpisodicMemory, Episode, EpisodeType, RetrievalQuery, ChromaDBStore
from engram_memory.utils import get_default_embeddings

async def main():
    # Initialize
    memory = EpisodicMemory(
        store=ChromaDBStore(persist_directory="./engram_data"),
        embeddings=get_default_embeddings()
    )

    # Store an experience
    episode = Episode(
        type=EpisodeType.CONVERSATION,
        summary="Implemented JWT authentication with httpOnly cookies",
        key_points=[
            "Used httpOnly cookies for security",
            "Added refresh token rotation",
        ],
        project="myapp",
        tags=["auth", "jwt", "security"]
    )
    await memory.store(episode)

    # Later, recall relevant experiences
    results = await memory.retrieve(
        RetrievalQuery(text="How did we handle authentication?", limit=5)
    )

    for result in results:
        print(f"[{result.relevance_score:.0%}] {result.entry.summary}")

asyncio.run(main())
```

## Memory Types

| Memory | Purpose | Storage | Status |
|--------|---------|---------|--------|
| **Working** | Current conversation context | In-memory + JSON | Done |
| **Episodic** | Past experiences & conversations | ChromaDB | Done |
| **Semantic** | Facts, entities, relationships | NetworkX | Done |
| **Procedural** | Learned preferences & patterns | JSON | Planned |

### Semantic Memory (Knowledge Graph)

```python
import asyncio
from engram_memory import SemanticMemory, EntityType

async def main():
    memory = SemanticMemory(persist_path="./engram_data/knowledge.json")

    # Store facts as triples
    await memory.add_fact("user", "prefers", "TypeScript",
                          subject_type=EntityType.USER,
                          object_type=EntityType.TECHNOLOGY)
    await memory.add_fact("myapp", "uses", "PostgreSQL",
                          subject_type=EntityType.PROJECT,
                          object_type=EntityType.DATABASE)

    # Query relationships
    user = await memory.get_entity_by_name("user")
    preferences = await memory.get_related(user.id, relation_type="prefers")
    print(f"User prefers: {[p.name for p in preferences]}")

    # Search facts
    results = await memory.search_facts(subject="myapp", predicate="uses")
    for subj, rel, obj in results:
        print(f"{subj.name} {rel.relation_type} {obj.name}")

asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR AI AGENT                        │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                  ENGRAM MEMORY                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ Working  │ │ Episodic │ │ Semantic │ │Procedural│   │
│  │ Memory   │ │ Memory   │ │ Memory   │ │ Memory   │   │
│  │   ✅     │ │    ✅    │ │    ✅    │ │   🔜     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Features

- **Zero Configuration** - Works out of the box with ChromaDB (embedded, no server)
- **Flexible Embeddings** - OpenAI, local sentence-transformers, or bring your own
- **Async First** - Built for modern async Python applications
- **Type Safe** - Full type hints and Protocol-based abstractions
- **Persistence** - Automatic state persistence with JSON and vector stores

## Development

```bash
git clone https://github.com/akercitocoder/engram.git
cd engram

pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy engram_memory
```

## Roadmap

- [x] Working Memory - current conversation tracking
- [x] Episodic Memory - vector-based experience recall
- [x] ChromaDB integration
- [x] Multiple embedding providers
- [x] Semantic Memory - knowledge graphs with NetworkX
- [ ] Procedural Memory - preference learning
- [ ] Memory Coordinator - unified retrieval
- [ ] Qdrant Cloud support

## License

MIT

## Author

Francisco Perez
