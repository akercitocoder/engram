"""
Working Memory implementation for Engram.

Working memory holds the current conversation context and active state.
It's the "RAM" of the agent - fast, limited capacity, and temporary.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid
import json
from pathlib import Path

from engram_memory.core.interfaces import (
    BaseMemory,
    MemoryType,
    MemoryEntry,
    RetrievalQuery,
    RetrievalResult,
    ConversationTurn,
)


@dataclass
class WorkingMemoryState(MemoryEntry):
    """The complete working memory state."""

    # Conversation history (recent turns)
    conversation: list = field(default_factory=list)
    max_turns: int = 50

    # Current task/goal
    current_task: Optional[str] = None
    task_context: dict = field(default_factory=dict)

    # Active files being discussed/edited
    active_files: list[str] = field(default_factory=list)

    # Scratchpad for intermediate reasoning
    scratchpad: dict = field(default_factory=dict)

    # Session info
    session_id: str = ""
    session_start: datetime = field(default_factory=datetime.utcnow)

    # Attention markers - what's important right now
    attention: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if not self.session_id:
            self.session_id = str(uuid.uuid4())[:8]


class WorkingMemory(BaseMemory[WorkingMemoryState]):
    """
    Working memory implementation.

    Characteristics:
    - Fast, in-memory storage
    - Limited capacity (mimics human working memory)
    - Automatically trims old conversation turns
    - Can persist to disk for session recovery
    """

    def __init__(
        self,
        max_conversation_turns: int = 50,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize working memory.

        Args:
            max_conversation_turns: Maximum conversation turns to keep
            persist_path: Optional path to persist state for recovery
        """
        self._max_turns = max_conversation_turns
        self._persist_path = Path(persist_path) if persist_path else None
        self._state = WorkingMemoryState(
            id="working_memory",
            max_turns=max_conversation_turns,
        )

        # Load persisted state if available
        if self._persist_path:
            self._load()

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType.WORKING

    @property
    def state(self) -> WorkingMemoryState:
        """Get the current working memory state."""
        return self._state

    async def store(self, entry: WorkingMemoryState) -> str:
        """Replace the entire working memory state."""
        self._state = entry
        self._persist()
        return entry.id

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Return current state as a single result."""
        return [
            RetrievalResult(
                entry=self._state,
                relevance_score=1.0,
                memory_type=MemoryType.WORKING,
            )
        ]

    async def update(self, entry_id: str, updates: dict) -> bool:
        """Update specific fields in working memory."""
        for key, value in updates.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self._state.updated_at = datetime.utcnow()
        self._persist()
        return True

    async def delete(self, entry_id: str) -> bool:
        """Reset working memory."""
        await self.clear()
        return True

    async def clear(self) -> None:
        """Clear working memory to fresh state."""
        self._state = WorkingMemoryState(
            id="working_memory",
            max_turns=self._max_turns,
        )
        self._persist()

    # ========== Working Memory Specific Methods ==========

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ConversationTurn:
        """
        Add a conversation turn.

        Args:
            role: "user" or "assistant"
            content: The message content
            metadata: Optional metadata for the turn

        Returns:
            The created ConversationTurn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self._state.conversation.append(turn)

        # Trim if over capacity
        while len(self._state.conversation) > self._max_turns:
            self._state.conversation.pop(0)

        self._state.updated_at = datetime.utcnow()
        self._persist()
        return turn

    def set_task(self, task: str, context: Optional[dict] = None) -> None:
        """
        Set the current task/goal.

        Args:
            task: Description of current task
            context: Additional context for the task
        """
        self._state.current_task = task
        self._state.task_context = context or {}
        self._state.updated_at = datetime.utcnow()
        self._persist()

    def clear_task(self) -> None:
        """Clear the current task."""
        self._state.current_task = None
        self._state.task_context = {}
        self._state.updated_at = datetime.utcnow()
        self._persist()

    def add_active_file(self, file_path: str) -> None:
        """Mark a file as being actively worked on."""
        if file_path not in self._state.active_files:
            self._state.active_files.append(file_path)
            # Keep list bounded
            if len(self._state.active_files) > 20:
                self._state.active_files.pop(0)
            self._persist()

    def remove_active_file(self, file_path: str) -> None:
        """Remove a file from active files."""
        if file_path in self._state.active_files:
            self._state.active_files.remove(file_path)
            self._persist()

    def add_attention(self, item: str) -> None:
        """Mark something as requiring attention."""
        if item not in self._state.attention:
            self._state.attention.append(item)
            # Keep attention list bounded
            if len(self._state.attention) > 10:
                self._state.attention.pop(0)
            self._persist()

    def clear_attention(self) -> None:
        """Clear all attention markers."""
        self._state.attention = []
        self._persist()

    def set_scratchpad(self, key: str, value: any) -> None:
        """Store something in the scratchpad."""
        self._state.scratchpad[key] = value
        self._persist()

    def get_scratchpad(self, key: str, default: any = None) -> any:
        """Get something from the scratchpad."""
        return self._state.scratchpad.get(key, default)

    def get_recent_context(self, n_turns: int = 10) -> list[ConversationTurn]:
        """
        Get the n most recent conversation turns.

        Args:
            n_turns: Number of turns to retrieve

        Returns:
            List of recent conversation turns
        """
        turns = self._state.conversation
        return turns[-n_turns:] if len(turns) > n_turns else turns[:]

    def get_conversation_for_llm(self, n_turns: int = 20) -> list[dict]:
        """
        Get conversation formatted for LLM API.

        Args:
            n_turns: Number of turns to include

        Returns:
            List of message dicts with 'role' and 'content'
        """
        turns = self.get_recent_context(n_turns)
        return [{"role": t.role, "content": t.content} for t in turns]

    def should_consolidate(self) -> bool:
        """
        Check if working memory should be consolidated to long-term.

        Returns True when conversation is getting long and should
        be summarized into episodic memory.
        """
        return len(self._state.conversation) >= self._max_turns * 0.8

    def get_token_estimate(self) -> int:
        """
        Estimate the number of tokens in working memory.

        Uses rough approximation of 4 chars per token.
        """
        total_chars = sum(len(t.content) for t in self._state.conversation)
        if self._state.current_task:
            total_chars += len(self._state.current_task)
        return total_chars // 4

    def to_dict(self) -> dict:
        """Export working memory as dictionary."""
        return {
            "session_id": self._state.session_id,
            "session_start": self._state.session_start.isoformat(),
            "conversation": [t.to_dict() for t in self._state.conversation],
            "current_task": self._state.current_task,
            "task_context": self._state.task_context,
            "active_files": self._state.active_files,
            "attention": self._state.attention,
            "scratchpad": self._state.scratchpad,
            "turn_count": len(self._state.conversation),
        }

    def summary(self) -> str:
        """Get a brief summary of working memory state."""
        parts = [f"Session: {self._state.session_id}"]
        parts.append(f"Turns: {len(self._state.conversation)}/{self._max_turns}")

        if self._state.current_task:
            parts.append(f"Task: {self._state.current_task[:50]}...")

        if self._state.active_files:
            parts.append(f"Files: {len(self._state.active_files)} active")

        if self._state.attention:
            parts.append(f"Attention: {len(self._state.attention)} items")

        return " | ".join(parts)

    # ========== Persistence ==========

    def _persist(self) -> None:
        """Persist state to disk if path configured."""
        if not self._persist_path:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "id": self._state.id,
            "session_id": self._state.session_id,
            "session_start": self._state.session_start.isoformat(),
            "created_at": self._state.created_at.isoformat(),
            "updated_at": self._state.updated_at.isoformat(),
            "conversation": [t.to_dict() for t in self._state.conversation],
            "current_task": self._state.current_task,
            "task_context": self._state.task_context,
            "active_files": self._state.active_files,
            "scratchpad": self._state.scratchpad,
            "attention": self._state.attention,
        }

        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load state from disk if available."""
        if not self._persist_path or not self._persist_path.exists():
            return

        try:
            with open(self._persist_path, "r") as f:
                data = json.load(f)

            self._state = WorkingMemoryState(
                id=data.get("id", "working_memory"),
                session_id=data.get("session_id", ""),
                session_start=datetime.fromisoformat(data["session_start"]),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                conversation=[
                    ConversationTurn.from_dict(t) for t in data.get("conversation", [])
                ],
                current_task=data.get("current_task"),
                task_context=data.get("task_context", {}),
                active_files=data.get("active_files", []),
                scratchpad=data.get("scratchpad", {}),
                attention=data.get("attention", []),
                max_turns=self._max_turns,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If loading fails, start fresh
            print(f"Warning: Could not load working memory state: {e}")
