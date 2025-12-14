"""
Tests for WorkingMemory implementation.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

from engram_memory import WorkingMemory
from engram_memory.core.interfaces import MemoryType, ConversationTurn
from engram_memory.working.memory import WorkingMemoryState


class TestWorkingMemoryBasics:
    """Test basic WorkingMemory functionality."""

    def test_init_default(self, working_memory):
        """Test default initialization."""
        assert working_memory.memory_type == MemoryType.WORKING
        assert working_memory.state is not None
        assert len(working_memory.state.conversation) == 0

    def test_init_with_max_turns(self):
        """Test initialization with custom max turns."""
        memory = WorkingMemory(max_conversation_turns=5)
        assert memory._max_turns == 5

    def test_add_turn(self, working_memory):
        """Test adding a conversation turn."""
        turn = working_memory.add_turn("user", "Hello, how are you?")

        assert isinstance(turn, ConversationTurn)
        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"
        assert len(working_memory.state.conversation) == 1

    def test_add_turn_with_metadata(self, working_memory):
        """Test adding a turn with metadata."""
        metadata = {"intent": "greeting", "confidence": 0.95}
        turn = working_memory.add_turn("user", "Hi there", metadata=metadata)

        assert turn.metadata == metadata

    def test_add_multiple_turns(self, working_memory):
        """Test adding multiple turns."""
        working_memory.add_turn("user", "Question 1")
        working_memory.add_turn("assistant", "Answer 1")
        working_memory.add_turn("user", "Question 2")
        working_memory.add_turn("assistant", "Answer 2")

        assert len(working_memory.state.conversation) == 4

    def test_turn_trimming(self):
        """Test that old turns are trimmed when capacity is exceeded."""
        memory = WorkingMemory(max_conversation_turns=3)

        memory.add_turn("user", "Message 1")
        memory.add_turn("assistant", "Response 1")
        memory.add_turn("user", "Message 2")
        memory.add_turn("assistant", "Response 2")  # This should trim the first

        assert len(memory.state.conversation) == 3
        assert memory.state.conversation[0].content == "Response 1"


class TestWorkingMemoryTask:
    """Test task management in WorkingMemory."""

    def test_set_task(self, working_memory):
        """Test setting a task."""
        working_memory.set_task("Implement authentication")

        assert working_memory.state.current_task == "Implement authentication"

    def test_set_task_with_context(self, working_memory):
        """Test setting a task with context."""
        context = {"project": "myapp", "priority": "high"}
        working_memory.set_task("Fix bug", context=context)

        assert working_memory.state.current_task == "Fix bug"
        assert working_memory.state.task_context == context

    def test_clear_task(self, working_memory):
        """Test clearing a task."""
        working_memory.set_task("Some task", context={"key": "value"})
        working_memory.clear_task()

        assert working_memory.state.current_task is None
        assert working_memory.state.task_context == {}


class TestWorkingMemoryFiles:
    """Test active files management."""

    def test_add_active_file(self, working_memory):
        """Test adding an active file."""
        working_memory.add_active_file("src/main.py")

        assert "src/main.py" in working_memory.state.active_files

    def test_add_duplicate_file(self, working_memory):
        """Test that duplicate files are not added."""
        working_memory.add_active_file("src/main.py")
        working_memory.add_active_file("src/main.py")

        assert working_memory.state.active_files.count("src/main.py") == 1

    def test_remove_active_file(self, working_memory):
        """Test removing an active file."""
        working_memory.add_active_file("src/main.py")
        working_memory.add_active_file("src/utils.py")
        working_memory.remove_active_file("src/main.py")

        assert "src/main.py" not in working_memory.state.active_files
        assert "src/utils.py" in working_memory.state.active_files

    def test_active_files_limit(self):
        """Test that active files list is bounded."""
        memory = WorkingMemory()

        # Add more than 20 files
        for i in range(25):
            memory.add_active_file(f"file_{i}.py")

        assert len(memory.state.active_files) == 20
        # First files should have been removed
        assert "file_0.py" not in memory.state.active_files
        assert "file_24.py" in memory.state.active_files


class TestWorkingMemoryAttention:
    """Test attention markers."""

    def test_add_attention(self, working_memory):
        """Test adding attention marker."""
        working_memory.add_attention("User prefers TypeScript")

        assert "User prefers TypeScript" in working_memory.state.attention

    def test_add_duplicate_attention(self, working_memory):
        """Test that duplicate attention markers are not added."""
        working_memory.add_attention("Important note")
        working_memory.add_attention("Important note")

        assert working_memory.state.attention.count("Important note") == 1

    def test_clear_attention(self, working_memory):
        """Test clearing attention markers."""
        working_memory.add_attention("Note 1")
        working_memory.add_attention("Note 2")
        working_memory.clear_attention()

        assert len(working_memory.state.attention) == 0

    def test_attention_limit(self):
        """Test that attention list is bounded to 10."""
        memory = WorkingMemory()

        for i in range(15):
            memory.add_attention(f"Attention item {i}")

        assert len(memory.state.attention) == 10


class TestWorkingMemoryScratchpad:
    """Test scratchpad functionality."""

    def test_set_scratchpad(self, working_memory):
        """Test setting scratchpad values."""
        working_memory.set_scratchpad("intermediate_result", {"count": 42})

        assert working_memory.get_scratchpad("intermediate_result") == {"count": 42}

    def test_get_scratchpad_default(self, working_memory):
        """Test getting non-existent scratchpad key with default."""
        result = working_memory.get_scratchpad("nonexistent", default="fallback")

        assert result == "fallback"

    def test_get_scratchpad_no_default(self, working_memory):
        """Test getting non-existent scratchpad key without default."""
        result = working_memory.get_scratchpad("nonexistent")

        assert result is None


class TestWorkingMemoryRetrieval:
    """Test context retrieval methods."""

    def test_get_recent_context(self, working_memory):
        """Test getting recent context."""
        for i in range(10):
            working_memory.add_turn("user", f"Message {i}")

        recent = working_memory.get_recent_context(n_turns=3)

        assert len(recent) == 3
        assert recent[-1].content == "Message 9"

    def test_get_recent_context_fewer_turns(self, working_memory):
        """Test getting recent context with fewer turns than requested."""
        working_memory.add_turn("user", "Only message")

        recent = working_memory.get_recent_context(n_turns=10)

        assert len(recent) == 1

    def test_get_conversation_for_llm(self, working_memory):
        """Test getting conversation formatted for LLM."""
        working_memory.add_turn("user", "Hello")
        working_memory.add_turn("assistant", "Hi there!")

        llm_format = working_memory.get_conversation_for_llm()

        assert isinstance(llm_format, list)
        assert llm_format[0] == {"role": "user", "content": "Hello"}
        assert llm_format[1] == {"role": "assistant", "content": "Hi there!"}


class TestWorkingMemoryUtilities:
    """Test utility methods."""

    def test_should_consolidate_false(self, working_memory):
        """Test should_consolidate returns False when below threshold."""
        working_memory.add_turn("user", "Message")

        assert working_memory.should_consolidate() is False

    def test_should_consolidate_true(self):
        """Test should_consolidate returns True when at threshold."""
        memory = WorkingMemory(max_conversation_turns=10)

        for i in range(8):  # 80% of capacity
            memory.add_turn("user", f"Message {i}")

        assert memory.should_consolidate() is True

    def test_get_token_estimate(self, working_memory):
        """Test token estimation."""
        working_memory.add_turn("user", "This is a test message")  # 22 chars
        working_memory.set_task("Task description")  # 16 chars

        estimate = working_memory.get_token_estimate()

        # (22 + 16) / 4 = 9 (integer division)
        assert estimate == 9

    def test_to_dict(self, working_memory):
        """Test exporting to dictionary."""
        working_memory.add_turn("user", "Test message")
        working_memory.set_task("Test task")
        working_memory.add_active_file("test.py")

        data = working_memory.to_dict()

        assert "session_id" in data
        assert "conversation" in data
        assert "current_task" in data
        assert data["current_task"] == "Test task"
        assert len(data["conversation"]) == 1

    def test_summary(self, working_memory):
        """Test summary generation."""
        working_memory.add_turn("user", "Message")
        working_memory.set_task("Important task")

        summary = working_memory.summary()

        assert "Session:" in summary
        assert "Turns:" in summary
        assert "Task:" in summary


class TestWorkingMemoryPersistence:
    """Test persistence functionality."""

    def test_persistence_save_and_load(self, temp_file):
        """Test saving and loading working memory state."""
        # Create and populate memory
        memory1 = WorkingMemory(persist_path=temp_file)
        memory1.add_turn("user", "Hello")
        memory1.add_turn("assistant", "Hi there")
        memory1.set_task("Test task", {"key": "value"})
        memory1.add_active_file("test.py")

        # Create new instance that loads from same path
        memory2 = WorkingMemory(persist_path=temp_file)

        assert len(memory2.state.conversation) == 2
        assert memory2.state.current_task == "Test task"
        assert memory2.state.task_context == {"key": "value"}
        assert "test.py" in memory2.state.active_files

    def test_persistence_file_created(self, temp_file):
        """Test that persistence file is created."""
        memory = WorkingMemory(persist_path=temp_file)
        memory.add_turn("user", "Test")

        assert Path(temp_file).exists()

    def test_no_persistence_by_default(self, working_memory, temp_dir):
        """Test that no persistence happens by default."""
        working_memory.add_turn("user", "Test")

        # Check no files were created
        files = list(Path(temp_dir).glob("*"))
        assert len(files) == 0


class TestWorkingMemoryAsync:
    """Test async methods."""

    @pytest.mark.asyncio
    async def test_store(self, working_memory):
        """Test async store method."""
        new_state = WorkingMemoryState(id="new_state")
        result = await working_memory.store(new_state)

        assert result == "new_state"
        assert working_memory.state.id == "new_state"

    @pytest.mark.asyncio
    async def test_retrieve(self, working_memory):
        """Test async retrieve method."""
        working_memory.add_turn("user", "Test")

        from engram_memory import RetrievalQuery
        results = await working_memory.retrieve(RetrievalQuery(text="anything"))

        assert len(results) == 1
        assert results[0].relevance_score == 1.0
        assert results[0].memory_type == MemoryType.WORKING

    @pytest.mark.asyncio
    async def test_update(self, working_memory):
        """Test async update method."""
        result = await working_memory.update(
            "working_memory",
            {"current_task": "Updated task"}
        )

        assert result is True
        assert working_memory.state.current_task == "Updated task"

    @pytest.mark.asyncio
    async def test_delete(self, working_memory):
        """Test async delete (clear) method."""
        working_memory.add_turn("user", "Test")
        result = await working_memory.delete("working_memory")

        assert result is True
        assert len(working_memory.state.conversation) == 0

    @pytest.mark.asyncio
    async def test_clear(self, working_memory):
        """Test async clear method."""
        working_memory.add_turn("user", "Test")
        working_memory.set_task("Task")

        await working_memory.clear()

        assert len(working_memory.state.conversation) == 0
        assert working_memory.state.current_task is None
