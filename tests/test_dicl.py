"""Tests for Dynamic In-Context Learning (DICL) module."""

import json
import pytest
import tempfile
from pathlib import Path

from src.core.dicl import (
    ToolCallExample,
    DICLHit,
    DICLStore,
    create_example_from_turn,
    DEFAULT_MAX_EXAMPLES,
)


# =============================================================================
# ToolCallExample Tests
# =============================================================================

class TestToolCallExample:
    """Test ToolCallExample dataclass."""

    def test_basic_creation(self):
        """ToolCallExample should store all fields."""
        example = ToolCallExample(
            task="List files in src",
            tool_calls=[{"name": "list_dir", "arguments": {"path": "src"}}],
            tool_results=[["main.py", "utils.py"]],
            response="The src directory contains main.py and utils.py.",
        )

        assert example.task == "List files in src"
        assert len(example.tool_calls) == 1
        assert example.tool_calls[0]["name"] == "list_dir"
        assert example.response.startswith("The src")
        assert example.success is True

    def test_to_dict_serialization(self):
        """ToolCallExample should serialize to dict."""
        example = ToolCallExample(
            task="Test task",
            tool_calls=[{"name": "test_tool"}],
            tool_results=[{"result": "ok"}],
            response="Done",
            tags=["test"],
        )

        d = example.to_dict()

        assert d["task"] == "Test task"
        assert d["tool_calls"] == [{"name": "test_tool"}]
        assert d["tags"] == ["test"]
        assert "timestamp" in d

    def test_from_dict_deserialization(self):
        """ToolCallExample should deserialize from dict."""
        data = {
            "task": "Read a file",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "test.py"}}],
            "tool_results": ["file contents"],
            "response": "File contents: ...",
            "tags": ["filesystem"],
            "success": True,
            "timestamp": 1234567890.0,
        }

        example = ToolCallExample.from_dict(data)

        assert example.task == "Read a file"
        assert example.tool_calls[0]["name"] == "read_file"
        assert example.tags == ["filesystem"]

    def test_from_dict_ignores_extra_fields(self):
        """from_dict should ignore unknown fields."""
        data = {
            "task": "Test",
            "tool_calls": [],
            "tool_results": [],
            "response": "Done",
            "unknown_field": "should be ignored",
        }

        example = ToolCallExample.from_dict(data)
        assert example.task == "Test"
        assert not hasattr(example, "unknown_field")


class TestToolCallExampleMatching:
    """Test ToolCallExample.matches() scoring."""

    def test_matches_exact_task(self):
        """Exact match in task should score high."""
        example = ToolCallExample(
            task="List files in the src directory",
            tool_calls=[{"name": "list_dir"}],
            tool_results=[],
            response="Files listed.",
        )

        score = example.matches("list files src directory")
        assert score > 0.5

    def test_matches_tool_name(self):
        """Match on tool name should contribute to score."""
        example = ToolCallExample(
            task="Read configuration",
            tool_calls=[{"name": "read_file"}, {"name": "parse_json"}],
            tool_results=[],
            response="Config loaded.",
        )

        score = example.matches("read_file parse")
        assert score > 0.3

    def test_matches_tags(self):
        """Match on tags should contribute to score."""
        example = ToolCallExample(
            task="Check status",
            tool_calls=[],
            tool_results=[],
            response="Status OK",
            tags=["health-check", "monitoring"],
        )

        score = example.matches("health monitoring")
        assert score > 0.3

    def test_no_match_returns_zero(self):
        """No matching terms should return 0."""
        example = ToolCallExample(
            task="List files",
            tool_calls=[{"name": "list_dir"}],
            tool_results=[],
            response="Files listed.",
        )

        score = example.matches("database connection query")
        assert score == 0.0

    def test_empty_query_returns_zero(self):
        """Empty query should return 0."""
        example = ToolCallExample(
            task="Test",
            tool_calls=[],
            tool_results=[],
            response="Done",
        )

        assert example.matches("") == 0.0
        assert example.matches("   ") == 0.0


# =============================================================================
# DICLStore Tests
# =============================================================================

class TestDICLStore:
    """Test DICLStore functionality."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary DICL store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dicl.jsonl"
            yield DICLStore(path=path)

    def test_record_and_count(self, temp_store):
        """Store should record examples and count correctly."""
        example = ToolCallExample(
            task="Test task",
            tool_calls=[{"name": "test"}],
            tool_results=["result"],
            response="Done",
        )

        temp_store.record(example)

        assert temp_store.count() == 1

    def test_record_multiple(self, temp_store):
        """Store should handle multiple examples."""
        for i in range(5):
            example = ToolCallExample(
                task=f"Task {i}",
                tool_calls=[{"name": f"tool_{i}"}],
                tool_results=[],
                response=f"Response {i}",
            )
            temp_store.record(example)

        assert temp_store.count() == 5

    def test_record_skips_failed(self, temp_store):
        """Store should skip examples with success=False."""
        example = ToolCallExample(
            task="Failed task",
            tool_calls=[],
            tool_results=[],
            response="Error",
            success=False,
        )

        temp_store.record(example)

        assert temp_store.count() == 0

    def test_get_all(self, temp_store):
        """get_all should return all stored examples."""
        for i in range(3):
            temp_store.record(ToolCallExample(
                task=f"Task {i}",
                tool_calls=[],
                tool_results=[],
                response=f"Response {i}",
            ))

        examples = temp_store.get_all()

        assert len(examples) == 3
        assert examples[0].task == "Task 0"

    def test_clear(self, temp_store):
        """clear should remove all examples."""
        temp_store.record(ToolCallExample(
            task="Test",
            tool_calls=[],
            tool_results=[],
            response="Done",
        ))
        assert temp_store.count() == 1

        temp_store.clear()

        assert temp_store.count() == 0

    def test_persistence(self, temp_store):
        """Examples should persist to disk."""
        temp_store.record(ToolCallExample(
            task="Persistent task",
            tool_calls=[{"name": "persist"}],
            tool_results=["ok"],
            response="Persisted",
        ))

        # Create new store pointing to same path
        new_store = DICLStore(path=temp_store.path)

        assert new_store.count() == 1
        examples = new_store.get_all()
        assert examples[0].task == "Persistent task"


class TestDICLStoreSearch:
    """Test DICLStore search functionality."""

    @pytest.fixture
    def populated_store(self):
        """Create a store with sample examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dicl.jsonl"
            store = DICLStore(path=path)

            # Add diverse examples
            examples = [
                ToolCallExample(
                    task="List files in the src directory",
                    tool_calls=[{"name": "list_dir", "arguments": {"path": "src"}}],
                    tool_results=[["main.py", "utils.py"]],
                    response="Found main.py and utils.py",
                    tags=["filesystem", "listing"],
                ),
                ToolCallExample(
                    task="Read the configuration file",
                    tool_calls=[{"name": "read_file", "arguments": {"path": "config.yaml"}}],
                    tool_results=["key: value"],
                    response="Config contains key: value",
                    tags=["filesystem", "config"],
                ),
                ToolCallExample(
                    task="Run the test suite",
                    tool_calls=[{"name": "run_shell", "arguments": {"command": "pytest"}}],
                    tool_results=["5 tests passed"],
                    response="All 5 tests passed",
                    tags=["testing", "pytest"],
                ),
                ToolCallExample(
                    task="Create a new Python file",
                    tool_calls=[{"name": "write_file", "arguments": {"path": "new.py"}}],
                    tool_results=["created"],
                    response="Created new.py",
                    tags=["filesystem", "writing"],
                ),
            ]

            for ex in examples:
                store.record(ex)

            yield store

    def test_search_finds_relevant(self, populated_store):
        """Search should find relevant examples."""
        hits = populated_store.search("list files directory")

        assert len(hits) > 0
        assert any("list_dir" in str(h.example.tool_calls) for h in hits)

    def test_search_respects_k(self, populated_store):
        """Search should limit results to k."""
        hits = populated_store.search("file", k=2)

        assert len(hits) <= 2

    def test_search_respects_min_score(self, populated_store):
        """Search should filter by min_score."""
        hits = populated_store.search("file", min_score=0.9)

        for hit in hits:
            assert hit.score >= 0.9

    def test_search_returns_scores(self, populated_store):
        """Search results should include relevance scores."""
        hits = populated_store.search("test pytest")

        assert len(hits) > 0
        assert all(isinstance(h.score, float) for h in hits)
        assert all(0 <= h.score <= 1.0 for h in hits)

    def test_search_sorted_by_score(self, populated_store):
        """Results should be sorted by score descending."""
        hits = populated_store.search("file", k=10)

        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_search_no_results(self, populated_store):
        """Search with no matches should return empty list."""
        hits = populated_store.search("database connection mongodb")

        assert len(hits) == 0


class TestDICLStoreFormatting:
    """Test DICLStore formatting functions."""

    @pytest.fixture
    def sample_hits(self):
        """Create sample search hits for formatting tests."""
        return [
            DICLHit(
                example=ToolCallExample(
                    task="List files in src",
                    tool_calls=[{"name": "list_dir", "arguments": {"path": "src"}}],
                    tool_results=[["main.py"]],
                    response="The src directory contains main.py.",
                ),
                score=0.8,
            ),
            DICLHit(
                example=ToolCallExample(
                    task="Read README",
                    tool_calls=[{"name": "read_file", "arguments": {"path": "README.md"}}],
                    tool_results=["# Project"],
                    response="README contains project documentation.",
                ),
                score=0.6,
            ),
        ]

    def test_format_few_shot_basic(self, sample_hits):
        """format_few_shot should produce readable output."""
        store = DICLStore()
        output = store.format_few_shot(sample_hits)

        assert "Examples of successful tool usage" in output
        assert "Example 1" in output
        assert "List files in src" in output
        assert "list_dir" in output

    def test_format_few_shot_includes_response(self, sample_hits):
        """format_few_shot should include agent response."""
        store = DICLStore()
        output = store.format_few_shot(sample_hits)

        assert "main.py" in output or "Response:" in output

    def test_format_few_shot_respects_max_chars(self, sample_hits):
        """format_few_shot should respect max_chars limit."""
        store = DICLStore()
        output = store.format_few_shot(sample_hits, max_chars=200)

        assert len(output) <= 300  # Some buffer for truncation

    def test_format_few_shot_empty_hits(self):
        """format_few_shot should handle empty hits."""
        store = DICLStore()
        output = store.format_few_shot([])

        assert output == ""

    def test_format_few_shot_with_results(self, sample_hits):
        """format_few_shot should optionally include tool results."""
        store = DICLStore()
        output = store.format_few_shot(sample_hits, include_results=True)

        assert "Results:" in output

    def test_format_messages_basic(self, sample_hits):
        """format_messages should produce message list."""
        store = DICLStore()
        messages = store.format_messages(sample_hits)

        assert len(messages) == 4  # 2 examples * (user + assistant)
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_format_messages_respects_max(self, sample_hits):
        """format_messages should respect max_examples."""
        store = DICLStore()
        messages = store.format_messages(sample_hits, max_examples=1)

        assert len(messages) == 2  # 1 example * (user + assistant)

    def test_format_messages_content(self, sample_hits):
        """format_messages should include task and tool calls."""
        store = DICLStore()
        messages = store.format_messages(sample_hits)

        user_msg = messages[0]
        assistant_msg = messages[1]

        assert "List files in src" in user_msg["content"]
        assert "list_dir" in assistant_msg["content"]


class TestDICLStoreStats:
    """Test DICLStore statistics."""

    def test_stats_empty(self):
        """Stats should work on empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DICLStore(path=Path(tmpdir) / "dicl.jsonl")
            stats = store.stats()

            assert stats["total_examples"] == 0
            assert stats["successful_examples"] == 0

    def test_stats_populated(self):
        """Stats should reflect stored examples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DICLStore(path=Path(tmpdir) / "dicl.jsonl")

            store.record(ToolCallExample(
                task="Test 1",
                tool_calls=[{"name": "tool_a"}, {"name": "tool_b"}],
                tool_results=[],
                response="Done",
                tags=["tag1", "tag2"],
            ))
            store.record(ToolCallExample(
                task="Test 2",
                tool_calls=[{"name": "tool_a"}],
                tool_results=[],
                response="Done",
                tags=["tag1"],
            ))

            stats = store.stats()

            assert stats["total_examples"] == 2
            assert stats["tool_usage"]["tool_a"] == 2
            assert stats["tool_usage"]["tool_b"] == 1
            assert stats["unique_tags"] == 2


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestCreateExampleFromTurn:
    """Test create_example_from_turn helper."""

    def test_basic_creation(self):
        """Should create example from turn data."""
        example = create_example_from_turn(
            user_message="List files",
            tool_calls=[{"name": "list_dir", "arguments": {"path": "."}}],
            tool_results=[["a.py", "b.py"]],
            response="Found a.py and b.py",
        )

        assert example.task == "List files"
        assert example.tool_calls[0]["name"] == "list_dir"
        assert example.success is True

    def test_auto_generates_tags(self):
        """Should auto-generate tags from tool names."""
        example = create_example_from_turn(
            user_message="Do stuff",
            tool_calls=[
                {"name": "read_file"},
                {"name": "write_file"},
            ],
            tool_results=[],
            response="Done",
        )

        assert "read_file" in example.tags
        assert "write_file" in example.tags

    def test_uses_provided_tags(self):
        """Should use provided tags over auto-generated."""
        example = create_example_from_turn(
            user_message="Do stuff",
            tool_calls=[{"name": "some_tool"}],
            tool_results=[],
            response="Done",
            tags=["custom-tag", "another"],
        )

        assert "custom-tag" in example.tags
        assert "another" in example.tags

    def test_includes_model(self):
        """Should include model name."""
        example = create_example_from_turn(
            user_message="Test",
            tool_calls=[],
            tool_results=[],
            response="Done",
            model="gpt-4",
        )

        assert example.model_used == "gpt-4"
