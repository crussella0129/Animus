"""Tests for run persistence module."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.core.run import Run, RunStatus, RunMetrics, RunStore


class TestRunMetrics:
    """Tests for RunMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = RunMetrics()
        assert metrics.tokens_used == 0
        assert metrics.turn_count == 0
        assert metrics.tool_calls == 0
        assert metrics.success_rate == 0.0

    def test_update_success_rate(self):
        """Test calculating success rate."""
        metrics = RunMetrics(tool_calls=10, tool_successes=8, tool_failures=2)
        metrics.update_success_rate()
        assert metrics.success_rate == 0.8

    def test_update_success_rate_zero_calls(self):
        """Test success rate with no tool calls."""
        metrics = RunMetrics()
        metrics.update_success_rate()
        assert metrics.success_rate == 0.0

    def test_to_dict_and_back(self):
        """Test serialization roundtrip."""
        metrics = RunMetrics(
            tokens_used=1000,
            turn_count=5,
            tool_calls=10,
            errors=["Error 1", "Error 2"],
        )
        data = metrics.to_dict()
        restored = RunMetrics.from_dict(data)
        assert restored.tokens_used == 1000
        assert restored.turn_count == 5
        assert len(restored.errors) == 2


class TestRun:
    """Tests for Run dataclass."""

    def test_create_run(self):
        """Test creating a run with factory method."""
        run = Run.create(
            goal="Implement feature X",
            model="llama-7b",
            provider="native",
            tags=["feature", "important"],
        )
        assert run.goal == "Implement feature X"
        assert run.status == RunStatus.PENDING
        assert run.model == "llama-7b"
        assert "feature" in run.tags

    def test_run_lifecycle(self):
        """Test run state transitions."""
        run = Run.create(goal="Test")
        assert run.status == RunStatus.PENDING

        run.start()
        assert run.status == RunStatus.RUNNING
        assert run.started_at is not None

        run.complete("Final output")
        assert run.status == RunStatus.COMPLETED
        assert run.ended_at is not None
        assert run.final_output == "Final output"

    def test_run_failure(self):
        """Test run failure handling."""
        run = Run.create(goal="Test")
        run.start()
        run.fail("Something went wrong")
        assert run.status == RunStatus.FAILED
        assert run.error == "Something went wrong"
        assert "Something went wrong" in run.metrics.errors

    def test_run_cancel(self):
        """Test run cancellation."""
        run = Run.create(goal="Test")
        run.start()
        run.cancel()
        assert run.status == RunStatus.CANCELLED
        assert run.ended_at is not None

    def test_record_tool_call(self):
        """Test recording tool calls."""
        run = Run.create(goal="Test")
        run.record_tool_call(success=True)
        run.record_tool_call(success=True)
        run.record_tool_call(success=False, error="Failed")

        assert run.metrics.tool_calls == 3
        assert run.metrics.tool_successes == 2
        assert run.metrics.tool_failures == 1
        assert "Failed" in run.metrics.errors

    def test_add_tokens(self):
        """Test adding token usage."""
        run = Run.create(goal="Test")
        run.add_tokens(100, 50)
        run.add_tokens(200, 100)

        assert run.metrics.tokens_input == 300
        assert run.metrics.tokens_output == 150
        assert run.metrics.tokens_used == 450

    def test_duration(self):
        """Test duration calculation."""
        run = Run.create(goal="Test")
        run.start()
        # Duration should be > 0 while running
        assert run.duration_ms >= 0

    def test_to_dict_and_back(self):
        """Test serialization roundtrip."""
        run = Run.create(goal="Test goal", model="test-model")
        run.start()
        run.add_tokens(100, 50)
        run.record_tool_call(success=True)
        run.complete("Done")

        data = run.to_dict()
        restored = Run.from_dict(data)

        assert restored.id == run.id
        assert restored.goal == run.goal
        assert restored.status == RunStatus.COMPLETED
        assert restored.metrics.tokens_used == 150


class TestRunStore:
    """Tests for RunStore class."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary run store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore(base_path=Path(tmpdir))
            yield store

    def test_save_and_load(self, temp_store):
        """Test saving and loading a run."""
        run = Run.create(goal="Test")
        run.start()
        temp_store.save(run)

        loaded = temp_store.load(run.id)
        assert loaded is not None
        assert loaded.id == run.id
        assert loaded.goal == "Test"

    def test_load_nonexistent(self, temp_store):
        """Test loading a nonexistent run."""
        loaded = temp_store.load("nonexistent-id")
        assert loaded is None

    def test_delete(self, temp_store):
        """Test deleting a run."""
        run = Run.create(goal="Test")
        temp_store.save(run)
        assert temp_store.delete(run.id)
        assert temp_store.load(run.id) is None

    def test_list_all(self, temp_store):
        """Test listing all runs."""
        for i in range(3):
            run = Run.create(goal=f"Test {i}")
            temp_store.save(run)

        all_ids = temp_store.list_all()
        assert len(all_ids) == 3

    def test_find_by_status(self, temp_store):
        """Test finding runs by status."""
        run1 = Run.create(goal="Running")
        run1.start()
        temp_store.save(run1)

        run2 = Run.create(goal="Completed")
        run2.start()
        run2.complete("Done")
        temp_store.save(run2)

        running = temp_store.find_by_status(RunStatus.RUNNING)
        assert len(running) == 1
        assert running[0].goal == "Running"

        completed = temp_store.find_by_status(RunStatus.COMPLETED)
        assert len(completed) == 1

    def test_find_by_goal(self, temp_store):
        """Test finding runs by goal."""
        temp_store.save(Run.create(goal="Implement feature X"))
        temp_store.save(Run.create(goal="Fix bug Y"))
        temp_store.save(Run.create(goal="Implement feature Z"))

        results = temp_store.find_by_goal("Implement")
        assert len(results) == 2

        results = temp_store.find_by_goal("bug")
        assert len(results) == 1

    def test_find_by_tag(self, temp_store):
        """Test finding runs by tag."""
        temp_store.save(Run.create(goal="Test 1", tags=["urgent"]))
        temp_store.save(Run.create(goal="Test 2", tags=["normal"]))
        temp_store.save(Run.create(goal="Test 3", tags=["urgent"]))

        urgent = temp_store.find_by_tag("urgent")
        assert len(urgent) == 2

    def test_get_recent(self, temp_store):
        """Test getting recent runs."""
        for i in range(5):
            run = Run.create(goal=f"Test {i}")
            run.start()
            temp_store.save(run)

        recent = temp_store.get_recent(limit=3)
        assert len(recent) == 3

    def test_get_stats(self, temp_store):
        """Test getting aggregate statistics."""
        for i in range(3):
            run = Run.create(goal=f"Test {i}")
            run.start()
            run.add_tokens(100, 50)
            run.record_tool_call(success=True)
            run.complete("Done")
            temp_store.save(run)

        stats = temp_store.get_stats()
        assert stats["total_runs"] == 3
        assert stats["by_status"]["completed"] == 3
        assert stats["avg_tokens"] == 150
        assert stats["overall_success_rate"] == 1.0

    def test_get_stats_empty(self, temp_store):
        """Test getting stats with no runs."""
        stats = temp_store.get_stats()
        assert stats["total_runs"] == 0
