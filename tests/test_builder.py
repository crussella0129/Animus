"""Tests for BuilderQuery interface."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from src.core.builder import (
    BuilderQuery,
    AnalysisResult,
    Suggestion,
    SuggestionPriority,
    SuggestionCategory,
)
from src.core.run import Run, RunStatus, RunMetrics, RunStore
from src.core.decision import Decision, DecisionType, Option


class TestSuggestion:
    """Test Suggestion dataclass."""

    def test_create_suggestion(self):
        """Test creating a suggestion."""
        suggestion = Suggestion(
            id="sug_0001",
            category=SuggestionCategory.ERROR_PATTERN,
            priority=SuggestionPriority.HIGH,
            title="Test suggestion",
            description="A test suggestion",
            evidence=["Evidence 1", "Evidence 2"],
            affected_runs=["run_1", "run_2"],
            suggested_actions=["Action 1", "Action 2"],
        )

        assert suggestion.id == "sug_0001"
        assert suggestion.category == SuggestionCategory.ERROR_PATTERN
        assert suggestion.priority == SuggestionPriority.HIGH
        assert len(suggestion.evidence) == 2
        assert len(suggestion.affected_runs) == 2

    def test_suggestion_to_dict(self):
        """Test suggestion serialization."""
        suggestion = Suggestion(
            id="sug_0001",
            category=SuggestionCategory.PERFORMANCE,
            priority=SuggestionPriority.MEDIUM,
            title="Performance issue",
            description="Description",
        )

        data = suggestion.to_dict()
        assert data["id"] == "sug_0001"
        assert data["category"] == "performance"
        assert data["priority"] == "medium"


class TestAnalysisResult:
    """Test AnalysisResult dataclass."""

    def test_create_result(self):
        """Test creating an analysis result."""
        result = AnalysisResult(
            total_runs=100,
            analyzed_runs=50,
            date_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
            suggestions=[],
            patterns={"key": "value"},
            summary="Test summary",
        )

        assert result.total_runs == 100
        assert result.analyzed_runs == 50
        assert result.summary == "Test summary"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = AnalysisResult(
            total_runs=10,
            analyzed_runs=10,
            date_range=(None, None),
            suggestions=[],
            patterns={},
            summary="Summary",
        )

        data = result.to_dict()
        assert data["total_runs"] == 10
        assert data["date_range"] == [None, None]


class TestBuilderQuery:
    """Test BuilderQuery class."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary RunStore."""
        with TemporaryDirectory() as tmpdir:
            yield RunStore(Path(tmpdir))

    @pytest.fixture
    def builder(self, temp_store):
        """Create BuilderQuery with temp store."""
        return BuilderQuery(temp_store)

    def _create_run(
        self,
        goal: str = "Test goal",
        status: RunStatus = RunStatus.COMPLETED,
        tokens: int = 1000,
        duration_ms: float = 5000,
        tool_calls: int = 5,
        tool_successes: int = 4,
        errors: list = None,
        decisions: list = None,
    ) -> Run:
        """Helper to create a test run."""
        run = Run.create(goal=goal)
        run.status = status
        run.started_at = datetime.now() - timedelta(hours=1)
        run.ended_at = datetime.now()
        run.metrics = RunMetrics(
            tokens_used=tokens,
            tokens_input=tokens // 2,
            tokens_output=tokens // 2,
            latency_ms=duration_ms,
            turn_count=5,
            tool_calls=tool_calls,
            tool_successes=tool_successes,
            tool_failures=tool_calls - tool_successes,
            errors=errors or [],
        )
        run.metrics.update_success_rate()
        run.decisions = decisions or []
        return run

    def test_analyze_no_runs(self, builder):
        """Test analysis with no runs."""
        result = builder.analyze()

        assert result.total_runs == 0
        assert result.analyzed_runs == 0
        assert result.suggestions == []
        assert "No runs found" in result.summary

    def test_analyze_single_run(self, builder, temp_store):
        """Test analysis with a single successful run."""
        run = self._create_run()
        temp_store.save(run)

        result = builder.analyze()

        assert result.total_runs == 1
        assert result.analyzed_runs == 1
        assert result.patterns["status_distribution"]["completed"] == 1

    def test_analyze_with_failures(self, builder, temp_store):
        """Test analysis detects high failure rate."""
        # Create mostly failed runs
        for i in range(7):
            run = self._create_run(
                goal=f"Failed task {i}",
                status=RunStatus.FAILED,
                errors=["Some error"],
            )
            temp_store.save(run)

        for i in range(3):
            run = self._create_run(goal=f"Success {i}")
            temp_store.save(run)

        result = builder.analyze()

        assert result.analyzed_runs == 10

        # Should have critical suggestion about failure rate
        critical = [s for s in result.suggestions if s.priority == SuggestionPriority.CRITICAL]
        assert len(critical) > 0
        assert "failure rate" in critical[0].title.lower()

    def test_analyze_error_patterns(self, builder, temp_store):
        """Test detection of recurring error patterns."""
        # Create runs with timeout errors
        for i in range(5):
            run = self._create_run(
                goal=f"Timeout task {i}",
                status=RunStatus.FAILED,
                errors=["Operation timeout after 30 seconds"],
            )
            temp_store.save(run)

        result = builder.analyze()

        # Should have suggestion about timeout errors
        timeout_suggestions = [
            s for s in result.suggestions
            if "timeout" in s.title.lower()
        ]
        assert len(timeout_suggestions) > 0

    def test_analyze_rate_limit_pattern(self, builder, temp_store):
        """Test detection of rate limit errors."""
        for i in range(3):
            run = self._create_run(
                goal=f"API task {i}",
                status=RunStatus.FAILED,
                errors=["Rate limit exceeded (429)"],
            )
            temp_store.save(run)

        result = builder.analyze()

        rate_suggestions = [
            s for s in result.suggestions
            if "rate limit" in s.title.lower()
        ]
        assert len(rate_suggestions) > 0

    def test_analyze_high_token_usage(self, builder, temp_store):
        """Test detection of high token usage."""
        for i in range(5):
            run = self._create_run(
                goal=f"Large task {i}",
                tokens=100000,  # Very high token usage
            )
            temp_store.save(run)

        result = builder.analyze()

        token_suggestions = [
            s for s in result.suggestions
            if s.category == SuggestionCategory.RESOURCE
        ]
        assert len(token_suggestions) > 0

    def test_analyze_low_tool_success(self, builder, temp_store):
        """Test detection of low tool success rate."""
        for i in range(5):
            run = self._create_run(
                goal=f"Tool task {i}",
                tool_calls=10,
                tool_successes=3,  # 30% success rate
            )
            temp_store.save(run)

        result = builder.analyze()

        tool_suggestions = [
            s for s in result.suggestions
            if s.category == SuggestionCategory.TOOL_USAGE
        ]
        assert len(tool_suggestions) > 0

    def test_analyze_goal_filter(self, builder, temp_store):
        """Test filtering by goal."""
        # Create runs with different goals
        for i in range(3):
            run = self._create_run(goal="Database migration task")
            temp_store.save(run)

        for i in range(3):
            run = self._create_run(goal="API development task")
            temp_store.save(run)

        result = builder.analyze(goal_filter="Database")

        assert result.analyzed_runs == 3
        assert result.total_runs == 6

    def test_analyze_days_filter(self, builder, temp_store):
        """Test filtering by days."""
        # Create recent run
        recent = self._create_run(goal="Recent task")
        temp_store.save(recent)

        # Create old run (manually set old date)
        old = self._create_run(goal="Old task")
        old.started_at = datetime.now() - timedelta(days=60)
        temp_store.save(old)

        result = builder.analyze(days=30)

        assert result.analyzed_runs == 1

    def test_analyze_limit(self, builder, temp_store):
        """Test limiting number of runs analyzed."""
        for i in range(10):
            run = self._create_run(goal=f"Task {i}")
            temp_store.save(run)

        result = builder.analyze(limit=5)

        assert result.analyzed_runs == 5
        assert result.total_runs == 10

    def test_get_run_details(self, builder, temp_store):
        """Test getting details for a specific run."""
        run = self._create_run(goal="Detailed task")
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Select a tool",
            context="Some context",
            options=[
                Option.create("Option A"),
                Option.create("Option B"),
            ],
            chosen_option_id=None,
            reasoning="Chose A because...",
        )
        run.decisions = [decision]
        temp_store.save(run)

        details = builder.get_run_details(run.id)

        assert details is not None
        assert details["goal"] == "Detailed task"
        assert details["decision_count"] == 1
        assert len(details["decisions"]) == 1

    def test_get_run_details_not_found(self, builder):
        """Test getting details for non-existent run."""
        details = builder.get_run_details("nonexistent")
        assert details is None

    def test_compare_runs(self, builder, temp_store):
        """Test comparing multiple runs."""
        run1 = self._create_run(goal="Fast run", duration_ms=1000, tokens=500)
        run2 = self._create_run(goal="Slow run", duration_ms=10000, tokens=5000)
        temp_store.save(run1)
        temp_store.save(run2)

        comparison = builder.compare_runs([run1.id, run2.id])

        assert len(comparison["runs"]) == 2
        assert "most_efficient" in comparison["analysis"]
        assert comparison["analysis"]["most_efficient"]["by_tokens"] == run1.id

    def test_compare_runs_insufficient(self, builder, temp_store):
        """Test comparison with insufficient runs."""
        run = self._create_run()
        temp_store.save(run)

        comparison = builder.compare_runs([run.id])

        assert "error" in comparison

    def test_get_trends(self, builder, temp_store):
        """Test trend analysis."""
        # Create runs over several days
        for i in range(10):
            run = self._create_run(goal=f"Task day {i}")
            run.started_at = datetime.now() - timedelta(days=i)
            temp_store.save(run)

        trends = builder.get_trends(days=30)

        assert trends["total_runs"] == 10
        assert len(trends["daily_stats"]) > 0
        assert "overall_trend" in trends

    def test_get_trends_no_runs(self, builder):
        """Test trends with no runs."""
        trends = builder.get_trends()
        assert "error" in trends

    def test_error_categorization(self, builder):
        """Test error message categorization."""
        assert builder._categorize_error("Connection timeout after 30s") == "timeout"
        assert builder._categorize_error("Rate limit exceeded (429)") == "rate_limit"
        assert builder._categorize_error("Authentication failed 401") == "authentication"
        assert builder._categorize_error("File not found 404") == "not_found"
        assert builder._categorize_error("Permission denied") == "permission_denied"
        assert builder._categorize_error("Out of memory error") == "memory"
        assert builder._categorize_error("Network connection failed") == "network"
        assert builder._categorize_error("Syntax error in code") == "syntax"
        assert builder._categorize_error("Import error: no module") == "import"
        assert builder._categorize_error("Unknown error occurred") == "other"

    def test_suggestion_priorities_sorted(self, builder, temp_store):
        """Test that suggestions are sorted by priority."""
        # Create runs with various issues
        for i in range(5):
            run = self._create_run(
                goal=f"Failed {i}",
                status=RunStatus.FAILED,
                errors=["timeout error"],
                tool_calls=10,
                tool_successes=3,
            )
            temp_store.save(run)

        result = builder.analyze()

        if len(result.suggestions) >= 2:
            priorities = [s.priority for s in result.suggestions]
            priority_values = {
                SuggestionPriority.CRITICAL: 0,
                SuggestionPriority.HIGH: 1,
                SuggestionPriority.MEDIUM: 2,
                SuggestionPriority.LOW: 3,
            }
            values = [priority_values[p] for p in priorities]
            assert values == sorted(values), "Suggestions should be sorted by priority"


class TestSuggestionPriority:
    """Test priority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert SuggestionPriority.CRITICAL.value == "critical"
        assert SuggestionPriority.HIGH.value == "high"
        assert SuggestionPriority.MEDIUM.value == "medium"
        assert SuggestionPriority.LOW.value == "low"


class TestSuggestionCategory:
    """Test category enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert SuggestionCategory.ERROR_PATTERN.value == "error_pattern"
        assert SuggestionCategory.PERFORMANCE.value == "performance"
        assert SuggestionCategory.TOOL_USAGE.value == "tool_usage"
        assert SuggestionCategory.STRATEGY.value == "strategy"
        assert SuggestionCategory.RESOURCE.value == "resource"
        assert SuggestionCategory.SUCCESS_RATE.value == "success_rate"
