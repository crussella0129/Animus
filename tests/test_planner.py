"""Tests for the planning module."""

import pytest
from datetime import datetime

from src.core.planner import (
    Planner,
    ExecutionPlan,
    PlanStep,
    StepStatus,
)


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_create_step(self):
        """Test creating a step with factory method."""
        step = PlanStep.create(
            description="Read the config file",
            reasoning="Need to understand current settings",
            tool_hints=["read_file"],
            estimated_complexity="low",
        )

        assert step.id  # Should have generated ID
        assert step.description == "Read the config file"
        assert step.reasoning == "Need to understand current settings"
        assert step.tool_hints == ["read_file"]
        assert step.estimated_complexity == "low"
        assert step.status == StepStatus.PENDING
        assert step.dependencies == []

    def test_step_to_dict(self):
        """Test serialization to dict."""
        step = PlanStep.create(
            description="Test step",
            reasoning="For testing",
        )
        step.status = StepStatus.COMPLETED
        step.output = "Step completed successfully"

        d = step.to_dict()

        assert d["id"] == step.id
        assert d["description"] == "Test step"
        assert d["status"] == "completed"
        assert d["output"] == "Step completed successfully"

    def test_step_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "abc123",
            "description": "Test step",
            "reasoning": "Testing",
            "dependencies": ["other123"],
            "status": "in_progress",
            "tool_hints": ["read_file", "write_file"],
            "estimated_complexity": "high",
            "output": None,
            "error": None,
            "started_at": "2024-01-01T10:00:00",
            "completed_at": None,
            "metadata": {},
        }

        step = PlanStep.from_dict(data)

        assert step.id == "abc123"
        assert step.description == "Test step"
        assert step.status == StepStatus.IN_PROGRESS
        assert step.dependencies == ["other123"]
        assert step.started_at is not None


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_create_plan(self):
        """Test creating a plan."""
        plan = ExecutionPlan.create(
            goal="Implement a new feature",
            summary="Add user authentication",
        )

        assert plan.id
        assert plan.goal == "Implement a new feature"
        assert plan.summary == "Add user authentication"
        assert plan.steps == []
        assert plan.revision_count == 0

    def test_add_step(self):
        """Test adding steps to plan."""
        plan = ExecutionPlan.create(goal="Test")

        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2", dependencies=[step1.id])

        plan.add_step(step1)
        plan.add_step(step2)

        assert len(plan.steps) == 2
        assert plan.steps[0].id == step1.id
        assert plan.steps[1].dependencies == [step1.id]

    def test_get_ready_steps_no_deps(self):
        """Test getting ready steps with no dependencies."""
        plan = ExecutionPlan.create(goal="Test")

        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2")

        plan.add_step(step1)
        plan.add_step(step2)

        ready = plan.get_ready_steps()

        assert len(ready) == 2  # Both should be ready

    def test_get_ready_steps_with_deps(self):
        """Test getting ready steps with dependencies."""
        plan = ExecutionPlan.create(goal="Test")

        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2", dependencies=[step1.id])
        step3 = PlanStep.create(description="Step 3", dependencies=[step1.id])

        plan.add_step(step1)
        plan.add_step(step2)
        plan.add_step(step3)

        # Initially only step1 is ready
        ready = plan.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].id == step1.id

        # Complete step1
        plan.mark_step_completed(step1.id, "Done")

        # Now step2 and step3 should be ready
        ready = plan.get_ready_steps()
        assert len(ready) == 2

    def test_get_blocked_steps(self):
        """Test getting blocked steps."""
        plan = ExecutionPlan.create(goal="Test")

        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2", dependencies=[step1.id])

        plan.add_step(step1)
        plan.add_step(step2)

        blocked = plan.get_blocked_steps()

        assert len(blocked) == 1
        assert blocked[0].id == step2.id

    def test_mark_step_started(self):
        """Test marking step as started."""
        plan = ExecutionPlan.create(goal="Test")
        step = PlanStep.create(description="Step 1")
        plan.add_step(step)

        result = plan.mark_step_started(step.id)

        assert result is True
        assert step.status == StepStatus.IN_PROGRESS
        assert step.started_at is not None

    def test_mark_step_completed(self):
        """Test marking step as completed."""
        plan = ExecutionPlan.create(goal="Test")
        step = PlanStep.create(description="Step 1")
        plan.add_step(step)

        result = plan.mark_step_completed(step.id, "Output text")

        assert result is True
        assert step.status == StepStatus.COMPLETED
        assert step.output == "Output text"
        assert step.completed_at is not None

    def test_mark_step_failed(self):
        """Test marking step as failed."""
        plan = ExecutionPlan.create(goal="Test")
        step = PlanStep.create(description="Step 1")
        plan.add_step(step)

        result = plan.mark_step_failed(step.id, "Error message")

        assert result is True
        assert step.status == StepStatus.FAILED
        assert step.error == "Error message"

    def test_is_complete(self):
        """Test checking plan completion."""
        plan = ExecutionPlan.create(goal="Test")
        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2")

        plan.add_step(step1)
        plan.add_step(step2)

        assert plan.is_complete() is False

        plan.mark_step_completed(step1.id)
        assert plan.is_complete() is False

        plan.mark_step_completed(step2.id)
        assert plan.is_complete() is True

    def test_is_complete_with_skipped(self):
        """Test that skipped steps count as complete."""
        plan = ExecutionPlan.create(goal="Test")
        step = PlanStep.create(description="Step 1")
        plan.add_step(step)

        plan.mark_step_skipped(step.id, "Not needed")

        assert plan.is_complete() is True

    def test_get_progress(self):
        """Test progress tracking."""
        plan = ExecutionPlan.create(goal="Test")

        for i in range(5):
            plan.add_step(PlanStep.create(description=f"Step {i+1}"))

        assert plan.get_progress() == (0, 5)
        assert plan.get_progress_percent() == 0.0

        plan.mark_step_completed(plan.steps[0].id)
        plan.mark_step_completed(plan.steps[1].id)

        assert plan.get_progress() == (2, 5)
        assert plan.get_progress_percent() == 40.0

    def test_plan_serialization(self):
        """Test plan to/from dict."""
        plan = ExecutionPlan.create(
            goal="Build feature",
            summary="Implementation plan",
        )
        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2", dependencies=[step1.id])

        plan.add_step(step1)
        plan.add_step(step2)
        plan.mark_step_completed(step1.id, "Done")

        # Serialize and deserialize
        d = plan.to_dict()
        restored = ExecutionPlan.from_dict(d)

        assert restored.id == plan.id
        assert restored.goal == plan.goal
        assert len(restored.steps) == 2
        assert restored.steps[0].status == StepStatus.COMPLETED

    def test_format_for_display(self):
        """Test human-readable display format."""
        plan = ExecutionPlan.create(
            goal="Test goal",
            summary="Test plan",
        )
        step1 = PlanStep.create(description="First step")
        step2 = PlanStep.create(description="Second step", dependencies=[step1.id])

        plan.add_step(step1)
        plan.add_step(step2)
        plan.mark_step_completed(step1.id)

        display = plan.format_for_display()

        assert "Test plan" in display
        assert "Test goal" in display
        assert "First step" in display
        assert "Second step" in display
        assert "50%" in display  # Progress


class TestPlanner:
    """Tests for Planner class."""

    def test_planner_init(self):
        """Test planner initialization."""
        # Mock provider
        class MockProvider:
            pass

        planner = Planner(
            provider=MockProvider(),
            available_tools=["read_file", "write_file"],
        )

        assert planner.available_tools == ["read_file", "write_file"]
        assert planner.current_plan is None

    def test_planner_get_next_step(self):
        """Test getting next step to execute."""
        class MockProvider:
            pass

        planner = Planner(provider=MockProvider())

        # No plan yet
        assert planner.get_next_step() is None

        # Create plan manually
        plan = ExecutionPlan.create(goal="Test")
        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2", dependencies=[step1.id])
        plan.add_step(step1)
        plan.add_step(step2)
        planner._current_plan = plan

        # Should return first step
        next_step = planner.get_next_step()
        assert next_step.id == step1.id

    def test_planner_get_parallel_steps(self):
        """Test getting steps for parallel execution."""
        class MockProvider:
            pass

        planner = Planner(provider=MockProvider())

        plan = ExecutionPlan.create(goal="Test")
        step1 = PlanStep.create(description="Step 1")
        step2 = PlanStep.create(description="Step 2")  # No deps, can run parallel
        step3 = PlanStep.create(description="Step 3", dependencies=[step1.id, step2.id])

        plan.add_step(step1)
        plan.add_step(step2)
        plan.add_step(step3)
        planner._current_plan = plan

        parallel = planner.get_parallel_steps()

        assert len(parallel) == 2
        assert step1 in parallel
        assert step2 in parallel

    def test_planner_step_lifecycle(self):
        """Test step start/complete/fail methods."""
        class MockProvider:
            pass

        planner = Planner(provider=MockProvider())

        plan = ExecutionPlan.create(goal="Test")
        step = PlanStep.create(description="Step 1")
        plan.add_step(step)
        planner._current_plan = plan

        # Start
        assert planner.start_step(step.id) is True
        assert step.status == StepStatus.IN_PROGRESS

        # Complete
        assert planner.complete_step(step.id, "Output") is True
        assert step.status == StepStatus.COMPLETED
        assert step.output == "Output"

    def test_planner_is_plan_complete(self):
        """Test checking if plan is complete."""
        class MockProvider:
            pass

        planner = Planner(provider=MockProvider())

        # No plan = complete
        assert planner.is_plan_complete() is True

        plan = ExecutionPlan.create(goal="Test")
        step = PlanStep.create(description="Step 1")
        plan.add_step(step)
        planner._current_plan = plan

        assert planner.is_plan_complete() is False

        planner.complete_step(step.id)
        assert planner.is_plan_complete() is True

    def test_planner_clear_plan(self):
        """Test clearing the plan."""
        class MockProvider:
            pass

        planner = Planner(provider=MockProvider())

        plan = ExecutionPlan.create(goal="Test")
        planner._current_plan = plan

        planner.clear_plan()

        assert planner.current_plan is None

    def test_parse_plan_response_valid_json(self):
        """Test parsing valid JSON plan response."""
        class MockProvider:
            pass

        planner = Planner(provider=MockProvider())

        response = '''```json
{
    "summary": "Implement feature X",
    "steps": [
        {
            "description": "Read existing code",
            "reasoning": "Need to understand current impl",
            "dependencies": [],
            "tool_hints": ["read_file"],
            "estimated_complexity": "low"
        },
        {
            "description": "Write new code",
            "reasoning": "Implement the feature",
            "dependencies": [0],
            "tool_hints": ["write_file"],
            "estimated_complexity": "medium"
        }
    ]
}
```'''

        plan = planner._parse_plan_response(response, "Test goal")

        assert plan.summary == "Implement feature X"
        assert len(plan.steps) == 2
        assert plan.steps[0].description == "Read existing code"
        assert plan.steps[1].dependencies == [plan.steps[0].id]

    def test_parse_plan_response_invalid_json(self):
        """Test parsing invalid JSON returns simple plan."""
        class MockProvider:
            pass

        planner = Planner(provider=MockProvider())

        response = "This is not valid JSON at all"

        plan = planner._parse_plan_response(response, "Test goal")

        assert plan.goal == "Test goal"
        assert plan.steps == []  # Empty plan as fallback


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.IN_PROGRESS.value == "in_progress"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.BLOCKED.value == "blocked"

    def test_status_from_string(self):
        """Test creating status from string."""
        assert StepStatus("pending") == StepStatus.PENDING
        assert StepStatus("completed") == StepStatus.COMPLETED
