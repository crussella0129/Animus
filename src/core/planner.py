"""Planning module for task decomposition and execution planning.

Provides explicit planning phase before implementation to:
1. Generate task breakdowns from user requests
2. Create execution DAGs for multi-step tasks
3. Track progress through the plan
4. Allow plan revision mid-execution
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.base import ModelProvider, Message, GenerationConfig


class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"  # Waiting on dependencies


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    id: str
    description: str
    reasoning: str  # Why this step is needed
    dependencies: list[str] = field(default_factory=list)  # IDs of steps that must complete first
    status: StepStatus = StepStatus.PENDING
    tool_hints: list[str] = field(default_factory=list)  # Suggested tools for this step
    estimated_complexity: str = "medium"  # low, medium, high
    output: Optional[str] = None  # Result after completion
    error: Optional[str] = None  # Error if failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        description: str,
        reasoning: str = "",
        dependencies: Optional[list[str]] = None,
        tool_hints: Optional[list[str]] = None,
        estimated_complexity: str = "medium",
    ) -> "PlanStep":
        """Create a new plan step with generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            description=description,
            reasoning=reasoning,
            dependencies=dependencies or [],
            tool_hints=tool_hints or [],
            estimated_complexity=estimated_complexity,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "reasoning": self.reasoning,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "tool_hints": self.tool_hints,
            "estimated_complexity": self.estimated_complexity,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanStep":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            reasoning=data.get("reasoning", ""),
            dependencies=data.get("dependencies", []),
            status=StepStatus(data.get("status", "pending")),
            tool_hints=data.get("tool_hints", []),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            output=data.get("output"),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionPlan:
    """A complete execution plan with steps forming a DAG."""
    id: str
    goal: str  # The original user request/goal
    summary: str  # Brief summary of the plan
    steps: list[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    revision_count: int = 0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, goal: str, summary: str = "") -> "ExecutionPlan":
        """Create a new execution plan."""
        return cls(
            id=str(uuid.uuid4())[:8],
            goal=goal,
            summary=summary,
        )

    def add_step(self, step: PlanStep) -> None:
        """Add a step to the plan."""
        self.steps.append(step)
        self.updated_at = datetime.now()

    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_ready_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute (dependencies satisfied).

        Returns steps that:
        - Have status PENDING
        - All dependencies are COMPLETED
        """
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        ready = []

        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            # Check if all dependencies are completed
            if all(dep_id in completed_ids for dep_id in step.dependencies):
                ready.append(step)

        return ready

    def get_blocked_steps(self) -> list[PlanStep]:
        """Get steps that are blocked on dependencies."""
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        blocked = []

        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            # Blocked if any dependency is not completed
            if step.dependencies and not all(dep_id in completed_ids for dep_id in step.dependencies):
                blocked.append(step)

        return blocked

    def mark_step_started(self, step_id: str) -> bool:
        """Mark a step as in progress."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.now()
            self.updated_at = datetime.now()
            return True
        return False

    def mark_step_completed(self, step_id: str, output: str = "") -> bool:
        """Mark a step as completed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.COMPLETED
            step.output = output
            step.completed_at = datetime.now()
            self.updated_at = datetime.now()
            return True
        return False

    def mark_step_failed(self, step_id: str, error: str) -> bool:
        """Mark a step as failed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.FAILED
            step.error = error
            step.completed_at = datetime.now()
            self.updated_at = datetime.now()
            return True
        return False

    def mark_step_skipped(self, step_id: str, reason: str = "") -> bool:
        """Mark a step as skipped."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.SKIPPED
            step.output = reason or "Skipped"
            step.completed_at = datetime.now()
            self.updated_at = datetime.now()
            return True
        return False

    def is_complete(self) -> bool:
        """Check if plan execution is complete."""
        if not self.steps:
            return True
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
            for s in self.steps
        )

    def get_progress(self) -> tuple[int, int]:
        """Get progress as (completed, total)."""
        completed = sum(
            1 for s in self.steps
            if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
        )
        return completed, len(self.steps)

    def get_progress_percent(self) -> float:
        """Get progress as percentage."""
        completed, total = self.get_progress()
        return (completed / total * 100) if total > 0 else 100.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "goal": self.goal,
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "revision_count": self.revision_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        """Create from dictionary."""
        plan = cls(
            id=data["id"],
            goal=data["goal"],
            summary=data.get("summary", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            revision_count=data.get("revision_count", 0),
            metadata=data.get("metadata", {}),
        )
        plan.steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return plan

    def format_for_display(self) -> str:
        """Format plan for human-readable display."""
        lines = [
            f"Plan: {self.summary}",
            f"Goal: {self.goal}",
            f"Progress: {self.get_progress_percent():.0f}%",
            "",
            "Steps:",
        ]

        status_icons = {
            StepStatus.PENDING: "○",
            StepStatus.IN_PROGRESS: "◐",
            StepStatus.COMPLETED: "●",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "○",
            StepStatus.BLOCKED: "◌",
        }

        for i, step in enumerate(self.steps, 1):
            icon = status_icons.get(step.status, "?")
            dep_str = ""
            if step.dependencies:
                dep_str = f" (after: {', '.join(step.dependencies)})"
            lines.append(f"  {icon} {i}. {step.description}{dep_str}")
            if step.status == StepStatus.FAILED and step.error:
                lines.append(f"      Error: {step.error}")

        return "\n".join(lines)


PLANNING_PROMPT = """You are a planning assistant. Given a user request, create a detailed execution plan.

Break down the task into discrete, actionable steps. Consider:
1. What information needs to be gathered first (reading files, exploring structure)
2. What dependencies exist between steps
3. What tools might be needed for each step
4. What could go wrong and how to handle it

Available tools: {tools}

IMPORTANT: Output your plan as a JSON object with this exact structure:
{{
    "summary": "Brief one-line summary of the plan",
    "steps": [
        {{
            "description": "Clear description of what this step accomplishes",
            "reasoning": "Why this step is necessary",
            "dependencies": [],  // IDs of steps this depends on (empty for first steps)
            "tool_hints": ["tool_name"],  // Suggested tools
            "estimated_complexity": "low|medium|high"
        }}
    ]
}}

Rules:
- Each step should be atomic and focused
- Dependencies reference step indices (0-based) or can be empty
- First steps should have no dependencies
- Order steps logically - gather info before modifying
- Include verification/testing steps where appropriate
- Keep the plan concise - avoid unnecessary steps

User Request: {request}

{context}

Output ONLY the JSON plan, no other text."""


REVISION_PROMPT = """You are revising an execution plan based on new information.

Original Plan:
{original_plan}

Completed Steps:
{completed_steps}

New Information / Issue:
{issue}

Create a revised plan that:
1. Keeps completed steps as-is
2. Adjusts remaining steps based on new information
3. Adds new steps if needed
4. Removes steps that are no longer necessary

Output the revised plan as JSON with the same structure as before.
Only include steps that still need to be done (not already completed ones).

Output ONLY the JSON plan, no other text."""


class Planner:
    """Generates and manages execution plans."""

    def __init__(
        self,
        provider: "ModelProvider",
        available_tools: Optional[list[str]] = None,
    ):
        """Initialize the planner.

        Args:
            provider: LLM provider for plan generation.
            available_tools: List of available tool names.
        """
        self.provider = provider
        self.available_tools = available_tools or [
            "read_file", "write_file", "list_dir", "run_shell",
            "git", "analyze_code", "find_symbols", "get_code_structure",
            "index_codebase", "search_codebase",
        ]
        self._current_plan: Optional[ExecutionPlan] = None

    @property
    def current_plan(self) -> Optional[ExecutionPlan]:
        """Get the current active plan."""
        return self._current_plan

    async def create_plan(
        self,
        request: str,
        context: str = "",
        model: str = "qwen2.5-coder:7b",
    ) -> ExecutionPlan:
        """Create an execution plan for a user request.

        Args:
            request: The user's request/goal.
            context: Additional context (e.g., relevant files, previous work).
            model: Model to use for planning.

        Returns:
            ExecutionPlan with steps to accomplish the goal.
        """
        from src.llm.base import Message, GenerationConfig

        # Build the planning prompt
        tools_str = ", ".join(self.available_tools)
        context_str = f"Context:\n{context}" if context else ""

        prompt = PLANNING_PROMPT.format(
            tools=tools_str,
            request=request,
            context=context_str,
        )

        messages = [Message(role="user", content=prompt)]
        config = GenerationConfig(temperature=0.3, max_tokens=2048)

        result = await self.provider.generate(
            messages=messages,
            model=model,
            config=config,
        )

        # Parse the plan from response
        plan = self._parse_plan_response(result.content, request)
        self._current_plan = plan
        return plan

    def _parse_plan_response(self, content: str, goal: str) -> ExecutionPlan:
        """Parse plan from LLM response."""
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                start = content.index("```json") + 7
                end = content.index("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.index("```") + 3
                end = content.index("```", start)
                content = content[start:end].strip()

            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # If JSON parsing fails, create a simple single-step plan
            return ExecutionPlan.create(
                goal=goal,
                summary="Direct execution",
            )

        # Create plan from parsed data
        plan = ExecutionPlan.create(
            goal=goal,
            summary=data.get("summary", "Execution plan"),
        )

        # Create step ID mapping (index -> generated ID)
        step_ids = []
        steps_data = data.get("steps", [])

        # First pass: create steps with generated IDs
        for step_data in steps_data:
            step = PlanStep.create(
                description=step_data.get("description", "Unknown step"),
                reasoning=step_data.get("reasoning", ""),
                tool_hints=step_data.get("tool_hints", []),
                estimated_complexity=step_data.get("estimated_complexity", "medium"),
            )
            step_ids.append(step.id)
            plan.add_step(step)

        # Second pass: resolve dependencies (indices to IDs)
        for i, step_data in enumerate(steps_data):
            deps = step_data.get("dependencies", [])
            resolved_deps = []
            for dep in deps:
                # Handle both index (int) and string index references
                try:
                    dep_idx = int(dep)
                    if 0 <= dep_idx < len(step_ids):
                        resolved_deps.append(step_ids[dep_idx])
                except (ValueError, TypeError):
                    # If it's already an ID, keep it
                    if dep in step_ids:
                        resolved_deps.append(dep)
            plan.steps[i].dependencies = resolved_deps

        return plan

    async def revise_plan(
        self,
        issue: str,
        model: str = "qwen2.5-coder:7b",
    ) -> Optional[ExecutionPlan]:
        """Revise the current plan based on new information.

        Args:
            issue: Description of the issue or new information.
            model: Model to use for revision.

        Returns:
            Revised ExecutionPlan, or None if no current plan.
        """
        if not self._current_plan:
            return None

        from src.llm.base import Message, GenerationConfig

        # Format completed steps
        completed = []
        for step in self._current_plan.steps:
            if step.status == StepStatus.COMPLETED:
                completed.append(f"- {step.description}: {step.output or 'Done'}")
        completed_str = "\n".join(completed) if completed else "None yet"

        # Build revision prompt
        prompt = REVISION_PROMPT.format(
            original_plan=self._current_plan.format_for_display(),
            completed_steps=completed_str,
            issue=issue,
        )

        messages = [Message(role="user", content=prompt)]
        config = GenerationConfig(temperature=0.3, max_tokens=2048)

        result = await self.provider.generate(
            messages=messages,
            model=model,
            config=config,
        )

        # Parse revised plan
        revised = self._parse_plan_response(result.content, self._current_plan.goal)
        revised.revision_count = self._current_plan.revision_count + 1

        # Keep completed steps from original plan
        completed_steps = [s for s in self._current_plan.steps if s.status == StepStatus.COMPLETED]
        revised.steps = completed_steps + revised.steps

        self._current_plan = revised
        return revised

    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next step to execute.

        Returns the first ready step (dependencies satisfied, status pending).
        """
        if not self._current_plan:
            return None

        ready = self._current_plan.get_ready_steps()
        return ready[0] if ready else None

    def get_parallel_steps(self) -> list[PlanStep]:
        """Get all steps that can be executed in parallel.

        Returns all ready steps with no interdependencies.
        """
        if not self._current_plan:
            return []

        return self._current_plan.get_ready_steps()

    def start_step(self, step_id: str) -> bool:
        """Mark a step as started."""
        if self._current_plan:
            return self._current_plan.mark_step_started(step_id)
        return False

    def complete_step(self, step_id: str, output: str = "") -> bool:
        """Mark a step as completed."""
        if self._current_plan:
            return self._current_plan.mark_step_completed(step_id, output)
        return False

    def fail_step(self, step_id: str, error: str) -> bool:
        """Mark a step as failed."""
        if self._current_plan:
            return self._current_plan.mark_step_failed(step_id, error)
        return False

    def skip_step(self, step_id: str, reason: str = "") -> bool:
        """Mark a step as skipped."""
        if self._current_plan:
            return self._current_plan.mark_step_skipped(step_id, reason)
        return False

    def is_plan_complete(self) -> bool:
        """Check if the current plan is complete."""
        if not self._current_plan:
            return True
        return self._current_plan.is_complete()

    def get_progress(self) -> tuple[int, int]:
        """Get current plan progress as (completed, total)."""
        if not self._current_plan:
            return (0, 0)
        return self._current_plan.get_progress()

    def clear_plan(self) -> None:
        """Clear the current plan."""
        self._current_plan = None
