"""Plan-Then-Execute: decompose complex tasks into atomic steps for small models.

Three-phase pipeline with hardcoded orchestration:
1. TaskDecomposer — LLM generates a numbered step plan (focused prompt, no tools, no history)
2. PlanParser — Hardcoded regex extracts steps into list[Step] dataclass
3. ChunkedExecutor — Executes each step with fresh context and filtered tools

Design philosophy: Parser and orchestrator are 100% hardcoded. Only decomposer and
per-step execution use the LLM. Memory between steps is filesystem/git state, not conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.core.context import ContextWindow, estimate_tokens
from src.core.errors import classify_error, RecoveryStrategy
from src.llm.base import ModelProvider
from src.tools.base import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class StepType(Enum):
    """Categorises what a step does. Used for tool filtering."""
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    GIT = "git"
    ANALYZE = "analyze"
    GENERATE = "generate"


@dataclass
class Step:
    """A single atomic step in a plan."""
    number: int
    description: str
    step_type: StepType = StepType.ANALYZE
    relevant_tools: list[str] = field(default_factory=list)


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Outcome of executing a single step."""
    step: Step
    status: StepStatus
    output: str = ""
    error: str = ""


@dataclass
class PlanResult:
    """Outcome of executing an entire plan."""
    original_request: str
    steps: list[Step]
    results: list[StepResult]

    @property
    def success(self) -> bool:
        return all(r.status == StepStatus.COMPLETED for r in self.results)

    @property
    def summary(self) -> str:
        lines = []
        for r in self.results:
            icon = {
                StepStatus.COMPLETED: "[OK]",
                StepStatus.FAILED: "[FAIL]",
                StepStatus.SKIPPED: "[SKIP]",
            }.get(r.status, "[??]")
            lines.append(f"  {icon} Step {r.step.number}: {r.step.description}")
            if r.error:
                lines.append(f"       Error: {r.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool filtering: map step types to tool name prefixes/names
# ---------------------------------------------------------------------------

_STEP_TYPE_TOOLS: dict[StepType, list[str]] = {
    StepType.READ: ["read_file", "list_dir", "git_status", "git_diff", "git_log"],
    StepType.WRITE: ["read_file", "write_file", "list_dir"],
    StepType.SHELL: ["run_shell", "read_file", "list_dir"],
    StepType.GIT: ["git_status", "git_diff", "git_log", "git_branch", "git_add", "git_commit", "git_checkout"],
    StepType.ANALYZE: ["read_file", "list_dir", "git_status", "git_diff", "git_log", "run_shell"],
    StepType.GENERATE: ["read_file", "write_file", "list_dir"],
}


def _filter_tools(registry: ToolRegistry, step_type: StepType) -> ToolRegistry:
    """Create a new registry containing only tools relevant to the step type."""
    allowed_names = set(_STEP_TYPE_TOOLS.get(step_type, []))
    filtered = ToolRegistry()
    for tool in registry.list_tools():
        if tool.name in allowed_names:
            filtered.register(tool)
    return filtered


# ---------------------------------------------------------------------------
# Step type inference: hardcoded keyword matching
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS: list[tuple[StepType, list[str]]] = [
    (StepType.GIT, ["commit", "branch", "checkout", "stage", "git add", "git status", "git diff", "git log", "push", "merge"]),
    (StepType.WRITE, ["write", "create file", "save", "modify", "edit", "update file", "add to file", "append"]),
    (StepType.READ, ["read", "view", "open", "inspect", "examine", "look at", "check file", "cat "]),
    (StepType.SHELL, ["run ", "execute", "install", "pip", "npm", "command", "terminal", "shell", "pytest", "test"]),
    (StepType.GENERATE, ["generate", "produce", "draft", "compose", "build", "construct", "implement", "code"]),
    (StepType.ANALYZE, ["analyze", "review", "understand", "determine", "figure out", "investigate", "find", "search", "list"]),
]


def _infer_step_type(description: str) -> StepType:
    """Infer step type from description using keyword matching. 100% hardcoded."""
    lower = description.lower()
    for stype, keywords in _TYPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return stype
    return StepType.ANALYZE


# ---------------------------------------------------------------------------
# 1. TaskDecomposer — LLM call with focused prompt
# ---------------------------------------------------------------------------

_PLANNING_PROMPT = """Break the following task into a numbered list of simple, atomic steps.
Each step should be one clear action that can be done independently.
Do NOT include tool names or code. Just describe what to do in plain English.
Number each step starting from 1.

Task: {task}

Steps:"""


class TaskDecomposer:
    """Use an LLM to decompose a complex task into numbered steps.

    Uses a stripped-down prompt: no tools, no history, just the task.
    Optionally uses a different (larger) provider for planning.
    """

    def __init__(
        self,
        provider: ModelProvider,
        planning_provider: Optional[ModelProvider] = None,
    ) -> None:
        self._provider = planning_provider or provider
        self._prompt_template = _PLANNING_PROMPT

    def decompose(self, task: str) -> str:
        """Send the task to the LLM and return raw plan text."""
        prompt = self._prompt_template.format(task=task)
        messages = [
            {"role": "system", "content": "You are a task planner. Output ONLY a numbered list of steps. No explanations, no preamble."},
            {"role": "user", "content": prompt},
        ]
        # No tools, no history — minimal context
        return self._provider.generate(messages, tools=None, max_tokens=1024, temperature=0.3)


# ---------------------------------------------------------------------------
# 2. PlanParser — 100% hardcoded regex extraction
# ---------------------------------------------------------------------------

# Matches lines like "1. Do something" or "1) Do something" or "Step 1: Do something"
_STEP_PATTERN = re.compile(
    r"^\s*(?:step\s+)?(\d+)[.):\-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


class PlanParser:
    """Parse LLM plan output into structured Step objects. Zero LLM involvement."""

    def parse(self, plan_text: str) -> list[Step]:
        """Extract numbered steps from plan text using regex."""
        steps: list[Step] = []
        seen_numbers: set[int] = set()

        for match in _STEP_PATTERN.finditer(plan_text):
            num = int(match.group(1))
            desc = match.group(2).strip()

            # Deduplicate by step number
            if num in seen_numbers:
                continue
            seen_numbers.add(num)

            # Strip trailing punctuation if it's just a period
            if desc.endswith("."):
                desc = desc[:-1].strip()

            step_type = _infer_step_type(desc)
            relevant = list(_STEP_TYPE_TOOLS.get(step_type, []))

            steps.append(Step(
                number=num,
                description=desc,
                step_type=step_type,
                relevant_tools=relevant,
            ))

        # Sort by step number
        steps.sort(key=lambda s: s.number)
        return steps


# ---------------------------------------------------------------------------
# 3. ChunkedExecutor — fresh context per step, tool filtering
# ---------------------------------------------------------------------------

_EXECUTION_PROMPT = """You are executing step {step_num} of {total_steps} in a plan.

Original request: {original_request}

Current step: {step_description}

Execute this step using the available tools. Be concise and focused.
When done, state what you accomplished."""


class ChunkedExecutor:
    """Execute plan steps one at a time with fresh context and filtered tools.

    Key design:
    - Each step gets a clean context window (no accumulated history)
    - Only tools relevant to the step type are available
    - Memory between steps is filesystem/git state, not conversation
    - Progress reporting via on_progress callback
    """

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        max_step_turns: int = 10,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_step_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._provider = provider
        self._full_registry = tool_registry
        self._max_step_turns = max_step_turns
        self._on_progress = on_progress
        self._on_step_output = on_step_output

        caps = provider.capabilities()
        self._context_window = ContextWindow(
            context_length=caps.context_length,
            size_tier=caps.size_tier,
        )

    def execute_plan(self, steps: list[Step], original_request: str) -> list[StepResult]:
        """Execute all steps sequentially, returning results for each."""
        results: list[StepResult] = []
        total = len(steps)

        for step in steps:
            if self._on_progress:
                self._on_progress(step.number, total, step.description)

            result = self._execute_step(step, total, original_request)
            results.append(result)

            # On failure: record but continue (user can inspect results)
            if result.status == StepStatus.FAILED:
                # Don't abort — let remaining steps attempt execution
                pass

        return results

    def _execute_step(self, step: Step, total: int, original_request: str) -> StepResult:
        """Execute a single step with fresh context and filtered tools."""
        # Filter tools for this step type
        filtered_registry = _filter_tools(self._full_registry, step.step_type)

        # Build fresh system prompt for this step
        system_prompt = _EXECUTION_PROMPT.format(
            step_num=step.number,
            total_steps=total,
            original_request=original_request,
            step_description=step.description,
        )

        # Fresh message history — no carryover from previous steps
        messages: list[dict[str, str]] = [
            {"role": "user", "content": f"Execute this step: {step.description}"},
        ]

        tool_schemas = filtered_registry.to_openai_schemas() if filtered_registry.names() else None

        # Agentic loop for this single step
        for turn in range(self._max_step_turns):
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            # Trim if needed
            trimmed = self._context_window.trim_messages(messages, system_prompt)
            full_messages = [{"role": "system", "content": system_prompt}] + trimmed

            try:
                response = self._provider.generate(full_messages, tools=tool_schemas)
            except Exception as e:
                classified = classify_error(e)
                if classified.recovery == RecoveryStrategy.REDUCE_CONTEXT and turn < self._max_step_turns - 1:
                    # Retry with less context
                    half = max(1, len(trimmed) // 2)
                    messages = trimmed[-half:]
                    continue
                return StepResult(
                    step=step,
                    status=StepStatus.FAILED,
                    error=str(e),
                )

            if response is None:
                return StepResult(step=step, status=StepStatus.FAILED, error="No response from model")

            # Emit output for streaming/display
            if self._on_step_output:
                self._on_step_output(response)

            # Check for tool calls (reuse Agent's parsing logic)
            tool_calls = _parse_tool_calls(response)

            if not tool_calls:
                # No tool calls — step is complete
                return StepResult(step=step, status=StepStatus.COMPLETED, output=response)

            # Execute tools and continue the loop
            messages.append({"role": "assistant", "content": response})
            for call in tool_calls:
                result = filtered_registry.execute(call["name"], call["arguments"])
                messages.append({
                    "role": "user",
                    "content": f"[Tool result for {call['name']}]: {result}",
                })

        # Exhausted turns — consider the step complete with what we have
        last_output = messages[-1].get("content", "") if messages else ""
        return StepResult(step=step, status=StepStatus.COMPLETED, output=last_output)


# ---------------------------------------------------------------------------
# Tool call parser (standalone, same logic as Agent._parse_tool_calls)
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from LLM response. Hardcoded regex, no LLM involvement."""
    import json

    calls: list[dict[str, Any]] = []

    # Strategy 1: JSON code blocks
    json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
            if "name" in data and "arguments" in data:
                calls.append({
                    "name": data["name"],
                    "arguments": data["arguments"] if isinstance(data["arguments"], dict) else {},
                })
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Strategy 2: Inline JSON
    inline_pattern = r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\}'
    for match in re.finditer(inline_pattern, text):
        try:
            name = match.group(1)
            args = json.loads(match.group(2))
            calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            continue

    return calls


# ---------------------------------------------------------------------------
# Orchestrator: ties all three phases together
# ---------------------------------------------------------------------------

class PlanExecutor:
    """Full plan-then-execute pipeline.

    Usage:
        executor = PlanExecutor(provider, tool_registry)
        result = executor.run("Refactor the auth module and add tests")
    """

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        planning_provider: Optional[ModelProvider] = None,
        max_step_turns: int = 10,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_step_output: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._decomposer = TaskDecomposer(provider, planning_provider)
        self._parser = PlanParser()
        self._executor = ChunkedExecutor(
            provider=provider,
            tool_registry=tool_registry,
            max_step_turns=max_step_turns,
            on_progress=on_progress,
            on_step_output=on_step_output,
        )

    def run(self, task: str) -> PlanResult:
        """Decompose, parse, and execute a task."""
        # Phase 1: Decompose
        raw_plan = self._decomposer.decompose(task)

        # Phase 2: Parse
        steps = self._parser.parse(raw_plan)

        if not steps:
            # Fallback: if parsing found no steps, create a single step
            steps = [Step(
                number=1,
                description=task,
                step_type=_infer_step_type(task),
                relevant_tools=list(_STEP_TYPE_TOOLS.get(_infer_step_type(task), [])),
            )]

        # Phase 3: Execute
        results = self._executor.execute_plan(steps, task)

        return PlanResult(
            original_request=task,
            steps=steps,
            results=results,
        )


# ---------------------------------------------------------------------------
# Helper: should we use plan-then-execute for this model?
# ---------------------------------------------------------------------------

def should_use_planner(provider: ModelProvider) -> bool:
    """Auto-detect whether to use plan-then-execute based on model size tier.

    Small and medium models benefit from chunked execution.
    Large/API models can handle direct execution.
    """
    caps = provider.capabilities()
    return caps.size_tier in ("small", "medium")
