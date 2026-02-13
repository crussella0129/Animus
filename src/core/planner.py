"""Plan-Then-Execute: decompose complex tasks into atomic steps for small models.

Three-phase pipeline with hardcoded orchestration:
1. TaskDecomposer — LLM generates a numbered step plan (focused prompt, no tools, no history)
2. PlanParser — Hardcoded regex extracts steps into list[Step] dataclass
3. ChunkedExecutor — Executes each step with fresh context and filtered tools

Design philosophy: Parser and orchestrator are 100% hardcoded. Only decomposer and
per-step execution use the LLM. Memory between steps is filesystem/git state, not conversation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.core.context import ContextWindow, estimate_tokens
from src.core.errors import classify_error, RecoveryStrategy
from src.core.tool_parsing import parse_tool_calls
from src.llm.base import ModelCapabilities, ModelProvider
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
    StepType.READ: ["read_file", "list_dir", "git_status", "git_diff", "git_log", "search_codebase"],
    StepType.WRITE: ["read_file", "write_file", "list_dir"],
    StepType.SHELL: ["run_shell", "read_file", "list_dir"],
    StepType.GIT: ["git_status", "git_diff", "git_log", "git_branch", "git_add", "git_commit", "git_checkout"],
    StepType.ANALYZE: ["read_file", "list_dir", "git_status", "git_diff", "git_log", "run_shell", "search_code_graph", "get_callers", "get_blast_radius", "search_codebase"],
    StepType.GENERATE: ["read_file", "write_file", "list_dir"],
}


# ---------------------------------------------------------------------------
# Tier-aware planning profiles: scale complexity to model capacity
# ---------------------------------------------------------------------------

_PLANNING_TIER_FACTORS: dict[str, dict[str, float]] = {
    "small": {
        "step_base": 1.5,
        "turn_base": 1.5,
        "output_ratio": 0.0625,
        "desc_ratio": 0.012,
    },
    "medium": {
        "step_base": 2.5,
        "turn_base": 2.5,
        "output_ratio": 0.125,
        "desc_ratio": 0.024,
    },
    "large": {
        "step_base": 3.5,
        "turn_base": 5.0,
        "output_ratio": 0.5,
        "desc_ratio": 0.0,
    },
}


def _compute_planning_profile(caps: ModelCapabilities) -> dict[str, int]:
    """Compute planning profile dynamically from model capabilities.

    Scales with log2(context_length / 1024). At ctx=4096 (log_scale=2),
    produces the same values as the original fixed profiles.
    Hard caps: steps<=10, turns<=15.
    """
    tier = caps.size_tier if caps.size_tier in _PLANNING_TIER_FACTORS else "medium"
    factors = _PLANNING_TIER_FACTORS[tier]
    log_scale = max(1.0, math.log2(max(1, caps.context_length) / 1024))

    steps = min(10, int(factors["step_base"] * log_scale))
    turns = min(15, int(factors["turn_base"] * log_scale))
    output = int(caps.context_length * factors["output_ratio"])
    desc = int(caps.context_length * factors["desc_ratio"]) if factors["desc_ratio"] > 0 else 0

    return {
        "max_plan_steps": max(1, steps),
        "max_step_turns": max(1, turns),
        "max_output_tokens": max(64, output),
        "max_step_desc_tokens": desc,
    }


# Deprecated: fixed alias for backward compatibility at ctx=4096.
_PLANNING_PROFILES: dict[str, dict[str, int]] = {
    "small": {
        "max_plan_steps": 3,
        "max_step_turns": 3,
        "max_output_tokens": 256,
        "max_step_desc_tokens": 50,
    },
    "medium": {
        "max_plan_steps": 5,
        "max_step_turns": 5,
        "max_output_tokens": 512,
        "max_step_desc_tokens": 100,
    },
    "large": {
        "max_plan_steps": 7,
        "max_step_turns": 10,
        "max_output_tokens": 2048,
        "max_step_desc_tokens": 0,
    },
}


def _get_planning_profile(caps: ModelCapabilities) -> dict[str, int]:
    """Get planning profile for a model's capabilities."""
    return _compute_planning_profile(caps)


def _filter_tools(registry: ToolRegistry, step_type: StepType, step_description: str = "") -> ToolRegistry:
    """Create a new registry containing only tools relevant to the step type.

    If the step description explicitly mentions a tool name (e.g. "read_file(grammar.py)"),
    narrows to just that tool for tighter GBNF grammar constraints.
    """
    allowed_names = set(_STEP_TYPE_TOOLS.get(step_type, []))
    available = [tool for tool in registry.list_tools() if tool.name in allowed_names]

    # Narrow to a single tool if the step description mentions one by name
    if step_description and available:
        desc_lower = step_description.lower()
        for tool in available:
            if tool.name in desc_lower:
                filtered = ToolRegistry()
                filtered.register(tool)
                return filtered

    filtered = ToolRegistry()
    for tool in available:
        filtered.register(tool)
    return filtered


# ---------------------------------------------------------------------------
# Step type inference: hardcoded keyword matching
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS: list[tuple[StepType, list[str]]] = [
    (StepType.GIT, ["commit", "branch", "checkout", "stage", "git add", "git status", "git diff", "git log", "push", "merge"]),
    (StepType.WRITE, ["write", "create file", "create a", "make a", "save", "modify", "edit", "update file", "add to file", "append"]),
    (StepType.READ, ["read", "view", "open", "inspect", "examine", "look at", "check file", "cat "]),
    (StepType.SHELL, ["run ", "execute", "install", "pip", "npm", "command", "terminal", "shell", "pytest", "test"]),
    (StepType.GENERATE, ["generate", "produce", "draft", "compose", "build", "construct", "implement", "code"]),
    (StepType.ANALYZE, ["analyze", "review", "understand", "determine", "figure out", "investigate", "find", "search", "list"]),
]


def _infer_step_type(description: str) -> StepType:
    """Infer step type from description using file pattern + keyword matching.

    File operation patterns take priority over keyword matching to avoid
    misclassification (e.g., "edit test_auth.py" should be WRITE, not SHELL).
    """
    lower = description.lower()

    # Priority 1: Check for file path patterns (e.g., "something.py", "file.txt")
    # This prevents "test_auth.py" from triggering SHELL due to the "test" keyword
    has_file_pattern = re.search(r'\b\w+\.\w{1,5}\b', description)

    if has_file_pattern:
        # If description mentions a file, check for file operation verbs first
        write_verbs = ["edit", "modify", "update", "change", "fix", "add to", "append", "write", "create"]
        read_verbs = ["read", "view", "check", "look at", "examine", "inspect", "show", "cat", "display"]

        if any(verb in lower for verb in write_verbs):
            return StepType.WRITE
        if any(verb in lower for verb in read_verbs):
            return StepType.READ

    # Priority 2: Keyword matching (original behavior)
    for stype, keywords in _TYPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return stype

    return StepType.ANALYZE


# ---------------------------------------------------------------------------
# 1. TaskDecomposer — LLM call with focused prompt
# ---------------------------------------------------------------------------

_PLANNING_PROMPT = """You are an AI agent. Your working directory is: {cwd}
Your tools: {tool_names}
Break this task into 1-5 numbered steps. Each step is one tool call.
Do NOT include "open terminal" or "press Enter". Keep it short.

Task: {task}

Steps:"""

_MAX_PLAN_STEPS = 7  # Hard cap: discard steps beyond this


class TaskDecomposer:
    """Use an LLM to decompose a complex task into numbered steps.

    Uses a stripped-down prompt: no tools, no history, just the task.
    Optionally uses a different (larger) provider for planning.
    """

    def __init__(
        self,
        provider: ModelProvider,
        planning_provider: Optional[ModelProvider] = None,
        tool_names: Optional[list[str]] = None,
    ) -> None:
        self._provider = planning_provider or provider
        self._prompt_template = _PLANNING_PROMPT
        self._tool_names = tool_names or []

    def decompose(self, task: str) -> str:
        """Send the task to the LLM and return raw plan text."""
        import os

        tool_str = ", ".join(self._tool_names) if self._tool_names else "read_file, write_file, list_dir, run_shell"
        prompt = self._prompt_template.format(task=task, tool_names=tool_str, cwd=os.getcwd())
        messages = [
            {"role": "system", "content": "You are a task planner. Output ONLY a numbered list of 1-5 steps. No explanations, no preamble, no conversational text."},
            {"role": "user", "content": prompt},
        ]
        # No tools, no history — minimal context
        return self._provider.generate(messages, tools=None, max_tokens=512, temperature=0.3)


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

    def __init__(self, max_steps: int = _MAX_PLAN_STEPS) -> None:
        self._max_steps = max_steps

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

        # Sort by step number and enforce cap
        steps.sort(key=lambda s: s.number)
        return steps[:self._max_steps]


# ---------------------------------------------------------------------------
# 3. ChunkedExecutor — fresh context per step, tool filtering
# ---------------------------------------------------------------------------

_EXECUTION_PROMPT = """You are an AI agent executing a single step. Be DIRECT, EFFICIENT, and COMPLETE.

CRITICAL RULES:
1. Make ONE tool call to complete this step, then STOP
2. If a tool fails, do NOT retry with minor variations - return the error
3. Do NOT read files you just wrote - trust that the write succeeded
4. Use absolute paths (e.g., C:\\Users\\...) not relative paths
5. If you've made 3+ tool calls, STOP and return what you have

QUALITY RULES:
- When writing code files: Include proper if __name__ == "__main__" blocks
- When writing scripts: Make them runnable with user input/output
- When writing plans: Be specific with package names and code examples
- Write COMPLETE, PRODUCTION-READY content - not minimal stubs

Working directory: {cwd}

{tool_schemas}

Respond with ONLY a JSON tool call. Example:
{{"name": "write_file", "arguments": {{"path": "C:\\\\Users\\\\example\\\\file.txt", "content": "..."}}}}

Step {step_num}/{total_steps}: {step_description}"""


class ChunkedExecutor:
    """Execute plan steps one at a time with fresh context and filtered tools.

    Key design:
    - Each step gets a clean context window (no accumulated history)
    - Only tools relevant to the step type are available
    - Memory between steps is filesystem/git state, not conversation
    - Progress reporting via on_progress callback
    - Tier-aware: output tokens and turn counts scale with model size
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
        self._on_progress = on_progress
        self._on_step_output = on_step_output

        caps = provider.capabilities()
        self._context_window = ContextWindow(
            context_length=caps.context_length,
            size_tier=caps.size_tier,
        )

        # Scale execution parameters to model capacity
        profile = _get_planning_profile(caps)
        self._max_step_turns = min(max_step_turns, profile["max_step_turns"])
        self._max_output_tokens = profile["max_output_tokens"]

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

    def _evaluate_tool_result(self, tool_name: str, result: str) -> str:
        """Evaluate tool result and inject reflection with forceful guidance."""
        # Detect errors - be VERY forceful about stopping
        if result.startswith("Error:") or "permission denied" in result.lower() or "failed" in result.lower():
            return (
                f"[Tool {tool_name} FAILED]\n"
                f"Result: {result}\n"
                f"ACTION REQUIRED: This tool failed. Do NOT retry it. Either use a completely different tool "
                f"or consider this step complete and return the error. Do NOT make another tool call for the same operation."
            )

        # Detect empty results
        if not result.strip() or result.strip() in ("None", "null", "", "(no output)"):
            return (
                f"[Tool {tool_name} returned empty/null result]\n"
                f"This usually means the operation succeeded with no output. "
                f"Consider this step COMPLETE. Do NOT retry or verify with another tool call."
            )

        # Detect success messages - treat as completion
        if "successfully" in result.lower() or "created" in result.lower() or "wrote" in result.lower():
            return (
                f"[Tool {tool_name} SUCCESS]: {result}\n"
                f"The operation succeeded. This step is COMPLETE. Do NOT make additional tool calls to verify."
            )

        # Summarize long outputs
        if len(result) > 1000:
            preview = result[:300]
            return (
                f"[Tool {tool_name} returned large output: {len(result)} chars]\n"
                f"Preview: {preview}...\n"
                f"The tool provided extensive output. Consider this step COMPLETE."
            )

        # Success case
        return f"[Tool result for {tool_name}]: {result}"

    def _execute_step(self, step: Step, total: int, original_request: str) -> StepResult:
        """Execute a single step with fresh context and filtered tools."""
        import os

        # Filter tools for this step type (narrows to single tool if description mentions one)
        filtered_registry = _filter_tools(self._full_registry, step.step_type, step.description)

        # Build GBNF grammar constraint for native models (None for API providers)
        from src.core.grammar import build_grammar

        grammar = build_grammar(filtered_registry.list_tools())

        # Build concise tool schema descriptions for the system prompt
        tool_lines = []
        for tool in filtered_registry.list_tools():
            params = tool.parameters.get("properties", {})
            param_strs = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                param_strs.append(f'{pname} ({ptype}): {pdesc}')
            params_text = "; ".join(param_strs) if param_strs else "none"
            tool_lines.append(f"- {tool.name}: {tool.description} Params: {params_text}")
        tool_schemas_str = "\n".join(tool_lines) if tool_lines else "No tools available."

        system_prompt = _EXECUTION_PROMPT.format(
            step_num=step.number,
            total_steps=total,
            step_description=step.description,
            tool_schemas=tool_schemas_str,
            cwd=os.getcwd(),
        )

        # Fresh message history — no carryover from previous steps
        messages: list[dict[str, str]] = [
            {"role": "user", "content": step.description},
        ]

        tool_schemas = filtered_registry.to_openai_schemas() if filtered_registry.names() else None

        # Agentic loop for this single step
        import json as _json

        prev_call_key: str | None = None
        repeat_count = 0
        last_tool_result = ""
        # Track tool names to detect thrashing (same tool, different args)
        tool_name_history: list[str] = []
        total_tool_calls = 0
        MAX_TOOLS_PER_STEP = 6  # Hard limit to prevent runaway execution

        for turn in range(self._max_step_turns):
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            # Trim if needed
            trimmed = self._context_window.trim_messages(messages, system_prompt)
            full_messages = [{"role": "system", "content": system_prompt}] + trimmed

            try:
                response = self._provider.generate(
                    full_messages, tools=tool_schemas,
                    max_tokens=self._max_output_tokens, grammar=grammar,
                )
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
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # No tool calls — step is complete
                return StepResult(step=step, status=StepStatus.COMPLETED, output=response)

            # Deduplicate tool calls within the same response
            # If model returns [{call}, {call}], only execute once
            seen_in_response = set()
            unique_calls = []
            for call in tool_calls:
                call_signature = _json.dumps((call["name"], call["arguments"]), sort_keys=True)
                if call_signature not in seen_in_response:
                    seen_in_response.add(call_signature)
                    unique_calls.append(call)

            if len(unique_calls) < len(tool_calls):
                # Duplicates were found and removed
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"[System]: Removed {len(tool_calls) - len(unique_calls)} duplicate tool calls from your response. "
                               f"Only executing unique calls."
                })

            tool_calls = unique_calls

            # Detect repeated identical tool calls to prevent infinite loops
            call_key = _json.dumps(
                [(c["name"], c["arguments"]) for c in tool_calls], sort_keys=True
            )
            if call_key == prev_call_key:
                repeat_count += 1
                if repeat_count >= 1:  # Break after first repeat (2 identical calls total)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": "[System]: That tool call was repeated and failed. Move on to a different action or return current result.",
                    })
                    if last_tool_result:
                        return StepResult(step=step, status=StepStatus.COMPLETED, output=last_tool_result)
                    # Force early exit if we have no result
                    return StepResult(
                        step=step,
                        status=StepStatus.COMPLETED,
                        output="Stopped due to repeated tool call."
                    )
            else:
                repeat_count = 0
            prev_call_key = call_key

            # Detect tool thrashing (same tool name repeatedly with different args)
            current_tool_names = [c["name"] for c in tool_calls]
            tool_name_history.extend(current_tool_names)

            # Check if we're stuck in a loop with the same tool
            if len(tool_name_history) >= 6:
                recent_tools = tool_name_history[-6:]
                # If the same tool appears 4+ times in last 6 calls, we're thrashing
                from collections import Counter
                tool_counts = Counter(recent_tools)
                max_count = max(tool_counts.values()) if tool_counts else 0

                if max_count >= 4:
                    most_common_tool = tool_counts.most_common(1)[0][0]
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": f"[System]: Tool '{most_common_tool}' has been called {max_count} times without success. "
                                   f"This approach is not working. Try a completely different strategy or return what you have.",
                    })
                    if last_tool_result:
                        return StepResult(step=step, status=StepStatus.COMPLETED, output=last_tool_result)
                    # Force return with current state
                    return StepResult(
                        step=step,
                        status=StepStatus.COMPLETED,
                        output=last_tool_result or "Stopped due to repeated unsuccessful attempts."
                    )

            # Execute tools and continue the loop
            messages.append({"role": "assistant", "content": response})
            for call in tool_calls:
                total_tool_calls += 1

                # Hard limit: prevent runaway tool execution
                if total_tool_calls >= MAX_TOOLS_PER_STEP:
                    messages.append({
                        "role": "user",
                        "content": f"[System]: Hard limit reached - {MAX_TOOLS_PER_STEP} tool calls executed in this step. "
                                   f"Returning current result to prevent runaway execution.",
                    })
                    return StepResult(
                        step=step,
                        status=StepStatus.COMPLETED,
                        output=last_tool_result or f"Stopped after {MAX_TOOLS_PER_STEP} tool calls."
                    )

                result = filtered_registry.execute(call["name"], call["arguments"])
                last_tool_result = result

                # Apply reflection/evaluation (same pattern as agent.py)
                evaluated_result = self._evaluate_tool_result(call["name"], result)
                messages.append({
                    "role": "user",
                    "content": evaluated_result,
                })

        # Exhausted turns — consider the step complete with what we have
        last_output = last_tool_result or (messages[-1].get("content", "") if messages else "")
        return StepResult(step=step, status=StepStatus.COMPLETED, output=last_output)


# ---------------------------------------------------------------------------
# Tool call parser (standalone, same logic as Agent._parse_tool_calls)
# ---------------------------------------------------------------------------

# Tool call parsing is now handled by shared utility in src.core.tool_parsing


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
        # Scale to model capacity
        caps = provider.capabilities()
        profile = _get_planning_profile(caps)

        self._decomposer = TaskDecomposer(
            provider, planning_provider, tool_names=tool_registry.names()
        )
        self._parser = PlanParser(max_steps=profile["max_plan_steps"])
        self._executor = ChunkedExecutor(
            provider=provider,
            tool_registry=tool_registry,
            max_step_turns=max_step_turns,
            on_progress=on_progress,
            on_step_output=on_step_output,
        )

    def run(self, task: str) -> PlanResult:
        """Decompose, parse, and execute a task.

        For simple tasks, skips the LLM planning step and creates a single
        step directly — still uses the execution framework (GBNF grammar, tool
        schemas) which is essential for small models to produce tool calls.

        For multi-step tasks, tries LLM decomposition first, then falls back
        to heuristic conjunction splitting if the LLM produces too few steps.
        """
        if _is_simple_task(task):
            # Skip planning — single-step execution with grammar constraints
            step_type = _infer_step_type(task)
            steps = [Step(
                number=1,
                description=task,
                step_type=step_type,
                relevant_tools=list(_STEP_TYPE_TOOLS.get(step_type, [])),
            )]
        else:
            # Phase 1: Decompose
            raw_plan = self._decomposer.decompose(task)

            # Phase 2: Parse
            steps = self._parser.parse(raw_plan)

            # Fallback: if LLM produced <=1 step for a multi-conjunction task,
            # use heuristic splitting instead. Small models often collapse
            # multi-step instructions into a single step.
            if len(steps) <= 1:
                heuristic_steps = _heuristic_decompose(task)
                if heuristic_steps:
                    steps = heuristic_steps

            if not steps:
                # Final fallback: single step with full task
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

def should_use_planner(provider: ModelProvider, task: str = "") -> bool:
    """Auto-detect whether to use plan-then-execute based on model size tier.

    Small and medium models always go through the planner framework.
    The framework itself handles complexity: simple tasks skip the LLM
    planning step and execute directly with grammar constraints.
    Large/API models use direct execution.
    """
    caps = provider.capabilities()
    return caps.size_tier in ("small", "medium")


def _is_simple_task(task: str) -> bool:
    """Heuristic: detect tasks simple enough to skip planning.

    Simple = short, single action, no multi-step conjunctions.
    """
    lower = task.lower()

    # Multi-step conjunctions always indicate complex tasks
    if " then " in lower or " and then " in lower:
        return False

    # Very short tasks are almost always single-action
    words = task.split()
    if len(words) <= 10:
        return True

    # Count action verbs — multiple actions suggest multi-step
    _action_words = {"read", "write", "create", "list", "show", "find", "check",
                     "run", "execute", "commit", "search", "open", "delete", "edit",
                     "update", "add", "remove", "install", "build", "test", "diff"}
    action_count = sum(1 for w in words if w.lower().strip(",.!?") in _action_words)
    if action_count <= 1:
        return True

    return False


def _heuristic_decompose(task: str) -> list[Step]:
    """Split a multi-step task into steps using conjunction splitting.

    Handles multiple patterns:
    - Temporal: "then", "and then", ". Then,"
    - Conjunctions: ", and verb", ", verb" (for action verbs)
    - Sentences: period-separated with capitalization

    Used as a fallback when the LLM planner fails to produce enough steps
    for a clearly multi-part task.
    """
    # Normalize separators into a common delimiter
    text = task

    # Strategy 1: Temporal indicators (highest confidence)
    # ". Then" / ", then" / "; then" → split point
    text = re.sub(r'[.;,]\s*[Tt]hen\s*,?\s*', '\n---SPLIT---\n', text)
    # " and then " / " then "
    text = re.sub(r'\s+and\s+then\s+', '\n---SPLIT---\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+then\s+', '\n---SPLIT---\n', text, flags=re.IGNORECASE)

    # Strategy 2: Conjunction with action verbs
    # ", and <verb>" where verb is a common action word
    action_verbs = r'(?:read|write|edit|update|fix|create|delete|run|execute|test|check|verify|add|remove|modify|analyze|search|find|list|show|get|set)'
    text = re.sub(
        rf',\s+and\s+({action_verbs})\b',
        r'\n---SPLIT---\n\1',
        text,
        flags=re.IGNORECASE
    )

    # Strategy 3: Comma-separated actions (lower confidence)
    # Only split on ", <verb>" if we haven't already found splits
    if '---SPLIT---' not in text:
        # ", <verb>" where verb starts the clause
        text = re.sub(
            rf',\s+({action_verbs})\b',
            r'\n---SPLIT---\n\1',
            text,
            flags=re.IGNORECASE
        )

    # Strategy 4: Sentence boundaries (period + capital letter)
    text = re.sub(r'\.\s+([A-Z])', r'\n---SPLIT---\n\1', text)

    parts = [p.strip() for p in text.split('---SPLIT---') if p.strip()]

    if len(parts) <= 1:
        return []  # No splits found, caller should fall back

    steps: list[Step] = []
    for i, part in enumerate(parts, 1):
        step_type = _infer_step_type(part)
        steps.append(Step(
            number=i,
            description=part,
            step_type=step_type,
            relevant_tools=list(_STEP_TYPE_TOOLS.get(step_type, [])),
        ))

    return steps
