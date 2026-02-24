"""Data structures, constants, utility functions, and PlanParser for plan-then-execute."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from src.llm.base import ModelCapabilities
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
    StepType.GIT: ["git_init", "git_status", "git_diff", "git_log", "git_branch", "git_add", "git_commit", "git_checkout"],
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
        "tools_per_step": 2,
    },
    "medium": {
        "step_base": 2.5,
        "turn_base": 2.5,
        "output_ratio": 0.125,
        "desc_ratio": 0.024,
        "tools_per_step": 4,
    },
    "large": {
        "step_base": 3.5,
        "turn_base": 5.0,
        "output_ratio": 0.5,
        "desc_ratio": 0.0,
        "tools_per_step": 6,
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
        "max_tools_per_step": int(factors["tools_per_step"]),
    }


# Deprecated: fixed alias for backward compatibility at ctx=4096.
_PLANNING_PROFILES: dict[str, dict[str, int]] = {
    "small": {
        "max_plan_steps": 3,
        "max_step_turns": 3,
        "max_output_tokens": 256,
        "max_step_desc_tokens": 50,
        "max_tools_per_step": 2,
    },
    "medium": {
        "max_plan_steps": 5,
        "max_step_turns": 5,
        "max_output_tokens": 512,
        "max_step_desc_tokens": 100,
        "max_tools_per_step": 4,
    },
    "large": {
        "max_plan_steps": 7,
        "max_step_turns": 10,
        "max_output_tokens": 2048,
        "max_step_desc_tokens": 0,
        "max_tools_per_step": 6,
    },
}


def _get_planning_profile(caps: ModelCapabilities) -> dict[str, int]:
    """Get planning profile for a model's capabilities."""
    return _compute_planning_profile(caps)


def _infer_expected_tools(description: str, available_tools: list[str]) -> set[str]:
    """Infer which tools a step description expects based on keywords.

    Returns the subset of available_tools that are mentioned or implied
    by the step description. Used for scope enforcement — tool calls
    outside this set trigger warnings.
    """
    desc_lower = description.lower()
    expected: set[str] = set()

    # Direct mention: description contains the tool name
    for tool_name in available_tools:
        if tool_name in desc_lower:
            expected.add(tool_name)

    # Keyword-based inference for common patterns
    _KEYWORD_TO_TOOLS: dict[str, list[str]] = {
        "init": ["git_init", "run_shell"],
        "mkdir": ["run_shell"],
        "create dir": ["run_shell"],
        "create folder": ["run_shell"],
        "write": ["write_file"],
        "create file": ["write_file"],
        "read": ["read_file"],
        "commit": ["git_commit"],
        "stage": ["git_add"],
        "add file": ["git_add"],
        "branch": ["git_branch"],
        "checkout": ["git_checkout"],
        "status": ["git_status"],
        "diff": ["git_diff"],
        "log": ["git_log"],
        "list": ["list_dir"],
    }

    for keyword, tools in _KEYWORD_TO_TOOLS.items():
        if keyword in desc_lower:
            for t in tools:
                if t in available_tools:
                    expected.add(t)

    return expected


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
    (StepType.GIT, ["commit", "branch", "checkout", "stage", "git init", "git add", "git status", "git diff", "git log", "push", "merge", "initialize a git", "initialize git"]),
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
        write_verbs = [
            "edit", "modify", "update", "change", "fix", "add to", "append",
            "write", "create", "save", "implement", "build", "make",
        ]
        read_verbs = [
            "read", "view", "check", "look at", "examine", "inspect",
            "show", "cat", "display", "open", "review",
        ]

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
# Step pattern and max plan steps
# ---------------------------------------------------------------------------

# Matches lines like "1. Do something" or "1) Do something" or "Step 1: Do something"
_STEP_PATTERN = re.compile(
    r"^\s*(?:step\s+)?(\d+)[.):\-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_MAX_PLAN_STEPS = 7  # Hard cap: discard steps beyond this


# ---------------------------------------------------------------------------
# PlanParser — 100% hardcoded regex extraction
# ---------------------------------------------------------------------------

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
