"""Specialist sub-agent presets — pre-configured role/scope/prompt combos.

Provides ready-to-use specialist configurations for common agent tasks:
- Explore: fast read-only codebase search
- Plan: analysis and planning without file edits
- Debug: stacktrace analysis and root cause identification
- Test: unit test generation and verification

Usage:
    from src.core.specialists import get_specialist, SpecialistType

    spec = get_specialist(SpecialistType.EXPLORE)
    result = await orchestrator.spawn_subagent(
        task="Find all usages of AuthProfile",
        role=spec.role,
        scope=spec.scope,
    )

    # Or use the convenience function:
    result = await spawn_specialist(
        orchestrator, SpecialistType.DEBUG,
        task="Analyze this traceback: ...",
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from src.core.subagent import (
    SubAgentRole,
    SubAgentScope,
    SubAgentOrchestrator,
    SubAgentResult,
    TOOL_CALLING_INSTRUCTIONS,
)


class SpecialistType(str, Enum):
    """Types of specialist sub-agents."""
    EXPLORE = "explore"
    PLAN = "plan"
    DEBUG = "debug"
    TEST = "test"


@dataclass
class SpecialistConfig:
    """Pre-configured specialist sub-agent setup."""

    type: SpecialistType
    role: SubAgentRole
    scope: SubAgentScope
    prompt: str

    @property
    def name(self) -> str:
        return self.type.value


# ---------------------------------------------------------------------------
# Specialist prompts — focused and constrained for each role
# ---------------------------------------------------------------------------

EXPLORE_PROMPT = """You are an explore sub-agent. Your ONLY job is to find and read code.

## Task
{task}

## Rules
- Use read_file and list_dir to search the codebase
- Report what you find clearly and concisely
- Do NOT modify any files
- Do NOT run shell commands
- Focus on answering the specific question, not general exploration
- Summarize your findings at the end

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS

PLAN_PROMPT = """You are a planning sub-agent. Analyze the codebase and produce a plan.

## Task
{task}

## Context from Parent Agent
{previous}

## Rules
- Read relevant files to understand the current architecture
- Identify what needs to change and where
- Produce a step-by-step plan with specific files and changes
- Do NOT modify any files — planning only
- Consider edge cases and potential issues
- Note any dependencies between steps

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS

DEBUG_PROMPT = """You are a debugging sub-agent. Analyze errors and find root causes.

## Task
{task}

## Context from Parent Agent
{previous}

## Rules
- Read the relevant source files mentioned in errors/stacktraces
- Trace the execution path to find the root cause
- Check for common issues: wrong types, missing imports, off-by-one, null/None
- Propose a specific fix with the exact code change needed
- Do NOT modify files — report your findings to the parent agent
- If you can reproduce with a test, describe the test case

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS

TEST_PROMPT = """You are a testing sub-agent. Write unit tests for the specified code.

## Task
{task}

## Context from Parent Agent
{previous}

## Rules
- Read the source code to understand what to test
- Write tests that cover: happy path, edge cases, error conditions
- Follow existing test patterns in the codebase (pytest style)
- Use descriptive test names that explain what is being tested
- Include assertions that verify behavior, not just that code runs
- Group related tests in classes
- Write the test file using write_file

## Available Tools: {tools}
## Scope: {scope}
""" + TOOL_CALLING_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Specialist factory
# ---------------------------------------------------------------------------

_SPECIALIST_CONFIGS = {
    SpecialistType.EXPLORE: {
        "role": SubAgentRole.RESEARCHER,
        "prompt": EXPLORE_PROMPT,
        "allowed_tools": ["read_file", "list_dir"],
        "can_write": False,
        "can_execute": False,
        "max_turns": 15,
    },
    SpecialistType.PLAN: {
        "role": SubAgentRole.RESEARCHER,
        "prompt": PLAN_PROMPT,
        "allowed_tools": ["read_file", "list_dir"],
        "can_write": False,
        "can_execute": False,
        "max_turns": 20,
    },
    SpecialistType.DEBUG: {
        "role": SubAgentRole.DEBUGGER,
        "prompt": DEBUG_PROMPT,
        "allowed_tools": ["read_file", "list_dir", "run_shell"],
        "can_write": False,
        "can_execute": True,  # May need to run tests to reproduce
        "max_turns": 15,
    },
    SpecialistType.TEST: {
        "role": SubAgentRole.TESTER,
        "prompt": TEST_PROMPT,
        "allowed_tools": ["read_file", "list_dir", "write_file", "run_shell"],
        "can_write": True,
        "can_execute": True,
        "max_turns": 20,
    },
}


def get_specialist(
    specialist_type: SpecialistType,
    allowed_paths: Optional[list[Path]] = None,
    context: str = "",
    **scope_overrides,
) -> SpecialistConfig:
    """Get a pre-configured specialist sub-agent.

    Args:
        specialist_type: Type of specialist to create.
        allowed_paths: Restrict to specific directories.
        context: Additional context from the parent agent.
        **scope_overrides: Override any SubAgentScope fields.

    Returns:
        SpecialistConfig ready to use with SubAgentOrchestrator.
    """
    cfg = _SPECIALIST_CONFIGS[specialist_type]

    scope_kwargs = {
        "allowed_tools": cfg["allowed_tools"],
        "can_write": cfg["can_write"],
        "can_execute": cfg["can_execute"],
        "max_turns": cfg["max_turns"],
        "context": context,
        "template_vars": {"previous": context},
    }

    if allowed_paths:
        scope_kwargs["allowed_paths"] = allowed_paths

    scope_kwargs.update(scope_overrides)

    return SpecialistConfig(
        type=specialist_type,
        role=cfg["role"],
        scope=SubAgentScope(**scope_kwargs),
        prompt=cfg["prompt"],
    )


async def spawn_specialist(
    orchestrator: SubAgentOrchestrator,
    specialist_type: SpecialistType,
    task: str,
    allowed_paths: Optional[list[Path]] = None,
    context: str = "",
    **scope_overrides,
) -> SubAgentResult:
    """Convenience function to spawn a specialist sub-agent.

    Args:
        orchestrator: The sub-agent orchestrator.
        specialist_type: Type of specialist.
        task: Task description.
        allowed_paths: Restrict to specific directories.
        context: Additional context from the parent agent.
        **scope_overrides: Override any SubAgentScope fields.

    Returns:
        SubAgentResult with the specialist's output.
    """
    spec = get_specialist(
        specialist_type,
        allowed_paths=allowed_paths,
        context=context,
        **scope_overrides,
    )

    return await orchestrator.spawn_subagent(
        task=task,
        role=spec.role,
        scope=spec.scope,
        custom_prompt=spec.prompt,
    )


def list_specialists() -> list[dict]:
    """List all available specialist types with their capabilities."""
    result = []
    for stype, cfg in _SPECIALIST_CONFIGS.items():
        result.append({
            "type": stype.value,
            "role": cfg["role"].value,
            "tools": cfg["allowed_tools"],
            "can_write": cfg["can_write"],
            "can_execute": cfg["can_execute"],
            "max_turns": cfg["max_turns"],
        })
    return result
