"""TaskDecomposer: LLM-based task decomposition into numbered steps."""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.llm.base import ModelCapabilities, ModelProvider
from src.core.workspace import Workspace

from src.core.planner.parser import (
    Step,
    StepType,
    _infer_step_type,
    _MAX_PLAN_STEPS,
    PlanParser,
    _compute_planning_profile,
    _get_planning_profile,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.transcript import TranscriptLogger


# ---------------------------------------------------------------------------
# 1. TaskDecomposer — LLM call with focused prompt
# ---------------------------------------------------------------------------

_PLANNING_PROMPT = """You are an AI agent. Your working directory is: {cwd}
Your tools: {tool_names}
Break this task into 1-5 numbered steps. Each step is one tool call.
Do NOT include "open terminal" or "press Enter". Keep it short.
IMPORTANT: Always use double quotes around paths/names that contain spaces (e.g., mkdir "my folder", cd "test 1").

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
        tool_names: Optional[list[str]] = None,
        session_cwd: Workspace | None = None,
        transcript: TranscriptLogger | None = None,
    ) -> None:
        self._provider = planning_provider or provider
        self._prompt_template = _PLANNING_PROMPT
        self._tool_names = tool_names or []
        self._session_cwd = session_cwd
        self._transcript = transcript

    def decompose(self, task: str) -> str:
        """Send the task to the LLM and return raw plan text."""
        import os

        cwd = str(self._session_cwd.path) if self._session_cwd else os.getcwd()
        tool_str = ", ".join(self._tool_names) if self._tool_names else "read_file, write_file, list_dir, run_shell"
        prompt = self._prompt_template.format(task=task, tool_names=tool_str, cwd=cwd)
        messages = [
            {"role": "system", "content": "You are a task planner. Output ONLY a numbered list of 1-5 steps. No explanations, no preamble, no conversational text."},
            {"role": "user", "content": prompt},
        ]
        if self._transcript:
            self._transcript.log_plan_request(prompt)
        # No tools, no history — minimal context
        result = self._provider.generate(messages, tools=None, max_tokens=512, temperature=0.3)
        if self._transcript:
            self._transcript.log_plan_response(result)
        return result
