"""ChunkedExecutor, PlanExecutor, and helper functions for plan execution."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional

from src.core.context import ContextWindow, estimate_tokens
from src.core.workspace import Workspace
from src.core.errors import classify_error, RecoveryStrategy
from src.core.tool_parsing import parse_tool_calls
from src.llm.base import ModelCapabilities, ModelProvider
from src.tools.base import Tool, ToolRegistry

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.transcript import TranscriptLogger

from src.core.planner.parser import (
    Step,
    StepType,
    StepStatus,
    StepResult,
    PlanResult,
    PlanParser,
    _STEP_TYPE_TOOLS,
    _filter_tools,
    _infer_expected_tools,
    _get_planning_profile,
    _compute_planning_profile,
    _infer_step_type,
)
from src.core.planner.decomposer import TaskDecomposer


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
6. ALWAYS double-quote paths and names containing spaces (e.g., mkdir "my folder", cd "test 1")

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
        session_cwd: Workspace | None = None,
        transcript: TranscriptLogger | None = None,
    ) -> None:
        self._provider = provider
        self._full_registry = tool_registry
        self._on_progress = on_progress
        self._on_step_output = on_step_output
        self._session_cwd = session_cwd
        self._transcript = transcript

        caps = provider.capabilities()
        self._context_window = ContextWindow(
            context_length=caps.context_length,
            size_tier=caps.size_tier,
        )

        # Scale execution parameters to model capacity
        profile = _get_planning_profile(caps)
        self._max_step_turns = min(max_step_turns, profile["max_step_turns"])
        self._max_output_tokens = profile["max_output_tokens"]
        self._max_tools_per_step = profile.get("max_tools_per_step", 6)

    def execute_plan(self, steps: list[Step], original_request: str) -> list[StepResult]:
        """Execute all steps sequentially with inter-step memory.

        Each step can learn from previous steps' outcomes to adapt its approach.
        """
        results: list[StepResult] = []
        total = len(steps)

        # Inter-step context: carries forward learnings from previous steps
        step_context = {
            "failed_paths": set(),  # Paths that failed in previous steps
            "successful_operations": [],  # Operations that succeeded
            "discovered_info": {},  # Information discovered during execution
        }

        for step in steps:
            if self._on_progress:
                self._on_progress(step.number, total, step.description)

            result = self._execute_step(step, total, original_request, step_context)
            results.append(result)

            if self._transcript:
                self._transcript.log_step_complete(
                    step.number, result.status.value, output=result.output, error=result.error,
                )

            # Update inter-step context based on result
            if result.status == StepStatus.FAILED and result.error:
                # Track failure patterns
                if "No such file" in result.error or "not found" in result.error.lower():
                    # Extract failed path if present
                    import re
                    path_match = re.search(r'["\']([^"\']+)["\']', result.error)
                    if path_match:
                        step_context["failed_paths"].add(path_match.group(1))
            elif result.status == StepStatus.COMPLETED:
                # Track successful operations
                step_context["successful_operations"].append({
                    "step": step.number,
                    "description": step.description,
                    "output_preview": result.output[:100] if result.output else "",
                })

            # On failure: record but continue (user can inspect results)
            if result.status == StepStatus.FAILED:
                # Don't abort — let remaining steps attempt execution with learned context
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

    def _execute_step(
        self,
        step: Step,
        total: int,
        original_request: str,
        step_context: dict | None = None,
    ) -> StepResult:
        """Execute a single step with fresh context, filtered tools, and inter-step memory.

        Args:
            step: The step to execute
            total: Total number of steps in plan
            original_request: Original user request
            step_context: Optional context from previous steps (failed paths, discoveries)

        Returns:
            StepResult with execution outcome
        """
        import os

        step_context = step_context or {}
        cwd = str(self._session_cwd.path) if self._session_cwd else os.getcwd()

        if self._transcript:
            self._transcript.log_step_start(step.number, total, step.description)

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

        # Explicit tool restriction to prevent scope bleed
        tool_schemas_str += (
            "\n\nIMPORTANT: You may ONLY use the tools listed above. "
            "Do not attempt to use any other tools. If the step cannot be completed "
            "with the available tools, return what you have."
        )

        # Build step context summary if available
        context_info = ""
        if step_context:
            context_parts = []
            if step_context.get("failed_paths"):
                paths = list(step_context["failed_paths"])[:3]
                context_parts.append(f"Failed paths from previous steps: {', '.join(paths)}")
            if step_context.get("successful_operations"):
                recent = step_context["successful_operations"][-2:]  # Last 2 successful ops
                for op in recent:
                    context_parts.append(f"Step {op['step']} succeeded: {op['output_preview']}")

            if context_parts:
                context_info = "\n\nLEARNINGS FROM PREVIOUS STEPS:\n" + "\n".join(f"- {p}" for p in context_parts)

        system_prompt = _EXECUTION_PROMPT.format(
            step_num=step.number,
            total_steps=total,
            step_description=step.description,
            tool_schemas=tool_schemas_str,
            cwd=cwd,
        ) + context_info

        # Fresh message history — no carryover from previous steps except learned context
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
        max_tools = self._max_tools_per_step
        out_of_scope_count = 0
        # Infer expected tools from step description for scope enforcement
        expected_tools = _infer_expected_tools(step.description, filtered_registry.names())

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
                if total_tool_calls >= max_tools:
                    messages.append({
                        "role": "user",
                        "content": f"[System]: Hard limit reached - {max_tools} tool calls executed in this step. "
                                   f"Returning current result to prevent runaway execution.",
                    })
                    return StepResult(
                        step=step,
                        status=StepStatus.COMPLETED,
                        output=last_tool_result or f"Stopped after {max_tools} tool calls."
                    )

                # Scope enforcement: block out-of-scope tool calls BEFORE execution
                if expected_tools and call["name"] not in expected_tools:
                    out_of_scope_count += 1
                    blocked_msg = (
                        f"[System]: Tool '{call['name']}' is not available for this step "
                        f"(\"{step.description}\"). Available tools: {', '.join(expected_tools)}. "
                        f"Use only the listed tools."
                    )
                    messages.append({"role": "user", "content": blocked_msg})

                    if out_of_scope_count >= 2:
                        return StepResult(
                            step=step,
                            status=StepStatus.COMPLETED,
                            output=last_tool_result or "Step terminated: repeated out-of-scope tool calls.",
                        )
                    continue  # Skip execution, give model another chance

                if self._transcript:
                    self._transcript.log_tool_call(call["name"], call["arguments"], step_num=step.number)
                _tool_t0 = time.time()
                result = filtered_registry.execute(call["name"], call["arguments"])
                if self._transcript:
                    self._transcript.log_tool_result(call["name"], result, duration=time.time() - _tool_t0, step_num=step.number)
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
        session_cwd: Workspace | None = None,
        transcript: TranscriptLogger | None = None,
    ) -> None:
        self._transcript = transcript

        # Scale to model capacity
        caps = provider.capabilities()
        profile = _get_planning_profile(caps)

        self._decomposer = TaskDecomposer(
            provider, planning_provider, tool_names=tool_registry.names(),
            session_cwd=session_cwd, transcript=transcript,
        )
        self._parser = PlanParser(max_steps=profile["max_plan_steps"])
        self._executor = ChunkedExecutor(
            provider=provider,
            tool_registry=tool_registry,
            max_step_turns=max_step_turns,
            on_progress=on_progress,
            on_step_output=on_step_output,
            session_cwd=session_cwd,
            transcript=transcript,
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

            if self._transcript and steps:
                self._transcript.log_plan_parsed([
                    {"number": s.number, "description": s.description, "type": s.step_type.value}
                    for s in steps
                ])

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
