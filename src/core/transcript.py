"""Transcript logging for Animus agent execution.

Records every LLM prompt/response, tool call/result, and planning step
into a structured timeline that can be rendered as readable Markdown.

Three main components:
- TranscriptLogger: collects timestamped events, renders to .md
- TranscriptToolRegistry: wraps ToolRegistry to intercept execute()
- TranscriptProvider: wraps ModelProvider to intercept generate()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.llm.base import ModelCapabilities, ModelProvider
from src.tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    TASK_START = "task_start"
    PLAN_DECOMPOSITION_REQUEST = "plan_decomposition_request"
    PLAN_DECOMPOSITION_RESPONSE = "plan_decomposition_response"
    PLAN_PARSED = "plan_parsed"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    TASK_COMPLETE = "task_complete"


@dataclass
class TranscriptEvent:
    """A single timestamped event in the transcript."""
    event_type: EventType
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TranscriptLogger
# ---------------------------------------------------------------------------

class TranscriptLogger:
    """Collects timestamped events and renders them as Markdown."""

    def __init__(self) -> None:
        self._events: list[TranscriptEvent] = []
        self._start_time: float = time.time()
        self._task: str = ""
        self._status: str = "IN_PROGRESS"
        self._current_step: int = 0
        self._total_steps: int = 0

    # --- Convenience logging methods ---

    def log_task_start(self, task: str) -> None:
        self._task = task
        self._start_time = time.time()
        self._events.append(TranscriptEvent(
            event_type=EventType.TASK_START,
            timestamp=time.time(),
            data={"task": task},
        ))

    def log_plan_request(self, prompt: str) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.PLAN_DECOMPOSITION_REQUEST,
            timestamp=time.time(),
            data={"prompt": prompt},
        ))

    def log_plan_response(self, raw_plan: str) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.PLAN_DECOMPOSITION_RESPONSE,
            timestamp=time.time(),
            data={"raw_plan": raw_plan},
        ))

    def log_plan_parsed(self, steps: list[dict[str, Any]]) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.PLAN_PARSED,
            timestamp=time.time(),
            data={"steps": steps},
        ))

    def log_step_start(self, step_num: int, total: int, description: str) -> None:
        self._current_step = step_num
        self._total_steps = total
        self._events.append(TranscriptEvent(
            event_type=EventType.STEP_START,
            timestamp=time.time(),
            data={"step_num": step_num, "total": total, "description": description},
        ))

    def log_step_complete(self, step_num: int, status: str, output: str = "", error: str = "") -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.STEP_COMPLETE,
            timestamp=time.time(),
            data={"step_num": step_num, "status": status, "output": output, "error": error},
        ))

    def log_llm_request(self, messages: list[dict[str, str]], step_num: int = 0) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.LLM_REQUEST,
            timestamp=time.time(),
            data={"messages": messages, "step_num": step_num},
        ))

    def log_llm_response(self, response: str, step_num: int = 0) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.LLM_RESPONSE,
            timestamp=time.time(),
            data={"response": response, "step_num": step_num},
        ))

    def log_tool_call(self, name: str, args: dict[str, Any], step_num: int = 0) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.TOOL_CALL,
            timestamp=time.time(),
            data={"name": name, "args": args, "step_num": step_num},
        ))

    def log_tool_result(self, name: str, result: str, duration: float = 0.0, step_num: int = 0) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.TOOL_RESULT,
            timestamp=time.time(),
            data={"name": name, "result": result, "duration": duration, "step_num": step_num},
        ))

    def log_error(self, message: str, step_num: int = 0) -> None:
        self._events.append(TranscriptEvent(
            event_type=EventType.ERROR,
            timestamp=time.time(),
            data={"message": message, "step_num": step_num},
        ))

    def log_task_complete(self, success: bool) -> None:
        self._status = "SUCCESS" if success else "FAILED"
        self._events.append(TranscriptEvent(
            event_type=EventType.TASK_COMPLETE,
            timestamp=time.time(),
            data={"success": success},
        ))

    # --- Rendering ---

    def render_markdown(self) -> str:
        """Render the full transcript as formatted Markdown."""
        duration = time.time() - self._start_time
        lines: list[str] = []

        # Header
        lines.append("# Animus Transcript")
        lines.append(f"**Task**: {self._task}")
        lines.append(f"**Started**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._start_time))}")
        lines.append(f"**Duration**: {duration:.1f}s")
        lines.append(f"**Status**: {self._status}")
        lines.append("")

        # Group events by phase
        plan_request_events = [e for e in self._events if e.event_type == EventType.PLAN_DECOMPOSITION_REQUEST]
        plan_response_events = [e for e in self._events if e.event_type == EventType.PLAN_DECOMPOSITION_RESPONSE]
        plan_parsed_events = [e for e in self._events if e.event_type == EventType.PLAN_PARSED]

        # Plan Decomposition section
        if plan_request_events or plan_response_events:
            lines.append("## Plan Decomposition")
            lines.append("")

            for evt in plan_request_events:
                lines.append("<details>")
                lines.append("<summary>Planner Prompt</summary>")
                lines.append("")
                lines.append("```")
                lines.append(evt.data["prompt"])
                lines.append("```")
                lines.append("</details>")
                lines.append("")

            for evt in plan_response_events:
                lines.append("### Raw Plan (LLM Response)")
                lines.append("```")
                lines.append(evt.data["raw_plan"])
                lines.append("```")
                lines.append("")

            for evt in plan_parsed_events:
                lines.append("### Parsed Steps")
                lines.append("")
                lines.append("| # | Description | Type |")
                lines.append("|---|-------------|------|")
                for step in evt.data["steps"]:
                    lines.append(f"| {step['number']} | {step['description']} | {step['type']} |")
                lines.append("")

        # Step Execution section
        step_events = self._group_step_events()
        if step_events:
            lines.append("## Step Execution")
            lines.append("")

            for step_num, events in sorted(step_events.items()):
                step_start = next((e for e in events if e.event_type == EventType.STEP_START), None)
                step_complete = next((e for e in events if e.event_type == EventType.STEP_COMPLETE), None)
                desc = step_start.data["description"] if step_start else f"Step {step_num}"
                total = step_start.data["total"] if step_start else "?"

                lines.append(f"### Step {step_num}/{total}: {desc}")
                lines.append("")

                # Group by turn (pairs of LLM request/response + tool calls)
                turn_num = 0
                for evt in events:
                    if evt.event_type == EventType.LLM_REQUEST:
                        turn_num += 1
                        lines.append(f"**Turn {turn_num}:**")
                        lines.append("")
                        lines.append("<details>")
                        lines.append("<summary>LLM Request</summary>")
                        lines.append("")
                        for msg in evt.data.get("messages", []):
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            lines.append(f"**{role}**: {content[:500]}")
                            if len(content) > 500:
                                lines.append(f"... ({len(content)} chars total)")
                            lines.append("")
                        lines.append("</details>")
                        lines.append("")

                    elif evt.event_type == EventType.LLM_RESPONSE:
                        lines.append("**LLM Response:**")
                        lines.append("```")
                        resp = evt.data.get("response", "")
                        lines.append(resp[:2000])
                        if len(resp) > 2000:
                            lines.append(f"... ({len(resp)} chars total)")
                        lines.append("```")
                        lines.append("")

                    elif evt.event_type == EventType.TOOL_CALL:
                        args_str = _format_args(evt.data.get("args", {}))
                        lines.append(f"**Tool Call**: `{evt.data['name']}({args_str})`")
                        lines.append("")

                    elif evt.event_type == EventType.TOOL_RESULT:
                        dur = evt.data.get("duration", 0)
                        dur_str = f" ({dur:.2f}s)" if dur > 0 else ""
                        lines.append(f"**Tool Result**{dur_str}:")
                        lines.append("```")
                        result = evt.data.get("result", "")
                        lines.append(result[:1000])
                        if len(result) > 1000:
                            lines.append(f"... ({len(result)} chars total)")
                        lines.append("```")
                        lines.append("")

                    elif evt.event_type == EventType.ERROR:
                        lines.append(f"> **ERROR**: {evt.data.get('message', '')}")
                        lines.append("")

                # Step result
                if step_complete:
                    status = step_complete.data.get("status", "UNKNOWN")
                    step_dur = step_complete.timestamp - (step_start.timestamp if step_start else self._start_time)
                    lines.append(f"**Step Result**: {status} ({step_dur:.1f}s)")
                    if step_complete.data.get("error"):
                        lines.append(f"> Error: {step_complete.data['error']}")
                    lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        step_completes = [e for e in self._events if e.event_type == EventType.STEP_COMPLETE]
        if step_completes:
            lines.append("| Step | Status | Duration |")
            lines.append("|------|--------|----------|")
            for evt in step_completes:
                snum = evt.data.get("step_num", "?")
                status = evt.data.get("status", "UNKNOWN")
                # Find matching step_start for duration
                matching_start = next(
                    (e for e in self._events
                     if e.event_type == EventType.STEP_START
                     and e.data.get("step_num") == snum),
                    None,
                )
                dur = evt.timestamp - matching_start.timestamp if matching_start else 0
                lines.append(f"| {snum} | {status} | {dur:.1f}s |")
            lines.append("")

        # Error summary
        errors = [e for e in self._events if e.event_type == EventType.ERROR]
        if errors:
            lines.append("### Errors")
            for evt in errors:
                lines.append(f"- {evt.data.get('message', '')}")
            lines.append("")

        return "\n".join(lines)

    def _group_step_events(self) -> dict[int, list[TranscriptEvent]]:
        """Group events by step number for rendering."""
        groups: dict[int, list[TranscriptEvent]] = {}
        current_step = 0
        step_types = {
            EventType.STEP_START, EventType.STEP_COMPLETE,
            EventType.LLM_REQUEST, EventType.LLM_RESPONSE,
            EventType.TOOL_CALL, EventType.TOOL_RESULT,
            EventType.ERROR,
        }
        for evt in self._events:
            if evt.event_type not in step_types:
                continue
            step_num = evt.data.get("step_num", 0)
            if evt.event_type == EventType.STEP_START:
                current_step = step_num
            if step_num == 0:
                step_num = current_step
            if step_num > 0:
                groups.setdefault(step_num, []).append(evt)
        return groups

    def save(self, path: str | Path) -> Path:
        """Render and write the transcript to a .md file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = self.render_markdown()
        p.write_text(content, encoding="utf-8")
        return p


# ---------------------------------------------------------------------------
# TranscriptToolRegistry — wraps ToolRegistry to intercept execute()
# ---------------------------------------------------------------------------

class TranscriptToolRegistry:
    """Wraps a ToolRegistry to log every tool call and result."""

    def __init__(self, registry: ToolRegistry, transcript: TranscriptLogger, step_num: int = 0) -> None:
        self._registry = registry
        self._transcript = transcript
        self._step_num = step_num

    def execute(self, name: str, args: dict[str, Any]) -> str:
        self._transcript.log_tool_call(name, args, step_num=self._step_num)
        t0 = time.time()
        result = self._registry.execute(name, args)
        duration = time.time() - t0
        self._transcript.log_tool_result(name, result, duration=duration, step_num=self._step_num)
        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the wrapped registry."""
        return getattr(self._registry, name)


# ---------------------------------------------------------------------------
# TranscriptProvider — wraps ModelProvider to intercept generate()
# ---------------------------------------------------------------------------

class TranscriptProvider:
    """Wraps a ModelProvider to log every LLM request and response."""

    def __init__(self, provider: ModelProvider, transcript: TranscriptLogger, step_num: int = 0) -> None:
        self._provider = provider
        self._transcript = transcript
        self._step_num = step_num

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        self._transcript.log_llm_request(messages, step_num=self._step_num)
        result = self._provider.generate(messages, tools=tools, **kwargs)
        self._transcript.log_llm_response(result, step_num=self._step_num)
        return result

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ):
        self._transcript.log_llm_request(messages, step_num=self._step_num)
        chunks: list[str] = []
        for chunk in self._provider.generate_stream(messages, tools=tools, **kwargs):
            chunks.append(chunk)
            yield chunk
        full_response = "".join(chunks)
        self._transcript.log_llm_response(full_response, step_num=self._step_num)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attribute access to the wrapped provider."""
        return getattr(self._provider, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_args(args: dict[str, Any], max_len: int = 200) -> str:
    """Format tool arguments for display, truncating long values."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > max_len:
            v_str = v_str[:max_len] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
