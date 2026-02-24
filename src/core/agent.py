"""Agent class with agentic loop, tool calling, and model-size-aware execution."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Generator
from typing import Any, Callable, Optional

from src.core.context import ContextWindow, estimate_tokens
from src.core.workspace import Workspace
from src.core.errors import RecoveryStrategy, classify_error
from src.core.tool_parsing import parse_tool_calls
from src.llm.base import ModelProvider
from src.tools.base import ToolRegistry

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.transcript import TranscriptLogger

# Appended to the system prompt when tools are available, so even small
# models know to respond with JSON tool calls instead of prose.
_TOOL_CALL_INSTRUCTION = """

When you need to perform an action, respond with ONLY a JSON tool call.
Available tools: {tool_names}

{tool_examples}
Respond with ONLY the JSON object. No explanation, no markdown."""


def _build_tool_examples(registry: ToolRegistry) -> str:
    """Build concrete tool call examples from registered tools."""
    examples = []
    for tool in registry.list_tools():
        # Build a minimal example with the actual parameter names
        params = tool.parameters.get("properties", {})
        example_args = {}
        for param_name, param_info in list(params.items())[:2]:
            if param_info.get("type") == "integer":
                example_args[param_name] = 10
            elif param_info.get("type") == "boolean":
                example_args[param_name] = False
            else:
                example_args[param_name] = f"example_{param_name}"
        import json
        example_json = json.dumps({"name": tool.name, "arguments": example_args})
        examples.append(f"  {tool.name}: {example_json}")
    return "Examples:\n" + "\n".join(examples)


class Agent:
    """Core agent: manages conversation, tool calls, and context window."""

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int = 20,
        planning_provider: Optional[ModelProvider] = None,
        session_cwd: Workspace | None = None,
        transcript: TranscriptLogger | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tool_registry
        self._base_system_prompt = system_prompt
        self._max_turns = max_turns
        self._messages: list[dict[str, str]] = []
        self._cumulative_tokens: int = 0
        self._planning_provider = planning_provider
        self._session_cwd = session_cwd
        self._transcript = transcript

        # Set up context window based on model capabilities
        caps = provider.capabilities()
        self._context_window = ContextWindow(
            context_length=caps.context_length,
            size_tier=caps.size_tier,
        )

        # Build GBNF grammar for native providers (None for API providers)
        self._grammar = self._build_tool_grammar()

        # Augment system prompt with tool call instructions when tools exist
        self._system_prompt = self._build_system_prompt()

    def _build_tool_grammar(self) -> Any:
        """Build GBNF grammar from registered tools. Returns None if unavailable."""
        if not self._tools.names():
            return None
        try:
            from src.core.grammar import build_grammar
            return build_grammar(self._tools.list_tools())
        except Exception:
            return None

    def _build_system_prompt(self) -> str:
        """Augment the base system prompt with tool call format instructions."""
        tool_names = self._tools.names()
        if not tool_names:
            return self._base_system_prompt
        return self._base_system_prompt + _TOOL_CALL_INSTRUCTION.format(
            tool_names=", ".join(tool_names),
            tool_examples=_build_tool_examples(self._tools),
        )

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages.copy()

    def run(self, user_input: str) -> str:
        """Run the agent loop: send user message, handle tool calls, return final response."""
        chunks = self._context_window.chunk_instruction(user_input)

        if len(chunks) > 1:
            return self._run_chunked(chunks)

        self._messages.append({"role": "user", "content": user_input})
        self._cumulative_tokens += estimate_tokens(user_input)
        return self._run_agentic_loop()

    def _run_chunked(self, chunks: list[str]) -> str:
        """Process multiple instruction chunks sequentially.

        Non-streaming version - delegates to streaming with null callback.
        """
        return self._run_chunked_stream(chunks, on_chunk=None)

    def _run_chunked_stream(
        self,
        chunks: list[str],
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Process multiple instruction chunks with optional streaming.

        Unified implementation for both streaming and non-streaming chunked execution.
        Streams only the final chunk to avoid confusing partial output.
        """
        total = len(chunks)
        last_response = ""

        for i, chunk in enumerate(chunks, 1):
            if i == 1:
                content = f"[Part {i}/{total}] {chunk}"
            else:
                summary = self._summarize_for_carry(last_response, max_tokens=100)
                content = f"[Part {i}/{total}] (Previous context: {summary})\n\n{chunk}"

            self._messages.append({"role": "user", "content": content})
            self._cumulative_tokens += estimate_tokens(content)

            # Only stream the final chunk's output to avoid confusing partial output
            if i == total and on_chunk is not None:
                last_response = self._run_agentic_loop_stream(on_chunk)
            else:
                last_response = self._run_agentic_loop_stream(on_chunk=None)

            if i < total:
                # Store assistant response for context continuity
                self._messages.append({"role": "assistant", "content": last_response})

        return last_response

    def _summarize_for_carry(self, response: str, max_tokens: int = 100) -> str:
        """Extract key outcomes from a response for inter-chunk context.

        Prioritizes:
        1. Tool results (most recent ones)
        2. Final statements
        3. First meaningful sentences

        Args:
            response: The response to summarize
            max_tokens: Maximum tokens for the summary

        Returns:
            Condensed summary preserving key context
        """
        if not response.strip():
            return "(no output)"

        lines = response.strip().split("\n")

        # Priority 1: Extract tool results
        tool_results = []
        for line in lines:
            if line.startswith("[Tool ") or "Tool result" in line:
                tool_results.append(line)

        if tool_results:
            # Take the last 3 tool results
            summary = " | ".join(tool_results[-3:])
            if estimate_tokens(summary) <= max_tokens:
                return summary
            # If tool results are too long, truncate
            return summary[:max_tokens * 4]

        # Priority 2: If response has clear conclusion markers
        conclusion_markers = ["In summary", "Therefore", "Finally", "Result:", "Output:"]
        for i, line in enumerate(lines):
            if any(marker.lower() in line.lower() for marker in conclusion_markers):
                # Take from this point to the end
                conclusion = " ".join(lines[i:])
                if estimate_tokens(conclusion) <= max_tokens:
                    return conclusion
                break

        # Priority 3: First meaningful sentences (skip empty lines)
        meaningful_lines = [l for l in lines if l.strip()]
        if not meaningful_lines:
            return response[:max_tokens * 4]

        # Take first few sentences up to token budget
        summary = ""
        for line in meaningful_lines[:3]:
            candidate = f"{summary} {line}".strip()
            if estimate_tokens(candidate) > max_tokens:
                break
            summary = candidate

        return summary if summary else response[:max_tokens * 4]

    def _evaluate_tool_result(self, tool_name: str, result: str) -> str:
        """Evaluate tool result and inject reflection/guidance for the agent.

        This implements the observation-reflection pattern: instead of just
        feeding raw tool results back, we classify outcomes and provide
        contextual guidance to help the agent reason about what happened.

        Args:
            tool_name: Name of the tool that was executed
            result: Raw result string from tool execution

        Returns:
            Enhanced result string with reflection metadata
        """
        # Detect errors and suggest alternative approaches
        if result.startswith("Error:") or "permission denied" in result.lower() or "failed" in result.lower():
            return (
                f"[Tool {tool_name} FAILED]\n"
                f"Result: {result}\n"
                f"Reflection: The tool encountered an error. Consider:\n"
                f"  - Trying a different approach or tool\n"
                f"  - Checking if required inputs were correct\n"
                f"  - Verifying permissions or prerequisites"
            )

        # Detect empty or null results
        if not result.strip() or result.strip() in ("None", "null", ""):
            return (
                f"[Tool {tool_name} returned empty result]\n"
                f"Reflection: The operation may have completed without output, "
                f"or there may be nothing to report. Verify the operation succeeded."
            )

        # Summarize very long outputs to preserve context budget
        MAX_RESULT_LENGTH = 2000
        if len(result) > MAX_RESULT_LENGTH:
            truncated = result[:500]
            preview_lines = truncated.count('\n')
            total_lines = result.count('\n')
            return (
                f"[Tool {tool_name} returned large output: {len(result)} chars, {total_lines} lines]\n"
                f"Preview (first 500 chars):\n{truncated}\n...\n"
                f"Reflection: Large output received. The full result is available but truncated here "
                f"to preserve context. Focus on the preview or request specific sections if needed."
            )

        # Success case: return result with minimal framing
        return f"[Tool result for {tool_name}]: {result}"

    def _run_agentic_loop(self) -> str:
        """Core agentic loop: generate, check for tool calls, execute, repeat.

        Non-streaming version - delegates to streaming with a null callback.
        """
        return self._run_agentic_loop_stream(on_chunk=None)

    def _run_agentic_loop_stream(
        self,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Core agentic loop with optional streaming support.

        Unified implementation for both streaming and non-streaming execution.
        When on_chunk is None, operates in non-streaming mode.

        Note: GBNF grammar is only applied on the first turn of non-streaming
        mode. Streaming mode cannot use grammar constraints because
        llama-cpp-python does not support grammar with streaming chat
        completions. This means streaming with small native models may produce
        malformed tool calls on the first turn. API providers are unaffected.
        """
        last_tool_result = ""
        prev_call_key: str | None = None
        repeat_count = 0
        for turn in range(self._max_turns):
            # Grammar on first turn only; streaming cannot use grammar
            # (llama-cpp-python limitation — see _step_stream docstring).
            use_grammar = (turn == 0)
            if on_chunk is not None:
                response_text = self._step_stream(on_chunk)
            else:
                response_text = self._step(use_grammar=use_grammar)
            if response_text is None:
                if last_tool_result:
                    return last_tool_result
                return "Error: Failed to get a response from the model."

            self._cumulative_tokens += estimate_tokens(response_text)

            tool_calls = self._parse_tool_calls(response_text)
            if not tool_calls:
                return response_text

            # Deduplicate tool calls within the same response
            seen_in_response = set()
            unique_calls = []
            for call in tool_calls:
                call_signature = json.dumps((call["name"], call["arguments"]), sort_keys=True)
                if call_signature not in seen_in_response:
                    seen_in_response.add(call_signature)
                    unique_calls.append(call)

            if len(unique_calls) < len(tool_calls):
                # Duplicates found in single response - use unique calls only
                tool_calls = unique_calls

            # Detect repeated identical tool calls
            call_key = json.dumps(
                [(c["name"], c["arguments"]) for c in tool_calls], sort_keys=True
            )
            if call_key == prev_call_key:
                repeat_count += 1
                if repeat_count >= 1:  # Break after first repeat (2 identical calls total)
                    self._messages.append({"role": "assistant", "content": response_text})
                    self._messages.append({
                        "role": "user",
                        "content": "[System]: That tool call was repeated and failed. Try a completely different approach or return your current result.",
                    })
                    if last_tool_result:
                        return last_tool_result
                    return "Stopped due to repeated tool call with no progress."
            else:
                repeat_count = 0
            prev_call_key = call_key

            self._messages.append({"role": "assistant", "content": response_text})
            for call in tool_calls:
                result = self._tools.execute(call["name"], call["arguments"])
                last_tool_result = result
                # Apply reflection/evaluation to tool result
                evaluated_result = self._evaluate_tool_result(call["name"], result)
                self._messages.append({
                    "role": "user",
                    "content": evaluated_result,
                })

            # Rate limiting: progressive slowdown for rapid tool calls
            if turn > 0:
                delay = min(0.5, turn * 0.1)
                time.sleep(delay)

        if last_tool_result:
            return last_tool_result
        return "Reached maximum turns without a final response."

    def _step(self, use_grammar: bool = True) -> str | None:
        """Single generation step with context management and error handling.

        Args:
            use_grammar: Whether to apply GBNF grammar constraint. Set False
                after tool results so the model can respond in natural language.
        """
        trimmed = self._context_window.trim_messages(self._messages, self._system_prompt)
        full_messages = [{"role": "system", "content": self._system_prompt}] + trimmed
        tool_schemas = self._tools.to_openai_schemas() if self._tools.names() else None

        kwargs: dict[str, Any] = {}
        if use_grammar and self._grammar is not None:
            kwargs["grammar"] = self._grammar

        try:
            return self._provider.generate(full_messages, tools=tool_schemas, **kwargs)
        except Exception as e:
            classified = classify_error(e)
            if classified.recovery == RecoveryStrategy.REDUCE_CONTEXT:
                half = max(1, len(trimmed) // 2)
                shorter = trimmed[-half:]
                full_messages = [{"role": "system", "content": self._system_prompt}] + shorter
                try:
                    return self._provider.generate(full_messages, tools=tool_schemas, **kwargs)
                except Exception:
                    return None
            return None

    def run_stream(
        self,
        user_input: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Run the agent loop with streaming output. Calls on_chunk for each token."""
        chunks = self._context_window.chunk_instruction(user_input)

        if len(chunks) > 1:
            return self._run_chunked_stream(chunks, on_chunk)

        self._messages.append({"role": "user", "content": user_input})
        self._cumulative_tokens += estimate_tokens(user_input)
        return self._run_agentic_loop_stream(on_chunk)

    def _step_stream(self, on_chunk: Optional[Callable[[str], None]] = None) -> str | None:
        """Single generation step with streaming support."""
        trimmed = self._context_window.trim_messages(self._messages, self._system_prompt)
        full_messages = [{"role": "system", "content": self._system_prompt}] + trimmed
        tool_schemas = self._tools.to_openai_schemas() if self._tools.names() else None

        # Note: GBNF grammar is not passed for streaming because llama-cpp-python
        # does not support grammar constraints with streaming chat completions.
        try:
            chunks: list[str] = []
            for chunk in self._provider.generate_stream(full_messages, tools=tool_schemas):
                chunks.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
            return "".join(chunks)
        except Exception as e:
            classified = classify_error(e)
            if classified.recovery == RecoveryStrategy.REDUCE_CONTEXT:
                half = max(1, len(trimmed) // 2)
                shorter = trimmed[-half:]
                full_messages = [{"role": "system", "content": self._system_prompt}] + shorter
                try:
                    chunks = []
                    for chunk in self._provider.generate_stream(full_messages, tools=tool_schemas):
                        chunks.append(chunk)
                        if on_chunk:
                            on_chunk(chunk)
                    return "".join(chunks)
                except Exception:
                    return None
            return None

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response (delegates to shared utility)."""
        return parse_tool_calls(text)

    def run_planned(
        self,
        user_input: str,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_step_output: Optional[Callable[[str], None]] = None,
        force: bool = False,
    ) -> str:
        """Run via plan-then-execute pipeline.

        Auto-detects whether to use planner based on model size tier,
        unless force=True which always uses it.
        Falls back to direct run() for large models unless forced.
        """
        from src.core.planner import PlanExecutor, should_use_planner

        if not force and not should_use_planner(self._provider):
            return self.run(user_input)

        executor = PlanExecutor(
            provider=self._provider,
            tool_registry=self._tools,
            planning_provider=self._planning_provider,
            max_step_turns=self._max_turns,
            on_progress=on_progress,
            on_step_output=on_step_output,
            session_cwd=self._session_cwd,
            transcript=self._transcript,
        )

        result = executor.run(user_input)

        # Store the plan result for inspection
        self._last_plan_result = result

        # Build a summary response
        if result.success:
            final_outputs = [r.output for r in result.results if r.output]
            return final_outputs[-1] if final_outputs else "Plan completed successfully."
        else:
            return f"Plan completed with issues:\n{result.summary}"

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
