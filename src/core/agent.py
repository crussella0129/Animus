"""Agent class with agentic loop, tool calling, and model-size-aware execution."""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from typing import Any, Callable, Optional

from src.core.context import ContextWindow, estimate_tokens
from src.core.errors import RecoveryStrategy, classify_error
from src.llm.base import ModelProvider
from src.tools.base import ToolRegistry

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
    ) -> None:
        self._provider = provider
        self._tools = tool_registry
        self._base_system_prompt = system_prompt
        self._max_turns = max_turns
        self._messages: list[dict[str, str]] = []
        self._cumulative_tokens: int = 0
        self._planning_provider = planning_provider

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

        Each chunk is sent as a separate user message with part numbering.
        After each chunk's response, a condensed summary carries context
        to the next chunk. The final chunk's response is the overall response.
        """
        total = len(chunks)
        last_response = ""

        for i, chunk in enumerate(chunks, 1):
            if i == 1:
                content = f"[Part {i}/{total}] {chunk}"
            else:
                summary = last_response[:200].strip()
                content = f"[Part {i}/{total}] (Previous context: {summary})\n\n{chunk}"

            self._messages.append({"role": "user", "content": content})
            self._cumulative_tokens += estimate_tokens(content)
            last_response = self._run_agentic_loop()

            if i < total:
                # Store assistant response for context continuity
                self._messages.append({"role": "assistant", "content": last_response})

        return last_response

    def _run_chunked_stream(
        self,
        chunks: list[str],
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Process multiple instruction chunks with streaming output."""
        total = len(chunks)
        last_response = ""

        for i, chunk in enumerate(chunks, 1):
            if i == 1:
                content = f"[Part {i}/{total}] {chunk}"
            else:
                summary = last_response[:200].strip()
                content = f"[Part {i}/{total}] (Previous context: {summary})\n\n{chunk}"

            self._messages.append({"role": "user", "content": content})
            self._cumulative_tokens += estimate_tokens(content)

            # Only stream the final chunk's output to avoid confusing partial output
            if i == total:
                last_response = self._run_agentic_loop_stream(on_chunk)
            else:
                last_response = self._run_agentic_loop()
                self._messages.append({"role": "assistant", "content": last_response})

        return last_response

    def _run_agentic_loop(self) -> str:
        """Core agentic loop: generate, check for tool calls, execute, repeat.

        Grammar is used on the first turn to get a valid tool call. After
        executing a tool, the result is fed back for the model to summarize
        (without grammar). If the model keeps producing tool calls, each is
        executed. After all tool calls are done, the last tool result is
        returned directly as a fallback.
        """
        last_tool_result = ""
        for turn in range(self._max_turns):
            # Use grammar on the first turn only; after tool results,
            # let the model respond freely.
            use_grammar = (turn == 0)
            response_text = self._step(use_grammar=use_grammar)
            if response_text is None:
                # If generation failed but we have a tool result, return it
                if last_tool_result:
                    return last_tool_result
                return "Error: Failed to get a response from the model."

            self._cumulative_tokens += estimate_tokens(response_text)

            tool_calls = self._parse_tool_calls(response_text)
            if not tool_calls:
                return response_text

            self._messages.append({"role": "assistant", "content": response_text})
            for call in tool_calls:
                result = self._tools.execute(call["name"], call["arguments"])
                last_tool_result = result
                self._messages.append({
                    "role": "user",
                    "content": f"[Tool result for {call['name']}]: {result}",
                })

        # Exhausted turns but have a tool result — return it directly
        if last_tool_result:
            return last_tool_result
        return "Reached maximum turns without a final response."

    def _run_agentic_loop_stream(
        self,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Core agentic loop with streaming support."""
        for turn in range(self._max_turns):
            response_text = self._step_stream(on_chunk)
            if response_text is None:
                return "Error: Failed to get a response from the model."

            self._cumulative_tokens += estimate_tokens(response_text)

            tool_calls = self._parse_tool_calls(response_text)
            if not tool_calls:
                return response_text

            self._messages.append({"role": "assistant", "content": response_text})
            for call in tool_calls:
                result = self._tools.execute(call["name"], call["arguments"])
                self._messages.append({
                    "role": "user",
                    "content": f"[Tool result for {call['name']}]: {result}",
                })

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
        """Parse tool calls from LLM response. Tries multiple strategies."""
        calls = []

        # Strategy 0: Raw JSON (GBNF grammar output — no wrapping)
        stripped = text.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                calls.append({
                    "name": data["name"],
                    "arguments": data["arguments"] if isinstance(data["arguments"], dict) else {},
                })
                return calls
        except (json.JSONDecodeError, TypeError):
            pass

        # Strategy 1: JSON code blocks with tool_call structure
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

        # Strategy 2: Inline JSON tool calls
        inline_pattern = r'\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\}'
        for match in re.finditer(inline_pattern, text):
            try:
                name = match.group(1)
                args = json.loads(match.group(2))
                calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                continue

        return calls

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
