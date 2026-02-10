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


class Agent:
    """Core agent: manages conversation, tool calls, and context window."""

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int = 20,
    ) -> None:
        self._provider = provider
        self._tools = tool_registry
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._messages: list[dict[str, str]] = []
        self._cumulative_tokens: int = 0

        # Set up context window based on model capabilities
        caps = provider.capabilities()
        self._context_window = ContextWindow(
            context_length=caps.context_length,
            size_tier=caps.size_tier,
        )

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages.copy()

    def run(self, user_input: str) -> str:
        """Run the agent loop: send user message, handle tool calls, return final response."""
        # For small models, chunk large instructions
        chunks = self._context_window.chunk_instruction(user_input)

        if len(chunks) == 1:
            self._messages.append({"role": "user", "content": user_input})
        else:
            # Feed instruction in chunks with a combining message
            combined = user_input
            self._messages.append({"role": "user", "content": combined})

        # Track input tokens
        self._cumulative_tokens += estimate_tokens(user_input)

        for turn in range(self._max_turns):
            response_text = self._step()
            if response_text is None:
                return "Error: Failed to get a response from the model."

            # Track output tokens
            self._cumulative_tokens += estimate_tokens(response_text)

            # Check for tool calls in the response
            tool_calls = self._parse_tool_calls(response_text)
            if not tool_calls:
                # No tool calls — this is the final response
                return response_text

            # Execute tools and append results
            self._messages.append({"role": "assistant", "content": response_text})
            for call in tool_calls:
                result = self._tools.execute(call["name"], call["arguments"])
                self._messages.append({
                    "role": "user",
                    "content": f"[Tool result for {call['name']}]: {result}",
                })

        return "Reached maximum turns without a final response."

    def _step(self) -> str | None:
        """Single generation step with context management and error handling."""
        # Trim messages to fit context window
        trimmed = self._context_window.trim_messages(self._messages, self._system_prompt)

        # Build message list with system prompt
        full_messages = [{"role": "system", "content": self._system_prompt}] + trimmed

        # Include tool schemas
        tool_schemas = self._tools.to_openai_schemas() if self._tools.names() else None

        try:
            return self._provider.generate(full_messages, tools=tool_schemas)
        except Exception as e:
            classified = classify_error(e)
            if classified.recovery == RecoveryStrategy.REDUCE_CONTEXT:
                # Try with fewer messages
                half = max(1, len(trimmed) // 2)
                shorter = trimmed[-half:]
                full_messages = [{"role": "system", "content": self._system_prompt}] + shorter
                try:
                    return self._provider.generate(full_messages, tools=tool_schemas)
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
        if len(chunks) == 1:
            self._messages.append({"role": "user", "content": user_input})
        else:
            self._messages.append({"role": "user", "content": user_input})

        # Track input tokens
        self._cumulative_tokens += estimate_tokens(user_input)

        for turn in range(self._max_turns):
            response_text = self._step_stream(on_chunk)
            if response_text is None:
                return "Error: Failed to get a response from the model."

            # Track output tokens
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

    def _step_stream(self, on_chunk: Optional[Callable[[str], None]] = None) -> str | None:
        """Single generation step with streaming support."""
        trimmed = self._context_window.trim_messages(self._messages, self._system_prompt)
        full_messages = [{"role": "system", "content": self._system_prompt}] + trimmed
        tool_schemas = self._tools.to_openai_schemas() if self._tools.names() else None

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
        """Parse tool calls from LLM response. Tries JSON blocks first, then regex fallback."""
        calls = []

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

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages.clear()
