"""Tests for GBNF grammar constraint building — all mocked, no llama-cpp-python required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.grammar import build_grammar, build_tool_call_schema
from src.tools.base import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeTool(Tool):
    """Configurable fake tool for testing."""

    def __init__(self, name: str, params: dict | None = None) -> None:
        self._name = name
        self._params = params or {"type": "object", "properties": {}, "required": []}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def parameters(self) -> dict:
        return self._params

    def execute(self, args: dict) -> str:
        return "ok"


# ---------------------------------------------------------------------------
# build_tool_call_schema tests
# ---------------------------------------------------------------------------


class TestBuildToolCallSchema:
    def test_empty_tools_returns_generic_object(self):
        schema = build_tool_call_schema([])
        assert schema == {"type": "object"}

    def test_single_tool_constrains_name_and_arguments(self):
        tool = FakeTool("read_file", {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        })
        schema = build_tool_call_schema([tool])

        assert schema["type"] == "object"
        assert schema["required"] == ["name", "arguments"]
        # Name constrained to single enum value
        assert schema["properties"]["name"]["enum"] == ["read_file"]
        # Arguments use the tool's full parameter schema
        assert schema["properties"]["arguments"]["properties"]["path"]["type"] == "string"
        assert schema["properties"]["arguments"]["required"] == ["path"]

    def test_multiple_tools_uses_name_enum(self):
        tools = [
            FakeTool("read_file"),
            FakeTool("list_dir"),
            FakeTool("write_file"),
        ]
        schema = build_tool_call_schema(tools)

        assert schema["type"] == "object"
        assert schema["required"] == ["name", "arguments"]
        assert set(schema["properties"]["name"]["enum"]) == {"read_file", "list_dir", "write_file"}
        # Multiple tools: generic object for arguments
        assert schema["properties"]["arguments"] == {"type": "object"}

    def test_single_tool_preserves_full_parameter_schema(self):
        params = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "max_lines": {"type": "integer", "description": "Max lines"},
            },
            "required": ["path"],
        }
        tool = FakeTool("read_file", params)
        schema = build_tool_call_schema([tool])

        args_schema = schema["properties"]["arguments"]
        assert "path" in args_schema["properties"]
        assert "max_lines" in args_schema["properties"]
        assert args_schema["required"] == ["path"]

    def test_schema_is_valid_json_schema_structure(self):
        tools = [FakeTool("read_file"), FakeTool("list_dir")]
        schema = build_tool_call_schema(tools)

        # Should be a valid JSON Schema object
        assert "type" in schema
        assert "properties" in schema
        assert "required" in schema
        assert "name" in schema["properties"]
        assert "arguments" in schema["properties"]


# ---------------------------------------------------------------------------
# build_grammar tests
# ---------------------------------------------------------------------------


class TestBuildGrammar:
    def test_empty_tools_returns_none(self):
        assert build_grammar([]) is None

    def test_returns_none_when_llama_cpp_not_installed(self):
        """Without llama-cpp-python, build_grammar should return None gracefully."""
        tools = [FakeTool("read_file")]

        import importlib
        import src.core.grammar as grammar_mod

        with patch.dict("sys.modules", {"llama_cpp": None}):
            # Force re-import to hit ImportError
            importlib.reload(grammar_mod)
            result = grammar_mod.build_grammar(tools)
            assert result is None

        # Restore module state
        importlib.reload(grammar_mod)

    def test_returns_grammar_when_llama_cpp_available(self):
        """When llama-cpp-python is available, build_grammar returns a grammar object."""
        tools = [FakeTool("read_file")]
        result = build_grammar(tools)
        # If llama_cpp is installed, we get a real grammar object; if not, None.
        # We test the mocked path separately, so just verify it doesn't crash.
        # On this machine llama_cpp IS installed, so we expect a grammar.
        try:
            import llama_cpp  # noqa: F401
            assert result is not None
        except ImportError:
            assert result is None

    def test_grammar_schema_passed_correctly(self):
        """Verify the JSON schema string passed to LlamaGrammar.from_json_schema."""
        import json

        mock_llama_grammar_cls = MagicMock()
        mock_llama_grammar_cls.from_json_schema.return_value = MagicMock()

        tools = [FakeTool("list_dir")]

        with patch("src.core.grammar.LlamaGrammar", mock_llama_grammar_cls, create=True):
            with patch.dict("sys.modules", {"llama_cpp": MagicMock(LlamaGrammar=mock_llama_grammar_cls)}):
                import importlib
                import src.core.grammar as grammar_mod
                importlib.reload(grammar_mod)
                grammar_mod.build_grammar(tools)
                importlib.reload(grammar_mod)

        mock_llama_grammar_cls.from_json_schema.assert_called_once()
        call_args = mock_llama_grammar_cls.from_json_schema.call_args[0][0]
        schema = json.loads(call_args)
        assert schema["properties"]["name"]["enum"] == ["list_dir"]

    def test_returns_none_on_grammar_conversion_failure(self):
        """If from_json_schema raises, build_grammar returns None."""
        mock_llama_grammar_cls = MagicMock()
        mock_llama_grammar_cls.from_json_schema.side_effect = RuntimeError("GBNF conversion failed")

        tools = [FakeTool("read_file")]

        import importlib
        import src.core.grammar as grammar_mod

        with patch.dict("sys.modules", {"llama_cpp": MagicMock(LlamaGrammar=mock_llama_grammar_cls)}):
            importlib.reload(grammar_mod)
            result = grammar_mod.build_grammar(tools)

        importlib.reload(grammar_mod)
        assert result is None


# ---------------------------------------------------------------------------
# NativeProvider grammar passthrough tests
# ---------------------------------------------------------------------------


class TestNativeProviderGrammar:
    def test_grammar_kwarg_passed_to_create_chat_completion(self):
        """NativeProvider should pass grammar= to create_chat_completion when provided."""
        from src.llm.native import NativeProvider

        provider = NativeProvider(model_path="/fake/model.gguf")

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "result"}}]
        }
        provider._model = mock_model  # Bypass lazy loading

        mock_grammar = MagicMock(name="grammar")
        result = provider.generate(
            [{"role": "user", "content": "test"}],
            grammar=mock_grammar,
        )

        call_kwargs = mock_model.create_chat_completion.call_args[1]
        assert call_kwargs.get("grammar") is mock_grammar
        assert result == "result"

    def test_no_grammar_kwarg_when_none(self):
        """When grammar is None, it should not be passed to create_chat_completion."""
        from src.llm.native import NativeProvider

        provider = NativeProvider(model_path="/fake/model.gguf")

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "result"}}]
        }
        provider._model = mock_model

        provider.generate(
            [{"role": "user", "content": "test"}],
            grammar=None,
        )

        call_kwargs = mock_model.create_chat_completion.call_args[1]
        assert "grammar" not in call_kwargs

    def test_no_grammar_kwarg_when_absent(self):
        """When grammar is not passed at all, it should not appear."""
        from src.llm.native import NativeProvider

        provider = NativeProvider(model_path="/fake/model.gguf")

        mock_model = MagicMock()
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "result"}}]
        }
        provider._model = mock_model

        provider.generate([{"role": "user", "content": "test"}])

        call_kwargs = mock_model.create_chat_completion.call_args[1]
        assert "grammar" not in call_kwargs


# ---------------------------------------------------------------------------
# Planner integration: grammar passed through to provider
# ---------------------------------------------------------------------------


class TestPlannerGrammarIntegration:
    def test_executor_passes_grammar_to_provider(self):
        """ChunkedExecutor should build grammar and pass it to provider.generate()."""
        from src.core.planner import ChunkedExecutor, Step, StepType
        from src.llm.base import ModelCapabilities

        generate_kwargs_list = []

        def capture_generate(messages, **kwargs):
            generate_kwargs_list.append(kwargs)
            return "Done."

        provider = MagicMock()
        provider.generate.side_effect = capture_generate
        provider.capabilities.return_value = ModelCapabilities(
            context_length=4096, size_tier="small"
        )

        registry = ToolRegistry()
        registry.register(FakeTool("read_file", {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }))

        executor = ChunkedExecutor(provider, registry)
        steps = [Step(number=1, description="Read the file", step_type=StepType.READ)]
        executor.execute_plan(steps, "Test grammar")

        # Verify grammar was passed (will be None since llama_cpp not installed,
        # but the kwarg should be present)
        assert len(generate_kwargs_list) >= 1
        assert "grammar" in generate_kwargs_list[0]

    def test_executor_grammar_is_none_without_llama_cpp(self):
        """Without llama-cpp-python, grammar should be None (graceful degradation)."""
        import importlib

        from src.core.planner import ChunkedExecutor, Step, StepType
        from src.llm.base import ModelCapabilities

        grammar_values = []

        def capture_generate(messages, **kwargs):
            grammar_values.append(kwargs.get("grammar"))
            return "Done."

        provider = MagicMock()
        provider.generate.side_effect = capture_generate
        provider.capabilities.return_value = ModelCapabilities(
            context_length=4096, size_tier="small"
        )

        registry = ToolRegistry()
        registry.register(FakeTool("read_file"))

        # Simulate llama_cpp not being installed
        import src.core.grammar as grammar_mod

        with patch.dict("sys.modules", {"llama_cpp": None}):
            importlib.reload(grammar_mod)
            executor = ChunkedExecutor(provider, registry)
            steps = [Step(number=1, description="Read stuff", step_type=StepType.READ)]
            executor.execute_plan(steps, "Test")

        importlib.reload(grammar_mod)

        # Grammar should be None since llama_cpp was simulated as missing
        assert grammar_values[0] is None
