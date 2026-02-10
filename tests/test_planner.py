"""Tests for plan-then-execute architecture — all mocked, no real inference."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from src.core.planner import (
    ChunkedExecutor,
    PlanExecutor,
    PlanParser,
    PlanResult,
    Step,
    StepResult,
    StepStatus,
    StepType,
    TaskDecomposer,
    _filter_tools,
    _infer_step_type,
    _is_simple_task,
    _parse_tool_calls,
    should_use_planner,
)
from src.llm.base import ModelCapabilities
from src.tools.base import Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeTool(Tool):
    """Configurable fake tool for testing."""

    def __init__(self, name: str, result: str = "ok") -> None:
        self._name = name
        self._result = result

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Fake {self._name}"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, args: dict) -> str:
        return self._result


def _make_provider(responses: list[str], size_tier: str = "small") -> MagicMock:
    """Create a mock provider that returns canned responses."""
    provider = MagicMock()
    provider.generate = MagicMock(side_effect=responses)
    provider.capabilities.return_value = ModelCapabilities(
        context_length=4096,
        size_tier=size_tier,
        supports_tools=True,
    )
    return provider


def _make_registry(*tool_names: str) -> ToolRegistry:
    """Create a registry with fake tools."""
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(FakeTool(name))
    return registry


# ---------------------------------------------------------------------------
# StepType inference tests
# ---------------------------------------------------------------------------


class TestInferStepType:
    def test_read_file(self):
        assert _infer_step_type("Read the contents of main.py") == StepType.READ

    def test_write_file(self):
        assert _infer_step_type("Write the new function to utils.py") == StepType.WRITE

    def test_shell_command(self):
        assert _infer_step_type("Run pytest on the test suite") == StepType.SHELL

    def test_git_operation(self):
        assert _infer_step_type("Commit the changes with a descriptive message") == StepType.GIT

    def test_generate_code(self):
        assert _infer_step_type("Generate a helper function for parsing") == StepType.GENERATE

    def test_analyze_code(self):
        assert _infer_step_type("Analyze the current error handling") == StepType.ANALYZE

    def test_default_is_analyze(self):
        assert _infer_step_type("Something completely unrecognizable") == StepType.ANALYZE

    def test_case_insensitive(self):
        assert _infer_step_type("READ the config file") == StepType.READ

    def test_git_keywords(self):
        assert _infer_step_type("Stage the modified files") == StepType.GIT
        assert _infer_step_type("Checkout the feature branch") == StepType.GIT

    def test_shell_keywords(self):
        assert _infer_step_type("Install the dependencies with pip") == StepType.SHELL
        assert _infer_step_type("Execute the build script") == StepType.SHELL


# ---------------------------------------------------------------------------
# PlanParser tests
# ---------------------------------------------------------------------------


class TestPlanParser:
    def setup_method(self):
        self.parser = PlanParser()

    def test_basic_numbered_list(self):
        text = """1. Read the main.py file
2. Write a new function
3. Run the tests"""
        steps = self.parser.parse(text)
        assert len(steps) == 3
        assert steps[0].number == 1
        assert steps[0].description == "Read the main.py file"
        assert steps[1].number == 2
        assert steps[2].number == 3

    def test_parenthesis_format(self):
        text = """1) Read the file
2) Modify the function
3) Save changes"""
        steps = self.parser.parse(text)
        assert len(steps) == 3
        assert steps[0].description == "Read the file"

    def test_step_prefix_format(self):
        text = """Step 1: Read the file
Step 2: Edit the function
Step 3: Run tests"""
        steps = self.parser.parse(text)
        assert len(steps) == 3
        assert steps[0].description == "Read the file"

    def test_dash_format(self):
        text = """1- Read the file
2- Write the code
3- Test it"""
        steps = self.parser.parse(text)
        assert len(steps) == 3

    def test_infers_step_types(self):
        text = """1. Read the configuration file
2. Write a new test file
3. Run pytest
4. Commit the changes"""
        steps = self.parser.parse(text)
        assert steps[0].step_type == StepType.READ
        assert steps[1].step_type == StepType.WRITE
        assert steps[2].step_type == StepType.SHELL
        assert steps[3].step_type == StepType.GIT

    def test_assigns_relevant_tools(self):
        text = "1. Read the config file"
        steps = self.parser.parse(text)
        assert "read_file" in steps[0].relevant_tools

    def test_strips_trailing_period(self):
        text = "1. Read the file."
        steps = self.parser.parse(text)
        assert steps[0].description == "Read the file"

    def test_empty_input(self):
        assert self.parser.parse("") == []

    def test_no_numbered_lines(self):
        text = "This is just a paragraph with no steps."
        assert self.parser.parse(text) == []

    def test_deduplicates_step_numbers(self):
        text = """1. First thing
1. Duplicate
2. Second thing"""
        steps = self.parser.parse(text)
        assert len(steps) == 2
        assert steps[0].description == "First thing"

    def test_sorts_by_number(self):
        text = """3. Third step
1. First step
2. Second step"""
        steps = self.parser.parse(text)
        assert [s.number for s in steps] == [1, 2, 3]

    def test_handles_mixed_content(self):
        text = """Here's my plan:

1. Read the current implementation
2. Identify the bug

Some explanation here.

3. Fix the function
4. Run tests to verify"""
        steps = self.parser.parse(text)
        assert len(steps) == 4
        assert steps[0].description == "Read the current implementation"
        assert steps[3].description == "Run tests to verify"


# ---------------------------------------------------------------------------
# TaskDecomposer tests
# ---------------------------------------------------------------------------


class TestTaskDecomposer:
    def test_sends_focused_prompt(self):
        provider = _make_provider(["1. Do something\n2. Do another thing"])
        decomposer = TaskDecomposer(provider)
        result = decomposer.decompose("Fix the login bug")
        assert "1." in result
        # Verify provider was called with minimal context
        call_args = provider.generate.call_args
        messages = call_args[0][0]
        assert len(messages) == 2  # system + user only
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # No tools passed
        assert call_args[1].get("tools") is None or call_args.kwargs.get("tools") is None

    def test_uses_planning_provider_when_given(self):
        execution_provider = _make_provider([])
        planning_provider = _make_provider(["1. Step one"])
        decomposer = TaskDecomposer(execution_provider, planning_provider=planning_provider)
        decomposer.decompose("Do something")
        # Planning provider should have been called, not execution provider
        planning_provider.generate.assert_called_once()
        execution_provider.generate.assert_not_called()

    def test_falls_back_to_main_provider(self):
        provider = _make_provider(["1. Single step"])
        decomposer = TaskDecomposer(provider)
        decomposer.decompose("Simple task")
        provider.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Tool filtering tests
# ---------------------------------------------------------------------------


class TestToolFiltering:
    def test_filter_read_tools(self):
        registry = _make_registry("read_file", "write_file", "list_dir", "run_shell", "git_status")
        filtered = _filter_tools(registry, StepType.READ)
        names = filtered.names()
        assert "read_file" in names
        assert "list_dir" in names
        assert "write_file" not in names
        assert "run_shell" not in names

    def test_filter_write_tools(self):
        registry = _make_registry("read_file", "write_file", "list_dir", "run_shell")
        filtered = _filter_tools(registry, StepType.WRITE)
        names = filtered.names()
        assert "write_file" in names
        assert "read_file" in names
        assert "run_shell" not in names

    def test_filter_git_tools(self):
        registry = _make_registry("git_status", "git_diff", "git_commit", "read_file", "run_shell")
        filtered = _filter_tools(registry, StepType.GIT)
        names = filtered.names()
        assert "git_status" in names
        assert "git_commit" in names
        assert "run_shell" not in names

    def test_filter_shell_tools(self):
        registry = _make_registry("run_shell", "read_file", "write_file", "git_commit")
        filtered = _filter_tools(registry, StepType.SHELL)
        names = filtered.names()
        assert "run_shell" in names
        assert "read_file" in names
        assert "write_file" not in names
        assert "git_commit" not in names

    def test_filter_empty_registry(self):
        registry = ToolRegistry()
        filtered = _filter_tools(registry, StepType.READ)
        assert filtered.names() == []


# ---------------------------------------------------------------------------
# ChunkedExecutor tests
# ---------------------------------------------------------------------------


class TestChunkedExecutor:
    def test_executes_simple_step(self):
        provider = _make_provider(["I read the file and found the data."])
        registry = _make_registry("read_file", "list_dir")
        executor = ChunkedExecutor(provider, registry)

        steps = [Step(number=1, description="Read the config file", step_type=StepType.READ)]
        results = executor.execute_plan(steps, "Check config")

        assert len(results) == 1
        assert results[0].status == StepStatus.COMPLETED
        assert "read the file" in results[0].output.lower()

    def test_executes_multiple_steps(self):
        provider = _make_provider([
            "Read the file successfully.",
            "Wrote the new function.",
            "Tests passed.",
        ])
        registry = _make_registry("read_file", "write_file", "run_shell", "list_dir")
        executor = ChunkedExecutor(provider, registry)

        steps = [
            Step(number=1, description="Read the code", step_type=StepType.READ),
            Step(number=2, description="Write the fix", step_type=StepType.WRITE),
            Step(number=3, description="Run tests", step_type=StepType.SHELL),
        ]
        results = executor.execute_plan(steps, "Fix the bug")

        assert len(results) == 3
        assert all(r.status == StepStatus.COMPLETED for r in results)

    def test_step_with_tool_call(self):
        provider = _make_provider([
            '```json\n{"name": "read_file", "arguments": {"path": "test.py"}}\n```',
            "The file contains a test function.",
        ])
        registry = _make_registry("read_file", "list_dir")
        executor = ChunkedExecutor(provider, registry)

        steps = [Step(number=1, description="Read test.py", step_type=StepType.READ)]
        results = executor.execute_plan(steps, "Review tests")

        assert len(results) == 1
        assert results[0].status == StepStatus.COMPLETED

    def test_provider_error_marks_step_failed(self):
        provider = MagicMock()
        provider.generate.side_effect = RuntimeError("Model crashed")
        provider.capabilities.return_value = ModelCapabilities(context_length=4096, size_tier="small")
        registry = _make_registry("read_file")
        executor = ChunkedExecutor(provider, registry)

        steps = [Step(number=1, description="Read something", step_type=StepType.READ)]
        results = executor.execute_plan(steps, "Do stuff")

        assert results[0].status == StepStatus.FAILED
        assert "Model crashed" in results[0].error

    def test_continues_after_step_failure(self):
        provider = MagicMock()
        provider.generate.side_effect = [
            RuntimeError("First step fails"),
            "Second step succeeds.",
        ]
        provider.capabilities.return_value = ModelCapabilities(context_length=4096, size_tier="small")
        registry = _make_registry("read_file", "write_file")
        executor = ChunkedExecutor(provider, registry)

        steps = [
            Step(number=1, description="Read the file", step_type=StepType.READ),
            Step(number=2, description="Write the output", step_type=StepType.WRITE),
        ]
        results = executor.execute_plan(steps, "Two step task")

        assert results[0].status == StepStatus.FAILED
        assert results[1].status == StepStatus.COMPLETED

    def test_progress_callback(self):
        provider = _make_provider(["Done step 1.", "Done step 2."])
        registry = _make_registry("read_file")
        progress_calls = []

        def on_progress(step_num, total, desc):
            progress_calls.append((step_num, total, desc))

        executor = ChunkedExecutor(provider, registry, on_progress=on_progress)
        steps = [
            Step(number=1, description="First", step_type=StepType.READ),
            Step(number=2, description="Second", step_type=StepType.READ),
        ]
        executor.execute_plan(steps, "Test")

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, "First")
        assert progress_calls[1] == (2, 2, "Second")

    def test_step_output_callback(self):
        provider = _make_provider(["Step output here."])
        registry = _make_registry("read_file")
        outputs = []

        executor = ChunkedExecutor(
            provider, registry, on_step_output=lambda s: outputs.append(s)
        )
        steps = [Step(number=1, description="Do it", step_type=StepType.READ)]
        executor.execute_plan(steps, "Test")

        assert len(outputs) == 1
        assert "Step output" in outputs[0]

    def test_fresh_context_per_step(self):
        """Verify each step starts with a clean message list (no carryover)."""
        call_messages = []

        def capture_generate(messages, **kwargs):
            # Record the messages sent to provider
            call_messages.append([m.copy() for m in messages])
            return "Done."

        provider = MagicMock()
        provider.generate.side_effect = capture_generate
        provider.capabilities.return_value = ModelCapabilities(context_length=4096, size_tier="small")
        registry = _make_registry("read_file")

        executor = ChunkedExecutor(provider, registry)
        steps = [
            Step(number=1, description="Step one", step_type=StepType.READ),
            Step(number=2, description="Step two", step_type=StepType.READ),
        ]
        executor.execute_plan(steps, "Test isolation")

        # Each call should have system + 1 user message (fresh context)
        for msgs in call_messages:
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert len(msgs) == 2  # No carryover from previous step


# ---------------------------------------------------------------------------
# PlanExecutor (full pipeline) tests
# ---------------------------------------------------------------------------


class TestPlanExecutor:
    def test_full_pipeline(self):
        # Decomposer returns a plan, then executor handles each step
        # Task must be complex enough to trigger planning (>10 words or multi-action)
        provider = _make_provider([
            "1. Read the file\n2. Fix the bug\n3. Run tests",  # decomposer output
            "Read the file content.",  # step 1 execution
            "Fixed the bug.",  # step 2 execution
            "All tests pass.",  # step 3 execution
        ])
        registry = _make_registry("read_file", "write_file", "run_shell", "list_dir")

        executor = PlanExecutor(provider, registry)
        result = executor.run("Read the auth module then fix the login bug and run the test suite to verify")

        assert isinstance(result, PlanResult)
        assert len(result.steps) == 3
        assert len(result.results) == 3
        assert result.success

    def test_hybrid_mode_different_providers(self):
        planning_provider = _make_provider([
            "1. Read config\n2. Update settings",
        ], size_tier="large")
        execution_provider = _make_provider([
            "Config contents loaded.",
            "Settings updated.",
        ], size_tier="small")

        registry = _make_registry("read_file", "write_file")
        executor = PlanExecutor(
            execution_provider, registry, planning_provider=planning_provider
        )
        result = executor.run("Read the current config file then update all the database settings to use the new credentials")

        # Planning provider should have been called once (decompose)
        planning_provider.generate.assert_called_once()
        # Execution provider handles steps
        assert execution_provider.generate.call_count == 2
        assert result.success

    def test_fallback_when_parser_finds_no_steps(self):
        # LLM returns garbage that parser can't extract
        # Task must be complex enough to trigger planning
        provider = _make_provider([
            "I don't know how to make a plan.",  # decomposer output
            "Ok I did the thing.",  # fallback single-step execution
        ])
        registry = _make_registry("read_file")
        executor = PlanExecutor(provider, registry)
        result = executor.run("Read all the configuration files then analyze them and create a summary report with recommendations")

        # Should fallback to a single step (parser found nothing)
        assert len(result.steps) == 1
        assert len(result.results) == 1

    def test_plan_result_summary(self):
        provider = _make_provider([
            "1. Read file\n2. Write output",
            "Read done.",
            "Write done.",
        ])
        registry = _make_registry("read_file", "write_file")
        executor = PlanExecutor(provider, registry)
        result = executor.run("Read the input data file then write the processed output to a new file")

        summary = result.summary
        assert "[OK]" in summary
        assert "Step 1" in summary
        assert "Step 2" in summary

    def test_plan_result_with_failure(self):
        provider = MagicMock()
        provider.generate.side_effect = [
            "1. Read file\n2. Crash here",
            "Read done.",
            RuntimeError("Boom"),
        ]
        provider.capabilities.return_value = ModelCapabilities(context_length=4096, size_tier="small")
        registry = _make_registry("read_file")
        executor = PlanExecutor(provider, registry)
        result = executor.run("Read all the source code files then compile and test the entire project")

        assert not result.success
        assert "[FAIL]" in result.summary


# ---------------------------------------------------------------------------
# should_use_planner tests
# ---------------------------------------------------------------------------


class TestShouldUsePlanner:
    def test_small_model_uses_planner(self):
        provider = _make_provider([], size_tier="small")
        assert should_use_planner(provider) is True

    def test_medium_model_uses_planner(self):
        provider = _make_provider([], size_tier="medium")
        assert should_use_planner(provider) is True

    def test_large_model_skips_planner(self):
        provider = _make_provider([], size_tier="large")
        assert should_use_planner(provider) is False


# ---------------------------------------------------------------------------
# Tier-aware planning profile tests
# ---------------------------------------------------------------------------


class TestTierAwarePlanning:
    def test_small_model_caps_steps_at_3(self):
        """Small model parser should enforce max 3 steps."""
        provider = _make_provider([
            "1. Step one\n2. Step two\n3. Step three\n4. Step four\n5. Step five",
            "Done 1.", "Done 2.", "Done 3.",
        ], size_tier="small")
        registry = _make_registry("read_file")
        executor = PlanExecutor(provider, registry)
        result = executor.run("Big task")
        # Small model: max 3 steps even though LLM produced 5
        assert len(result.steps) <= 3

    def test_large_model_allows_more_steps(self):
        provider = _make_provider([
            "1. A\n2. B\n3. C\n4. D\n5. E\n6. F\n7. G",
            "Done.", "Done.", "Done.", "Done.", "Done.", "Done.", "Done.",
        ], size_tier="large")
        registry = _make_registry("read_file")
        executor = PlanExecutor(provider, registry)
        result = executor.run("Read all config files then analyze each one and write a summary then run tests and commit changes")
        assert len(result.steps) <= 7
        assert len(result.steps) >= 5  # Large model should keep more steps

    def test_small_model_fewer_step_turns(self):
        """Small model executor should use fewer turns per step."""
        # Produce tool calls that would exhaust turns
        tool_response = '```json\n{"name": "read_file", "arguments": {"path": "x"}}\n```'
        provider = _make_provider([tool_response] * 10, size_tier="small")
        registry = _make_registry("read_file")
        executor = ChunkedExecutor(provider, registry)
        # Small model: max_step_turns=3, so it should stop early
        steps = [Step(number=1, description="Read things", step_type=StepType.READ)]
        results = executor.execute_plan(steps, "Test")
        # Should complete (not error) even with many tool calls
        assert len(results) == 1
        # Provider called at most max_step_turns times (3 for small)
        assert provider.generate.call_count <= 3

    def test_planning_prompt_includes_tool_names(self):
        """Planning prompt should list available tools."""
        provider = _make_provider(["1. List the files"])
        decomposer = TaskDecomposer(provider, tool_names=["read_file", "list_dir"])
        decomposer.decompose("Show me the files")
        call_args = provider.generate.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "read_file" in user_msg
        assert "list_dir" in user_msg

    def test_execution_prompt_includes_tool_names(self):
        """Execution prompt should tell the model which tools are available."""
        call_messages = []

        def capture_generate(messages, **kwargs):
            call_messages.append([m.copy() for m in messages])
            return "Done."

        provider = MagicMock()
        provider.generate.side_effect = capture_generate
        provider.capabilities.return_value = ModelCapabilities(context_length=4096, size_tier="small")
        registry = _make_registry("read_file", "list_dir")
        executor = ChunkedExecutor(provider, registry)

        steps = [Step(number=1, description="List files", step_type=StepType.READ)]
        executor.execute_plan(steps, "Test")

        system_msg = call_messages[0][0]["content"]
        assert "read_file" in system_msg or "list_dir" in system_msg


# ---------------------------------------------------------------------------
# _parse_tool_calls tests
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_json_code_block(self):
        text = '```json\n{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n```'
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "/tmp/x"

    def test_inline_json(self):
        text = 'I will call {"name": "list_dir", "arguments": {"path": "/tmp"}} now'
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_dir"

    def test_no_tool_calls(self):
        assert _parse_tool_calls("Just a regular response.") == []

    def test_invalid_json_ignored(self):
        text = '```json\n{not valid json}\n```'
        assert _parse_tool_calls(text) == []

    def test_raw_json_from_gbnf(self):
        """GBNF-constrained output is raw JSON (no code blocks)."""
        text = '{"name": "read_file", "arguments": {"path": "/tmp/test.py"}}'
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"
        assert calls[0]["arguments"]["path"] == "/tmp/test.py"

    def test_raw_json_with_whitespace(self):
        """GBNF output may have leading/trailing whitespace."""
        text = '  {"name": "list_dir", "arguments": {"path": "."}}  '
        calls = _parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "list_dir"


# ---------------------------------------------------------------------------
# Complexity heuristic tests
# ---------------------------------------------------------------------------


class TestSimpleTaskDetection:
    def test_short_task_is_simple(self):
        assert _is_simple_task("List the files") is True

    def test_very_short_task_is_simple(self):
        assert _is_simple_task("What files are here?") is True

    def test_single_action_is_simple(self):
        assert _is_simple_task("Read the README.md file in the current directory") is True

    def test_multi_action_with_then_is_complex(self):
        assert _is_simple_task("Read the file then write a summary and save it") is False

    def test_multi_action_with_and_then_is_complex(self):
        assert _is_simple_task("Check git status and then commit the changes and push") is False

    def test_simple_task_skips_llm_planning(self):
        """PlanExecutor should skip LLM decomposer for simple tasks."""
        provider = _make_provider([
            "Done.",  # Only execution, no planning call
        ], size_tier="small")
        registry = _make_registry("read_file", "list_dir")
        executor = PlanExecutor(provider, registry)
        result = executor.run("List the files")
        # Should have 1 step (created directly, not from LLM)
        assert len(result.steps) == 1
        # Provider called once (execution only, no decompose call)
        assert provider.generate.call_count == 1

    def test_complex_task_uses_llm_planning(self):
        """PlanExecutor should call LLM decomposer for complex tasks."""
        provider = _make_provider([
            "1. Read the file\n2. Analyze it\n3. Write summary",  # Planning call
            "Done 1.", "Done 2.", "Done 3.",  # Execution calls
        ], size_tier="small")
        registry = _make_registry("read_file", "write_file", "list_dir")
        executor = PlanExecutor(provider, registry)
        result = executor.run("Read all Python files then write a summary of each")
        # Should have multiple steps from LLM decomposition
        assert len(result.steps) >= 2
        # Provider called more than once (planning + execution)
        assert provider.generate.call_count >= 2


# ---------------------------------------------------------------------------
# Tool narrowing tests
# ---------------------------------------------------------------------------


class TestToolNarrowing:
    def test_narrows_to_read_file_when_mentioned(self):
        registry = _make_registry("read_file", "list_dir", "git_status")
        filtered = _filter_tools(registry, StepType.READ, "read_file(config.py)")
        assert filtered.names() == ["read_file"]

    def test_narrows_to_list_dir_when_mentioned(self):
        registry = _make_registry("read_file", "list_dir", "git_status")
        filtered = _filter_tools(registry, StepType.READ, "Use list_dir to see the contents")
        assert filtered.names() == ["list_dir"]

    def test_no_narrowing_when_no_tool_mentioned(self):
        registry = _make_registry("read_file", "list_dir", "git_status")
        filtered = _filter_tools(registry, StepType.READ, "Check the file contents")
        # All READ-relevant tools present in registry
        assert len(filtered.names()) >= 2

    def test_no_narrowing_with_empty_description(self):
        registry = _make_registry("read_file", "list_dir")
        filtered = _filter_tools(registry, StepType.READ, "")
        assert len(filtered.names()) >= 2
