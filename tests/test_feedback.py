"""Tests for the Feedback Flywheel module."""

import json
import pytest
import tempfile
from pathlib import Path

from src.core.feedback import (
    InferenceLog,
    ValidationCheck,
    ValidationResult,
    FeedbackStore,
    hash_prompt,
    validate_inference,
    _validate_json_parse,
    _validate_tool_calls,
    _validate_tool_exists,
    _validate_response_not_empty,
)


# =============================================================================
# InferenceLog Tests
# =============================================================================

class TestInferenceLog:
    """Test InferenceLog dataclass."""

    def test_basic_creation(self):
        log = InferenceLog(
            model="qwen2.5-coder-7b",
            prompt_hash="abc123",
            response="Hello world",
            latency_ms=500.0,
        )
        assert log.model == "qwen2.5-coder-7b"
        assert log.success is True
        assert log.latency_ms == 500.0

    def test_to_dict_serialization(self):
        log = InferenceLog(
            model="test",
            prompt_hash="hash",
            response="response",
            validations=[
                ValidationCheck("test_check", ValidationResult.PASS),
            ],
        )
        d = log.to_dict()
        assert d["model"] == "test"
        assert d["validations"][0]["result"] == "pass"

    def test_from_dict_deserialization(self):
        data = {
            "model": "test",
            "prompt_hash": "hash",
            "response": "response",
            "validations": [
                {"name": "json_parse", "result": "pass", "detail": ""},
            ],
            "success": True,
            "timestamp": 1234567890.0,
        }
        log = InferenceLog.from_dict(data)
        assert log.model == "test"
        assert log.validations[0].result == ValidationResult.PASS

    def test_all_validations_passed_true(self):
        log = InferenceLog(
            model="test", prompt_hash="h", response="r",
            validations=[
                ValidationCheck("a", ValidationResult.PASS),
                ValidationCheck("b", ValidationResult.PASS),
                ValidationCheck("c", ValidationResult.SKIP),
            ],
        )
        assert log.all_validations_passed is True

    def test_all_validations_passed_false(self):
        log = InferenceLog(
            model="test", prompt_hash="h", response="r",
            validations=[
                ValidationCheck("a", ValidationResult.PASS),
                ValidationCheck("b", ValidationResult.FAIL, "bad"),
            ],
        )
        assert log.all_validations_passed is False

    def test_failed_validations(self):
        log = InferenceLog(
            model="test", prompt_hash="h", response="r",
            validations=[
                ValidationCheck("a", ValidationResult.PASS),
                ValidationCheck("b", ValidationResult.FAIL, "error1"),
                ValidationCheck("c", ValidationResult.FAIL, "error2"),
            ],
        )
        failed = log.failed_validations
        assert len(failed) == 2
        assert failed[0].name == "b"


# =============================================================================
# Hash Prompt Tests
# =============================================================================

class TestHashPrompt:
    """Test prompt hashing."""

    def test_consistent_hash(self):
        msgs = [{"role": "user", "content": "hello"}]
        h1 = hash_prompt(msgs)
        h2 = hash_prompt(msgs)
        assert h1 == h2

    def test_different_prompts_different_hash(self):
        h1 = hash_prompt([{"role": "user", "content": "hello"}])
        h2 = hash_prompt([{"role": "user", "content": "world"}])
        assert h1 != h2

    def test_hash_length(self):
        h = hash_prompt([{"role": "user", "content": "test"}])
        assert len(h) == 16  # First 16 chars of SHA-256


# =============================================================================
# Validator Tests
# =============================================================================

class TestValidateJsonParse:
    """Test JSON parse validator."""

    def test_valid_json_object(self):
        result = _validate_json_parse('{"name": "read_file", "arguments": {}}')
        assert result.result == ValidationResult.PASS

    def test_valid_json_array(self):
        result = _validate_json_parse('[{"name": "test"}]')
        assert result.result == ValidationResult.PASS

    def test_invalid_json(self):
        result = _validate_json_parse('{"name": "test"')
        assert result.result == ValidationResult.FAIL

    def test_json_in_code_block(self):
        result = _validate_json_parse('Here is the result:\n```json\n{"key": "value"}\n```')
        assert result.result == ValidationResult.PASS

    def test_no_json(self):
        result = _validate_json_parse("Just a plain text response with no JSON.")
        assert result.result == ValidationResult.SKIP

    def test_empty_response(self):
        result = _validate_json_parse("")
        assert result.result == ValidationResult.SKIP


class TestValidateToolCalls:
    """Test tool call validator."""

    def test_valid_tool_call(self):
        result = _validate_tool_calls(
            [{"name": "read_file", "arguments": {"path": "test.py"}}],
        )
        assert result.result == ValidationResult.PASS

    def test_missing_name(self):
        result = _validate_tool_calls(
            [{"arguments": {"path": "test.py"}}],
        )
        assert result.result == ValidationResult.FAIL
        assert "missing 'name'" in result.detail

    def test_arguments_not_dict(self):
        result = _validate_tool_calls(
            [{"name": "test", "arguments": "not a dict"}],
        )
        assert result.result == ValidationResult.FAIL

    def test_missing_required_arg(self):
        schemas = {
            "read_file": {
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        }
        result = _validate_tool_calls(
            [{"name": "read_file", "arguments": {}}],
            tool_schemas=schemas,
        )
        assert result.result == ValidationResult.FAIL
        assert "missing required" in result.detail

    def test_unknown_arg(self):
        schemas = {
            "read_file": {
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        }
        result = _validate_tool_calls(
            [{"name": "read_file", "arguments": {"path": "x", "unknown": "y"}}],
            tool_schemas=schemas,
        )
        assert result.result == ValidationResult.FAIL
        assert "unknown arg" in result.detail

    def test_no_tool_calls(self):
        result = _validate_tool_calls([])
        assert result.result == ValidationResult.SKIP


class TestValidateToolExists:
    """Test tool existence validator."""

    def test_known_tool(self):
        result = _validate_tool_exists(
            [{"name": "read_file"}],
            known_tools={"read_file", "write_file"},
        )
        assert result.result == ValidationResult.PASS

    def test_unknown_tool(self):
        result = _validate_tool_exists(
            [{"name": "hack_mainframe"}],
            known_tools={"read_file", "write_file"},
        )
        assert result.result == ValidationResult.FAIL
        assert "hack_mainframe" in result.detail

    def test_no_known_tools(self):
        result = _validate_tool_exists(
            [{"name": "anything"}],
            known_tools=None,
        )
        assert result.result == ValidationResult.SKIP


class TestValidateResponseNotEmpty:
    """Test non-empty response validator."""

    def test_non_empty(self):
        result = _validate_response_not_empty("Hello")
        assert result.result == ValidationResult.PASS

    def test_empty(self):
        result = _validate_response_not_empty("")
        assert result.result == ValidationResult.FAIL

    def test_whitespace_only(self):
        result = _validate_response_not_empty("   \n\t  ")
        assert result.result == ValidationResult.FAIL


# =============================================================================
# validate_inference Integration Test
# =============================================================================

class TestValidateInference:
    """Test the validate_inference function."""

    def test_validates_all_checks(self):
        log = InferenceLog(
            model="test",
            prompt_hash="h",
            response='{"name": "read_file", "arguments": {"path": "x"}}',
            tool_calls=[{"name": "read_file", "arguments": {"path": "x"}}],
        )
        schemas = {
            "read_file": {
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        }
        result = validate_inference(log, tool_schemas=schemas, known_tools={"read_file"})

        assert len(result.validations) == 4
        assert result.all_validations_passed is True

    def test_catches_failures(self):
        log = InferenceLog(
            model="test",
            prompt_hash="h",
            response="",
            tool_calls=[{"arguments": {}}],  # Missing name
        )
        result = validate_inference(log)

        assert not result.all_validations_passed
        assert len(result.failed_validations) >= 1


# =============================================================================
# FeedbackStore Tests
# =============================================================================

class TestFeedbackStore:
    """Test FeedbackStore persistence and analysis."""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FeedbackStore(path=Path(tmpdir) / "feedback.jsonl")

    def test_record_and_count(self, store):
        log = InferenceLog(model="test", prompt_hash="h", response="ok")
        store.record(log)
        assert store.count() == 1

    def test_get_all(self, store):
        for i in range(3):
            store.record(InferenceLog(
                model=f"model-{i}", prompt_hash=f"h{i}", response=f"r{i}",
            ))
        logs = store.get_all()
        assert len(logs) == 3

    def test_clear(self, store):
        store.record(InferenceLog(model="t", prompt_hash="h", response="r"))
        store.clear()
        assert store.count() == 0

    def test_persistence(self, store):
        store.record(InferenceLog(model="persistent", prompt_hash="h", response="r"))
        new_store = FeedbackStore(path=store.path)
        assert new_store.count() == 1
        assert new_store.get_all()[0].model == "persistent"

    def test_stats_empty(self, store):
        stats = store.stats()
        assert stats["total_inferences"] == 0

    def test_stats_populated(self, store):
        for i in range(5):
            log = InferenceLog(
                model="qwen",
                prompt_hash=f"h{i}",
                response="ok",
                latency_ms=100.0 * (i + 1),
                success=i < 4,  # 4 successes, 1 failure
                tool_calls=[{"name": "read_file"}] if i % 2 == 0 else [],
            )
            store.record(log)

        stats = store.stats()
        assert stats["total_inferences"] == 5
        assert stats["success_rate"] == 0.8
        assert stats["avg_latency_ms"] == 300.0
        assert stats["model_usage"]["qwen"] == 5
        assert stats["tool_usage"].get("read_file", 0) == 3

    def test_failure_patterns(self, store):
        for i in range(5):
            log = InferenceLog(
                model="qwen",
                prompt_hash=f"h{i}",
                response="",
                validations=[
                    ValidationCheck("non_empty", ValidationResult.FAIL, "Empty response"),
                ],
            )
            store.record(log)

        patterns = store.failure_patterns(min_occurrences=3)
        assert len(patterns) >= 1
        assert patterns[0]["check"] == "non_empty"
        assert patterns[0]["occurrences"] == 5

    def test_model_comparison(self, store):
        for model in ["qwen", "qwen", "gpt4", "gpt4", "gpt4"]:
            log = InferenceLog(
                model=model,
                prompt_hash=f"h-{model}",
                response="ok",
                latency_ms=100 if model == "qwen" else 500,
                success=True,
            )
            store.record(log)

        comparison = store.model_comparison()
        assert "qwen" in comparison
        assert "gpt4" in comparison
        assert comparison["qwen"]["total_inferences"] == 2
        assert comparison["gpt4"]["total_inferences"] == 3
        assert comparison["qwen"]["avg_latency_ms"] == 100.0
