"""Feedback Flywheel — structured inference telemetry and automated validation.

Logs every inference with structured telemetry: model used, prompt hash,
response content, tool calls, latency, and success indicators. Automated
validators check whether JSON parsed, tool calls matched schemas, and
code compiled. This data feeds prompt optimization.

Storage: ~/.animus/data/feedback.jsonl

Implementation Principle: 100% hardcoded validation. No LLM involvement
in the feedback pipeline itself.

Usage:
    store = FeedbackStore()

    # Log an inference
    log = InferenceLog(
        model="qwen2.5-coder-7b",
        prompt_hash="abc123",
        response="...",
        tool_calls=[{"name": "read_file", "arguments": {"path": "x"}}],
        latency_ms=1200,
    )
    log = validate_inference(log, tool_schemas=schemas)
    store.record(log)

    # Analyze patterns
    stats = store.stats()
    failures = store.failure_patterns()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_PATH = Path.home() / ".animus" / "data" / "feedback.jsonl"


class ValidationResult(str, Enum):
    """Result of an automated validation check."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # Validation not applicable


@dataclass
class ValidationCheck:
    """A single validation check result."""
    name: str  # e.g. "json_parse", "tool_schema", "code_compile"
    result: ValidationResult
    detail: str = ""  # Error message if failed


@dataclass
class InferenceLog:
    """Structured telemetry for a single LLM inference."""

    # Identity
    model: str
    prompt_hash: str  # SHA-256 of the prompt (for dedup)

    # Response
    response: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"

    # Performance
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Validation results (populated by validate_inference)
    validations: list[ValidationCheck] = field(default_factory=list)

    # Outcome
    success: bool = True  # Overall: did the agent accomplish its goal?
    error: Optional[str] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    turn_number: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def all_validations_passed(self) -> bool:
        """Check if all applicable validations passed."""
        applicable = [v for v in self.validations if v.result != ValidationResult.SKIP]
        return all(v.result == ValidationResult.PASS for v in applicable)

    @property
    def failed_validations(self) -> list[ValidationCheck]:
        """Get list of failed validations."""
        return [v for v in self.validations if v.result == ValidationResult.FAIL]

    def to_dict(self) -> dict:
        """Serialize to dict."""
        d = asdict(self)
        # Convert enums to strings
        d["validations"] = [
            {"name": v.name, "result": v.result.value, "detail": v.detail}
            for v in self.validations
        ]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> InferenceLog:
        """Deserialize from dict."""
        validations = []
        for v in data.pop("validations", []):
            validations.append(ValidationCheck(
                name=v["name"],
                result=ValidationResult(v["result"]),
                detail=v.get("detail", ""),
            ))

        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        fields["validations"] = validations
        return cls(**fields)


def hash_prompt(messages: list[dict[str, str]]) -> str:
    """Create a stable hash of a prompt for deduplication.

    Args:
        messages: List of message dicts with role/content.

    Returns:
        Hex string SHA-256 hash.
    """
    content = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# Automated Validators
# =============================================================================

def _validate_json_parse(response: str) -> ValidationCheck:
    """Check if the response contains valid JSON.

    Looks for JSON objects or arrays in the response text.
    """
    # Try to find and parse JSON in the response
    text = response.strip()

    # Direct JSON
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return ValidationCheck("json_parse", ValidationResult.PASS)
        except json.JSONDecodeError as e:
            return ValidationCheck("json_parse", ValidationResult.FAIL, str(e))

    # JSON in code blocks
    if "```json" in text or "```" in text:
        import re
        blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        for block in blocks:
            block = block.strip()
            if block.startswith("{") or block.startswith("["):
                try:
                    json.loads(block)
                    return ValidationCheck("json_parse", ValidationResult.PASS)
                except json.JSONDecodeError as e:
                    return ValidationCheck("json_parse", ValidationResult.FAIL, str(e))

    # No JSON expected — skip
    return ValidationCheck("json_parse", ValidationResult.SKIP, "No JSON found in response")


def _validate_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_schemas: Optional[dict[str, dict]] = None,
) -> ValidationCheck:
    """Check if tool calls are well-formed and match schemas.

    Args:
        tool_calls: List of tool call dicts.
        tool_schemas: Optional dict of tool_name -> parameter schema.
    """
    if not tool_calls:
        return ValidationCheck("tool_schema", ValidationResult.SKIP, "No tool calls")

    errors = []
    for i, tc in enumerate(tool_calls):
        # Check required fields
        if "name" not in tc:
            errors.append(f"Tool call {i}: missing 'name'")
            continue

        name = tc["name"]
        args = tc.get("arguments", {})

        # Check arguments is a dict
        if not isinstance(args, dict):
            errors.append(f"Tool call {i} ({name}): arguments is not a dict")
            continue

        # Check against schema if available
        if tool_schemas and name in tool_schemas:
            schema = tool_schemas[name]
            required = schema.get("required", [])
            properties = schema.get("properties", {})

            for req in required:
                if req not in args:
                    errors.append(f"Tool call {i} ({name}): missing required arg '{req}'")

            for arg_name, arg_val in args.items():
                if arg_name not in properties:
                    errors.append(f"Tool call {i} ({name}): unknown arg '{arg_name}'")

    if errors:
        return ValidationCheck("tool_schema", ValidationResult.FAIL, "; ".join(errors))

    return ValidationCheck("tool_schema", ValidationResult.PASS)


def _validate_tool_exists(
    tool_calls: list[dict[str, Any]],
    known_tools: Optional[set[str]] = None,
) -> ValidationCheck:
    """Check if tool names reference known tools.

    Args:
        tool_calls: List of tool call dicts.
        known_tools: Set of known tool names.
    """
    if not tool_calls or not known_tools:
        return ValidationCheck("tool_exists", ValidationResult.SKIP)

    unknown = []
    for tc in tool_calls:
        name = tc.get("name", "")
        if name and name not in known_tools:
            unknown.append(name)

    if unknown:
        return ValidationCheck(
            "tool_exists", ValidationResult.FAIL,
            f"Unknown tools: {', '.join(unknown)}",
        )

    return ValidationCheck("tool_exists", ValidationResult.PASS)


def _validate_response_not_empty(response: str) -> ValidationCheck:
    """Check that the response is not empty or whitespace-only."""
    if not response or not response.strip():
        return ValidationCheck("non_empty", ValidationResult.FAIL, "Empty response")
    return ValidationCheck("non_empty", ValidationResult.PASS)


def validate_inference(
    log: InferenceLog,
    tool_schemas: Optional[dict[str, dict]] = None,
    known_tools: Optional[set[str]] = None,
) -> InferenceLog:
    """Run all automated validators on an inference log.

    Mutates the log's validations list in-place.

    Args:
        log: The inference log to validate.
        tool_schemas: Optional tool name -> parameter schema mapping.
        known_tools: Optional set of known tool names.

    Returns:
        The same log with validations populated.
    """
    log.validations = [
        _validate_response_not_empty(log.response),
        _validate_json_parse(log.response),
        _validate_tool_calls(log.tool_calls, tool_schemas),
        _validate_tool_exists(log.tool_calls, known_tools),
    ]

    return log


# =============================================================================
# Feedback Store
# =============================================================================

class FeedbackStore:
    """JSONL-based store for inference telemetry.

    Stores inference logs for analysis and prompt optimization.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
    ):
        self._path = path or DEFAULT_FEEDBACK_PATH
        self._cache: Optional[list[InferenceLog]] = None

    @property
    def path(self) -> Path:
        return self._path

    def record(self, log: InferenceLog) -> None:
        """Append an inference log to the store."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log.to_dict()) + "\n")

        self._cache = None

    def get_all(self) -> list[InferenceLog]:
        """Get all stored logs."""
        return list(self._load_all())

    def count(self) -> int:
        """Count total logs."""
        return len(self._load_all())

    def clear(self) -> None:
        """Clear all logs."""
        if self._path.exists():
            self._path.unlink()
        self._cache = None

    def stats(self) -> dict:
        """Get aggregate statistics across all logs.

        Returns dict with:
        - total_inferences, success_rate, avg_latency_ms
        - validation_pass_rates per check name
        - model_usage counts
        - tool_usage counts
        """
        logs = self._load_all()
        if not logs:
            return {"total_inferences": 0}

        total = len(logs)
        successes = sum(1 for log in logs if log.success)
        latencies = [log.latency_ms for log in logs if log.latency_ms > 0]

        # Validation pass rates
        val_counts: dict[str, dict[str, int]] = {}
        for log in logs:
            for v in log.validations:
                if v.name not in val_counts:
                    val_counts[v.name] = {"pass": 0, "fail": 0, "skip": 0}
                val_counts[v.name][v.result.value] += 1

        val_rates = {}
        for name, counts in val_counts.items():
            applicable = counts["pass"] + counts["fail"]
            if applicable > 0:
                val_rates[name] = counts["pass"] / applicable
            else:
                val_rates[name] = None

        # Model usage
        model_counts: dict[str, int] = {}
        for log in logs:
            model_counts[log.model] = model_counts.get(log.model, 0) + 1

        # Tool usage
        tool_counts: dict[str, int] = {}
        for log in logs:
            for tc in log.tool_calls:
                name = tc.get("name", "unknown")
                tool_counts[name] = tool_counts.get(name, 0) + 1

        return {
            "total_inferences": total,
            "success_rate": successes / total if total else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "validation_pass_rates": val_rates,
            "model_usage": model_counts,
            "tool_usage": tool_counts,
        }

    def failure_patterns(
        self,
        min_occurrences: int = 2,
    ) -> list[dict[str, Any]]:
        """Identify recurring failure patterns.

        Groups failures by validation check and error detail.

        Args:
            min_occurrences: Minimum times a pattern must occur.

        Returns:
            List of pattern dicts sorted by frequency.
        """
        logs = self._load_all()

        patterns: dict[str, int] = {}
        pattern_details: dict[str, dict] = {}

        for log in logs:
            for v in log.validations:
                if v.result == ValidationResult.FAIL:
                    key = f"{v.name}:{v.detail[:100]}"
                    patterns[key] = patterns.get(key, 0) + 1
                    if key not in pattern_details:
                        pattern_details[key] = {
                            "check": v.name,
                            "detail": v.detail,
                            "models": set(),
                        }
                    pattern_details[key]["models"].add(log.model)

        result = []
        for key, count in sorted(patterns.items(), key=lambda x: -x[1]):
            if count >= min_occurrences:
                info = pattern_details[key]
                result.append({
                    "check": info["check"],
                    "detail": info["detail"],
                    "occurrences": count,
                    "models": list(info["models"]),
                })

        return result

    def model_comparison(self) -> dict[str, dict[str, Any]]:
        """Compare performance across models.

        Returns dict of model_name -> {success_rate, avg_latency, validation_rates}.
        """
        logs = self._load_all()

        by_model: dict[str, list[InferenceLog]] = {}
        for log in logs:
            by_model.setdefault(log.model, []).append(log)

        result = {}
        for model, model_logs in by_model.items():
            total = len(model_logs)
            successes = sum(1 for l in model_logs if l.success)
            latencies = [l.latency_ms for l in model_logs if l.latency_ms > 0]

            val_pass: dict[str, int] = {}
            val_total: dict[str, int] = {}
            for l in model_logs:
                for v in l.validations:
                    if v.result != ValidationResult.SKIP:
                        val_total[v.name] = val_total.get(v.name, 0) + 1
                        if v.result == ValidationResult.PASS:
                            val_pass[v.name] = val_pass.get(v.name, 0) + 1

            val_rates = {}
            for name, t in val_total.items():
                val_rates[name] = val_pass.get(name, 0) / t if t else 0

            result[model] = {
                "total_inferences": total,
                "success_rate": successes / total if total else 0,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "validation_pass_rates": val_rates,
            }

        return result

    def _load_all(self) -> list[InferenceLog]:
        """Load all logs from disk (cached)."""
        if self._cache is not None:
            return self._cache

        logs = []
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        logs.append(InferenceLog.from_dict(data))
                    except (json.JSONDecodeError, TypeError, KeyError):
                        logger.warning("Skipping malformed feedback log")

        self._cache = logs
        return logs
