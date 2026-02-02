"""Triangulated Verification for agent output validation.

This module provides a multi-layer verification system that combines:
1. Rule-based checks (fast, deterministic)
2. LLM evaluation (flexible, contextual)
3. Human escalation (when confidence is low)

The goal is to catch errors early with cheap rule checks, use LLM
for nuanced validation, and escalate to humans when necessary.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Awaitable

from src.core.decision import OutcomeStatus


class VerificationLevel(Enum):
    """Level of verification confidence."""
    HIGH = "high"           # Strong confidence in result
    MEDIUM = "medium"       # Moderate confidence
    LOW = "low"             # Low confidence, may need escalation
    UNCERTAIN = "uncertain" # Cannot determine, needs escalation


class VerificationSource(Enum):
    """Source of the verification result."""
    RULE = "rule"           # Rule-based check
    LLM = "llm"             # LLM evaluation
    HUMAN = "human"         # Human verification


@dataclass
class VerificationResult:
    """Result of a verification check."""
    passed: bool
    confidence: VerificationLevel
    source: VerificationSource
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "confidence": self.confidence.value,
            "source": self.source.value,
            "reason": self.reason,
            "details": self.details,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
        }

    @property
    def needs_escalation(self) -> bool:
        """Check if result needs human escalation."""
        return (
            self.confidence in (VerificationLevel.LOW, VerificationLevel.UNCERTAIN)
            or (not self.passed and self.source == VerificationSource.LLM)
        )


@dataclass
class RuleCheck:
    """A rule-based verification check."""
    name: str
    description: str
    check_fn: Callable[[str, dict[str, Any]], bool]
    severity: str = "warning"  # "warning", "error", "critical"
    enabled: bool = True


class RuleEngine:
    """Engine for running rule-based verification checks.

    Rule checks are fast, deterministic, and don't require LLM inference.
    They catch obvious issues like:
    - Empty or too-short outputs
    - Missing required sections
    - Invalid format/structure
    - Known error patterns
    """

    def __init__(self):
        """Initialize with default rules."""
        self._rules: list[RuleCheck] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default verification rules."""
        # Output quality rules
        self.add_rule(RuleCheck(
            name="non_empty",
            description="Output must not be empty",
            check_fn=lambda output, ctx: len(output.strip()) > 0,
            severity="error",
        ))

        self.add_rule(RuleCheck(
            name="min_length",
            description="Output must be at least 10 characters",
            check_fn=lambda output, ctx: len(output.strip()) >= 10,
            severity="warning",
        ))

        self.add_rule(RuleCheck(
            name="no_error_indicators",
            description="Output should not contain error indicators",
            check_fn=lambda output, ctx: not self._has_error_indicators(output),
            severity="warning",
        ))

        self.add_rule(RuleCheck(
            name="no_placeholder_text",
            description="Output should not contain placeholder text",
            check_fn=lambda output, ctx: not self._has_placeholders(output),
            severity="warning",
        ))

        self.add_rule(RuleCheck(
            name="no_hallucination_markers",
            description="Output should not contain hallucination markers",
            check_fn=lambda output, ctx: not self._has_hallucination_markers(output),
            severity="error",
        ))

        # Code-specific rules (only apply when context indicates code)
        self.add_rule(RuleCheck(
            name="balanced_brackets",
            description="Code should have balanced brackets",
            check_fn=lambda output, ctx: (
                not ctx.get("is_code", False) or self._has_balanced_brackets(output)
            ),
            severity="error",
        ))

        self.add_rule(RuleCheck(
            name="no_syntax_markers",
            description="Code should not have obvious syntax errors",
            check_fn=lambda output, ctx: (
                not ctx.get("is_code", False) or not self._has_syntax_error_markers(output)
            ),
            severity="warning",
        ))

    def _has_error_indicators(self, output: str) -> bool:
        """Check for common error indicators."""
        error_patterns = [
            r"(?i)^error:",
            r"(?i)^fatal:",
            r"(?i)^exception:",
            r"(?i)traceback \(most recent call last\)",
            r"(?i)^\s*raise\s+\w+Error",
            r"(?i)I cannot",
            r"(?i)I'm unable to",
            r"(?i)I apologize, but",
        ]
        for pattern in error_patterns:
            if re.search(pattern, output):
                return True
        return False

    def _has_placeholders(self, output: str) -> bool:
        """Check for placeholder text."""
        placeholder_patterns = [
            r"\[TODO\]",
            r"\[PLACEHOLDER\]",
            r"\[INSERT",
            r"\[YOUR",
            r"<TODO>",
            r"<PLACEHOLDER>",
            r"\.{3,}",  # Multiple dots as placeholder
            r"___+",    # Multiple underscores
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return True
        return False

    def _has_hallucination_markers(self, output: str) -> bool:
        """Check for potential hallucination markers."""
        # These indicate the model might be making things up
        hallucination_patterns = [
            r"(?i)I don't actually have access",
            r"(?i)I cannot actually",
            r"(?i)I made up",
            r"(?i)I fabricated",
            r"(?i)As an AI, I",
            r"(?i)I'm just an AI",
            r"(?i)I don't have the ability to",
            r"(?i)this is hypothetical",
            r"(?i)assuming the file exists",
        ]
        for pattern in hallucination_patterns:
            if re.search(pattern, output):
                return True
        return False

    def _has_balanced_brackets(self, output: str) -> bool:
        """Check if brackets are balanced in code."""
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}

        # Skip strings (simplified - doesn't handle escaped quotes)
        in_string = False
        string_char = None

        for char in output:
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif not in_string:
                if char in '([{':
                    stack.append(char)
                elif char in ')]}':
                    if not stack or stack[-1] != pairs[char]:
                        return False
                    stack.pop()

        return len(stack) == 0

    def _has_syntax_error_markers(self, output: str) -> bool:
        """Check for syntax error markers in code."""
        syntax_patterns = [
            r"SyntaxError:",
            r"IndentationError:",
            r"(?i)unexpected token",
            r"(?i)missing semicolon",
            r"(?i)unterminated string",
        ]
        for pattern in syntax_patterns:
            if re.search(pattern, output):
                return True
        return False

    def add_rule(self, rule: RuleCheck) -> None:
        """Add a verification rule."""
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def verify(
        self,
        output: str,
        context: Optional[dict[str, Any]] = None,
    ) -> VerificationResult:
        """Run all rules against the output.

        Args:
            output: The output to verify.
            context: Additional context (e.g., is_code, expected_format).

        Returns:
            VerificationResult with aggregated check results.
        """
        context = context or {}
        passed_checks = []
        failed_checks = []
        critical_failure = False
        error_failure = False

        for rule in self._rules:
            if not rule.enabled:
                continue

            try:
                result = rule.check_fn(output, context)
                if result:
                    passed_checks.append(rule.name)
                else:
                    failed_checks.append(rule.name)
                    if rule.severity == "critical":
                        critical_failure = True
                    elif rule.severity == "error":
                        error_failure = True
            except Exception:
                # Rule execution failed, treat as warning
                failed_checks.append(f"{rule.name} (execution error)")

        # Determine overall result
        if critical_failure or error_failure:
            passed = False
            confidence = VerificationLevel.HIGH
            reason = f"Failed {len(failed_checks)} check(s): {', '.join(failed_checks)}"
        elif failed_checks:
            # Only warnings
            passed = True
            confidence = VerificationLevel.MEDIUM
            reason = f"Passed with {len(failed_checks)} warning(s)"
        else:
            passed = True
            confidence = VerificationLevel.HIGH
            reason = f"All {len(passed_checks)} check(s) passed"

        return VerificationResult(
            passed=passed,
            confidence=confidence,
            source=VerificationSource.RULE,
            reason=reason,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
        )


class LLMEvaluator:
    """LLM-based verification for nuanced evaluation.

    Used when rule-based checks pass but we need deeper validation,
    such as:
    - Does the output correctly address the user's intent?
    - Is the code logic correct (not just syntax)?
    - Is the explanation accurate?
    """

    def __init__(
        self,
        evaluate_fn: Optional[Callable[[str, str, dict], Awaitable[dict]]] = None,
    ):
        """Initialize LLM evaluator.

        Args:
            evaluate_fn: Async function that takes (output, prompt, context)
                        and returns {"passed": bool, "reason": str, "confidence": float}
        """
        self._evaluate_fn = evaluate_fn

    async def verify(
        self,
        output: str,
        original_prompt: str,
        context: Optional[dict[str, Any]] = None,
    ) -> VerificationResult:
        """Use LLM to evaluate output quality.

        Args:
            output: The output to verify.
            original_prompt: The original user prompt.
            context: Additional context.

        Returns:
            VerificationResult from LLM evaluation.
        """
        if not self._evaluate_fn:
            # No LLM evaluator configured, return uncertain
            return VerificationResult(
                passed=True,
                confidence=VerificationLevel.UNCERTAIN,
                source=VerificationSource.LLM,
                reason="LLM evaluator not configured",
            )

        try:
            result = await self._evaluate_fn(output, original_prompt, context or {})

            confidence_score = result.get("confidence", 0.5)
            if confidence_score >= 0.8:
                confidence = VerificationLevel.HIGH
            elif confidence_score >= 0.5:
                confidence = VerificationLevel.MEDIUM
            else:
                confidence = VerificationLevel.LOW

            return VerificationResult(
                passed=result.get("passed", False),
                confidence=confidence,
                source=VerificationSource.LLM,
                reason=result.get("reason", "LLM evaluation complete"),
                details={
                    "confidence_score": confidence_score,
                    "llm_feedback": result.get("feedback", ""),
                },
            )

        except Exception as e:
            return VerificationResult(
                passed=True,  # Don't block on LLM failure
                confidence=VerificationLevel.UNCERTAIN,
                source=VerificationSource.LLM,
                reason=f"LLM evaluation failed: {str(e)}",
            )


class HumanEscalator:
    """Handles escalation to human review.

    Used when automated verification has low confidence or
    detects potential issues that need human judgment.
    """

    def __init__(
        self,
        escalation_fn: Optional[Callable[[str, str, VerificationResult], Awaitable[bool]]] = None,
    ):
        """Initialize human escalator.

        Args:
            escalation_fn: Async function that presents output to human and
                          returns True if approved, False if rejected.
        """
        self._escalation_fn = escalation_fn

    async def escalate(
        self,
        output: str,
        prompt: str,
        prior_result: VerificationResult,
    ) -> VerificationResult:
        """Escalate to human for verification.

        Args:
            output: The output to verify.
            prompt: The original prompt.
            prior_result: Result from automated verification.

        Returns:
            VerificationResult with human decision.
        """
        if not self._escalation_fn:
            # No escalation handler, return prior result
            return VerificationResult(
                passed=prior_result.passed,
                confidence=VerificationLevel.LOW,
                source=VerificationSource.HUMAN,
                reason="Human escalation not configured, using prior result",
                details={"prior_result": prior_result.to_dict()},
            )

        try:
            approved = await self._escalation_fn(output, prompt, prior_result)

            return VerificationResult(
                passed=approved,
                confidence=VerificationLevel.HIGH,  # Human decision is authoritative
                source=VerificationSource.HUMAN,
                reason="Approved by human" if approved else "Rejected by human",
                details={"prior_result": prior_result.to_dict()},
            )

        except Exception as e:
            return VerificationResult(
                passed=prior_result.passed,
                confidence=VerificationLevel.LOW,
                source=VerificationSource.HUMAN,
                reason=f"Human escalation failed: {str(e)}",
            )


class HybridJudge:
    """Triangulated verification combining rules, LLM, and human review.

    Verification flow:
    1. Run fast rule-based checks
    2. If rules pass with high confidence, accept
    3. If rules have warnings, optionally run LLM evaluation
    4. If LLM is uncertain or fails, escalate to human
    """

    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        llm_evaluator: Optional[LLMEvaluator] = None,
        human_escalator: Optional[HumanEscalator] = None,
        auto_escalate_on_failure: bool = True,
        require_llm_for_medium_confidence: bool = False,
    ):
        """Initialize hybrid judge.

        Args:
            rule_engine: Rule-based verification engine.
            llm_evaluator: LLM-based evaluator.
            human_escalator: Human escalation handler.
            auto_escalate_on_failure: Auto-escalate to human on failure.
            require_llm_for_medium_confidence: Require LLM check for medium confidence.
        """
        self.rule_engine = rule_engine or RuleEngine()
        self.llm_evaluator = llm_evaluator or LLMEvaluator()
        self.human_escalator = human_escalator or HumanEscalator()
        self.auto_escalate_on_failure = auto_escalate_on_failure
        self.require_llm_for_medium_confidence = require_llm_for_medium_confidence

    async def verify(
        self,
        output: str,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
        skip_llm: bool = False,
        skip_human: bool = False,
    ) -> VerificationResult:
        """Run triangulated verification.

        Args:
            output: The output to verify.
            prompt: The original prompt.
            context: Additional context.
            skip_llm: Skip LLM evaluation.
            skip_human: Skip human escalation.

        Returns:
            Final VerificationResult.
        """
        context = context or {}

        # Step 1: Rule-based verification (always run)
        rule_result = self.rule_engine.verify(output, context)

        # Early exit if rules pass with high confidence
        if rule_result.passed and rule_result.confidence == VerificationLevel.HIGH:
            return rule_result

        # Early exit if rules fail critically
        if not rule_result.passed:
            if self.auto_escalate_on_failure and not skip_human:
                return await self.human_escalator.escalate(output, prompt, rule_result)
            return rule_result

        # Step 2: LLM evaluation (if configured and needed)
        if not skip_llm and (
            self.require_llm_for_medium_confidence
            or rule_result.confidence != VerificationLevel.HIGH
        ):
            llm_result = await self.llm_evaluator.verify(output, prompt, context)

            # If LLM passes with high confidence, accept
            if llm_result.passed and llm_result.confidence == VerificationLevel.HIGH:
                return llm_result

            # If LLM fails or is uncertain, consider escalation
            if llm_result.needs_escalation and not skip_human:
                return await self.human_escalator.escalate(output, prompt, llm_result)

            return llm_result

        # Step 3: Human escalation (if needed)
        if rule_result.needs_escalation and not skip_human:
            return await self.human_escalator.escalate(output, prompt, rule_result)

        return rule_result

    def verify_sync(
        self,
        output: str,
        context: Optional[dict[str, Any]] = None,
    ) -> VerificationResult:
        """Synchronous rule-only verification.

        Use this for fast, synchronous verification when LLM and
        human escalation are not needed.

        Args:
            output: The output to verify.
            context: Additional context.

        Returns:
            VerificationResult from rule checks only.
        """
        return self.rule_engine.verify(output, context)

    def add_rule(self, rule: RuleCheck) -> None:
        """Add a custom verification rule."""
        self.rule_engine.add_rule(rule)

    def to_outcome_status(self, result: VerificationResult) -> OutcomeStatus:
        """Convert verification result to outcome status.

        Args:
            result: The verification result.

        Returns:
            OutcomeStatus for decision recording.
        """
        if result.passed:
            if result.confidence == VerificationLevel.HIGH:
                return OutcomeStatus.SUCCESS
            else:
                return OutcomeStatus.PARTIAL
        else:
            return OutcomeStatus.FAILURE
