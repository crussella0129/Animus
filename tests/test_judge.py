"""Tests for Triangulated Verification (HybridJudge)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.judge import (
    HybridJudge,
    RuleEngine,
    RuleCheck,
    LLMEvaluator,
    HumanEscalator,
    VerificationResult,
    VerificationLevel,
    VerificationSource,
)
from src.core.decision import OutcomeStatus


class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_create_result(self):
        """Test creating a verification result."""
        result = VerificationResult(
            passed=True,
            confidence=VerificationLevel.HIGH,
            source=VerificationSource.RULE,
            reason="All checks passed",
        )
        assert result.passed is True
        assert result.confidence == VerificationLevel.HIGH
        assert result.source == VerificationSource.RULE

    def test_needs_escalation_low_confidence(self):
        """Test escalation needed for low confidence."""
        result = VerificationResult(
            passed=True,
            confidence=VerificationLevel.LOW,
            source=VerificationSource.RULE,
            reason="Low confidence",
        )
        assert result.needs_escalation is True

    def test_needs_escalation_uncertain(self):
        """Test escalation needed for uncertain."""
        result = VerificationResult(
            passed=True,
            confidence=VerificationLevel.UNCERTAIN,
            source=VerificationSource.LLM,
            reason="Uncertain",
        )
        assert result.needs_escalation is True

    def test_needs_escalation_llm_failure(self):
        """Test escalation needed for LLM failure."""
        result = VerificationResult(
            passed=False,
            confidence=VerificationLevel.MEDIUM,
            source=VerificationSource.LLM,
            reason="LLM rejected",
        )
        assert result.needs_escalation is True

    def test_no_escalation_high_confidence(self):
        """Test no escalation for high confidence pass."""
        result = VerificationResult(
            passed=True,
            confidence=VerificationLevel.HIGH,
            source=VerificationSource.RULE,
            reason="All good",
        )
        assert result.needs_escalation is False

    def test_to_dict(self):
        """Test serialization."""
        result = VerificationResult(
            passed=True,
            confidence=VerificationLevel.HIGH,
            source=VerificationSource.RULE,
            reason="Test",
            checks_passed=["check1"],
            checks_failed=["check2"],
        )
        data = result.to_dict()
        assert data["passed"] is True
        assert data["confidence"] == "high"
        assert data["source"] == "rule"
        assert "check1" in data["checks_passed"]


class TestRuleEngine:
    """Test RuleEngine class."""

    @pytest.fixture
    def engine(self):
        """Create rule engine with defaults."""
        return RuleEngine()

    def test_verify_empty_output(self, engine):
        """Test verification of empty output."""
        result = engine.verify("")
        assert result.passed is False
        assert "non_empty" in result.checks_failed

    def test_verify_short_output(self, engine):
        """Test verification of too-short output."""
        result = engine.verify("Hello")
        assert "min_length" in result.checks_failed

    def test_verify_valid_output(self, engine):
        """Test verification of valid output."""
        result = engine.verify("This is a perfectly valid output with enough content.")
        assert result.passed is True
        assert result.confidence == VerificationLevel.HIGH

    def test_verify_error_indicators(self, engine):
        """Test detection of error indicators."""
        result = engine.verify("Error: Something went wrong with the process.")
        assert "no_error_indicators" in result.checks_failed

    def test_verify_placeholder_text(self, engine):
        """Test detection of placeholder text."""
        result = engine.verify("The function should [TODO] implement the logic here.")
        assert "no_placeholder_text" in result.checks_failed

    def test_verify_hallucination_markers(self, engine):
        """Test detection of hallucination markers."""
        result = engine.verify("As an AI, I cannot actually access your files.")
        assert "no_hallucination_markers" in result.checks_failed

    def test_verify_code_balanced_brackets(self, engine):
        """Test balanced brackets in code."""
        # Unbalanced brackets
        result = engine.verify("function test() { return [1, 2, 3; }", {"is_code": True})
        assert "balanced_brackets" in result.checks_failed

        # Balanced brackets
        result = engine.verify("function test() { return [1, 2, 3]; }", {"is_code": True})
        assert "balanced_brackets" not in result.checks_failed

    def test_verify_syntax_error_markers(self, engine):
        """Test detection of syntax errors in code."""
        result = engine.verify("SyntaxError: unexpected token", {"is_code": True})
        assert "no_syntax_markers" in result.checks_failed

    def test_add_custom_rule(self, engine):
        """Test adding custom rule."""
        custom_rule = RuleCheck(
            name="no_profanity",
            description="Output should not contain profanity",
            check_fn=lambda output, ctx: "badword" not in output.lower(),
            severity="warning",
        )
        engine.add_rule(custom_rule)

        result = engine.verify("This contains badword in it.")
        assert "no_profanity" in result.checks_failed

    def test_remove_rule(self, engine):
        """Test removing a rule."""
        assert engine.remove_rule("min_length") is True
        assert engine.remove_rule("nonexistent") is False

    def test_disabled_rule(self, engine):
        """Test that disabled rules are skipped."""
        for rule in engine._rules:
            if rule.name == "min_length":
                rule.enabled = False

        result = engine.verify("Short")
        assert "min_length" not in result.checks_failed
        assert "min_length" not in result.checks_passed


class TestLLMEvaluator:
    """Test LLMEvaluator class."""

    @pytest.fixture
    def evaluator_with_fn(self):
        """Create evaluator with mock function."""
        async def mock_evaluate(output, prompt, context):
            return {
                "passed": "good" in output.lower(),
                "reason": "LLM evaluation",
                "confidence": 0.9 if "good" in output.lower() else 0.3,
            }
        return LLMEvaluator(evaluate_fn=mock_evaluate)

    @pytest.fixture
    def evaluator_without_fn(self):
        """Create evaluator without function."""
        return LLMEvaluator()

    @pytest.mark.asyncio
    async def test_verify_with_function(self, evaluator_with_fn):
        """Test LLM verification with function."""
        result = await evaluator_with_fn.verify(
            "This is a good output",
            "Generate something",
        )
        assert result.passed is True
        assert result.confidence == VerificationLevel.HIGH
        assert result.source == VerificationSource.LLM

    @pytest.mark.asyncio
    async def test_verify_failure(self, evaluator_with_fn):
        """Test LLM verification failure."""
        result = await evaluator_with_fn.verify(
            "This is a bad output",
            "Generate something",
        )
        assert result.passed is False
        assert result.confidence == VerificationLevel.LOW

    @pytest.mark.asyncio
    async def test_verify_without_function(self, evaluator_without_fn):
        """Test LLM verification without function."""
        result = await evaluator_without_fn.verify(
            "Some output",
            "Some prompt",
        )
        assert result.passed is True
        assert result.confidence == VerificationLevel.UNCERTAIN

    @pytest.mark.asyncio
    async def test_verify_handles_exception(self):
        """Test that evaluator handles exceptions."""
        async def failing_fn(output, prompt, context):
            raise RuntimeError("LLM failure")

        evaluator = LLMEvaluator(evaluate_fn=failing_fn)
        result = await evaluator.verify("output", "prompt")

        assert result.passed is True  # Don't block on failure
        assert result.confidence == VerificationLevel.UNCERTAIN


class TestHumanEscalator:
    """Test HumanEscalator class."""

    @pytest.fixture
    def escalator_approve(self):
        """Create escalator that approves."""
        async def approve_fn(output, prompt, prior):
            return True
        return HumanEscalator(escalation_fn=approve_fn)

    @pytest.fixture
    def escalator_reject(self):
        """Create escalator that rejects."""
        async def reject_fn(output, prompt, prior):
            return False
        return HumanEscalator(escalation_fn=reject_fn)

    @pytest.fixture
    def prior_result(self):
        """Create a prior result for escalation."""
        return VerificationResult(
            passed=True,
            confidence=VerificationLevel.LOW,
            source=VerificationSource.RULE,
            reason="Low confidence",
        )

    @pytest.mark.asyncio
    async def test_escalate_approved(self, escalator_approve, prior_result):
        """Test human approval."""
        result = await escalator_approve.escalate(
            "Output",
            "Prompt",
            prior_result,
        )
        assert result.passed is True
        assert result.confidence == VerificationLevel.HIGH
        assert result.source == VerificationSource.HUMAN
        assert "Approved" in result.reason

    @pytest.mark.asyncio
    async def test_escalate_rejected(self, escalator_reject, prior_result):
        """Test human rejection."""
        result = await escalator_reject.escalate(
            "Output",
            "Prompt",
            prior_result,
        )
        assert result.passed is False
        assert result.source == VerificationSource.HUMAN
        assert "Rejected" in result.reason

    @pytest.mark.asyncio
    async def test_escalate_no_function(self, prior_result):
        """Test escalation without function."""
        escalator = HumanEscalator()
        result = await escalator.escalate("Output", "Prompt", prior_result)

        assert result.passed == prior_result.passed
        assert result.confidence == VerificationLevel.LOW


class TestHybridJudge:
    """Test HybridJudge class."""

    @pytest.fixture
    def judge(self):
        """Create basic hybrid judge."""
        return HybridJudge()

    @pytest.fixture
    def judge_with_llm(self):
        """Create judge with LLM evaluator."""
        async def mock_llm(output, prompt, context):
            return {
                "passed": len(output) > 20,
                "reason": "Length check",
                "confidence": 0.9,
            }
        return HybridJudge(
            llm_evaluator=LLMEvaluator(evaluate_fn=mock_llm),
            require_llm_for_medium_confidence=True,
        )

    def test_sync_verify_passes(self, judge):
        """Test synchronous verification pass."""
        result = judge.verify_sync(
            "This is a valid output with sufficient content for verification."
        )
        assert result.passed is True
        assert result.source == VerificationSource.RULE

    def test_sync_verify_fails(self, judge):
        """Test synchronous verification failure."""
        result = judge.verify_sync("")
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_async_verify_rules_only(self, judge):
        """Test async verification with rules only."""
        result = await judge.verify(
            "This is a valid output with sufficient content.",
            "Generate something",
            skip_llm=True,
            skip_human=True,
        )
        assert result.passed is True
        assert result.source == VerificationSource.RULE

    @pytest.mark.asyncio
    async def test_async_verify_with_llm(self, judge_with_llm):
        """Test async verification with LLM."""
        result = await judge_with_llm.verify(
            "This is a moderately long output that should pass.",
            "Generate something",
            skip_human=True,
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_async_verify_early_exit_high_confidence(self, judge):
        """Test early exit on high confidence."""
        result = await judge.verify(
            "This is a perfectly valid and complete output for verification.",
            "Generate something",
        )
        assert result.passed is True
        assert result.confidence == VerificationLevel.HIGH
        assert result.source == VerificationSource.RULE

    def test_add_custom_rule(self, judge):
        """Test adding custom rule to judge."""
        judge.add_rule(RuleCheck(
            name="custom",
            description="Custom check",
            check_fn=lambda o, c: "test" in o,
        ))

        result = judge.verify_sync("This contains test word")
        assert "custom" in result.checks_passed

    def test_to_outcome_status_success(self, judge):
        """Test conversion to outcome status - success."""
        result = VerificationResult(
            passed=True,
            confidence=VerificationLevel.HIGH,
            source=VerificationSource.RULE,
            reason="Success",
        )
        assert judge.to_outcome_status(result) == OutcomeStatus.SUCCESS

    def test_to_outcome_status_partial(self, judge):
        """Test conversion to outcome status - partial."""
        result = VerificationResult(
            passed=True,
            confidence=VerificationLevel.MEDIUM,
            source=VerificationSource.RULE,
            reason="Partial",
        )
        assert judge.to_outcome_status(result) == OutcomeStatus.PARTIAL

    def test_to_outcome_status_failure(self, judge):
        """Test conversion to outcome status - failure."""
        result = VerificationResult(
            passed=False,
            confidence=VerificationLevel.HIGH,
            source=VerificationSource.RULE,
            reason="Failed",
        )
        assert judge.to_outcome_status(result) == OutcomeStatus.FAILURE


class TestVerificationLevel:
    """Test VerificationLevel enum."""

    def test_levels(self):
        """Test all verification levels."""
        assert VerificationLevel.HIGH.value == "high"
        assert VerificationLevel.MEDIUM.value == "medium"
        assert VerificationLevel.LOW.value == "low"
        assert VerificationLevel.UNCERTAIN.value == "uncertain"


class TestVerificationSource:
    """Test VerificationSource enum."""

    def test_sources(self):
        """Test all verification sources."""
        assert VerificationSource.RULE.value == "rule"
        assert VerificationSource.LLM.value == "llm"
        assert VerificationSource.HUMAN.value == "human"


class TestRuleCheck:
    """Test RuleCheck dataclass."""

    def test_create_rule(self):
        """Test creating a rule check."""
        rule = RuleCheck(
            name="test_rule",
            description="A test rule",
            check_fn=lambda o, c: True,
            severity="error",
        )
        assert rule.name == "test_rule"
        assert rule.severity == "error"
        assert rule.enabled is True

    def test_rule_execution(self):
        """Test rule execution."""
        rule = RuleCheck(
            name="length_check",
            description="Check length",
            check_fn=lambda o, c: len(o) > 5,
        )
        assert rule.check_fn("hello world", {}) is True
        assert rule.check_fn("hi", {}) is False
