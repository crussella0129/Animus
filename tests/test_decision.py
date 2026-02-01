"""Tests for decision recording module."""

import pytest
from datetime import datetime

from src.core.decision import (
    Decision,
    DecisionType,
    Option,
    Outcome,
    OutcomeStatus,
    DecisionRecord,
    DecisionRecorder,
)


class TestOption:
    """Tests for Option dataclass."""

    def test_create_option(self):
        """Test creating an option with factory method."""
        opt = Option.create(
            description="Use tool X",
            pros=["Fast", "Reliable"],
            cons=["Complex"],
            confidence=0.8,
        )
        assert opt.description == "Use tool X"
        assert opt.pros == ["Fast", "Reliable"]
        assert opt.cons == ["Complex"]
        assert opt.confidence == 0.8
        assert len(opt.id) == 8  # Short UUID

    def test_option_with_metadata(self):
        """Test option with metadata."""
        opt = Option.create(
            description="Test",
            tool_name="read_file",
            custom="value",
        )
        assert opt.metadata["tool_name"] == "read_file"
        assert opt.metadata["custom"] == "value"


class TestDecision:
    """Tests for Decision dataclass."""

    def test_create_decision(self):
        """Test creating a decision with factory method."""
        options = [
            Option.create("Option A", pros=["Good"]),
            Option.create("Option B", cons=["Bad"]),
        ]
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Select best tool",
            context="User wants to read a file",
            options=options,
            chosen_option_id=options[0].id,
            reasoning="Option A is faster",
            turn_number=1,
        )
        assert decision.decision_type == DecisionType.TOOL_SELECTION
        assert decision.intent == "Select best tool"
        assert len(decision.options) == 2
        assert decision.chosen_option_id == options[0].id
        assert decision.turn_number == 1

    def test_decision_chosen_option_property(self):
        """Test getting the chosen option."""
        options = [
            Option.create("Option A"),
            Option.create("Option B"),
        ]
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test",
            context="Context",
            options=options,
            chosen_option_id=options[1].id,
            reasoning="Chose B",
        )
        assert decision.chosen_option is not None
        assert decision.chosen_option.description == "Option B"

    def test_decision_to_dict_and_back(self):
        """Test serialization roundtrip."""
        options = [Option.create("Test option")]
        decision = Decision.create(
            decision_type=DecisionType.STRATEGY,
            intent="Test intent",
            context="Test context",
            options=options,
            chosen_option_id=options[0].id,
            reasoning="Test reasoning",
        )
        data = decision.to_dict()
        restored = Decision.from_dict(data)
        assert restored.id == decision.id
        assert restored.decision_type == decision.decision_type
        assert restored.intent == decision.intent
        assert len(restored.options) == 1


class TestOutcome:
    """Tests for Outcome dataclass."""

    def test_create_outcome_success(self):
        """Test creating a successful outcome."""
        outcome = Outcome.create(
            decision_id="test-decision-id",
            status=OutcomeStatus.SUCCESS,
            result="File was read successfully",
            summary="Read file.txt",
        )
        assert outcome.status == OutcomeStatus.SUCCESS
        assert outcome.error is None

    def test_create_outcome_failure(self):
        """Test creating a failed outcome."""
        outcome = Outcome.create(
            decision_id="test-decision-id",
            status=OutcomeStatus.FAILURE,
            result="",
            summary="Failed to read file",
            error="File not found",
        )
        assert outcome.status == OutcomeStatus.FAILURE
        assert outcome.error == "File not found"

    def test_outcome_with_metrics(self):
        """Test outcome with metrics."""
        outcome = Outcome.create(
            decision_id="test-id",
            status=OutcomeStatus.SUCCESS,
            result="Done",
            summary="Completed",
            metrics={"latency_ms": 50.0, "tokens": 100},
        )
        assert outcome.metrics["latency_ms"] == 50.0
        assert outcome.metrics["tokens"] == 100

    def test_outcome_to_dict_and_back(self):
        """Test serialization roundtrip."""
        outcome = Outcome.create(
            decision_id="test-id",
            status=OutcomeStatus.PARTIAL,
            result="Partial result",
            summary="Partially done",
        )
        data = outcome.to_dict()
        restored = Outcome.from_dict(data)
        assert restored.id == outcome.id
        assert restored.status == OutcomeStatus.PARTIAL


class TestDecisionRecord:
    """Tests for DecisionRecord dataclass."""

    def test_record_without_outcome(self):
        """Test record with decision but no outcome."""
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test",
            context="Context",
            options=[],
            chosen_option_id=None,
            reasoning="Reasoning",
        )
        record = DecisionRecord(decision=decision)
        assert record.outcome is None
        assert not record.was_successful

    def test_record_with_successful_outcome(self):
        """Test record with successful outcome."""
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test",
            context="Context",
            options=[],
            chosen_option_id=None,
            reasoning="Reasoning",
        )
        outcome = Outcome.create(
            decision_id=decision.id,
            status=OutcomeStatus.SUCCESS,
            result="Done",
            summary="Success",
        )
        record = DecisionRecord(decision=decision, outcome=outcome)
        assert record.was_successful


class TestDecisionRecorder:
    """Tests for DecisionRecorder class."""

    def test_record_and_retrieve_decision(self):
        """Test recording and retrieving a decision."""
        recorder = DecisionRecorder()
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test",
            context="Context",
            options=[],
            chosen_option_id=None,
            reasoning="Reasoning",
        )
        recorder.record_decision(decision)
        assert len(recorder.decisions) == 1
        assert recorder.get_decision(decision.id) == decision

    def test_record_outcome(self):
        """Test recording an outcome."""
        recorder = DecisionRecorder()
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test",
            context="Context",
            options=[],
            chosen_option_id=None,
            reasoning="Reasoning",
        )
        recorder.record_decision(decision)

        outcome = Outcome.create(
            decision_id=decision.id,
            status=OutcomeStatus.SUCCESS,
            result="Done",
            summary="Success",
        )
        recorder.record_outcome(outcome)
        assert recorder.get_outcome(decision.id) == outcome

    def test_get_records(self):
        """Test getting all decision records."""
        recorder = DecisionRecorder()
        decision = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test",
            context="Context",
            options=[],
            chosen_option_id=None,
            reasoning="Reasoning",
        )
        recorder.record_decision(decision)
        outcome = Outcome.create(
            decision_id=decision.id,
            status=OutcomeStatus.SUCCESS,
            result="Done",
            summary="Success",
        )
        recorder.record_outcome(outcome)

        records = recorder.get_records()
        assert len(records) == 1
        assert records[0].decision.id == decision.id
        assert records[0].outcome.id == outcome.id

    def test_get_decisions_by_type(self):
        """Test filtering decisions by type."""
        recorder = DecisionRecorder()
        recorder.record_decision(Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Tool 1", context="", options=[],
            chosen_option_id=None, reasoning="",
        ))
        recorder.record_decision(Decision.create(
            decision_type=DecisionType.STRATEGY,
            intent="Strategy 1", context="", options=[],
            chosen_option_id=None, reasoning="",
        ))
        recorder.record_decision(Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Tool 2", context="", options=[],
            chosen_option_id=None, reasoning="",
        ))

        tool_decisions = recorder.get_decisions_by_type(DecisionType.TOOL_SELECTION)
        assert len(tool_decisions) == 2

        strategy_decisions = recorder.get_decisions_by_type(DecisionType.STRATEGY)
        assert len(strategy_decisions) == 1

    def test_success_rate(self):
        """Test calculating success rate."""
        recorder = DecisionRecorder()

        # Add 3 decisions with 2 successes
        for i in range(3):
            d = Decision.create(
                decision_type=DecisionType.TOOL_SELECTION,
                intent=f"Test {i}", context="", options=[],
                chosen_option_id=None, reasoning="",
            )
            recorder.record_decision(d)
            recorder.record_outcome(Outcome.create(
                decision_id=d.id,
                status=OutcomeStatus.SUCCESS if i < 2 else OutcomeStatus.FAILURE,
                result="", summary="",
            ))

        rate = recorder.get_success_rate()
        assert rate == pytest.approx(2/3)

    def test_clear(self):
        """Test clearing the recorder."""
        recorder = DecisionRecorder()
        recorder.record_decision(Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test", context="", options=[],
            chosen_option_id=None, reasoning="",
        ))
        recorder.clear()
        assert len(recorder.decisions) == 0
        assert len(recorder.outcomes) == 0

    def test_to_dict_and_back(self):
        """Test serialization roundtrip."""
        recorder = DecisionRecorder()
        d = Decision.create(
            decision_type=DecisionType.TOOL_SELECTION,
            intent="Test", context="", options=[],
            chosen_option_id=None, reasoning="",
        )
        recorder.record_decision(d)
        recorder.record_outcome(Outcome.create(
            decision_id=d.id,
            status=OutcomeStatus.SUCCESS,
            result="", summary="",
        ))

        data = recorder.to_dict()
        restored = DecisionRecorder.from_dict(data)
        assert len(restored.decisions) == 1
        assert len(restored.outcomes) == 1
