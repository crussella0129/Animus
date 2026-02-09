"""Tests for sub-agent graph architecture (Phase 11)."""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.subagents.goal import (
    SubAgentGoal,
    SuccessCriterion,
    Constraint,
    ConstraintType,
)
from src.subagents.node import SubAgentNode, NodeType
from src.subagents.edge import SubAgentEdge, EdgeCondition
from src.subagents.graph import SubAgentGraph, GraphValidationError
from src.subagents.executor import SubAgentExecutor, ExecutionResult, StepResult
from src.subagents.session import SessionState, SessionStore
from src.subagents.cleaner import OutputCleaner


# ============================================================
# Goal Tests
# ============================================================


class TestSuccessCriterion:
    def test_evaluate_full_match(self):
        c = SuccessCriterion(id="c1", description="tests", metric="tests_passing", target=10, weight=0.5)
        assert c.evaluate(10) == pytest.approx(0.5)

    def test_evaluate_partial_match(self):
        c = SuccessCriterion(id="c1", description="tests", metric="tests_passing", target=10, weight=1.0)
        assert c.evaluate(5) == pytest.approx(0.5)

    def test_evaluate_over_target_capped(self):
        c = SuccessCriterion(id="c1", description="tests", metric="tests_passing", target=10, weight=0.8)
        assert c.evaluate(20) == pytest.approx(0.8)  # Capped at weight

    def test_evaluate_zero_target(self):
        c = SuccessCriterion(id="c1", description="zero", metric="errors", target=0, weight=0.3)
        assert c.evaluate(0) == pytest.approx(0.3)
        assert c.evaluate(1) == pytest.approx(0.0)


class TestConstraint:
    def test_hard_constraint(self):
        c = Constraint(id="c1", description="no write", constraint_type=ConstraintType.HARD)
        assert c.is_hard()

    def test_soft_constraint(self):
        c = Constraint(id="c1", description="prefer fast", constraint_type=ConstraintType.SOFT)
        assert not c.is_hard()


class TestSubAgentGoal:
    def test_valid_goal(self):
        goal = SubAgentGoal(
            id="g1",
            name="Fix bug",
            description="Find and fix the null pointer",
            criteria=[
                SuccessCriterion(id="c1", description="tests", metric="tests_passing", target=1, weight=0.6),
                SuccessCriterion(id="c2", description="errors", metric="errors", target=0, weight=0.4),
            ],
        )
        assert goal.validate() == []

    def test_criteria_weights_must_sum_to_one(self):
        goal = SubAgentGoal(
            id="g1",
            name="Test",
            description="desc",
            criteria=[
                SuccessCriterion(id="c1", description="a", metric="m1", target=1, weight=0.3),
                SuccessCriterion(id="c2", description="b", metric="m2", target=1, weight=0.3),
            ],
        )
        errors = goal.validate()
        assert any("sum to 1.0" in e for e in errors)

    def test_duplicate_criteria_ids(self):
        goal = SubAgentGoal(
            id="g1",
            name="Test",
            description="desc",
            criteria=[
                SuccessCriterion(id="dup", description="a", metric="m1", target=1, weight=0.5),
                SuccessCriterion(id="dup", description="b", metric="m2", target=1, weight=0.5),
            ],
        )
        errors = goal.validate()
        assert any("unique" in e for e in errors)

    def test_missing_id(self):
        goal = SubAgentGoal(id="", name="Test", description="desc")
        errors = goal.validate()
        assert any("id" in e for e in errors)

    def test_evaluate_success(self):
        goal = SubAgentGoal(
            id="g1",
            name="Test",
            description="desc",
            criteria=[
                SuccessCriterion(id="c1", description="a", metric="tests", target=10, weight=0.7),
                SuccessCriterion(id="c2", description="b", metric="coverage", target=80, weight=0.3),
            ],
        )
        score = goal.evaluate({"tests": 10, "coverage": 80})
        assert score == pytest.approx(1.0)

    def test_evaluate_no_criteria(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        assert goal.evaluate({}) == 1.0

    def test_hard_soft_constraints(self):
        goal = SubAgentGoal(
            id="g1",
            name="Test",
            description="desc",
            constraints=[
                Constraint(id="h1", description="no delete", constraint_type=ConstraintType.HARD),
                Constraint(id="s1", description="prefer fast", constraint_type=ConstraintType.SOFT),
                Constraint(id="h2", description="no write", constraint_type=ConstraintType.HARD),
            ],
        )
        assert len(goal.hard_constraints()) == 2
        assert len(goal.soft_constraints()) == 1


# ============================================================
# Node Tests
# ============================================================


class TestSubAgentNode:
    def test_llm_generate_valid(self):
        node = SubAgentNode(
            id="n1", name="Generate", node_type=NodeType.LLM_GENERATE,
            system_prompt="You are a coder.", output_keys=["code"],
        )
        assert node.validate() == []

    def test_llm_generate_requires_prompt(self):
        node = SubAgentNode(id="n1", name="Gen", node_type=NodeType.LLM_GENERATE)
        errors = node.validate()
        assert any("system_prompt" in e for e in errors)

    def test_llm_tool_use_requires_tools(self):
        node = SubAgentNode(
            id="n1", name="Tool", node_type=NodeType.LLM_TOOL_USE,
            system_prompt="Use tools.",
        )
        errors = node.validate()
        assert any("tool" in e.lower() for e in errors)

    def test_function_requires_callable(self):
        node = SubAgentNode(id="n1", name="Func", node_type=NodeType.FUNCTION)
        errors = node.validate()
        assert any("callable" in e for e in errors)

    def test_router_requires_rules(self):
        node = SubAgentNode(id="n1", name="Route", node_type=NodeType.ROUTER)
        errors = node.validate()
        assert any("routing_rules" in e for e in errors)

    def test_interpolate_prompt(self):
        node = SubAgentNode(
            id="n1", name="Gen", node_type=NodeType.LLM_GENERATE,
            system_prompt="Fix {issue} in {file}.",
        )
        result = node.interpolate_prompt({"issue": "bug", "file": "main.py"})
        assert result == "Fix bug in main.py."

    def test_interpolate_missing_key_preserved(self):
        node = SubAgentNode(
            id="n1", name="Gen", node_type=NodeType.LLM_GENERATE,
            system_prompt="Fix {issue} in {file}.",
        )
        result = node.interpolate_prompt({"issue": "bug"})
        assert "{file}" in result


# ============================================================
# Edge Tests
# ============================================================


class TestSubAgentEdge:
    def test_on_success(self):
        edge = SubAgentEdge(id="e1", source="a", target="b", condition=EdgeCondition.ON_SUCCESS)
        assert edge.evaluate(success=True, context={})
        assert not edge.evaluate(success=False, context={})

    def test_on_failure(self):
        edge = SubAgentEdge(id="e1", source="a", target="b", condition=EdgeCondition.ON_FAILURE)
        assert edge.evaluate(success=False, context={})
        assert not edge.evaluate(success=True, context={})

    def test_always(self):
        edge = SubAgentEdge(id="e1", source="a", target="b", condition=EdgeCondition.ALWAYS)
        assert edge.evaluate(success=True, context={})
        assert edge.evaluate(success=False, context={})

    def test_conditional_truthiness(self):
        edge = SubAgentEdge(
            id="e1", source="a", target="b",
            condition=EdgeCondition.CONDITIONAL,
            condition_key="has_errors",
        )
        assert edge.evaluate(success=True, context={"has_errors": True})
        assert not edge.evaluate(success=True, context={"has_errors": False})
        assert not edge.evaluate(success=True, context={})

    def test_conditional_value_match(self):
        edge = SubAgentEdge(
            id="e1", source="a", target="b",
            condition=EdgeCondition.CONDITIONAL,
            condition_key="status",
            condition_value="ready",
        )
        assert edge.evaluate(success=True, context={"status": "ready"})
        assert not edge.evaluate(success=True, context={"status": "pending"})

    def test_conditional_requires_key(self):
        edge = SubAgentEdge(
            id="e1", source="a", target="b",
            condition=EdgeCondition.CONDITIONAL,
        )
        errors = edge.validate()
        assert any("condition_key" in e for e in errors)

    def test_validate_missing_source(self):
        edge = SubAgentEdge(id="e1", source="", target="b")
        errors = edge.validate()
        assert any("source" in e for e in errors)


# ============================================================
# Graph Tests
# ============================================================


def _make_simple_graph() -> SubAgentGraph:
    """Create a simple 2-node graph for testing."""
    goal = SubAgentGoal(id="g1", name="Test", description="Test graph")

    start = SubAgentNode(
        id="start", name="Start", node_type=NodeType.LLM_GENERATE,
        system_prompt="Generate code.", output_keys=["code"],
    )
    end = SubAgentNode(
        id="end", name="End", node_type=NodeType.LLM_GENERATE,
        system_prompt="Review {code}.", input_keys=["code"], output_keys=["review"],
    )

    edge = SubAgentEdge(id="e1", source="start", target="end", condition=EdgeCondition.ON_SUCCESS)

    graph = SubAgentGraph(
        id="test-graph", goal=goal,
        entry_node="start", terminal_nodes=["end"],
    )
    graph.add_node(start)
    graph.add_node(end)
    graph.add_edge(edge)
    return graph


class TestSubAgentGraph:
    def test_valid_graph(self):
        graph = _make_simple_graph()
        assert graph.validate(strict=False) == []

    def test_missing_entry_node(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        graph = SubAgentGraph(id="g", goal=goal, entry_node="")
        errors = graph.validate(strict=False)
        assert any("entry_node" in e for e in errors)

    def test_invalid_entry_node(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        node = SubAgentNode(id="n1", name="N", node_type=NodeType.LLM_GENERATE, system_prompt="p")
        graph = SubAgentGraph(id="g", goal=goal, entry_node="missing")
        graph.add_node(node)
        errors = graph.validate(strict=False)
        assert any("missing" in e for e in errors)

    def test_invalid_terminal_node(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        node = SubAgentNode(id="n1", name="N", node_type=NodeType.LLM_GENERATE, system_prompt="p")
        graph = SubAgentGraph(
            id="g", goal=goal, entry_node="n1", terminal_nodes=["missing"],
        )
        graph.add_node(node)
        errors = graph.validate(strict=False)
        assert any("Terminal node" in e and "missing" in e for e in errors)

    def test_unreachable_node(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        n1 = SubAgentNode(id="n1", name="A", node_type=NodeType.LLM_GENERATE, system_prompt="p")
        n2 = SubAgentNode(id="n2", name="B", node_type=NodeType.LLM_GENERATE, system_prompt="p")
        graph = SubAgentGraph(
            id="g", goal=goal, entry_node="n1", terminal_nodes=["n1", "n2"],
        )
        graph.add_node(n1)
        graph.add_node(n2)
        errors = graph.validate(strict=False)
        assert any("unreachable" in e for e in errors)

    def test_node_without_outgoing_edge(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        n1 = SubAgentNode(id="n1", name="A", node_type=NodeType.LLM_GENERATE, system_prompt="p")
        n2 = SubAgentNode(id="n2", name="B", node_type=NodeType.LLM_GENERATE, system_prompt="p")
        edge = SubAgentEdge(id="e1", source="n1", target="n2")
        graph = SubAgentGraph(id="g", goal=goal, entry_node="n1", terminal_nodes=["n2"])
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(edge)
        # n1 has edge, n2 is terminal — should be valid
        errors = graph.validate(strict=False)
        assert errors == []

    def test_strict_raises(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        graph = SubAgentGraph(id="g", goal=goal, entry_node="")
        with pytest.raises(GraphValidationError):
            graph.validate(strict=True)

    def test_get_outgoing_edges_sorted(self):
        graph = _make_simple_graph()
        e2 = SubAgentEdge(id="e2", source="start", target="end", condition=EdgeCondition.ON_FAILURE, priority=1)
        graph.add_edge(e2)
        edges = graph.get_outgoing_edges("start")
        assert edges[0].priority <= edges[1].priority

    def test_edge_references_unknown_node(self):
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        node = SubAgentNode(id="n1", name="N", node_type=NodeType.LLM_GENERATE, system_prompt="p")
        bad_edge = SubAgentEdge(id="e1", source="n1", target="ghost")
        graph = SubAgentGraph(id="g", goal=goal, entry_node="n1", terminal_nodes=["n1"])
        graph.add_node(node)
        graph.add_edge(bad_edge)
        errors = graph.validate(strict=False)
        assert any("ghost" in e for e in errors)


# ============================================================
# Executor Tests
# ============================================================


class TestSubAgentExecutor:
    @pytest.mark.asyncio
    async def test_execute_function_graph(self):
        """Test a graph with only FUNCTION nodes (no LLM needed)."""
        goal = SubAgentGoal(id="g1", name="Math", description="Add numbers")

        async def add(ctx):
            return {"result": ctx.get("a", 0) + ctx.get("b", 0)}

        async def double(ctx):
            return {"final": ctx.get("result", 0) * 2}

        n1 = SubAgentNode(
            id="add", name="Add", node_type=NodeType.FUNCTION,
            input_keys=["a", "b"], output_keys=["result"], function=add,
        )
        n2 = SubAgentNode(
            id="double", name="Double", node_type=NodeType.FUNCTION,
            input_keys=["result"], output_keys=["final"], function=double,
        )
        edge = SubAgentEdge(id="e1", source="add", target="double")

        graph = SubAgentGraph(
            id="math", goal=goal, entry_node="add", terminal_nodes=["double"],
        )
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_edge(edge)

        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        executor = SubAgentExecutor(mock_provider, mock_registry)
        result = await executor.execute(graph, initial_context={"a": 3, "b": 7})

        assert result.success
        assert result.output["result"] == 10
        assert result.output["final"] == 20
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_execute_router_node(self):
        """Test a graph with a ROUTER node."""
        goal = SubAgentGoal(id="g1", name="Route", description="Route by type")

        async def process_a(ctx):
            return {"output": "processed A"}

        async def process_b(ctx):
            return {"output": "processed B"}

        router = SubAgentNode(
            id="router", name="Router", node_type=NodeType.ROUTER,
            input_keys=["type"],
            routing_rules=[
                ("type", "a", "proc_a"),
                ("type", "b", "proc_b"),
                ("default", None, "proc_b"),
            ],
        )
        proc_a = SubAgentNode(
            id="proc_a", name="Process A", node_type=NodeType.FUNCTION,
            output_keys=["output"], function=process_a,
        )
        proc_b = SubAgentNode(
            id="proc_b", name="Process B", node_type=NodeType.FUNCTION,
            output_keys=["output"], function=process_b,
        )

        graph = SubAgentGraph(
            id="route", goal=goal, entry_node="router",
            terminal_nodes=["proc_a", "proc_b"],
        )
        graph.add_node(router)
        graph.add_node(proc_a)
        graph.add_node(proc_b)
        # Router uses routing_rules, but we still need edges for validation
        graph.add_edge(SubAgentEdge(id="e1", source="router", target="proc_a", condition=EdgeCondition.ALWAYS))
        graph.add_edge(SubAgentEdge(id="e2", source="router", target="proc_b", condition=EdgeCondition.ALWAYS))

        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        executor = SubAgentExecutor(mock_provider, mock_registry)
        result = await executor.execute(graph, initial_context={"type": "a"})

        assert result.success
        assert result.output["output"] == "processed A"

    @pytest.mark.asyncio
    async def test_execute_failure_edge(self):
        """Test that ON_FAILURE edges are followed on node failure."""
        goal = SubAgentGoal(id="g1", name="Fail", description="Handle failure")

        async def fail_func(ctx):
            raise RuntimeError("intentional error")

        async def recover_func(ctx):
            return {"recovered": True}

        fail_node = SubAgentNode(
            id="fail", name="Fail", node_type=NodeType.FUNCTION,
            function=fail_func, max_retries=0,
        )
        recover_node = SubAgentNode(
            id="recover", name="Recover", node_type=NodeType.FUNCTION,
            output_keys=["recovered"], function=recover_func,
        )

        graph = SubAgentGraph(
            id="failover", goal=goal, entry_node="fail",
            terminal_nodes=["recover"],
        )
        graph.add_node(fail_node)
        graph.add_node(recover_node)
        graph.add_edge(SubAgentEdge(
            id="e1", source="fail", target="recover", condition=EdgeCondition.ON_FAILURE,
        ))

        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        executor = SubAgentExecutor(mock_provider, mock_registry)
        result = await executor.execute(graph)

        assert result.success
        assert result.output.get("recovered") is True
        assert result.steps[0].success is False  # First node failed
        assert result.steps[1].success is True   # Recovery succeeded

    @pytest.mark.asyncio
    async def test_execute_retry_on_failure(self):
        """Test that nodes retry before propagating failure."""
        goal = SubAgentGoal(id="g1", name="Retry", description="Retry test")

        call_count = 0

        async def flaky_func(ctx):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return {"done": True}

        node = SubAgentNode(
            id="flaky", name="Flaky", node_type=NodeType.FUNCTION,
            function=flaky_func, max_retries=3,
        )

        graph = SubAgentGraph(
            id="retry", goal=goal, entry_node="flaky",
            terminal_nodes=["flaky"],
        )
        graph.add_node(node)

        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        executor = SubAgentExecutor(mock_provider, mock_registry)
        result = await executor.execute(graph)

        assert result.success
        assert result.steps[0].retries == 2  # Succeeded on 3rd attempt
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_invalid_start_node(self):
        """Test error when start node doesn't exist."""
        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        graph = SubAgentGraph(id="g", goal=goal, entry_node="missing")

        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        executor = SubAgentExecutor(mock_provider, mock_registry)
        # Can't use validate() since it would raise, test executor directly
        result = await executor.execute(graph, resume_from="missing")

        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Test that infinite loops are caught."""
        goal = SubAgentGoal(id="g1", name="Loop", description="Infinite loop")

        async def noop(ctx):
            return {}

        node = SubAgentNode(
            id="loop", name="Loop", node_type=NodeType.FUNCTION,
            function=noop,
        )

        graph = SubAgentGraph(id="loop", goal=goal, entry_node="loop")
        graph.add_node(node)
        # Self-loop edge
        graph.add_edge(SubAgentEdge(id="e1", source="loop", target="loop", condition=EdgeCondition.ALWAYS))

        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        executor = SubAgentExecutor(mock_provider, mock_registry)
        result = await executor.execute(graph)

        assert not result.success
        assert "Circuit breaker" in result.error

    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        """Test graph pause at pause node and resume."""
        goal = SubAgentGoal(id="g1", name="Pause", description="Pause test")

        async def step1(ctx):
            return {"step1_done": True}

        async def step2(ctx):
            return {"step2_done": True}

        n1 = SubAgentNode(
            id="step1", name="Step1", node_type=NodeType.FUNCTION,
            output_keys=["step1_done"], function=step1,
        )
        n2 = SubAgentNode(
            id="pause", name="Pause", node_type=NodeType.FUNCTION,
            output_keys=["step2_done"], function=step2,
        )
        n3 = SubAgentNode(
            id="step3", name="Step3", node_type=NodeType.FUNCTION,
            output_keys=["final"], function=lambda ctx: asyncio.coroutine(lambda: {"final": True})(),
        )

        graph = SubAgentGraph(
            id="pause-test", goal=goal, entry_node="step1",
            terminal_nodes=["step3"], pause_nodes=["pause"],
        )
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        graph.add_edge(SubAgentEdge(id="e1", source="step1", target="pause"))
        graph.add_edge(SubAgentEdge(id="e2", source="pause", target="step3"))

        mock_provider = MagicMock()
        mock_registry = MagicMock()
        mock_registry.list_tools.return_value = []

        executor = SubAgentExecutor(mock_provider, mock_registry)

        # First execution — should pause
        result1 = await executor.execute(graph)
        assert result1.success
        assert result1.paused_at == "pause"
        assert result1.output.get("step1_done") is True
        assert len(result1.steps) == 1  # Only step1 executed

        # Resume from pause
        result2 = await executor.execute(
            graph,
            resume_from="pause",
            session_state=result1.output,
        )
        assert result2.success
        assert result2.paused_at is None
        assert result2.output.get("step2_done") is True


# ============================================================
# Session Tests
# ============================================================


class TestSessionState:
    def test_round_trip(self):
        state = SessionState(
            session_id="s1",
            graph_id="g1",
            paused_at="node2",
            context={"key": "value"},
            steps_completed=["node1"],
        )
        d = state.to_dict()
        restored = SessionState.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.paused_at == "node2"
        assert restored.context == {"key": "value"}


class TestSessionStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            state = SessionState(
                session_id="test-123",
                graph_id="g1",
                paused_at="n2",
                context={"step": 1},
            )
            store.save(state)

            loaded = store.load("test-123")
            assert loaded is not None
            assert loaded.session_id == "test-123"
            assert loaded.paused_at == "n2"
            assert loaded.context == {"step": 1}
            assert loaded.created_at > 0

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            assert store.load("nonexistent") is None

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            state = SessionState(session_id="del-me", graph_id="g1", paused_at="n1")
            store.save(state)
            assert store.delete("del-me")
            assert store.load("del-me") is None
            assert not store.delete("del-me")  # Already deleted

    def test_list_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            for i in range(3):
                state = SessionState(session_id=f"s{i}", graph_id="g1", paused_at=f"n{i}")
                store.save(state)
            sessions = store.list_sessions()
            assert len(sessions) == 3

    def test_list_for_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir))
            store.save(SessionState(session_id="s1", graph_id="g1", paused_at="n1"))
            store.save(SessionState(session_id="s2", graph_id="g2", paused_at="n1"))
            store.save(SessionState(session_id="s3", graph_id="g1", paused_at="n2"))
            assert len(store.list_for_graph("g1")) == 2
            assert len(store.list_for_graph("g2")) == 1


# ============================================================
# Cleaner Tests
# ============================================================


class TestOutputCleaner:
    def test_validate_missing_keys(self):
        cleaner = OutputCleaner()
        errors = cleaner.validate({"a": 1}, expected_keys=["a", "b"])
        assert len(errors) == 1
        assert "b" in errors[0]

    def test_validate_all_keys_present(self):
        cleaner = OutputCleaner()
        errors = cleaner.validate({"a": 1, "b": 2}, expected_keys=["a", "b"])
        assert errors == []

    def test_validate_type_check(self):
        cleaner = OutputCleaner()
        errors = cleaner.validate(
            {"count": "not_a_number"},
            expected_keys=["count"],
            schema={"count": "integer"},
        )
        assert any("type" in e.lower() for e in errors)

    def test_detect_json_trap(self):
        cleaner = OutputCleaner()
        # Long structured content in single key = trap
        long_content = "## Header\n\n" + "- item\n" * 100
        errors = cleaner.validate(
            {"response": long_content},
            expected_keys=["response"],
        )
        assert any("JSON trap" in e for e in errors)

    def test_no_json_trap_for_short_values(self):
        cleaner = OutputCleaner()
        errors = cleaner.validate(
            {"response": "short answer"},
            expected_keys=["response"],
        )
        assert not any("JSON trap" in e for e in errors)

    def test_clean_json(self):
        cleaner = OutputCleaner()
        result = cleaner.clean('{"code": "print(1)", "lang": "python"}', ["code"])
        assert result is not None
        assert result["code"] == "print(1)"

    def test_clean_markdown_code_block(self):
        cleaner = OutputCleaner()
        text = 'Here is the result:\n```json\n{"code": "x = 1"}\n```'
        result = cleaner.clean(text, ["code"])
        assert result is not None
        assert result["code"] == "x = 1"

    def test_clean_fallback_to_text(self):
        cleaner = OutputCleaner()
        result = cleaner.clean("just some text", ["output"])
        assert result is not None
        assert result["output"] == "just some text"

    def test_clean_no_keys_no_json(self):
        cleaner = OutputCleaner()
        result = cleaner.clean("just some text", [])
        assert result is None  # Can't extract without JSON or keys


# ============================================================
# Integration: Orchestrator + Graph
# ============================================================


class TestOrchestratorGraphIntegration:
    def test_validate_graph_tools_warns(self):
        """Test that orchestrator warns about missing tools."""
        from src.core.subagent import SubAgentOrchestrator

        mock_provider = MagicMock()
        mock_registry = MagicMock()

        # Registry has no tools
        mock_tool = MagicMock()
        mock_tool.name = "read_file"
        mock_registry.list_tools.return_value = [mock_tool]

        orchestrator = SubAgentOrchestrator(mock_provider, mock_registry)

        goal = SubAgentGoal(id="g1", name="Test", description="desc")
        node = SubAgentNode(
            id="n1", name="Tool User", node_type=NodeType.LLM_TOOL_USE,
            system_prompt="Use tools.", tools=["read_file", "nonexistent_tool"],
        )
        graph = SubAgentGraph(
            id="g", goal=goal, entry_node="n1", terminal_nodes=["n1"],
        )
        graph.add_node(node)

        # Should log warning, not raise
        with patch("src.core.subagent.logger") as mock_logger:
            orchestrator._validate_graph_tools(graph)
            mock_logger.warning.assert_called_once()
