"""Tests for knowledge compounding."""

import json
import tempfile
from pathlib import Path

import pytest
from src.core.knowledge import (
    SolutionRecord,
    SearchHit,
    KnowledgeStore,
)
from src.core.agent import Agent, AgentConfig


class TestSolutionRecord:
    """Tests for SolutionRecord."""

    def test_defaults(self):
        r = SolutionRecord(task="fix bug", approach="patched it")
        assert r.task == "fix bug"
        assert r.approach == "patched it"
        assert r.outcome == "success"
        assert r.files_changed == []
        assert r.tags == []
        assert r.timestamp > 0

    def test_with_metadata(self):
        r = SolutionRecord(
            task="add feature",
            approach="wrote code",
            files_changed=["main.py"],
            tags=["feature", "ui"],
            model_used="gpt-4o",
        )
        assert r.files_changed == ["main.py"]
        assert r.tags == ["feature", "ui"]
        assert r.model_used == "gpt-4o"

    def test_to_dict_from_dict(self):
        r = SolutionRecord(
            task="fix NaN",
            approach="analytical rotation",
            tags=["numerical", "cuda"],
            files_changed=["processor.py"],
        )
        d = r.to_dict()
        r2 = SolutionRecord.from_dict(d)
        assert r2.task == r.task
        assert r2.approach == r.approach
        assert r2.tags == r.tags
        assert r2.files_changed == r.files_changed

    def test_from_dict_ignores_extra_keys(self):
        d = {"task": "x", "approach": "y", "unknown_field": 42}
        r = SolutionRecord.from_dict(d)
        assert r.task == "x"

    def test_matches_basic(self):
        r = SolutionRecord(task="fix NaN bug", approach="use analytical rotation")
        assert r.matches("NaN") > 0
        assert r.matches("NaN bug") >= r.matches("NaN")
        assert r.matches("completely unrelated xyz") == 0

    def test_matches_tags(self):
        r = SolutionRecord(task="fix bug", approach="patch", tags=["cuda", "numerical"])
        assert r.matches("cuda") > 0
        assert r.matches("numerical cuda") >= r.matches("cuda")

    def test_matches_empty_query(self):
        r = SolutionRecord(task="anything", approach="whatever")
        assert r.matches("") == 0

    def test_matches_task_bonus(self):
        r = SolutionRecord(task="authentication error", approach="rotate keys")
        score_task = r.matches("authentication")
        # A term that matches only in approach, not task
        r2 = SolutionRecord(task="fix issue", approach="authentication fix")
        score_approach = r2.matches("authentication")
        # Task match should score higher due to bonus
        assert score_task >= score_approach


class TestKnowledgeStore:
    """Tests for KnowledgeStore."""

    @pytest.fixture
    def tmp_store(self, tmp_path):
        path = tmp_path / "solutions.jsonl"
        return KnowledgeStore(path=path)

    def test_empty_store(self, tmp_store):
        assert tmp_store.count() == 0
        assert tmp_store.get_all() == []

    def test_record_and_count(self, tmp_store):
        tmp_store.record(SolutionRecord(task="t1", approach="a1"))
        assert tmp_store.count() == 1
        tmp_store.record(SolutionRecord(task="t2", approach="a2"))
        assert tmp_store.count() == 2

    def test_record_persists(self, tmp_path):
        path = tmp_path / "solutions.jsonl"
        store1 = KnowledgeStore(path=path)
        store1.record(SolutionRecord(task="persist test", approach="write"))
        # New store instance reads same file
        store2 = KnowledgeStore(path=path)
        assert store2.count() == 1
        assert store2.get_all()[0].task == "persist test"

    def test_search_basic(self, tmp_store):
        tmp_store.record(SolutionRecord(task="fix NaN bug", approach="analytical rotation"))
        tmp_store.record(SolutionRecord(task="add auth", approach="JWT tokens"))
        results = tmp_store.search("NaN")
        assert len(results) == 1
        assert results[0].record.task == "fix NaN bug"
        assert results[0].score > 0

    def test_search_no_results(self, tmp_store):
        tmp_store.record(SolutionRecord(task="something", approach="else"))
        results = tmp_store.search("completely nonexistent xyzzy")
        assert len(results) == 0

    def test_search_outcome_filter(self, tmp_store):
        tmp_store.record(SolutionRecord(task="failed task", approach="bad approach", outcome="failed"))
        tmp_store.record(SolutionRecord(task="good task", approach="good approach", outcome="success"))
        results = tmp_store.search("task", outcome_filter="success")
        assert len(results) == 1
        assert results[0].record.outcome == "success"

    def test_search_k_limit(self, tmp_store):
        for i in range(10):
            tmp_store.record(SolutionRecord(
                task=f"task number {i}",
                approach=f"approach {i}",
            ))
        results = tmp_store.search("task", k=3)
        assert len(results) <= 3

    def test_search_min_score(self, tmp_store):
        tmp_store.record(SolutionRecord(task="exact match query", approach="found"))
        tmp_store.record(SolutionRecord(task="vague relation", approach="tangential"))
        results = tmp_store.search("exact match query", min_score=0.5)
        assert all(h.score >= 0.5 for h in results)

    def test_format_context(self, tmp_store):
        tmp_store.record(SolutionRecord(
            task="fix auth bug",
            approach="rotate API keys",
            tags=["auth", "api"],
        ))
        hits = tmp_store.search("auth")
        ctx = tmp_store.format_context(hits)
        assert "Past solutions" in ctx
        assert "fix auth bug" in ctx
        assert "rotate API keys" in ctx
        assert "auth, api" in ctx

    def test_format_context_empty(self, tmp_store):
        ctx = tmp_store.format_context([])
        assert ctx == ""

    def test_format_context_max_chars(self, tmp_store):
        for i in range(20):
            tmp_store.record(SolutionRecord(
                task=f"very long task description number {i} " * 5,
                approach=f"detailed approach {i} " * 5,
            ))
        hits = tmp_store.search("task", k=20)
        ctx = tmp_store.format_context(hits, max_chars=200)
        assert len(ctx) <= 300  # Some tolerance for last entry

    def test_clear(self, tmp_store):
        tmp_store.record(SolutionRecord(task="t", approach="a"))
        assert tmp_store.count() == 1
        tmp_store.clear()
        assert tmp_store.count() == 0

    def test_stats(self, tmp_store):
        tmp_store.record(SolutionRecord(task="t1", approach="a1", tags=["a", "b"]))
        tmp_store.record(SolutionRecord(task="t2", approach="a2", tags=["b", "c"], outcome="failed"))
        stats = tmp_store.stats()
        assert stats["total_records"] == 2
        assert stats["outcomes"]["success"] == 1
        assert stats["outcomes"]["failed"] == 1
        assert stats["unique_tags"] == 3

    def test_malformed_jsonl_line_skipped(self, tmp_path):
        path = tmp_path / "solutions.jsonl"
        with open(path, "w") as f:
            f.write('{"task":"good","approach":"ok"}\n')
            f.write('not valid json\n')
            f.write('{"task":"also good","approach":"fine"}\n')
        store = KnowledgeStore(path=path)
        assert store.count() == 2

    def test_cache_invalidation(self, tmp_store):
        tmp_store.record(SolutionRecord(task="t1", approach="a1"))
        assert tmp_store.count() == 1  # populates cache
        tmp_store.record(SolutionRecord(task="t2", approach="a2"))
        assert tmp_store.count() == 2  # cache should be invalidated


class TestAgentKnowledgeIntegration:
    """Tests for Agent integration with knowledge store."""

    @pytest.fixture
    def mock_provider(self):
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = ""
                    tool_calls = None
                return Result()
        return MockProvider()

    def test_knowledge_enabled_by_default(self, mock_provider):
        agent = Agent(provider=mock_provider)
        assert agent._knowledge_store is not None

    def test_knowledge_disabled(self, mock_provider):
        config = AgentConfig(enable_knowledge=False)
        agent = Agent(provider=mock_provider, config=config)
        assert agent._knowledge_store is None

    def test_record_solution(self, mock_provider, tmp_path):
        agent = Agent(provider=mock_provider)
        agent._knowledge_store = KnowledgeStore(path=tmp_path / "solutions.jsonl")
        agent.record_solution(
            task="fix NaN",
            approach="analytical rotation",
            tags=["numerical"],
            files_changed=["processor.py"],
        )
        assert agent._knowledge_store.count() == 1
        records = agent._knowledge_store.get_all()
        assert records[0].task == "fix NaN"

    def test_record_solution_noop_when_disabled(self, mock_provider):
        config = AgentConfig(enable_knowledge=False)
        agent = Agent(provider=mock_provider, config=config)
        agent.record_solution(task="x", approach="y")  # should not raise

    def test_search_knowledge(self, mock_provider, tmp_path):
        agent = Agent(provider=mock_provider)
        agent._knowledge_store = KnowledgeStore(path=tmp_path / "solutions.jsonl")
        agent._knowledge_store.record(SolutionRecord(
            task="fix auth failure",
            approach="rotate API keys",
            tags=["auth"],
        ))
        result = agent._search_knowledge("auth failure")
        assert result is not None
        assert "auth" in result

    def test_search_knowledge_no_results(self, mock_provider, tmp_path):
        agent = Agent(provider=mock_provider)
        agent._knowledge_store = KnowledgeStore(path=tmp_path / "solutions.jsonl")
        result = agent._search_knowledge("nonexistent xyzzy")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_context_includes_knowledge(self, tmp_path):
        class MockProvider:
            is_available = True
            async def generate(self, **kwargs):
                class Result:
                    content = ""
                    tool_calls = None
                return Result()

        config = AgentConfig(use_memory=False, enable_compaction=False)
        agent = Agent(provider=MockProvider(), config=config)
        agent._knowledge_store = KnowledgeStore(path=tmp_path / "solutions.jsonl")
        agent._knowledge_store.record(SolutionRecord(
            task="fix NaN divergence",
            approach="use analytical phase rotation",
            tags=["numerical"],
        ))

        ctx = await agent._retrieve_context("NaN divergence")
        assert ctx is not None
        assert "analytical phase rotation" in ctx
