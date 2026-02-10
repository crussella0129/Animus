"""Tests for the code knowledge graph: parser, graph_db, indexer, tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge.parser import PythonParser, NodeInfo, EdgeInfo, FileParseResult
from src.knowledge.graph_db import GraphDB, NodeRow
from src.knowledge.indexer import Indexer, IndexResult


# =====================================================================
# Parser tests
# =====================================================================


class TestPythonParser:

    def test_parse_file_returns_result(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        assert isinstance(result, FileParseResult)
        assert result.file_path == str(sample_python_file)

    def test_parse_file_extracts_module_node(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        modules = [n for n in result.nodes if n.kind == "module"]
        assert len(modules) == 1
        assert modules[0].name == "animals"
        assert "Sample module docstring" in modules[0].docstring

    def test_parse_file_extracts_classes(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        classes = [n for n in result.nodes if n.kind == "class"]
        names = {c.name for c in classes}
        assert names == {"Animal", "Dog", "Cat"}

    def test_parse_file_extracts_methods(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        methods = [n for n in result.nodes if n.kind == "method"]
        names = {m.name for m in methods}
        assert "speak" in names
        assert "__init__" in names
        assert "fetch" in names

    def test_parse_file_extracts_functions(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        functions = [n for n in result.nodes if n.kind == "function"]
        names = {f.name for f in functions}
        assert "make_animal" in names
        assert "main" in names

    def test_parse_file_extracts_inheritance(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        inherits = [e for e in result.edges if e.kind == "INHERITS"]
        sources = {e.source_qname.split(".")[-1] for e in inherits}
        assert "Dog" in sources
        assert "Cat" in sources

    def test_parse_file_extracts_calls(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        calls = [e for e in result.edges if e.kind == "CALLS"]
        # main() calls make_animal and animal.speak
        main_calls = [e for e in calls if "main" in e.source_qname]
        target_names = {e.target_name for e in main_calls}
        assert "make_animal" in target_names

    def test_parse_file_extracts_imports(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        imports = [e for e in result.edges if e.kind == "IMPORTS"]
        target_names = {e.target_name for e in imports}
        assert "os" in target_names
        assert "pathlib.Path" in target_names

    def test_parse_file_extracts_contains_edges(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        contains = [e for e in result.edges if e.kind == "CONTAINS"]
        assert len(contains) > 0
        # Module contains Animal class
        module_contains = [e for e in contains if e.kind == "CONTAINS" and "Animal" in e.target_name]
        assert len(module_contains) >= 1

    def test_parse_file_docstrings(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        animal = next(n for n in result.nodes if n.name == "Animal")
        assert "Base animal class" in animal.docstring

    def test_parse_file_line_numbers(self, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        animal = next(n for n in result.nodes if n.name == "Animal")
        assert animal.line_start > 0
        assert animal.line_end >= animal.line_start

    def test_parse_file_syntax_error(self, tmp_path: Path):
        bad = tmp_path / "bad.py"
        bad.write_text("def broken(:\n  pass\n")
        parser = PythonParser()
        result = parser.parse_file(bad)
        assert result.nodes == []
        assert result.edges == []

    def test_parse_file_nonexistent(self, tmp_path: Path):
        parser = PythonParser()
        result = parser.parse_file(tmp_path / "nope.py")
        assert result.nodes == []


# =====================================================================
# GraphDB tests
# =====================================================================


class TestGraphDB:

    def test_empty_db_stats(self, graph_db: GraphDB):
        stats = graph_db.get_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["files"] == 0

    def test_upsert_and_search(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)

        nodes = graph_db.search_nodes("Animal")
        assert len(nodes) >= 1
        assert any(n.name == "Animal" for n in nodes)

    def test_upsert_idempotent(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)
        stats1 = graph_db.get_stats()

        # Upsert same file again
        graph_db.upsert_file_results(result, "abc123", 1001.0)
        stats2 = graph_db.get_stats()
        assert stats2["nodes"] == stats1["nodes"]

    def test_search_by_kind(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)

        classes = graph_db.search_nodes("", kind="class")
        assert all(n.kind == "class" for n in classes)
        assert len(classes) == 3  # Animal, Dog, Cat

    def test_get_callers(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)

        # Find make_animal's qualified name
        make_nodes = graph_db.search_nodes("make_animal", kind="function")
        assert len(make_nodes) >= 1
        qname = make_nodes[0].qualified_name

        callers = graph_db.get_callers(qname)
        caller_names = {c.name for c in callers}
        assert "main" in caller_names

    def test_get_callees(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)

        # main calls make_animal
        main_nodes = graph_db.search_nodes("main", kind="function")
        assert len(main_nodes) >= 1
        qname = main_nodes[0].qualified_name
        callees = graph_db.get_callees(qname)
        callee_names = {c.name for c in callees}
        assert "make_animal" in callee_names

    def test_get_inheritance_tree(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)

        animal_nodes = graph_db.search_nodes("Animal", kind="class")
        # Get the one that is exactly "Animal" (not Dog or Cat)
        animal = next(n for n in animal_nodes if n.name == "Animal")

        subclasses = graph_db.get_inheritance_tree(animal.qualified_name)
        sub_names = {s.name for s in subclasses}
        assert "Dog" in sub_names
        assert "Cat" in sub_names

    def test_get_blast_radius(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)

        # make_animal is called by main → blast radius depth 1 includes main
        make_nodes = graph_db.search_nodes("make_animal", kind="function")
        qname = make_nodes[0].qualified_name

        radius = graph_db.get_blast_radius(qname, max_depth=3)
        all_affected = [n for nodes in radius.values() for n in nodes]
        affected_names = {n.name for n in all_affected}
        assert "main" in affected_names

    def test_remove_file(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)
        assert graph_db.get_stats()["nodes"] > 0

        graph_db.remove_file(str(sample_python_file))
        # Only external phantom nodes may remain
        remaining = graph_db.search_nodes("Animal", kind="class")
        assert len(remaining) == 0


# =====================================================================
# Indexer tests
# =====================================================================


class TestIndexer:

    def test_index_directory(self, graph_db: GraphDB, tmp_path: Path):
        # Create a small project
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("def main():\n    pass\n")
        (src / "utils.py").write_text("def helper():\n    pass\n")

        indexer = Indexer(graph_db)
        result = indexer.index_directory(tmp_path)
        assert result.files_scanned == 2
        assert result.files_parsed == 2
        assert result.files_skipped == 0
        assert result.total_nodes > 0

    def test_index_skips_unchanged(self, graph_db: GraphDB, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("def main():\n    pass\n")

        indexer = Indexer(graph_db)
        r1 = indexer.index_directory(tmp_path)
        assert r1.files_parsed == 1

        r2 = indexer.index_directory(tmp_path)
        assert r2.files_skipped == 1
        assert r2.files_parsed == 0

    def test_index_single_file(self, graph_db: GraphDB, tmp_path: Path):
        f = tmp_path / "single.py"
        f.write_text("class Foo:\n    pass\n")

        indexer = Indexer(graph_db)
        result = indexer.index_file(f)
        assert result.files_parsed == 1
        assert result.total_nodes > 0


# =====================================================================
# Tools tests
# =====================================================================


class TestGraphTools:

    def _populate_db(self, graph_db: GraphDB, sample_python_file: Path):
        parser = PythonParser()
        result = parser.parse_file(sample_python_file)
        graph_db.upsert_file_results(result, "abc123", 1000.0)

    def test_search_code_graph_tool(self, graph_db: GraphDB, sample_python_file: Path):
        from src.tools.graph import SearchCodeGraphTool
        self._populate_db(graph_db, sample_python_file)
        tool = SearchCodeGraphTool(graph_db)
        result = tool.execute({"pattern": "Animal"})
        assert "Animal" in result

    def test_search_code_graph_no_results(self, graph_db: GraphDB):
        from src.tools.graph import SearchCodeGraphTool
        tool = SearchCodeGraphTool(graph_db)
        result = tool.execute({"pattern": "NonexistentSymbol"})
        assert "No symbols found" in result

    def test_get_callers_tool(self, graph_db: GraphDB, sample_python_file: Path):
        from src.tools.graph import GetCallersTool
        self._populate_db(graph_db, sample_python_file)

        # Find make_animal qname
        make_nodes = graph_db.search_nodes("make_animal", kind="function")
        qname = make_nodes[0].qualified_name

        tool = GetCallersTool(graph_db)
        result = tool.execute({"symbol": qname})
        assert "main" in result

    def test_get_callers_tool_no_results(self, graph_db: GraphDB):
        from src.tools.graph import GetCallersTool
        tool = GetCallersTool(graph_db)
        result = tool.execute({"symbol": "nonexistent.symbol"})
        assert "No callers found" in result

    def test_get_blast_radius_tool(self, graph_db: GraphDB, sample_python_file: Path):
        from src.tools.graph import GetBlastRadiusTool
        self._populate_db(graph_db, sample_python_file)

        make_nodes = graph_db.search_nodes("make_animal", kind="function")
        qname = make_nodes[0].qualified_name

        tool = GetBlastRadiusTool(graph_db)
        result = tool.execute({"symbol": qname})
        assert "main" in result

    def test_register_graph_tools(self, graph_db: GraphDB):
        from src.tools.base import ToolRegistry
        from src.tools.graph import register_graph_tools

        registry = ToolRegistry()
        register_graph_tools(registry, graph_db)
        names = registry.names()
        assert "search_code_graph" in names
        assert "get_callers" in names
        assert "get_blast_radius" in names
