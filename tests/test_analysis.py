"""Tests for code analysis module."""

import pytest
import tempfile
from pathlib import Path

from src.analysis.parser import (
    CodeParser,
    ParsedCode,
    CodeSymbol,
    SymbolType,
    is_tree_sitter_available,
    get_supported_languages,
)


class TestParserBasics:
    """Basic parser tests that don't require tree-sitter."""

    def test_symbol_type_enum(self):
        """Test SymbolType enum values."""
        assert SymbolType.FUNCTION.value == "function"
        assert SymbolType.CLASS.value == "class"
        assert SymbolType.METHOD.value == "method"
        assert SymbolType.IMPORT.value == "import"

    def test_code_symbol_to_dict(self):
        """Test CodeSymbol serialization."""
        symbol = CodeSymbol(
            name="test_func",
            type=SymbolType.FUNCTION,
            line=10,
            column=0,
            end_line=20,
            end_column=0,
            docstring="A test function",
            signature="(arg1, arg2)",
        )
        d = symbol.to_dict()
        assert d["name"] == "test_func"
        assert d["type"] == "function"
        assert d["line"] == 10
        assert d["signature"] == "(arg1, arg2)"

    def test_parsed_code_get_functions(self):
        """Test ParsedCode filtering methods."""
        symbols = [
            CodeSymbol("func1", SymbolType.FUNCTION, 1, 0, 5, 0),
            CodeSymbol("MyClass", SymbolType.CLASS, 10, 0, 30, 0),
            CodeSymbol("func2", SymbolType.FUNCTION, 40, 0, 45, 0),
            CodeSymbol("method1", SymbolType.METHOD, 15, 4, 20, 4, parent="MyClass"),
        ]
        parsed = ParsedCode(
            language="python",
            source="",
            symbols=symbols,
        )

        assert len(parsed.get_functions()) == 2
        assert len(parsed.get_classes()) == 1
        assert len(parsed.get_methods()) == 1

    def test_parsed_code_find_symbol(self):
        """Test finding symbols by name."""
        symbols = [
            CodeSymbol("func1", SymbolType.FUNCTION, 1, 0, 5, 0),
            CodeSymbol("MyClass", SymbolType.CLASS, 10, 0, 30, 0),
        ]
        parsed = ParsedCode(language="python", source="", symbols=symbols)

        found = parsed.find_symbol("MyClass")
        assert found is not None
        assert found.type == SymbolType.CLASS

        not_found = parsed.find_symbol("nonexistent")
        assert not_found is None

    def test_parser_detect_language(self):
        """Test language detection from file extension."""
        parser = CodeParser()
        assert parser.detect_language("test.py") == "python"
        assert parser.detect_language("test.js") == "javascript"
        assert parser.detect_language("test.ts") == "typescript"
        assert parser.detect_language("test.tsx") == "tsx"
        assert parser.detect_language("test.rs") == "rust"
        assert parser.detect_language("test.go") == "go"
        assert parser.detect_language("test.java") == "java"
        assert parser.detect_language("test.c") == "c"
        assert parser.detect_language("test.cpp") == "cpp"
        assert parser.detect_language("test.unknown") is None


class TestParserWithTreeSitter:
    """Tests that require tree-sitter to be installed."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return CodeParser()

    @pytest.mark.skipif(
        not is_tree_sitter_available(),
        reason="tree-sitter not installed"
    )
    def test_is_tree_sitter_available(self):
        """Test tree-sitter availability check."""
        assert is_tree_sitter_available() is True

    @pytest.mark.skipif(
        not is_tree_sitter_available(),
        reason="tree-sitter not installed"
    )
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        langs = get_supported_languages()
        assert isinstance(langs, list)
        # At minimum, python should be supported if tree-sitter is available
        # (depends on which language modules are installed)

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    def test_parse_python_function(self, parser):
        """Test parsing a Python function."""
        source = '''
def hello_world(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        parsed = parser.parse(source, language="python")

        assert parsed.language == "python"
        assert len(parsed.get_functions()) == 1

        func = parsed.get_functions()[0]
        assert func.name == "hello_world"
        assert func.type == SymbolType.FUNCTION
        assert "Say hello" in (func.docstring or "")

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    def test_parse_python_class(self, parser):
        """Test parsing a Python class."""
        source = '''
class MyClass:
    """A test class."""

    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
'''
        parsed = parser.parse(source, language="python")

        assert len(parsed.get_classes()) == 1
        cls = parsed.get_classes()[0]
        assert cls.name == "MyClass"

        methods = parsed.get_methods()
        assert len(methods) == 2
        method_names = {m.name for m in methods}
        assert "__init__" in method_names
        assert "get_value" in method_names

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    def test_parse_python_imports(self, parser):
        """Test parsing Python imports."""
        source = '''
import os
from pathlib import Path
from typing import Optional, List
'''
        parsed = parser.parse(source, language="python")

        assert len(parsed.imports) == 3
        assert any("import os" in imp for imp in parsed.imports)
        assert any("pathlib" in imp for imp in parsed.imports)

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    def test_parse_file(self, parser):
        """Test parsing a file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def test_func():\n    pass\n")
            f.flush()

            parsed = parser.parse_file(f.name)

            assert parsed.language == "python"
            assert len(parsed.get_functions()) == 1
            assert parsed.get_functions()[0].name == "test_func"

            # Cleanup
            Path(f.name).unlink()

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    def test_parse_nonexistent_file(self, parser):
        """Test parsing a nonexistent file."""
        parsed = parser.parse_file("/nonexistent/file.py")

        assert len(parsed.errors) > 0
        assert "not found" in parsed.errors[0].lower()

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    def test_get_structure(self, parser):
        """Test getting code structure."""
        source = '''
import os

class Calculator:
    def add(self, a, b):
        return a + b

def main():
    pass
'''
        parsed = parser.parse(source, language="python")
        structure = parsed.get_structure()

        assert structure["language"] == "python"
        assert len(structure["imports"]) == 1
        assert len(structure["classes"]) == 1
        assert structure["classes"][0]["name"] == "Calculator"
        assert len(structure["functions"]) == 1


class TestAnalysisTools:
    """Tests for analysis tools."""

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    @pytest.mark.asyncio
    async def test_analyze_code_tool(self):
        """Test AnalyzeCodeTool."""
        from src.analysis.tools import AnalyzeCodeTool

        tool = AnalyzeCodeTool()
        assert tool.name == "analyze_code"
        assert tool.requires_confirmation is False

        result = await tool.execute(
            code="def hello(): pass",
            language="python",
        )

        assert result.success is True
        assert "hello" in result.output.lower()

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    @pytest.mark.asyncio
    async def test_analyze_code_tool_no_input(self):
        """Test AnalyzeCodeTool with no input."""
        from src.analysis.tools import AnalyzeCodeTool

        tool = AnalyzeCodeTool()
        result = await tool.execute()

        assert result.success is False
        assert "must be provided" in result.error.lower()

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    @pytest.mark.asyncio
    async def test_find_symbols_tool(self):
        """Test FindSymbolsTool."""
        from src.analysis.tools import FindSymbolsTool

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def hello():\n    pass\n\ndef world():\n    pass\n")
            f.flush()

            tool = FindSymbolsTool()
            result = await tool.execute(
                file_path=f.name,
                pattern="hello",
            )

            assert result.success is True
            assert "hello" in result.output.lower()
            assert result.metadata["count"] == 1

            Path(f.name).unlink()

    @pytest.mark.skipif(
        not is_tree_sitter_available() or "python" not in get_supported_languages(),
        reason="tree-sitter-python not installed"
    )
    @pytest.mark.asyncio
    async def test_get_code_structure_tool(self):
        """Test GetCodeStructureTool."""
        from src.analysis.tools import GetCodeStructureTool
        import json

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("class Foo:\n    def bar(self):\n        pass\n")
            f.flush()

            tool = GetCodeStructureTool()
            result = await tool.execute(file_path=f.name)

            assert result.success is True
            structure = json.loads(result.output)
            assert structure["language"] == "python"
            assert len(structure["classes"]) == 1
            assert structure["classes"][0]["name"] == "Foo"

            Path(f.name).unlink()
