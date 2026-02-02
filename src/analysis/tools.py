"""Code analysis tools using Tree-sitter."""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

from src.tools.base import Tool, ToolParameter, ToolResult, ToolCategory
from src.analysis.parser import (
    CodeParser,
    ParsedCode,
    SymbolType,
    is_tree_sitter_available,
    get_supported_languages,
)


class AnalyzeCodeTool(Tool):
    """Tool to analyze code structure using AST parsing."""

    def __init__(self):
        self._parser = CodeParser()

    @property
    def name(self) -> str:
        return "analyze_code"

    @property
    def description(self) -> str:
        langs = get_supported_languages() if is_tree_sitter_available() else []
        lang_str = ", ".join(langs) if langs else "Python, JavaScript, TypeScript, etc."
        return (
            f"Analyze code structure using AST parsing. "
            f"Extracts functions, classes, methods, and imports. "
            f"Supported languages: {lang_str}."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the source file to analyze.",
                required=False,
            ),
            ToolParameter(
                name="code",
                type="string",
                description="Source code to analyze (if file_path not provided).",
                required=False,
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language (auto-detected if file_path provided).",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ANALYSIS

    @property
    def requires_confirmation(self) -> bool:
        return False

    async def execute(
        self,
        file_path: Optional[str] = None,
        code: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Analyze code structure."""
        if not file_path and not code:
            return ToolResult(
                success=False,
                output="",
                error="Either file_path or code must be provided.",
            )

        if file_path:
            parsed = self._parser.parse_file(file_path)
        else:
            parsed = self._parser.parse(code or "", language=language)

        if parsed.errors and not parsed.symbols:
            return ToolResult(
                success=False,
                output="",
                error="\n".join(parsed.errors),
            )

        # Build output
        output_parts = [f"Language: {parsed.language}"]

        if parsed.imports:
            output_parts.append(f"\nImports ({len(parsed.imports)}):")
            for imp in parsed.imports[:10]:  # Limit to first 10
                output_parts.append(f"  {imp}")
            if len(parsed.imports) > 10:
                output_parts.append(f"  ... and {len(parsed.imports) - 10} more")

        classes = parsed.get_classes()
        if classes:
            output_parts.append(f"\nClasses ({len(classes)}):")
            for cls in classes:
                methods = [s for s in parsed.symbols if s.parent == cls.name]
                output_parts.append(f"  {cls.name} (line {cls.line}, {len(methods)} methods)")
                for method in methods[:5]:
                    output_parts.append(f"    - {method.name}{method.signature or '()'}")
                if len(methods) > 5:
                    output_parts.append(f"    ... and {len(methods) - 5} more methods")

        functions = parsed.get_functions()
        if functions:
            output_parts.append(f"\nFunctions ({len(functions)}):")
            for func in functions[:15]:
                sig = func.signature or "()"
                output_parts.append(f"  {func.name}{sig} (line {func.line})")
            if len(functions) > 15:
                output_parts.append(f"  ... and {len(functions) - 15} more")

        if parsed.errors:
            output_parts.append(f"\nWarnings: {', '.join(parsed.errors)}")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata={
                "language": parsed.language,
                "class_count": len(classes),
                "function_count": len(functions),
                "import_count": len(parsed.imports),
            },
        )


class FindSymbolsTool(Tool):
    """Tool to find specific symbols in code."""

    def __init__(self):
        self._parser = CodeParser()

    @property
    def name(self) -> str:
        return "find_symbols"

    @property
    def description(self) -> str:
        return (
            "Find specific symbols (functions, classes, methods) in code. "
            "Search by name pattern or symbol type."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the source file to search.",
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="Name pattern to search for (case-insensitive substring match).",
                required=False,
            ),
            ToolParameter(
                name="symbol_type",
                type="string",
                description="Type of symbol: function, class, method, import.",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ANALYSIS

    @property
    def requires_confirmation(self) -> bool:
        return False

    async def execute(
        self,
        file_path: str,
        pattern: Optional[str] = None,
        symbol_type: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Find symbols in code."""
        parsed = self._parser.parse_file(file_path)

        if parsed.errors and not parsed.symbols:
            return ToolResult(
                success=False,
                output="",
                error="\n".join(parsed.errors),
            )

        # Filter symbols
        results = parsed.symbols

        if symbol_type:
            try:
                sym_type = SymbolType(symbol_type.lower())
                results = [s for s in results if s.type == sym_type]
            except ValueError:
                valid_types = [t.value for t in SymbolType]
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Invalid symbol_type. Valid types: {', '.join(valid_types)}",
                )

        if pattern:
            pattern_lower = pattern.lower()
            results = [s for s in results if pattern_lower in s.name.lower()]

        if not results:
            return ToolResult(
                success=True,
                output="No symbols found matching criteria.",
                metadata={"count": 0},
            )

        # Build output
        output_parts = [f"Found {len(results)} symbol(s):"]
        for sym in results:
            parent_info = f" in {sym.parent}" if sym.parent else ""
            sig = sym.signature or ""
            output_parts.append(
                f"  {sym.type.value}: {sym.name}{sig} (line {sym.line}){parent_info}"
            )

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata={"count": len(results)},
        )


class GetCodeStructureTool(Tool):
    """Tool to get a structured overview of code."""

    def __init__(self):
        self._parser = CodeParser()

    @property
    def name(self) -> str:
        return "get_code_structure"

    @property
    def description(self) -> str:
        return (
            "Get a structured JSON overview of code including "
            "classes, methods, functions, and imports."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the source file.",
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ANALYSIS

    @property
    def requires_confirmation(self) -> bool:
        return False

    async def execute(
        self,
        file_path: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Get code structure as JSON."""
        import json

        parsed = self._parser.parse_file(file_path)

        if parsed.errors and not parsed.symbols:
            return ToolResult(
                success=False,
                output="",
                error="\n".join(parsed.errors),
            )

        structure = parsed.get_structure()

        return ToolResult(
            success=True,
            output=json.dumps(structure, indent=2),
            metadata=structure,
        )


class IndexCodebaseTool(Tool):
    """Tool to index a codebase for fast symbol search."""

    def __init__(self):
        self._indexer = None
        self._index = None

    def _get_indexer(self):
        """Get indexer, creating if needed."""
        if self._indexer is None:
            from src.analysis.indexer import CodebaseIndexer
            self._indexer = CodebaseIndexer()
        return self._indexer

    @property
    def name(self) -> str:
        return "index_codebase"

    @property
    def description(self) -> str:
        return (
            "Index a codebase directory for fast symbol search. "
            "Scans all supported source files and extracts symbols."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="directory",
                type="string",
                description="Path to the directory to index.",
            ),
            ToolParameter(
                name="save_path",
                type="string",
                description="Path to save the index file (optional).",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ANALYSIS

    @property
    def requires_confirmation(self) -> bool:
        return False

    async def execute(
        self,
        directory: str,
        save_path: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Index a codebase."""
        from pathlib import Path

        if not Path(directory).exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Directory not found: {directory}",
            )

        indexer = self._get_indexer()

        try:
            self._index = indexer.index_directory(directory)

            if save_path:
                indexer.save_index(self._index, save_path)

            stats = self._index.get_statistics()
            output = (
                f"Indexed {stats['total_files']} files\n"
                f"Found {stats['total_symbols']} symbols:\n"
                f"  - {stats['total_classes']} classes\n"
                f"  - {stats['total_functions']} functions/methods\n"
                f"Languages: {', '.join(stats['languages'])}"
            )
            if save_path:
                output += f"\nIndex saved to: {save_path}"

            return ToolResult(
                success=True,
                output=output,
                metadata=stats,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Indexing failed: {e}",
            )


class SearchCodebaseTool(Tool):
    """Tool to search an indexed codebase."""

    def __init__(self):
        self._index = None

    @property
    def name(self) -> str:
        return "search_codebase"

    @property
    def description(self) -> str:
        return (
            "Search for symbols in an indexed codebase. "
            "Requires prior indexing with index_codebase or loading an index."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query (symbol name or pattern).",
            ),
            ToolParameter(
                name="symbol_type",
                type="string",
                description="Filter by type: function, class, method.",
                required=False,
            ),
            ToolParameter(
                name="index_path",
                type="string",
                description="Path to load index from (if not already loaded).",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum results to return (default: 20).",
                required=False,
            ),
        ]

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ANALYSIS

    @property
    def requires_confirmation(self) -> bool:
        return False

    async def execute(
        self,
        query: str,
        symbol_type: Optional[str] = None,
        index_path: Optional[str] = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> ToolResult:
        """Search the codebase."""
        from src.analysis.indexer import CodebaseIndexer

        # Load index if path provided
        if index_path:
            try:
                indexer = CodebaseIndexer()
                self._index = indexer.load_index(index_path)
            except Exception as e:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Failed to load index: {e}",
                )

        if self._index is None:
            return ToolResult(
                success=False,
                output="",
                error="No index loaded. Use index_codebase first or provide index_path.",
            )

        # Parse symbol type
        sym_type = None
        if symbol_type:
            try:
                sym_type = SymbolType(symbol_type.lower())
            except ValueError:
                pass

        # Search
        results = self._index.search_symbols(query, symbol_type=sym_type, limit=limit)

        if not results:
            return ToolResult(
                success=True,
                output=f"No results found for '{query}'",
                metadata={"count": 0},
            )

        # Format output
        output_parts = [f"Found {len(results)} result(s) for '{query}':"]
        for r in results:
            sym = r.symbol
            if sym:
                output_parts.append(
                    f"  {sym.type.value}: {sym.name} - {r.file_path}:{r.line}"
                )
            else:
                output_parts.append(f"  {r.file_path}:{r.line}")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            metadata={"count": len(results)},
        )


def create_analysis_tools() -> list[Tool]:
    """Create all analysis tools."""
    return [
        AnalyzeCodeTool(),
        FindSymbolsTool(),
        GetCodeStructureTool(),
        IndexCodebaseTool(),
        SearchCodebaseTool(),
    ]
