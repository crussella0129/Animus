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


def create_analysis_tools() -> list[Tool]:
    """Create all analysis tools."""
    return [
        AnalyzeCodeTool(),
        FindSymbolsTool(),
        GetCodeStructureTool(),
    ]
