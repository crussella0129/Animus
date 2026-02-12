"""Language-agnostic code parser framework with Python implementation.

Provides an abstract LanguageParser base class that can be extended
to support multiple programming languages.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NodeInfo:
    """A symbol extracted from source code."""

    kind: str  # "module", "class", "function", "method"
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str = ""
    args: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


@dataclass
class EdgeInfo:
    """A relationship between two symbols."""

    source_qname: str
    target_name: str
    kind: str  # "CALLS", "INHERITS", "CONTAINS", "IMPORTS"


@dataclass
class FileParseResult:
    """Result of parsing a single file."""

    file_path: str
    nodes: list[NodeInfo] = field(default_factory=list)
    edges: list[EdgeInfo] = field(default_factory=list)


class LanguageParser(ABC):
    """Abstract base class for language-specific code parsers.

    Implement this interface to add support for new programming languages
    to the knowledge graph system.
    """

    @abstractmethod
    def parse_file(self, path: Path) -> FileParseResult:
        """Parse a source file and extract code structure.

        Args:
            path: Path to the source file

        Returns:
            FileParseResult containing nodes (symbols) and edges (relationships)
        """
        pass

    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return the set of file extensions this parser supports.

        Returns:
            Set of extensions including the dot (e.g., {".py", ".pyi"})
        """
        pass

    def can_parse(self, path: Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            path: Path to check

        Returns:
            True if the file extension is supported
        """
        return path.suffix.lower() in self.supported_extensions()


class PythonParser(LanguageParser):
    """Extract code structure from Python files using the ast module."""

    def supported_extensions(self) -> set[str]:
        """Return supported Python file extensions."""
        return {".py", ".pyi"}

    def parse_file(self, path: Path) -> FileParseResult:
        """Parse a Python file, returning nodes and edges. Syntax errors → empty result."""
        file_str = str(path)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return FileParseResult(file_path=file_str)

        try:
            tree = ast.parse(source, filename=file_str)
        except SyntaxError:
            return FileParseResult(file_path=file_str)

        module_qname = self._module_name_from_path(path)
        result = FileParseResult(file_path=file_str)

        # Module node
        result.nodes.append(NodeInfo(
            kind="module",
            name=module_qname.split(".")[-1],
            qualified_name=module_qname,
            file_path=file_str,
            line_start=1,
            line_end=len(source.splitlines()),
            docstring=ast.get_docstring(tree) or "",
        ))

        self._walk(tree, module_qname, file_str, result)
        return result

    def _module_name_from_path(self, path: Path) -> str:
        """Derive a dotted module name from a file path."""
        parts = list(path.resolve().with_suffix("").parts)

        # Walk backwards to find the last 'src' directory or project root
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "src":
                return ".".join(parts[i:])

        # Fallback: use filename stem
        return path.stem

    def _walk(
        self,
        node: ast.AST,
        parent_qname: str,
        file_str: str,
        result: FileParseResult,
    ) -> None:
        """Recursively walk the AST collecting nodes and edges."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._handle_class(child, parent_qname, file_str, result)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                self._handle_function(child, parent_qname, file_str, result)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                self._handle_import(child, parent_qname, result)

    def _handle_class(
        self,
        node: ast.ClassDef,
        parent_qname: str,
        file_str: str,
        result: FileParseResult,
    ) -> None:
        qname = f"{parent_qname}.{node.name}"

        bases = []
        for base in node.bases:
            bases.append(self._name_from_node(base))

        decorators = [self._name_from_node(d) for d in node.decorator_list]

        result.nodes.append(NodeInfo(
            kind="class",
            name=node.name,
            qualified_name=qname,
            file_path=file_str,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node) or "",
            bases=bases,
            decorators=decorators,
        ))

        # CONTAINS edge from parent
        result.edges.append(EdgeInfo(
            source_qname=parent_qname,
            target_name=qname,
            kind="CONTAINS",
        ))

        # INHERITS edges
        for base_name in bases:
            result.edges.append(EdgeInfo(
                source_qname=qname,
                target_name=base_name,
                kind="INHERITS",
            ))

        # Recurse into class body
        self._walk(node, qname, file_str, result)

    def _handle_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_qname: str,
        file_str: str,
        result: FileParseResult,
    ) -> None:
        qname = f"{parent_qname}.{node.name}"

        # Determine if it's a method (parent is a class)
        is_method = any(
            n.kind == "class" and n.qualified_name == parent_qname
            for n in result.nodes
        )
        kind = "method" if is_method else "function"

        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        decorators = [self._name_from_node(d) for d in node.decorator_list]

        result.nodes.append(NodeInfo(
            kind=kind,
            name=node.name,
            qualified_name=qname,
            file_path=file_str,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node) or "",
            args=args,
            decorators=decorators,
        ))

        # CONTAINS edge from parent
        result.edges.append(EdgeInfo(
            source_qname=parent_qname,
            target_name=qname,
            kind="CONTAINS",
        ))

        # Extract CALLS edges from function body
        self._extract_calls(node, qname, result)

    def _extract_calls(
        self,
        node: ast.AST,
        caller_qname: str,
        result: FileParseResult,
    ) -> None:
        """Walk function body for ast.Call nodes and create CALLS edges."""
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            target = self._name_from_node(child.func)
            if target:
                result.edges.append(EdgeInfo(
                    source_qname=caller_qname,
                    target_name=target,
                    kind="CALLS",
                ))

    def _handle_import(
        self,
        node: ast.Import | ast.ImportFrom,
        parent_qname: str,
        result: FileParseResult,
    ) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.edges.append(EdgeInfo(
                    source_qname=parent_qname,
                    target_name=alias.name,
                    kind="IMPORTS",
                ))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                target = f"{node.module}.{alias.name}"
                result.edges.append(EdgeInfo(
                    source_qname=parent_qname,
                    target_name=target,
                    kind="IMPORTS",
                ))

    def _name_from_node(self, node: ast.expr) -> str:
        """Extract a dotted name from an AST expression node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            value = self._name_from_node(node.value)
            if value:
                return f"{value}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Call):
            return self._name_from_node(node.func)
        return ""


# ---------------------------------------------------------------------------
# Multi-language parser registry
# ---------------------------------------------------------------------------

class ParserRegistry:
    """Registry for managing multiple language parsers.

    Allows automatic parser selection based on file extension.
    """

    def __init__(self) -> None:
        self._parsers: list[LanguageParser] = []

    def register(self, parser: LanguageParser) -> None:
        """Register a new parser."""
        self._parsers.append(parser)

    def get_parser(self, path: Path) -> LanguageParser | None:
        """Get the appropriate parser for a file path.

        Args:
            path: File path to parse

        Returns:
            LanguageParser instance if supported, None otherwise
        """
        for parser in self._parsers:
            if parser.can_parse(path):
                return parser
        return None

    def supported_extensions(self) -> set[str]:
        """Get all supported file extensions across all parsers."""
        extensions = set()
        for parser in self._parsers:
            extensions.update(parser.supported_extensions())
        return extensions


# Default registry with Python support
_default_registry = ParserRegistry()
_default_registry.register(PythonParser())


def get_parser(path: Path) -> LanguageParser | None:
    """Get a parser for the given file path using the default registry.

    Args:
        path: File path to parse

    Returns:
        LanguageParser instance if supported, None otherwise
    """
    return _default_registry.get_parser(path)


# ---------------------------------------------------------------------------
# Stub parsers for future language support
# ---------------------------------------------------------------------------
# These are placeholders demonstrating the pluggable architecture.
# To implement, replace with actual parsing logic (tree-sitter, language AST, etc.)


class GoParser(LanguageParser):
    """Parser for Go source files (stub implementation).

    To implement: Use 'go doc -json' or tree-sitter-go for AST parsing.
    """

    def supported_extensions(self) -> set[str]:
        return {".go"}

    def parse_file(self, path: Path) -> FileParseResult:
        """Parse a Go file (not yet implemented)."""
        # TODO: Implement using tree-sitter or go/ast via subprocess
        return FileParseResult(file_path=str(path))


class RustParser(LanguageParser):
    """Parser for Rust source files (stub implementation).

    To implement: Use tree-sitter-rust or syn crate via PyO3 bindings.
    """

    def supported_extensions(self) -> set[str]:
        return {".rs"}

    def parse_file(self, path: Path) -> FileParseResult:
        """Parse a Rust file (not yet implemented)."""
        # TODO: Implement using tree-sitter-rust
        return FileParseResult(file_path=str(path))


class TypeScriptParser(LanguageParser):
    """Parser for TypeScript/JavaScript files (stub implementation).

    To implement: Use tree-sitter-typescript or TypeScript compiler API.
    """

    def supported_extensions(self) -> set[str]:
        return {".ts", ".tsx", ".js", ".jsx"}

    def parse_file(self, path: Path) -> FileParseResult:
        """Parse a TypeScript/JavaScript file (not yet implemented)."""
        # TODO: Implement using tree-sitter-typescript
        return FileParseResult(file_path=str(path))


# To enable additional language support, instantiate and register parsers:
# _default_registry.register(GoParser())
# _default_registry.register(RustParser())
# _default_registry.register(TypeScriptParser())
