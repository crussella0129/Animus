"""Tree-sitter based code parser for AST analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Any

# Global state for lazy loading
_TREE_SITTER_AVAILABLE: Optional[bool] = None
_LANGUAGES: dict[str, Any] = {}


def _check_tree_sitter() -> bool:
    """Check if tree-sitter is available and cache the result."""
    global _TREE_SITTER_AVAILABLE

    if _TREE_SITTER_AVAILABLE is not None:
        return _TREE_SITTER_AVAILABLE

    try:
        import tree_sitter  # noqa: F401
        _TREE_SITTER_AVAILABLE = True
    except ImportError:
        _TREE_SITTER_AVAILABLE = False

    return _TREE_SITTER_AVAILABLE


def is_tree_sitter_available() -> bool:
    """Check if tree-sitter is available."""
    return _check_tree_sitter()


def get_supported_languages() -> list[str]:
    """Get list of supported languages."""
    if not _check_tree_sitter():
        return []

    supported = []
    lang_modules = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "rust": "tree_sitter_rust",
        "go": "tree_sitter_go",
        "java": "tree_sitter_java",
        "c": "tree_sitter_c",
        "cpp": "tree_sitter_cpp",
    }

    for lang, module in lang_modules.items():
        try:
            __import__(module)
            supported.append(lang)
        except ImportError:
            pass

    return supported


def _get_language(lang_name: str) -> Optional[Any]:
    """Get a tree-sitter language, caching the result."""
    global _LANGUAGES

    if lang_name in _LANGUAGES:
        return _LANGUAGES[lang_name]

    if not _check_tree_sitter():
        return None

    from tree_sitter import Language

    lang_modules = {
        "python": "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "tsx": "tree_sitter_typescript",
        "rust": "tree_sitter_rust",
        "go": "tree_sitter_go",
        "java": "tree_sitter_java",
        "c": "tree_sitter_c",
        "cpp": "tree_sitter_cpp",
    }

    module_name = lang_modules.get(lang_name)
    if not module_name:
        return None

    try:
        module = __import__(module_name)
        # For typescript, check if tsx is requested
        if lang_name == "tsx":
            lang = Language(module.language_tsx())
        elif lang_name == "typescript":
            lang = Language(module.language_typescript())
        else:
            lang = Language(module.language())
        _LANGUAGES[lang_name] = lang
        return lang
    except (ImportError, AttributeError):
        return None


class SymbolType(str, Enum):
    """Types of code symbols."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"
    TYPE = "type"
    MODULE = "module"
    PARAMETER = "parameter"
    DECORATOR = "decorator"


@dataclass
class CodeSymbol:
    """A symbol found in code."""
    name: str
    type: SymbolType
    line: int
    column: int
    end_line: int
    end_column: int
    parent: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "parent": self.parent,
            "docstring": self.docstring,
            "signature": self.signature,
            "metadata": self.metadata,
        }


@dataclass
class ParsedCode:
    """Result of parsing source code."""
    language: str
    source: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    tree: Optional[Any] = None  # tree_sitter.Tree

    def get_functions(self) -> list[CodeSymbol]:
        """Get all functions."""
        return [s for s in self.symbols if s.type == SymbolType.FUNCTION]

    def get_classes(self) -> list[CodeSymbol]:
        """Get all classes."""
        return [s for s in self.symbols if s.type == SymbolType.CLASS]

    def get_methods(self) -> list[CodeSymbol]:
        """Get all methods."""
        return [s for s in self.symbols if s.type == SymbolType.METHOD]

    def find_symbol(self, name: str) -> Optional[CodeSymbol]:
        """Find a symbol by name."""
        for s in self.symbols:
            if s.name == name:
                return s
        return None

    def get_structure(self) -> dict:
        """Get a structured representation of the code."""
        return {
            "language": self.language,
            "imports": self.imports,
            "classes": [
                {
                    "name": c.name,
                    "line": c.line,
                    "methods": [
                        {"name": m.name, "line": m.line}
                        for m in self.symbols
                        if m.type == SymbolType.METHOD and m.parent == c.name
                    ],
                }
                for c in self.get_classes()
            ],
            "functions": [
                {"name": f.name, "line": f.line, "signature": f.signature}
                for f in self.get_functions()
            ],
            "errors": self.errors,
        }


# Language-specific symbol extraction queries
PYTHON_QUERIES = """
(function_definition name: (identifier) @function.name) @function
(class_definition name: (identifier) @class.name) @class
(import_statement) @import
(import_from_statement) @import
(decorated_definition) @decorated
"""

JAVASCRIPT_QUERIES = """
(function_declaration name: (identifier) @function.name) @function
(class_declaration name: (identifier) @class.name) @class
(method_definition name: (property_identifier) @method.name) @method
(import_statement) @import
(arrow_function) @arrow
"""


class CodeParser:
    """Parser for extracting code structure using tree-sitter."""

    # Map file extensions to languages
    EXTENSION_MAP = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
    }

    def __init__(self):
        """Initialize the code parser."""
        self._parsers: dict[str, Any] = {}

    def _get_parser(self, language: str) -> Optional[Any]:
        """Get a parser for the given language."""
        if language in self._parsers:
            return self._parsers[language]

        if not _check_tree_sitter():
            return None

        from tree_sitter import Parser

        lang = _get_language(language)
        if not lang:
            return None

        parser = Parser(lang)
        self._parsers[language] = parser
        return parser

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        return self.EXTENSION_MAP.get(ext)

    def parse(
        self,
        source: str,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> ParsedCode:
        """Parse source code and extract symbols.

        Args:
            source: Source code to parse.
            language: Language name (python, javascript, etc.).
            file_path: Optional file path for language detection.

        Returns:
            ParsedCode with extracted symbols.
        """
        # Detect language if not provided
        if not language and file_path:
            language = self.detect_language(file_path)

        if not language:
            return ParsedCode(
                language="unknown",
                source=source,
                errors=["Could not detect language"],
            )

        if not _check_tree_sitter():
            return ParsedCode(
                language=language,
                source=source,
                errors=["tree-sitter not available. Install with: pip install animus[analysis]"],
            )

        parser = self._get_parser(language)
        if not parser:
            return ParsedCode(
                language=language,
                source=source,
                errors=[f"Language not supported: {language}"],
            )

        try:
            tree = parser.parse(source.encode())

            # Extract symbols based on language
            if language == "python":
                symbols, imports = self._extract_python_symbols(tree, source)
            elif language in ("javascript", "typescript", "tsx"):
                symbols, imports = self._extract_js_symbols(tree, source)
            else:
                symbols, imports = self._extract_generic_symbols(tree, source, language)

            # Collect syntax errors
            errors = []
            if tree.root_node.has_error:
                errors.append("Source contains syntax errors")

            return ParsedCode(
                language=language,
                source=source,
                symbols=symbols,
                imports=imports,
                errors=errors,
                tree=tree,
            )
        except Exception as e:
            return ParsedCode(
                language=language,
                source=source,
                errors=[f"Parse error: {e}"],
            )

    def _extract_python_symbols(
        self,
        tree: Any,
        source: str,
    ) -> tuple[list[CodeSymbol], list[str]]:
        """Extract symbols from Python code."""
        symbols = []
        imports = []
        lines = source.split("\n")

        def get_text(node: Any) -> str:
            return source[node.start_byte:node.end_byte]

        def get_docstring(node: Any) -> Optional[str]:
            """Extract docstring from function/class body."""
            body = None
            for child in node.children:
                if child.type == "block":
                    body = child
                    break

            if body and body.children:
                first_stmt = body.children[0]
                if first_stmt.type == "expression_statement":
                    expr = first_stmt.children[0] if first_stmt.children else None
                    if expr and expr.type == "string":
                        return get_text(expr).strip('"""').strip("'''").strip()
            return None

        def get_signature(node: Any) -> str:
            """Get function signature."""
            for child in node.children:
                if child.type == "parameters":
                    return f"({get_text(child)[1:-1]})"
            return "()"

        def visit(node: Any, parent_class: Optional[str] = None) -> None:
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = get_text(name_node)
                    sym_type = SymbolType.METHOD if parent_class else SymbolType.FUNCTION
                    symbols.append(CodeSymbol(
                        name=name,
                        type=sym_type,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                        parent=parent_class,
                        docstring=get_docstring(node),
                        signature=get_signature(node),
                    ))

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = get_text(name_node)
                    symbols.append(CodeSymbol(
                        name=name,
                        type=SymbolType.CLASS,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                        docstring=get_docstring(node),
                    ))
                    # Visit class body with class name as parent
                    for child in node.children:
                        visit(child, name)
                    return  # Don't recurse again

            elif node.type in ("import_statement", "import_from_statement"):
                imports.append(get_text(node))

            # Recurse
            for child in node.children:
                visit(child, parent_class)

        visit(tree.root_node)
        return symbols, imports

    def _extract_js_symbols(
        self,
        tree: Any,
        source: str,
    ) -> tuple[list[CodeSymbol], list[str]]:
        """Extract symbols from JavaScript/TypeScript code."""
        symbols = []
        imports = []

        def get_text(node: Any) -> str:
            return source[node.start_byte:node.end_byte]

        def visit(node: Any, parent_class: Optional[str] = None) -> None:
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(CodeSymbol(
                        name=get_text(name_node),
                        type=SymbolType.FUNCTION,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                    ))

            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = get_text(name_node)
                    symbols.append(CodeSymbol(
                        name=name,
                        type=SymbolType.CLASS,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                    ))
                    for child in node.children:
                        visit(child, name)
                    return

            elif node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(CodeSymbol(
                        name=get_text(name_node),
                        type=SymbolType.METHOD,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                        parent=parent_class,
                    ))

            elif node.type == "import_statement":
                imports.append(get_text(node))

            elif node.type == "interface_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(CodeSymbol(
                        name=get_text(name_node),
                        type=SymbolType.INTERFACE,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                    ))

            for child in node.children:
                visit(child, parent_class)

        visit(tree.root_node)
        return symbols, imports

    def _extract_generic_symbols(
        self,
        tree: Any,
        source: str,
        language: str,
    ) -> tuple[list[CodeSymbol], list[str]]:
        """Extract symbols from other languages with generic patterns."""
        symbols = []
        imports = []

        def get_text(node: Any) -> str:
            return source[node.start_byte:node.end_byte]

        # Language-specific node types
        func_types = {
            "rust": ["function_item"],
            "go": ["function_declaration", "method_declaration"],
            "java": ["method_declaration", "constructor_declaration"],
            "c": ["function_definition"],
            "cpp": ["function_definition"],
        }

        class_types = {
            "rust": ["struct_item", "impl_item", "enum_item"],
            "go": ["type_declaration"],
            "java": ["class_declaration", "interface_declaration"],
            "c": ["struct_specifier"],
            "cpp": ["class_specifier", "struct_specifier"],
        }

        def visit(node: Any, parent: Optional[str] = None) -> None:
            # Check for function-like nodes
            if node.type in func_types.get(language, []):
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append(CodeSymbol(
                        name=get_text(name_node),
                        type=SymbolType.FUNCTION if not parent else SymbolType.METHOD,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                        parent=parent,
                    ))

            # Check for class-like nodes
            elif node.type in class_types.get(language, []):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = get_text(name_node)
                    symbols.append(CodeSymbol(
                        name=name,
                        type=SymbolType.CLASS,
                        line=node.start_point[0] + 1,
                        column=node.start_point[1],
                        end_line=node.end_point[0] + 1,
                        end_column=node.end_point[1],
                    ))
                    for child in node.children:
                        visit(child, name)
                    return

            for child in node.children:
                visit(child, parent)

        visit(tree.root_node)
        return symbols, imports

    def parse_file(self, file_path: str) -> ParsedCode:
        """Parse a file and extract symbols.

        Args:
            file_path: Path to the file.

        Returns:
            ParsedCode with extracted symbols.
        """
        path = Path(file_path)
        if not path.exists():
            return ParsedCode(
                language="unknown",
                source="",
                errors=[f"File not found: {file_path}"],
            )

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            return ParsedCode(
                language="unknown",
                source="",
                errors=[f"Could not read file: {e}"],
            )

        return self.parse(source, file_path=file_path)
