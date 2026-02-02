"""Code analysis module using Tree-sitter for AST-based parsing."""

from src.analysis.parser import (
    CodeParser,
    ParsedCode,
    CodeSymbol,
    SymbolType,
    is_tree_sitter_available,
    get_supported_languages,
)
from src.analysis.indexer import (
    CodebaseIndexer,
    CodebaseIndex,
    IndexedFile,
    SearchResult,
)
from src.analysis.tools import (
    AnalyzeCodeTool,
    FindSymbolsTool,
    GetCodeStructureTool,
)

__all__ = [
    # Parser
    "CodeParser",
    "ParsedCode",
    "CodeSymbol",
    "SymbolType",
    "is_tree_sitter_available",
    "get_supported_languages",
    # Indexer
    "CodebaseIndexer",
    "CodebaseIndex",
    "IndexedFile",
    "SearchResult",
    # Tools
    "AnalyzeCodeTool",
    "FindSymbolsTool",
    "GetCodeStructureTool",
]
