"""Session management helpers for Animus CLI."""

from __future__ import annotations

from src.core.config import AnimusConfig
from src.ui import console, info


def make_confirm_callback(cfg: AnimusConfig):
    """Create a Rich-based confirmation callback for dangerous tool operations."""
    def _confirm(message: str) -> bool:
        if not cfg.agent.confirm_dangerous:
            return True
        try:
            # IMPORTANT: No timeout on user input!
            # Users need time to research commands, read docs, or make informed decisions.
            # See docs/DESIGN_PRINCIPLES.md - Rule #1: Never Timeout User Input
            response = console.input(f"[bold yellow]\\[!][/] {message} [dim]\\[y/N][/] ")
            return response.strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
    return _confirm


def build_tool_registry(cfg: AnimusConfig, session_cwd, confirm_cb):
    """Build and return a fully-configured ToolRegistry for a session.

    Args:
        cfg: Loaded AnimusConfig.
        session_cwd: Workspace instance for CWD/boundary tracking.
        confirm_cb: Confirmation callback for dangerous operations.

    Returns:
        Configured ToolRegistry instance.
    """
    from src.tools.base import ToolRegistry
    from src.tools.filesystem import register_filesystem_tools
    from src.tools.shell import register_shell_tools
    from src.isolation.ornstein import create_sandbox as _create_sandbox

    registry = ToolRegistry()
    register_filesystem_tools(registry, session_cwd=session_cwd)

    shell_sandbox = (
        _create_sandbox(memory_mb=512, timeout_seconds=30)
        if cfg.isolation.ornstein_enabled
        else None
    )
    register_shell_tools(registry, confirm_callback=confirm_cb, session_cwd=session_cwd, sandbox=shell_sandbox)

    # Register git tools if available
    try:
        from src.tools.git import register_git_tools
        register_git_tools(registry, confirm_callback=confirm_cb, session_cwd=session_cwd)
    except ImportError:
        pass

    # Register graph tools if the knowledge graph exists
    graph_db_path = cfg.graph_dir / "code_graph.db"
    graph_db = None
    if graph_db_path.exists():
        try:
            from src.knowledge.graph_db import GraphDB
            from src.tools.graph import register_graph_tools
            graph_db = GraphDB(graph_db_path)
            register_graph_tools(registry, graph_db)
        except Exception:
            pass

    # Register search tools if the vector store exists
    vector_db_path = cfg.vector_dir / "vectors.db"
    vector_store = None
    search_embedder = None
    if vector_db_path.exists():
        try:
            from src.memory.embedder import NativeEmbedder
            from src.memory.vectorstore import SQLiteVectorStore
            from src.tools.search import register_search_tools
            vector_store = SQLiteVectorStore(vector_db_path)
            search_embedder = NativeEmbedder()  # Real embeddings for semantic search
            register_search_tools(registry, vector_store, search_embedder)
        except Exception:
            pass

    # Register Manifold unified search (if any backend is available)
    if vector_store or graph_db:
        try:
            from pathlib import Path as PathLib
            from src.retrieval.executor import RetrievalExecutor
            from src.tools.manifold_search import register_manifold_search

            executor = RetrievalExecutor(
                vector_store=vector_store,
                embedder=search_embedder,
                graph_db=graph_db,
                project_root=PathLib.cwd(),  # Use current working directory
            )
            register_manifold_search(registry, executor)
            info("[Manifold] Unified search tool registered")
        except Exception:
            # Manifold optional - degrade gracefully
            pass

    return registry
