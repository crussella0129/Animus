"""Animus CLI entry point using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from src.core.config import AnimusConfig
from src.core.detection import detect_system
from src.ui import console, error, info, success

app = typer.Typer(
    name="animus",
    help="Local-first AI agent with RAG and tool use.",
    no_args_is_help=True,
)


@app.command()
def detect() -> None:
    """Detect system hardware and capabilities."""
    sys_info = detect_system()
    table = Table(title="System Detection")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("OS", f"{sys_info.os_name} {sys_info.os_version}")
    if sys_info.is_windows_11:
        table.add_row("Windows 11", "Yes (build >= 22000)")
    table.add_row("Architecture", sys_info.architecture)
    table.add_row("CPU Cores", str(sys_info.cpu_count))
    table.add_row("Hardware Type", sys_info.hardware_type)
    table.add_row("Python", sys_info.python_version)
    table.add_row("GPU", sys_info.gpu.name)
    table.add_row("GPU Memory", f"{sys_info.gpu.memory_mb} MB")
    table.add_row("CUDA Available", str(sys_info.gpu.cuda_available))
    if sys_info.gpu.driver_version:
        table.add_row("Driver Version", sys_info.gpu.driver_version)
    console.print(table)


@app.command()
def init() -> None:
    """Initialize Animus configuration directory."""
    config = AnimusConfig()
    config.save()
    success(f"Configuration initialized at {config.config_dir}")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    path: bool = typer.Option(False, "--path", "-p", help="Show config file path"),
) -> None:
    """View or manage configuration."""
    cfg = AnimusConfig.load()
    if path:
        info(str(cfg.config_file))
        return
    if show:
        table = Table(title="Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        data = cfg.model_dump(exclude={"config_dir"})
        _flat_table(table, data)
        console.print(table)
        return
    info(f"Config dir: {cfg.config_dir}")
    info(f"Config file exists: {cfg.config_file.exists()}")


def _flat_table(table: Table, data: dict, prefix: str = "") -> None:
    """Flatten nested dict into table rows."""
    for key, value in data.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            _flat_table(table, value, full_key)
        else:
            table.add_row(full_key, str(value))


# --- Model commands ---


@app.command()
def models() -> None:
    """List available model providers and their status."""
    from src.llm.factory import ProviderFactory

    factory = ProviderFactory()
    table = Table(title="Model Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Available", style="green")
    for name in factory.provider_names():
        provider = factory.create(name)
        available = provider.available() if provider else False
        status = "[green]Yes[/]" if available else "[red]No[/]"
        table.add_row(name, status)
    console.print(table)


@app.command()
def status() -> None:
    """Show overall Animus system status."""
    cfg = AnimusConfig.load()
    sys_info = detect_system()
    table = Table(title="Animus Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_row("Config", "[green]OK[/]" if cfg.config_file.exists() else "[yellow]Not initialized[/]")
    table.add_row("GPU", f"[green]{sys_info.gpu.name}[/]" if sys_info.gpu.cuda_available else "[yellow]None[/]")
    table.add_row("Provider", cfg.model.provider)
    table.add_row("Model", cfg.model.model_name)
    console.print(table)


@app.command()
def pull(model_name: str = typer.Argument(..., help="Model name to pull (e.g. llama3.2)")) -> None:
    """Pull a model via the configured provider."""
    from src.llm.factory import ProviderFactory

    cfg = AnimusConfig.load()
    factory = ProviderFactory()
    provider = factory.create(cfg.model.provider)
    if provider is None:
        error(f"Provider '{cfg.model.provider}' not available")
        raise typer.Exit(1)
    info(f"Pulling {model_name} via {cfg.model.provider}...")
    try:
        provider.pull(model_name)
        success(f"Model {model_name} pulled successfully")
    except Exception as e:
        error(f"Failed to pull model: {e}")
        raise typer.Exit(1)


# --- RAG commands ---


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Directory or file to ingest"),
    glob: str = typer.Option("**/*", "--glob", "-g", help="Glob pattern for file matching"),
) -> None:
    """Ingest files into the vector store for RAG."""
    from src.memory.chunker import Chunker
    from src.memory.embedder import MockEmbedder
    from src.memory.scanner import Scanner
    from src.memory.vectorstore import InMemoryVectorStore

    cfg = AnimusConfig.load()
    scanner = Scanner()
    chunker = Chunker(chunk_size=cfg.rag.chunk_size, overlap=cfg.rag.chunk_overlap)
    embedder = MockEmbedder()
    store = InMemoryVectorStore()

    target = Path(path)
    if not target.exists():
        error(f"Path does not exist: {path}")
        raise typer.Exit(1)

    files = list(scanner.scan(target))
    info(f"Found {len(files)} files")

    total_chunks = 0
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunker.chunk(text, metadata={"source": str(file_path)})
            if chunks:
                texts = [c["text"] for c in chunks]
                embeddings = embedder.embed(texts)
                store.add(texts, embeddings, [c.get("metadata", {}) for c in chunks])
                total_chunks += len(chunks)
        except Exception as e:
            error(f"Failed to process {file_path}: {e}")

    success(f"Ingested {total_chunks} chunks from {len(files)} files")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """Search the vector store."""
    info(f"Search for: {query} (top_k={top_k})")
    info("Note: In-memory store does not persist between runs. Use within a session.")


# --- Chat command ---


@app.command()
def chat() -> None:
    """Start an interactive chat session with the agent."""
    from src.core.agent import Agent
    from src.llm.factory import ProviderFactory
    from src.tools.base import ToolRegistry
    from src.tools.filesystem import register_filesystem_tools
    from src.tools.shell import register_shell_tools

    cfg = AnimusConfig.load()
    factory = ProviderFactory()
    provider = factory.create(cfg.model.provider)

    if provider is None or not provider.available():
        error(f"Provider '{cfg.model.provider}' is not available. Run 'animus status' to check.")
        raise typer.Exit(1)

    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_shell_tools(registry)

    agent = Agent(
        provider=provider,
        tool_registry=registry,
        system_prompt=cfg.agent.system_prompt,
        max_turns=cfg.agent.max_turns,
    )

    info("Animus chat session started. Type 'exit' or 'quit' to end.")
    console.print()

    while True:
        try:
            user_input = console.input("[bold cyan]You>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            info("Session ended.")
            break

        if user_input.strip().lower() in ("exit", "quit", "q"):
            info("Session ended.")
            break

        if not user_input.strip():
            continue

        response = agent.run(user_input)
        console.print(f"[bold green]Animus>[/] {response}")
        console.print()


if __name__ == "__main__":
    app()
