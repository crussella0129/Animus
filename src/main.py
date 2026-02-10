"""Animus CLI entry point using Typer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from src.core.config import AnimusConfig
from src.core.detection import detect_system
from src.ui import console, error, info, print_logo, success, warn


def _startup_callback(ctx: typer.Context) -> None:
    """Show the logo when any command is invoked."""
    print_logo()


app = typer.Typer(
    name="animus",
    help="Local-first AI agent with RAG and tool use.",
    no_args_is_help=True,
    callback=_startup_callback,
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
def pull(
    model_name: str = typer.Argument(None, help="Model name (e.g. llama-3.2-1b) or HuggingFace URL"),
    list_models: bool = typer.Option(False, "--list", "-l", help="List available models"),
) -> None:
    """Pull/download a GGUF model for local inference."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    from src.llm.native import MODEL_CATALOG, download_gguf, list_available_models

    if list_models or model_name is None:
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Repository", style="dim")
        for name in list_available_models():
            repo, filename, params = MODEL_CATALOG[name]
            table.add_row(name, f"~{params}B params", repo)
        console.print(table)
        info("Usage: animus pull <model-name>")
        info("Or: animus pull <huggingface-url-to-gguf>")
        return

    cfg = AnimusConfig.load()
    info(f"Pulling {model_name}...")

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    )

    task_id = None

    def _on_progress(downloaded: int, total: int) -> None:
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task("Downloading", total=total)
            progress.start()
        progress.update(task_id, completed=downloaded)

    try:
        model_path = download_gguf(model_name, cfg.models_dir, on_progress=_on_progress)
    except Exception as e:
        if task_id is not None:
            progress.stop()
        error(f"Failed to pull model: {e}")
        raise typer.Exit(1)

    if task_id is not None:
        progress.stop()

    success(f"Model downloaded: {model_path}")

    # Auto-configure: set model_path and model_name in config
    cfg.model.model_path = str(model_path)
    cfg.model.model_name = model_name
    cfg.model.provider = "native"
    cfg.save()
    success(f"Config updated: provider=native, model_path={model_path}")


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


# --- Session commands ---


@app.command()
def sessions() -> None:
    """List saved conversation sessions."""
    from src.core.session import Session

    cfg = AnimusConfig.load()
    listing = Session.list_sessions(cfg.sessions_dir)
    if not listing:
        info("No saved sessions.")
        return

    table = Table(title="Saved Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Messages", style="yellow")
    table.add_column("Created", style="dim")
    table.add_column("Preview", style="white", max_width=50)
    for s in listing:
        created = time.strftime("%Y-%m-%d %H:%M", time.localtime(s["created"])) if s["created"] else "?"
        table.add_row(s["id"], s["provider"], str(s["messages"]), created, s["preview"])
    console.print(table)


# --- Rise command ---


def _make_confirm_callback(cfg: AnimusConfig):
    """Create a Rich-based confirmation callback for dangerous tool operations."""
    def _confirm(message: str) -> bool:
        if not cfg.agent.confirm_dangerous:
            return True
        try:
            response = console.input(f"[bold yellow]\\[!][/] {message} [dim]\\[y/N][/] ")
            return response.strip().lower() in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
    return _confirm


def _handle_slash_command(command: str, agent, session, cfg: AnimusConfig) -> bool:
    """Handle slash commands. Returns True if command was handled."""
    from src.core.context import estimate_messages_tokens

    cmd = command.strip().lower()

    if cmd == "/help":
        console.print("[bold cyan]Slash Commands:[/]")
        console.print("  /save    — Save current session")
        console.print("  /clear   — Reset conversation history")
        console.print("  /tools   — List available tools")
        console.print("  /tokens  — Show context usage estimate")
        console.print("  /plan    — Force plan-then-execute mode for next message")
        console.print("  /help    — Show this help message")
        return True

    if cmd == "/save":
        path = session.save()
        success(f"Session saved: {path}")
        return True

    if cmd == "/clear":
        agent.reset()
        session.clear_messages()
        success("Conversation history cleared.")
        return True

    if cmd == "/tools":
        tools = agent._tools.list_tools()
        if not tools:
            info("No tools registered.")
        else:
            table = Table(title="Available Tools")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green")
            for tool in tools:
                table.add_row(tool.name, tool.description)
            console.print(table)
        return True

    if cmd == "/tokens":
        token_est = estimate_messages_tokens(agent.messages)
        budget = agent._context_window.compute_budget(agent._system_prompt, agent.messages)
        cumulative = getattr(agent, '_cumulative_tokens', 0)
        console.print(f"[bold cyan]Context Usage:[/]")
        console.print(f"  History tokens: ~{token_est} / {budget.history_tokens} budget")
        console.print(f"  Context window: {budget.total}")
        console.print(f"  Available for output: {budget.available_for_output}")
        if cumulative:
            console.print(f"  Cumulative session tokens: ~{cumulative}")
        return True

    if cmd == "/plan":
        # Toggle plan mode — next message will use plan-then-execute
        _plan_mode_state["active"] = not _plan_mode_state["active"]
        state = "ON" if _plan_mode_state["active"] else "OFF"
        style = "bold green" if _plan_mode_state["active"] else "bold red"
        console.print(f"[{style}]Plan mode: {state}[/]")
        if _plan_mode_state["active"]:
            info("Next message will use plan-then-execute pipeline.")
        return True

    return False


# Mutable state for /plan toggle (module-level to avoid closure issues)
_plan_mode_state: dict[str, bool] = {"active": False}


@app.command()
def rise(
    resume: bool = typer.Option(False, "--resume", help="Resume the most recent session"),
    session_id: Optional[str] = typer.Option(None, "--session", help="Resume a specific session by ID"),
) -> None:
    """Awaken Animus. Start an interactive agent session."""
    from src.core.agent import Agent
    from src.core.session import Session
    from src.llm.factory import ProviderFactory
    from src.tools.base import ToolRegistry
    from src.tools.filesystem import register_filesystem_tools
    from src.tools.shell import register_shell_tools

    cfg = AnimusConfig.load()
    factory = ProviderFactory()
    provider = factory.create(
        cfg.model.provider,
        model_path=cfg.model.model_path,
        model_name=cfg.model.model_name,
        context_length=cfg.model.context_length,
        gpu_layers=cfg.model.gpu_layers,
        size_tier=cfg.model.size_tier,
        max_tokens=cfg.model.max_tokens,
    )

    if provider is None or not provider.available():
        error(f"Provider '{cfg.model.provider}' is not available. Run 'animus status' to check.")
        raise typer.Exit(1)

    # Set up tool registry with confirmation callback
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    confirm_cb = _make_confirm_callback(cfg)
    register_shell_tools(registry, confirm_callback=confirm_cb)

    # Register git tools if available
    try:
        from src.tools.git import register_git_tools
        register_git_tools(registry, confirm_callback=confirm_cb)
    except ImportError:
        pass

    agent = Agent(
        provider=provider,
        tool_registry=registry,
        system_prompt=cfg.agent.system_prompt,
        max_turns=cfg.agent.max_turns,
    )

    # Session handling: resume or create new
    session: Session
    if session_id:
        try:
            session = Session.load_by_id(session_id, cfg.sessions_dir)
            agent._messages = list(session.messages)
            info(f"Resumed session: {session.id}")
        except FileNotFoundError:
            error(f"Session not found: {session_id}")
            raise typer.Exit(1)
    elif resume:
        loaded = Session.load_latest(cfg.sessions_dir)
        if loaded:
            session = loaded
            agent._messages = list(session.messages)
            info(f"Resumed session: {session.id} ({len(session.messages)} messages)")
        else:
            session = Session(
                sessions_dir=cfg.sessions_dir,
                provider=cfg.model.provider,
                model=cfg.model.model_name,
            )
            info("No previous session found. Starting new session.")
    else:
        session = Session(
            sessions_dir=cfg.sessions_dir,
            provider=cfg.model.provider,
            model=cfg.model.model_name,
        )

    # Show session info
    info(f"Provider: [bold]{cfg.model.provider}[/]  Model: [bold]{cfg.model.model_name}[/]")
    info(f"Session: {session.id}")
    info("Type 'exit' or 'quit' to end. Type '/help' for commands.")
    console.print()

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]You>[/] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            stripped = user_input.strip()
            if stripped.lower() in ("exit", "quit", "q"):
                break

            if not stripped:
                continue

            # Handle slash commands
            if stripped.startswith("/"):
                if _handle_slash_command(stripped, agent, session, cfg):
                    console.print()
                    continue
                else:
                    warn(f"Unknown command: {stripped}. Type /help for available commands.")
                    console.print()
                    continue

            # Determine execution mode
            from src.core.planner import should_use_planner

            use_plan = _plan_mode_state["active"] or should_use_planner(provider)

            if use_plan:
                # Plan-then-execute mode
                session.add_message("user", stripped)

                def _on_progress(step_num: int, total: int, desc: str) -> None:
                    console.print(f"[bold cyan]  [{step_num}/{total}][/] {desc}")

                def _on_step_output(text: str) -> None:
                    console.print(f"[green]  > {text[:200]}[/]")

                console.print("[bold yellow]Planning...[/]")
                response = agent.run_planned(
                    stripped,
                    on_progress=_on_progress,
                    on_step_output=_on_step_output,
                    force=_plan_mode_state["active"],
                )
                console.print(f"[bold green]Animus>[/] {response}")

                # Reset plan mode toggle after use
                if _plan_mode_state["active"]:
                    _plan_mode_state["active"] = False
            else:
                # Direct streaming mode (large models)
                session.add_message("user", stripped)
                collected: list[str] = []
                first_chunk = True

                def _on_chunk(chunk: str) -> None:
                    nonlocal first_chunk
                    if first_chunk:
                        console.print("[bold green]Animus>[/] ", end="")
                        first_chunk = False
                    console.print(chunk, end="", highlight=False)
                    collected.append(chunk)

                response = agent.run_stream(stripped, on_chunk=_on_chunk)
                if not collected:
                    console.print(f"[bold green]Animus>[/] {response}")
                else:
                    console.print()  # newline after streamed output

            session.add_message("assistant", response)
            console.print()

    finally:
        # Auto-save on exit
        session.messages = list(agent.messages)
        session.save()
        info(f"Session saved: {session.id}")
        info("Session ended.")


if __name__ == "__main__":
    app()
