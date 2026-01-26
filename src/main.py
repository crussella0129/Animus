"""Animus CLI - Main entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import __version__
from src.core.config import ConfigManager
from src.core.detection import detect_environment, HardwareType, OperatingSystem

# Initialize Typer app
app = typer.Typer(
    name="animus",
    help="A high-performance, cross-platform CLI coding agent.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]Animus[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Animus - A high-performance, cross-platform CLI coding agent."""
    pass


@app.command()
def detect(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show detailed information.",
    ),
) -> None:
    """Detect system environment (OS, hardware, GPU)."""
    info = detect_environment()

    if json_output:
        import json
        output = {
            "os": info.os.value,
            "os_version": info.os_version,
            "architecture": info.architecture.value,
            "hardware_type": info.hardware_type.value,
            "python_version": info.python_version,
            "hostname": info.hostname,
            "cpu_count": info.cpu_count,
            "is_wsl": info.is_wsl,
            "gpu": None,
        }
        if info.gpu:
            output["gpu"] = {
                "name": info.gpu.name,
                "vendor": info.gpu.vendor,
                "memory_mb": info.gpu.memory_mb,
                "cuda_available": info.gpu.cuda_available,
                "cuda_version": info.gpu.cuda_version,
            }
        console.print_json(json.dumps(output, indent=2))
        return

    # Create rich table output
    table = Table(title="System Environment", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    # OS info
    os_emoji = {
        OperatingSystem.WINDOWS: "Windows",
        OperatingSystem.MACOS: "macOS",
        OperatingSystem.LINUX: "Linux",
    }.get(info.os, "Unknown")

    table.add_row("Operating System", f"{os_emoji} ({info.os_version})")
    table.add_row("Architecture", info.architecture.value)

    # Hardware type with description
    hw_desc = {
        HardwareType.JETSON: "NVIDIA Jetson (Edge AI)",
        HardwareType.APPLE_SILICON: "Apple Silicon (M-series)",
        HardwareType.STANDARD_X86: "Standard x86_64",
        HardwareType.STANDARD_ARM: "Standard ARM",
    }.get(info.hardware_type, "Unknown")
    table.add_row("Hardware Type", hw_desc)

    table.add_row("Python Version", info.python_version)
    table.add_row("CPU Cores", str(info.cpu_count))

    if info.is_wsl:
        table.add_row("WSL", "Yes (Windows Subsystem for Linux)")

    # GPU info
    if info.gpu:
        gpu_str = f"{info.gpu.name}"
        if info.gpu.memory_mb > 0:
            gpu_str += f" ({info.gpu.memory_mb} MB)"
        table.add_row("GPU", gpu_str)
        if info.gpu.cuda_available:
            cuda_str = "Available"
            if info.gpu.cuda_version:
                cuda_str += f" (Driver: {info.gpu.cuda_version})"
            table.add_row("CUDA", cuda_str)
    else:
        table.add_row("GPU", "Not detected")

    if verbose:
        table.add_row("Hostname", info.hostname)

    console.print(table)

    # Print recommendation
    console.print()
    if info.hardware_type == HardwareType.JETSON:
        console.print(Panel(
            "[yellow]Jetson detected![/yellow] Use TensorRT-LLM provider for optimal performance.",
            title="Recommendation",
        ))
    elif info.hardware_type == HardwareType.APPLE_SILICON:
        console.print(Panel(
            "[yellow]Apple Silicon detected![/yellow] Use Ollama with Metal acceleration.",
            title="Recommendation",
        ))
    elif info.gpu and info.gpu.cuda_available:
        console.print(Panel(
            "[yellow]NVIDIA GPU detected![/yellow] Use Ollama or vLLM for GPU acceleration.",
            title="Recommendation",
        ))


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Show current configuration.",
    ),
    init: bool = typer.Option(
        False,
        "--init",
        "-i",
        help="Initialize configuration with defaults.",
    ),
    path: bool = typer.Option(
        False,
        "--path",
        "-p",
        help="Show configuration file path.",
    ),
) -> None:
    """Manage Animus configuration."""
    manager = ConfigManager()

    if path:
        console.print(f"Config path: [cyan]{manager.config_path}[/cyan]")
        return

    if init:
        manager.ensure_directories()
        manager.save()
        console.print(f"[green]Configuration initialized at:[/green] {manager.config_path}")
        return

    if show:
        if not manager.config_path.exists():
            console.print("[yellow]No configuration file found. Run 'animus config --init' to create one.[/yellow]")
            raise typer.Exit(1)

        import yaml
        with open(manager.config_path, "r") as f:
            content = f.read()
        console.print(Panel(content, title=str(manager.config_path), border_style="blue"))
        return

    # Default: show help
    console.print("Use [cyan]--show[/cyan] to view config, [cyan]--init[/cyan] to initialize, or [cyan]--path[/cyan] to show path.")


@app.command()
def init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration.",
    ),
) -> None:
    """Initialize Animus in the current directory or home."""
    manager = ConfigManager()

    if manager.config_path.exists() and not force:
        console.print(f"[yellow]Configuration already exists at {manager.config_path}[/yellow]")
        console.print("Use [cyan]--force[/cyan] to overwrite.")
        raise typer.Exit(1)

    # Detect environment
    console.print("[bold]Detecting system environment...[/bold]")
    info = detect_environment()

    # Create directories and save config
    manager.ensure_directories()

    # Adjust defaults based on detected hardware
    config = manager.config
    if info.hardware_type == HardwareType.JETSON:
        config.model.provider = "trtllm"
        console.print("[green]Jetson detected - configured for TensorRT-LLM[/green]")
    elif info.hardware_type == HardwareType.APPLE_SILICON:
        config.model.provider = "ollama"
        console.print("[green]Apple Silicon detected - configured for Ollama with Metal[/green]")
    else:
        config.model.provider = "ollama"
        console.print("[green]Configured for Ollama[/green]")

    manager.save(config)

    console.print(f"\n[bold green]Animus initialized![/bold green]")
    console.print(f"Config: [cyan]{manager.config_path}[/cyan]")
    console.print(f"Data:   [cyan]{config.data_dir}[/cyan]")
    console.print(f"\nNext steps:")
    console.print("  1. Run [cyan]animus detect[/cyan] to verify your environment")
    console.print("  2. Run [cyan]animus pull <model>[/cyan] to download a model")
    console.print("  3. Run [cyan]animus chat[/cyan] to start chatting")


@app.command()
def models(
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to list models from (ollama, api).",
    ),
) -> None:
    """List available models."""
    import asyncio
    from src.llm import OllamaProvider, APIProvider, ProviderType
    from src.core.config import ConfigManager

    async def list_models() -> None:
        config = ConfigManager().config
        provider_type = provider or config.model.provider

        if provider_type == "ollama":
            prov = OllamaProvider(
                host=config.ollama.host,
                port=config.ollama.port,
            )
            if not prov.is_available:
                console.print("[red]Ollama server not running.[/red]")
                console.print("Start Ollama with: [cyan]ollama serve[/cyan]")
                raise typer.Exit(1)
        elif provider_type == "api":
            if not config.model.api_key:
                console.print("[red]API key not configured.[/red]")
                console.print("Set it in [cyan]~/.animus/config.yaml[/cyan]")
                raise typer.Exit(1)
            prov = APIProvider(
                api_base=config.model.api_base or "https://api.openai.com/v1",
                api_key=config.model.api_key,
            )
        else:
            console.print(f"[red]Unknown provider: {provider_type}[/red]")
            raise typer.Exit(1)

        try:
            model_list = await prov.list_models()

            if not model_list:
                console.print(f"[yellow]No models found for {provider_type}.[/yellow]")
                if provider_type == "ollama":
                    console.print("Pull a model with: [cyan]animus pull <model>[/cyan]")
                return

            table = Table(title=f"Available Models ({provider_type})")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Quantization", style="yellow")

            for model in model_list:
                size = ""
                if model.size_bytes:
                    size_gb = model.size_bytes / (1024 ** 3)
                    size = f"{size_gb:.1f} GB"
                elif model.parameter_count:
                    size = model.parameter_count

                table.add_row(
                    model.name,
                    size,
                    model.quantization or "",
                )

            console.print(table)
        finally:
            if hasattr(prov, 'close'):
                await prov.close()

    asyncio.run(list_models())


@app.command()
def pull(
    model_name: str = typer.Argument(..., help="Name of the model to pull."),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider to pull from (default: ollama).",
    ),
) -> None:
    """Pull/download a model."""
    import asyncio
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from src.llm import OllamaProvider
    from src.core.config import ConfigManager

    async def pull_model() -> None:
        config = ConfigManager().config
        provider_type = provider or config.model.provider

        if provider_type != "ollama":
            console.print(f"[yellow]Only Ollama supports model pulling.[/yellow]")
            console.print("API models are accessed remotely and don't need to be downloaded.")
            console.print("TensorRT models must be compiled locally.")
            return

        prov = OllamaProvider(
            host=config.ollama.host,
            port=config.ollama.port,
        )

        if not prov.is_available:
            console.print("[red]Ollama server not running.[/red]")
            console.print("Start Ollama with: [cyan]ollama serve[/cyan]")
            raise typer.Exit(1)

        console.print(f"[bold]Pulling model:[/bold] {model_name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)

            try:
                async for update in prov.pull_model(model_name):
                    status = update.get("status", "")
                    completed = update.get("completed", 0)
                    total = update.get("total", 0)

                    if total > 0:
                        progress.update(task, total=total, completed=completed)

                    if "pulling" in status.lower():
                        digest = update.get("digest", "")[:12]
                        progress.update(task, description=f"Pulling {digest}...")
                    elif status:
                        progress.update(task, description=status)

                progress.update(task, description="Done!", completed=progress.tasks[0].total or 100)
            finally:
                await prov.close()

        console.print(f"\n[bold green]Model '{model_name}' pulled successfully![/bold green]")

    asyncio.run(pull_model())


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest."),
    chunk_size: int = typer.Option(512, "--chunk-size", "-c", help="Chunk size in tokens."),
    overlap: int = typer.Option(50, "--overlap", "-o", help="Overlap between chunks."),
) -> None:
    """Ingest documents into the knowledge base."""
    import asyncio
    from pathlib import Path as PathLib
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from src.memory import Ingester, IngestionProgress
    from src.core.config import ConfigManager

    async def run_ingestion() -> None:
        config = ConfigManager().config
        target_path = PathLib(path).resolve()

        if not target_path.exists():
            console.print(f"[red]Path not found:[/red] {target_path}")
            raise typer.Exit(1)

        console.print(f"[bold]Ingesting:[/bold] {target_path}")

        ingester = Ingester(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Starting...", total=None)

            def on_progress(p: IngestionProgress):
                if p.total > 0:
                    progress.update(task, total=p.total, completed=p.current)
                desc = p.message or p.stage
                if p.current_file:
                    # Show just filename, not full path
                    filename = PathLib(p.current_file).name
                    desc = f"{p.stage}: {filename}"
                progress.update(task, description=desc)

            try:
                stats = await ingester.ingest(
                    target_path,
                    progress_callback=on_progress,
                    persist_dir=config.data_dir / "vectordb",
                )

                progress.update(task, description="Done!", completed=progress.tasks[0].total or 100)

            finally:
                await ingester.close()

        # Print summary
        console.print()
        console.print("[bold green]Ingestion complete![/bold green]")
        console.print(f"  Files scanned:  {stats.files_scanned}")
        console.print(f"  Files processed: {stats.files_processed}")
        console.print(f"  Files skipped:   {stats.files_skipped}")
        console.print(f"  Chunks created:  {stats.chunks_created}")
        console.print(f"  Embeddings:      {stats.embeddings_generated}")

        if stats.errors:
            console.print(f"\n[yellow]Errors ({len(stats.errors)}):[/yellow]")
            for file, error in stats.errors[:5]:
                console.print(f"  {file}: {error}")
            if len(stats.errors) > 5:
                console.print(f"  ... and {len(stats.errors) - 5} more")

    asyncio.run(run_ingestion())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(5, "--results", "-k", help="Number of results."),
) -> None:
    """Search the knowledge base."""
    import asyncio
    from src.memory import Ingester
    from src.core.config import ConfigManager

    async def run_search() -> None:
        config = ConfigManager().config
        ingester = Ingester()

        try:
            results = await ingester.search(
                query,
                k=k,
                persist_dir=config.data_dir / "vectordb",
            )

            if not results:
                console.print("[yellow]No results found.[/yellow]")
                return

            console.print(f"[bold]Results for:[/bold] {query}\n")

            for i, (content, score, metadata) in enumerate(results, 1):
                source = metadata.get("source", "Unknown")
                console.print(f"[cyan]{i}.[/cyan] [dim]({score:.3f})[/dim] {source}")
                # Show first 200 chars of content
                preview = content[:200].replace("\n", " ")
                if len(content) > 200:
                    preview += "..."
                console.print(f"   {preview}\n")

        finally:
            await ingester.close()

    asyncio.run(run_search())


@app.command()
def status() -> None:
    """Show provider status and configured model."""
    import asyncio
    from src.llm import OllamaProvider, TRTLLMProvider, APIProvider
    from src.core.config import ConfigManager

    async def check_status() -> None:
        config = ConfigManager().config

        console.print("[bold]Animus Status[/bold]\n")

        # Show configured provider and model
        console.print(f"Configured Provider: [cyan]{config.model.provider}[/cyan]")
        console.print(f"Configured Model: [cyan]{config.model.model_name}[/cyan]")
        console.print()

        # Check Ollama
        ollama = OllamaProvider(host=config.ollama.host, port=config.ollama.port)
        ollama_status = "[green]Running[/green]" if ollama.is_available else "[red]Not Running[/red]"
        console.print(f"Ollama ({config.ollama.base_url}): {ollama_status}")

        if ollama.is_available:
            models = await ollama.list_models()
            if models:
                console.print(f"  Models: {len(models)} available")
            await ollama.close()

        # Check TensorRT-LLM
        trtllm = TRTLLMProvider(engine_dir=config.data_dir / "engines")
        trtllm_status = "[green]Available[/green]" if trtllm.is_available else "[dim]Not Installed[/dim]"
        console.print(f"TensorRT-LLM: {trtllm_status}")

        # Check API
        if config.model.api_key:
            api = APIProvider(
                api_base=config.model.api_base or "https://api.openai.com/v1",
                api_key=config.model.api_key,
            )
            api_status = "[green]Configured[/green]"
            console.print(f"API ({config.model.api_base or 'OpenAI'}): {api_status}")
            await api.close()
        else:
            console.print("API: [dim]Not Configured[/dim]")

    asyncio.run(check_status())


@app.command()
def chat(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use for chat.",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Skip tool confirmation prompts.",
    ),
) -> None:
    """Start an interactive chat session with the agent."""
    import asyncio
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from src.llm import get_default_provider
    from src.core import Agent, AgentConfig
    from src.core.config import ConfigManager

    async def run_chat() -> None:
        config = ConfigManager().config
        provider = get_default_provider(config)

        if not provider.is_available:
            console.print("[red]No LLM provider available.[/red]")
            console.print("Start Ollama with: [cyan]ollama serve[/cyan]")
            console.print("Or configure an API key in [cyan]~/.animus/config.yaml[/cyan]")
            raise typer.Exit(1)

        async def confirm_tool(tool_name: str, description: str) -> bool:
            if no_confirm:
                return True
            console.print(f"\n[yellow]Tool request:[/yellow] {tool_name}")
            console.print(f"[dim]{description}[/dim]")
            return Confirm.ask("Execute this tool?", default=True)

        agent_config = AgentConfig(
            model=model or config.model.model_name,
            temperature=config.model.temperature,
            require_tool_confirmation=not no_confirm,
        )

        agent = Agent(
            provider=provider,
            config=agent_config,
            confirm_callback=confirm_tool,
        )

        console.print("[bold blue]Animus Chat[/bold blue]")
        console.print("Type your message. Use [cyan]exit[/cyan] or [cyan]quit[/cyan] to end.\n")

        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")

                if user_input.lower() in ("exit", "quit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                if not user_input.strip():
                    continue

                console.print()

                async for turn in agent.run(user_input):
                    if turn.role == "assistant":
                        console.print("[bold blue]Animus[/bold blue]")
                        # Render as markdown
                        console.print(Markdown(turn.content))

                        if turn.tool_calls:
                            console.print(f"\n[dim]Executed {len(turn.tool_calls)} tool(s)[/dim]")

                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

        # Cleanup
        if hasattr(provider, 'close'):
            await provider.close()

    asyncio.run(run_chat())


if __name__ == "__main__":
    app()
