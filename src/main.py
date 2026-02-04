"""Animus CLI - Main entry point.

A techromantic CLI coding agent that runs locally.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

# Configure UTF-8 encoding for Windows to support Unicode banners
if sys.platform == "win32":
    # Try to enable UTF-8 mode for Windows console
    try:
        import codecs
        # Set UTF-8 encoding for stdout/stderr if not already set
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass  # Ignore encoding setup failures

    # ConnectionResetError from ProactorEventLoop is suppressed
    # via a custom exception handler set inside run_chat()

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import __version__
from src.core.config import ConfigManager
from src.core.detection import detect_environment, HardwareType, OperatingSystem
from src.incantations import speak, whisper, show_banner, get_response

# Initialize Typer app
app = typer.Typer(
    name="animus",
    help="A techromantic CLI coding agent that runs locally.",
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


@app.command("install")
def install_cmd(
    skip_native: bool = typer.Option(
        False,
        "--skip-native",
        help="Skip llama-cpp-python installation.",
    ),
    skip_embeddings: bool = typer.Option(
        False,
        "--skip-embeddings",
        help="Skip sentence-transformers installation.",
    ),
    force_cpu: bool = typer.Option(
        False,
        "--cpu",
        help="Force CPU-only installation (no GPU acceleration).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show detailed installation output.",
    ),
) -> None:
    """Install Animus - auto-detect system and install all dependencies.

    This command detects your system (OS, architecture, GPU) and installs
    the appropriate dependencies for optimal performance.

    Supported platforms:
    - Windows (x86_64)
    - macOS (Intel & Apple Silicon)
    - Linux (x86_64, ARM64)
    - NVIDIA Jetson (Nano, TX2, Xavier, Orin)
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from src.install import install_animus, InstallProgress, InstallStep

    console.print("[bold magenta]Installing Animus[/bold magenta]\n")

    steps_done = set()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        current_task = None

        def on_progress(p: InstallProgress):
            nonlocal current_task

            # Create new task for each step
            if p.step.value not in steps_done:
                if current_task is not None:
                    progress.update(current_task, description=f"[green]✓[/green] {p.message}")
                current_task = progress.add_task(p.message)
                steps_done.add(p.step.value)
            else:
                # Update current task
                if p.success:
                    if p.warning:
                        progress.update(current_task, description=f"[yellow]![/yellow] {p.message}")
                    else:
                        progress.update(current_task, description=f"[dim]{p.message}[/dim]")
                else:
                    progress.update(current_task, description=f"[red]✗[/red] {p.message}")

            if verbose and p.detail:
                console.print(f"  [dim]{p.detail}[/dim]")

        result = install_animus(
            skip_native=skip_native,
            skip_embeddings=skip_embeddings,
            force_cpu=force_cpu,
            verbose=verbose,
            progress_callback=on_progress,
        )

        # Mark final task complete
        if current_task is not None:
            progress.update(current_task, description="[green]✓[/green] Verification complete")

    console.print()

    # Show results
    if result.success:
        console.print("[bold green]Installation complete![/bold green]\n")
    else:
        console.print("[bold red]Installation completed with errors[/bold red]\n")

    # System info
    if result.system_info:
        info = result.system_info
        console.print(f"[bold]System:[/bold] {info.os.value} {info.architecture.value}")
        console.print(f"[bold]Hardware:[/bold] {info.hardware_type.value}")
        if info.gpu:
            console.print(f"[bold]GPU:[/bold] {info.gpu.name}")
        console.print()

    # Components installed
    if result.installed_components:
        console.print("[bold]Installed:[/bold]")
        for comp in result.installed_components:
            console.print(f"  [green]✓[/green] {comp}")
        console.print()

    # Warnings
    if result.warnings:
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for warn in result.warnings:
            console.print(f"  [yellow]![/yellow] {warn}")
        console.print()

    # Errors
    if result.errors:
        console.print("[bold red]Errors:[/bold red]")
        for err in result.errors:
            console.print(f"  [red]✗[/red] {err}")
        console.print()

    # Next steps
    if result.next_steps:
        console.print("[bold]Next steps:[/bold]")
        for i, step in enumerate(result.next_steps, 1):
            console.print(f"\n{i}. {step}")


@app.command("detect")
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
            "[yellow]Apple Silicon detected![/yellow] Use native provider with Metal acceleration.",
            title="Recommendation",
        ))
    elif info.gpu and info.gpu.cuda_available:
        console.print(Panel(
            "[yellow]NVIDIA GPU detected![/yellow] Use native provider with CUDA acceleration.",
            title="Recommendation",
        ))


@app.command("config")
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


@app.command("init")
def init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration.",
    ),
) -> None:
    """Initialize Animus in the current directory."""
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
    else:
        # Default: LiteLLM provider with managed llama-server
        config.model.provider = "litellm"
        if info.hardware_type == HardwareType.APPLE_SILICON:
            console.print("[green]Apple Silicon detected - configured for LiteLLM + llama-server (Metal)[/green]")
        elif info.gpu and info.gpu.cuda_available:
            console.print("[green]NVIDIA GPU detected - configured for LiteLLM + llama-server (CUDA)[/green]")
        else:
            console.print("[green]Configured for LiteLLM + llama-server (CPU)[/green]")

    manager.save(config)

    console.print(f"\n[bold green]Animus initialized![/bold green]")
    console.print(f"Config: [cyan]{manager.config_path}[/cyan]")
    console.print(f"Data:   [cyan]{config.data_dir}[/cyan]")
    console.print(f"\nNext steps:")
    console.print("  1. Run [cyan]animus detect[/cyan] to verify your environment")
    console.print("  2. Run [cyan]animus pull <repo_id/filename.gguf>[/cyan] to download a model")
    console.print("  3. Run [cyan]animus rise[/cyan] to start chatting")


@app.command("models")
def models() -> None:
    """List available local GGUF models."""
    import asyncio
    from src.llm import LiteLLMProvider, LITELLM_AVAILABLE
    from src.core.config import ConfigManager

    async def list_models() -> None:
        config = ConfigManager().config
        provider = LiteLLMProvider(models_dir=config.native.models_dir)

        console.print(f"[bold]Local Models[/bold]")
        console.print(f"[dim]Directory: {config.native.models_dir}[/dim]\n")

        model_list = await provider.list_models()

        if not model_list:
            console.print("[yellow]No local models found.[/yellow]")
            console.print("\nDownload a model with:")
            console.print("  [cyan]animus pull <repo_id/filename.gguf>[/cyan]")
            return

        table = Table(title="Local GGUF Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Quantization", style="yellow")

        for model in model_list:
            size = ""
            if model.size_bytes:
                size_gb = model.size_bytes / (1024 ** 3)
                if size_gb >= 1:
                    size = f"{size_gb:.2f} GB"
                else:
                    size_mb = model.size_bytes / (1024 ** 2)
                    size = f"{size_mb:.0f} MB"
            table.add_row(
                model.name,
                size,
                model.quantization or "",
            )

        console.print(table)

        # Show provider status
        console.print()
        if LITELLM_AVAILABLE:
            console.print(f"[green]LiteLLM:[/green] Installed")
        else:
            console.print("[yellow]LiteLLM:[/yellow] Not installed")
            console.print("Install with: [cyan]pip install litellm[/cyan]")

        # Show llama-server status
        from src.llm.server import LlamaServer
        server = LlamaServer(bin_dir=config.native.bin_dir)
        if server.is_installed:
            console.print(f"[green]llama-server:[/green] Installed ({server.binary_path})")
        else:
            console.print("[dim]llama-server:[/dim] Not installed (will auto-download on first use)")

    asyncio.run(list_models())


@app.command("pull")
def pull_model(
    model_name: str = typer.Argument(
        ...,
        help="Model to download (e.g., Qwen/Qwen2.5-Coder-7B-Instruct-GGUF)",
    ),
) -> None:
    """Pull a model - download a GGUF model from Hugging Face."""
    import asyncio
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from src.llm import LiteLLMProvider, HF_HUB_AVAILABLE
    from src.core.config import ConfigManager

    if not HF_HUB_AVAILABLE:
        console.print("[red]huggingface-hub not installed.[/red]")
        console.print("Install with: [cyan]pip install huggingface-hub[/cyan]")
        raise typer.Exit(1)

    async def download() -> None:
        config = ConfigManager().config
        provider = LiteLLMProvider(models_dir=config.native.models_dir)

        console.print(f"[bold]Downloading model:[/bold] {model_name}")
        console.print(f"[dim]Target directory: {config.native.models_dir}[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Starting...", total=None)

            async for update in provider.pull_model(model_name):
                status = update.get("status", "")

                if status == "error":
                    progress.update(task, description=f"[red]Error: {update.get('error')}[/red]")
                    console.print(f"\n[red]Download failed:[/red] {update.get('error')}")
                    raise typer.Exit(1)
                elif status == "searching":
                    progress.update(task, description=update.get("message", "Searching..."))
                elif status == "selected":
                    progress.update(task, description=f"Selected: {update.get('filename')}")
                elif status == "downloading":
                    progress.update(task, description=f"Downloading: {update.get('filename')}")
                elif status == "complete":
                    progress.update(task, description="Done!")
                    size_mb = update.get("size", 0) / (1024 * 1024)
                    console.print(f"\n[bold green]Download complete![/bold green]")
                    console.print(f"  Path: [cyan]{update.get('path')}[/cyan]")
                    console.print(f"  Size: {size_mb:.1f} MB")

    asyncio.run(download())


@app.command("ingest")
def ingest(
    path: str = typer.Argument(..., help="File or directory to consume."),
    chunk_size: int = typer.Option(512, "--chunk-size", "-c", help="Chunk size in tokens."),
    overlap: int = typer.Option(50, "--overlap", "-o", help="Overlap between chunks."),
) -> None:
    """Ingest documents into Animus's memory for RAG."""
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


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(5, "--results", "-k", help="Number of results."),
) -> None:
    """Search Animus's accumulated knowledge."""
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


# Model management subcommand group (vessel)
model_app = typer.Typer(
    name="model",
    help="Manage GGUF models.",
    no_args_is_help=True,
)
app.add_typer(model_app, name="model")


@model_app.command("download")
def model_download(
    model_name: str = typer.Argument(
        ...,
        help="Model to download (format: repo_id or repo_id/filename.gguf)",
    ),
) -> None:
    """Download a GGUF model from Hugging Face."""
    import asyncio
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from src.llm import LiteLLMProvider, HF_HUB_AVAILABLE
    from src.core.config import ConfigManager

    if not HF_HUB_AVAILABLE:
        console.print("[red]huggingface-hub not installed.[/red]")
        console.print("Install with: [cyan]pip install huggingface-hub[/cyan]")
        raise typer.Exit(1)

    async def download() -> None:
        config = ConfigManager().config
        provider = LiteLLMProvider(models_dir=config.native.models_dir)

        console.print(f"[bold]Downloading model:[/bold] {model_name}")
        console.print(f"[dim]Target directory: {config.native.models_dir}[/dim]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Starting...", total=None)

            async for update in provider.pull_model(model_name):
                status = update.get("status", "")

                if status == "error":
                    progress.update(task, description=f"[red]Error: {update.get('error')}[/red]")
                    console.print(f"\n[red]Download failed:[/red] {update.get('error')}")
                    raise typer.Exit(1)
                elif status == "searching":
                    progress.update(task, description=update.get("message", "Searching..."))
                elif status == "selected":
                    progress.update(task, description=f"Selected: {update.get('filename')}")
                elif status == "downloading":
                    progress.update(task, description=f"Downloading: {update.get('filename')}")
                elif status == "complete":
                    progress.update(task, description="Done!")
                    size_mb = update.get("size", 0) / (1024 * 1024)
                    console.print(f"\n[bold green]Download complete![/bold green]")
                    console.print(f"  Path: [cyan]{update.get('path')}[/cyan]")
                    console.print(f"  Size: {size_mb:.1f} MB")

    asyncio.run(download())


@model_app.command("list")
def model_list() -> None:
    """List locally downloaded GGUF models."""
    import asyncio
    from src.llm import LiteLLMProvider, LITELLM_AVAILABLE
    from src.core.config import ConfigManager

    async def list_local() -> None:
        config = ConfigManager().config
        provider = LiteLLMProvider(models_dir=config.native.models_dir)

        console.print(f"[bold]Local Models[/bold]")
        console.print(f"[dim]Directory: {config.native.models_dir}[/dim]\n")

        model_list = await provider.list_models()

        if not model_list:
            console.print("[yellow]No local models found.[/yellow]")
            console.print("\nDownload a model with:")
            console.print("  [cyan]animus pull <repo_id/filename.gguf>[/cyan]")
            return

        table = Table(title="Local GGUF Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Quantization", style="yellow")

        for m in model_list:
            size = ""
            if m.size_bytes:
                size_gb = m.size_bytes / (1024 ** 3)
                if size_gb >= 1:
                    size = f"{size_gb:.2f} GB"
                else:
                    size_mb = m.size_bytes / (1024 ** 2)
                    size = f"{size_mb:.0f} MB"

            table.add_row(
                m.name,
                size,
                m.quantization or "Unknown",
            )

        console.print(table)

        # Show provider status
        console.print()
        if LITELLM_AVAILABLE:
            console.print(f"[green]LiteLLM:[/green] Installed")
        else:
            console.print("[yellow]LiteLLM:[/yellow] Not installed")

        from src.llm.server import LlamaServer
        server = LlamaServer(bin_dir=config.native.bin_dir)
        if server.is_installed:
            console.print(f"[green]llama-server:[/green] Installed")
        else:
            console.print("[dim]llama-server:[/dim] Not installed (auto-downloads on first use)")

    asyncio.run(list_local())


@model_app.command("remove")
def model_remove(
    model_name: str = typer.Argument(..., help="Name of the model file to remove."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Remove a locally downloaded model."""
    from rich.prompt import Confirm
    from src.core.config import ConfigManager

    config = ConfigManager().config
    model_path = config.native.models_dir / model_name

    if not model_path.exists():
        # Try with .gguf extension
        model_path = config.native.models_dir / f"{model_name}.gguf"
        if not model_path.exists():
            console.print(f"[red]Model not found:[/red] {model_name}")
            raise typer.Exit(1)

    size_mb = model_path.stat().st_size / (1024 * 1024)

    if not force:
        console.print(f"Model: [cyan]{model_path.name}[/cyan]")
        console.print(f"Size: {size_mb:.1f} MB")
        if not Confirm.ask("Delete this model?", default=False):
            console.print("[dim]Cancelled.[/dim]")
            return

    model_path.unlink()
    console.print(f"[green]Deleted:[/green] {model_path.name}")


# Skills subcommand group (tomes)
skill_app = typer.Typer(
    name="skill",
    help="Manage skills that extend Animus's capabilities.",
    no_args_is_help=True,
)
app.add_typer(skill_app, name="skill")


@skill_app.command("list")
def skill_list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info."),
) -> None:
    """List available skills."""
    from pathlib import Path
    from src.skills import SkillRegistry

    registry = SkillRegistry()
    registry.discover(Path.cwd())
    skills = registry.list_enabled()

    if not skills:
        console.print("[yellow]No skills found.[/yellow]")
        console.print("\nCreate a skill with: [cyan]animus skill create <name>[/cyan]")
        console.print("Or install one: [cyan]animus skill install <url>[/cyan]")
        return

    table = Table(title="Available Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    if verbose:
        table.add_column("Version", style="dim")
        table.add_column("Tags", style="yellow")
        table.add_column("Source", style="dim")

    for skill in skills:
        if verbose:
            source = str(skill.source_path.parent.name) if skill.source_path else "unknown"
            tags = ", ".join(skill.metadata.tags) if skill.metadata.tags else "-"
            table.add_row(
                skill.name,
                skill.description[:50] + "..." if len(skill.description) > 50 else skill.description,
                skill.metadata.version,
                tags,
                source,
            )
        else:
            table.add_row(
                skill.name,
                skill.description[:60] + "..." if len(skill.description) > 60 else skill.description,
            )

    console.print(table)
    console.print(f"\n[dim]Found {len(skills)} skill(s)[/dim]")


@skill_app.command("show")
def skill_show(
    name: str = typer.Argument(..., help="Name of the skill to show."),
) -> None:
    """Show details of a specific skill."""
    from pathlib import Path
    from src.skills import SkillRegistry

    registry = SkillRegistry()
    registry.discover(Path.cwd())
    skill = registry.get(name)

    if not skill:
        console.print(f"[red]Skill not found:[/red] {name}")
        console.print("\nAvailable skills:")
        for s in registry.list():
            console.print(f"  - {s.name}")
        raise typer.Exit(1)

    console.print(f"[bold cyan]{skill.name}[/bold cyan] v{skill.metadata.version}")
    console.print(f"\n{skill.description}")

    if skill.metadata.author:
        console.print(f"\n[dim]Author:[/dim] {skill.metadata.author}")

    if skill.metadata.tags:
        console.print(f"[dim]Tags:[/dim] {', '.join(skill.metadata.tags)}")

    if skill.metadata.requires:
        console.print(f"[dim]Requires:[/dim] {', '.join(skill.metadata.requires)}")

    if skill.source_path:
        console.print(f"[dim]Source:[/dim] {skill.source_path}")

    console.print(f"\n[bold]Instructions:[/bold]")
    console.print(Panel(skill.instructions[:500] + "..." if len(skill.instructions) > 500 else skill.instructions))


@skill_app.command("inscribe")
def skill_create(
    name: str = typer.Argument(..., help="Name for the new skill."),
    description: str = typer.Option("A custom Animus skill", "--description", "-d", help="Skill description."),
    project: bool = typer.Option(False, "--project", "-p", help="Create in project directory instead of user."),
) -> None:
    """Inscribe a new spell - create a skill from template."""
    from pathlib import Path
    from src.skills import SkillRegistry

    target_dir = Path.cwd() / "skills" if project else None
    registry = SkillRegistry()

    try:
        skill_path = registry.create(name, description, target_dir)
        console.print(f"[green]Skill created:[/green] {skill_path}")
        console.print(f"\nEdit [cyan]{skill_path}[/cyan] to customize your skill.")
    except Exception as e:
        console.print(f"[red]Failed to create skill:[/red] {e}")
        raise typer.Exit(1)


@skill_app.command("install")
def skill_install(
    url: str = typer.Argument(..., help="URL to skill (GitHub repo or raw SKILL.md)."),
) -> None:
    """Install a skill from URL."""
    from src.skills import SkillRegistry

    registry = SkillRegistry()

    console.print(f"[bold]Installing skill from:[/bold] {url}")

    try:
        skill = registry.install_from_url(url)
        console.print(f"[green]Installed:[/green] {skill.name} v{skill.metadata.version}")
        console.print(f"[dim]Location:[/dim] {skill.source_path}")
    except Exception as e:
        console.print(f"[red]Installation failed:[/red] {e}")
        raise typer.Exit(1)


@skill_app.command("run")
def skill_run(
    name: str = typer.Argument(..., help="Name of the skill to run."),
    prompt: str = typer.Argument(..., help="Prompt to run with the skill."),
) -> None:
    """Run a skill with a prompt."""
    from pathlib import Path
    from src.skills import SkillLoader
    from src.llm import get_default_provider
    from src.core.config import ConfigManager
    import asyncio

    async def run_with_skill() -> None:
        config = ConfigManager().config
        provider = get_default_provider(config)

        if not provider.is_available:
            console.print("[red]No LLM provider available.[/red]")
            raise typer.Exit(1)

        loader = SkillLoader()
        loader.discover_all(Path.cwd())
        skill = loader.load_skill(name)

        if not skill:
            console.print(f"[red]Skill not found:[/red] {name}")
            raise typer.Exit(1)

        console.print(f"[bold]Running skill:[/bold] {skill.name}")
        console.print()

        # Create enhanced prompt with skill
        from src.llm.base import Message
        messages = [
            Message(role="system", content=skill.to_prompt()),
            Message(role="user", content=prompt),
        ]

        try:
            result = await provider.generate(messages)
            console.print(result.text)
        finally:
            if hasattr(provider, 'close'):
                await provider.close()

    asyncio.run(run_with_skill())


@model_app.command("info")
def model_info(
    model_name: str = typer.Argument(..., help="Name of the model to inspect."),
) -> None:
    """Show information about a local model."""
    from src.core.config import ConfigManager

    config = ConfigManager().config
    models_dir = config.native.models_dir

    # Try to find the model file
    model_path = models_dir / model_name
    if not model_path.exists():
        model_path = models_dir / f"{model_name}.gguf"
        if not model_path.exists():
            # Fuzzy match
            matches = [f for f in models_dir.glob("*.gguf") if model_name.lower() in f.name.lower()]
            if matches:
                model_path = matches[0]
            else:
                console.print(f"[red]Model not found:[/red] {model_name}")
                raise typer.Exit(1)

    stat = model_path.stat()

    # Detect quantization from filename
    quant = None
    for q in ["Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32"]:
        if q.lower() in model_path.name.lower():
            quant = q
            break

    console.print(f"[bold]Model Information[/bold]\n")
    console.print(f"  Name: [cyan]{model_path.name}[/cyan]")
    console.print(f"  Path: {model_path}")
    console.print(f"  Size: {stat.st_size / (1024**3):.2f} GB")
    console.print(f"  Quantization: {quant or 'Unknown'}")

    # Show llama-server status
    from src.llm.server import LlamaServer
    server = LlamaServer(bin_dir=config.native.bin_dir)
    if server.is_installed:
        console.print(f"\n[green]Ready for inference via llama-server[/green]")
    else:
        console.print(f"\n[dim]llama-server will auto-download on first use[/dim]")


@app.command("analyze")
def analyze(
    goal: Optional[str] = typer.Argument(
        None,
        help="Filter runs by goal (substring match).",
    ),
    days: Optional[int] = typer.Option(
        None,
        "--days",
        "-d",
        help="Only analyze runs from last N days.",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Maximum number of runs to analyze.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run",
        "-r",
        help="Show details for a specific run ID.",
    ),
    trends: bool = typer.Option(
        False,
        "--trends",
        "-t",
        help="Show trends over time.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
) -> None:
    """Analyze past runs and suggest improvements."""
    from src.core.builder import BuilderQuery, SuggestionPriority

    builder = BuilderQuery()

    # Show specific run details
    if run_id:
        details = builder.get_run_details(run_id)
        if not details:
            console.print(f"[red]Run not found:[/red] {run_id}")
            raise typer.Exit(1)

        if json_output:
            import json
            console.print_json(json.dumps(details, indent=2, default=str))
        else:
            console.print(f"[bold]Run Details[/bold]\n")
            console.print(f"ID: [cyan]{details['id']}[/cyan]")
            console.print(f"Goal: {details['goal']}")
            console.print(f"Status: {details['status']}")
            console.print(f"Duration: {details['duration_ms']/1000:.1f}s")
            console.print(f"Tokens: {details['metrics']['tokens_used']}")
            console.print(f"Tool Success Rate: {details['success_rate']*100:.1f}%")

            if details['errors']:
                console.print(f"\n[bold red]Errors:[/bold red]")
                for err in details['errors'][:5]:
                    console.print(f"  - {err}")

            if details['decisions']:
                console.print(f"\n[bold]Decisions ({details['decision_count']}):[/bold]")
                for dec in details['decisions'][:5]:
                    console.print(f"  [{dec['type']}] {dec['intent'][:50]}...")
        return

    # Show trends
    if trends:
        trend_data = builder.get_trends(days=days or 30)

        if "error" in trend_data:
            console.print(f"[yellow]{trend_data['error']}[/yellow]")
            return

        if json_output:
            import json
            console.print_json(json.dumps(trend_data, indent=2, default=str))
        else:
            console.print(f"[bold]Trends (Last {trend_data['period_days']} Days)[/bold]\n")
            console.print(f"Total Runs: {trend_data['total_runs']}")
            console.print(f"Overall Trend: [{'green' if trend_data['overall_trend'] == 'improving' else 'yellow'}]{trend_data['overall_trend']}[/]")

            if trend_data['daily_stats']:
                console.print(f"\n[bold]Daily Stats:[/bold]")
                table = Table()
                table.add_column("Date", style="cyan")
                table.add_column("Runs", justify="right")
                table.add_column("Success", justify="right")
                table.add_column("Failed", justify="right")
                table.add_column("Rate", justify="right")

                for day in trend_data['daily_stats'][-10:]:  # Last 10 days
                    rate_color = "green" if day['success_rate'] >= 0.8 else "yellow" if day['success_rate'] >= 0.5 else "red"
                    table.add_row(
                        day['date'],
                        str(day['runs']),
                        str(day['completed']),
                        str(day['failed']),
                        f"[{rate_color}]{day['success_rate']*100:.0f}%[/]",
                    )
                console.print(table)
        return

    # Full analysis
    result = builder.analyze(goal_filter=goal, days=days, limit=limit)

    if json_output:
        import json
        console.print_json(json.dumps(result.to_dict(), indent=2, default=str))
        return

    # Print summary
    console.print(f"[bold]Run Analysis[/bold]\n")
    console.print(result.summary)

    # Print suggestions
    if result.suggestions:
        console.print(f"\n[bold]Suggestions ({len(result.suggestions)}):[/bold]\n")

        priority_colors = {
            SuggestionPriority.CRITICAL: "red",
            SuggestionPriority.HIGH: "yellow",
            SuggestionPriority.MEDIUM: "blue",
            SuggestionPriority.LOW: "dim",
        }

        for i, sug in enumerate(result.suggestions[:10], 1):
            color = priority_colors.get(sug.priority, "white")
            console.print(f"[{color}]{i}. [{sug.priority.value.upper()}] {sug.title}[/]")
            console.print(f"   {sug.description}")
            if sug.suggested_actions:
                console.print(f"   [dim]Actions:[/dim]")
                for action in sug.suggested_actions[:2]:
                    console.print(f"     - {action}")
            console.print()
    else:
        console.print("\n[green]No significant issues found.[/green]")


@app.command("status")
def status() -> None:
    """Show provider status and available models."""
    import asyncio
    from src.llm import LiteLLMProvider, LITELLM_AVAILABLE
    from src.llm.server import LlamaServer
    from src.core.config import ConfigManager

    async def check_status() -> None:
        config = ConfigManager().config

        console.print("[bold]Animus Status[/bold]\n")

        # Show configured provider and model
        console.print(f"Configured Provider: [cyan]{config.model.provider}[/cyan]")
        console.print(f"Configured Model: [cyan]{config.model.model_name or '(auto-detect)'}[/cyan]")
        console.print()

        # Check LiteLLM
        litellm_status = "[green]Installed[/green]" if LITELLM_AVAILABLE else "[dim]Not Installed[/dim]"
        console.print(f"LiteLLM: {litellm_status}")

        # Check llama-server
        server = LlamaServer(bin_dir=config.native.bin_dir)
        if server.is_installed:
            console.print(f"llama-server: [green]Installed[/green] ({server.binary_path})")
        else:
            console.print("llama-server: [dim]Not installed[/dim] (auto-downloads on first use)")

        # Check local models
        provider = LiteLLMProvider(models_dir=config.native.models_dir)
        local_models = await provider.list_models()
        console.print()
        if local_models:
            console.print(f"[bold]Local GGUF Models: {len(local_models)}[/bold]")
            for m in local_models[:5]:
                size = ""
                if m.size_bytes:
                    size_gb = m.size_bytes / (1024 ** 3)
                    size = f" ({size_gb:.1f} GB)" if size_gb >= 1 else f" ({m.size_bytes / (1024**2):.0f} MB)"
                console.print(f"  - {m.name}{size}")
            if len(local_models) > 5:
                console.print(f"  ... and {len(local_models) - 5} more")
        else:
            console.print("[yellow]No local models found[/yellow]")
            console.print("Download with: [cyan]animus pull <repo_id/filename.gguf>[/cyan]")

        # Check API configuration
        console.print()
        if config.model.api_key:
            console.print(f"API: [green]Configured[/green] ({config.model.api_base or 'default'})")
        else:
            console.print("API: [dim]No API key configured[/dim]")

        # Check legacy providers
        try:
            from src.llm import TRTLLMProvider
            trtllm = TRTLLMProvider(engine_dir=config.data_dir / "engines")
            if trtllm.is_available:
                console.print(f"TensorRT-LLM: [green]Available[/green]")
        except Exception:
            pass

    asyncio.run(check_status())


@app.command("rise")
def chat(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use.",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Skip tool confirmation prompts.",
    ),
    max_context: Optional[int] = typer.Option(
        None,
        "--max-context",
        "-c",
        help="Maximum context window size in tokens (default: 8192).",
    ),
    show_tokens: bool = typer.Option(
        False,
        "--show-tokens",
        help="Show token usage after each turn.",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream tokens in real-time (default: enabled).",
    ),
    plan: bool = typer.Option(
        False,
        "--plan",
        "-p",
        help="Enable planning mode (create plan before execution).",
    ),
    delegate: bool = typer.Option(
        False,
        "--delegate",
        "-d",
        help="Enable multi-agent delegation (spawn sub-agents for tasks).",
    ),
) -> None:
    """Rise! Begin an interactive session with Animus."""
    import asyncio
    from rich.prompt import Prompt, Confirm
    from rich.markdown import Markdown
    from src.llm import get_default_provider
    from src.core import Agent, AgentConfig
    from src.core.config import ConfigManager
    from src.core.context import ContextWindow, ContextConfig, ContextStatus

    async def run_chat() -> None:
        # Suppress ConnectionResetError noise on Windows ProactorEventLoop
        if sys.platform == "win32":
            loop = asyncio.get_running_loop()
            def _suppress_connection_reset(loop, context):
                exc = context.get('exception')
                if isinstance(exc, ConnectionResetError):
                    return  # Benign — llama-server connection closed
                loop.default_exception_handler(context)
            loop.set_exception_handler(_suppress_connection_reset)

        config = ConfigManager().config
        provider = get_default_provider(config)
        model_name = model or config.model.model_name

        if not provider.is_available:
            console.print("[red]Provider not available.[/red]")
            if config.model.provider == "litellm":
                console.print("litellm not available.\n")
                console.print("Install with: [cyan]pip install litellm[/cyan]")
            else:
                console.print(f"{config.model.provider} not available.\n")
            console.print("Or configure an API in [cyan]~/.animus/config.yaml[/cyan]")
            raise typer.Exit(1)

        # For LiteLLM provider, check if we have local models or API key
        if config.model.provider in ("litellm", "native"):
            local_models = await provider.list_models()
            if model_name:
                # Specific model requested — check it exists
                exists = await provider.model_exists(model_name)
                if not exists:
                    console.print("[yellow]Model not found.[/yellow]")
                    console.print(f"Model [cyan]{model_name}[/cyan] not found.\n")
                    console.print("Download a GGUF model:")
                    console.print("  [cyan]animus pull <repo_id/filename.gguf>[/cyan]\n")
                    console.print("Or list available models:")
                    console.print("  [cyan]animus models[/cyan]")
                    raise typer.Exit(1)
            elif not local_models and not config.model.api_key:
                # No model specified and none found
                console.print("[yellow]No models found.[/yellow]")
                console.print("No local models found and no API key configured.\n")
                console.print("Download a model:")
                console.print("  [cyan]animus pull <repo_id/filename.gguf>[/cyan]")
                raise typer.Exit(1)
            elif not model_name and local_models:
                # Auto-select first local model
                model_name = f"local/{local_models[0].name}"
                console.print(f"[dim]Using model: {local_models[0].name}[/dim]")

        async def confirm_tool(tool_name: str, description: str) -> bool:
            if no_confirm:
                return True
            console.print(f"\n[yellow]Tool request:[/yellow] {tool_name}")
            console.print(f"[dim]{description}[/dim]")
            return Confirm.ask("Execute this tool?", default=True)

        agent_config = AgentConfig(
            model=model_name or "",
            temperature=config.model.temperature,
            require_tool_confirmation=not no_confirm,
            enable_planning=plan,
            enable_delegation=delegate,
        )

        agent = Agent(
            provider=provider,
            config=agent_config,
            animus_config=config,
            confirm_callback=confirm_tool,
        )

        # Initialize context window tracking
        context_config = ContextConfig(max_tokens=max_context or 8192)
        context_window = ContextWindow(config=context_config)
        context_window.set_system_prompt(agent.system_prompt)
        turn_number = 0

        # Show awakening banner and response
        show_banner("awakening")
        speak("rise", newline_before=False)

        # Speak greeting via TTS (always on startup)
        try:
            from src.audio import AudioPlayer
            greeter = AudioPlayer(volume=config.audio.volume)
            greeter.speak("Greetings, Master.", blocking=True)
        except Exception:
            pass  # TTS not available, continue silently

        if max_context:
            console.print(f"[dim]Max context: {max_context} tokens[/dim]")
        if plan:
            console.print("[cyan]Planning mode enabled[/cyan] - I'll create a plan before acting.")
        if delegate:
            console.print("[cyan]Delegation enabled[/cyan] - I can spawn sub-agents for complex tasks.")
        console.print("Say and I shall do. Say [cyan]to-dust[/cyan] to end.\n")

        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")

                if user_input.lower() in ("exit", "quit", "q", "to-dust"):
                    show_banner("farewell")
                    break

                if not user_input.strip():
                    continue

                # Track user input tokens
                turn_number += 1
                context_window.add_turn(turn_number, "user", user_input)

                # Check context status before generating
                if context_window.status == ContextStatus.WARNING:
                    console.print(f"[yellow]Warning: Context at {context_window.usage_ratio:.0%} capacity[/yellow]")
                elif context_window.needs_compaction():
                    console.print(f"[red]Context full ({context_window.usage_ratio:.0%}). Consider starting a new session.[/red]")

                console.print()

                # Planning mode - create and display plan first
                if plan and agent.is_planning_enabled():
                    from src.core import ExecutionPlan
                    console.print("[cyan]Creating plan...[/cyan]")
                    execution_plan = await agent.create_plan(user_input)

                    if execution_plan and execution_plan.steps:
                        # Display the plan
                        console.print("\n[bold cyan]Execution Plan[/bold cyan]")
                        console.print(f"[dim]{execution_plan.summary}[/dim]\n")

                        for i, step in enumerate(execution_plan.steps, 1):
                            dep_str = ""
                            if step.dependencies:
                                dep_str = f" [dim](after step {', '.join(str(execution_plan.steps.index(s)+1) for s in execution_plan.steps if s.id in step.dependencies)})[/dim]"
                            tools_str = f" [dim][{', '.join(step.tool_hints)}][/dim]" if step.tool_hints else ""
                            console.print(f"  {i}. {step.description}{tools_str}{dep_str}")

                        console.print()

                        # Ask for confirmation
                        if not Confirm.ask("Execute this plan?", default=True):
                            console.print("[yellow]Plan cancelled.[/yellow]")
                            continue

                        console.print()

                if stream:
                    # Streaming mode - display tokens as they arrive
                    from src.core import StreamChunk

                    streaming_started = False
                    current_content = []

                    async for chunk in agent.run_stream(user_input):
                        if chunk.type == "token":
                            # Print header on first token
                            if not streaming_started:
                                console.print("[bold blue]Animus[/bold blue]", end="")
                                console.print()  # newline after header
                                streaming_started = True
                            # Print token immediately (raw, no markdown)
                            print(chunk.token, end="", flush=True)
                            current_content.append(chunk.token)
                        elif chunk.type == "turn" and chunk.turn:
                            turn = chunk.turn
                            if turn.role == "assistant":
                                # End the streaming line
                                if streaming_started:
                                    print()  # newline after streamed content
                                    streaming_started = False
                                    current_content = []

                                # Track assistant response tokens
                                turn_number += 1
                                context_window.add_turn(turn_number, "assistant", turn.content)

                                if turn.tool_calls:
                                    console.print(f"\n[dim]Executed {len(turn.tool_calls)} tool(s)[/dim]")

                                # Show token usage if requested
                                if show_tokens:
                                    stats = context_window.get_stats()
                                    console.print(f"[dim]Tokens: {stats['total_tokens']}/{stats['max_tokens']} ({stats['usage_ratio']:.0%})[/dim]")
                else:
                    # Non-streaming mode - display complete responses
                    async for turn in agent.run(user_input):
                        if turn.role == "assistant":
                            # Track assistant response tokens
                            turn_number += 1
                            context_window.add_turn(turn_number, "assistant", turn.content)

                            console.print("[bold blue]Animus[/bold blue]")
                            # Render as markdown
                            console.print(Markdown(turn.content))

                            if turn.tool_calls:
                                console.print(f"\n[dim]Executed {len(turn.tool_calls)} tool(s)[/dim]")

                            # Show token usage if requested
                            if show_tokens:
                                stats = context_window.get_stats()
                                console.print(f"[dim]Tokens: {stats['total_tokens']}/{stats['max_tokens']} ({stats['usage_ratio']:.0%})[/dim]")

                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

        # Cleanup
        if hasattr(provider, 'close'):
            await provider.close()

    asyncio.run(run_chat())


# MCP subcommand group (portal)
mcp_app = typer.Typer(
    name="mcp",
    help="Model Context Protocol (MCP) server.",
    no_args_is_help=True,
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.command("server")
def mcp_server(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport type: stdio or http."),
    port: int = typer.Option(8338, "--port", "-p", help="Port for HTTP transport."),
) -> None:
    """Start the MCP server to expose Animus tools."""
    import asyncio
    from src.mcp import MCPServer

    server = MCPServer(name="animus", version="1.0.0")
    server.register_animus_tools()

    tools = list(server._tools.keys())
    console.print(f"[bold blue]Animus MCP Server[/bold blue]")
    console.print(f"Transport: [cyan]{transport}[/cyan]")
    console.print(f"Tools: [cyan]{len(tools)}[/cyan]")

    if transport == "stdio":
        console.print("\n[dim]Reading from stdin, writing to stdout...[/dim]")
        asyncio.run(server.run_stdio())
    elif transport == "http":
        console.print(f"Running on [cyan]http://localhost:{port}[/cyan]")
        asyncio.run(server.run_http(port=port))
    else:
        console.print(f"[red]Unknown transport: {transport}[/red]")
        raise typer.Exit(1)


@mcp_app.command("tools")
def mcp_tools() -> None:
    """List tools that would be exposed via MCP."""
    from src.mcp import MCPServer

    server = MCPServer()
    server.register_animus_tools()

    table = Table(title="MCP Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    for name, tool in server._tools.items():
        desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
        table.add_row(name, desc)

    console.print(table)
    console.print(f"\n[dim]{len(server._tools)} tool(s) available[/dim]")


@mcp_app.command("list")
def mcp_list(
    connect: bool = typer.Option(
        False,
        "--connect",
        "-c",
        help="Test connection to each server.",
    ),
) -> None:
    """List configured MCP servers and their status."""
    import asyncio
    from src.core.config import ConfigManager
    from src.mcp.client import MCPClient, MCPServerConfig

    config = ConfigManager().config
    servers = config.mcp.servers

    if not servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        console.print("\nTo add an MCP server, edit [cyan]~/.animus/config.yaml[/cyan]:")
        console.print("""
[dim]mcp:
  servers:
    - name: my-server
      command: npx
      args: ["-y", "@my/mcp-server"]
      enabled: true[/dim]
""")
        return

    table = Table(title="Configured MCP Servers")
    table.add_column("Name", style="cyan")
    table.add_column("Transport", style="white")
    table.add_column("Enabled", style="white")
    if connect:
        table.add_column("Status", style="white")
        table.add_column("Tools", style="white")

    async def check_servers():
        client = MCPClient()
        results = []

        for server in servers:
            transport = "stdio" if server.command else "http" if server.url else "unknown"
            enabled = "[green]Yes[/green]" if server.enabled else "[dim]No[/dim]"

            if connect and server.enabled:
                try:
                    server_config = MCPServerConfig(
                        name=server.name,
                        command=server.command,
                        args=server.args,
                        url=server.url,
                        env=server.env,
                        enabled=server.enabled,
                    )
                    connected = await client.connect(server_config)
                    status = "[green]Connected[/green]"
                    tool_count = str(len(connected.tools))
                    results.append((server.name, transport, enabled, status, tool_count))
                except Exception as e:
                    status = f"[red]Failed[/red]"
                    results.append((server.name, transport, enabled, status, "-"))
            else:
                if connect:
                    results.append((server.name, transport, enabled, "[dim]Skipped[/dim]", "-"))
                else:
                    results.append((server.name, transport, enabled))

        client.disconnect_all()
        return results

    if connect:
        with console.status("[bold blue]Connecting to servers..."):
            results = asyncio.run(check_servers())
        for result in results:
            table.add_row(*result)
    else:
        for server in servers:
            transport = "stdio" if server.command else "http" if server.url else "unknown"
            enabled = "[green]Yes[/green]" if server.enabled else "[dim]No[/dim]"
            table.add_row(server.name, transport, enabled)

    console.print(table)
    console.print(f"\n[dim]{len(servers)} server(s) configured[/dim]")
    if not connect:
        console.print("[dim]Use --connect to test connections[/dim]")


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(..., help="Unique name for this server"),
    command: Optional[str] = typer.Option(None, "--command", "-c", help="Command for stdio transport"),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="URL for HTTP transport"),
    args: Optional[str] = typer.Option(None, "--args", "-a", help="Comma-separated args for command"),
) -> None:
    """Add a new MCP server configuration."""
    from src.core.config import ConfigManager, MCPServerEntry

    if not command and not url:
        console.print("[red]Error: Must specify either --command or --url[/red]")
        raise typer.Exit(1)

    manager = ConfigManager()
    config = manager.config

    # Check for duplicate name
    for server in config.mcp.servers:
        if server.name == name:
            console.print(f"[red]Error: Server '{name}' already exists[/red]")
            raise typer.Exit(1)

    # Create new server entry
    server = MCPServerEntry(
        name=name,
        command=command,
        url=url,
        args=args.split(",") if args else [],
        enabled=True,
    )

    config.mcp.servers.append(server)
    manager.save(config)

    console.print(f"[green]Added MCP server: {name}[/green]")
    transport = "stdio" if command else "http"
    console.print(f"Transport: [cyan]{transport}[/cyan]")


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Name of server to remove"),
) -> None:
    """Remove an MCP server configuration."""
    from src.core.config import ConfigManager

    manager = ConfigManager()
    config = manager.config

    # Find and remove
    for i, server in enumerate(config.mcp.servers):
        if server.name == name:
            config.mcp.servers.pop(i)
            manager.save(config)
            console.print(f"[green]Removed MCP server: {name}[/green]")
            return

    console.print(f"[red]Error: Server '{name}' not found[/red]")
    raise typer.Exit(1)


@app.command("serve")
def serve(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to."),
    port: int = typer.Option(8337, "--port", "-p", help="Port to listen on."),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for authentication."),
) -> None:
    """Start the OpenAI-compatible API server."""
    from src.api import create_app
    from http.server import HTTPServer

    handler = create_app(api_key)
    server = HTTPServer((host, port), handler)

    console.print(f"[bold magenta]✦ Animus Manifests ✦[/bold magenta]")
    console.print(f"Running on [cyan]http://{host}:{port}[/cyan]")
    console.print()
    console.print("[dim]Endpoints:[/dim]")
    console.print("  GET  /v1/models")
    console.print("  POST /v1/chat/completions")
    console.print("  POST /v1/embeddings")
    console.print("  POST /v1/agent/chat")
    console.print("  POST /v1/agent/search")
    console.print("  GET  /health")
    console.print()
    console.print("Press [yellow]Ctrl+C[/yellow] to stop")
    console.print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")
        server.shutdown()


@app.command("ide")
def ide_server(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to."),
    port: int = typer.Option(8765, "--port", "-p", help="Port to listen on."),
) -> None:
    """Start the WebSocket server for IDE integration (VSCode extension)."""
    import asyncio

    try:
        from src.api.websocket_server import run_websocket_server
    except ImportError:
        console.print("[red]websockets package not installed.[/red]")
        console.print("Install with: [cyan]pip install websockets[/cyan]")
        raise typer.Exit(1)

    console.print(f"[bold magenta]✦ Animus IDE Server ✦[/bold magenta]")
    console.print(f"WebSocket server on [cyan]ws://{host}:{port}[/cyan]")
    console.print()
    console.print("[dim]Connect your VSCode extension to this server.[/dim]")
    console.print("[dim]Configure the URL in VSCode settings: animus.serverUrl[/dim]")
    console.print()
    console.print("Press [yellow]Ctrl+C[/yellow] to stop")
    console.print()

    try:
        asyncio.run(run_websocket_server(host=host, port=port))
    except KeyboardInterrupt:
        console.print("\n[dim]Shutting down...[/dim]")


@app.command("speak")
def speak_command(
    off: bool = typer.Option(
        False,
        "--off",
        help="Disable Animus voice synthesis.",
    ),
) -> None:
    """
    Toggle Animus voice synthesis.

    When enabled, Animus will speak phrases like "Yes, Master" and "It will be done"
    with a low, square-wave, robotic voice.
    """
    manager = ConfigManager()
    config = manager.config

    if off:
        config.audio.speak_enabled = False
        manager.save()
        console.print("[yellow]Animus voice disabled.[/yellow]")
    else:
        config.audio.speak_enabled = True
        manager.save()
        console.print("[green]Animus voice enabled.[/green]")
        console.print("[dim]Animus voice synthesis enabled.[/dim]")


@app.command("praise")
def praise_command(
    fanfare: bool = typer.Option(
        False,
        "--fanfare",
        help="Enable Mozart fanfare on task completion.",
    ),
    sophisticated: bool = typer.Option(
        False,
        "--sophisticated",
        help="Enable Bach Invention 13 on task completion.",
    ),
    moto: bool = typer.Option(
        False,
        "--moto",
        help="Enable Paganini Moto Perpetuo background music during execution.",
    ),
    motoff: bool = typer.Option(
        False,
        "--motoff",
        help="Disable Moto Perpetuo background music.",
    ),
    off: bool = typer.Option(
        False,
        "--off",
        help="Disable all praise audio.",
    ),
) -> None:
    """
    Configure task completion audio and background music.

    Modes:
      --fanfare: Play Mozart's "Eine kleine Nachtmusik" on completion
      --sophisticated: Play Bach's Invention 13 on completion
      --moto:    Play Paganini's "Moto Perpetuo" during task execution
      --motoff:  Disable Moto Perpetuo background music
      --off:     Disable all praise audio
    """
    manager = ConfigManager()
    config = manager.config

    if off:
        config.audio.praise_mode = "off"
        config.audio.moto_enabled = False
        manager.save()
        console.print("[yellow]All praise audio disabled.[/yellow]")
        return

    if fanfare:
        config.audio.praise_mode = "fanfare"
        manager.save()
        console.print("[green]Mozart fanfare enabled for task completion.[/green]")
        console.print("[dim]Eine kleine Nachtmusik will play when tasks complete.[/dim]")

    if sophisticated:
        config.audio.praise_mode = "sophisticated"
        manager.save()
        console.print("[green]Bach sophisticated mode enabled for task completion.[/green]")
        console.print("[dim]Invention 13 will play when tasks complete.[/dim]")

    if moto:
        config.audio.moto_enabled = True
        manager.save()
        console.print("[green]Moto Perpetuo background music enabled.[/green]")
        console.print("[dim]Paganini's Moto Perpetuo will play during task execution.[/dim]")

    if motoff:
        config.audio.moto_enabled = False
        manager.save()
        console.print("[yellow]Moto Perpetuo background music disabled.[/yellow]")

    # Show current settings if no options provided
    if not any([fanfare, sophisticated, moto, motoff, off]):
        console.print(Panel(
            f"[bold]Current Audio Settings[/bold]\n\n"
            f"Praise Mode: [cyan]{config.audio.praise_mode}[/cyan]\n"
            f"Moto Perpetuo: [cyan]{'enabled' if config.audio.moto_enabled else 'disabled'}[/cyan]\n"
            f"Volume: [cyan]{config.audio.volume:.1f}[/cyan]",
            title="Audio Configuration",
            border_style="blue"
        ))


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

@app.command("chat", hidden=True)
def _chat_alias(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    no_confirm: bool = typer.Option(False, "--no-confirm"),
    max_context: Optional[int] = typer.Option(None, "--max-context", "-c"),
    show_tokens: bool = typer.Option(False, "--show-tokens"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    plan: bool = typer.Option(False, "--plan", "-p"),
    delegate: bool = typer.Option(False, "--delegate", "-d"),
) -> None:
    """Alias for 'rise'."""
    chat(model, no_confirm, max_context, show_tokens, stream, plan, delegate)


@app.command("download", hidden=True)
def _download_alias(
    model_name: str = typer.Argument(...),
) -> None:
    """Alias for 'pull'."""
    pull_model(model_name)


if __name__ == "__main__":
    app()
