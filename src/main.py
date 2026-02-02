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

    # Adjust defaults based on detected hardware and available providers
    from src.llm import LLAMA_CPP_AVAILABLE

    config = manager.config
    if info.hardware_type == HardwareType.JETSON:
        config.model.provider = "trtllm"
        console.print("[green]Jetson detected - configured for TensorRT-LLM[/green]")
    elif LLAMA_CPP_AVAILABLE:
        # Native provider available - use it for full independence
        config.model.provider = "native"
        if info.hardware_type == HardwareType.APPLE_SILICON:
            console.print("[green]Apple Silicon detected - configured for native provider with Metal[/green]")
        elif info.gpu and info.gpu.cuda_available:
            console.print("[green]NVIDIA GPU detected - configured for native provider with CUDA[/green]")
        else:
            console.print("[green]Configured for native provider (CPU)[/green]")
    else:
        console.print("[yellow]Note: llama-cpp-python not installed. Install for local inference.[/yellow]")

    manager.save(config)

    console.print(f"\n[bold green]Animus initialized![/bold green]")
    console.print(f"Config: [cyan]{manager.config_path}[/cyan]")
    console.print(f"Data:   [cyan]{config.data_dir}[/cyan]")
    console.print(f"\nNext steps:")
    console.print("  1. Run [cyan]animus sense[/cyan] to verify your environment")
    console.print("  2. Run [cyan]animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF[/cyan] to download a model")
    console.print("  3. Run [cyan]animus rise[/cyan] to start chatting")


@app.command("models")
def models() -> None:
    """List available local GGUF models."""
    import asyncio
    from src.llm import NativeProvider, LLAMA_CPP_AVAILABLE
    from src.core.config import ConfigManager

    async def list_models() -> None:
        config = ConfigManager().config
        provider = NativeProvider(models_dir=config.native.models_dir)

        console.print(f"[bold]Local Models[/bold]")
        console.print(f"[dim]Directory: {config.native.models_dir}[/dim]\n")

        model_list = await provider.list_models()

        if not model_list:
            console.print("[yellow]No local models found.[/yellow]")
            console.print("\nDownload a model with:")
            console.print("  [cyan]animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF[/cyan]")
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

        # Show llama-cpp-python status
        console.print()
        if LLAMA_CPP_AVAILABLE:
            gpu_backend = provider.gpu_backend
            console.print(f"[green]llama-cpp-python:[/green] Installed ({gpu_backend})")
        else:
            console.print("[yellow]llama-cpp-python:[/yellow] Not installed")
            console.print("Install with: [cyan]pip install llama-cpp-python[/cyan]")

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
    from src.llm import NativeProvider, HF_HUB_AVAILABLE
    from src.core.config import ConfigManager

    if not HF_HUB_AVAILABLE:
        console.print("[red]huggingface-hub not installed.[/red]")
        console.print("Install with: [cyan]pip install huggingface-hub[/cyan]")
        raise typer.Exit(1)

    async def download() -> None:
        config = ConfigManager().config
        provider = NativeProvider(models_dir=config.native.models_dir)

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
    from src.llm import NativeProvider, HF_HUB_AVAILABLE
    from src.core.config import ConfigManager

    if not HF_HUB_AVAILABLE:
        console.print("[red]huggingface-hub not installed.[/red]")
        console.print("Install with: [cyan]pip install huggingface-hub[/cyan]")
        raise typer.Exit(1)

    async def download() -> None:
        config = ConfigManager().config
        provider = NativeProvider(models_dir=config.native.models_dir)

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
    from src.llm import NativeProvider, LLAMA_CPP_AVAILABLE
    from src.core.config import ConfigManager

    async def list_local() -> None:
        config = ConfigManager().config
        provider = NativeProvider(models_dir=config.native.models_dir)

        console.print(f"[bold]Local Models[/bold]")
        console.print(f"[dim]Directory: {config.native.models_dir}[/dim]\n")

        models = await provider.list_models()

        if not models:
            console.print("[yellow]No local models found.[/yellow]")
            console.print("\nDownload a model with:")
            console.print("  [cyan]animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF[/cyan]")
            return

        table = Table(title="Local GGUF Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Quantization", style="yellow")

        for model in models:
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
                model.quantization or "Unknown",
            )

        console.print(table)

        # Show llama-cpp-python status
        console.print()
        if LLAMA_CPP_AVAILABLE:
            gpu_backend = provider.gpu_backend
            console.print(f"[green]llama-cpp-python:[/green] Installed")
            console.print(f"[green]GPU Backend:[/green] {gpu_backend}")
        else:
            console.print("[yellow]llama-cpp-python:[/yellow] Not installed")
            console.print("Install with: [cyan]pip install llama-cpp-python[/cyan]")

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
    from src.llm import NativeProvider, LLAMA_CPP_AVAILABLE
    from src.llm.native import detect_quantization
    from src.core.config import ConfigManager

    config = ConfigManager().config
    provider = NativeProvider(models_dir=config.native.models_dir)

    model_path = provider._get_model_path(model_name)
    if not model_path:
        console.print(f"[red]Model not found:[/red] {model_name}")
        raise typer.Exit(1)

    stat = model_path.stat()
    quant = detect_quantization(model_path.name)

    console.print(f"[bold]Model Information[/bold]\n")
    console.print(f"  Name: [cyan]{model_path.name}[/cyan]")
    console.print(f"  Path: {model_path}")
    console.print(f"  Size: {stat.st_size / (1024**3):.2f} GB")
    console.print(f"  Quantization: {quant or 'Unknown'}")

    if LLAMA_CPP_AVAILABLE:
        console.print(f"\n[green]Ready for native inference[/green]")
        console.print(f"  GPU Backend: {provider.gpu_backend}")
    else:
        console.print(f"\n[yellow]Install llama-cpp-python to use this model[/yellow]")


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
    from src.llm import NativeProvider, TRTLLMProvider, APIProvider, LLAMA_CPP_AVAILABLE
    from src.core.config import ConfigManager

    async def check_status() -> None:
        config = ConfigManager().config

        console.print("[bold]Animus Status[/bold]\n")

        # Show configured provider and model
        console.print(f"Configured Provider: [cyan]{config.model.provider}[/cyan]")
        console.print(f"Configured Model: [cyan]{config.model.model_name or '(auto-detect)'}[/cyan]")
        console.print()

        # Check Native (llama-cpp-python)
        native = NativeProvider(models_dir=config.native.models_dir)
        if LLAMA_CPP_AVAILABLE:
            models = await native.list_models()
            native_status = f"[green]Available[/green] ({native.gpu_backend})"
            console.print(f"Native (llama-cpp-python): {native_status}")
            if models:
                console.print(f"  Local models: {len(models)}")
                for m in models[:3]:
                    console.print(f"    - {m.name}")
                if len(models) > 3:
                    console.print(f"    ... and {len(models) - 3} more")
            else:
                console.print("  [yellow]No local models found[/yellow]")
                console.print("  Download with: [cyan]animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF[/cyan]")
        else:
            console.print("Native (llama-cpp-python): [dim]Not Installed[/dim]")
            console.print("  Install with: [cyan]pip install llama-cpp-python[/cyan]")

        # Check TensorRT-LLM (Jetson)
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
        config = ConfigManager().config
        provider = get_default_provider(config)
        model_name = model or config.model.model_name

        if not provider.is_available:
            console.print("[red]Provider not available.[/red]")
            console.print("llama-cpp-python not available.\n")
            console.print("Install with: [cyan]pip install llama-cpp-python[/cyan]")
            console.print("Or configure an API in [cyan]~/.animus/config.yaml[/cyan]")
            raise typer.Exit(1)

        # For native provider, check if we have any models
        if config.model.provider == "native":
            from src.llm.native import NativeProvider
            native = provider if isinstance(provider, NativeProvider) else None
            if native:
                # If specific model requested, check if it exists
                if model_name:
                    model_path = native._get_model_path(model_name)
                    if not model_path:
                        console.print("[yellow]Model not found.[/yellow]")
                        console.print(f"Model [cyan]{model_name}[/cyan] not found.\n")
                        console.print("Download a GGUF model:")
                        console.print("  [cyan]animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF[/cyan]\n")
                        console.print("Or list available models:")
                        console.print("  [cyan]animus vessels[/cyan]")
                        raise typer.Exit(1)
                else:
                    # No model specified, check if any exist
                    models = await native.list_models()
                    if not models:
                        console.print("[yellow]No models found.[/yellow]")
                        console.print("No local models found.\n")
                        console.print("Download a model:")
                        console.print("  [cyan]animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF[/cyan]")
                        raise typer.Exit(1)
                    # Auto-select first model
                    model_name = models[0].name
                    console.print(f"[dim]Using model: {model_name}[/dim]")

        async def confirm_tool(tool_name: str, description: str) -> bool:
            if no_confirm:
                return True
            console.print(f"\n[yellow]Tool request:[/yellow] {tool_name}")
            console.print(f"[dim]{description}[/dim]")
            return Confirm.ask("Execute this tool?", default=True)

        agent_config = AgentConfig(
            model=model_name,
            temperature=config.model.temperature,
            require_tool_confirmation=not no_confirm,
        )

        agent = Agent(
            provider=provider,
            config=agent_config,
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
        if max_context:
            console.print(f"[dim]Max context: {max_context} tokens[/dim]")
        console.print("Speak your command. Say [cyan]farewell[/cyan] to end.\n")

        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")

                if user_input.lower() in ("exit", "quit", "q", "farewell", "dismiss"):
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


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

@app.command("chat", hidden=True)
def _chat_alias(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    no_confirm: bool = typer.Option(False, "--no-confirm"),
    max_context: Optional[int] = typer.Option(None, "--max-context", "-c"),
    show_tokens: bool = typer.Option(False, "--show-tokens"),
) -> None:
    """Alias for 'rise'."""
    chat(model, no_confirm, max_context, show_tokens)


@app.command("download", hidden=True)
def _download_alias(
    model_name: str = typer.Argument(...),
) -> None:
    """Alias for 'pull'."""
    pull_model(model_name)


if __name__ == "__main__":
    app()
