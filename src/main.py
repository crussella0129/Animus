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


if __name__ == "__main__":
    app()
