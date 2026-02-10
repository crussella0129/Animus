"""Minimal Rich console helpers."""

from rich.console import Console

console = Console()


def info(msg: str) -> None:
    console.print(f"[bold blue]\\[i][/] {msg}")


def success(msg: str) -> None:
    console.print(f"[bold green]\\[+][/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]\\[!][/] {msg}")


def error(msg: str) -> None:
    console.print(f"[bold red]\\[-][/] {msg}")
