"""Minimal Rich console helpers."""

import io
import sys

from rich.console import Console

# Force UTF-8 on Windows to support the logo's Unicode block characters
if sys.platform == "win32" and not isinstance(sys.stdout, io.TextIOWrapper):
    pass  # non-standard stdout, leave it alone
elif sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

LOGO = r"""
 .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--.
/ .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
\ \/\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ \/ /
 \/ /`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\/ /
 / /\   ▄▀▀█▄   ▄▀▀▄ ▀▄  ▄▀▀█▀▄    ▄▀▀▄ ▄▀▄  ▄▀▀▄ ▄▀▀▄  ▄▀▀▀▀▄   / /\
/ /\ \ ▐ ▄▀ ▀▄ █  █ █ █ █   █  █  █  █ ▀  █ █   █    █ █ █   ▐  / /\ \
\ \/ /   █▄▄▄█ ▐  █  ▀█ ▐   █  ▐  ▐  █    █ ▐  █    █     ▀▄    \ \/ /
 \/ /   ▄▀   █   █   █      █       █    █    █    █   ▀▄   █    \/ /
 / /\  █   ▄▀  ▄▀   █    ▄▀▀▀▀▀▄  ▄▀   ▄▀      ▀▄▄▄▄▀   █▀▀▀     / /\
/ /\ \ ▐   ▐   █    ▐   █       █ █    █                ▐       / /\ \
\ \/ /         ▐        ▐       ▐ ▐    ▐                        \ \/ /
 \/ /                                                            \/ /
 / /\.--..--..--..--..--..--..--..--..--..--..--..--..--..--..--./ /\
/ /\ \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \/\ \
\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
 `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'
"""


def print_logo() -> None:
    """Display the Animus ASCII logo."""
    console.print(LOGO, style="bold cyan", highlight=False)


def info(msg: str) -> None:
    console.print(f"[bold blue]\\[i][/] {msg}")


def success(msg: str) -> None:
    console.print(f"[bold green]\\[+][/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]\\[!][/] {msg}")


def error(msg: str) -> None:
    console.print(f"[bold red]\\[-][/] {msg}")
