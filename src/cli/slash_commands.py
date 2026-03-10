"""Slash command handler for the Animus interactive session."""

from __future__ import annotations

from rich.table import Table

from src.core.config import AnimusConfig
from src.ui import console, info, success, warn


# Mutable state for /plan toggle (module-level to avoid closure issues)
_plan_mode_state: dict[str, bool] = {"active": False}


def handle_slash_command(command: str, agent, session, cfg: AnimusConfig) -> bool:
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
            table.add_column("Isolation", style="yellow")

            for tool in tools:
                isolation = tool.isolation_level
                isolation_display = {
                    'none': '[dim]none[/]',
                    'ornstein': '[yellow]ornstein[/]',
                    'smough': '[red]smough[/]',
                }.get(isolation, isolation)

                table.add_row(tool.name, tool.description, isolation_display)

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
