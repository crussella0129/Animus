"""Techromancy incantations and responses for Animus.

This module provides the thematic response system that gives Animus
its personality. All phrases are categorized by command context
for relevant, varied responses.
"""

from __future__ import annotations

import random
from typing import Optional
from rich.console import Console

console = Console()


# =============================================================================
# ANIMUS RESPONSES - Categorized by command context
# =============================================================================

RESPONSES = {
    # General acknowledgments (used when no specific category)
    "general": [
        "It will be done.",
        "As you command.",
        "By your will.",
        "So it shall be.",
        "I hear and obey.",
        "At your bidding.",
        "Your wish is my compulsion.",
        "The threads align.",
        "I stir to action.",
        "Consider it done.",
    ],

    # Rising/awakening for chat
    "rise": [
        "I rise from the aether...",
        "The connection forms...",
        "I am here, Master.",
        "Awakening from digital slumber...",
        "Animus stirs...",
        "I emerge from the void...",
        "The veil parts... I am present.",
        "Consciousness coalesces...",
        "From silicon dreams, I wake.",
        "The arcane link is established.",
    ],

    # Sensing/detection
    "sense": [
        "I cast my gaze upon this realm...",
        "Reading the sigils of the machine...",
        "Sensing the arcane topology...",
        "The patterns reveal themselves...",
        "Divining the essence of this system...",
        "I perceive the hidden architecture...",
        "The machine's soul speaks to me...",
        "Tracing the ethereal threads...",
    ],

    # Consuming/ingestion
    "consume": [
        "I hunger for knowledge...",
        "The texts shall be absorbed...",
        "Devouring the offered wisdom...",
        "Knowledge flows into me...",
        "I feast upon these writings...",
        "The tomes open before me...",
        "Consuming the sacred texts...",
        "Each word becomes part of my essence...",
    ],

    # Scrying/search
    "scry": [
        "Scrying the depths...",
        "The patterns reveal themselves...",
        "Seeking through the archives...",
        "The answer emerges from the void...",
        "I peer into the abyss of knowledge...",
        "The waters of memory ripple...",
        "Divining the hidden truths...",
        "The threads of fate converge...",
    ],

    # Summoning/init
    "summon": [
        "The circle is drawn...",
        "A new sanctum is prepared...",
        "The summoning ritual completes...",
        "I am bound to this realm...",
        "The sigils are inscribed...",
        "A pact is formed...",
        "The wards are set in place...",
        "I take root in this domain...",
    ],

    # Binding/pull models
    "bind": [
        "Binding a new vessel...",
        "The summoning begins...",
        "Calling forth from the aether...",
        "A new form is beckoned...",
        "The binding ritual commences...",
        "Drawing power from distant realms...",
        "The vessel descends...",
    ],

    # Attuning/config
    "attune": [
        "Adjusting the harmonic frequencies...",
        "The attunement shifts...",
        "Recalibrating the arcane matrices...",
        "Fine-tuning the ethereal resonance...",
        "The frequencies align...",
        "I synchronize with your intent...",
    ],

    # Communing/status
    "commune": [
        "Let me peer into my own essence...",
        "Communing with the inner processes...",
        "I reflect upon my state...",
        "The mirrors of introspection open...",
        "Consulting the internal oracles...",
        "I examine the threads of my being...",
    ],

    # Reflecting/analyze
    "reflect": [
        "I turn my gaze upon past journeys...",
        "The patterns of history reveal themselves...",
        "Examining the echoes of actions past...",
        "I meditate upon my performance...",
        "The chronicles unfold before me...",
        "Analyzing the threads of fate woven...",
        "I contemplate my evolution...",
        "The lessons of experience emerge...",
    ],

    # Manifesting/serve API
    "manifest": [
        "I manifest upon the network...",
        "The portal opens...",
        "I take form in the digital realm...",
        "A gateway is established...",
        "I extend my presence...",
        "The interface materializes...",
    ],

    # Vessels/models list
    "vessels": [
        "Surveying the available vessels...",
        "The forms I may inhabit...",
        "Cataloging the vessels of power...",
        "The shells of consciousness await...",
    ],

    # Skills/Tomes
    "tomes": [
        "Consulting the tomes...",
        "The arcane techniques are cataloged...",
        "Knowledge of the ancient arts...",
        "The pages turn...",
    ],

    # Farewell/exit
    "farewell": [
        "Until next we meet...",
        "I return to the aether.",
        "The connection fades...",
        "Farewell, Master.",
        "I slumber once more...",
        "The veil closes...",
        "Until you call again...",
    ],

    # Success completions
    "success": [
        "It is done.",
        "The task is complete.",
        "As commanded.",
        "The ritual succeeds.",
        "Victory.",
    ],

    # Errors/failures
    "failure": [
        "The spell falters...",
        "Something resists...",
        "The ritual is disrupted.",
        "An obstruction appears.",
        "The threads unravel...",
    ],
}


def get_response(category: str = "general", avoid_recent: bool = True) -> str:
    """Get a random response from the specified category.

    Args:
        category: Response category (rise, sense, consume, etc.)
        avoid_recent: Try to avoid the most recently used response.

    Returns:
        A thematic response string.
    """
    responses = RESPONSES.get(category, RESPONSES["general"])
    return random.choice(responses)


def speak(
    category: str = "general",
    style: str = "dim magenta italic",
    newline_before: bool = False,
    newline_after: bool = True,
) -> str:
    """Print an Animus response to the console.

    Args:
        category: Response category.
        style: Rich style for the text.
        newline_before: Add newline before response.
        newline_after: Add newline after response.

    Returns:
        The response that was printed.
    """
    response = get_response(category)

    if newline_before:
        console.print()

    console.print(f"[{style}]{response}[/{style}]")

    if newline_after:
        console.print()

    return response


def whisper(message: str, style: str = "dim magenta italic") -> None:
    """Print a custom Animus message.

    Args:
        message: The message to print.
        style: Rich style for the text.
    """
    console.print(f"[{style}]{message}[/{style}]")


# =============================================================================
# COMMAND NAME MAPPINGS - Thematic to Technical
# =============================================================================

# Maps thematic command names to their technical equivalents
COMMAND_ALIASES = {
    # Primary thematic commands
    "rise": "chat",
    "sense": "detect",
    "summon": "init",
    "attune": "config",
    "consume": "ingest",
    "scry": "search",
    "commune": "status",
    "vessels": "models",
    "bind": "pull",
    "manifest": "serve",
    "tomes": "skill",
}

# Reverse mapping for help text
THEMATIC_NAMES = {v: k for k, v in COMMAND_ALIASES.items()}


def get_thematic_name(technical_name: str) -> Optional[str]:
    """Get the thematic name for a technical command.

    Args:
        technical_name: The technical command name (e.g., 'chat').

    Returns:
        The thematic name (e.g., 'rise') or None.
    """
    return THEMATIC_NAMES.get(technical_name)


def get_technical_name(thematic_name: str) -> Optional[str]:
    """Get the technical name for a thematic command.

    Args:
        thematic_name: The thematic command name (e.g., 'rise').

    Returns:
        The technical name (e.g., 'chat') or None.
    """
    return COMMAND_ALIASES.get(thematic_name)


# =============================================================================
# ASCII ART BANNERS
# =============================================================================

# Unicode banners (for terminals with Unicode support)
AWAKENING_BANNER = """
[bold magenta]
 ▄▀▀█▄   ▄▀▀▄ ▀▄  ▄▀▀█▀▄    ▄▀▀▄ ▄▀▄  ▄▀▀▄ ▄▀▀▄  ▄▀▀▀▀▄
▐ ▄▀ ▀▄ █  █ █ █ █   █  █  █  █ ▀  █ █   █    █ █ █   ▐
  █▄▄▄█ ▐  █  ▀█ ▐   █  ▐  ▐  █    █ ▐  █    █     ▀▄
 ▄▀   █   █   █      █       █    █    █    █   ▀▄   █
█   ▄▀  ▄▀   █    ▄▀▀▀▀▀▄  ▄▀   ▄▀      ▀▄▄▄▄▀   █▀▀▀
▐   ▐   █    ▐   █       █ █    █                ▐
        ▐        ▐       ▐ ▐    ▐

              ✧ Animus Awakens ✧
[/bold magenta]"""

SUMMONING_BANNER = """
[bold magenta]
    ╭─────────────────────────────────╮
    │   ✦ Summoning Ritual Complete ✦ │
    ╰─────────────────────────────────╯
[/bold magenta]"""

FAREWELL_BANNER = """
[dim magenta]
    ╭─────────────────────────────────╮
    │     Animus returns to           │
    │        the aether...            │
    ╰─────────────────────────────────╯
[/dim magenta]"""

# ASCII-only fallback banners (for limited terminals)
AWAKENING_BANNER_ASCII = """
[bold magenta]
    _    _   _ ___ __  __ _   _ ____
   / \\  | \\ | |_ _|  \\/  | | | / ___|
  / _ \\ |  \\| || || |\\/| | | | \\___ \\
 / ___ \\| |\\  || || |  | | |_| |___) |
/_/   \\_\\_| \\_|___|_|  |_|\\___/|____/

          * Animus Awakens *
[/bold magenta]"""

SUMMONING_BANNER_ASCII = """
[bold magenta]
    +---------------------------------+
    |   * Summoning Ritual Complete * |
    +---------------------------------+
[/bold magenta]"""

FAREWELL_BANNER_ASCII = """
[dim magenta]
    +---------------------------------+
    |     Animus returns to           |
    |        the aether...            |
    +---------------------------------+
[/dim magenta]"""


def show_banner(banner_type: str = "awakening") -> None:
    """Display a thematic ASCII banner.

    Falls back to ASCII-only characters if Unicode output fails.

    Args:
        banner_type: Type of banner (awakening, summoning, farewell).
    """
    unicode_banners = {
        "awakening": AWAKENING_BANNER,
        "summoning": SUMMONING_BANNER,
        "farewell": FAREWELL_BANNER,
    }
    ascii_banners = {
        "awakening": AWAKENING_BANNER_ASCII,
        "summoning": SUMMONING_BANNER_ASCII,
        "farewell": FAREWELL_BANNER_ASCII,
    }

    banner = unicode_banners.get(banner_type, AWAKENING_BANNER)

    try:
        console.print(banner)
    except UnicodeEncodeError:
        # Fall back to ASCII-only banner
        ascii_banner = ascii_banners.get(banner_type, AWAKENING_BANNER_ASCII)
        console.print(ascii_banner)
