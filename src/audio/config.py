"""
Audio configuration for Animus voice and music features.

Note: AudioConfig is defined in src/core/config.py and imported here.
"""

from enum import Enum


class PraiseMode(str, Enum):
    """Task completion praise modes."""
    FANFARE = "fanfare"  # Mozart tune
    SOPHISTICATED = "sophisticated"  # Bach tune
    OFF = "off"          # Silent
