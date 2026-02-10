"""Structured logging for Animus: rotating file handler, LLM/tool call logging."""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


_logger: logging.Logger | None = None


def setup_logging(
    log_level: str = "INFO",
    logs_dir: Path | None = None,
    log_file: str = "animus.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3,
) -> logging.Logger:
    """Configure the Animus logger with optional rotating file handler.

    Returns the configured logger.
    """
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("animus")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (WARNING and above only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            logs_dir / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the Animus logger. Sets up with defaults if not yet configured."""
    global _logger
    if _logger is None:
        return setup_logging()
    return _logger


def reset_logger() -> None:
    """Reset the global logger (for testing)."""
    global _logger
    if _logger is not None:
        _logger.handlers.clear()
        _logger = None


def log_llm_call(
    provider: str,
    token_estimate: int,
    latency_ms: float,
    error: str | None = None,
) -> None:
    """Log an LLM API call."""
    logger = get_logger()
    if error:
        logger.error("LLM call provider=%s tokens=~%d latency=%.0fms error=%s", provider, token_estimate, latency_ms, error)
    else:
        logger.info("LLM call provider=%s tokens=~%d latency=%.0fms", provider, token_estimate, latency_ms)


def log_tool_execution(
    tool_name: str,
    args_summary: str,
    duration_ms: float,
    error: str | None = None,
) -> None:
    """Log a tool execution."""
    logger = get_logger()
    if error:
        logger.error("Tool exec name=%s args=%s duration=%.0fms error=%s", tool_name, args_summary, duration_ms, error)
    else:
        logger.info("Tool exec name=%s args=%s duration=%.0fms", tool_name, args_summary, duration_ms)


def log_error(
    category: str,
    message: str,
    **extra: Any,
) -> None:
    """Log a classified error."""
    logger = get_logger()
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    logger.error("Error category=%s message=%s %s", category, message, extra_str)
