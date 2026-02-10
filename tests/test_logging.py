"""Tests for structured logging and token tracking."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.context import TokenUsage
from src.core.logging import (
    get_logger,
    log_error,
    log_llm_call,
    log_tool_execution,
    reset_logger,
    setup_logging,
)


class TestSetupLogging:
    def setup_method(self):
        reset_logger()

    def teardown_method(self):
        reset_logger()

    def test_setup_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "animus"

    def test_setup_default_level(self):
        logger = setup_logging(log_level="INFO")
        assert logger.level == logging.INFO

    def test_setup_debug_level(self):
        logger = setup_logging(log_level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_setup_with_file_handler(self, tmp_path: Path):
        logs_dir = tmp_path / "logs"
        logger = setup_logging(logs_dir=logs_dir)
        assert logs_dir.exists()
        # Should have console + file handler
        assert len(logger.handlers) == 2

    def test_setup_without_file_handler(self):
        logger = setup_logging()
        # Console handler only
        assert len(logger.handlers) == 1

    def test_setup_idempotent(self):
        logger1 = setup_logging()
        logger2 = setup_logging()
        assert logger1 is logger2

    def test_get_logger_auto_setup(self):
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    def test_reset_logger(self):
        setup_logging()
        reset_logger()
        # After reset, get_logger should create a new one
        logger = get_logger()
        assert isinstance(logger, logging.Logger)


class TestLogFunctions:
    def setup_method(self):
        reset_logger()

    def teardown_method(self):
        reset_logger()

    def test_log_llm_call(self, tmp_path: Path):
        logs_dir = tmp_path / "logs"
        setup_logging(log_level="DEBUG", logs_dir=logs_dir)
        log_llm_call(provider="openai", token_estimate=100, latency_ms=250.5)
        # Read the log file
        log_file = logs_dir / "animus.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "LLM call" in content
        assert "openai" in content

    def test_log_llm_call_with_error(self, tmp_path: Path):
        logs_dir = tmp_path / "logs"
        setup_logging(log_level="DEBUG", logs_dir=logs_dir)
        log_llm_call(provider="anthropic", token_estimate=50, latency_ms=100, error="timeout")
        content = (logs_dir / "animus.log").read_text()
        assert "ERROR" in content
        assert "timeout" in content

    def test_log_tool_execution(self, tmp_path: Path):
        logs_dir = tmp_path / "logs"
        setup_logging(log_level="DEBUG", logs_dir=logs_dir)
        log_tool_execution(tool_name="read_file", args_summary="path=/tmp/test", duration_ms=15.3)
        content = (logs_dir / "animus.log").read_text()
        assert "Tool exec" in content
        assert "read_file" in content

    def test_log_tool_execution_with_error(self, tmp_path: Path):
        logs_dir = tmp_path / "logs"
        setup_logging(log_level="DEBUG", logs_dir=logs_dir)
        log_tool_execution(tool_name="write_file", args_summary="path=/bad", duration_ms=5, error="denied")
        content = (logs_dir / "animus.log").read_text()
        assert "denied" in content

    def test_log_error(self, tmp_path: Path):
        logs_dir = tmp_path / "logs"
        setup_logging(log_level="DEBUG", logs_dir=logs_dir)
        log_error("NETWORK", "Connection refused", host="api.openai.com")
        content = (logs_dir / "animus.log").read_text()
        assert "NETWORK" in content
        assert "Connection refused" in content


class TestTokenUsage:
    def test_initial_state(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total == 0

    def test_add_input(self):
        usage = TokenUsage()
        usage.add_input(100)
        assert usage.input_tokens == 100
        assert usage.total == 100

    def test_add_output(self):
        usage = TokenUsage()
        usage.add_output(50)
        assert usage.output_tokens == 50
        assert usage.total == 50

    def test_cumulative(self):
        usage = TokenUsage()
        usage.add_input(100)
        usage.add_output(50)
        usage.add_input(200)
        usage.add_output(75)
        assert usage.input_tokens == 300
        assert usage.output_tokens == 125
        assert usage.total == 425
