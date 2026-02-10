"""Shared test fixtures and pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Provide a temporary config directory."""
    config_dir = tmp_path / ".animus"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_files(tmp_path: Path) -> Path:
    """Create sample files for testing."""
    (tmp_path / "hello.py").write_text("def hello():\n    return 'world'\n")
    (tmp_path / "readme.md").write_text("# Test Project\nThis is a test.\n")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "data.txt").write_text("Some data content here.\n")
    return tmp_path


@pytest.fixture
def mock_llama_cpp():
    """Mock llama-cpp-python to prevent real model loading."""
    mock = MagicMock()
    mock.Llama = MagicMock()
    sys.modules["llama_cpp"] = mock
    yield mock
    sys.modules.pop("llama_cpp", None)


@pytest.fixture
def mock_sentence_transformers():
    """Mock sentence-transformers to prevent real model loading."""
    mock = MagicMock()
    mock.SentenceTransformer = MagicMock()
    sys.modules["sentence_transformers"] = mock
    yield mock
    sys.modules.pop("sentence_transformers", None)
