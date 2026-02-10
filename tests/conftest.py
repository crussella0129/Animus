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


SAMPLE_PYTHON_SOURCE = '''\
"""Sample module docstring."""

import os
from pathlib import Path


class Animal:
    """Base animal class."""

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return ""


class Dog(Animal):
    """A dog."""

    def speak(self) -> str:
        return f"{self.name} says woof"

    def fetch(self, item: str) -> str:
        print(f"Fetching {item}")
        return item


class Cat(Animal):
    """A cat."""

    def speak(self) -> str:
        return f"{self.name} says meow"


def make_animal(kind: str) -> Animal:
    """Factory function."""
    if kind == "dog":
        return Dog(kind)
    return Cat(kind)


def main():
    animal = make_animal("dog")
    animal.speak()
'''


@pytest.fixture
def sample_python_source() -> str:
    """Return sample Python source code for parser testing."""
    return SAMPLE_PYTHON_SOURCE


@pytest.fixture
def sample_python_file(tmp_path: Path, sample_python_source: str) -> Path:
    """Write sample Python source to a temp file and return its path."""
    # Put it in a src/ dir so the module name derivation works
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    f = src_dir / "animals.py"
    f.write_text(sample_python_source)
    return f


@pytest.fixture
def graph_db(tmp_path: Path):
    """Create a temporary GraphDB instance."""
    from src.knowledge.graph_db import GraphDB
    db = GraphDB(tmp_path / "test_graph.db")
    yield db
    db.close()


@pytest.fixture
def sqlite_vector_store(tmp_path: Path):
    """Create a temporary SQLiteVectorStore instance."""
    from src.memory.vectorstore import SQLiteVectorStore
    store = SQLiteVectorStore(tmp_path / "test_vectors.db")
    yield store
    store.close()
