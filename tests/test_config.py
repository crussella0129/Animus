"""Tests for configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core.config import AnimusConfig, ModelConfig, RAGConfig


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.provider == "native"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
        assert cfg.size_tier == "auto"


class TestRAGConfig:
    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 64
        assert cfg.top_k == 5


class TestAnimusConfig:
    def test_defaults(self):
        cfg = AnimusConfig()
        assert cfg.log_level == "INFO"
        assert cfg.model.provider == "native"

    def test_save_and_load(self, tmp_config_dir: Path):
        cfg = AnimusConfig(config_dir=tmp_config_dir)
        cfg.model.model_name = "test-model"
        cfg.save()

        assert cfg.config_file.exists()

        loaded = AnimusConfig.load(tmp_config_dir)
        assert loaded.model.model_name == "test-model"

    def test_load_missing_file(self, tmp_path: Path):
        cfg = AnimusConfig.load(tmp_path / "nonexistent")
        assert cfg.model.provider == "native"

    def test_config_file_path(self, tmp_config_dir: Path):
        cfg = AnimusConfig(config_dir=tmp_config_dir)
        assert cfg.config_file == tmp_config_dir / "config.yaml"

    @pytest.mark.skipif(os.name == "nt", reason="chmod 600 is a no-op on Windows")
    def test_save_sets_restrictive_permissions(self, tmp_config_dir: Path):
        cfg = AnimusConfig(config_dir=tmp_config_dir)
        cfg.save()
        mode = cfg.config_file.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"
