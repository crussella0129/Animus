"""Pydantic configuration with YAML persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


def _default_config_dir() -> Path:
    return Path(os.environ.get("ANIMUS_CONFIG_DIR", Path.home() / ".animus"))


class ModelConfig(BaseModel):
    model_config = {"extra": "ignore"}

    provider: str = "native"  # "native" (local GGUF), "openai", or "anthropic"
    model_name: str = ""
    model_path: str = ""  # path to GGUF file for native provider
    temperature: float = 0.7
    max_tokens: int = 2048
    context_length: int = 4096
    gpu_layers: int = -1  # -1 = auto
    # Model size tier: "small" (<4B), "medium" (4-13B), "large" (13B+), "auto"
    size_tier: str = "auto"


class RAGConfig(BaseModel):
    model_config = {"extra": "ignore"}

    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5


class AgentConfig(BaseModel):
    model_config = {"extra": "ignore"}

    max_turns: int = 20
    system_prompt: str = "You are Animus, a helpful local AI assistant with tool use capabilities."
    confirm_dangerous: bool = True


class AnimusConfig(BaseSettings):
    """Main application configuration."""

    config_dir: Path = Field(default_factory=_default_config_dir)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    log_level: str = "INFO"

    model_config = {"env_prefix": "ANIMUS_", "extra": "ignore"}

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def sessions_dir(self) -> Path:
        return self.config_dir / "sessions"

    @property
    def logs_dir(self) -> Path:
        return self.config_dir / "logs"

    @property
    def models_dir(self) -> Path:
        return self.config_dir / "models"

    @property
    def graph_dir(self) -> Path:
        return self.config_dir / "graph"

    @property
    def vector_dir(self) -> Path:
        return self.config_dir / "vectorstore"

    def save(self) -> None:
        """Save configuration to YAML file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(exclude={"config_dir"})
        with open(self.config_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> "AnimusConfig":
        """Load configuration from YAML file, falling back to defaults."""
        base_dir = config_dir or _default_config_dir()
        config_file = base_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, "r") as f:
                data = yaml.safe_load(f) or {}
            data["config_dir"] = base_dir
            return cls(**data)
        return cls(config_dir=base_dir)
