"""Automated end-to-end gauntlet test for Animus agent.

Feeds the agent a real multi-step task (create a program in Downloads),
verifies each outcome, and saves the full transcript to GECK/tests/.

Requires a real LLM provider to be available (skipped otherwise).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

GECK_TESTS_DIR = Path(r"C:\Users\charl\GECK\tests")
DOWNLOADS_DIR = Path(os.path.expanduser("~/Downloads"))
TEST_DIR_NAME = "animus_gauntlet_test"


def _provider_available() -> bool:
    """Check if any LLM provider is available for real inference."""
    try:
        from src.core.config import AnimusConfig
        from src.llm.factory import ProviderFactory

        cfg = AnimusConfig.load()
        factory = ProviderFactory()
        provider = factory.create(
            cfg.model.provider,
            model_path=cfg.model.model_path,
            model_name=cfg.model.model_name,
            context_length=cfg.model.context_length,
            gpu_layers=cfg.model.gpu_layers,
            size_tier=cfg.model.size_tier,
            max_tokens=cfg.model.max_tokens,
        )
        return provider is not None and provider.available()
    except Exception:
        return False


@pytest.mark.skipif(not _provider_available(), reason="No LLM provider available")
@pytest.mark.timeout(300)
def test_gauntlet_create_calculator():
    """End-to-end test: create calculator program, git init, commit.

    Sends the agent a multi-step task and verifies:
    1. Directory created in Downloads
    2. calculator.py contains add/subtract/multiply/divide functions
    3. .git directory exists (git init)
    4. git log shows at least one commit
    """
    from src.core.agent import Agent
    from src.core.config import AnimusConfig
    from src.core.cwd import SessionCwd
    from src.core.transcript import TranscriptLogger
    from src.llm.factory import ProviderFactory
    from src.tools.base import ToolRegistry
    from src.tools.filesystem import register_filesystem_tools
    from src.tools.shell import register_shell_tools

    # Set up
    cfg = AnimusConfig.load()
    factory = ProviderFactory()
    provider = factory.create(
        cfg.model.provider,
        model_path=cfg.model.model_path,
        model_name=cfg.model.model_name,
        context_length=cfg.model.context_length,
        gpu_layers=cfg.model.gpu_layers,
        size_tier=cfg.model.size_tier,
        max_tokens=cfg.model.max_tokens,
    )
    assert provider is not None and provider.available()

    session_cwd = SessionCwd()
    registry = ToolRegistry()
    register_filesystem_tools(registry, session_cwd=session_cwd)
    register_shell_tools(registry, confirm_callback=lambda _: True, session_cwd=session_cwd)

    try:
        from src.tools.git import register_git_tools
        register_git_tools(registry, confirm_callback=lambda _: True, session_cwd=session_cwd)
    except ImportError:
        pass

    transcript = TranscriptLogger()
    agent = Agent(
        provider=provider,
        tool_registry=registry,
        system_prompt=cfg.agent.system_prompt,
        max_turns=cfg.agent.max_turns,
        session_cwd=session_cwd,
        transcript=transcript,
    )

    test_dir = DOWNLOADS_DIR / TEST_DIR_NAME
    task = (
        f'Create a folder called "{TEST_DIR_NAME}" in {DOWNLOADS_DIR}, '
        f"then write a file called calculator.py inside it with 4 functions: "
        f"add(a, b), subtract(a, b), multiply(a, b), divide(a, b) that each "
        f"return the result of the operation. Include a proper if __name__ == '__main__' block. "
        f"Then git init the folder, git add all files, and git commit with message 'Initial commit'."
    )

    transcript.log_task_start(task)

    # Execute
    response = agent.run_planned(
        task,
        on_progress=lambda s, t, d: None,
        on_step_output=lambda t: None,
        force=True,
    )

    plan_result = getattr(agent, '_last_plan_result', None)
    transcript.log_task_complete(success=plan_result.success if plan_result else True)

    # Save transcript BEFORE assertions (so we always get it)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    transcript_path = GECK_TESTS_DIR / f"gauntlet_{timestamp}.md"
    GECK_TESTS_DIR.mkdir(parents=True, exist_ok=True)
    transcript.save(transcript_path)

    # Assertions
    try:
        # 1. Directory exists
        assert test_dir.exists(), f"Test directory {test_dir} was not created"

        # 2. calculator.py with required functions
        calc_file = test_dir / "calculator.py"
        assert calc_file.exists(), "calculator.py was not created"
        calc_content = calc_file.read_text(encoding="utf-8")
        for func_name in ["def add", "def subtract", "def multiply", "def divide"]:
            assert func_name in calc_content, f"calculator.py missing {func_name}"

        # 3. .git directory
        git_dir = test_dir / ".git"
        assert git_dir.exists(), ".git directory was not created (git init failed)"

        # 4. git log shows commits
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=str(test_dir),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"git log failed: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "No commits found in git log"

    finally:
        # Cleanup test directory
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
