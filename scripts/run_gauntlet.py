"""Standalone gauntlet runner for Animus agent.

Runs the same multi-step task as the gauntlet test but standalone:
    python -m scripts.run_gauntlet

Prints step progress to console, a verification checklist (PASS/FAIL),
and saves transcript to tests/gauntlet_transcripts/.

Override transcript output with ANIMUS_GAUNTLET_TRANSCRIPT_DIR env var.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = os.environ.get(
    "ANIMUS_GAUNTLET_TRANSCRIPT_DIR",
    str(_PROJECT_ROOT / "tests" / "gauntlet_transcripts"),
)
TEST_DIR_NAME = "animus_gauntlet_test"


def main() -> int:
    from src.core.agent import Agent
    from src.core.config import AnimusConfig
    from src.core.cwd import SessionCwd
    from src.core.transcript import TranscriptLogger
    from src.llm.factory import ProviderFactory
    from src.tools.base import ToolRegistry
    from src.tools.filesystem import register_filesystem_tools
    from src.tools.shell import register_shell_tools

    print("=" * 60)
    print("  ANIMUS GAUNTLET TEST")
    print("=" * 60)
    print()

    # Set up provider
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

    if provider is None or not provider.available():
        print("ERROR: No LLM provider available. Run 'animus status' to check.")
        return 1

    print(f"Provider: {cfg.model.provider}")
    print(f"Model:    {cfg.model.model_name}")
    print()

    # Set up tools
    session_cwd = SessionCwd()
    registry = ToolRegistry()
    register_filesystem_tools(registry, session_cwd=session_cwd)
    register_shell_tools(registry, confirm_callback=lambda _: True, session_cwd=session_cwd)

    try:
        from src.tools.git import register_git_tools
        register_git_tools(registry, confirm_callback=lambda _: True, session_cwd=session_cwd)
    except ImportError:
        print("WARNING: Git tools not available")

    # Use a temp directory so the runner is portable
    tmp_base = Path(tempfile.mkdtemp(prefix="animus_gauntlet_"))
    test_dir = tmp_base / TEST_DIR_NAME

    # Set up agent with transcript
    transcript = TranscriptLogger()
    agent = Agent(
        provider=provider,
        tool_registry=registry,
        system_prompt=cfg.agent.system_prompt,
        max_turns=cfg.agent.max_turns,
        session_cwd=session_cwd,
        transcript=transcript,
    )

    task = (
        f'Create a folder called "{TEST_DIR_NAME}" in {tmp_base}, '
        f"then write a file called calculator.py inside it with 4 functions: "
        f"add(a, b), subtract(a, b), multiply(a, b), divide(a, b) that each "
        f"return the result of the operation. Include a proper if __name__ == '__main__' block. "
        f"Then git init the folder, git add all files, and git commit with message 'Initial commit'."
    )

    def _on_progress(step_num: int, total: int, desc: str) -> None:
        print(f"  [{step_num}/{total}] {desc}")

    def _on_step_output(text: str) -> None:
        print(f"  > {text[:200]}")

    # Execute
    print("Task:", task[:100] + "...")
    print()
    print("Executing...")
    print("-" * 40)

    transcript.log_task_start(task)
    t0 = time.time()

    response = agent.run_planned(
        task,
        on_progress=_on_progress,
        on_step_output=_on_step_output,
        force=True,
    )

    elapsed = time.time() - t0
    plan_result = getattr(agent, '_last_plan_result', None)
    transcript.log_task_complete(success=plan_result.success if plan_result else True)

    print("-" * 40)
    print(f"Completed in {elapsed:.1f}s")
    print()

    # Save transcript BEFORE verification
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    transcript_out = Path(TRANSCRIPT_DIR)
    transcript_out.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_out / f"gauntlet_{timestamp}.md"
    transcript.save(transcript_path)
    print(f"Transcript saved: {transcript_path}")
    print()

    # Verification checklist
    print("VERIFICATION CHECKLIST")
    print("=" * 40)
    all_pass = True

    # Check 1: Directory exists
    dir_exists = test_dir.exists()
    _print_check("Directory created", dir_exists)
    all_pass &= dir_exists

    # Check 2: calculator.py with functions
    calc_file = test_dir / "calculator.py"
    calc_exists = calc_file.exists()
    _print_check("calculator.py exists", calc_exists)
    all_pass &= calc_exists

    if calc_exists:
        calc_content = calc_file.read_text(encoding="utf-8")
        for func_name in ["def add", "def subtract", "def multiply", "def divide"]:
            has_func = func_name in calc_content
            _print_check(f"  {func_name}()", has_func)
            all_pass &= has_func
    else:
        for func_name in ["def add", "def subtract", "def multiply", "def divide"]:
            _print_check(f"  {func_name}()", False)
            all_pass = False

    # Check 3: .git directory
    git_exists = (test_dir / ".git").exists()
    _print_check("git initialized", git_exists)
    all_pass &= git_exists

    # Check 4: git log
    has_commits = False
    if git_exists:
        try:
            result = subprocess.run(
                ["git", "log", "--oneline"],
                cwd=str(test_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            has_commits = result.returncode == 0 and len(result.stdout.strip()) > 0
        except Exception:
            pass
    _print_check("git commit exists", has_commits)
    all_pass &= has_commits

    print()
    if all_pass:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print("RESULT: SOME CHECKS FAILED")
    print()

    # Cleanup
    if tmp_base.exists():
        shutil.rmtree(tmp_base, ignore_errors=True)
        print(f"Cleaned up: {tmp_base}")

    return 0 if all_pass else 1


def _print_check(label: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    marker = "[+]" if passed else "[-]"
    print(f"  {marker} {label}: {status}")


if __name__ == "__main__":
    sys.exit(main())
