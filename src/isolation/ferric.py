"""FerricSandbox: subprocess wrapper using the ferric-sandbox Rust binary.

Provides the same interface as OrnsteinSandbox.run_command() so it can be
used as a drop-in replacement via the sandbox= parameter in register_shell_tools().

NOTE: Isolation constraints (memory, network, read-only) are scaffolded but not
yet enforced. The binary currently acts as a process relay that records timing.
Kernel-level isolation (seccomp, namespaces) is reserved for a future release.
The isolation_level in results will read "ornsmo-stub" to reflect this.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FerricResult:
    """Result from ferric-sandbox execution. Compatible with OrnsteinResult interface."""

    success: bool
    output: Any          # stdout string
    error: Optional[str] # stderr string or None
    isolation_level: str
    resource_usage: dict[str, Any]
    execution_time: float = 0.0


class FerricSandbox:
    """Subprocess wrapper that delegates command execution to the ferric-sandbox binary.

    When ferric-sandbox is not found on PATH or in src/bin/, falls back to direct
    subprocess execution with a warning logged.
    """

    def __init__(
        self,
        memory_mb: int = 512,
        timeout_seconds: int = 30,
        no_network: bool = False,
        read_only: bool = False,
    ) -> None:
        self._memory_mb = memory_mb
        self._timeout_seconds = timeout_seconds
        self._no_network = no_network
        self._read_only = read_only

        from src.ferric import find_ferric_binary
        self._binary = find_ferric_binary("ferric-sandbox")
        if self._binary is None:
            logger.warning(
                "[FerricSandbox] ferric-sandbox binary not found — "
                "falling back to direct subprocess execution. "
                "Build with: cargo build -p ferric-sandbox"
            )

    def run_command(self, cmd_list: list, cwd: Any, timeout: int) -> FerricResult:
        """Run a shell command, routing through ferric-sandbox when available.

        Args:
            cmd_list: Command and arguments as a list (e.g. ["python", "script.py"])
            cwd: Working directory for the command
            timeout: Per-command timeout in seconds

        Returns:
            FerricResult compatible with OrnsteinResult interface
        """
        import time

        start = time.time()

        if self._binary is not None:
            return self._run_via_ferric(cmd_list, cwd, timeout, start)
        else:
            return self._run_direct(cmd_list, cwd, timeout, start)

    def _run_via_ferric(
        self,
        cmd_list: list,
        cwd: Any,
        timeout: int,
        start: float,
    ) -> FerricResult:
        """Execute through the ferric-sandbox binary."""
        import time

        sandbox_input = json.dumps({"args": [str(a) for a in cmd_list]})
        args = [
            self._binary,
            "--memory", str(self._memory_mb),
            "--timeout", str(timeout),
        ]
        if self._no_network:
            args.append("--no-network")
        if self._read_only:
            args.append("--read-only")

        try:
            proc = subprocess.run(
                args,
                input=sandbox_input,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                cwd=cwd if cwd else None,
            )
            elapsed = time.time() - start

            if proc.returncode != 0:
                # Non-zero exit from ferric-sandbox itself (not the sandboxed command)
                return FerricResult(
                    success=False,
                    output="",
                    error=f"ferric-sandbox exited with code {proc.returncode}: {proc.stderr}",
                    isolation_level="ornsmo-stub",
                    resource_usage={"wall_time_ms": int(elapsed * 1000)},
                    execution_time=elapsed,
                )

            data = json.loads(proc.stdout)
            resource = data.get("resource_usage", {})
            return FerricResult(
                success=data.get("success", False),
                output=data.get("output", ""),
                error=data.get("error"),
                isolation_level="ornsmo-stub",
                resource_usage={
                    "wall_time_ms": resource.get("wall_time_ms", int(elapsed * 1000)),
                    "cpu_time_ms": resource.get("cpu_time_ms", 0),
                    "peak_memory_kb": resource.get("peak_memory_kb", 0),
                },
                execution_time=elapsed,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logger.warning("[FerricSandbox] ferric-sandbox timed out — falling back to direct execution")
            return FerricResult(
                success=False,
                output="",
                error="ferric-sandbox timed out",
                isolation_level="ornsmo-stub",
                resource_usage={"wall_time_ms": int(elapsed * 1000)},
                execution_time=elapsed,
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            elapsed = time.time() - start
            logger.warning(f"[FerricSandbox] Binary error ({exc!r}), falling back to direct execution")
            return self._run_direct(cmd_list, cwd, timeout, start)

    def _run_direct(
        self,
        cmd_list: list,
        cwd: Any,
        timeout: int,
        start: float,
    ) -> FerricResult:
        """Direct subprocess execution fallback."""
        import time

        try:
            result = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd if cwd else None,
                stdin=subprocess.DEVNULL,
            )
            elapsed = time.time() - start
            return FerricResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                isolation_level="ornsmo-fallback",
                resource_usage={"wall_time_ms": int(elapsed * 1000)},
                execution_time=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return FerricResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s",
                isolation_level="ornsmo-fallback",
                resource_usage={"wall_time_ms": int(elapsed * 1000)},
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - start
            return FerricResult(
                success=False,
                output="",
                error=str(exc),
                isolation_level="ornsmo-fallback",
                resource_usage={"wall_time_ms": int(elapsed * 1000)},
                execution_time=elapsed,
            )


def create_ferric_sandbox(
    memory_mb: int = 512,
    timeout_seconds: int = 30,
) -> FerricSandbox:
    """Convenience factory for FerricSandbox."""
    return FerricSandbox(
        memory_mb=memory_mb,
        timeout_seconds=timeout_seconds,
    )
