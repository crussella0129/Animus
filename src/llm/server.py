"""
Managed llama-server subprocess.

Spawns llama-server on a random localhost port for the duration of a session.
No persistent daemon, no network exposure. Killed when session ends.
"""

import atexit
import json
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import urllib.request


# Release info for downloading llama-server
LLAMA_CPP_RELEASE_TAG = "b7926"
LLAMA_CPP_BASE_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE_TAG}"
)


def _get_binary_name() -> str:
    """Get the platform-specific binary name."""
    if sys.platform == "win32":
        return "llama-server.exe"
    return "llama-server"


def _get_download_asset() -> str:
    """Get the platform-specific download asset name."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        # TODO: Add CUDA/Vulkan detection for GPU builds
        return f"llama-{LLAMA_CPP_RELEASE_TAG}-bin-win-cpu-x64.zip"
    elif system == "darwin":
        return f"llama-{LLAMA_CPP_RELEASE_TAG}-bin-macos-arm64.zip"
    elif system == "linux":
        if "aarch64" in machine or "arm64" in machine:
            return f"llama-{LLAMA_CPP_RELEASE_TAG}-bin-ubuntu-x64.zip"
        return f"llama-{LLAMA_CPP_RELEASE_TAG}-bin-ubuntu-x64.zip"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


class LlamaServer:
    """
    Manages a llama-server subprocess bound to localhost.

    Usage:
        server = LlamaServer(bin_dir=Path("~/.animus/bin"))
        url = server.start(model_path=Path("~/.animus/models/qwen3-8b.gguf"))
        # url is "http://127.0.0.1:{port}/v1"
        # Use with LiteLLM: litellm.acompletion(model="openai/model", api_base=url)
        server.stop()
    """

    def __init__(
        self,
        bin_dir: Optional[Path] = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
    ):
        self.bin_dir = bin_dir or (Path.home() / ".animus" / "bin")
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._model_path: Optional[Path] = None

        # Register cleanup on exit
        atexit.register(self.stop)

    @property
    def binary_path(self) -> Path:
        """Path to the llama-server binary."""
        return self.bin_dir / _get_binary_name()

    @property
    def is_installed(self) -> bool:
        """Check if llama-server binary exists."""
        return self.binary_path.exists()

    @property
    def is_running(self) -> bool:
        """Check if the server subprocess is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def base_url(self) -> Optional[str]:
        """Get the OpenAI-compatible base URL."""
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}/v1"

    @property
    def port(self) -> Optional[int]:
        return self._port

    def install(self, progress_callback=None) -> Path:
        """
        Download and install llama-server binary.

        Args:
            progress_callback: Optional callable(status, completed, total)

        Returns:
            Path to the installed binary.
        """
        if self.is_installed:
            return self.binary_path

        self.bin_dir.mkdir(parents=True, exist_ok=True)

        asset_name = _get_download_asset()
        download_url = f"{LLAMA_CPP_BASE_URL}/{asset_name}"
        zip_path = self.bin_dir / asset_name

        if progress_callback:
            progress_callback("downloading", 0, 1)

        # Download the zip
        urllib.request.urlretrieve(download_url, str(zip_path))

        if progress_callback:
            progress_callback("extracting", 0, 1)

        # Extract llama-server binary and all required DLLs
        binary_name = _get_binary_name()
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the server binary in the archive
            server_entries = [
                n for n in zf.namelist()
                if n.endswith(binary_name)
            ]
            if not server_entries:
                raise FileNotFoundError(
                    f"{binary_name} not found in {asset_name}. "
                    f"Contents: {zf.namelist()[:10]}"
                )

            # Determine the directory prefix in the archive
            entry = server_entries[0]
            entry_dir = str(Path(entry).parent)

            # Extract the binary and all sibling files (DLLs, .so, .dylib)
            for name in zf.namelist():
                # Skip directories
                if name.endswith("/"):
                    continue
                # Extract files from the same directory as the binary
                file_dir = str(Path(name).parent)
                if file_dir == entry_dir:
                    filename = Path(name).name
                    target = self.bin_dir / filename
                    with zf.open(name) as src:
                        with open(target, "wb") as dst:
                            dst.write(src.read())

        # Make executable on Unix
        if sys.platform != "win32":
            self.binary_path.chmod(0o755)

        # Clean up zip
        zip_path.unlink(missing_ok=True)

        if progress_callback:
            progress_callback("complete", 1, 1)

        return self.binary_path

    def start(
        self,
        model_path: Path,
        port: Optional[int] = None,
        timeout: float = 60.0,
    ) -> str:
        """
        Start llama-server with the given model.

        Args:
            model_path: Path to the GGUF model file.
            port: Port to bind to (random if None).
            timeout: Max seconds to wait for server to be ready.

        Returns:
            Base URL for the OpenAI-compatible API (http://127.0.0.1:{port}/v1).

        Raises:
            FileNotFoundError: If binary or model not found.
            TimeoutError: If server doesn't start in time.
            RuntimeError: If server process exits unexpectedly.
        """
        if self.is_running:
            self.stop()

        if not self.is_installed:
            self.install()

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Pick a random port if not specified
        if port is None:
            port = random.randint(49152, 65535)

        self._port = port
        self._model_path = model_path

        cmd = [
            str(self.binary_path),
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--jinja",                        # Required for native tool calling
            "--flash-attn", "on",             # Flash attention
        ]

        # Start the process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" else 0
            ),
        )

        # Wait for the server to be ready
        health_url = f"http://127.0.0.1:{port}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process died
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}. "
                    f"stderr: {stderr[:500]}"
                )

            # Try health check
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read())
                    if data.get("status") == "ok":
                        return self.base_url
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                pass

            time.sleep(0.5)

        # Timeout — kill and report
        self.stop()
        raise TimeoutError(
            f"llama-server did not become ready within {timeout}s. "
            f"Model: {model_path.name}, Port: {port}"
        )

    def stop(self) -> None:
        """Stop the llama-server subprocess."""
        if self._process is not None:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=2)
            except (OSError, ProcessLookupError):
                pass
            finally:
                self._process = None
                self._port = None

    def __del__(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()
