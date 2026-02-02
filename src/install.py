"""Animus installation and setup module.

Provides cross-platform automated installation of Animus and its dependencies.
Supports: Windows, macOS (Intel & Apple Silicon), Linux (x86_64, ARM64, Jetson).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from src.core.detection import (
    detect_environment,
    SystemInfo,
    HardwareType,
    OperatingSystem,
    Architecture,
)


class InstallStep(Enum):
    """Installation step types."""
    DETECT = "detect"
    PYTHON_DEPS = "python_deps"
    NATIVE_BACKEND = "native_backend"
    EMBEDDINGS = "embeddings"
    CONFIGURE = "configure"
    VERIFY = "verify"


@dataclass
class InstallProgress:
    """Progress update during installation."""
    step: InstallStep
    message: str
    success: bool = True
    warning: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class InstallResult:
    """Result of installation."""
    success: bool
    system_info: Optional[SystemInfo]
    installed_components: list[str]
    warnings: list[str]
    errors: list[str]
    next_steps: list[str]


class AnimusInstaller:
    """Cross-platform Animus installer."""

    def __init__(
        self,
        skip_native: bool = False,
        skip_embeddings: bool = False,
        force_cpu: bool = False,
        verbose: bool = False,
    ):
        """Initialize installer.

        Args:
            skip_native: Skip llama-cpp-python installation.
            skip_embeddings: Skip sentence-transformers installation.
            force_cpu: Force CPU-only installation (no GPU acceleration).
            verbose: Show detailed output.
        """
        self.skip_native = skip_native
        self.skip_embeddings = skip_embeddings
        self.force_cpu = force_cpu
        self.verbose = verbose
        self.system_info: Optional[SystemInfo] = None
        self.installed: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def _run_command(
        self,
        cmd: list[str],
        check: bool = True,
        capture: bool = True,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        """Run a shell command."""
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        if self.verbose:
            print(f"  Running: {' '.join(cmd)}")

        return subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            env=full_env,
        )

    def _pip_install(
        self,
        packages: list[str],
        extra_args: Optional[list[str]] = None,
        env: Optional[dict] = None,
    ) -> bool:
        """Install packages via pip."""
        cmd = [sys.executable, "-m", "pip", "install"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(packages)

        try:
            self._run_command(cmd, env=env)
            return True
        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"  pip install failed: {e}")
            return False

    def _check_package_installed(self, package: str) -> bool:
        """Check if a Python package is installed."""
        try:
            self._run_command([sys.executable, "-c", f"import {package}"])
            return True
        except subprocess.CalledProcessError:
            return False

    def install(
        self,
        progress_callback: Optional[Callable[[InstallProgress], None]] = None,
    ) -> InstallResult:
        """Run the full installation process.

        Args:
            progress_callback: Called with progress updates.

        Returns:
            InstallResult with success status and details.
        """
        def report(step: InstallStep, message: str, **kwargs):
            progress = InstallProgress(step=step, message=message, **kwargs)
            if progress_callback:
                progress_callback(progress)

        # Step 1: Detect system
        report(InstallStep.DETECT, "Detecting system environment...")
        self.system_info = detect_environment()
        report(
            InstallStep.DETECT,
            f"Detected: {self.system_info.os.value} {self.system_info.architecture.value}",
            detail=f"Hardware: {self.system_info.hardware_type.value}",
        )

        # Step 2: Install base Python dependencies
        report(InstallStep.PYTHON_DEPS, "Installing base dependencies...")
        if not self._install_base_deps():
            self.errors.append("Failed to install base dependencies")
            report(InstallStep.PYTHON_DEPS, "Base dependencies failed", success=False)
        else:
            self.installed.append("base-deps")
            report(InstallStep.PYTHON_DEPS, "Base dependencies installed")

        # Step 3: Install native backend (llama-cpp-python)
        if not self.skip_native:
            report(InstallStep.NATIVE_BACKEND, "Installing native inference backend...")
            if self._install_native_backend():
                self.installed.append("llama-cpp-python")
                report(InstallStep.NATIVE_BACKEND, "Native backend installed")
            else:
                self.warnings.append("Native backend installation failed")
                report(
                    InstallStep.NATIVE_BACKEND,
                    "Native backend skipped",
                    warning="Local inference will not be available",
                )
        else:
            report(InstallStep.NATIVE_BACKEND, "Skipping native backend (--skip-native)")

        # Step 4: Install embeddings
        if not self.skip_embeddings:
            report(InstallStep.EMBEDDINGS, "Installing embedding model support...")
            if self._install_embeddings():
                self.installed.append("sentence-transformers")
                report(InstallStep.EMBEDDINGS, "Embedding support installed")
            else:
                self.warnings.append("Embedding support installation failed")
                report(
                    InstallStep.EMBEDDINGS,
                    "Embedding support skipped",
                    warning="Memory/search features may be limited",
                )
        else:
            report(InstallStep.EMBEDDINGS, "Skipping embeddings (--skip-embeddings)")

        # Step 5: Configure Animus
        report(InstallStep.CONFIGURE, "Configuring Animus...")
        if self._configure():
            self.installed.append("configuration")
            report(InstallStep.CONFIGURE, "Configuration complete")
        else:
            self.errors.append("Configuration failed")
            report(InstallStep.CONFIGURE, "Configuration failed", success=False)

        # Step 6: Verify installation
        report(InstallStep.VERIFY, "Verifying installation...")
        verification = self._verify()
        report(InstallStep.VERIFY, f"Verified {len(verification)} components")

        # Generate next steps
        next_steps = self._generate_next_steps()

        return InstallResult(
            success=len(self.errors) == 0,
            system_info=self.system_info,
            installed_components=self.installed,
            warnings=self.warnings,
            errors=self.errors,
            next_steps=next_steps,
        )

    def _install_base_deps(self) -> bool:
        """Install base Python dependencies."""
        packages = [
            "typer>=0.9.0",
            "rich>=13.0.0",
            "pyyaml>=6.0",
            "pydantic>=2.0.0",
            "pydantic-settings>=2.0.0",
            "httpx>=0.25.0",
        ]
        return self._pip_install(packages)

    def _install_native_backend(self) -> bool:
        """Install llama-cpp-python with appropriate GPU support."""
        if self.force_cpu:
            return self._install_llama_cpp_cpu()

        info = self.system_info
        if not info:
            return self._install_llama_cpp_cpu()

        # Jetson uses pre-built wheels or source compilation
        if info.hardware_type == HardwareType.JETSON:
            return self._install_llama_cpp_jetson()

        # Apple Silicon uses Metal
        if info.hardware_type == HardwareType.APPLE_SILICON:
            return self._install_llama_cpp_metal()

        # NVIDIA GPU uses CUDA
        if info.gpu and info.gpu.cuda_available:
            return self._install_llama_cpp_cuda()

        # AMD GPU uses ROCm (if available)
        if info.gpu and info.gpu.vendor == "AMD":
            return self._install_llama_cpp_rocm()

        # Fallback to CPU
        return self._install_llama_cpp_cpu()

    def _install_llama_cpp_cpu(self) -> bool:
        """Install llama-cpp-python for CPU only."""
        return self._pip_install(["llama-cpp-python>=0.2.0"])

    def _install_llama_cpp_cuda(self) -> bool:
        """Install llama-cpp-python with CUDA support."""
        # Set environment variable to enable CUDA during build
        env = {"CMAKE_ARGS": "-DGGML_CUDA=on"}
        success = self._pip_install(
            ["llama-cpp-python>=0.2.0"],
            extra_args=["--force-reinstall", "--no-cache-dir"],
            env=env,
        )

        if not success:
            # Fallback to CPU if CUDA build fails
            self.warnings.append("CUDA build failed, falling back to CPU")
            return self._install_llama_cpp_cpu()

        return success

    def _install_llama_cpp_metal(self) -> bool:
        """Install llama-cpp-python with Metal support for Apple Silicon."""
        # Metal is automatically enabled on macOS ARM64
        env = {"CMAKE_ARGS": "-DGGML_METAL=on"}
        return self._pip_install(
            ["llama-cpp-python>=0.2.0"],
            extra_args=["--force-reinstall", "--no-cache-dir"],
            env=env,
        )

    def _install_llama_cpp_rocm(self) -> bool:
        """Install llama-cpp-python with ROCm support for AMD GPUs."""
        env = {"CMAKE_ARGS": "-DGGML_HIPBLAS=on"}
        success = self._pip_install(
            ["llama-cpp-python>=0.2.0"],
            extra_args=["--force-reinstall", "--no-cache-dir"],
            env=env,
        )

        if not success:
            self.warnings.append("ROCm build failed, falling back to CPU")
            return self._install_llama_cpp_cpu()

        return success

    def _install_llama_cpp_jetson(self) -> bool:
        """Install llama-cpp-python for NVIDIA Jetson.

        Jetson requires special handling due to its ARM64 + CUDA architecture.
        We try multiple approaches in order of preference.
        """
        # Approach 1: Try pre-built Jetson wheel from JetPack/NVIDIA
        # These are sometimes available for specific JetPack versions
        jetson_wheel_success = self._try_jetson_prebuilt()
        if jetson_wheel_success:
            return True

        # Approach 2: Build from source with CUDA
        # Jetson uses a different CUDA architecture than desktop GPUs
        cuda_arch = self._detect_jetson_cuda_arch()
        env = {
            "CMAKE_ARGS": f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            "CUDACXX": "/usr/local/cuda/bin/nvcc",
        }

        # Check if CUDA toolkit is installed
        if not Path("/usr/local/cuda/bin/nvcc").exists():
            self.warnings.append(
                "CUDA toolkit not found on Jetson. "
                "Install with: sudo apt install nvidia-jetpack"
            )
            # Try CPU-only as fallback
            return self._install_llama_cpp_cpu()

        success = self._pip_install(
            ["llama-cpp-python>=0.2.0"],
            extra_args=["--force-reinstall", "--no-cache-dir"],
            env=env,
        )

        if not success:
            # Final fallback to CPU
            self.warnings.append("Jetson CUDA build failed, using CPU")
            return self._install_llama_cpp_cpu()

        return success

    def _try_jetson_prebuilt(self) -> bool:
        """Try to install pre-built Jetson wheel if available."""
        # Check for JetPack version
        jetpack_version = self._detect_jetpack_version()
        if not jetpack_version:
            return False

        # Known pre-built wheel URLs (these would need to be maintained)
        # For now, we skip this and go straight to source build
        return False

    def _detect_jetpack_version(self) -> Optional[str]:
        """Detect NVIDIA JetPack version on Jetson."""
        try:
            # Try dpkg query
            result = self._run_command(
                ["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"],
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        # Try reading from /etc/nv_tegra_release
        try:
            tegra_path = Path("/etc/nv_tegra_release")
            if tegra_path.exists():
                content = tegra_path.read_text()
                # Parse version from content
                for line in content.split("\n"):
                    if "R" in line:
                        # Format: "# R35 (release), REVISION: 4.1..."
                        parts = line.split()
                        for part in parts:
                            if part.startswith("R"):
                                return part
        except Exception:
            pass

        return None

    def _detect_jetson_cuda_arch(self) -> str:
        """Detect CUDA compute capability for Jetson device.

        Returns appropriate CMAKE_CUDA_ARCHITECTURES value.
        """
        model_path = Path("/sys/firmware/devicetree/base/model")
        model = ""
        if model_path.exists():
            try:
                model = model_path.read_text().lower()
            except Exception:
                pass

        # Jetson device CUDA architectures:
        # - Jetson Nano: 5.3 (Maxwell)
        # - Jetson TX1: 5.3 (Maxwell)
        # - Jetson TX2: 6.2 (Pascal)
        # - Jetson Xavier: 7.2 (Volta)
        # - Jetson Orin: 8.7 (Ampere)

        if "orin" in model:
            return "87"  # Ampere
        elif "xavier" in model:
            return "72"  # Volta
        elif "tx2" in model:
            return "62"  # Pascal
        elif "nano" in model or "tx1" in model:
            return "53"  # Maxwell

        # Default to supporting multiple architectures
        return "53;62;72;87"

    def _install_embeddings(self) -> bool:
        """Install sentence-transformers for embeddings."""
        # sentence-transformers has many dependencies (torch, transformers, etc.)
        # This can be a large download

        if self.system_info and self.system_info.hardware_type == HardwareType.JETSON:
            # Jetson needs special PyTorch installation
            return self._install_embeddings_jetson()

        return self._pip_install(["sentence-transformers>=2.2.0"])

    def _install_embeddings_jetson(self) -> bool:
        """Install sentence-transformers for Jetson.

        Jetson requires PyTorch from NVIDIA's wheel repository.
        """
        # Try to install PyTorch for Jetson first
        # NVIDIA provides pre-built wheels at https://developer.download.nvidia.com/compute/redist/jp/

        # Check if PyTorch is already installed
        if self._check_package_installed("torch"):
            # PyTorch exists, just install sentence-transformers
            return self._pip_install(["sentence-transformers>=2.2.0"])

        # Try NVIDIA's PyTorch wheel
        jetpack_version = self._detect_jetpack_version()

        # For JetPack 5.x (L4T R35.x), use the corresponding PyTorch
        # This is a common case for Orin devices
        nvidia_wheel_url = (
            "https://developer.download.nvidia.com/compute/redist/jp/v511/"
            "pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl"
        )

        # The exact URL depends on JetPack version and Python version
        # For now, try the standard pip install which may work for some versions
        self.warnings.append(
            "For best Jetson performance, consider installing PyTorch from NVIDIA: "
            "https://forums.developer.nvidia.com/t/pytorch-for-jetson/"
        )

        # Try standard install
        success = self._pip_install(["sentence-transformers>=2.2.0"])

        if not success:
            self.warnings.append(
                "sentence-transformers installation failed on Jetson. "
                "Memory/search features will be limited."
            )

        return success

    def _configure(self) -> bool:
        """Configure Animus with detected settings."""
        try:
            from src.core.config import ConfigManager

            manager = ConfigManager()
            manager.ensure_directories()

            config = manager.config
            info = self.system_info

            if info:
                # Set provider based on hardware
                if info.hardware_type == HardwareType.JETSON:
                    # Prefer TensorRT-LLM on Jetson if available
                    config.model.provider = "trtllm"
                elif self._check_package_installed("llama_cpp"):
                    config.model.provider = "native"
                # Default to native even if llama_cpp not installed yet
                # User will need to install it for local inference

                # Set GPU layers based on detection
                if info.gpu and not self.force_cpu:
                    config.native.n_gpu_layers = -1  # All layers on GPU
                else:
                    config.native.n_gpu_layers = 0  # CPU only

            manager.save(config)
            return True

        except Exception as e:
            if self.verbose:
                print(f"  Configuration error: {e}")
            return False

    def _verify(self) -> list[str]:
        """Verify installed components."""
        verified = []

        # Check base dependencies
        for pkg in ["typer", "rich", "yaml", "pydantic", "httpx"]:
            if self._check_package_installed(pkg):
                verified.append(pkg)

        # Check native backend
        if self._check_package_installed("llama_cpp"):
            verified.append("llama-cpp-python")

        # Check embeddings
        if self._check_package_installed("sentence_transformers"):
            verified.append("sentence-transformers")

        # Check huggingface-hub
        if self._check_package_installed("huggingface_hub"):
            verified.append("huggingface-hub")

        return verified

    def _generate_next_steps(self) -> list[str]:
        """Generate next steps based on installation result."""
        steps = []
        info = self.system_info

        # Step 1: Get a model
        steps.append(
            "Download a model:\n"
            "  animus pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
        )

        # Step 2: Start chatting
        steps.append(
            "Start an interactive session:\n"
            "  animus rise"
        )

        # Platform-specific notes
        if info:
            if info.hardware_type == HardwareType.JETSON:
                steps.insert(0,
                    "For optimal Jetson performance:\n"
                    "  - Ensure JetPack is up to date: sudo apt update && sudo apt upgrade\n"
                    "  - Use quantized models (Q4_K_M or Q4_0) to fit in memory"
                )
            elif info.hardware_type == HardwareType.APPLE_SILICON:
                steps.insert(0,
                    "Apple Silicon detected - Metal acceleration enabled.\n"
                    "  Use animus sense to verify GPU detection."
                )

        # Add warning about missing components
        if "llama-cpp-python" not in self.installed and "sentence-transformers" not in self.installed:
            steps.insert(0,
                "Optional: Install native components for offline use:\n"
                "  pip install llama-cpp-python sentence-transformers"
            )

        return steps


def install_animus(
    skip_native: bool = False,
    skip_embeddings: bool = False,
    force_cpu: bool = False,
    verbose: bool = False,
    progress_callback: Optional[Callable[[InstallProgress], None]] = None,
) -> InstallResult:
    """Install Animus with all dependencies.

    Args:
        skip_native: Skip llama-cpp-python installation.
        skip_embeddings: Skip sentence-transformers installation.
        force_cpu: Force CPU-only installation.
        verbose: Show detailed output.
        progress_callback: Called with progress updates.

    Returns:
        InstallResult with installation details.
    """
    installer = AnimusInstaller(
        skip_native=skip_native,
        skip_embeddings=skip_embeddings,
        force_cpu=force_cpu,
        verbose=verbose,
    )
    return installer.install(progress_callback)
