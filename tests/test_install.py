"""Tests for the Animus installer module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.install import (
    AnimusInstaller,
    InstallProgress,
    InstallStep,
    InstallResult,
    install_animus,
)
from src.core.detection import (
    SystemInfo,
    OperatingSystem,
    Architecture,
    HardwareType,
    GPUInfo,
)


class TestInstallProgress:
    """Test InstallProgress dataclass."""

    def test_create_progress(self):
        """Test creating progress update."""
        progress = InstallProgress(
            step=InstallStep.DETECT,
            message="Detecting system...",
            success=True,
        )
        assert progress.step == InstallStep.DETECT
        assert progress.message == "Detecting system..."
        assert progress.success is True

    def test_progress_with_warning(self):
        """Test progress with warning."""
        progress = InstallProgress(
            step=InstallStep.NATIVE_BACKEND,
            message="Backend installed",
            warning="GPU not detected",
        )
        assert progress.warning == "GPU not detected"


class TestInstallResult:
    """Test InstallResult dataclass."""

    def test_successful_result(self):
        """Test successful installation result."""
        result = InstallResult(
            success=True,
            system_info=None,
            installed_components=["base-deps", "llama-cpp-python"],
            warnings=[],
            errors=[],
            next_steps=["Download a model"],
        )
        assert result.success is True
        assert len(result.installed_components) == 2

    def test_failed_result(self):
        """Test failed installation result."""
        result = InstallResult(
            success=False,
            system_info=None,
            installed_components=["base-deps"],
            warnings=["GPU not found"],
            errors=["CUDA build failed"],
            next_steps=[],
        )
        assert result.success is False
        assert len(result.errors) == 1


class TestAnimusInstaller:
    """Test AnimusInstaller class."""

    @pytest.fixture
    def installer(self):
        """Create installer instance."""
        return AnimusInstaller(verbose=False)

    @pytest.fixture
    def mock_system_info_standard(self):
        """Create mock standard x86 system info."""
        return SystemInfo(
            os=OperatingSystem.LINUX,
            os_version="Ubuntu 22.04",
            architecture=Architecture.X86_64,
            hardware_type=HardwareType.STANDARD_X86,
            python_version="3.11.0",
            hostname="test",
            cpu_count=8,
            gpu=None,
        )

    @pytest.fixture
    def mock_system_info_cuda(self):
        """Create mock system with CUDA GPU."""
        return SystemInfo(
            os=OperatingSystem.LINUX,
            os_version="Ubuntu 22.04",
            architecture=Architecture.X86_64,
            hardware_type=HardwareType.STANDARD_X86,
            python_version="3.11.0",
            hostname="test",
            cpu_count=8,
            gpu=GPUInfo(
                name="NVIDIA RTX 3080",
                vendor="NVIDIA",
                memory_mb=10240,
                cuda_available=True,
                cuda_version="12.0",
            ),
        )

    @pytest.fixture
    def mock_system_info_jetson(self):
        """Create mock Jetson system info."""
        return SystemInfo(
            os=OperatingSystem.LINUX,
            os_version="L4T R35.4",
            architecture=Architecture.ARM64,
            hardware_type=HardwareType.JETSON,
            python_version="3.10.0",
            hostname="jetson-orin",
            cpu_count=8,
            gpu=GPUInfo(
                name="NVIDIA Orin",
                vendor="NVIDIA",
                memory_mb=8192,
                cuda_available=True,
                cuda_version="11.4",
            ),
        )

    @pytest.fixture
    def mock_system_info_apple(self):
        """Create mock Apple Silicon system info."""
        return SystemInfo(
            os=OperatingSystem.MACOS,
            os_version="14.0",
            architecture=Architecture.ARM64,
            hardware_type=HardwareType.APPLE_SILICON,
            python_version="3.11.0",
            hostname="macbook",
            cpu_count=10,
            gpu=GPUInfo(
                name="Apple M2",
                vendor="Apple",
            ),
        )

    def test_installer_init(self):
        """Test installer initialization."""
        installer = AnimusInstaller(
            skip_native=True,
            skip_embeddings=True,
            force_cpu=True,
            verbose=True,
        )
        assert installer.skip_native is True
        assert installer.skip_embeddings is True
        assert installer.force_cpu is True
        assert installer.verbose is True

    def test_installer_defaults(self, installer):
        """Test installer default values."""
        assert installer.skip_native is False
        assert installer.skip_embeddings is False
        assert installer.force_cpu is False
        assert installer.system_info is None
        assert installer.installed == []
        assert installer.errors == []

    @patch.object(AnimusInstaller, '_run_command')
    def test_pip_install(self, mock_run, installer):
        """Test pip install helper."""
        mock_run.return_value = MagicMock(returncode=0)

        result = installer._pip_install(["package1", "package2"])

        assert result is True
        mock_run.assert_called_once()

    @patch.object(AnimusInstaller, '_run_command')
    def test_pip_install_failure(self, mock_run, installer):
        """Test pip install failure handling."""
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, "pip")

        result = installer._pip_install(["nonexistent-package"])

        assert result is False

    @patch.object(AnimusInstaller, '_run_command')
    def test_check_package_installed(self, mock_run, installer):
        """Test package installation check."""
        mock_run.return_value = MagicMock(returncode=0)

        result = installer._check_package_installed("rich")

        assert result is True

    @patch.object(AnimusInstaller, '_run_command')
    def test_check_package_not_installed(self, mock_run, installer):
        """Test package not installed check."""
        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, "python")

        result = installer._check_package_installed("nonexistent")

        assert result is False

    def test_detect_jetson_cuda_arch_orin(self, installer):
        """Test Jetson Orin CUDA architecture detection."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value="NVIDIA Orin Nano"):
                arch = installer._detect_jetson_cuda_arch()
                assert arch == "87"

    def test_detect_jetson_cuda_arch_xavier(self, installer):
        """Test Jetson Xavier CUDA architecture detection."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value="NVIDIA Jetson AGX Xavier"):
                arch = installer._detect_jetson_cuda_arch()
                assert arch == "72"

    def test_detect_jetson_cuda_arch_nano(self, installer):
        """Test Jetson Nano CUDA architecture detection."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.read_text', return_value="NVIDIA Jetson Nano"):
                arch = installer._detect_jetson_cuda_arch()
                assert arch == "53"

    def test_detect_jetson_cuda_arch_unknown(self, installer):
        """Test unknown Jetson CUDA architecture."""
        with patch('pathlib.Path.exists', return_value=False):
            arch = installer._detect_jetson_cuda_arch()
            # Should return multi-arch fallback
            assert "53" in arch or "72" in arch or "87" in arch

    @patch.object(AnimusInstaller, '_pip_install')
    def test_install_llama_cpp_cpu(self, mock_pip, installer):
        """Test CPU-only llama-cpp-python installation."""
        mock_pip.return_value = True

        result = installer._install_llama_cpp_cpu()

        assert result is True
        mock_pip.assert_called_once_with(["llama-cpp-python>=0.2.0"])

    @patch.object(AnimusInstaller, '_pip_install')
    def test_install_llama_cpp_cuda(self, mock_pip, installer):
        """Test CUDA llama-cpp-python installation."""
        mock_pip.return_value = True

        result = installer._install_llama_cpp_cuda()

        assert result is True
        # Should be called with CUDA cmake args
        call_args = mock_pip.call_args
        assert "env" in call_args.kwargs
        assert "CMAKE_ARGS" in call_args.kwargs["env"]

    @patch.object(AnimusInstaller, '_pip_install')
    def test_install_llama_cpp_cuda_fallback(self, mock_pip, installer):
        """Test CUDA installation fallback to CPU."""
        # First call (CUDA) fails, second call (CPU) succeeds
        mock_pip.side_effect = [False, True]

        result = installer._install_llama_cpp_cuda()

        assert result is True
        assert "CUDA build failed" in installer.warnings[0]

    @patch.object(AnimusInstaller, '_pip_install')
    def test_install_llama_cpp_metal(self, mock_pip, installer):
        """Test Metal llama-cpp-python installation."""
        mock_pip.return_value = True

        result = installer._install_llama_cpp_metal()

        assert result is True
        call_args = mock_pip.call_args
        assert "DGGML_METAL=on" in call_args.kwargs["env"]["CMAKE_ARGS"]

    @patch.object(AnimusInstaller, '_pip_install')
    @patch('pathlib.Path.exists')
    def test_install_llama_cpp_jetson_no_cuda(self, mock_exists, mock_pip, installer):
        """Test Jetson installation without CUDA toolkit."""
        # No CUDA toolkit found
        mock_exists.return_value = False
        mock_pip.return_value = True

        installer._install_llama_cpp_jetson()

        # Should warn about missing CUDA and fall back to CPU
        assert any("CUDA toolkit not found" in w for w in installer.warnings)

    def test_generate_next_steps_with_native(self, installer):
        """Test next steps generation with native backend."""
        installer.installed = ["base-deps", "llama-cpp-python"]
        installer.system_info = None

        steps = installer._generate_next_steps()

        assert len(steps) >= 2
        assert any("vessel download" in s for s in steps)
        assert any("rise" in s for s in steps)

    def test_generate_next_steps_without_native(self, installer):
        """Test next steps generation without native backend."""
        installer.installed = ["base-deps"]
        installer.system_info = None

        steps = installer._generate_next_steps()

        assert any("ollama" in s.lower() for s in steps)

    def test_generate_next_steps_jetson(self, installer, mock_system_info_jetson):
        """Test Jetson-specific next steps."""
        installer.installed = ["base-deps", "llama-cpp-python"]
        installer.system_info = mock_system_info_jetson

        steps = installer._generate_next_steps()

        # Should have Jetson-specific advice
        assert any("jetson" in s.lower() for s in steps)

    @patch('src.install.detect_environment')
    @patch.object(AnimusInstaller, '_install_base_deps')
    @patch.object(AnimusInstaller, '_install_native_backend')
    @patch.object(AnimusInstaller, '_install_embeddings')
    @patch.object(AnimusInstaller, '_configure')
    @patch.object(AnimusInstaller, '_verify')
    def test_full_install_success(
        self,
        mock_verify,
        mock_configure,
        mock_embeddings,
        mock_native,
        mock_base,
        mock_detect,
        installer,
        mock_system_info_standard,
    ):
        """Test full installation success path."""
        mock_detect.return_value = mock_system_info_standard
        mock_base.return_value = True
        mock_native.return_value = True
        mock_embeddings.return_value = True
        mock_configure.return_value = True
        mock_verify.return_value = ["typer", "rich", "llama-cpp-python"]

        progress_updates = []

        def on_progress(p):
            progress_updates.append(p)

        result = installer.install(progress_callback=on_progress)

        assert result.success is True
        assert "base-deps" in result.installed_components
        assert "llama-cpp-python" in result.installed_components
        assert len(progress_updates) > 0

    @patch('src.install.detect_environment')
    @patch.object(AnimusInstaller, '_install_base_deps')
    def test_full_install_base_failure(
        self,
        mock_base,
        mock_detect,
        installer,
        mock_system_info_standard,
    ):
        """Test installation with base deps failure."""
        mock_detect.return_value = mock_system_info_standard
        mock_base.return_value = False

        result = installer.install()

        assert result.success is False
        assert "base dependencies" in result.errors[0].lower()

    def test_install_with_skip_native(self, mock_system_info_standard):
        """Test installation with skip_native flag."""
        installer = AnimusInstaller(skip_native=True)

        with patch('src.install.detect_environment', return_value=mock_system_info_standard):
            with patch.object(installer, '_install_base_deps', return_value=True):
                with patch.object(installer, '_install_embeddings', return_value=True):
                    with patch.object(installer, '_configure', return_value=True):
                        with patch.object(installer, '_verify', return_value=[]):
                            result = installer.install()

        assert "llama-cpp-python" not in result.installed_components

    def test_install_with_force_cpu(self, mock_system_info_cuda):
        """Test installation with force_cpu flag."""
        installer = AnimusInstaller(force_cpu=True)
        installer.system_info = mock_system_info_cuda

        with patch.object(installer, '_install_llama_cpp_cpu', return_value=True) as mock_cpu:
            result = installer._install_native_backend()

        mock_cpu.assert_called_once()
        assert result is True


class TestInstallAnimusFunction:
    """Test the install_animus convenience function."""

    @patch('src.install.AnimusInstaller')
    def test_install_animus_default(self, mock_installer_class):
        """Test default install_animus call."""
        mock_installer = MagicMock()
        mock_installer.install.return_value = InstallResult(
            success=True,
            system_info=None,
            installed_components=[],
            warnings=[],
            errors=[],
            next_steps=[],
        )
        mock_installer_class.return_value = mock_installer

        result = install_animus()

        mock_installer_class.assert_called_once_with(
            skip_native=False,
            skip_embeddings=False,
            force_cpu=False,
            verbose=False,
        )
        assert result.success is True

    @patch('src.install.AnimusInstaller')
    def test_install_animus_with_options(self, mock_installer_class):
        """Test install_animus with options."""
        mock_installer = MagicMock()
        mock_installer.install.return_value = InstallResult(
            success=True,
            system_info=None,
            installed_components=[],
            warnings=[],
            errors=[],
            next_steps=[],
        )
        mock_installer_class.return_value = mock_installer

        install_animus(
            skip_native=True,
            skip_embeddings=True,
            force_cpu=True,
            verbose=True,
        )

        mock_installer_class.assert_called_once_with(
            skip_native=True,
            skip_embeddings=True,
            force_cpu=True,
            verbose=True,
        )


class TestInstallStep:
    """Test InstallStep enum."""

    def test_all_steps_defined(self):
        """Test all installation steps are defined."""
        assert InstallStep.DETECT.value == "detect"
        assert InstallStep.PYTHON_DEPS.value == "python_deps"
        assert InstallStep.NATIVE_BACKEND.value == "native_backend"
        assert InstallStep.EMBEDDINGS.value == "embeddings"
        assert InstallStep.CONFIGURE.value == "configure"
        assert InstallStep.VERIFY.value == "verify"
