"""Tests for system detection."""

from __future__ import annotations

from unittest.mock import patch

from src.core.detection import GPUInfo, SystemInfo, detect_system


class TestSystemInfo:
    def test_is_windows(self):
        info = SystemInfo(os_name="Windows")
        assert info.is_windows is True
        info2 = SystemInfo(os_name="Linux")
        assert info2.is_windows is False

    def test_is_windows_11_high_build(self):
        info = SystemInfo(os_name="Windows", os_version="10.0.22621")
        assert info.is_windows_11 is True

    def test_is_windows_11_low_build(self):
        info = SystemInfo(os_name="Windows", os_version="10.0.19045")
        assert info.is_windows_11 is False

    def test_is_windows_11_on_linux(self):
        info = SystemInfo(os_name="Linux", os_version="6.1.0")
        assert info.is_windows_11 is False

    def test_is_windows_11_bad_version(self):
        info = SystemInfo(os_name="Windows", os_version="unknown")
        assert info.is_windows_11 is False


class TestGPUInfo:
    def test_defaults(self):
        gpu = GPUInfo()
        assert gpu.name == "Unknown"
        assert gpu.memory_mb == 0
        assert gpu.cuda_available is False


class TestDetectSystem:
    @patch("src.core.detection._detect_gpu")
    @patch("src.core.detection._detect_hardware_type")
    def test_detect_system_returns_system_info(self, mock_hw, mock_gpu):
        mock_hw.return_value = "x86"
        mock_gpu.return_value = GPUInfo()
        result = detect_system()
        assert isinstance(result, SystemInfo)
        assert result.os_name != ""
        assert result.cpu_count >= 1
        assert result.python_version != ""

    @patch("src.core.detection.shutil.which", return_value=None)
    def test_detect_gpu_no_nvidia_smi(self, mock_which):
        from src.core.detection import _detect_gpu

        gpu = _detect_gpu()
        assert gpu.cuda_available is False
        assert gpu.name == "Unknown"
