"""Tests for native model provider."""

import pytest
from pathlib import Path
import tempfile

from src.llm.native import (
    NativeProvider,
    LLAMA_CPP_AVAILABLE,
    HF_HUB_AVAILABLE,
    detect_quantization,
    detect_gpu_backend,
)
from src.llm.base import ProviderType, Message, GenerationConfig


class TestQuantizationDetection:
    def test_detect_q4_k_m(self):
        assert detect_quantization("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_detect_q5_k_m(self):
        assert detect_quantization("codellama-7b-Q5_K_M.gguf") == "Q5_K_M"

    def test_detect_q8_0(self):
        assert detect_quantization("model.Q8_0.gguf") == "Q8_0"

    def test_detect_f16(self):
        assert detect_quantization("model-F16.gguf") == "F16"

    def test_detect_case_insensitive(self):
        assert detect_quantization("model-q4_k_m.gguf") == "Q4_K_M"

    def test_detect_unknown(self):
        assert detect_quantization("model.gguf") is None


class TestGPUBackendDetection:
    def test_returns_string(self):
        backend = detect_gpu_backend()
        assert isinstance(backend, str)
        assert backend in ["cuda", "metal", "rocm", "cpu"]


class TestNativeProvider:
    def test_provider_type(self):
        provider = NativeProvider()
        assert provider.provider_type == ProviderType.NATIVE

    def test_is_available(self):
        provider = NativeProvider()
        assert provider.is_available == LLAMA_CPP_AVAILABLE

    def test_gpu_backend(self):
        provider = NativeProvider()
        assert provider.gpu_backend in ["cuda", "metal", "rocm", "cpu"]

    def test_custom_models_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = NativeProvider(models_dir=Path(tmpdir))
            assert provider.models_dir == Path(tmpdir)

    def test_default_config(self):
        provider = NativeProvider()
        assert provider.n_ctx == 4096
        assert provider.n_batch == 512
        assert provider.n_gpu_layers == -1

    def test_custom_config(self):
        provider = NativeProvider(
            n_ctx=2048,
            n_batch=256,
            n_gpu_layers=20,
        )
        assert provider.n_ctx == 2048
        assert provider.n_batch == 256
        assert provider.n_gpu_layers == 20


class TestNativeProviderModelPaths:
    def test_get_model_path_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = NativeProvider(models_dir=Path(tmpdir))
            assert provider._get_model_path("nonexistent.gguf") is None

    def test_get_model_path_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test-model.gguf"
            model_path.touch()

            provider = NativeProvider(models_dir=Path(tmpdir))
            result = provider._get_model_path("test-model.gguf")
            assert result == model_path

    def test_get_model_path_without_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test-model.gguf"
            model_path.touch()

            provider = NativeProvider(models_dir=Path(tmpdir))
            result = provider._get_model_path("test-model")
            assert result == model_path

    def test_get_model_path_partial_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "codellama-7b-Q4_K_M.gguf"
            model_path.touch()

            provider = NativeProvider(models_dir=Path(tmpdir))
            result = provider._get_model_path("codellama")
            assert result == model_path


class TestNativeProviderListModels:
    @pytest.mark.asyncio
    async def test_list_models_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = NativeProvider(models_dir=Path(tmpdir))
            models = await provider.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_list_models_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test model files
            (Path(tmpdir) / "model1-Q4_K_M.gguf").write_bytes(b"x" * 1000)
            (Path(tmpdir) / "model2-Q8_0.gguf").write_bytes(b"x" * 2000)
            (Path(tmpdir) / "not-a-model.txt").write_text("test")

            provider = NativeProvider(models_dir=Path(tmpdir))
            models = await provider.list_models()

            assert len(models) == 2
            names = [m.name for m in models]
            assert "model1-Q4_K_M.gguf" in names
            assert "model2-Q8_0.gguf" in names

    @pytest.mark.asyncio
    async def test_list_models_detects_quantization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model-Q5_K_M.gguf").write_bytes(b"x" * 1000)

            provider = NativeProvider(models_dir=Path(tmpdir))
            models = await provider.list_models()

            assert len(models) == 1
            assert models[0].quantization == "Q5_K_M"
            assert models[0].provider == ProviderType.NATIVE


class TestNativeProviderModelExists:
    @pytest.mark.asyncio
    async def test_model_exists_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test-model.gguf").touch()

            provider = NativeProvider(models_dir=Path(tmpdir))
            assert await provider.model_exists("test-model.gguf") is True

    @pytest.mark.asyncio
    async def test_model_exists_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = NativeProvider(models_dir=Path(tmpdir))
            assert await provider.model_exists("nonexistent.gguf") is False


class TestNativeProviderMessages:
    def test_messages_to_prompt_basic(self):
        provider = NativeProvider()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello!"),
        ]

        prompt = provider._messages_to_prompt(messages)

        assert "<|im_start|>system" in prompt
        assert "You are helpful." in prompt
        assert "<|im_start|>user" in prompt
        assert "Hello!" in prompt
        assert "<|im_start|>assistant" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_messages_to_prompt_with_assistant(self):
        provider = NativeProvider()
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="How are you?"),
        ]

        prompt = provider._messages_to_prompt(messages)

        assert prompt.count("<|im_start|>user") == 2
        assert prompt.count("<|im_start|>assistant") == 2  # One from history, one for generation


class TestNativeProviderHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self):
        provider = NativeProvider()
        result = await provider.health_check()
        assert result == LLAMA_CPP_AVAILABLE


class TestNativeProviderUnload:
    def test_unload_model(self):
        provider = NativeProvider()
        provider._loaded_models["test"] = "dummy"
        provider.unload_model("test")
        assert "test" not in provider._loaded_models

    def test_unload_all(self):
        provider = NativeProvider()
        provider._loaded_models["test1"] = "dummy1"
        provider._loaded_models["test2"] = "dummy2"
        provider.unload_all()
        assert len(provider._loaded_models) == 0

    @pytest.mark.asyncio
    async def test_close(self):
        provider = NativeProvider()
        provider._loaded_models["test"] = "dummy"
        await provider.close()
        assert len(provider._loaded_models) == 0
