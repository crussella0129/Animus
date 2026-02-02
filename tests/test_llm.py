"""Tests for LLM providers."""

import pytest
from src.llm.base import (
    Message,
    GenerationConfig,
    GenerationResult,
    ModelInfo,
    ProviderType,
)
from src.llm.native import NativeProvider
from src.llm.api import APIProvider
from src.llm.trtllm import TRTLLMProvider
from src.llm.factory import create_provider


def test_message_creation():
    """Test Message dataclass creation."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.name is None
    assert msg.tool_calls is None


def test_generation_config_defaults():
    """Test GenerationConfig has sensible defaults."""
    config = GenerationConfig()
    assert config.temperature == 0.7
    assert config.max_tokens == 4096
    assert config.top_p == 1.0
    assert config.stream is False


def test_model_info_creation():
    """Test ModelInfo dataclass creation."""
    info = ModelInfo(
        name="test-model",
        provider=ProviderType.NATIVE,
        size_bytes=1024 * 1024 * 1024,
    )
    assert info.name == "test-model"
    assert info.provider == ProviderType.NATIVE
    assert info.size_bytes == 1024 * 1024 * 1024


def test_api_provider_type():
    """Test APIProvider returns correct type."""
    provider = APIProvider(api_key="test-key")
    assert provider.provider_type == ProviderType.API


def test_trtllm_provider_type():
    """Test TRTLLMProvider returns correct type."""
    provider = TRTLLMProvider()
    assert provider.provider_type == ProviderType.TRTLLM


def test_native_provider_type():
    """Test NativeProvider returns correct type."""
    provider = NativeProvider()
    assert provider.provider_type == ProviderType.NATIVE


def test_api_provider_not_available_without_key():
    """Test APIProvider reports unavailable without key."""
    provider = APIProvider()
    assert provider.is_available is False


def test_api_provider_available_with_key():
    """Test APIProvider reports available with key."""
    provider = APIProvider(api_key="test-key")
    assert provider.is_available is True


def test_create_provider_api():
    """Test factory creates APIProvider."""
    provider = create_provider("api")
    assert isinstance(provider, APIProvider)


def test_create_provider_trtllm():
    """Test factory creates TRTLLMProvider."""
    provider = create_provider("trtllm")
    assert isinstance(provider, TRTLLMProvider)


def test_create_provider_native():
    """Test factory creates NativeProvider."""
    provider = create_provider("native")
    assert isinstance(provider, NativeProvider)


def test_create_provider_invalid():
    """Test factory raises for unknown provider."""
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("unknown")


class TestTRTLLMProvider:
    """Tests for TRTLLMProvider."""

    def test_init_defaults(self):
        """Test TRTLLMProvider initializes with defaults."""
        provider = TRTLLMProvider()
        assert provider.model_dir is None
        assert provider.engine_dir is None
        assert provider.tokenizer_dir is None
        assert provider.tp_size == 1
        assert provider.pp_size == 1

    def test_init_with_paths(self, tmp_path):
        """Test TRTLLMProvider initializes with paths."""
        model_dir = tmp_path / "models"
        engine_dir = tmp_path / "engines"
        model_dir.mkdir()
        engine_dir.mkdir()

        provider = TRTLLMProvider(
            model_dir=model_dir,
            engine_dir=engine_dir,
            tp_size=2,
            pp_size=2,
        )
        assert provider.model_dir == model_dir
        assert provider.engine_dir == engine_dir
        assert provider.tp_size == 2
        assert provider.pp_size == 2

    def test_messages_to_prompt(self):
        """Test message to prompt conversion."""
        provider = TRTLLMProvider()
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]

        prompt = provider._messages_to_prompt(messages)

        assert "<|system|>" in prompt
        assert "You are helpful." in prompt
        assert "<|user|>" in prompt
        assert "Hello" in prompt
        assert "<|assistant|>" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt
        # Should end with assistant prefix for generation
        assert prompt.endswith("<|assistant|>\n")

    def test_config_to_sampling_params_without_trtllm(self):
        """Test config conversion raises when TensorRT-LLM not installed."""
        provider = TRTLLMProvider()
        config = GenerationConfig(temperature=0.5, max_tokens=100)

        # Should raise if TensorRT-LLM is not installed
        if not provider.is_available:
            with pytest.raises(RuntimeError, match="not installed"):
                provider._config_to_sampling_params(config)

    def test_check_jetson(self):
        """Test Jetson hardware detection."""
        provider = TRTLLMProvider()
        # This test just ensures the method runs without error
        # Result depends on actual hardware
        result = provider._check_jetson()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_model_exists_hf_model(self):
        """Test model_exists returns True for HuggingFace model names."""
        provider = TRTLLMProvider()
        # HuggingFace model names contain a slash
        exists = await provider.model_exists("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        assert exists is True

    @pytest.mark.asyncio
    async def test_model_exists_local_not_found(self, tmp_path):
        """Test model_exists returns False for non-existent local model."""
        provider = TRTLLMProvider(
            model_dir=tmp_path / "models",
            engine_dir=tmp_path / "engines",
        )
        exists = await provider.model_exists("nonexistent-model")
        assert exists is False

    @pytest.mark.asyncio
    async def test_model_exists_local_engine(self, tmp_path):
        """Test model_exists finds local engine files."""
        engine_dir = tmp_path / "engines"
        engine_dir.mkdir()
        (engine_dir / "test-model.engine").touch()

        provider = TRTLLMProvider(engine_dir=engine_dir)
        exists = await provider.model_exists("test-model")
        assert exists is True

    @pytest.mark.asyncio
    async def test_model_exists_local_model_dir(self, tmp_path):
        """Test model_exists finds local model directories."""
        model_dir = tmp_path / "models"
        test_model = model_dir / "test-model"
        test_model.mkdir(parents=True)
        (test_model / "config.json").touch()

        provider = TRTLLMProvider(model_dir=model_dir)
        exists = await provider.model_exists("test-model")
        assert exists is True

    @pytest.mark.asyncio
    async def test_list_models_empty(self, tmp_path):
        """Test list_models returns empty list when no models."""
        provider = TRTLLMProvider(
            model_dir=tmp_path / "models",
            engine_dir=tmp_path / "engines",
        )
        models = await provider.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_with_engines(self, tmp_path):
        """Test list_models finds engine files."""
        engine_dir = tmp_path / "engines"
        engine_dir.mkdir()
        engine_file = engine_dir / "test-model.engine"
        engine_file.write_bytes(b"fake engine data")

        provider = TRTLLMProvider(engine_dir=engine_dir)
        models = await provider.list_models()

        assert len(models) == 1
        assert models[0].name == "test-model"
        assert models[0].provider == ProviderType.TRTLLM

    @pytest.mark.asyncio
    async def test_health_check_not_available(self):
        """Test health_check when TensorRT-LLM not installed."""
        provider = TRTLLMProvider()
        if not provider.is_available:
            result = await provider.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check_missing_dirs(self, tmp_path):
        """Test health_check fails with missing configured directories."""
        provider = TRTLLMProvider(
            engine_dir=tmp_path / "nonexistent",
        )
        # Even if trtllm is available, missing dirs should fail
        if provider.is_available:
            result = await provider.health_check()
            assert result is False

    @pytest.mark.asyncio
    async def test_generate_not_available(self):
        """Test generate raises when TensorRT-LLM not installed."""
        provider = TRTLLMProvider()
        if not provider.is_available:
            messages = [Message(role="user", content="Hello")]
            with pytest.raises(RuntimeError, match="not installed"):
                await provider.generate(messages, "test-model")

    @pytest.mark.asyncio
    async def test_generate_stream_not_available(self):
        """Test generate_stream raises when TensorRT-LLM not installed."""
        provider = TRTLLMProvider()
        if not provider.is_available:
            messages = [Message(role="user", content="Hello")]
            with pytest.raises(RuntimeError, match="not installed"):
                async for _ in provider.generate_stream(messages, "test-model"):
                    pass

    @pytest.mark.asyncio
    async def test_pull_model_hf(self):
        """Test pull_model for HuggingFace models."""
        provider = TRTLLMProvider()
        messages = []
        async for msg in provider.pull_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
            messages.append(msg)

        # Should get success messages for HF models
        if provider.is_available:
            assert any(m.get("status") == "success" for m in messages)
        else:
            assert any(m.get("status") == "error" for m in messages)

    @pytest.mark.asyncio
    async def test_pull_model_local(self):
        """Test pull_model for local model names."""
        provider = TRTLLMProvider()
        messages = []
        async for msg in provider.pull_model("local-model"):
            messages.append(msg)

        # Should get info messages about building engines
        if provider.is_available:
            assert any("build" in m.get("message", "").lower() for m in messages)

    def test_cleanup(self):
        """Test cleanup method runs without error."""
        provider = TRTLLMProvider()
        provider.cleanup()
        assert provider._llm_instances == {}
        assert provider._current_model is None
