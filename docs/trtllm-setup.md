# TensorRT-LLM Setup Guide

This guide covers setting up TensorRT-LLM for high-performance inference with Animus.

## Overview

TensorRT-LLM is NVIDIA's high-performance inference library for Large Language Models. It provides:
- 2-4x faster inference compared to standard frameworks
- Optimized memory usage with paged KV caching
- Support for quantization (FP8, INT8, INT4)
- Multi-GPU support with tensor/pipeline parallelism

## Requirements

### Hardware
- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Hopper, Ada)
- Minimum 8GB GPU memory for 7B models
- For Jetson: Orin series (AGX Orin, Orin NX, Orin Nano)

### Software
- CUDA 12.0+
- cuDNN 8.9+
- TensorRT 9.0+
- Python 3.10+

## Installation

### Desktop/Server (x86_64)

```bash
# Install TensorRT-LLM from PyPI
pip install tensorrt-llm

# Verify installation
python -c "from tensorrt_llm import LLM; print('TensorRT-LLM installed successfully')"
```

### NVIDIA Jetson (Orin)

Jetson requires building TensorRT-LLM from source due to the ARM64 architecture.

#### Prerequisites

```bash
# Ensure JetPack 5.1+ is installed
sudo apt update && sudo apt upgrade

# Install build dependencies
sudo apt install -y \
    python3-dev \
    python3-pip \
    cmake \
    build-essential \
    git

# Install CUDA toolkit (if not already present)
sudo apt install -y nvidia-jetpack
```

#### Build from Source

```bash
# Clone TensorRT-LLM
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# Install Python dependencies
pip install -r requirements.txt

# Build for Jetson (this takes 30-60 minutes)
python scripts/build_wheel.py --clean --trt_root /usr/lib/aarch64-linux-gnu

# Install the built wheel
pip install build/tensorrt_llm*.whl
```

## Configuration

### Configure Animus for TensorRT-LLM

Edit `~/.animus/config.yaml`:

```yaml
model:
  provider: trtllm
  model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0

trtllm:
  model_dir: ~/.animus/models/trtllm
  engine_dir: ~/.animus/engines
  tp_size: 1  # Tensor parallelism (multi-GPU)
  pp_size: 1  # Pipeline parallelism (multi-GPU)
```

## Usage

### Using HuggingFace Models Directly

TensorRT-LLM can load and optimize models from HuggingFace automatically:

```bash
# Start Animus with a HuggingFace model
animus rise

# The model will be downloaded and optimized on first use
```

Supported models include:
- LLaMA / LLaMA 2 / LLaMA 3
- Mistral / Mixtral
- Qwen / Qwen2
- Phi-2 / Phi-3
- GPT-2 / GPT-J / GPT-NeoX
- Falcon
- And many more

### Building Custom Engines (Advanced)

For maximum performance, you can pre-compile TensorRT engines:

```bash
# Download model weights
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./tinyllama

# Build TensorRT engine
trtllm-build \
    --checkpoint_dir ./tinyllama \
    --output_dir ./engines/tinyllama \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 2048 \
    --max_output_len 512

# Use the pre-built engine with Animus
# Edit config.yaml to point engine_dir to ./engines
```

#### Jetson-Specific Build Options

```bash
# For Jetson Orin Nano (8GB)
trtllm-build \
    --checkpoint_dir ./model \
    --output_dir ./engine \
    --gemm_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_output_len 256 \
    --use_paged_context_fmha enable

# For larger Jetson devices (AGX Orin)
trtllm-build \
    --checkpoint_dir ./model \
    --output_dir ./engine \
    --gemm_plugin float16 \
    --max_batch_size 4 \
    --max_input_len 4096 \
    --max_output_len 1024
```

## Recommended Models for Jetson

| Device | RAM | Recommended Models |
|--------|-----|-------------------|
| Orin Nano (8GB) | 8GB shared | TinyLlama-1.1B, Phi-2 (2.7B) |
| Orin NX (16GB) | 16GB shared | Mistral-7B-Q4, LLaMA-7B-Q4 |
| AGX Orin (32GB) | 32GB shared | LLaMA-13B-Q4, Mistral-7B-FP16 |
| AGX Orin (64GB) | 64GB shared | LLaMA-30B-Q4, Mixtral-8x7B-Q4 |

## Troubleshooting

### "TensorRT-LLM is not installed"

Ensure TensorRT-LLM is properly installed:

```bash
pip install tensorrt-llm
# or for Jetson, follow the build instructions above
```

### "CUDA out of memory"

Try:
1. Use a smaller model or more aggressive quantization (Q4)
2. Reduce `max_batch_size` in engine build
3. Reduce `max_input_len` and `max_output_len`
4. Enable paged context FMHA: `--use_paged_context_fmha enable`

### "Engine build fails on Jetson"

Common issues:
1. **Insufficient swap space**: Add more swap
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Wrong CUDA architecture**: Specify the correct compute capability
   ```bash
   # Orin: sm_87
   export CUDA_ARCHITECTURES=87
   ```

3. **Missing dependencies**: Reinstall JetPack
   ```bash
   sudo apt install --reinstall nvidia-jetpack
   ```

### Slow first inference

The first inference is slow because TensorRT-LLM:
1. Downloads the model (if using HuggingFace)
2. Converts and optimizes the model
3. Builds the TensorRT engine

Subsequent runs use the cached engine and are much faster.

## Performance Tips

1. **Use pre-built engines** for production deployments
2. **Enable FP8 quantization** on Hopper/Ada GPUs for 2x speedup
3. **Use tensor parallelism** for multi-GPU setups
4. **Tune batch size** based on your workload
5. **Enable KV cache reuse** for chat applications

## Further Reading

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/)
- [LLM API Reference](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html)
