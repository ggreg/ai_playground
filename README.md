# AI Playground

A hands-on playground for experimenting with LLM internals, training optimization, distributed systems, and GPU performance. Built around a clean, from-scratch LLaMA-style transformer implementation.

## What's Inside

### Core Model (`src/ai_playground/models/`)

A modern decoder-only transformer with all the components used in production LLMs:

- **Grouped-Query Attention** — configurable MHA, GQA, or MQA via `n_kv_heads`
- **RoPE** — rotary positional embeddings
- **RMSNorm** — faster alternative to LayerNorm
- **SwiGLU** — gated FFN activation (LLaMA, Mistral, Gemma)
- **KV cache** — for efficient autoregressive generation
- Model configs from ~10M (tiny) to ~350M (medium) parameters

### Training (`src/ai_playground/training/`)

- Mixed precision training (FP16/BF16) with automatic loss scaling
- Gradient accumulation for simulating large batch sizes
- Cosine LR schedule with linear warmup
- AdamW from scratch (for learning) + PyTorch fused AdamW
- DDP and FSDP distributed training wrappers

### Inference (`src/ai_playground/inference/`)

- Autoregressive generation with KV cache (prefill + decode)
- Top-p (nucleus) sampling
- Inference benchmarking (tokens/sec, TTFT, peak memory)
- Basic INT8 absmax quantization

### Profiling (`src/ai_playground/profiling/`)

- GPU memory tracking and snapshots
- FLOP counting for transformer operations
- Model FLOP Utilization (MFU) calculation with GPU peak TFLOPS database
- Nsight Systems/Compute integration helpers
- PyTorch profiler wrapper (outputs Chrome traces)

### Notebooks (`notebooks/`)

Interactive Jupyter notebooks organized by topic:

| Module | Topics |
|--------|--------|
| `01_transformer_internals/` | Attention (MHA/GQA/MQA), positional encodings, activations, normalization, tokenizers |
| `02_training_optimization/` | Mixed precision, gradient accumulation, LR schedules, optimizers, data loading |
| `03_distributed_training/` | DDP, FSDP, tensor parallelism, pipeline parallelism, DeepSpeed ZeRO |
| `04_inference_optimization/` | KV cache, quantization, speculative decoding, continuous batching, Flash Attention |
| `05_gpu_nvidia_tools/` | CUDA basics, Triton kernels, Nsight profiling, TensorRT, NCCL |

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Train a tiny model (CPU/MPS, ~20M params)
uv run python scripts/train.py --config configs/tiny.yaml

# Benchmark inference
uv run python scripts/benchmark.py --config configs/tiny.yaml

# Profile a model
uv run python scripts/profile_model.py --config configs/tiny.yaml

# Open notebooks
uv run jupyter notebook notebooks/
```

### GPU Training

```bash
# Install GPU extras
uv sync --extra gpu --extra dev

# Single GPU with mixed precision
uv run python scripts/train.py --config configs/small.yaml --dtype bfloat16

# Multi-GPU with DDP
torchrun --nproc_per_node=4 scripts/train.py --config configs/small.yaml --distributed ddp

# Multi-GPU with FSDP (for models that don't fit on one GPU)
uv run python scripts/launch_distributed.py --nproc 8 --mode fsdp --config configs/medium.yaml
```

## Model Configs

| Config | Params | Layers | Dim | Heads | KV Heads | Use Case |
|--------|--------|--------|-----|-------|----------|----------|
| `tiny.yaml` | ~20M | 6 | 256 | 8 | 4 | Quick iteration, CPU/single GPU |
| `small.yaml` | ~125M | 12 | 768 | 12 | 4 | Single GPU training |
| `medium.yaml` | ~350M | 24 | 1024 | 16 | 4 | Multi-GPU training |

## Project Structure

```
ai-playground/
├── src/ai_playground/       # Python package
│   ├── models/              # Transformer, attention, layers, config
│   ├── training/            # Trainer, data, distributed, optimizers
│   ├── inference/           # Generation, quantization, benchmarking
│   └── profiling/           # Memory, FLOPs, Nsight, torch profiler
├── notebooks/               # Jupyter notebooks (5 topic modules)
├── scripts/                 # CLI entry points (train, benchmark, profile)
├── configs/                 # YAML model/training configs
└── tests/                   # pytest test suite
```
