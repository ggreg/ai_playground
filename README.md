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

## How This Compares to Similar Projects

| | **ai-playground** | **[nanochat](https://github.com/karpathy/nanochat)** (Karpathy) | **[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)** (Raschka) |
|---|---|---|---|
| **Architecture** | LLaMA-style (RMSNorm, RoPE, SwiGLU, GQA) | GPT-style (LayerNorm, learned pos, GELU) | GPT-2 style (LayerNorm, learned pos, GELU) |
| **Purpose** | Learn internals + training + GPU performance | Train a real chatbot end-to-end for <$100 | Book companion — build an LLM step by step |
| **Model sizes** | 10M / 125M / 350M | Variable via `--depth` (up to ~1.6B) | Small educational; loads GPT-2 weights (124M–1.5B) |
| **Scope** | Architecture, training, inference, profiling, distributed | Full pipeline: pretrain, SFT, RL, eval, chat UI | Tokenization, pretraining, finetuning (classification + instruction) |
| **Attention** | MHA/GQA/MQA configurable | MHA | MHA (GQA in bonus) |
| **Distributed** | DDP + FSDP | DDP via torchrun | DDP in bonus appendix |
| **Unique strengths** | GPU profiling, FLOP/MFU analysis, quantization, benchmarking | Speedrun leaderboard, auto-scaling hyperparams, RL + chat UI | Best-in-class pedagogy, 90K+ stars, published book |

**In short:** This project uses a more modern architecture (LLaMA conventions) and uniquely focuses on GPU performance and profiling. nanochat goes further on the product side (SFT, RL, chat). Raschka's project is the most beginner-friendly with structured chapters and exercises.

## Hands-On Curriculum

A practical path through the repo. Each phase involves running code, modifying it, and verifying your understanding.

### Phase 1: Understand the Model

1. **Run the tests** — `uv run pytest -v`. Read `tests/test_attention.py` to see how MHA/GQA/MQA are exercised, then read the implementation in `src/ai_playground/models/attention.py`.

2. **Work through the attention notebook** — `uv run jupyter notebook notebooks/01_transformer_internals/01_attention_mechanisms.ipynb`. Try changing `n_kv_heads`, visualize attention patterns, break things on purpose.

3. **Inspect a model in the REPL:**
   ```python
   from ai_playground.models import Transformer, TransformerConfig
   model = Transformer(TransformerConfig.TINY)
   print(model)
   print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
   ```
   Change config values and see how parameter counts change.

### Phase 2: Train and Benchmark

4. **Train the tiny model on CPU** (~30 seconds):
   ```bash
   uv run python scripts/train.py --config configs/tiny.yaml --max-steps 20
   ```

5. **Benchmark inference** — run on `tiny.yaml`, then `small.yaml` and compare tokens/sec, TTFT, and peak memory:
   ```bash
   uv run python scripts/benchmark.py --config configs/tiny.yaml
   ```

6. **Profile memory and FLOPs** — try to predict the numbers before running, then check:
   ```bash
   uv run python scripts/profile_model.py --config configs/tiny.yaml
   ```

### Phase 3: Modify and Experiment

Pick one or more exercises:

- **Swap an architecture component** — replace RMSNorm with LayerNorm in `layers.py`, or SwiGLU with GELU, or GQA with MHA. Measure the impact on loss, speed, and memory using the profiler and benchmarks.
- **Build a missing notebook** — the [learning path](docs/LEARNING_PATH.md) references many notebooks that don't exist yet. Pick one (e.g., mixed precision, positional encodings, or LR schedules) and build it.
- **Extend the training loop** — add wandb logging to `trainer.py`, implement gradient checkpointing, or add a perplexity eval on a held-out set.
- **Try quantization** — quantize the tiny model to INT8 using `inference/quantize.py`, benchmark it vs FP32, and compare quality.

### Phase 4: Scale Up (GPU)

Follow [docs/CLOUD_SETUP.md](docs/CLOUD_SETUP.md) to rent a cloud GPU, then:

- Train `small.yaml` on a single A100 with BF16
- Train `medium.yaml` across multiple GPUs with DDP and FSDP
- Compare MFU between setups using `profiling/flops.py`

### The Learning Loop

For each topic: **read the code, form a hypothesis, run an experiment, verify.** For example: "GQA with 4 KV heads should use ~3x less KV cache memory than MHA with 12 heads" — then write a script to measure it and see if you're right.

See [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md) for the full topic-by-topic sequence.

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
