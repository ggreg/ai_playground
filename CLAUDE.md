# CLAUDE.md — AI Playground

## Project Overview

Educational repository for learning LLM internals, training optimization, distributed training, and GPU performance. Contains a from-scratch LLaMA-style transformer and interactive notebooks.

## Build & Test Commands

```bash
# Install all dependencies (including dev tools)
uv sync --extra dev

# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_attention.py -v

# Lint
uv run ruff check src/ tests/

# Quick training sanity check (CPU/MPS, ~30 seconds)
uv run python scripts/train.py --config configs/tiny.yaml --max-steps 20

# Benchmark inference
uv run python scripts/benchmark.py --config configs/tiny.yaml
```

## Architecture

**LLaMA-style decoder-only transformer** in `src/ai_playground/models/`:
- `config.py` — `TransformerConfig` dataclass with predefined TINY/SMALL/MEDIUM configs
- `layers.py` — RMSNorm, RoPE (precompute + apply), SwiGLU
- `attention.py` — `GroupedQueryAttention` supporting MHA/GQA/MQA via `n_kv_heads`
- `transformer.py` — Full model: embedding, N x TransformerBlock, output head

**Training** in `src/ai_playground/training/`:
- `trainer.py` — `Trainer` class with mixed precision, grad accumulation, cosine LR
- `data.py` — `TextDataset` from pre-tokenized tensors, `create_dataloader`
- `distributed.py` — DDP/FSDP wrappers (used with torchrun)
- `optimizers.py` — `AdamWFromScratch` for educational purposes

**Inference** in `src/ai_playground/inference/`:
- `generate.py` — Autoregressive generation with KV cache, top-p sampling
- `benchmark.py` — Measures prefill/decode latency, tokens/sec, TTFT, peak memory
- `quantize.py` — Basic INT8 absmax quantization

**Profiling** in `src/ai_playground/profiling/`:
- `flops.py` — FLOP estimation, MFU calculation, GPU peak TFLOPS database
- `memory.py` — `MemoryTracker`, `track_memory` context manager
- `nsight.py` — NVTX range markers, PyTorch profiler wrapper

**Agents** in `src/ai_playground/agents/` (optional extra: `uv sync --extra agents`):
- `llm.py` — `LLMBackend` protocol with `ClaudeBackend` (Anthropic API) and `LocalTransformerBackend` (this repo's tiny transformer)
- `tools.py` — `Tool` dataclass, `ToolRegistry`, builtin tools (file I/O, calculator, shell, web fetch)
- `memory.py` — `ConversationMemory` (bounded + optional summarization), `VectorMemory` (similarity-search long-term store)
- `agent.py` — `Agent` ReAct loop — model + tools + memory in <100 lines
- `planner.py` — `PlanAndExecute` for multi-step decomposition; optional reflection
- `multi_agent.py` — `Supervisor` + `Worker` orchestration

**Documentation** in `docs/`:
- `LEARNING_PATH.md` — 6-phase curriculum from internals to agents
- `PAPERS.md` — Key papers organized by topic with links and summaries
- `CONCEPTS.md` — FAQ-style deep dives on conceptual questions (embeddings, cross-entropy, bias, etc.). When the user asks a substantive conceptual question that would benefit future readers, add the answer here following the existing Q&A format.

## Code Conventions

- **Python 3.11+** with type hints (use `X | None` not `Optional[X]`)
- **No bias terms** in Linear layers (modern LLM convention) — use `bias=False`
- **Pre-norm architecture** — always normalize before attention/FFN, not after
- Line length: 100 characters (ruff)
- Tests use pytest with parametrize for testing attention variants
- Configs are YAML files in `configs/` with `model:` and `training:` sections

## Educational Focus

This is a learning repository. When adding or modifying code:

- **Explain the "why" in docstrings** — not just what the code does, but why this approach matters for LLM performance. Reference papers where relevant.
- **Include concrete numbers** — parameter counts, memory usage, speedup factors. Vague claims like "faster" are not useful; "2.3x faster, 0.4x memory" is.
- **Make tradeoffs explicit** — every optimization has a cost (complexity, quality, compatibility). State both sides.
- **Keep implementations readable over clever** — this is for learning, not production. Prefer explicit loops over obscure tensor tricks when the logic is clearer.
- **Benchmark claims** — if you say something is faster, include or reference a benchmark. The `inference/benchmark.py` and `profiling/` modules exist for this purpose.

## Notebook Conventions

Notebooks live in `notebooks/` organized by topic module:
- `01_transformer_internals/` — model architecture components
- `02_training_optimization/` — training efficiency techniques
- `03_distributed_training/` — multi-GPU and multi-node
- `04_inference_optimization/` — serving and generation speed
- `05_gpu_nvidia_tools/` — CUDA, Triton, profiling tools
- `06_agents/` — building LLM agents from scratch (tool use, memory, planning)

Each notebook should:
1. Start with a markdown cell explaining what we're exploring and why it matters
2. Build concepts incrementally — implement from scratch before using library versions
3. Include visualizations (matplotlib) for attention patterns, memory usage, latency, etc.
4. End with a "Key Takeaways" section and pointer to the next notebook
5. Import from `ai_playground` package for reusable components: `from ai_playground.models import Transformer, TransformerConfig`
6. Use `sys.path.insert(0, '../src')` at the top for imports

## Important Constraints

- Model weights (`.pt`, `.bin`, `.safetensors`) are gitignored — never commit them
- The `data/` and `checkpoints/` directories are gitignored
- Wandb logs (`wandb/`) are gitignored
- GPU-specific packages (`triton`, `flash-attn`, `bitsandbytes`) are optional extras — core code must work on CPU/MPS
- All attention variants (MHA, GQA, MQA) must be testable via the `n_kv_heads` parameter on `TransformerConfig`
- The `TransformerConfig.num_params()` method should stay accurate when model architecture changes
