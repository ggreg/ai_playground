---
name: benchmark
description: Run benchmarks comparing model variants, measuring inference speed, or profiling GPU performance. Use when the user wants to measure performance, compare approaches, or optimize.
user_invocable: true
---

# Run Benchmarks

When benchmarking in this AI playground, follow these practices:

## Available Benchmarking Tools

1. **Inference benchmark** (`scripts/benchmark.py`):
   ```bash
   uv run python scripts/benchmark.py --config configs/tiny.yaml --dtype float32
   uv run python scripts/benchmark.py --config configs/small.yaml --dtype bfloat16 --prompt-len 512
   ```
   Measures: prefill latency, decode tokens/sec, TTFT, peak GPU memory.

2. **PyTorch profiler** (`scripts/profile_model.py`):
   ```bash
   uv run python scripts/profile_model.py --config configs/tiny.yaml
   ```
   Outputs: Chrome trace, top CUDA kernels, FLOP estimate, memory breakdown.

3. **In-code profiling** — Use utilities from `ai_playground.profiling`:
   ```python
   from ai_playground.profiling import MemoryTracker, estimate_flops, compute_mfu
   from ai_playground.profiling.memory import track_memory
   ```

4. **Quick training benchmark**:
   ```bash
   uv run python scripts/train.py --config configs/tiny.yaml --max-steps 50
   ```
   Reports: tokens/sec, ms/step, loss progression.

## Benchmarking Rules

- **Always warm up** — Run at least 2-3 warmup iterations before measuring. Cold starts include JIT compilation and memory allocation overhead.
- **Sync CUDA** — Call `torch.cuda.synchronize()` before timing CUDA operations. Without this, you're measuring kernel launch time, not execution time.
- **Report multiple runs** — Use at least 5 benchmark runs and report the average. Single measurements are noisy.
- **State the hardware** — Always report: GPU model, dtype, batch size, sequence length.
- **Compare apples to apples** — Same batch size, sequence length, and dtype when comparing approaches. Only vary one thing at a time.
- **Report absolute AND relative** — "GQA: 12.3ms (1.4x faster than MHA at 17.2ms)" is better than just "GQA is faster".

## When Comparing Approaches

Structure the output as a table:

```
Variant          | Params | Latency  | Memory  | Tokens/sec
MHA (8 KV heads) |  20.1M |  15.2ms  | 142MB   |    33,800
GQA (4 KV heads) |  18.3M |  12.1ms  | 98MB    |    42,500
MQA (1 KV head)  |  17.1M |  10.8ms  | 71MB    |    47,600
```

## MFU Calculation

For training benchmarks, compute Model FLOP Utilization:

```python
from ai_playground.profiling.flops import compute_mfu
mfu = compute_mfu(config, seq_len=1024, batch_size=8, step_time_sec=0.15, gpu_name="A100_80GB")
print(f"MFU: {mfu['mfu_percent']:.1f}%")
```

Reference MFU values: Naive PyTorch ~20-30%, with Flash Attention + compile ~40-50%, highly optimized ~50-60%.
