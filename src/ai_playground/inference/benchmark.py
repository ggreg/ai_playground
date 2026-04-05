"""Inference benchmarking: measure tokens/sec, latency, and memory."""

import time

import torch

from ..models.transformer import Transformer
from .generate import generate


@torch.no_grad()
def benchmark_generation(
    model: Transformer,
    prompt_len: int = 128,
    gen_len: int = 256,
    batch_size: int = 1,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Benchmark autoregressive generation performance.

    Measures:
    - Prefill latency and tokens/sec (processing the prompt)
    - Decode latency and tokens/sec (generating new tokens)
    - Peak GPU memory usage
    - Time to first token (TTFT)

    These are the key metrics for LLM serving systems.
    """
    device = next(model.parameters()).device
    model.eval()

    prompt = torch.randint(0, model.config.vocab_size, (batch_size, prompt_len), device=device)

    # Track GPU memory if available
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup_runs):
        model.reset_caches()
        _ = generate(model, prompt, max_new_tokens=min(16, gen_len), temperature=0)

    # Benchmark
    prefill_times = []
    decode_times = []
    total_times = []

    for _ in range(benchmark_runs):
        model.reset_caches()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Prefill phase
        t_start = time.perf_counter()
        with torch.amp.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits = model(prompt, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        # Decode phase
        tokens = prompt
        for _ in range(gen_len):
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            with torch.amp.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                logits = model(next_token, use_cache=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        model.reset_caches()

        prefill_times.append(t_prefill - t_start)
        decode_times.append(t_end - t_prefill)
        total_times.append(t_end - t_start)

    # Compute statistics
    avg = lambda xs: sum(xs) / len(xs)
    results = {
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "batch_size": batch_size,
        "device": str(device),
        "dtype": str(dtype),
        "prefill_ms": avg(prefill_times) * 1000,
        "prefill_tokens_per_sec": prompt_len * batch_size / avg(prefill_times),
        "decode_ms": avg(decode_times) * 1000,
        "decode_tokens_per_sec": gen_len * batch_size / avg(decode_times),
        "time_to_first_token_ms": avg(prefill_times) * 1000,
        "total_ms": avg(total_times) * 1000,
    }

    if device.type == "cuda":
        results["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return results


def print_benchmark(results: dict):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"Inference Benchmark Results")
    print(f"{'=' * 60}")
    print(f"Config: prompt={results['prompt_len']}, gen={results['gen_len']}, "
          f"batch={results['batch_size']}")
    print(f"Device: {results['device']}, dtype: {results['dtype']}")
    print(f"{'-' * 60}")
    print(f"Prefill:  {results['prefill_ms']:.1f}ms "
          f"({results['prefill_tokens_per_sec']:.0f} tok/s)")
    print(f"Decode:   {results['decode_ms']:.1f}ms "
          f"({results['decode_tokens_per_sec']:.0f} tok/s)")
    print(f"TTFT:     {results['time_to_first_token_ms']:.1f}ms")
    print(f"Total:    {results['total_ms']:.1f}ms")
    if "peak_memory_mb" in results:
        print(f"Peak GPU: {results['peak_memory_mb']:.1f} MB")
    print(f"{'=' * 60}\n")
