#!/usr/bin/env python3
"""Benchmark inference performance.

Usage:
    uv run python scripts/benchmark.py --config configs/tiny.yaml
    uv run python scripts/benchmark.py --config configs/small.yaml --dtype float16 --prompt-len 512
"""

import argparse
import yaml

import torch

from ai_playground.models import Transformer, TransformerConfig
from ai_playground.inference.benchmark import benchmark_generation, print_benchmark


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--gen-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = TransformerConfig(**cfg.get("model", {}))
    model = Transformer(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params")

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    results = benchmark_generation(
        model,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        batch_size=args.batch_size,
        benchmark_runs=args.runs,
        dtype=dtype_map[args.dtype],
    )
    print_benchmark(results)

    # Memory estimate
    mem = model.estimate_memory_mb(args.batch_size, args.prompt_len + args.gen_len)
    print("Memory estimate:")
    for k, v in mem.items():
        print(f"  {k}: {v:.1f} MB")


if __name__ == "__main__":
    main()
