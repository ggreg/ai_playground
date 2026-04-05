#!/usr/bin/env python3
"""Profile a model with PyTorch profiler.

Usage:
    # Basic profile
    uv run python scripts/profile_model.py --config configs/tiny.yaml

    # With Nsight Systems (requires NVIDIA tools)
    nsys profile -o trace uv run python scripts/profile_model.py --config configs/tiny.yaml
"""

import argparse
import yaml

import torch

from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.nsight import profile_with_torch
from ai_playground.profiling.flops import estimate_flops
from ai_playground.profiling.memory import print_memory_stats


def main():
    parser = argparse.ArgumentParser(description="Profile model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output", type=str, default="profile_trace")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = TransformerConfig(**cfg.get("model", {}))
    model = Transformer(model_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params on {device}")

    # FLOP estimate
    flops = estimate_flops(model_cfg, args.seq_len, args.batch_size)
    print(f"\nFLOP estimate (forward): {flops['forward_tflops']:.3f} TFLOPS")
    print(f"  Attention fraction: {flops['attention_fraction']:.1%}")
    print(f"  FFN fraction: {flops['ffn_fraction']:.1%}")

    # Memory estimate
    mem = model.estimate_memory_mb(args.batch_size, args.seq_len)
    print(f"\nMemory estimate:")
    for k, v in mem.items():
        print(f"  {k}: {v:.1f} MB")

    if torch.cuda.is_available():
        print_memory_stats()

    # Run profiler
    def make_input():
        return torch.randint(0, model_cfg.vocab_size, (args.batch_size, args.seq_len), device=device)

    print(f"\nProfiling {5} steps...")
    prof = profile_with_torch(model, make_input, output_path=args.output)
    print(f"\nTrace saved to {args.output}/ — open with TensorBoard or chrome://tracing")


if __name__ == "__main__":
    main()
