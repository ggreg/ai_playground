#!/usr/bin/env python3
"""Convenience wrapper for launching distributed training.

Usage:
    # DDP on 4 GPUs
    uv run python scripts/launch_distributed.py --nproc 4 --mode ddp --config configs/small.yaml

    # FSDP on 8 GPUs
    uv run python scripts/launch_distributed.py --nproc 8 --mode fsdp --config configs/medium.yaml

This is equivalent to running:
    torchrun --nproc_per_node=N scripts/train.py --config ... --distributed MODE
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch distributed training")
    parser.add_argument("--nproc", type=int, required=True, help="Number of GPUs")
    parser.add_argument("--mode", type=str, required=True, choices=["ddp", "fsdp"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        "scripts/train.py",
        "--config", args.config,
        "--distributed", args.mode,
        "--dtype", args.dtype,
    ]

    if args.max_steps:
        cmd.extend(["--max-steps", str(args.max_steps)])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
