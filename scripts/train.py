#!/usr/bin/env python3
"""Train a transformer model.

Usage:
    # Quick test on CPU/MPS with tiny model
    uv run python scripts/train.py --config configs/tiny.yaml

    # Train with mixed precision on GPU
    uv run python scripts/train.py --config configs/small.yaml --dtype bfloat16

    # Distributed (multi-GPU)
    torchrun --nproc_per_node=4 scripts/train.py --config configs/medium.yaml --distributed fsdp
"""

import argparse
import yaml

import torch

from ai_playground.models import Transformer, TransformerConfig
from ai_playground.training import Trainer, TrainingConfig, create_dataloader


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dtype", type=str, default=None, help="Override dtype (float32/float16/bfloat16)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--distributed", type=str, default=None, choices=["ddp", "fsdp"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Apply CLI overrides
    if args.dtype:
        train_cfg["dtype"] = args.dtype
    if args.max_steps:
        train_cfg["max_steps"] = args.max_steps

    # Build model
    transformer_config = TransformerConfig(**model_cfg)
    model = Transformer(transformer_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters")
    print(f"Config: {transformer_config}")

    # Handle distributed
    if args.distributed:
        from ai_playground.training.distributed import setup_distributed, wrap_ddp, wrap_fsdp
        rank, local_rank, world_size = setup_distributed()
        print(f"Rank {rank}/{world_size}, local_rank={local_rank}")
        if args.distributed == "ddp":
            model = wrap_ddp(model, local_rank)
        else:
            model = wrap_fsdp(model, local_rank)

    # Create synthetic data for testing (replace with real data for actual training)
    seq_len = model_cfg.get("max_seq_len", 512)
    vocab_size = model_cfg.get("vocab_size", 32000)
    total_tokens = train_cfg.get("batch_size", 8) * seq_len * 100
    data = torch.randint(0, vocab_size, (total_tokens,))

    split = int(0.9 * len(data))
    train_loader = create_dataloader(data[:split], seq_len, train_cfg.get("batch_size", 8))
    eval_loader = create_dataloader(data[split:], seq_len, train_cfg.get("batch_size", 8), shuffle=False)

    # Train
    training_config = TrainingConfig(**train_cfg)
    trainer = Trainer(model, training_config, train_loader, eval_loader)
    print(f"Training on {trainer.device}")
    logs = trainer.train()

    # Final eval
    eval_loss = trainer.evaluate()
    print(f"\nFinal eval loss: {eval_loss:.4f}")

    if args.distributed:
        from ai_playground.training.distributed import cleanup_distributed
        cleanup_distributed()


if __name__ == "__main__":
    main()
