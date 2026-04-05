"""Tests for training utilities."""

import pytest
import torch

from ai_playground.models import Transformer, TransformerConfig
from ai_playground.training.data import TextDataset, create_dataloader
from ai_playground.training.trainer import Trainer, TrainingConfig
from ai_playground.training.optimizers import AdamWFromScratch
from ai_playground.profiling.flops import estimate_flops, GPU_PEAK_TFLOPS


class TestDataset:
    def test_text_dataset(self):
        data = torch.arange(1000)
        ds = TextDataset(data, seq_len=100)
        assert len(ds) == 9  # (1000 - 1) // 100

        x, y = ds[0]
        assert x.shape == (100,)
        assert y.shape == (100,)
        assert (y[:-1] == x[1:]).all()  # y is shifted by 1

    def test_dataloader(self):
        data = torch.arange(10000)
        loader = create_dataloader(data, seq_len=64, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        x, y = batch
        assert x.shape == (4, 64)
        assert y.shape == (4, 64)


class TestTrainer:
    def test_training_runs(self):
        cfg = TransformerConfig(vocab_size=100, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
        model = Transformer(cfg)

        data = torch.randint(0, 100, (5000,))
        loader = create_dataloader(data, seq_len=32, batch_size=4)

        train_cfg = TrainingConfig(max_steps=5, log_interval=1, warmup_steps=2)
        trainer = Trainer(model, train_cfg, loader)

        logs = trainer.train()
        assert len(logs) == 5
        assert all("loss" in log for log in logs)


class TestAdamWFromScratch:
    def test_converges(self):
        """AdamW from scratch should minimize a simple quadratic."""
        x = torch.randn(10, requires_grad=True)
        opt = AdamWFromScratch([x], lr=0.1, weight_decay=0.0)

        for _ in range(100):
            loss = (x**2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert x.abs().max().item() < 0.1


class TestFlops:
    def test_estimate_flops(self):
        cfg = TransformerConfig(dim=512, n_layers=8, n_heads=8)
        flops = estimate_flops(cfg, seq_len=512, batch_size=4)

        assert flops["forward_tflops"] > 0
        assert 0 < flops["attention_fraction"] < 1
        assert 0 < flops["ffn_fraction"] < 1
        assert abs(flops["attention_fraction"] + flops["ffn_fraction"] - 1.0) < 0.01

    def test_gpu_peak_tflops(self):
        assert "A100_80GB" in GPU_PEAK_TFLOPS
        assert "H100_SXM" in GPU_PEAK_TFLOPS
