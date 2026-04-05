"""Tests for the full transformer model."""

import pytest
import torch

from ai_playground.models import Transformer, TransformerConfig
from ai_playground.models.config import TINY, SMALL


class TestTransformerConfig:
    def test_tiny_params(self):
        assert TINY.num_params() > 5_000_000  # At least 5M
        assert TINY.num_params() < 20_000_000  # Less than 20M

    def test_small_params(self):
        assert SMALL.num_params() > 80_000_000

    def test_head_dim(self):
        cfg = TransformerConfig(dim=512, n_heads=8)
        assert cfg.head_dim == 64

    def test_kv_heads_default(self):
        cfg = TransformerConfig(n_heads=8, n_kv_heads=None)
        assert cfg.kv_heads == 8  # MHA by default

    def test_ffn_hidden_dim_multiple_of_64(self):
        cfg = TransformerConfig(dim=512)
        assert cfg.ffn_hidden_dim % 64 == 0


class TestTransformer:
    @pytest.fixture
    def tiny_model(self):
        cfg = TransformerConfig(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
        )
        return Transformer(cfg)

    def test_forward_shape(self, tiny_model):
        tokens = torch.randint(0, 1000, (2, 32))
        logits = tiny_model(tokens)
        assert logits.shape == (2, 32, 1000)

    def test_forward_no_nan(self, tiny_model):
        tokens = torch.randint(0, 1000, (1, 16))
        logits = tiny_model(tokens)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_generation_with_cache(self, tiny_model):
        prompt = torch.randint(0, 1000, (1, 8))

        # Prefill
        logits = tiny_model(prompt, use_cache=True)
        assert logits.shape == (1, 8, 1000)

        # Decode
        next_token = logits[:, -1:, :].argmax(-1)
        logits2 = tiny_model(next_token, use_cache=True)
        assert logits2.shape == (1, 1, 1000)

        tiny_model.reset_caches()

    def test_loss_decreases(self, tiny_model):
        """Sanity check: loss should decrease on a single batch."""
        tokens = torch.randint(0, 1000, (4, 32))
        x, y = tokens[:, :-1], tokens[:, 1:]

        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        losses = []

        for _ in range(10):
            logits = tiny_model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 1000), y.reshape(-1))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease over 10 steps on same data
        assert losses[-1] < losses[0]

    def test_memory_estimate(self, tiny_model):
        mem = tiny_model.estimate_memory_mb(batch_size=1, seq_len=64)
        assert "parameters_mb" in mem
        assert "kv_cache_mb" in mem
        assert mem["parameters_mb"] > 0
