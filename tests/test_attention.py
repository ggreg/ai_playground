"""Tests for attention mechanisms."""

import pytest
import torch

from ai_playground.models.attention import GroupedQueryAttention
from ai_playground.models.config import TransformerConfig
from ai_playground.models.layers import precompute_rope_frequencies, apply_rope, RMSNorm, SwiGLU


@pytest.fixture
def device():
    return torch.device("cpu")


class TestRoPE:
    def test_frequencies_shape(self):
        freqs = precompute_rope_frequencies(64, 128)
        assert freqs.shape == (128, 32)  # (seq_len, head_dim // 2)
        assert freqs.is_complex()

    def test_rope_preserves_shape(self):
        head_dim = 64
        seq_len = 32
        freqs = precompute_rope_frequencies(head_dim, seq_len)

        q = torch.randn(2, seq_len, 8, head_dim)
        k = torch.randn(2, seq_len, 4, head_dim)
        q_rot, k_rot = apply_rope(q, k, freqs)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_changes_values(self):
        head_dim = 64
        freqs = precompute_rope_frequencies(head_dim, 16)
        q = torch.randn(1, 16, 4, head_dim)
        k = torch.randn(1, 16, 4, head_dim)
        q_rot, k_rot = apply_rope(q, k, freqs)

        # RoPE should change values (not be identity)
        assert not torch.allclose(q, q_rot, atol=1e-5)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(512)
        x = torch.randn(2, 32, 512)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(256)
        x = torch.randn(4, 16, 256) * 100  # Large values
        out = norm(x)
        # RMS of output should be ~1 (modulo learned weight)
        rms = out.pow(2).mean(-1).sqrt()
        assert rms.mean().item() < 5  # Reasonable range


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLU(512, 1024)
        x = torch.randn(2, 32, 512)
        out = ffn(x)
        assert out.shape == (2, 32, 512)


class TestGroupedQueryAttention:
    @pytest.mark.parametrize("n_kv_heads", [8, 4, 2, 1])
    def test_attention_variants(self, n_kv_heads):
        """Test MHA (8), GQA (4, 2), and MQA (1)."""
        config = TransformerConfig(dim=512, n_heads=8, n_kv_heads=n_kv_heads, max_seq_len=64)
        attn = GroupedQueryAttention(config)
        freqs = precompute_rope_frequencies(config.head_dim, 64)

        x = torch.randn(2, 32, 512)
        out = attn(x, freqs)
        assert out.shape == (2, 32, 512)

    def test_kv_cache(self):
        config = TransformerConfig(dim=256, n_heads=4, n_kv_heads=2, max_seq_len=64)
        attn = GroupedQueryAttention(config)
        freqs = precompute_rope_frequencies(config.head_dim, 64)

        # Prefill
        x = torch.randn(1, 16, 256)
        out1 = attn(x, freqs, use_cache=True)
        assert attn.cache_k is not None
        assert attn.cache_k.shape[1] == 16

        # Decode one token
        x2 = torch.randn(1, 1, 256)
        freqs2 = precompute_rope_frequencies(config.head_dim, 17)
        out2 = attn(x2, freqs2, use_cache=True)
        assert attn.cache_k.shape[1] == 17

        # Reset
        attn.reset_cache()
        assert attn.cache_k is None

    def test_causal_masking(self):
        """Verify that future tokens don't affect current token output."""
        config = TransformerConfig(dim=128, n_heads=4, n_kv_heads=2, max_seq_len=32)
        attn = GroupedQueryAttention(config)
        freqs = precompute_rope_frequencies(config.head_dim, 32)

        torch.manual_seed(0)
        x = torch.randn(1, 8, 128)

        # Full sequence
        out_full = attn(x, freqs)

        # Only first 4 tokens
        out_partial = attn(x[:, :4], freqs)

        # First 4 positions should be identical (causal = no future leakage)
        assert torch.allclose(out_full[:, :4], out_partial, atol=1e-5)
