"""Attention implementations: MHA, GQA, MQA with optional Flash Attention."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TransformerConfig
from .layers import apply_rope


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (Ainslie et al., 2023).

    Generalizes MHA, MQA, and GQA through n_kv_heads:
    - n_kv_heads == n_heads → standard Multi-Head Attention (MHA)
    - n_kv_heads == 1       → Multi-Query Attention (MQA, Shazeer 2019)
    - 1 < n_kv_heads < n_heads → Grouped-Query Attention (GQA)

    GQA reduces KV cache memory by sharing K/V heads across groups of Q heads,
    with minimal quality loss compared to full MHA.

    Papers:
    - GQA: https://arxiv.org/abs/2305.13245
    - MQA: https://arxiv.org/abs/1911.02150
    - Flash Attention (used via SDPA): https://arxiv.org/abs/2205.14135
    See also: docs/PAPERS.md § Attention Variants
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # number of Q heads per KV head

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # KV cache (populated during inference)
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA.

        (batch, seq, n_kv_heads, head_dim) -> (batch, seq, n_heads, head_dim)
        """
        if self.n_rep == 1:
            return x
        bs, seq_len, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bs, seq_len, n_kv_heads, self.n_rep, head_dim)
            .reshape(bs, seq_len, self.n_heads, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q, k = apply_rope(q, k, freqs_cis)

        # KV cache for autoregressive generation
        if use_cache:
            if self.cache_k is not None:
                k = torch.cat([self.cache_k, k], dim=1)
                v = torch.cat([self.cache_v, v], dim=1)
            self.cache_k = k
            self.cache_v = v

        # Expand KV heads for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Transpose to (batch, n_heads, seq_len, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Try to use Flash Attention via scaled_dot_product_attention
        if mask is None and not use_cache and hasattr(F, "scaled_dot_product_attention"):
            # PyTorch SDPA will use Flash Attention or memory-efficient attention
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0
            )
        else:
            # Manual attention for when we need explicit mask control
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
            scores = self.dropout(scores)
            output = torch.matmul(scores, v)

        # (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, dim)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        return self.wo(output)
