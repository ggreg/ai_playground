"""Full transformer model — LLaMA-style decoder-only architecture."""

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .config import TransformerConfig
from .layers import RMSNorm, SwiGLU, precompute_rope_frequencies


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Uses pre-norm (norm before attention/FFN) which is more stable
    for training and is the standard in all modern LLMs.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config.dim, config.ffn_hidden_dim, config.dropout)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Pre-norm + attention + residual
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask, use_cache)
        # Pre-norm + FFN + residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """Decoder-only transformer (LLaMA-style).

    Architecture choices:
    - Pre-norm with RMSNorm (simpler, faster than LayerNorm)
    - RoPE for positional encoding (relative, extrapolates well)
    - SwiGLU activation (better than GELU, used in LLaMA/Mistral)
    - Grouped-Query Attention (reduces KV cache, configurable)
    - No bias terms anywhere (modern convention)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_rope_frequencies(config.head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: (batch, seq_len) integer token IDs
            mask: optional attention mask
            use_cache: enable KV cache for autoregressive generation

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[:seq_len]

        # Build causal mask if not using flash attention path
        if mask is None and use_cache and seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask, use_cache)

        h = self.norm(h)
        return self.output(h)

    def reset_caches(self):
        for layer in self.layers:
            layer.attention.reset_cache()

    @torch.no_grad()
    def estimate_memory_mb(self, batch_size: int = 1, seq_len: int | None = None) -> dict:
        """Estimate GPU memory usage for parameters, activations, and KV cache."""
        seq_len = seq_len or self.config.max_seq_len
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())

        # KV cache: 2 * n_layers * batch * seq * n_kv_heads * head_dim * dtype_size
        kv_bytes = (
            2 * self.config.n_layers * batch_size * seq_len
            * self.config.kv_heads * self.config.head_dim * 2  # fp16
        )

        # Rough activation estimate: ~2x parameters for a single forward pass
        act_bytes = 2 * param_bytes * batch_size

        mb = 1024 * 1024
        return {
            "parameters_mb": param_bytes / mb,
            "kv_cache_mb": kv_bytes / mb,
            "activations_mb (est)": act_bytes / mb,
            "total_mb (est)": (param_bytes + kv_bytes + act_bytes) / mb,
        }
