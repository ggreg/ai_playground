"""Full transformer model — LLaMA-style decoder-only architecture.

The transformer is built from a small set of first-principles concepts:
1. Embeddings — tokens (integers) → continuous vectors
2. Positional encoding — inject sequence order (RoPE)
3. Linear projections — learned matrix multiplies (Q, K, V, FFN)
4. Dot-product similarity — Q · K measures relevance between tokens
5. Scaling — divide by sqrt(head_dim) to stabilize softmax
6. Softmax — normalize scores into attention probabilities
7. Weighted sum — blend Values by attention weights
8. Causal mask — prevent attending to future tokens
9. Multi-head attention — parallel independent attention patterns
10. Residual connections — x + f(x) for gradient flow
11. Layer normalization — stabilize activations (RMSNorm variant)
12. Feed-forward network — per-token transformation (SwiGLU variant)
13. Transformer block — compose: norm → attention → residual → norm → FFN → residual
14. Output head — project to vocabulary size → logits
15. Sampling — logits → probabilities → next token (see inference/generate.py)

Reading list:
- "Attention Is All You Need" (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762
- The Annotated Transformer (Harvard NLP): https://nlp.seas.harvard.edu/annotated-transformer/
- LLaMA (Touvron et al., 2023): https://arxiv.org/abs/2302.13971
- See docs/PAPERS.md for the full reference list
"""

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention
from .config import TransformerConfig
from .layers import RMSNorm, SwiGLU, precompute_rope_frequencies


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture (concept 13).

    Composes four concepts into a repeatable unit:
    - Layer norm before each sub-layer (concept 11, "pre-norm")
    - Multi-head attention (concept 9)
    - Feed-forward network (concept 12)
    - Residual connections around both (concept 10)

    Pre-norm (norm before attention/FFN) is more stable for training than
    post-norm (original paper) and is the standard in all modern LLMs.

    References:
    - Original post-norm: https://arxiv.org/abs/1706.03762 (Section 3.1)
    - Pre-norm analysis: https://arxiv.org/abs/2002.04745
    - RMSNorm: https://arxiv.org/abs/1910.07467
    - SwiGLU: https://arxiv.org/abs/2002.05202
    See also: docs/PAPERS.md § Normalization & Activations
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)  # Concept 9: multi-head attention
        self.feed_forward = SwiGLU(config.dim, config.ffn_hidden_dim, config.dropout)  # Concept 12: FFN
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)  # Concept 11: layer norm
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)  # Concept 11: layer norm

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Concept 10 (residual) + 11 (norm) + 9 (attention):
        # x + attention(norm(x)) — the residual lets gradients flow directly
        # through the skip connection, making deep networks trainable.
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask, use_cache)
        # Concept 10 (residual) + 11 (norm) + 12 (FFN):
        # x + ffn(norm(x)) — FFN processes each token independently,
        # storing "knowledge" in its weight matrices.
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

    Papers:
    - LLaMA: https://arxiv.org/abs/2302.13971
    - LLaMA 2 (GQA): https://arxiv.org/abs/2307.09288
    See also: docs/PAPERS.md § Transformer Architecture
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Concept 1: Embedding — lookup table mapping token IDs to dense vectors.
        # Each of the vocab_size tokens gets its own dim-dimensional vector.
        # "Attention Is All You Need" §3.4: https://arxiv.org/abs/1706.03762
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # Concept 13: Stack N transformer blocks — depth is where the model
        # builds increasingly abstract representations of the input.
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Concept 11: Final layer norm before the output projection.
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # Concept 14: Output head — linear projection from dim → vocab_size.
        # Produces one logit (score) per vocabulary token. The highest logit
        # is the model's best guess for the next token.
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: reuse embedding weights for the output projection.
        # Reduces parameters and often improves quality — the model uses
        # the same representation for "understanding" and "producing" tokens.
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017):
        # https://arxiv.org/abs/1608.05859
        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Concept 2: Positional encoding — precompute RoPE rotation frequencies.
        # Without this, the model has no sense of token order ("cat sat" = "sat cat").
        # RoPE encodes position by rotating Q and K vectors in 2D subspaces.
        # RoPE (Su et al., 2021): https://arxiv.org/abs/2104.09864
        # See also: docs/PAPERS.md § Positional Encodings
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
        """Forward pass: tokens in → logits out.

        The data flow maps directly to the concept stack:
        tokens → embedding (1) → + position (2) → N × block (13) → norm (11) → output (14)

        Args:
            tokens: (batch, seq_len) integer token IDs
            mask: optional attention mask
            use_cache: enable KV cache for autoregressive generation

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = tokens.shape

        # Concept 1: Embedding lookup — integers → vectors
        h = self.tok_embeddings(tokens)

        # Concept 2: RoPE frequencies for this sequence length
        # (actual rotation happens inside attention when applied to Q and K)
        freqs_cis = self.freqs_cis[:seq_len]

        # Concept 8: Causal mask — upper triangle of -inf so that after softmax,
        # future positions get zero attention weight. This ensures token t can
        # only "see" tokens 0..t, which is required for autoregressive generation.
        # "Attention Is All You Need" §3.1 (Masked Multi-Head Attention)
        if mask is None and use_cache and seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Concept 13: Pass through N transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, mask, use_cache)

        # Concept 11: Final normalization
        h = self.norm(h)

        # Concept 14: Project to vocabulary → logits
        # After this, use softmax + sampling (concept 15) to pick the next token.
        # See inference/generate.py for top-p/nucleus sampling.
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
