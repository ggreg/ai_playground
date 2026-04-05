"""Model configurations from tiny (~10M) to medium (~350M) scale."""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None  # None = MHA, < n_heads = GQA, 1 = MQA
    max_seq_len: int = 1024
    ffn_dim_multiplier: float = 2.667  # SwiGLU: hidden = dim * multiplier * 2/3 rounded
    norm_eps: float = 1e-5
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        hidden = int(self.dim * self.ffn_dim_multiplier * 2 / 3)
        # Round to nearest multiple of 64 for GPU efficiency
        return ((hidden + 63) // 64) * 64

    def num_params(self, include_embeddings: bool = True) -> int:
        """Estimate total parameter count."""
        # Embeddings
        emb = self.vocab_size * self.dim if include_embeddings else 0
        # Attention: Q, K, V projections + output
        attn_q = self.dim * self.dim
        attn_kv = 2 * self.dim * (self.kv_heads * self.head_dim)
        attn_o = self.dim * self.dim
        attn = attn_q + attn_kv + attn_o
        # FFN: gate, up, down (SwiGLU)
        ffn = 3 * self.dim * self.ffn_hidden_dim
        # Norms (2 per layer)
        norms = 2 * self.dim
        # Per layer total
        per_layer = attn + ffn + norms
        # Final norm + output head
        final = self.dim + (0 if self.tie_word_embeddings else self.vocab_size * self.dim)
        return emb + self.n_layers * per_layer + final


# Predefined configurations
TINY = TransformerConfig(
    dim=256, n_layers=6, n_heads=8, n_kv_heads=4,
    max_seq_len=512, vocab_size=32000,
)  # ~10M params

SMALL = TransformerConfig(
    dim=768, n_layers=12, n_heads=12, n_kv_heads=4,
    max_seq_len=2048, vocab_size=32000,
)  # ~125M params

MEDIUM = TransformerConfig(
    dim=1024, n_layers=24, n_heads=16, n_kv_heads=4,
    max_seq_len=2048, vocab_size=32000,
)  # ~350M params
