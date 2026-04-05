"""Core building blocks: RMSNorm, RoPE, SwiGLU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Simpler and faster than LayerNorm — no mean subtraction, no bias.
    Used in LLaMA, Mistral, and most modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the complex exponentials for RoPE.

    Returns: (max_seq_len, head_dim // 2) complex tensor
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors.

    Args:
        xq: (batch, seq_len, n_heads, head_dim)
        xk: (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: (seq_len, head_dim // 2) complex tensor
    """
    # Reshape to complex: (..., head_dim) -> (..., head_dim // 2, 2) -> complex
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis to match (1, seq_len, 1, head_dim // 2)
    freqs = freqs_cis[: xq.shape[1]].unsqueeze(0).unsqueeze(2)

    # Apply rotation
    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer, 2020). Used in LLaMA, PaLM, etc.

    gate = swish(x @ W_gate)
    out  = gate * (x @ W_up) @ W_down

    Compared to standard FFN, SwiGLU uses 3 weight matrices instead of 2,
    but the hidden dimension is 2/3 of what a standard FFN would use,
    so total params are similar while performance is better.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
