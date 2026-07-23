"""p3 scaffolding — the GIVEN model. Read it, use its pieces, don't edit it.

The model is deliberately plain (learned positions, LayerNorm, GELU — no RoPE) so that
re-implementing its forward pass with a KV cache is about caching, not about rotary
subtleties. Every submodule you need is exposed by name.

Exact computation of `forward(ids)`, which your cached decoder must reproduce:

    x = tok_emb(ids) + pos_emb(positions)          # positions = 0..T-1
    for block in blocks:
        h = block.ln1(x)
        q, k, v = block.wq(h), block.wk(h), block.wv(h)   # each (B, T, dim)
        split heads: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        att = softmax(q @ k^T / sqrt(head_dim) + causal_mask) @ v
        x = x + block.wo(att merged back to (B, T, dim))
        h = block.ln2(x)
        x = x + block.w_down(gelu(block.w_up(h)))
    logits = head(ln_f(x))                          # (B, T, vocab)

For a cached decode step, position matters: token t must be embedded with pos_emb(t),
and its q attends over the cached k/v of positions 0..t.
"""

import math

import torch
from torch import nn

VOCAB = 61
DIM = 32
N_LAYERS = 2
N_HEADS = 4
HEAD_DIM = DIM // N_HEADS
MAX_SEQ = 64


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DIM)
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.ln2 = nn.LayerNorm(DIM)
        self.w_up = nn.Linear(DIM, 4 * DIM, bias=False)
        self.w_down = nn.Linear(4 * DIM, DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.ln1(x)
        q, k, v = self.wq(h), self.wk(h), self.wv(h)
        q = q.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        att = torch.softmax(scores + mask, dim=-1) @ v
        x = x + self.wo(att.transpose(1, 2).reshape(B, T, DIM))
        h = self.ln2(x)
        return x + self.w_down(torch.nn.functional.gelu(self.w_up(h)))


class GivenGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.ln_f = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) int64 -> (B, T, VOCAB) logits. Full sequence only — no cache hooks."""
        B, T = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(T))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def tiny_model(seed: int = 0) -> GivenGPT:
    """The given model: deterministic random weights, eval mode, no grad needed."""
    torch.manual_seed(seed)
    model = GivenGPT()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
