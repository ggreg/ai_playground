"""p1 — your decoder-only transformer, in one file. See README.md for the brief and the
import rules (short version: torch primitives yes, torch's ready-made attention no).
"""

import torch

# ---------------------------------------------------------------- SCAFFOLDING (given) --
VOCAB = 17
SEQ_LEN = 32


def copy_task_batch(batch_size: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token prediction on periodic sequences — learnable to ~zero loss.

    Each row repeats a random period-4 motif, so token t is fully determined by tokens
    t-4..t-1. Returns (inputs, targets), both (batch_size, SEQ_LEN), targets shifted by 1.
    """
    g = torch.Generator().manual_seed(seed)
    motifs = torch.randint(0, VOCAB, (batch_size, 4), generator=g)
    rows = motifs.repeat(1, (SEQ_LEN + 4) // 4 + 1)[:, : SEQ_LEN + 1]
    return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


# ------------------------------------------------------------------- YOUR CODE (build) --


def build_model(
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
) -> torch.nn.Module:
    """Return your transformer: forward(ids: (B, T) int64) -> logits (B, T, vocab_size)."""
    raise NotImplementedError


if __name__ == "__main__":
    model = build_model(VOCAB, dim=64, n_layers=2, n_heads=4, max_seq_len=SEQ_LEN)
    x, y = copy_task_batch()
    print("logits:", tuple(model(x).shape), "— expected", (16, SEQ_LEN, VOCAB))
