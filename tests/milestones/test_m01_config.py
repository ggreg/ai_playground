"""M1 (session S1.2) — design your model under a 2M-parameter budget.

Certifies: the reader made every architecture choice (dim, n_layers, n_heads,
vocab_size) themselves, the resulting parameter count fits the budget, and the shapes
actually compose — a (4, 32) token batch flows through to (4, 32, vocab_size) logits.
"""

import torch


def test_parameter_budget(reader_config):
    n = reader_config.num_params()
    assert n <= 2_000_000, (
        f"Your config is {n:,} parameters — over the 2M budget. The embedding table "
        f"({reader_config.vocab_size} x {reader_config.dim}) is usually the culprit."
    )


def test_forward_pass_shapes(reader_config):
    from ai_playground.models import Transformer

    model = Transformer(reader_config)
    tokens = torch.randint(0, reader_config.vocab_size, (4, 32))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (4, 32, reader_config.vocab_size), (
        f"Expected (4, 32, {reader_config.vocab_size}) logits, got {tuple(logits.shape)}."
    )
