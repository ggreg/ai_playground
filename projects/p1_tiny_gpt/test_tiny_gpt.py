"""Acceptance tests for p1 — a tiny GPT from a blank file.

Behavioral checks only: output shape, strict causality, and the ability to learn a
deterministic next-token task. Architecture choices inside are free.
"""

from pathlib import Path

import torch

from tiny_gpt import SEQ_LEN, VOCAB, build_model, copy_task_batch


def _model(attempt):
    torch.manual_seed(0)
    m = attempt(build_model, VOCAB, 64, 2, 4, SEQ_LEN)
    m.eval()
    return m


def test_no_forbidden_imports():
    lines = (Path(__file__).parent / "tiny_gpt.py").read_text().splitlines()
    src = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    for banned in (
        "MultiheadAttention",
        "nn.Transformer",
        "scaled_dot_product_attention",
        "ai_playground.models",
    ):
        assert banned not in src, f"tiny_gpt.py must not use '{banned}' — see README rules"


def test_scaffold_copy_task_is_deterministic():
    x1, y1 = copy_task_batch()
    x2, y2 = copy_task_batch()
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(x1[:, 1:], y1[:, :-1])  # targets are inputs shifted by one


def test_output_shape(attempt):
    model = _model(attempt)
    x, _ = copy_task_batch(batch_size=3)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (3, SEQ_LEN, VOCAB), f"got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all()


def test_causality(attempt):
    model = _model(attempt)
    x, _ = copy_task_batch(batch_size=1)
    t = SEQ_LEN // 2
    x_perturbed = x.clone()
    x_perturbed[0, t + 1 :] = (x_perturbed[0, t + 1 :] + 1) % VOCAB
    with torch.no_grad():
        a, b = model(x), model(x_perturbed)
    diff = (a[0, : t + 1] - b[0, : t + 1]).abs().max().item()
    assert diff < 1e-5, (
        f"changing tokens after position {t} changed logits at <= {t} (max diff {diff:.2e}) "
        "— the causal mask is leaking future information"
    )


def test_learns_the_copy_task(attempt):
    torch.manual_seed(0)
    model = attempt(build_model, VOCAB, 64, 2, 4, SEQ_LEN)
    x, y = copy_task_batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    def step() -> float:
        opt.zero_grad()
        loss = loss_fn(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step()
        return loss.item()

    initial = step()
    final = initial
    for _ in range(299):
        final = step()
    assert final < 0.5 and final < 0.5 * initial, (
        f"loss went {initial:.3f} -> {final:.3f} on a deterministic task; a working "
        "transformer reaches < 0.5. Suspects: mask applied after softmax, missing "
        "residuals, or positions never added."
    )


def test_respects_vocab_and_length_args(attempt):
    torch.manual_seed(0)
    m = attempt(build_model, 7, 32, 1, 2, 16)
    m.eval()
    ids = torch.randint(0, 7, (2, 16))
    with torch.no_grad():
        out = m(ids)
    assert out.shape == (2, 16, 7)
