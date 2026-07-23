"""Acceptance tests for p2 — optimizer / schedule / accumulation, each proved
numerically against an independent reference."""

from pathlib import Path

import pytest
import torch

from trainer_scratch import (
    AdamWScratch,
    accumulate_gradients,
    loss_fn,
    make_batch,
    make_model,
    warmup_cosine_lr,
)

HP = dict(lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)


def test_no_forbidden_imports():
    lines = (Path(__file__).parent / "trainer_scratch.py").read_text().splitlines()
    code = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    for banned in ("torch.optim", "ai_playground.training"):
        assert banned not in code, f"trainer_scratch.py must not use '{banned}'"


def test_adamw_matches_torch(attempt):
    ours_model, ref_model = make_model(seed=1), make_model(seed=1)
    x, y = make_batch(seed=2)
    ours = attempt(AdamWScratch, ours_model.parameters(), **HP)
    ref = torch.optim.AdamW(ref_model.parameters(), **HP)

    for _ in range(20):
        for model, opt in ((ours_model, ours), (ref_model, ref)):
            attempt(opt.zero_grad)
            loss_fn(model, x, y).backward()
            attempt(opt.step)

    worst = max(
        (p1 - p2).abs().max().item()
        for p1, p2 in zip(ours_model.parameters(), ref_model.parameters())
    )
    assert worst < 1e-5, (
        f"max param deviation vs torch.optim.AdamW after 20 steps: {worst:.2e} (need < 1e-5). "
        "Usual suspects: bias correction, or weight decay coupled into the gradient."
    )


def test_schedule_endpoints_and_shape(attempt):
    max_steps, warmup, max_lr, min_lr = 1000, 100, 3e-4, 3e-5
    lr = lambda s: attempt(warmup_cosine_lr, s, max_steps, max_lr, warmup, min_lr)  # noqa: E731

    assert lr(0) == pytest.approx(0.0, abs=1e-12)
    assert lr(warmup // 2) == pytest.approx(max_lr / 2, rel=0.02), "warmup should be linear"
    assert lr(warmup) == pytest.approx(max_lr)
    assert lr(max_steps) == pytest.approx(min_lr)
    mid = warmup + (max_steps - warmup) // 2
    assert lr(mid) == pytest.approx((max_lr + min_lr) / 2, rel=0.02), "cosine midpoint"
    decayed = [lr(s) for s in range(warmup, max_steps + 1, 50)]
    assert all(a >= b for a, b in zip(decayed, decayed[1:])), "must decay monotonically"


def test_accumulation_equals_full_batch(attempt):
    x, y = make_batch(seed=3)

    full = make_model(seed=4)
    loss_fn(full, x, y).backward()

    accum = make_model(seed=4)
    attempt(accumulate_gradients, accum, x, y, 4)  # 64 rows -> 16 micro-batches

    worst = max(
        (p1.grad - p2.grad).abs().max().item()
        for p1, p2 in zip(full.parameters(), accum.parameters())
    )
    assert worst < 1e-6, (
        f"accumulated grads differ from full-batch grads by {worst:.2e} (need < 1e-6). "
        "If it's off by ~16x, you summed micro-batch mean losses without rescaling."
    )
