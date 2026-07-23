"""Acceptance tests for p4 — hand-rolled DDP.

The referee is a single-process full-batch run: with equal shards and averaged
gradients, data parallelism must be numerically invisible.
"""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import ddp_scratch
from ddp_scratch import loss_fn, make_data, make_model

N_STEPS, LR, WORLD = 10, 0.05, 2

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with gloo is unavailable on this build",
)


def test_no_forbidden_imports():
    lines = (Path(__file__).parent / "ddp_scratch.py").read_text().splitlines()
    code = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    for banned in ("DistributedDataParallel", "FullyShardedDataParallel", "fsdp",
                   "ai_playground.training"):
        assert banned not in code, f"ddp_scratch.py must not use '{banned}' — see README"


def test_scaffold_collectives_work():
    assert ddp_scratch.collective_smoke_test(WORLD) == pytest.approx(3.0)


def _single_process_reference() -> dict:
    model = make_model(seed=0)
    x, y = make_data(seed=0)
    for _ in range(N_STEPS):
        model.zero_grad()
        loss_fn(model, x, y).backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= LR * p.grad
    return model.state_dict()


def test_ddp_matches_single_process(attempt):
    state = attempt(ddp_scratch.run_workers, ddp_scratch.train_worker, WORLD, N_STEPS, LR)
    ref = _single_process_reference()
    assert set(state.keys()) == set(ref.keys())
    worst = max((state[k] - ref[k]).abs().max().item() for k in ref)
    assert worst < 1e-5, (
        f"max param deviation from the single-process run: {worst:.2e} (need < 1e-5). "
        "If it's ~2x off in gradient terms, you summed instead of averaged; if it "
        "diverges over steps, the replicas didn't start identical."
    )
