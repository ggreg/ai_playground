"""p2 — your optimizer, schedule, and gradient accumulation. See README.md for the
brief and the import rules (short version: the optimizer package is the referee, not a
dependency). Everything below the SCAFFOLDING line is yours.
"""

import math  # noqa: F401  (you'll want it for the cosine)

import torch

# ---------------------------------------------------------------- SCAFFOLDING (given) --


def make_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
    )


def make_batch(seed: int = 0, n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    y = (x[:, :1] * 0.5 + torch.sin(x[:, 1:2]) + 0.1 * torch.randn(n, 1, generator=g))
    return x, y


def loss_fn(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean-squared error over the given (micro-)batch."""
    return torch.nn.functional.mse_loss(model(x), y)


# ------------------------------------------------------------------- YOUR CODE (build) --


class AdamWScratch:
    """AdamW, reimplemented. Must match torch's to < 1e-5 over 20 identical steps."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def step(self) -> None:
        raise NotImplementedError


def warmup_cosine_lr(
    step: int, max_steps: int, max_lr: float, warmup_steps: int, min_lr: float = 0.0
) -> float:
    """LR at `step` (0-indexed).

    Convention (the tests hold you to exactly this):
    - steps 0..warmup_steps: linear from 0 at step 0 to max_lr at step == warmup_steps
    - warmup_steps..max_steps: cosine from max_lr down to min_lr at step == max_steps
    """
    raise NotImplementedError


def accumulate_gradients(
    model: torch.nn.Module, xb: torch.Tensor, yb: torch.Tensor, micro_batch_size: int
) -> None:
    """Leave grads on model.parameters() equal to the full-batch mean-loss gradients,
    while never doing a forward pass larger than micro_batch_size rows."""
    raise NotImplementedError
