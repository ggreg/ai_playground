"""Custom optimizer implementations for experimentation.

Includes standard AdamW reference and notes on modern alternatives.
"""

import math

import torch
from torch.optim import Optimizer


class AdamWFromScratch(Optimizer):
    """AdamW implemented from scratch for learning purposes.

    This is functionally identical to torch.optim.AdamW but written
    explicitly so you can see every step of the algorithm:

    1. Compute gradient g_t
    2. Update biased first moment:  m_t = β1 * m_{t-1} + (1 - β1) * g_t
    3. Update biased second moment: v_t = β2 * v_{t-1} + (1 - β2) * g_t²
    4. Bias correction: m̂_t = m_t / (1 - β1^t), v̂_t = v_t / (1 - β2^t)
    5. Parameter update: θ_t = θ_{t-1} - lr * (m̂_t / (√v̂_t + ε) + λ * θ_{t-1})

    Note: AdamW applies weight decay DIRECTLY to parameters (decoupled),
    not through the gradient like L2 regularization. This is the key
    difference from the original Adam + L2 regularization.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)  # First moment (mean)
                    state["v"] = torch.zeros_like(p)  # Second moment (variance)

                state["step"] += 1
                m, v = state["m"], state["v"]
                t = state["step"]

                # Update biased moments
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # Decoupled weight decay (the "W" in AdamW)
                p.mul_(1 - lr * weight_decay)

                # Parameter update
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss
