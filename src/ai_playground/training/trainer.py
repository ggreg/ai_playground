"""Training loop with mixed precision, gradient accumulation, and LR scheduling."""

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    max_steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    dtype: str = "float32"  # "float32", "float16", "bfloat16"
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    save_dir: str = "checkpoints"
    wandb_project: str | None = None
    seed: int = 42


class Trainer:
    """Training loop with mixed precision and gradient accumulation.

    Key features for learning:
    - Mixed precision training with automatic loss scaling
    - Gradient accumulation to simulate large batches
    - Cosine LR schedule with warmup
    - Gradient clipping
    - Tokens/sec and MFU tracking

    Papers:
    - Mixed Precision Training: https://arxiv.org/abs/1710.03740
    - AdamW (Decoupled Weight Decay): https://arxiv.org/abs/1711.05101
    See also: docs/PAPERS.md § Training Optimization
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Resolve device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # Mixed precision setup
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.amp_dtype = self.dtype_map[config.dtype]
        self.use_amp = config.dtype != "float32" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and config.dtype == "float16")

        # Optimizer: AdamW with weight decay only on 2D parameters (not biases/norms)
        decay_params = [p for p in model.parameters() if p.dim() >= 2]
        no_decay_params = [p for p in model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            fused=self.device.type == "cuda",
        )

        self.step = 0

    def get_lr(self) -> float:
        """Cosine learning rate schedule with linear warmup."""
        cfg = self.config
        if self.step < cfg.warmup_steps:
            return cfg.learning_rate * (self.step + 1) / cfg.warmup_steps

        progress = (self.step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        import math
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine

    def train(self) -> list[dict]:
        """Run training loop. Returns list of logged metrics."""
        cfg = self.config
        self.model.train()
        logs = []
        data_iter = iter(self.train_loader)

        while self.step < cfg.max_steps:
            t0 = time.perf_counter()
            total_loss = 0.0
            total_tokens = 0

            # Update learning rate
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation loop
            self.optimizer.zero_grad(set_to_none=True)
            for micro_step in range(cfg.grad_accum_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    x, y = next(data_iter)

                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(x)
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                    loss = loss / cfg.grad_accum_steps

                self.scaler.scale(loss).backward()
                total_loss += loss.item()
                total_tokens += y.numel()

            # Gradient clipping
            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            dt = time.perf_counter() - t0
            tokens_per_sec = total_tokens / dt

            if self.step % cfg.log_interval == 0:
                log_entry = {
                    "step": self.step,
                    "loss": total_loss,
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                    "dt_ms": dt * 1000,
                }
                logs.append(log_entry)
                print(
                    f"step {self.step:5d} | loss {total_loss:.4f} | "
                    f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s | {dt*1000:.1f}ms"
                )

            self.step += 1

        return logs

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation and return average loss."""
        if self.eval_loader is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for x, y in self.eval_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(1, n_batches)
