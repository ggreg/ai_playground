"""p4 — hand-rolled data parallelism. See README.md for the brief and rules (short
version: raw collectives yes, the ready-made DDP/FSDP wrappers no).

Everything above the YOUR CODE line is process plumbing — read it once, then forget it.
"""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------- SCAFFOLDING (given) --


def make_model(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
    )


def make_data(seed: int = 0, n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    y = (x[:, :1] * 0.5 + torch.sin(x[:, 1:2]) + 0.1 * torch.randn(n, 1, generator=g))
    return x, y


def loss_fn(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(x), y)


def _worker(rank: int, world_size: int, fn_name: str, args: tuple, workdir: str):
    """Spawned in each process: rendezvous via FileStore, run the named module-level
    function, save rank 0's return value for the parent, tear down."""
    store = dist.FileStore(os.path.join(workdir, "rendezvous"), world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    try:
        result = globals()[fn_name](rank, world_size, *args)
        if rank == 0:
            torch.save(result, os.path.join(workdir, "result.pt"))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_workers(fn, world_size: int, *args):
    """Run `fn(rank, world_size, *args)` in `world_size` CPU processes; return rank 0's
    result. `fn` must be a module-level function in this file."""
    with tempfile.TemporaryDirectory() as workdir:
        try:
            mp.spawn(_worker, args=(world_size, fn.__name__, args, workdir), nprocs=world_size)
        except Exception as e:  # child exceptions arrive wrapped; keep skip-on-stub working
            if "NotImplementedError" in str(e):
                raise NotImplementedError from e
            raise
        return torch.load(os.path.join(workdir, "result.pt"), weights_only=True)


def _smoke(rank: int, world_size: int) -> float:
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def collective_smoke_test(world_size: int = 2) -> float:
    """Proves spawn + gloo + all_reduce work here: returns sum of (rank+1) = 3.0 for 2."""
    return run_workers(_smoke, world_size)


# ------------------------------------------------------------------- YOUR CODE (build) --


def train_worker(rank: int, world_size: int, n_steps: int, lr: float):
    """Data-parallel training, synchronized by you. Runs inside every spawned process.

    Recipe to implement (each deviation shows up as a parity failure):
    - model = make_model(seed=0) on every rank (identical replicas)
    - your shard of make_data(seed=0) is x[rank::world_size], y[rank::world_size]
    - per step: zero grads -> mean loss on the shard -> backward -> average grads
      across ranks (all_reduce, then divide) -> p -= lr * p.grad (plain SGD, no momentum)
    - return the model's state_dict
    """
    raise NotImplementedError


if __name__ == "__main__":
    print("collective smoke test (expect 3.0):", collective_smoke_test())
