# p4 — Data parallelism from a blank file (CPU, 2 processes)

**After Phase 4 (Distributed Training).** No GPUs needed: `torch.distributed`'s gloo
backend runs on CPU processes, and the *idea* of DDP — replicate the model, shard the
data, average the gradients, take identical steps — is hardware-independent. You build
the synchronization yourself; `DistributedDataParallel` is banned. ~2–3 hours.

## You build (the contract in `ddp_scratch.py`)

- `train_worker(rank, world_size, n_steps, lr) -> state_dict` — runs inside each spawned
  process (the process/rendezvous plumbing is scaffolding). Per step, on every rank:
  compute the mean loss on **your shard** (`x[rank::world_size]`), backward, **average
  gradients across ranks yourself** (`dist.all_reduce` + divide — or send/recv if you
  want the stretch goal), then the exact SGD step `p -= lr * p.grad`.

The acceptance bar: after `n_steps`, every rank's parameters must match a *single*-process
full-batch run to 1e-5 — data parallelism is only correct when it's invisible.

## Scaffolding provided

- `run_workers(...)` — all the painful parts: `mp.spawn`, FileStore rendezvous (no port
  fights), process-group init/teardown, shipping rank 0's result back. Also
  `collective_smoke_test()` plus a test that runs it, so you know the harness works on
  your machine before a line of your code exists.
- `make_model` / `make_data` / `loss_fn` — deterministic and shared with the reference.

## Rules

- Banned: `DistributedDataParallel`, FSDP, and `ai_playground.training.distributed`.
  The collectives themselves (`dist.all_reduce`, `dist.send`/`recv`, `dist.barrier`)
  are exactly what you're supposed to use.

## Done-when

```bash
uv run pytest projects/p4_ddp/ -v
```

Afterwards: read `src/ai_playground/training/distributed.py`, then
[ZeRO (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) (docs/PAPERS.md)
for what sharding adds when replicas stop fitting.
