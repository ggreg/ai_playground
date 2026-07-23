# From-Scratch Projects

The capstone milestones (M0–M13) have you *adapt* this repo's code to your own model. These
projects are the complement: at the end of each phase of the
[Learning Path](../docs/LEARNING_PATH.md), **rebuild the phase's core idea from a blank
file** — no peeking at the repo's implementation, no modifying something that already works.
Building without a reference is a different (and harsher) test of understanding than editing
working code, which is why these exist.

## The deal

Each project gives you exactly three things:

1. **A contract** — function/class stubs in the starter file. The stubs are the *interface*
   the tests call; everything beneath them is yours to design. They all start as
   `raise NotImplementedError`.
2. **Scaffolding for the boring parts** — dataset generators, a deterministic fake-LLM,
   process-spawn plumbing, a pre-built model to serve. Anything that would burn an hour
   without teaching you the phase's idea is provided. Nothing to install: the repo's
   `uv sync --extra dev` environment covers every project (p6 needs no API key — the LLM
   is scripted).
3. **An acceptance test** — the same policy as the milestone tests: a stub that still
   raises `NotImplementedError` **skips** (you haven't started that part), wrong output
   **fails**. Green means done; nothing else does.

Write your implementation directly in the starter file — `git diff projects/` is then an
exact record of what you built vs. what you were given.

## The projects

| # | After phase | You build from scratch | Done-when |
|---|-------------|------------------------|-----------|
| [p0](p0_grad_engine/README.md) | [DNN Refresher](../docs/LEARNING_PATH.md#phase-0-dnn-refresher-optional) | Scalar autograd engine + MLP + training loop | `uv run pytest projects/p0_grad_engine/` |
| [p1](p1_tiny_gpt/README.md) | [Transformer Internals](../docs/LEARNING_PATH.md#phase-1-transformer-internals) | A tiny GPT (attention included) in one file | `uv run pytest projects/p1_tiny_gpt/` |
| [p2](p2_trainer/README.md) | [Training Optimization](../docs/LEARNING_PATH.md#phase-2-training-optimization) | AdamW, a cosine-warmup schedule, grad accumulation | `uv run pytest projects/p2_trainer/` |
| [p3](p3_kv_serve/README.md) | [Inference Optimization](../docs/LEARNING_PATH.md#phase-3-inference-optimization) | A KV-cached greedy/top-p decoder for a given model | `uv run pytest projects/p3_kv_serve/` |
| [p4](p4_ddp/README.md) | [Distributed Training](../docs/LEARNING_PATH.md#phase-4-distributed-training) | Data-parallel training with raw collectives (CPU, 2 procs) | `uv run pytest projects/p4_ddp/` |
| [p5](p5_roofline/README.md) | [GPU & NVIDIA Tools](../docs/LEARNING_PATH.md#phase-5-gpu--nvidia-tools) | A decode roofline model (FLOPs, bytes, tokens/sec) | `uv run pytest projects/p5_roofline/` |
| [p6](p6_agent/README.md) | [Building AI Agents](../docs/LEARNING_PATH.md#phase-6-building-ai-agents) | A ReAct agent loop with tools and error handling | `uv run pytest projects/p6_agent/` |

## Rules

- **Don't import the repo's implementation of the thing you're building.** Each README
  names exactly what's off-limits (e.g. p0 bans `torch` and `ai_playground.fundamentals`;
  p1 bans `ai_playground.models` and `nn.MultiheadAttention`). Everything else is fair game.
- **Struggle first, compare after.** Once your tests are green, diff your design against the
  repo's version — the divergences are where the interesting lessons are.
- **The agent may not write your project code.** Same ground rules as `/session`: hints via
  the ladder (`/stuck`), review, and test interpretation only.

These are deliberately *not* in the default `uv run pytest` run (`testpaths = ["tests"]`),
so a fresh clone stays green; run each project's own command above.
