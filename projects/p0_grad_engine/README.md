# p0 — Autograd from a blank file

**After Phase 0 (DNN Refresher).** In the chapters you read and ran
`ai_playground.fundamentals.autograd`. Now close it and rebuild the whole stack yourself:
a scalar reverse-mode autograd engine, an MLP on top of it, and a training loop that
separates the two moons. ~2–4 hours.

## You build (the contract in `grad_engine.py`)

- `Value` — a scalar with `.data`, `.grad`, supporting at least `+`, `*`, `-`,
  `**` (int powers), `tanh()`, and `backward()`. Gradients **accumulate** (`+=`) so shared
  subexpressions are correct — that's what the first test checks.
- `train_moons() -> float` — build an MLP out of your `Value`s (architecture, loss, LR,
  step count: all yours) and train it on the provided moons data until it separates them.
  Returns final train accuracy.

## Scaffolding provided

- Moons data via `ai_playground.fundamentals.datasets.make_moons` (numpy-only data
  generation is boilerplate, not the lesson) — already imported and seeded in the starter.
- Nothing to install; the repo env has everything.

## Rules

- No `torch`, no `numpy` in the engine itself (numpy only to read the dataset arrays),
  and no imports from `ai_playground.fundamentals.autograd` / `.nn`. The tests check the
  gradients against central finite differences, not against any library.

## Done-when

```bash
uv run pytest projects/p0_grad_engine/ -v
```

Stuck? The hint ladder applies (`/stuck`). After it's green, diff your engine against
`src/ai_playground/fundamentals/autograd.py` — did you both need topological sort? —
and reread [Rumelhart, Hinton & Williams (1986)](https://doi.org/10.1038/323533a0)
(see also docs/PAPERS.md).
