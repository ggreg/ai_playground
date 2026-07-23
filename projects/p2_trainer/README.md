# p2 — The training stack from a blank file

**After Phase 2 (Training Optimization).** You've watched the repo's `Trainer` do it.
Now write the three load-bearing pieces yourself and prove each one numerically:
your own AdamW, your own warmup+cosine schedule, and gradient accumulation that is
*exactly* equivalent to the big batch. ~2–4 hours.

## You build (the contract in `trainer_scratch.py`)

- `AdamWScratch(params, lr, betas, eps, weight_decay)` with `.step()` and `.zero_grad()`
  — must match `torch.optim.AdamW` to < 1e-5 over 20 steps. Decoupled weight decay
  ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)) is the part people
  get wrong: decay is applied to the *weights*, not mixed into the gradient.
- `warmup_cosine_lr(step, max_steps, max_lr, warmup_steps, min_lr)` — the exact
  convention is in the stub's docstring, so the tests are unambiguous.
- `accumulate_gradients(model, xb, yb, micro_batch_size)` — populate `param.grad` with
  gradients of the **mean loss over the full batch** using micro-batches. The classic
  trap (each micro-batch's mean loss summing to `n_micro ×` the right gradient) is
  exactly what the test measures.

## Scaffolding provided

- `make_model(seed)` / `make_batch(seed)` — a small deterministic MLP + regression batch,
  and `loss_fn` (MSE), so "what model, what loss?" costs you zero minutes.

## Rules

- No `torch.optim` in your code (the *test* uses it as the referee) and no imports from
  `ai_playground.training`. Autograd (`loss.backward()`) is allowed and expected — you
  built backprop yourself in p0; here the lesson is what happens *after* `.backward()`.

## Done-when

```bash
uv run pytest projects/p2_trainer/ -v
```
