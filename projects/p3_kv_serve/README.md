# p3 — A KV-cached decoder from a blank file

**After Phase 3 (Inference Optimization).** The model is *given* — a small GPT whose
internals are all exposed. What you build is the inference engine on top of it: greedy
decoding by full recompute, then greedy decoding through **your own KV cache** (token-
for-token identical — that's the acceptance bar, same as vLLM's correctness tests), and
top-p sampling. ~3–5 hours.

## You build (the contract in `kv_generate.py`)

- `generate_full(model, prompt_ids, n_new)` — greedy, re-running the whole sequence
  every step. The O(n²) baseline; ~10 lines.
- `generate_cached(model, prompt_ids, n_new)` — greedy again, but each decode step feeds
  **one token**, reusing cached K/V. You'll re-implement the forward pass out of the
  model's exposed pieces (`blocks[i].wq` etc. — the exact math is in `given_model.py`'s
  docstrings), because the given `forward()` has no cache hooks. That's the point.
- `top_p_sample(logits, p, generator)` — nucleus sampling
  ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)): sample from the smallest
  set of tokens whose probability mass reaches `p`, renormalized.

## Scaffolding provided

- `given_model.py` — `tiny_model()`: 2 layers, 4 heads, dim 32, deterministic weights,
  fully documented math. Building a model is p1; here it's boilerplate.

## Rules

- No imports from `ai_playground.inference`. The test also counts calls to the model's
  full `forward()` inside `generate_cached` — at most one (the prefill); if every decode
  step calls it, it isn't a cache.

## Done-when

```bash
uv run pytest projects/p3_kv_serve/ -v
```

Afterwards: compare with `src/ai_playground/inference/generate.py`, then read what paging
adds on top — [vLLM / PagedAttention (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
(docs/PAPERS.md) — which is exactly the finale's M13.
