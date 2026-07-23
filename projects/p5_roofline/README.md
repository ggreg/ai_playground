# p5 — A decode roofline model from a blank file

**After Phase 5 (GPU & NVIDIA Tools).** No GPU required — this project is the *mental
model* on which the whole phase runs: given a transformer config and a GPU's peak specs,
predict decode FLOPs, bytes moved, arithmetic intensity, tokens/sec, and the batch size
where decoding flips from bandwidth-bound to compute-bound. M9/M11 had you do this once
for your model; here you build the general calculator, from scratch. ~2–3 hours.

## You build (the contract in `roofline.py`)

- `decode_flops_per_token(cfg)` — the ≈ `2 × params` rule, derived from the actual GEMMs.
- `decode_bytes_per_step(cfg, context_len, batch_size)` — weights once + K/V per request.
- `arithmetic_intensity(cfg, context_len, batch_size)` — FLOPs/byte at that batch.
- `predict_decode_tokens_per_sec(cfg, gpu, context_len, batch_size)` — the roofline
  `min(compute ceiling, bandwidth ceiling)` ([Williams et al., 2009](https://doi.org/10.1145/1498765.1498785)).
- `crossover_batch(cfg, gpu, context_len)` — smallest batch that is compute-bound, or
  `None` when no batch is. (At real context lengths the KV asymptote sits below the
  ridge and the answer *is* None — discovering that is the project's punchline.)

Exact conventions (dtype bytes, what counts) are pinned in the stubs' docstrings so the
tests are unambiguous.

## Scaffolding provided

- `GPU_SPECS` — peak TFLOPS and memory bandwidth for T4 / A100 / H100, matching the
  numbers used elsewhere in the repo.
- Configs come from `ai_playground.models.TransformerConfig` (allowed — you're analyzing
  models, not rebuilding them).

## Rules

- No imports from `ai_playground.profiling` — that module is the *after* comparison. One
  test cross-checks your FLOP count against `cfg.num_params()` (the Chinchilla 2N rule);
  the rest are internal-consistency checks your formulas must satisfy.

## Done-when

```bash
uv run pytest projects/p5_roofline/ -v
```

Afterwards: diff against `src/ai_playground/profiling/flops.py`, and sanity-check your
T4 predictions against your own M9/M11 numbers in `checkpoints/myllm/metrics.json`.
