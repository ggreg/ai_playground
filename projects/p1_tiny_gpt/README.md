# p1 — A tiny GPT from a blank file

**After Phase 1 (Transformer Internals).** You've read every component of the repo's
LLaMA-style model. Now build a working decoder-only transformer in one file without
looking at it: embeddings, multi-head **causal** attention, MLP blocks, a head — wired
well enough to actually learn. ~3–5 hours.

## You build (the contract in `tiny_gpt.py`)

- `build_model(vocab_size, dim, n_layers, n_heads, max_seq_len) -> torch.nn.Module`
  whose `forward(ids)` maps `(B, T)` int64 token ids to `(B, T, vocab_size)` logits.

Everything inside is your design: pre/post-norm, LayerNorm or RMSNorm, learned positions
or RoPE, GELU or SwiGLU — the tests check behavior (shapes, causality, learning), not
your choices. Causality is the test that bites: position `t` must be unaffected by any
token after `t`.

## Scaffolding provided

- `copy_task_batch()` in the starter — a deterministic periodic-sequence dataset where
  the next token is a pure function of recent context, so a correct transformer can
  drive the loss near zero. Data plumbing is boilerplate; attention is the lesson.

## Rules

- `torch` and `torch.nn` primitives (`Linear`, `Embedding`, `LayerNorm`, …) are allowed.
- Banned (the whole point is to build them): `nn.MultiheadAttention`, `nn.Transformer*`,
  `F.scaled_dot_product_attention`, and any import from `ai_playground.models`.

## Done-when

```bash
uv run pytest projects/p1_tiny_gpt/ -v
```

Afterwards, diff against `src/ai_playground/models/` and the original papers:
[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762),
[LLaMA (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) — both in docs/PAPERS.md.
