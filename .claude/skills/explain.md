---
name: explain
description: Explain a concept, component, or technique in the AI playground codebase. Use when the user asks "how does X work", "why do we use X", or "explain X".
user_invocable: true
---

# Explain a Concept

When explaining AI/ML concepts in this playground, follow these guidelines:

## Structure

1. **One-sentence summary** — What is it, in plain language?
2. **Why it matters** — What problem does it solve? What's the performance/quality impact?
3. **How it works** — Walk through the implementation in this codebase, referencing specific files and line numbers.
4. **Concrete numbers** — Memory savings, speedup factors, quality impact. Reference benchmarks in the code or papers.
5. **Tradeoffs** — What do you give up? When should you NOT use this?
6. **Paper reference** — Author, title, year for the original paper.

## Code References

Always point to the actual implementation in this repo:
- Attention mechanisms: `src/ai_playground/models/attention.py`
- Layers (RMSNorm, RoPE, SwiGLU): `src/ai_playground/models/layers.py`
- Model configs: `src/ai_playground/models/config.py`
- Training loop: `src/ai_playground/training/trainer.py`
- KV cache + generation: `src/ai_playground/inference/generate.py`
- FLOP counting: `src/ai_playground/profiling/flops.py`

## Depth Guidelines

- **For architecture components** (attention, normalization, etc.): Show the math, then the code, then the intuition. Use tensor shape annotations.
- **For training techniques** (mixed precision, grad accumulation): Explain what happens to gradients/parameters at each step. Show the failure mode this prevents.
- **For GPU/systems concepts** (KV cache, memory bandwidth, MFU): Use concrete numbers for specific hardware. Reference the GPU TFLOPS table in `profiling/flops.py`.
- **For distributed training** (DDP, FSDP, tensor parallel): Draw the communication pattern. Explain what's sent between GPUs and when.

## Style

- Assume the reader knows PyTorch and linear algebra, but may not know the specific LLM technique
- Use tensor shape comments liberally: `# (batch, seq_len, n_heads, head_dim)`
- Compare to alternatives: "Unlike LayerNorm, RMSNorm skips the mean subtraction, making it ~15% faster with equivalent quality"
- If there's a relevant notebook, point to it for interactive exploration
