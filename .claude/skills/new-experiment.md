---
name: new-experiment
description: Add a new module, layer, or experiment to the AI playground source code. Use when the user wants to implement a new model component, training technique, inference optimization, or profiling tool.
user_invocable: true
---

# Add a New Experiment

When adding new source code to the AI playground, follow these guidelines:

## Where Code Goes

- **New model components** (attention variants, normalization, activations, positional encodings) → `src/ai_playground/models/`
- **Training techniques** (optimizers, schedulers, data strategies) → `src/ai_playground/training/`
- **Inference optimizations** (decoding strategies, quantization, caching) → `src/ai_playground/inference/`
- **Profiling/measurement tools** → `src/ai_playground/profiling/`
- **New model configs** → `configs/` as YAML files

## Implementation Checklist

1. **Read the existing code first** — Check if similar functionality exists. Reuse `TransformerConfig`, existing attention implementations, etc. Don't duplicate.

2. **Write the implementation** with educational docstrings:
   - Explain WHAT the technique is (1-2 sentences)
   - Explain WHY it matters (performance gain, memory reduction, quality improvement)
   - Reference the paper or source (author, year, paper title)
   - Include concrete numbers where possible

3. **Ensure CPU/MPS compatibility** — GPU-specific code (CUDA kernels, Flash Attention, Triton) must be gated behind availability checks. The core path must work on CPU.

4. **Write tests** in the corresponding `tests/test_*.py` file:
   - Test output shapes
   - Test numerical correctness (no NaN/Inf)
   - Test against reference implementations where possible
   - Use `@pytest.mark.parametrize` for testing variants
   - For attention: test causal masking (future tokens don't affect current output)
   - For training: test that loss decreases on a toy problem

5. **Run the full test suite** to verify nothing is broken:
   ```bash
   uv run pytest -v
   ```

6. **Update `__init__.py`** exports if adding new public APIs

7. **If adding a new model component**, verify `TransformerConfig.num_params()` still returns accurate counts

## Code Conventions

- Python 3.11+ type hints: `X | None` not `Optional[X]`, `list[int]` not `List[int]`
- No bias in Linear layers: `nn.Linear(in, out, bias=False)`
- Pre-norm pattern: normalize → compute → residual add
- Round hidden dimensions to multiples of 64 for GPU efficiency
- Use `torch.amp.autocast` for mixed precision, not the deprecated `torch.cuda.amp`
- Line length: 100 chars

## Config Format

If adding a new model variant, create a YAML config:
```yaml
model:
  vocab_size: 32000
  dim: 512
  n_layers: 8
  n_heads: 8
  n_kv_heads: 4
  max_seq_len: 1024

training:
  max_steps: 1000
  batch_size: 8
  learning_rate: 3.0e-4
  # ... etc
```

All `TransformerConfig` fields are valid under `model:`. All `TrainingConfig` fields are valid under `training:`.
