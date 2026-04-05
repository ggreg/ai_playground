---
name: new-notebook
description: Create a new Jupyter notebook for the AI playground. Use when the user wants to add a new experiment, tutorial, or exploration notebook.
user_invocable: true
---

# Create a New Notebook

When creating a new notebook for this educational AI playground, follow these conventions:

## Placement

Notebooks go in `notebooks/` under the appropriate topic module:
- `01_transformer_internals/` — model architecture components (attention, embeddings, normalization, etc.)
- `02_training_optimization/` — training efficiency (mixed precision, grad accumulation, LR schedules, optimizers)
- `03_distributed_training/` — multi-GPU/multi-node (DDP, FSDP, tensor parallel, pipeline parallel)
- `04_inference_optimization/` — serving speed (KV cache, quantization, speculative decoding, batching)
- `05_gpu_nvidia_tools/` — GPU programming (CUDA, Triton kernels, Nsight, TensorRT, NCCL)

Filename format: `NN_topic_name.ipynb` where NN is the next sequential number in that directory.

## Structure

Every notebook must follow this structure:

1. **Title + Introduction** (markdown) — What are we exploring? Why does it matter for LLMs? What will the reader learn?

2. **Setup cell** — Standard imports:
```python
import sys
sys.path.insert(0, '../src')

import torch
import matplotlib.pyplot as plt
from ai_playground.models import Transformer, TransformerConfig
# ... other imports as needed

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')
```

3. **Build from scratch first** — Implement the concept manually with explicit, readable code before using library/optimized versions. Show every step.

4. **Compare approaches** — When exploring alternatives (e.g., GQA vs MQA, RoPE vs ALiBi), implement all variants and benchmark them side by side.

5. **Visualize** — Use matplotlib to show attention patterns, memory profiles, training curves, latency comparisons, etc. Every notebook should produce at least one visualization.

6. **Include concrete numbers** — Parameter counts, memory in MB, tokens/sec, speedup ratios. Not "faster" but "2.3x faster".

7. **Key Takeaways** (markdown) — Summarize the 3-5 most important lessons. Include a pointer to the next related notebook.

## Code Style in Notebooks

- Use the `ai_playground` package for reusable components — don't redefine what's already in `src/`
- Add explanatory comments inline, especially for tensor shape transformations
- Show tensor shapes in comments: `# (batch, seq_len, n_heads, head_dim)`
- When benchmarking, always include warmup runs before measurement
- Gate CUDA-specific code behind `if device == 'cuda':` checks so notebooks run on CPU/MPS too

## After Creating

- Run all cells to verify the notebook executes without errors
- Ensure outputs are cleared before committing (no stale output cells)
- Update the README.md notebook table if adding a notebook to a new topic area
