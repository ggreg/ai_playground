# The Project: Serve Your Own LLM

One project runs through every chapter of this book: **design, train, and analyze your own
small LLM, then serve it on the mini-vLLM engine you build in module 04** — paged KV cache,
continuous batching, real text from your own weights through your own inference engine.

Each chapter ends in a **🏗️ Project milestone**: a coding exercise that produces a concrete
artifact toward the finale. Do them in book order and the finale is an afternoon; skip around
and each milestone still stands alone (it states what it needs from earlier ones).

## Workspace

All artifacts live in `checkpoints/myllm/` at the repo root — it's gitignored, so your model
stays local:

```
checkpoints/myllm/
├── config.json      # your model's design (M1, refined in M2/M5/M6)
├── DECISIONS.md     # design decisions with numbers (M3, ...)
├── metrics.json     # accumulating measurements (M0, M5, M6, M9, M11, M12)
├── step300.pt       # first trained checkpoint (M4)
└── final.pt         # the release checkpoint you'll serve (M7)
```

From a notebook, the workspace is:

```python
from pathlib import Path
WS = Path('../../checkpoints/myllm')
WS.mkdir(parents=True, exist_ok=True)
```

## Milestones

| # | Chapter | You build | Artifact |
|---|---------|-----------|----------|
| M0 | [Softmax & Cross-Entropy](../notebooks/00_dnn_refresher/03_softmax_crossentropy.ipynb) | An MLP classifier trained with your own autograd engine, gradients verified against PyTorch | `metrics.json` |
| M1 | [Transformer Overview](../notebooks/01_transformer_internals/00_transformer_overview.ipynb) | Your model's config under a 2M-parameter budget | `config.json` |
| M2 | [Attention Mechanisms](../notebooks/01_transformer_internals/01_attention_mechanisms.ipynb) | KV-cache budget → choose `n_kv_heads` (MHA/GQA/MQA) | updated `config.json` |
| M3 | [Mixture of Experts](../notebooks/01_transformer_internals/02_mixture_of_experts.ipynb) | Dense-vs-MoE decision, with the serving cost spelled out | `DECISIONS.md` |
| M4 | [Training From Scratch](../notebooks/02_training_optimization/00_training_from_scratch.ipynb) | First real training run of *your* config | `step300.pt` + loss curve |
| M5 | [Learning Rate Schedules](../notebooks/02_training_optimization/01_learning_rate_schedules.ipynb) | Schedule sweep on your model; adopt the winner | `metrics.json` |
| M6 | [Mixed Precision](../notebooks/02_training_optimization/02_mixed_precision.ipynb) | BF16 speedup measured on your model (GPU/Colab; CPU baseline otherwise) | `metrics.json` |
| M7 | [Gradient Accumulation](../notebooks/02_training_optimization/03_gradient_accumulation.ipynb) | Effective-batch choice + the release training run | `final.pt` |
| M8 | [AdamW From Scratch](../notebooks/02_training_optimization/04_adamw_from_scratch.ipynb) | Prove the from-scratch optimizer on *your* model | passing assert |
| M9 | [CUDA Basics](../notebooks/05_gpu_nvidia_tools/01_cuda_basics.ipynb) | Your model's decode roofline: where does it sit on a T4? | `metrics.json` |
| M10 | [SIMT Simulator](../notebooks/05_gpu_nvidia_tools/01b_simt_simulator.ipynb) | A `paged_gather` kernel + its coalescing analysis | verified kernel |
| M11 | [Virtual GPU](../notebooks/05_gpu_nvidia_tools/01c_virtual_gpu.ipynb) | Predicted decode-step time for your config on the virtual T4 | `metrics.json` |
| M12 | [SGEMM Optimization](../notebooks/05_gpu_nvidia_tools/02_sgemm_optimization.ipynb) | Your decode GEMM shapes, benchmarked against cuBLAS (Colab) | `metrics.json` |
| M13 | [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) | **The finale**: your trained model served through your paged engine | generated text |

## The finale (M13), concretely

Load `final.pt` into the mini-vLLM decode path (the chapter's `MiniLayer` maps 1:1 onto the
repo's `Transformer` weights — q/k/v/o projections, SwiGLU, RMSNorms), then:

1. **Prove it**: greedy tokens from the paged engine must equal a full-recompute reference for
   3 prompts — the same assert the chapter uses for its toy model, now on your weights.
2. **Serve it**: a 16-request workload through continuous batching, block pool deliberately
   smaller than worst-case so preemption fires at least once.
3. **Read it**: decode the first request's output with your tokenizer. It's your model's text,
   through your engine — the whole book in one function call.

## Ground rules

- **Milestones state acceptance criteria** — an assert or a number to record. Green means done.
- **`metrics.json` accumulates**; never overwrite earlier keys — the finale prints the whole
  journey.
- GPU-flagged milestones (M6, M9, M11, M12) run on the chapter's Colab badge (T4); each has a
  stated CPU fallback so no milestone hard-blocks the finale.
- Stuck? Every chapter's own code is the worked example: milestones ask you to re-apply it to
  *your* model, never to invent unseen machinery.
