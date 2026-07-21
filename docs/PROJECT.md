# The Project: Serve Your Own LLM

One project runs through every chapter of this book: **design, train, and analyze your own
small LLM, then serve it on the mini-vLLM engine you build in module 04** — paged KV cache,
continuous batching, real text from your own weights through your own inference engine.

Each chapter ends in a **🏗️ Project milestone**: a coding exercise that produces a concrete
artifact toward the finale. Do them in book order and the finale is an afternoon; skip around
and each milestone still stands alone (it states what it needs from earlier ones).

The book is paced for **one-hour sessions** — the [Session Guide](SESSIONS.md) maps every
milestone (splitting the big ones) onto session cards, and every milestone has an executable
acceptance test in `tests/milestones/` (see [How to Read This Book](HOW_TO_READ.md)):

```bash
uv run pytest tests/milestones/   # skips what you haven't reached; green means done
```

## Workspace

All artifacts live in `checkpoints/myllm/` at the repo root — it's gitignored, so your model
stays local:

```
checkpoints/myllm/
├── config.json           # your model's design (M1, refined in M2/M5/M6)
├── DECISIONS.md          # design decisions with numbers (M3, M7a, ...)
├── metrics.json          # accumulating measurements (M0, M5, M6, M8–M12, M13d)
├── loss_smoke.json       # 20-step smoke-run losses (M4a)
├── step300.pt            # first trained checkpoint (M4b)
├── loss_step300.json     # its loss curve (M4b)
├── final.pt              # the release checkpoint you'll serve (M7b)
├── loss_final.json       # its loss curve (M7b)
├── PROFILE.md            # your reader profile (written by /onboard)
└── src/
    └── serve_myllm.py    # your engine adaptation (M13a–c) — importable by the tests
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
| M4a | [Training From Scratch](../notebooks/02_training_optimization/00_training_from_scratch.ipynb) | Your config wired into the training loop; 20-step smoke run, loss decreasing | `loss_smoke.json` |
| M4b | [Training From Scratch](../notebooks/02_training_optimization/00_training_from_scratch.ipynb) | The real 300-step run (launched at M4a's end, finishes between sessions) + curve diagnosis | `step300.pt` + `loss_step300.json` |
| M5 | [Learning Rate Schedules](../notebooks/02_training_optimization/01_learning_rate_schedules.ipynb) | Schedule sweep on your model; adopt the winner | `metrics.json` |
| M6 | [Mixed Precision](../notebooks/02_training_optimization/02_mixed_precision.ipynb) | BF16 speedup measured on your model (GPU/Colab; CPU baseline otherwise) | `metrics.json` |
| M7a | [Gradient Accumulation](../notebooks/02_training_optimization/03_gradient_accumulation.ipynb) | Effective-batch decision with numbers; release run launched | `DECISIONS.md` (`## M7`) |
| M7b | [Gradient Accumulation](../notebooks/02_training_optimization/03_gradient_accumulation.ipynb) | Release checkpoint verified; beats `step300.pt`; first sampled text | `final.pt` + `loss_final.json` |
| M8 | [AdamW From Scratch](../notebooks/02_training_optimization/04_adamw_from_scratch.ipynb) | Prove the from-scratch optimizer on *your* model | `metrics.json` (`m8_optimizer_max_diff`) |
| M9 | [CUDA Basics](../notebooks/05_gpu_nvidia_tools/01_cuda_basics.ipynb) | Your model's decode roofline: where does it sit on a T4? | `metrics.json` |
| M10 | [SIMT Simulator](../notebooks/05_gpu_nvidia_tools/01b_simt_simulator.ipynb) | A `paged_gather` kernel + its coalescing analysis | verified kernel |
| M11 | [Virtual GPU](../notebooks/05_gpu_nvidia_tools/01c_virtual_gpu.ipynb) | Predicted decode-step time for your config on the virtual T4 | `metrics.json` |
| M12 | [SGEMM Optimization](../notebooks/05_gpu_nvidia_tools/02_sgemm_optimization.ipynb) | Your decode GEMM shapes, benchmarked against cuBLAS (Colab) | `metrics.json` |
| M13a | [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) | `final.pt`'s weights mapped onto the engine's layer structure | `src/serve_myllm.py` (`load_model`) |
| M13b | [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) | **Prove it**: paged greedy tokens ≡ full-recompute reference, 3 prompts × 20 tokens | `serve_myllm.py` (`greedy_paged`) |
| M13c | [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) | **Serve it**: 16 requests, continuous batching, ≥1 forced preemption | `serve_myllm.py` (`serve`) |
| M13d | [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) | **Read it — the finale**: your model's text through your engine + the metrics retrospective | `metrics.json` (`m13_finale`) |

## The finale (M13), concretely

The finale is four sessions ([S4.3–S4.6](SESSIONS.md)), and its code lives in a real module —
`checkpoints/myllm/src/serve_myllm.py` — so the milestone tests can import *your* engine:

1. **M13a — Adapt it**: `load_model(ws)` loads `final.pt` and maps every weight onto the
   engine's layer structure (the chapter's `MiniLayer` maps 1:1 onto the repo's
   `Transformer` — q/k/v/o projections, SwiGLU, RMSNorms).
2. **M13b — Prove it**: `greedy_paged` tokens must equal `greedy_reference` (full recompute)
   for 3 prompts × 20 greedy tokens — the same assert the chapter uses for its toy model,
   now on your weights.
3. **M13c — Serve it**: `serve(requests, max_new, block_budget)` runs a 16-request workload
   through continuous batching, block pool deliberately smaller than worst-case so preemption
   fires at least once.
4. **M13d — Read it**: decode the first request's output with your tokenizer. It's your
   model's text, through your engine — the whole book in one function call.

## Ground rules

- **Milestones state acceptance criteria** — an assert or a number to record, executable as
  `uv run pytest tests/milestones/`. Green means done; nothing else does.
- **Long runs never block a session**: M4b's 300-step run and M7b's release run are launched
  at the end of one session and harvested at the start of the next.
- **`metrics.json` accumulates**; never overwrite earlier keys — the finale prints the whole
  journey.
- GPU-flagged milestones (M6, M9, M11, M12) run on the chapter's Colab badge (T4); each has a
  stated CPU fallback so no milestone hard-blocks the finale.
- Stuck? Every chapter's own code is the worked example: milestones ask you to re-apply it to
  *your* model, never to invent unseen machinery.
