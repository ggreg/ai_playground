# Session Guide

This book is designed to be read in **one-hour sessions**, each pairing reading and
experimentation with a hands-on step of [the capstone project](PROJECT.md) — done alongside
a coding agent acting as your tutor (see [How to Read This Book](HOW_TO_READ.md)).

Every session has the same shape:

1. **Read & experiment** (~20 min) — the chapter section and notebook cells for the hour.
2. **Build** (~30 min) — *you* write the project code; the agent hints, reviews, and keeps time.
3. **Verify & log** (~10 min) — run the session's **Done-when** check, log the session.

Each card marks a **Midpoint**: stop there if you're splitting the session into two
30-minute sittings. **Needs** lists prerequisite sessions; the agent checks them before
starting. Milestone acceptance checks live in `tests/milestones/` — a session is done when
its test passes, not when anyone says it looks done. The full minute-by-minute protocol,
the tutor's rules, and the hint ladder are documented in
[Inside a Session](SESSION_PROTOCOL.md).

Sessions are numbered `S<module>.<k>` after the notebook modules (`notebooks/00_…` → S0.x).
**Read them in the order of this page** — note the GPU module (S5.x) comes before the
finale (S4.x), matching the milestone order in [The Project](PROJECT.md).

---

## Phase 0 — DNN Refresher (optional)

Skippable if MLPs, backprop, and cross-entropy are second nature — but skipping requires
passing the phase self-check the agent administers during `/onboard`. M0's artifacts are
referenced again at the finale, so if you skip the sessions, still do S0.5.

## S0.1 — Neurons, layers, and the forward pass

- **Chapter**: [Neurons & MLPs](../notebooks/00_dnn_refresher/00_neurons_and_mlps.ipynb)
- **Read** (~40 min): the whole chapter; run every cell. Rebuild the XOR truth table by
  hand before running the XOR section.
- **Build** (~15 min): in a scratch cell, change the MLP width/depth on `make_moons` and
  find the smallest network that still separates the moons cleanly
  (`plot_decision_boundary` is your instrument).
- **Midpoint**: after the universal-approximation section.
- **Done-when**: self-check quiz with the agent (topics: why stacked linear layers
  collapse; what a hidden unit's decision boundary looks like; parameter count of
  `MLP(2, [8, 3])`).
- **Needs**: nothing.
- **Artifact**: none (read-only session).

## S0.2 — Backprop from scratch

- **Chapter**: [Backprop & Micrograd](../notebooks/00_dnn_refresher/01_backprop_micrograd.ipynb)
- **Read** (~35 min): the whole chapter; run every cell. Before running the `.backward()`
  cell, predict the gradients of the worked example by hand.
- **Build** (~20 min): pick one expression with shared subexpressions (e.g.
  `y = x*x + x*z`), derive `dy/dx` by hand, then verify with `Value` — explain to the
  agent why `grad +=` (not `=`) is what makes the shared case correct.
- **Midpoint**: after the chain-rule-on-a-graph section, before the `Value` class.
- **Done-when**: self-check quiz (topics: topological order; gradient accumulation;
  what `zero_grad` prevents).
- **Needs**: S0.1.
- **Artifact**: none.

## S0.3 — The training loop

- **Chapter**: [The Training Loop](../notebooks/00_dnn_refresher/02_training_loop.ipynb)
- **Read** (~30 min): the whole chapter; run every cell.
- **Build** (~25 min): break the loop on purpose, one line at a time — remove `zero_grad`,
  flip the sign of the step, set the LR 100x too high — predict each failure mode before
  running it, then watch the loss curve confirm or refute you.
- **Midpoint**: before the minibatch section.
- **Done-when**: self-check quiz (topics: the five lines of the loop; what each failure
  mode above looks like on a loss curve).
- **Needs**: S0.2.
- **Artifact**: none.

## S0.4 — Softmax and cross-entropy

- **Chapter**: [Softmax & Cross-Entropy](../notebooks/00_dnn_refresher/03_softmax_crossentropy.ipynb)
- **Read** (~40 min): the chapter up to (not including) the M0 milestone; run every cell.
  Derive the `p − y` gradient on paper before the notebook derives it for you.
- **Build** (~15 min): with vocab_size = 3, compute one full softmax + CE forward pass
  with a calculator and match the notebook's numbers.
- **Midpoint**: after the CE-vs-MSE section.
- **Done-when**: self-check quiz (topics: subtract-max stability; temperature; why MSE's
  gradient vanishes where CE's doesn't).
- **Needs**: S0.3.
- **Artifact**: none.

## S0.5 — 🏗️ M0: train an MLP classifier with your own engine

- **Chapter**: [Softmax & Cross-Entropy](../notebooks/00_dnn_refresher/03_softmax_crossentropy.ipynb) § Project milestone M0
- **Read** (~10 min): the M0 milestone cell; skim
  `tests/test_fundamentals.py::TestMLP::test_grad_parity_with_torch` for the weight-copy
  recipe.
- **Build** (~40 min): part 1 — gradient parity on one 16-point spiral batch
  (`fundamentals` engine vs identical-weight torch model); record
  `m0_grad_max_abs_diff`. Part 2 — train an MLP on the full 300-point spiral; record
  `m0_spiral_train_acc`. Both keys go in `checkpoints/myllm/metrics.json`.
- **Midpoint**: between part 1 and part 2.
- **Done-when**: `uv run pytest tests/milestones/test_m00_classifier.py`
- **Needs**: S0.4 (or a passed Phase 0 skip-quiz).
- **Artifact**: `checkpoints/myllm/metrics.json` (`m0_grad_max_abs_diff < 1e-6`,
  `m0_spiral_train_acc >= 0.95`).

---

## Phase 1 — Transformer Internals

## S1.1 — Transformer overview, part I: the parts

- **Chapter**: [Transformer Overview](../notebooks/01_transformer_internals/00_transformer_overview.ipynb)
- **Read** (~45 min): the first half — embeddings through the transformer block; run every
  cell. Before each parameter-count cell, predict the count from the shapes.
- **Build** (~10 min): change one knob (`dim`, `n_layers`, `n_heads`) in the tiny config
  and predict, then verify, the new parameter count.
- **Midpoint**: after the attention section.
- **Done-when**: self-check quiz (topics: where the parameters live; what the embedding
  table costs; what `n_heads` changes and what it doesn't).
- **Needs**: S0.5 (or Phase 0 skip-quiz).
- **Artifact**: none.

## S1.2 — 🏗️ M1: design your model (transformer overview, part II)

- **Chapter**: [Transformer Overview](../notebooks/01_transformer_internals/00_transformer_overview.ipynb), second half + § Project milestone M1
- **Read** (~20 min): finish the chapter — training the tiny model and `generate_tiny`.
- **Build** (~30 min): design *your* model under the 2M-parameter budget: pick `dim`,
  `n_layers`, `n_heads`, `vocab_size`; build a `TransformerConfig`; verify
  `config.num_params() <= 2_000_000`; run a `(4, 32)` batch through it; save
  `checkpoints/myllm/config.json`.
- **Midpoint**: after finishing the chapter reading.
- **Done-when**: `uv run pytest tests/milestones/test_m01_config.py`
- **Needs**: S1.1.
- **Artifact**: `checkpoints/myllm/config.json`.

## S1.3 — 🏗️ M2: attention shapes and your KV-cache budget

- **Chapter**: [Attention Mechanisms](../notebooks/01_transformer_internals/01_attention_mechanisms.ipynb)
- **Read** (~25 min): the chapter through the MHA/GQA/MQA quality comparison; run every
  cell. Redo the chapter's cache-size table for your model's `dim` before the Build.
- **Build** (~25 min): compute KV bytes/token for your config
  (`2 × n_layers × n_kv_heads × head_dim × 4` in fp32) across MHA/GQA/MQA; choose the
  largest `n_kv_heads` that keeps batch 8 × 512 tokens under 8 MB; rerun the quality
  comparison at your sizes; update `config.json`.
- **Midpoint**: after the Read block.
- **Done-when**: `uv run pytest tests/milestones/test_m02_kv_budget.py`
- **Needs**: S1.2.
- **Artifact**: updated `checkpoints/myllm/config.json` (`n_kv_heads`).

## S1.4 — 🏗️ M3: dense or MoE, decided with numbers

- **Chapter**: [Mixture of Experts](../notebooks/01_transformer_internals/02_mixture_of_experts.ipynb)
- **Read** (~30 min): the whole chapter; run every cell.
- **Build** (~20 min): compute a 4-expert top-1 MoE variant of your config — total vs
  *active* parameters — and write the dense-vs-MoE decision to
  `checkpoints/myllm/DECISIONS.md` under a `## M3` heading, with both counts and one
  paragraph on the M13 serving cost either way.
- **Midpoint**: after the routing/load-balancing sections.
- **Done-when**: `uv run pytest tests/milestones/test_m03_decisions.py`
- **Needs**: S1.3.
- **Artifact**: `checkpoints/myllm/DECISIONS.md` (`## M3` entry).

---

## Phase 2 — Training Optimization

## S2.1 — The training loop, at full scale

- **Chapter**: [Training From Scratch](../notebooks/02_training_optimization/00_training_from_scratch.ipynb)
- **Read** (~45 min): the whole chapter; run every cell. This is S0.3's loop with real
  tensors — name the correspondence line by line as you go.
- **Build** (~10 min): change one training knob (batch size or LR) and predict the loss
  curve's reaction before rerunning.
- **Midpoint**: before the loss-curve-reading section.
- **Done-when**: self-check quiz (topics: what a healthy loss curve looks like; the three
  pathologies and their causes).
- **Needs**: S1.2 (M2/M3 recommended but not required).
- **Artifact**: none.

## S2.2 — 🏗️ M4a: your model trains (smoke run)

- **Chapter**: [Training From Scratch](../notebooks/02_training_optimization/00_training_from_scratch.ipynb) § Project milestone M4
- **Read** (~10 min): the M4 milestone cell.
- **Build** (~40 min): load `checkpoints/myllm/config.json`, build the `Transformer`, and
  point the chapter's loop at it. Run a **20-step smoke run** on CPU/MPS; save the losses
  to `checkpoints/myllm/loss_smoke.json` (a JSON list of floats). Loss must end lower than
  it starts. **At the end of the session**, launch the real 300-step run
  (it finishes between sessions): save `step300.pt`
  (`{'config': cfg_dict, 'model': state_dict}`) and `loss_step300.json` when it completes.
- **Midpoint**: once the smoke run's first losses print.
- **Done-when**: `uv run pytest tests/milestones/test_m04a_smoke.py`
- **Needs**: S2.1.
- **Artifact**: `checkpoints/myllm/loss_smoke.json`; 300-step run launched.

## S2.3 — 🏗️ M4b: read your first real loss curve

- **Chapter**: [Training From Scratch](../notebooks/02_training_optimization/00_training_from_scratch.ipynb) (loss-curve sections, revisited)
- **Read** (~15 min): revisit the chapter's loss-curve pathology gallery with *your*
  curve on screen.
- **Build** (~35 min): verify `step300.pt` reloads into a fresh model without key errors;
  plot `loss_step300.json`; annotate the curve — warmup transient? plateau? final loss
  < 60% of initial? — and write one paragraph of diagnosis into your session log.
- **Midpoint**: after the reload check.
- **Done-when**: `uv run pytest tests/milestones/test_m04b_step300.py`
- **Needs**: S2.2 (with the 300-step run finished).
- **Artifact**: `checkpoints/myllm/step300.pt` + `loss_step300.json`, verified.

## S2.4 — 🏗️ M5: sweep schedules on your model

- **Chapter**: [Learning Rate Schedules](../notebooks/02_training_optimization/01_learning_rate_schedules.ipynb)
- **Read** (~20 min): the whole chapter. **Start the sweep first** (3 × 200 steps runs in
  the background while you read).
- **Build** (~30 min): sweep constant / cosine / warmup+cosine on your model — same seed
  and init, 200 steps each. Record all three final losses and your choice in
  `metrics.json['m5_schedule_sweep']` (`{'constant': …, 'cosine': …, 'warmup_cosine': …,
  'chosen': '…'}`) and add `"schedule": "<chosen>"` to `config.json`.
- **Midpoint**: after the Read block (sweep still running is fine — log PARTIAL and
  verify next sitting).
- **Done-when**: `uv run pytest tests/milestones/test_m05_schedule.py`
- **Needs**: S2.3.
- **Artifact**: `metrics.json` (`m5_schedule_sweep`) + `config.json` (`schedule`).

## S2.5 — 🏗️ M6: measure your precision speedup

- **Chapter**: [Mixed Precision](../notebooks/02_training_optimization/02_mixed_precision.ipynb)
- **Read** (~30 min): the whole chapter; run every cell (Colab badge for the GPU parts).
- **Build** (~20 min): GPU (Colab T4): 200 steps fp32 vs bf16 autocast on your model;
  record tokens/sec for both and the loss gap in `metrics.json['m6_precision']`. CPU
  only: record the fp32 baseline plus a note — the discipline is the milestone, not the
  hardware. Add `"precision": "bf16"` (or `"fp32"`) to `config.json`.
- **Midpoint**: after the loss-scaling section.
- **Done-when**: `uv run pytest tests/milestones/test_m06_precision.py`
- **Needs**: S2.3.
- **Artifact**: `metrics.json` (`m6_precision`) + `config.json` (`precision`).

## S2.6 — 🏗️ M7a: the effective-batch decision (and launch the release run)

- **Chapter**: [Gradient Accumulation](../notebooks/02_training_optimization/03_gradient_accumulation.ipynb)
- **Read** (~30 min): the whole chapter; run every cell — especially the
  loss-scaling-by-accumulation-steps trap.
- **Build** (~20 min): find the largest micro-batch your memory allows, set
  `accum_steps = 64 / micro_batch`, and record the decision (numbers + reasoning) in
  `DECISIONS.md` under `## M7`. **Then launch the release run**: 1,000 steps, your adopted
  schedule (M5) and precision (M6) → `final.pt` + `loss_final.json`, finishing between
  sessions.
- **Midpoint**: after the Read block.
- **Done-when**: `uv run pytest tests/milestones/test_m07a_decision.py`
- **Needs**: S2.4, S2.5.
- **Artifact**: `DECISIONS.md` (`## M7` entry); release run launched.

## S2.7 — 🏗️ M7b: the release checkpoint — and first words

- **Chapter**: [Gradient Accumulation](../notebooks/02_training_optimization/03_gradient_accumulation.ipynb) § Project milestone M7
- **Read** (~10 min): none new — this session harvests the release run.
- **Build** (~40 min): verify `final.pt` reloads cleanly and `loss_final.json` beats
  `step300.pt`'s final loss; plot both curves together. Then the payoff: load `final.pt`
  into the repo's `inference/generate.py` and sample your model's first text (it will be
  bad — it's 2M parameters; note what "bad" looks like, you'll compare at M13).
- **Midpoint**: after the verification, before sampling.
- **Done-when**: `uv run pytest tests/milestones/test_m07b_release.py`
- **Needs**: S2.6 (with the release run finished).
- **Artifact**: `checkpoints/myllm/final.pt` + `loss_final.json`, verified.

## S2.8 — 🏗️ M8: trust your optimizer

- **Chapter**: [AdamW From Scratch](../notebooks/02_training_optimization/04_adamw_from_scratch.ipynb)
- **Read** (~30 min): the whole chapter; run every cell. Predict what removing bias
  correction does before the chapter shows you.
- **Build** (~20 min): 20 steps of *your* model twice from identical init and data order —
  `AdamWFromScratch` vs `torch.optim.AdamW`, same hyperparameters, fp32. Record the worst
  per-parameter deviation in `metrics.json['m8_optimizer_max_diff']` (must be < 1e-5).
- **Midpoint**: after the bias-correction section.
- **Done-when**: `uv run pytest tests/milestones/test_m08_adamw.py`
- **Needs**: S2.3.
- **Artifact**: `metrics.json` (`m8_optimizer_max_diff`).

---

## Phase 5 — GPU & NVIDIA Tools

Yes, before the finale: M9–M12 build the performance vocabulary the mini-vLLM chapter
assumes, and M10's paged gather is the finale's memory access pattern.

## S5.1 — The GPU mental model

- **Chapter**: [CUDA Basics](../notebooks/05_gpu_nvidia_tools/01_cuda_basics.ipynb)
- **Read** (~45 min): the whole chapter; run every cell.
- **Build** (~10 min): compute arithmetic intensity for one op you know well (your
  model's output head GEMM at batch 1) and place it on the chapter's roofline by hand.
- **Midpoint**: after the memory-hierarchy section.
- **Done-when**: self-check quiz (topics: coalescing; occupancy; why decode is
  bandwidth-bound).
- **Needs**: S2.3 (your model must exist and mean something to you).
- **Artifact**: none.

## S5.2 — 🏗️ M9: your model's decode roofline

- **Chapter**: [CUDA Basics](../notebooks/05_gpu_nvidia_tools/01_cuda_basics.ipynb) § Project milestone M9
- **Read** (~10 min): the M9 milestone cell; the T4 roofline section again.
- **Build** (~40 min): decode FLOPs/token ≈ `2 × num_params`; bytes/token ≈ weight bytes +
  KV bytes at the current context. Compute your model's decode arithmetic intensity at
  batch 1, 8, 32; find the bandwidth→compute crossover batch; record
  `metrics.json['m9_roofline']` as `{'intensity_b1': …, 'intensity_b8': …,
  'intensity_b32': …, 'crossover_batch': …, 'conclusion': '…'}`. Cross-check FLOPs with
  `ai_playground.profiling.flops`.
- **Midpoint**: after the batch-1 intensity.
- **Done-when**: `uv run pytest tests/milestones/test_m09_roofline.py`
- **Needs**: S5.1.
- **Artifact**: `metrics.json` (`m9_roofline`).

## S5.3 — SIMT on paper that runs

- **Chapter**: [SIMT Simulator](../notebooks/05_gpu_nvidia_tools/01b_simt_simulator.ipynb)
- **Read** (~45 min): the whole chapter; run every cell. Predict each
  `coalescing_report` before running it.
- **Build** (~10 min): write one deliberately awful (strided) access pattern and confirm
  the transaction count the report predicts.
- **Midpoint**: after the warp-execution sections.
- **Done-when**: self-check quiz (topics: warps vs threads; what makes an access
  coalesced; transactions-vs-ideal).
- **Needs**: S5.1.
- **Artifact**: none.

## S5.4 — 🏗️ M10: the paged gather, verified

- **Chapter**: [SIMT Simulator](../notebooks/05_gpu_nvidia_tools/01b_simt_simulator.ipynb) § Project milestone M10
- **Read** (~10 min): the M10 milestone cell; skim the mini-vLLM chapter's block-table
  figure so you know what you're building toward.
- **Build** (~40 min): write `paged_gather(ctx, pool, table, out)` as a simulator kernel —
  each thread copies one element of one token-row from its physical block into contiguous
  output. Verify against numpy fancy-indexing; run `coalescing_report`; record
  `metrics.json['m10_coalescing']` as `{'block_size': …, 'transactions_ratio': …}`.
- **Midpoint**: kernel matches the reference (before the coalescing analysis).
- **Done-when**: `uv run pytest tests/milestones/test_m10_paged_gather.py`
- **Needs**: S5.3.
- **Artifact**: `metrics.json` (`m10_coalescing`).

## S5.5 — A virtual T4 with a time axis

- **Chapter**: [Virtual GPU](../notebooks/05_gpu_nvidia_tools/01c_virtual_gpu.ipynb)
- **Read** (~45 min): the whole chapter; run every cell.
- **Build** (~10 min): change one occupancy knob in a chapter experiment and predict the
  trace before rerunning.
- **Midpoint**: after the latency-hiding sections.
- **Done-when**: self-check quiz (topics: latency hiding; why more resident warps help
  until they don't; reading achieved GB/s from a trace).
- **Needs**: S5.3.
- **Artifact**: none.

## S5.6 — 🏗️ M11: predict your decode step

- **Chapter**: [Virtual GPU](../notebooks/05_gpu_nvidia_tools/01c_virtual_gpu.ipynb) § Project milestone M11
- **Read** (~10 min): the M11 milestone cell.
- **Build** (~40 min): one decode step at batch 8 moves ≈ `weight_bytes + 8 × kv_bytes(t)`
  from DRAM; simulate that traffic on the virtual T4 (a copy kernel sized to those bytes),
  read predicted cycles and achieved GB/s, convert to predicted tokens/sec; record
  `metrics.json['m11_predicted']` as `{'tokens_per_sec': …, 'assumptions': '…'}`. On
  Colab, run the chapter's validation cells and record the real bandwidth beside it.
- **Midpoint**: after computing the bytes, before simulating.
- **Done-when**: `uv run pytest tests/milestones/test_m11_predicted.py`
- **Needs**: S5.5, S5.2.
- **Artifact**: `metrics.json` (`m11_predicted`).

## S5.7 — SGEMM: naive to near-cuBLAS

- **Chapter**: [SGEMM Optimization](../notebooks/05_gpu_nvidia_tools/02_sgemm_optimization.ipynb) (Colab)
- **Read** (~50 min): the whole chapter on the Colab T4; run each kernel and watch the
  GFLOPs climb. Before each optimization, say what it buys and why.
- **Build** (~5 min): rank the optimizations by measured payoff — is the ranking what the
  chapter's narrative led you to expect?
- **Midpoint**: after the shared-memory tiling kernel.
- **Done-when**: self-check quiz (topics: tiling; why occupancy alone doesn't finish the
  job; where cuBLAS's remaining margin comes from).
- **Needs**: S5.1.
- **Artifact**: none.

## S5.8 — 🏗️ M12: your model's GEMM shapes

- **Chapter**: [SGEMM Optimization](../notebooks/05_gpu_nvidia_tools/02_sgemm_optimization.ipynb) § Project milestone M12
- **Read** (~10 min): the M12 milestone cell.
- **Build** (~40 min): list every GEMM in one decode step of your model —
  `(B×dim)@(dim×dim)` projections, `(B×dim)@(dim×hidden)` FFN, `(B×dim)@(dim×vocab)`
  head. On the Colab T4, time `torch.matmul` at those shapes for B ∈ {1, 8, 32}; record
  GFLOPs vs the chapter's kernels in `metrics.json['m12_gemms']` (CPU fallback: record
  the shape table only). Connect the batch-1 numbers to M9's conclusion.
- **Midpoint**: shape table done, before timing.
- **Done-when**: `uv run pytest tests/milestones/test_m12_gemms.py`
- **Needs**: S5.7, S5.2.
- **Artifact**: `metrics.json` (`m12_gemms`).

---

## Phase 4 — The Finale: Serve Your Own LLM

The finale's Build steps live in `checkpoints/myllm/src/serve_myllm.py` — a real module,
not notebook cells, so the milestone tests can import *your* engine
(`tests/milestones/conftest.py` puts `checkpoints/myllm/src` on `sys.path`). Start it at
S4.3; it grows one function per session.

## S4.1 — Mini-vLLM, part I: the paged KV cache

- **Chapter**: [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb), first half
- **Read** (~45 min): from the KV-cache memory problem through the paged cache and block
  tables; run every cell. Draw the block table for one of the chapter's examples by hand.
- **Build** (~10 min): change the block size in a chapter experiment and predict the
  fragmentation change before rerunning.
- **Midpoint**: after the block-table section.
- **Done-when**: self-check quiz (topics: internal fragmentation; block tables; why paging
  beats one contiguous buffer per request).
- **Needs**: S2.7, S5.4 recommended.
- **Artifact**: none.

## S4.2 — Mini-vLLM, part II: continuous batching

- **Chapter**: [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb), second half
- **Read** (~45 min): scheduling, continuous batching, preemption; run every cell through
  the chapter's toy-model correctness assert.
- **Build** (~10 min): shrink the toy example's block pool until preemption fires;
  narrate to the agent exactly what the scheduler did and why.
- **Midpoint**: after the scheduling section.
- **Done-when**: self-check quiz (topics: prefill vs decode scheduling; preemption and
  recompute; why the correctness assert compares against full recompute).
- **Needs**: S4.1.
- **Artifact**: none.

## S4.3 — 🏗️ M13a: your weights in the engine

- **Chapter**: [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) § Project milestone M13, step 1
- **Read** (~10 min): the M13 milestone cell, step 1; the `MiniLayer` ↔ repo
  `Transformer` weight correspondence (q/k/v/o projections, SwiGLU, RMSNorms).
- **Build** (~40 min): create `checkpoints/myllm/src/serve_myllm.py` with
  `load_model(ws: Path) -> model` — load `final.pt`, map every weight onto the engine's
  layer structure, expose the config. No generation yet; correct shapes and a clean load
  are the whole session.
- **Midpoint**: attention weights mapped; FFN and norms remaining.
- **Done-when**: `uv run pytest tests/milestones/test_m13a_weights.py`
- **Needs**: S4.2, S2.7.
- **Artifact**: `checkpoints/myllm/src/serve_myllm.py` (`load_model`).

## S4.4 — 🏗️ M13b: prove it — paged equals reference

- **Chapter**: [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) § Project milestone M13, step 2
- **Read** (~10 min): the chapter's correctness assert for the toy model — you are about
  to earn the same assert on your own weights.
- **Build** (~40 min): add to `serve_myllm.py`:
  `greedy_paged(prompt_ids: list[int], n: int) -> list[int]` (prefill + decode through
  the paged cache) and `greedy_reference(prompt_ids, n)` (full recompute each step). They
  must agree token-for-token: 3 prompts × 20 greedy tokens.
- **Midpoint**: `greedy_reference` working; paged path remaining.
- **Done-when**: `uv run pytest tests/milestones/test_m13b_parity.py`
- **Needs**: S4.3.
- **Artifact**: `serve_myllm.py` (`greedy_paged`, `greedy_reference`).

## S4.5 — 🏗️ M13c: serve it — continuous batching under pressure

- **Chapter**: [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) § Project milestone M13, step 3
- **Read** (~10 min): the chapter's scheduler loop once more.
- **Build** (~40 min): add `serve(requests: list[list[int]], max_new: int,
  block_budget: int) -> dict` to `serve_myllm.py` — continuous batching over your model,
  returning at least `{'completed': <int>, 'preemptions': <int>, 'outputs':
  [list[int], …]}`. Run 16 requests with a block pool deliberately smaller than
  worst-case: all 16 must complete and preemption must fire at least once.
- **Midpoint**: batching works with an oversized pool; now shrink it.
- **Done-when**: `uv run pytest tests/milestones/test_m13c_serve.py`
- **Needs**: S4.4.
- **Artifact**: `serve_myllm.py` (`serve`).

## S4.6 — 🏗️ M13d: read it — the whole book in one function call

- **Chapter**: [Build a Mini vLLM](../notebooks/04_inference_optimization/00_mini_vllm.ipynb) § Project milestone M13, step 4
- **Read** (~5 min): nothing new. You've read the book.
- **Build** (~40 min): decode the first request's output with your tokenizer — your
  model's text through your paged engine. Record
  `metrics.json['m13_finale'] = {'text': '…', 'tokens_per_sec': …}` and compare the
  measured rate against M11's prediction. Then print the accumulated `metrics.json` —
  M0's gradient parity to M13's text — and write the retrospective in your session log.
- **Midpoint**: text decoded; retrospective remaining.
- **Done-when**: `uv run pytest tests/milestones/test_m13d_finale.py`
- **Needs**: S4.5.
- **Artifact**: `metrics.json` (`m13_finale`) — and the paper trail complete.

---

## Phase 6 — Building AI Agents (epilogue, optional)

Read-only sessions; the agents package (`uv sync --extra agents`) is the worked example.

## S6.1 — What is an agent?

- **Chapter**: [What is an Agent?](../notebooks/06_agents/00_what_is_an_agent.ipynb)
- **Read** (~45 min): the whole chapter; run every cell.
- **Build** (~10 min): trace one loop iteration of `ai_playground.agents.agent.Agent`
  on paper — model call, tool call, observation — and match it to the chapter's diagram.
- **Midpoint**: after the loop diagram.
- **Done-when**: self-check quiz (topics: the ReAct loop; where the loop terminates; what
  the model actually sees each turn).
- **Needs**: S1.2 recommended.
- **Artifact**: none.

## S6.2 — Tool use

- **Chapter**: [Tool Use](../notebooks/06_agents/01_tool_use.ipynb)
- **Read** (~45 min): the whole chapter; run every cell.
- **Build** (~10 min): register one new tool of your own design in the `ToolRegistry`
  and watch the agent decide when to call it.
- **Midpoint**: after the tool-schema section.
- **Done-when**: self-check quiz (topics: tool schemas; when the model calls vs answers;
  error handling on tool failure).
- **Needs**: S6.1.
- **Artifact**: none.

## S6.3 — Memory

- **Chapter**: [Memory](../notebooks/06_agents/02_memory.ipynb)
- **Read** (~45 min): the whole chapter; run every cell.
- **Build** (~10 min): overflow `ConversationMemory` on purpose and inspect what the
  summarizer kept vs dropped.
- **Midpoint**: after the bounded-memory section.
- **Done-when**: self-check quiz (topics: bounded vs vector memory; what summarization
  loses; when similarity search retrieves the wrong thing).
- **Needs**: S6.2.
- **Artifact**: none.

---

## Not yet sessioned

- **Phase 3 — Distributed Training** (`notebooks/03_distributed_training/`): no chapters
  yet. Sessions will be added when the notebooks land.
- The [Performance Exercises](PERFORMANCE_EXERCISES.md) are excellent between-session
  material for readers with GPU time; they are deliberately outside the session sequence.
