# Daily Progress Tracker

## Current Status

**Week:** 1
**Phase:** 1 — Understand the Model
**Cloud setup:** Not yet

---

## Daily Log

### Day 1 — Date: ____
- [ ] Session completed (30 min)
- **What I did:**
- **What surprised me:**
- **Tomorrow:**

---

## Phase 1: Understand the Model (Days 1–4)

### Day 1: Run tests, read attention code
- [ ] `uv run pytest -v` — all tests pass
- [ ] Read `tests/test_attention.py` — understand MHA/GQA/MQA parametrize
- [ ] Read `src/ai_playground/models/attention.py` — trace one forward pass mentally
- [ ] REPL: instantiate TINY config, print model, count params

### Day 2: Attention notebook + configs
- [ ] Work through `notebooks/01_transformer_internals/01_attention_mechanisms.ipynb`
- [ ] Change `n_kv_heads` to 1 (MQA), 8 (MHA), 4 (GQA) — predict param count before checking
- [ ] Read `models/config.py` — understand `num_params()` calculation

### Day 3: Layers deep dive
- [ ] Read `models/layers.py` — RMSNorm, RoPE, SwiGLU
- [ ] REPL: instantiate each layer, pass tensors through, check shapes
- [ ] Experiment: replace RMSNorm with nn.LayerNorm, run tests — do they still pass?

### Day 4: Full model + training loop
- [ ] Read `models/transformer.py` — trace a full forward pass
- [ ] Read `training/trainer.py` — understand the training loop, LR schedule, grad accum
- [ ] `uv run python scripts/train.py --config configs/tiny.yaml --max-steps 20`

## Phase 2: Train and Benchmark (Days 5–8)

### Day 5: Cloud setup
- [ ] Pick a provider (Lambda Cloud recommended — cheapest A100s)
- [ ] Create account, add SSH key, launch 1x A100 instance
- [ ] Clone repo, `uv sync --extra dev`, verify GPU with `torch.cuda.is_available()`
- [ ] Run: `uv run python scripts/train.py --config configs/tiny.yaml --max-steps 20`
- [ ] **Stop the instance when done**

### Day 6: Train small model on GPU
- [ ] Launch GPU instance, tmux session
- [ ] `uv run python scripts/train.py --config configs/small.yaml --dtype bfloat16 --max-steps 100`
- [ ] Watch loss curve — is it decreasing?
- [ ] Run benchmark: `uv run python scripts/benchmark.py --config configs/small.yaml --dtype bfloat16`
- [ ] Note: tokens/sec, TTFT, peak memory
- [ ] **Stop the instance**

### Day 7: Profile and measure
- [ ] `uv run python scripts/profile_model.py --config configs/small.yaml`
- [ ] Performance Exercise 1: Where Did My Memory Go? (use MemoryTracker)
- [ ] Predict activation memory before measuring — were you right?
- [ ] **Stop the instance**

### Day 8: MFU and compilation
- [ ] Performance Exercise 3: Compute MFU (use `profiling.flops.compute_mfu`)
- [ ] Try eager FP32 → BF16 → torch.compile → compile+BF16, record MFU for each
- [ ] Performance Exercise 8: torch.compile warmup — measure compilation cost
- [ ] **Stop the instance**

## Phase 3: Modify and Experiment (Days 9–14)

### Day 9: Swap architecture components
- [ ] Replace SwiGLU with GELU in `layers.py`, train 50 steps, compare loss
- [ ] Revert. Replace RMSNorm with LayerNorm, train 50 steps, compare loss
- [ ] Revert. Change GQA to MHA (`n_kv_heads=12`), benchmark — memory difference?

### Day 10: Attention scaling
- [ ] Performance Exercise 5: Attention vs FFN at different seq lengths
- [ ] Find the crossover point for small.yaml
- [ ] Compare MHA vs GQA vs MQA KV cache memory

### Day 11: Quantization
- [ ] Performance Exercise 4: Quantize small model to INT8
- [ ] Measure compression ratio, logit error, top-1 agreement
- [ ] Find which layers have the worst outlier ratios

### Day 12: Decode bottleneck
- [ ] Performance Exercise 7: Prefill vs decode speed
- [ ] Compute arithmetic intensity for both, compare to A100 roofline
- [ ] Vary batch size during decode — when does it become compute-bound?

### Day 13: Data loading
- [ ] Performance Exercise 9: Measure DataLoader with 0, 1, 4, 8 workers
- [ ] Compare data time vs compute time — are you data-bound?
- [ ] Test pin_memory effect

### Day 14: Build a notebook
- [ ] Pick one: mixed precision, positional encodings, or LR schedules
- [ ] Create the notebook from scratch in the appropriate `notebooks/` directory
- [ ] Include: explanation, from-scratch implementation, visualization, key takeaways

## Phase 4: Scale Up (Days 15–18)

### Day 15: DDP
- [ ] Launch 4-GPU instance
- [ ] Performance Exercise 6: Calculate DDP vs FSDP memory analytically
- [ ] Train medium.yaml with DDP: `scripts/launch_distributed.py --nproc 4 --mode ddp`
- [ ] Note step time and per-GPU memory
- [ ] **Stop the instance**

### Day 16: FSDP
- [ ] Train medium.yaml with FSDP: `scripts/launch_distributed.py --nproc 4 --mode fsdp`
- [ ] Compare step time and memory with DDP from Day 15
- [ ] Was DDP or FSDP faster? Did the memory match your predictions?
- [ ] **Stop the instance**

### Day 17: Full optimization workflow
- [ ] Performance Exercise 10: Profile-guided optimization on small.yaml
- [ ] Apply optimizations one at a time, build the optimization log
- [ ] Target: achieve >40% MFU
- [ ] **Stop the instance**

### Day 18: Review and next steps
- [ ] Re-read your daily log — what concepts clicked? What's still fuzzy?
- [ ] Re-run the tests: `uv run pytest -v`
- [ ] Pick what to explore next: build more notebooks, add SFT, write a Triton kernel

---

## Cloud Cost Tracker

| Date | Provider | Instance | Hours | Est. Cost |
|------|----------|----------|-------|-----------|
| | | | | |

**Running total:** $0

---

## Key Numbers I've Measured

Fill these in as you go — having your own measurements beats memorizing someone else's.

| Metric | tiny | small | medium |
|--------|------|-------|--------|
| Param count | | | |
| Weight memory (FP32) | | | |
| Weight memory (BF16) | | | |
| Training tokens/sec (CPU) | | | |
| Training tokens/sec (A100) | | | |
| MFU (eager FP32) | | | |
| MFU (compile+BF16) | | | |
| Inference tok/s (decode, batch=1) | | | |
| Inference tok/s (prefill, seq=512) | | | |
| INT8 compression ratio | | | |
| KV cache memory (seq=2048, BF16) | | | |
