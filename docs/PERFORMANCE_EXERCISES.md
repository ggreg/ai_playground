# Performance Debugging Exercises

Hands-on exercises for debugging model and ML infrastructure performance. Each exercise presents a scenario, asks you to diagnose the issue, and provides hints, the right tool to use, and a full solution.

Work through these on a GPU instance (see [CLOUD_SETUP.md](CLOUD_SETUP.md)). Most exercises also work on CPU/MPS with the tiny config, though the numbers will be less dramatic.

---

## Exercise 1: Where Did My Memory Go?

**Scenario:** You're training `small.yaml` on a 24 GB GPU. At batch_size=16, you get an OOM error. You try batch_size=4 and it works, but GPU utilization is only 30%.

**Task:** Figure out exactly where the memory goes. Break it down into model weights, optimizer states, activations, and gradients.

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.memory import MemoryTracker

config = TransformerConfig.SMALL
model = Transformer(config).cuda()

tracker = MemoryTracker()
tracker.snapshot("after_model_load")

x = torch.randint(0, config.vocab_size, (16, 2048)).cuda()
tracker.snapshot("after_input")

output = model(x)
tracker.snapshot("after_forward")

loss = output[:, :-1].sum()
loss.backward()
tracker.snapshot("after_backward")

tracker.report()
```

**Questions:**
1. How much memory do the model weights use? (Hint: `config.num_params() * 4` bytes for FP32)
2. How much do activations add during the forward pass?
3. Why does the backward pass allocate even more memory?
4. What's the theoretical minimum batch size for this GPU?

<details>
<summary>Hints</summary>

- Model weights (~125M params * 4 bytes) = ~500 MB
- Optimizer states (AdamW stores 2 extra copies: momentum + variance) = ~1000 MB
- Gradients = another ~500 MB
- That's ~2 GB before a single activation is stored
- Activations scale with `batch_size * seq_len * dim * n_layers` — this is usually the biggest chunk
- Use `tracker.delta("after_model_load", "after_forward")` to isolate activation memory

</details>

<details>
<summary>Tool to use</summary>

**`ai_playground.profiling.memory.MemoryTracker`** — take snapshots at key points (after model load, after forward, after backward, after optimizer step) and use `.delta()` to isolate each component. Also use **`torch.cuda.memory_summary()`** for the allocator-level breakdown showing block sizes, fragmentation, and the split between allocated vs reserved memory.

For a quick check without writing code: `nvidia-smi` in another terminal shows total GPU memory used, but can't distinguish weights from activations.

</details>

<details>
<summary>Solution</summary>

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.memory import MemoryTracker

config = TransformerConfig.SMALL
print(f"Model params: {config.num_params():,}")
print(f"Theoretical weight memory (FP32): {config.num_params() * 4 / 1024**2:.1f} MB")

model = Transformer(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

tracker = MemoryTracker()
tracker.snapshot("01_model_loaded")

# Optimizer init doesn't allocate states until first step, but let's check
optimizer.zero_grad()
tracker.snapshot("02_optimizer_init")

x = torch.randint(0, config.vocab_size, (16, 2048)).cuda()
tracker.snapshot("03_input_created")

output = model(x)
tracker.snapshot("04_after_forward")

loss = output[:, :-1].sum()
loss.backward()
tracker.snapshot("05_after_backward")

optimizer.step()
tracker.snapshot("06_after_optimizer_step")

tracker.report()

# Isolate each component
print(f"\nMemory breakdown:")
print(f"  Model weights:    {tracker.delta('01_model_loaded', '02_optimizer_init'):.1f} MB")
print(f"  Input tensors:    {tracker.delta('02_optimizer_init', '03_input_created'):.1f} MB")
print(f"  Activations:     {tracker.delta('03_input_created', '04_after_forward'):.1f} MB")
print(f"  Gradients:       {tracker.delta('04_after_forward', '05_after_backward'):.1f} MB")
print(f"  Optimizer states: {tracker.delta('05_after_backward', '06_after_optimizer_step'):.1f} MB")

# Full allocator breakdown
print(torch.cuda.memory_summary())
```

**Expected results on A100 (small.yaml, batch_size=16, seq_len=2048):**

| Component | Memory | Why |
|-----------|--------|-----|
| Weights | ~500 MB | 125M params * 4 bytes (FP32) |
| Activations | ~8-12 GB | Saved for backward: every layer stores input, attention scores, etc. Scales linearly with batch_size * seq_len |
| Gradients | ~500 MB | Same size as weights — one gradient per parameter |
| Optimizer states | ~1000 MB | AdamW: momentum (4 bytes/param) + variance (4 bytes/param) |
| **Total** | **~10-14 GB** | |

**Why batch_size=16 OOMs on 24 GB:** Activations alone can use 8-12 GB at this batch size. Combined with weights (500 MB), gradients (500 MB), optimizer (1 GB), and CUDA context (~500 MB), you exceed 24 GB.

**Fixes, in order of impact:**
1. **BF16 mixed precision** — halves activation and weight memory. Add `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
2. **Gradient checkpointing** — recomputes activations during backward instead of storing them. Trades ~30% extra compute for ~60% less activation memory. Use `torch.utils.checkpoint.checkpoint()` on each transformer block.
3. **Reduce sequence length** — activations scale linearly with seq_len. 1024 instead of 2048 roughly halves activation memory.
4. **Gradient accumulation** — use batch_size=4 with grad_accum_steps=4 for effective batch_size=16 at 4x less peak memory.

</details>

---

## Exercise 2: Why Is My GPU Idle?

**Scenario:** You're training and `nvidia-smi` shows GPU utilization bouncing between 0% and 90%. Training is 3x slower than expected.

**Task:** Identify the bottleneck. Profile a training step to find where time is spent.

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.nsight import profile_with_torch

config = TransformerConfig.SMALL
model = Transformer(config).cuda()

def make_batch():
    return (torch.randint(0, config.vocab_size, (8, 2048)).cuda(),)

prof = profile_with_torch(model, make_batch, num_steps=5, warmup_steps=2)
```

**Questions:**
1. Look at the trace — is the GPU stalled waiting for data?
2. Are there many small CUDA kernels instead of a few large ones?
3. Is CPU-GPU synchronization happening unexpectedly?

<details>
<summary>Hints</summary>

Common causes of GPU idle time:

- **Data loading on CPU:** The GPU finishes a step and waits for the next batch. Fix: use `DataLoader(num_workers=4, pin_memory=True, prefetch_factor=2)`.
- **CPU-GPU sync points:** Calling `.item()`, `print(loss)`, or `if loss < threshold` forces the GPU to finish before the CPU can continue. Fix: only sync at log intervals.
- **Small kernels:** Many tiny operations (e.g., per-element ops, reshape) have high launch overhead. Fix: use `torch.compile()` to fuse them.
- **Host-to-device copies:** Moving tensors to GPU inside the training loop instead of in the data loader. Fix: do `.cuda()` in the dataloader's `collate_fn` or use `pin_memory`.

</details>

<details>
<summary>Tool to use</summary>

**`ai_playground.profiling.nsight.profile_with_torch`** — generates a Chrome trace showing CPU and CUDA activity side by side. Open the trace in `chrome://tracing` or `perfetto.dev`. Look for:
- **Gaps between CUDA kernels** on the GPU timeline row — these are stalls
- **Long CPU bars** aligned with GPU gaps — these reveal the blocking operation
- **`cudaMemcpy`/`aten::to`** bars — these indicate data transfer overhead

For deeper kernel-level analysis, use **Nsight Systems** on the command line:
```bash
nsys profile -o trace uv run python scripts/train.py --config configs/tiny.yaml --max-steps 5
```
Then open `trace.nsys-rep` in Nsight Systems GUI to see the full timeline with CUDA API calls, kernel launches, and memory operations.

**`nvidia-smi dmon -s u -d 1`** — lightweight continuous monitoring of GPU utilization at 1-second intervals.

</details>

<details>
<summary>Solution</summary>

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.nsight import profile_with_torch, nsight_range

config = TransformerConfig.SMALL
model = Transformer(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Profile a full training step with labeled regions
def profiled_step():
    with nsight_range("data_loading"):
        x = torch.randint(0, config.vocab_size, (8, 2048)).cuda()

    with nsight_range("forward"):
        output = model(x)

    with nsight_range("loss"):
        loss = output.sum()

    with nsight_range("backward"):
        loss.backward()

    with nsight_range("optimizer"):
        optimizer.step()
        optimizer.zero_grad()

    # THIS IS THE TRAP: .item() forces a CPU-GPU sync
    with nsight_range("logging_sync"):
        loss_val = loss.item()  # <-- blocks until GPU finishes
        print(f"Loss: {loss_val:.4f}")

    return (x,)

prof = profile_with_torch(model, profiled_step, num_steps=5, warmup_steps=2)
```

**How to read the trace:**

1. Open the output in `chrome://tracing` or `perfetto.dev`
2. Look at the GPU row — it should have continuous kernel execution
3. Common patterns you'll see:

```
GPU:  [======forward======]  [===backward===]  [gap]  [======forward======]
CPU:  [forward launch]  [backward launch]  [.item() BLOCKS HERE]  [next step]
```

**The `.item()` trap:** Every call to `.item()`, `print(tensor)`, or Python-side conditionals on tensor values forces `cudaDeviceSynchronize()`. The CPU blocks until the GPU finishes all queued work, then the GPU sits idle while the CPU prepares the next operation.

**Fix:**
```python
# BAD: syncs every step
for step in range(1000):
    loss = train_step()
    print(f"Step {step}: {loss.item():.4f}")  # sync every step!

# GOOD: sync only at log intervals
for step in range(1000):
    loss = train_step()
    if step % 10 == 0:
        print(f"Step {step}: {loss.item():.4f}")  # sync every 10 steps
```

**Other fixes:**
- Move data to GPU asynchronously: `x = x.to(device, non_blocking=True)` with `pin_memory=True`
- Use `torch.compile(model)` to fuse small kernels into larger ones
- Overlap data loading with compute using `DataLoader(num_workers=4, prefetch_factor=2)`

</details>

---

## Exercise 3: MFU — How Hard Is Your GPU Working?

**Scenario:** Training runs, the GPU shows 95% utilization, but it still feels slow. You want to know if you're actually using the GPU's compute capacity efficiently.

**Task:** Compute MFU for your training run and identify what's leaving performance on the table.

```python
import time
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.flops import compute_mfu

config = TransformerConfig.SMALL
model = Transformer(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x = torch.randint(0, config.vocab_size, (16, 2048)).cuda()

# Warmup
for _ in range(3):
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

# Measure
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    output = model(x)
    output.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
step_time = (time.perf_counter() - t0) / 10

mfu = compute_mfu(config, seq_len=2048, batch_size=16,
                   step_time_sec=step_time, gpu_name="A100_80GB")
print(f"MFU: {mfu['mfu_percent']:.1f}%")
print(f"Achieved: {mfu['achieved_tflops_per_sec']:.1f} TFLOPS")
print(f"Peak: {mfu['peak_tflops']} TFLOPS")
```

**Questions:**
1. What MFU do you get? (Naive PyTorch is typically 20-30%)
2. Try adding `torch.compile(model)` — how much does MFU improve?
3. Try BF16 with `torch.amp.autocast("cuda", dtype=torch.bfloat16)` — what changes?
4. What's the theoretical maximum MFU and why can't you reach 100%?

<details>
<summary>Hints</summary>

- **GPU utilization != MFU.** The GPU can be 100% utilized doing memory transfers. MFU measures how much actual math gets done vs the chip's peak math throughput.
- `torch.compile` fuses small operations and reduces kernel launch overhead — expect 1.3-1.8x speedup (MFU from ~25% to ~40%).
- BF16 halves memory bandwidth pressure and doubles tensor core throughput — often another 1.5-2x.
- You can never reach 100% because: memory reads/writes take time, not all ops use tensor cores, kernel launches have overhead, and some operations are inherently sequential.
- Target 40-50% for a well-optimized single-GPU setup. 50-60% requires expert-level fused kernels (Flash Attention, Triton).

</details>

<details>
<summary>Tool to use</summary>

**`ai_playground.profiling.flops.compute_mfu`** — takes your config, batch size, sequence length, and measured step time, and returns the MFU percentage. This is the primary tool.

**`ai_playground.profiling.flops.estimate_flops`** — breaks down FLOPs by component (attention vs FFN) so you can see where compute goes.

**`time.perf_counter()` with `torch.cuda.synchronize()`** — critical for accurate GPU timing. Without `synchronize()`, you're only timing kernel launch (microseconds), not execution (milliseconds).

For deeper analysis, use **`torch.profiler`** (via `profile_with_torch`) to see which CUDA kernels are using tensor cores vs scalar cores. Tensor core kernels have names containing `gemm` or `mma`; scalar kernels (like softmax, dropout, normalization) can't achieve peak FLOPS.

</details>

<details>
<summary>Solution</summary>

```python
import time
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.flops import compute_mfu, estimate_flops

config = TransformerConfig.SMALL

def measure_mfu(model, label, use_amp=False, dtype=torch.bfloat16):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, config.vocab_size, (16, 2048)).cuda()

    # Warmup (important: torch.compile compiles during warmup)
    for _ in range(5):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 10
    for _ in range(steps):
        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            output = model(x)
            output.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    step_time = (time.perf_counter() - t0) / steps

    mfu = compute_mfu(config, seq_len=2048, batch_size=16,
                       step_time_sec=step_time, gpu_name="A100_80GB")
    print(f"{label:>30}: MFU={mfu['mfu_percent']:.1f}%, "
          f"{mfu['achieved_tflops_per_sec']:.1f}/{mfu['peak_tflops']} TFLOPS, "
          f"{step_time*1000:.0f} ms/step")
    return mfu

# 1. Baseline: eager FP32
model = Transformer(config).cuda()
measure_mfu(model, "Eager FP32")

# 2. BF16 mixed precision
model = Transformer(config).cuda()
measure_mfu(model, "Eager BF16", use_amp=True)

# 3. torch.compile + FP32
model = torch.compile(Transformer(config).cuda())
measure_mfu(model, "Compiled FP32")

# 4. torch.compile + BF16 (best single-GPU setup)
model = torch.compile(Transformer(config).cuda())
measure_mfu(model, "Compiled BF16", use_amp=True)

# Show where FLOPs go
flops = estimate_flops(config, seq_len=2048, batch_size=16)
print(f"\nFLOP breakdown per step:")
print(f"  Forward:   {flops['forward_tflops']:.3f} TFLOPS")
print(f"  Backward:  {flops['backward_tflops']:.3f} TFLOPS")
print(f"  Total:     {flops['total_tflops']:.3f} TFLOPS")
print(f"  Attention: {flops['attention_fraction']:.1%} of per-layer compute")
print(f"  FFN:       {flops['ffn_fraction']:.1%} of per-layer compute")
```

**Expected results on A100-80GB:**

| Configuration | MFU | Step Time | Speedup |
|--------------|-----|-----------|---------|
| Eager FP32 | ~22% | ~850 ms | 1.0x |
| Eager BF16 | ~30% | ~520 ms | 1.6x |
| Compiled FP32 | ~35% | ~480 ms | 1.8x |
| Compiled BF16 | ~45% | ~300 ms | 2.8x |

**Why you can't reach 100% MFU:**
1. **Memory bandwidth** — loading weights from HBM takes time that isn't compute. The roofline model sets the upper bound based on arithmetic intensity.
2. **Non-tensor-core ops** — softmax, RMSNorm, RoPE, dropout all run on scalar cores at much lower throughput.
3. **Kernel launch overhead** — each CUDA kernel launch has ~5-10 μs overhead. Hundreds of launches per step add up.
4. **Python overhead** — PyTorch dispatches ops from Python. `torch.compile` eliminates most of this.
5. **Memory allocation** — PyTorch's caching allocator occasionally does cudaMalloc, which stalls.

**To push beyond 45-50%:**
- Flash Attention (fuses the entire attention computation, eliminates O(n^2) memory reads)
- Triton fused kernels for RMSNorm + residual + dropout
- CUDA graphs to eliminate kernel launch overhead

</details>

---

## Exercise 4: The Quantization Quality–Speed Tradeoff

**Scenario:** You need to serve the model and want to reduce memory and latency using INT8 quantization. But how much quality do you lose?

**Task:** Quantize the model, measure the speedup, and check if outputs diverge.

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.inference.quantize import quantize_model_weights
from ai_playground.inference.benchmark import benchmark_generation, print_benchmark

config = TransformerConfig.SMALL
model = Transformer(config).cuda().eval()

# Baseline benchmark
results_fp32 = benchmark_generation(model, prompt_len=128, gen_len=256)
print_benchmark(results_fp32)

# Quantize and check compression
stats = quantize_model_weights(model)
print(f"Original:    {stats['original_mb']:.1f} MB")
print(f"Quantized:   {stats['quantized_mb']:.1f} MB")
print(f"Compression: {stats['compression_ratio']:.2f}x")

# Compare outputs before and after quantization
prompt = torch.randint(0, config.vocab_size, (1, 64)).cuda()
with torch.no_grad():
    logits_fp32 = model(prompt)
    # TODO: load quantized weights back and compare logits
```

**Questions:**
1. What compression ratio do you get with INT8 absmax?
2. How does the max absolute error in logits compare to the mean?
3. Is absmax quantization per-tensor or per-channel? Why does that matter?
4. At what point does quantization error actually affect generation quality?

<details>
<summary>Hints</summary>

- INT8 absmax on FP32 weights gives ~4x compression (4 bytes → 1 byte per weight).
- Per-tensor quantization (what `quantize.py` implements) uses one scale for the entire tensor. Outlier values waste dynamic range for the rest. Per-channel quantization uses one scale per output channel — much better quality.
- Check for outliers: `weight.abs().max() / weight.abs().mean()`. Ratios above 10-20 indicate outliers that hurt per-tensor quantization.
- In practice, absmax INT8 works well for weights >100M params. For smaller models, the quantization noise is proportionally larger.

</details>

<details>
<summary>Tool to use</summary>

**`ai_playground.inference.quantize.quantize_tensor_absmax`** and **`dequantize_tensor`** — quantize individual tensors and measure round-trip error. Use these to inspect specific layers.

**`ai_playground.inference.quantize.quantize_model_weights`** — quantize all linear layers and get compression stats.

**`ai_playground.inference.benchmark.benchmark_generation`** — measure end-to-end inference speed before and after quantization.

For analyzing outliers, use plain **PyTorch tensor operations** — `.abs().max()`, `.abs().mean()`, histograms with `torch.histc()`.

</details>

<details>
<summary>Solution</summary>

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.inference.quantize import (
    quantize_tensor_absmax, dequantize_tensor, quantize_model_weights
)

config = TransformerConfig.SMALL
model = Transformer(config).cuda().eval()

# Step 1: Understand per-tensor quantization error
print("=== Per-layer quantization analysis ===\n")
for name, param in model.named_parameters():
    if param.dim() < 2:
        continue  # skip norms

    q, scale = quantize_tensor_absmax(param.data, bits=8)
    reconstructed = dequantize_tensor(q, scale)
    error = (param.data - reconstructed).abs()

    outlier_ratio = param.data.abs().max().item() / param.data.abs().mean().item()

    print(f"{name:>50}: "
          f"max_err={error.max().item():.6f}, "
          f"mean_err={error.mean().item():.6f}, "
          f"outlier_ratio={outlier_ratio:.1f}")

# Step 2: Compare model outputs before and after quantization
prompt = torch.randint(0, config.vocab_size, (1, 128)).cuda()

with torch.no_grad():
    logits_original = model(prompt).clone()

# Quantize and dequantize all weights (simulating INT8 inference)
for name, param in model.named_parameters():
    if param.dim() >= 2:
        q, scale = quantize_tensor_absmax(param.data, bits=8)
        param.data = dequantize_tensor(q, scale)

with torch.no_grad():
    logits_quantized = model(prompt)

# Compare logits
logit_error = (logits_original - logits_quantized).abs()
print(f"\n=== Logit comparison ===")
print(f"Max absolute error:  {logit_error.max().item():.4f}")
print(f"Mean absolute error: {logit_error.mean().item():.4f}")
print(f"Logit range:         [{logits_original.min().item():.2f}, {logits_original.max().item():.2f}]")

# Do the top-1 predictions change?
top1_original = logits_original.argmax(dim=-1)
top1_quantized = logits_quantized.argmax(dim=-1)
agreement = (top1_original == top1_quantized).float().mean()
print(f"Top-1 agreement:     {agreement.item():.1%}")

# Step 3: Check compression
stats = quantize_model_weights(model)
print(f"\n=== Compression stats ===")
print(f"Original:    {stats['original_mb']:.1f} MB")
print(f"Quantized:   {stats['quantized_mb']:.1f} MB")
print(f"Compression: {stats['compression_ratio']:.2f}x")
```

**Expected results:**

- **Compression:** ~3.8x (close to theoretical 4x; the scale factor per tensor adds a tiny overhead)
- **Outlier ratios:** Layers with ratios >20 will have worse quantization quality. Attention output projections and the first/last layers often have the worst outliers.
- **Top-1 agreement:** Typically 90-98% for a 125M model. Errors accumulate across layers, so deeper models show more divergence.
- **Max logit error:** Can be 10-100x larger than mean error due to outliers. This is the fundamental weakness of per-tensor quantization.

**Why per-channel is better:**
```python
# Per-tensor: one scale for entire [out, in] matrix
# If one output channel has a value of 100 and others are <1,
# the scale is set by the 100, wasting 7 bits of range for all other channels.

# Per-channel: one scale per output row
# Each channel uses its full INT8 range independently.
# Outliers in one channel don't affect others.
```

**When quality degrades noticeably:**
- Models <50M params: too few parameters to absorb the quantization noise
- Per-tensor on models with activation outliers (common in LLMs at >1B scale)
- When cascading: small errors in early layers amplify through the network

</details>

---

## Exercise 5: Attention Is the Bottleneck (or Is It?)

**Scenario:** Someone tells you "attention is O(n^2) so it's always the bottleneck." You want to verify this claim.

**Task:** Profile a forward pass and find what actually dominates compute time at different sequence lengths.

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.flops import estimate_flops

config = TransformerConfig.SMALL

for seq_len in [128, 512, 2048, 8192]:
    flops = estimate_flops(config, seq_len=seq_len, batch_size=1)
    print(f"seq_len={seq_len:>5}: "
          f"attention={flops['attention_fraction']:.1%}, "
          f"ffn={flops['ffn_fraction']:.1%}")
```

**Questions:**
1. At what sequence length does attention surpass FFN in compute cost?
2. At `seq_len=512`, which dominates? Does this surprise you?
3. Where does the KV cache memory formula `2 * n_layers * seq_len * n_kv_heads * head_dim * bytes_per_element` come from?
4. How does GQA change the crossover point vs MHA?

<details>
<summary>Hints</summary>

- Attention has two components: **projections** (QKV and output, which are matrix multiplications scaling with `seq_len * dim^2`) and **score computation** (which scales with `seq_len^2 * dim`).
- At short sequences, the QKV projections dominate because `dim^2` >> `seq_len`. The O(n^2) part only dominates when `seq_len` > `dim` (e.g., above 1024 for dim=768).
- FFN (SwiGLU) does 3 matrix multiplications per layer with hidden_dim ≈ 2.67 * dim. This is often 2/3 of total FLOPs at typical sequence lengths.

</details>

<details>
<summary>Tool to use</summary>

**`ai_playground.profiling.flops.estimate_flops`** — breaks down FLOPs into attention fraction vs FFN fraction for any config and sequence length. This is the analytical tool.

For empirical verification, use **`ai_playground.profiling.nsight.profile_with_torch`** — profile a forward pass and look at kernel-level timing. Sort by `cuda_time_total` to see which operations actually take the most wall-clock time. Attention kernels will have names containing `bmm` or `sdpa`; FFN kernels will show as large `mm` (matrix multiply) calls.

Also use **`torch.utils.flop_counter.FlopCounterMode`** (PyTorch 2.1+) for automatic per-operator FLOP counting to cross-check the analytical estimates.

</details>

<details>
<summary>Solution</summary>

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.flops import estimate_flops

# Part 1: Analytical FLOP breakdown across sequence lengths
config = TransformerConfig.SMALL
print(f"Model: dim={config.dim}, layers={config.n_layers}, "
      f"heads={config.n_heads}, kv_heads={config.kv_heads}\n")

print(f"{'seq_len':>8} | {'Attention':>10} | {'FFN':>10} | {'Dominant':>10}")
print("-" * 50)
for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    flops = estimate_flops(config, seq_len=seq_len, batch_size=1)
    dominant = "Attention" if flops['attention_fraction'] > 0.5 else "FFN"
    print(f"{seq_len:>8} | {flops['attention_fraction']:>9.1%} | "
          f"{flops['ffn_fraction']:>9.1%} | {dominant:>10}")

# Part 2: Compare MHA vs GQA vs MQA
print(f"\n--- Crossover comparison at seq_len where attention > FFN ---\n")
configs = {
    "MHA (12 KV heads)": TransformerConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=12),
    "GQA (4 KV heads)":  TransformerConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4),
    "MQA (1 KV head)":   TransformerConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=1),
}

for name, cfg in configs.items():
    # Find crossover point
    for seq_len in range(128, 32768, 128):
        flops = estimate_flops(cfg, seq_len=seq_len, batch_size=1)
        if flops['attention_fraction'] > 0.5:
            print(f"{name:>25}: attention dominates at seq_len >= {seq_len}")
            break

# Part 3: KV cache memory breakdown
print(f"\n--- KV cache memory ---\n")
print("Formula: 2 * n_layers * seq_len * n_kv_heads * head_dim * bytes")
print("  2       = one K tensor + one V tensor per layer")
print("  n_layers = each layer has its own KV cache")
print("  seq_len  = one KV entry per token in the sequence")
print("  n_kv_heads * head_dim = size of each K or V vector")
print()

for seq_len in [1024, 2048, 8192, 32768]:
    bytes_per = 2  # BF16
    kv_bytes = 2 * config.n_layers * seq_len * config.kv_heads * config.head_dim * bytes_per
    kv_mb = kv_bytes / 1024**2
    # Compare with MHA
    kv_mha = 2 * config.n_layers * seq_len * config.n_heads * config.head_dim * bytes_per / 1024**2
    print(f"seq_len={seq_len:>6}: GQA(4 heads)={kv_mb:.1f} MB, "
          f"MHA(12 heads)={kv_mha:.1f} MB, "
          f"savings={kv_mha/kv_mb:.1f}x")
```

**Expected output (small.yaml, dim=768, 12 heads, 4 KV heads):**

```
 seq_len |  Attention |        FFN |   Dominant
--------------------------------------------------
     128 |      20.5% |      79.5% |        FFN
     512 |      25.8% |      74.2% |        FFN
    2048 |      39.1% |      60.9% |        FFN
    4096 |      52.3% |      47.7% |  Attention
    8192 |      67.8% |      32.2% |  Attention
```

**Key insight:** At typical training sequence lengths (512-2048), FFN dominates — attention is NOT the bottleneck. The O(n^2) claim only matters for long-context inference (>4K tokens for this model size). This is why Flash Attention gives modest gains for short sequences but transformative gains for 32K+ contexts.

**GQA vs MHA:** GQA reduces KV projection FLOPs (fewer KV heads = smaller K, V matrices) but the attention score computation (`Q @ K^T`, which is the O(n^2) part) stays the same size because it depends on the number of Q heads, not KV heads. GQA's main benefit is **memory savings** in the KV cache, not compute savings.

</details>

---

## Exercise 6: DDP vs FSDP — When to Shard

**Scenario:** You have 4 GPUs and want to train `medium.yaml` (~350M params). Should you use DDP or FSDP?

**Task:** Calculate memory requirements for each strategy and verify empirically.

**Questions:**
1. In DDP, each GPU holds a full copy of: model weights + optimizer states + gradients. How much memory is that per GPU for the medium model in FP32? In BF16 with FP32 optimizer states?
2. In FSDP (ZeRO Stage 3), weights, optimizer states, and gradients are all sharded. What's the per-GPU memory now?
3. What's the communication overhead of each? (DDP: all-reduce gradients once per step. FSDP: all-gather weights before each layer's forward/backward, reduce-scatter gradients.)
4. At what model size does FSDP become necessary vs just convenient?

```bash
# Try both and compare step times
uv run python scripts/launch_distributed.py --nproc 4 --mode ddp --config configs/medium.yaml --max-steps 20
uv run python scripts/launch_distributed.py --nproc 4 --mode fsdp --config configs/medium.yaml --max-steps 20
```

<details>
<summary>Hints</summary>

**DDP memory per GPU (medium, FP32):**
- Weights: 350M * 4 bytes = 1.4 GB
- Gradients: 1.4 GB
- Optimizer (AdamW, 2 states): 2.8 GB
- Total fixed: **5.6 GB** per GPU (identical on all GPUs)
- Plus activations (scales with batch size)

**FSDP memory per GPU (4 GPUs, FP32):**
- Weights sharded: 1.4 / 4 = 0.35 GB
- Gradients sharded: 0.35 GB
- Optimizer sharded: 0.7 GB
- Total fixed: **1.4 GB** per GPU
- But: during forward/backward, one layer's full weights are gathered temporarily

</details>

<details>
<summary>Tool to use</summary>

**`ai_playground.profiling.memory.MemoryTracker`** — run on each rank to compare per-GPU memory between DDP and FSDP. Use snapshots after model init and after the first training step to capture optimizer state allocation.

**`scripts/launch_distributed.py`** — run both DDP and FSDP modes and compare reported step times.

**`ai_playground.models.config.TransformerConfig.num_params()`** — calculate parameter count to derive memory estimates analytically before measuring.

For communication profiling, use **`torch.profiler`** with `ProfilerActivity.CUDA` — look for NCCL kernels (`ncclAllReduceKernel`, `ncclAllGatherKernel`, `ncclReduceScatterKernel`). Their time relative to compute kernels tells you whether communication is the bottleneck.

**`nvidia-smi`** on all GPUs simultaneously (or `nvidia-smi dmon`) to verify that memory usage is uniform (DDP) or reduced (FSDP).

</details>

<details>
<summary>Solution</summary>

```python
# Run this calculation locally, then verify on a multi-GPU instance
from ai_playground.models.config import TransformerConfig

config = TransformerConfig.MEDIUM
params = config.num_params()
n_gpus = 4

print(f"Model: {params:,} parameters\n")

# Memory calculation helper
def memory_breakdown(params, n_gpus, strategy, weight_bytes=4, optim_bytes=8):
    """
    weight_bytes: 4 for FP32, 2 for BF16
    optim_bytes: 8 for AdamW FP32 (momentum + variance, 4 bytes each)
    """
    weights = params * weight_bytes
    grads = params * weight_bytes  # gradients match weight precision
    optim = params * optim_bytes

    if strategy == "ddp":
        # Full replica on every GPU
        per_gpu_fixed = weights + grads + optim
    elif strategy == "fsdp":
        # Everything sharded across GPUs
        per_gpu_fixed = (weights + grads + optim) / n_gpus
        # But one layer's weights are gathered during forward/backward
        layer_params = params / config.n_layers
        per_gpu_fixed += layer_params * weight_bytes  # temporary gather buffer

    mb = per_gpu_fixed / 1024**2
    return mb

print(f"{'Strategy':<20} {'FP32 per GPU':>15} {'BF16+FP32opt per GPU':>22}")
print("-" * 60)

for strategy in ["ddp", "fsdp"]:
    fp32 = memory_breakdown(params, n_gpus, strategy, weight_bytes=4, optim_bytes=8)
    mixed = memory_breakdown(params, n_gpus, strategy, weight_bytes=2, optim_bytes=8)
    print(f"{strategy.upper():<20} {fp32:>12.0f} MB {mixed:>19.0f} MB")

print(f"\n--- Communication analysis ---\n")
grad_size_mb = params * 4 / 1024**2  # FP32 gradients

print(f"DDP all-reduce per step:  {grad_size_mb:.0f} MB")
print(f"  = 2x gradient size (reduce + broadcast)")
print(f"  = happens once, overlapped with backward via bucketing\n")

fsdp_per_layer_mb = params / config.n_layers * 4 / 1024**2
print(f"FSDP per-layer all-gather: {fsdp_per_layer_mb:.0f} MB")
print(f"  = full weights gathered before each layer's forward AND backward")
print(f"  = {config.n_layers} layers * 2 (fwd+bwd) = {config.n_layers * 2} all-gathers per step")
print(f"  + reduce-scatter for gradients: {config.n_layers} per step")
print(f"  Total FSDP comms: {config.n_layers * 3} collective ops per step vs 1 for DDP")

# When does FSDP become necessary?
print(f"\n--- When to use FSDP ---\n")
for gpu_vram_gb in [24, 40, 80]:
    # Available for model state (reserve ~30% for activations, CUDA context)
    available_gb = gpu_vram_gb * 0.7
    # DDP: need 16 bytes/param (4 weights + 4 grads + 8 optimizer)
    max_ddp_params = available_gb * 1024**3 / 16
    # With BF16: 2 weights + 2 grads + 8 optimizer = 12 bytes/param
    max_ddp_bf16 = available_gb * 1024**3 / 12
    print(f"{gpu_vram_gb}GB GPU: DDP handles up to {max_ddp_params/1e9:.1f}B params (FP32), "
          f"{max_ddp_bf16/1e9:.1f}B params (BF16)")
```

**Expected output:**

```
Strategy             FP32 per GPU   BF16+FP32opt per GPU
------------------------------------------------------------
DDP                      5376 MB              4032 MB
FSDP                     1400 MB              1050 MB

24GB GPU: DDP handles up to 1.1B params (FP32), 1.4B params (BF16)
40GB GPU: DDP handles up to 1.8B params (FP32), 2.3B params (BF16)
80GB GPU: DDP handles up to 3.7B params (FP32), 4.7B params (BF16)
```

**Decision framework:**
- **<1B params on 40GB+ GPUs:** Use DDP. Simpler, faster (less communication).
- **1-3B params:** Use DDP with BF16 on 80GB GPUs. Use FSDP on 40GB GPUs.
- **>3B params:** FSDP required regardless of GPU size.
- **Measure both:** If DDP fits in memory, it's almost always faster for this model size due to less communication overhead.

</details>

---

## Exercise 7: Decode Is Memory-Bound, Not Compute-Bound

**Scenario:** During autoregressive generation, your GPU utilization drops to 5% even though you're processing tokens as fast as possible.

**Task:** Understand why decode is fundamentally different from prefill and what determines decode speed.

```python
import torch
from ai_playground.inference.benchmark import benchmark_generation, print_benchmark
from ai_playground.models import Transformer, TransformerConfig

config = TransformerConfig.SMALL
model = Transformer(config).cuda().eval()

# Compare prefill vs decode rates
results = benchmark_generation(model, prompt_len=512, gen_len=512,
                                batch_size=1, dtype=torch.bfloat16)
print_benchmark(results)
print(f"Prefill/Decode ratio: {results['prefill_tokens_per_sec'] / results['decode_tokens_per_sec']:.1f}x")
```

**Questions:**
1. Why is prefill so much faster per token than decode?
2. What's the arithmetic intensity (FLOPs per byte loaded) of a decode step vs a prefill step?
3. How does batching multiple requests help decode throughput?
4. Why do serving systems like vLLM use continuous batching?

<details>
<summary>Hints</summary>

- **Prefill** processes all prompt tokens in one big matrix multiplication (e.g., `[batch, seq_len, dim] @ [dim, dim]`). This is compute-bound — the GPU's tensor cores are fully utilized.
- **Decode** processes one token at a time (`[batch, 1, dim] @ [dim, dim]`). The matrix multiply is tiny, but you still need to load the entire weight matrix from GPU memory. This is memory-bandwidth-bound.
- Arithmetic intensity: prefill with seq_len=512 does 512x more FLOPs per weight byte loaded. Decode does ~2 FLOPs per byte (one multiply, one add) — far below the GPU's compute-to-bandwidth ratio.
- **Batching helps** because you load weights once and multiply against multiple sequences. Going from batch=1 to batch=32 is nearly free in latency but 32x the throughput.

</details>

<details>
<summary>Tool to use</summary>

**`ai_playground.inference.benchmark.benchmark_generation`** — measures prefill and decode separately, reporting tokens/sec and latency for each phase. This is the primary diagnostic tool.

For understanding **why** decode is slow, compute the **roofline model** analytically:
- **Arithmetic intensity** = FLOPs / bytes loaded from memory
- Compare to GPU's **ops:byte ratio** (A100: 312 TFLOPS / 2 TB/s = 156 FLOP/byte)
- If your operation's arithmetic intensity < 156, it's memory-bound on A100

Use **`ai_playground.profiling.flops.estimate_flops`** with `batch_size=1, seq_len=1` for decode vs `seq_len=512` for prefill to see the compute difference analytically.

To verify empirically, vary batch size and measure decode tokens/sec — if throughput scales linearly with batch size, you're memory-bound (loading weights once, doing more compute per load).

</details>

<details>
<summary>Solution</summary>

```python
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.inference.benchmark import benchmark_generation, print_benchmark
from ai_playground.profiling.flops import estimate_flops

config = TransformerConfig.SMALL
model = Transformer(config).cuda().eval()

# Part 1: Measure prefill vs decode at different batch sizes
print("=== Prefill vs Decode ===\n")
for batch_size in [1, 4, 16, 32]:
    try:
        results = benchmark_generation(
            model, prompt_len=512, gen_len=128,
            batch_size=batch_size, dtype=torch.bfloat16
        )
        ratio = results['prefill_tokens_per_sec'] / results['decode_tokens_per_sec']
        print(f"batch={batch_size:>2}: prefill={results['prefill_tokens_per_sec']:>8.0f} tok/s, "
              f"decode={results['decode_tokens_per_sec']:>8.0f} tok/s, "
              f"ratio={ratio:.0f}x")
    except torch.cuda.OutOfMemoryError:
        print(f"batch={batch_size:>2}: OOM")
        break

# Part 2: Roofline analysis
print(f"\n=== Roofline Analysis ===\n")
params = config.num_params()
weight_bytes_bf16 = params * 2  # BF16

# A100 specs
a100_bandwidth_bytes = 2e12  # 2 TB/s
a100_peak_tflops = 312e12   # 312 TFLOPS BF16

ops_byte_ratio = a100_peak_tflops / a100_bandwidth_bytes
print(f"A100 ops:byte ratio: {ops_byte_ratio:.0f} FLOP/byte")
print(f"(Operations below this ratio are memory-bound)\n")

# Decode: [1, 1, 768] @ [768, 768] per linear layer
decode_flops = estimate_flops(config, seq_len=1, batch_size=1)
decode_intensity = decode_flops['forward_tflops'] * 1e12 / weight_bytes_bf16
print(f"Decode (batch=1):  {decode_intensity:.1f} FLOP/byte  → {'MEMORY-BOUND' if decode_intensity < ops_byte_ratio else 'COMPUTE-BOUND'}")

# Prefill: [1, 512, 768] @ [768, 768] per linear layer
prefill_flops = estimate_flops(config, seq_len=512, batch_size=1)
prefill_intensity = prefill_flops['forward_tflops'] * 1e12 / weight_bytes_bf16
print(f"Prefill (seq=512): {prefill_intensity:.1f} FLOP/byte → {'MEMORY-BOUND' if prefill_intensity < ops_byte_ratio else 'COMPUTE-BOUND'}")

# Batched decode
for batch_size in [1, 8, 32, 128]:
    decode_flops = estimate_flops(config, seq_len=1, batch_size=batch_size)
    intensity = decode_flops['forward_tflops'] * 1e12 / weight_bytes_bf16
    bound = 'MEMORY' if intensity < ops_byte_ratio else 'COMPUTE'
    print(f"Decode (batch={batch_size:>3}): {intensity:>6.1f} FLOP/byte → {bound}-BOUND")

# Part 3: Theoretical minimum decode latency
min_latency_ms = weight_bytes_bf16 / a100_bandwidth_bytes * 1000
print(f"\nTheoretical minimum per-token latency (A100): {min_latency_ms:.2f} ms")
print(f"  = time to load all {params/1e6:.0f}M params from HBM")
print(f"  = {1000/min_latency_ms:.0f} tokens/sec max (batch=1)")
```

**Expected results (small.yaml on A100):**

```
A100 ops:byte ratio: 156 FLOP/byte

Decode (batch=1):    2.0 FLOP/byte  → MEMORY-BOUND    (78x below roofline!)
Prefill (seq=512): 512.0 FLOP/byte  → COMPUTE-BOUND
Decode (batch=128): 256.0 FLOP/byte → COMPUTE-BOUND

Theoretical minimum per-token latency: 0.12 ms = 8,000 tokens/sec max (batch=1)
```

**Key takeaways:**
1. **Decode at batch=1 wastes 98.7% of the GPU's compute.** You load the entire model just to compute one vector-matrix multiply.
2. **Batching is the primary fix.** At batch=128, you're doing 128x more FLOPs per weight byte loaded, crossing into compute-bound territory.
3. **Continuous batching** (vLLM, TGI) keeps the batch full by injecting new requests as old ones finish. Static batching wastes GPU time when sequences finish at different times.
4. **Speculative decoding** is another fix: use a small draft model to predict N tokens, then verify all N in one batched forward pass of the large model.

</details>

---

## Exercise 8: The torch.compile Warmup Trap

**Scenario:** You add `torch.compile` and your first training step takes 45 seconds instead of 0.5 seconds. Your benchmark shows a slowdown.

**Task:** Understand compilation overhead and measure it correctly.

```python
import time
import torch
from ai_playground.models import Transformer, TransformerConfig

config = TransformerConfig.SMALL
model = Transformer(config).cuda()
x = torch.randint(0, config.vocab_size, (8, 2048)).cuda()

# Without compile
torch.cuda.synchronize()
t0 = time.perf_counter()
for i in range(5):
    model(x).sum().backward()
torch.cuda.synchronize()
eager_time = (time.perf_counter() - t0) / 5

# With compile
compiled = torch.compile(model)
times = []
for i in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled(x).sum().backward()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

print(f"Eager: {eager_time*1000:.1f} ms/step")
for i, t in enumerate(times):
    print(f"Compiled step {i}: {t*1000:.1f} ms {'(compiling)' if t > eager_time * 2 else ''}")
```

**Questions:**
1. How many steps does it take for `torch.compile` to finish compiling?
2. Does changing the input shape trigger recompilation?
3. What's the steady-state speedup after warmup?
4. When is `torch.compile` not worth it? (Short training runs? Small models?)

<details>
<summary>Hints</summary>

- `torch.compile` traces the model on the first call, then compiles an optimized version. This takes 10-60 seconds depending on model size. Subsequent calls with the same shapes are fast.
- **Dynamic shapes** (different batch sizes, sequence lengths) trigger recompilation. Use `torch.compile(model, dynamic=True)` to use dynamic shape tracing, or pad inputs to fixed sizes.
- Steady-state speedup is typically 1.3-2x from kernel fusion, reduced memory traffic, and eliminated Python overhead.
- Not worth it when: total training is <100 steps (compilation dominates), model is very small (not enough work to amortize), or you need dynamic shapes without padding.
- For benchmarking, always exclude the first 3-5 steps. Report median, not mean, to exclude outliers.

</details>

<details>
<summary>Tool to use</summary>

**`time.perf_counter()` with `torch.cuda.synchronize()`** — the fundamental timing tool. Measure each step individually to see compilation warmup clearly.

**`torch._dynamo.utils.CompileProfiler`** — shows what `torch.compile` decided to compile, what it skipped (graph breaks), and why:
```python
with torch._dynamo.utils.CompileProfiler() as prof:
    compiled_model(x)
print(prof.report())
```

**`TORCH_LOGS="+dynamo" python your_script.py`** — environment variable that enables verbose logging of what Dynamo is tracing, where graph breaks happen, and compilation time. Extremely noisy but useful for diagnosing graph breaks.

**`torch._dynamo.explain(model, x)`** — shows graph breaks and their causes without actually compiling. Use this to check if your model is compile-friendly before paying the compilation cost.

</details>

<details>
<summary>Solution</summary>

```python
import time
import statistics
import torch
from ai_playground.models import Transformer, TransformerConfig

config = TransformerConfig.SMALL

# Part 1: Measure compilation overhead
print("=== Compilation Overhead ===\n")

model = Transformer(config).cuda()
x = torch.randint(0, config.vocab_size, (8, 2048)).cuda()

# Eager baseline
torch.cuda.synchronize()
eager_times = []
for i in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(x).sum().backward()
    torch.cuda.synchronize()
    eager_times.append(time.perf_counter() - t0)
eager_median = statistics.median(eager_times[3:])  # skip warmup

# Compiled
compiled = torch.compile(Transformer(config).cuda())
compiled_times = []
for i in range(15):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled(x).sum().backward()
    torch.cuda.synchronize()
    compiled_times.append(time.perf_counter() - t0)

for i, t in enumerate(compiled_times):
    marker = ""
    if t > eager_median * 5:
        marker = " ← COMPILING"
    elif t < eager_median * 0.8:
        marker = " ← faster than eager"
    print(f"Step {i:>2}: {t*1000:>8.1f} ms{marker}")

compiled_median = statistics.median(compiled_times[5:])
print(f"\nEager median:    {eager_median*1000:.1f} ms")
print(f"Compiled median: {compiled_median*1000:.1f} ms")
print(f"Speedup:         {eager_median/compiled_median:.2f}x")
print(f"Compilation cost: {sum(t for t in compiled_times[:3])*1000:.0f} ms")
print(f"Break-even step:  ~{sum(compiled_times[:3]) / (eager_median - compiled_median):.0f}")

# Part 2: Dynamic shapes trigger recompilation
print(f"\n=== Dynamic Shape Recompilation ===\n")

compiled2 = torch.compile(Transformer(config).cuda())

for seq_len in [512, 1024, 2048, 512, 1024]:  # revisit old shapes
    x = torch.randint(0, config.vocab_size, (8, seq_len)).cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled2(x).sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    recompiled = elapsed > eager_median * 5
    print(f"seq_len={seq_len:>5}: {elapsed*1000:>8.1f} ms "
          f"{'← RECOMPILED' if recompiled else ''}")

# Part 3: Fix with dynamic=True
print(f"\n=== With dynamic=True ===\n")

compiled3 = torch.compile(Transformer(config).cuda(), dynamic=True)

for seq_len in [512, 1024, 2048, 512, 1024]:
    x = torch.randint(0, config.vocab_size, (8, seq_len)).cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    compiled3(x).sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"seq_len={seq_len:>5}: {elapsed*1000:>8.1f} ms")

# Part 4: Check for graph breaks
print(f"\n=== Graph Break Analysis ===\n")
explanation = torch._dynamo.explain(Transformer(config).cuda(), x)
print(explanation)
```

**Expected results:**

```
Step  0: 35000.0 ms ← COMPILING (forward graph)
Step  1: 18000.0 ms ← COMPILING (backward graph)
Step  2:   420.0 ms
Step  3:   380.0 ms ← faster than eager
...

Eager median:    520.0 ms
Compiled median: 380.0 ms
Speedup:         1.37x
Break-even step: ~380
```

**Decision: when to use torch.compile:**

| Training length | Compilation cost | Net benefit? |
|----------------|-----------------|--------------|
| 20 steps | ~50 sec compile, save ~3 sec | No (17x overhead) |
| 200 steps | ~50 sec compile, save ~28 sec | Borderline |
| 2000 steps | ~50 sec compile, save ~280 sec | Yes (5.6x ROI) |
| 20000 steps | ~50 sec compile, save ~2800 sec | Yes (56x ROI) |

**Best practices:**
- Always use for training runs >500 steps
- Use `dynamic=True` if sequence lengths vary
- Check for graph breaks with `torch._dynamo.explain()` before relying on speedup
- Report benchmark numbers excluding compilation warmup (use `median(steps[5:])`, not `mean(all_steps)`)

</details>

---

## Exercise 9: Finding the Data Loading Bottleneck

**Scenario:** You doubled the GPU count but training only got 1.3x faster, not 2x. You suspect the data pipeline can't keep up.

**Task:** Determine if you're data-bound or compute-bound.

**Diagnostic steps:**

1. **Measure data loading time separately:**
   ```python
   import time
   from ai_playground.training.data import create_dataloader

   loader = create_dataloader(dataset, batch_size=16, num_workers=0)

   # Time 20 batches with 0 workers
   t0 = time.perf_counter()
   for i, batch in enumerate(loader):
       if i == 20: break
   print(f"0 workers: {(time.perf_counter()-t0)/20*1000:.1f} ms/batch")

   # Now with 4 workers
   loader = create_dataloader(dataset, batch_size=16, num_workers=4)
   t0 = time.perf_counter()
   for i, batch in enumerate(loader):
       if i == 20: break
   print(f"4 workers: {(time.perf_counter()-t0)/20*1000:.1f} ms/batch")
   ```

2. **Compare data time vs compute time:** If data loading takes longer than a training step, you're data-bound.

**Questions:**
1. What's the GPU doing while data is loading?
2. How many `num_workers` saturates the data pipeline?
3. Does `pin_memory=True` help? Why?
4. Would pre-tokenized data (stored as tensors) vs on-the-fly tokenization change the picture?

<details>
<summary>Hints</summary>

- **Rule of thumb:** data loading should take <10% of step time. If it's more, you're leaving GPU idle.
- `num_workers=4` is a good starting point. More workers use more CPU memory. Too many can cause contention. Optimal is usually `num_workers = 2-4 * num_gpus`.
- `pin_memory=True` uses page-locked memory for CPU→GPU transfers, which enables async DMA copies. This overlaps data transfer with compute. Only helps when data is loaded on CPU.
- Pre-tokenized tensors (what `TextDataset` uses) are much faster than on-the-fly tokenization. Tokenization is CPU-intensive and becomes the bottleneck at scale.

</details>

<details>
<summary>Tool to use</summary>

**`time.perf_counter()`** — measure data loading time in isolation by iterating the DataLoader without any model computation. Compare this to measured step time.

**`ai_playground.profiling.nsight.profile_with_torch`** — the Chrome trace shows data loading as CPU-only bars between GPU kernel bursts. If you see long CPU gaps with no GPU activity, data loading is the bottleneck.

**`htop` or `top`** — check CPU utilization across cores. If DataLoader workers are pinned at 100% on a few cores while others are idle, you need more workers. If all cores are busy, the CPU itself is the bottleneck (consider pre-tokenizing).

**`iostat -x 1`** — check disk I/O utilization. If `%util` is near 100%, storage bandwidth is the bottleneck. This is common with NFS or network storage on cloud instances.

**For distributed training specifically:** use **`torch.profiler`** and look for `ncclAllReduce` timing. If all-reduce takes longer than data loading, the bottleneck is communication, not data.

</details>

<details>
<summary>Solution</summary>

```python
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

# Create a synthetic dataset that mimics pre-tokenized text
seq_len = 2048
dataset_size = 10000
data = torch.randint(0, 32000, (dataset_size, seq_len))
dataset = TensorDataset(data)

# Part 1: Measure data loading at different num_workers
print("=== Data Loading Speed ===\n")

for num_workers in [0, 1, 2, 4, 8]:
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    # Warmup
    it = iter(loader)
    next(it)

    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= 50:
            break
    data_time = (time.perf_counter() - t0) / 50 * 1000
    print(f"num_workers={num_workers}: {data_time:.1f} ms/batch")

# Part 2: Measure compute time
print(f"\n=== Compute Time ===\n")
from ai_playground.models import Transformer, TransformerConfig
config = TransformerConfig.SMALL
model = Transformer(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x = torch.randint(0, 32000, (16, seq_len)).cuda()
# Warmup
for _ in range(3):
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    out = model(x)
    out.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
compute_time = (time.perf_counter() - t0) / 10 * 1000
print(f"Compute time: {compute_time:.1f} ms/step")

# Part 3: Compare and diagnose
print(f"\n=== Diagnosis ===\n")

# Use the num_workers=0 time as worst case
loader_slow = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
it = iter(loader_slow)
next(it)  # warmup
t0 = time.perf_counter()
for i, batch in enumerate(loader_slow):
    if i >= 50: break
data_time_slow = (time.perf_counter() - t0) / 50 * 1000

loader_fast = DataLoader(dataset, batch_size=16, shuffle=True,
                          num_workers=4, pin_memory=True)
it = iter(loader_fast)
next(it)
t0 = time.perf_counter()
for i, batch in enumerate(loader_fast):
    if i >= 50: break
data_time_fast = (time.perf_counter() - t0) / 50 * 1000

print(f"Data (0 workers): {data_time_slow:.1f} ms/batch")
print(f"Data (4 workers): {data_time_fast:.1f} ms/batch")
print(f"Compute:          {compute_time:.1f} ms/step")
print()

if data_time_slow > compute_time:
    print("⚠ DATA-BOUND with 0 workers! GPU is idle waiting for data.")
    print(f"  GPU idle time per step: ~{data_time_slow - compute_time:.0f} ms")
    print(f"  Wasted: {(data_time_slow - compute_time) / data_time_slow * 100:.0f}% of wall time")
else:
    print("✓ Compute-bound with 0 workers (no bottleneck)")

if data_time_fast > compute_time:
    print("⚠ Still data-bound even with 4 workers. Consider:")
    print("  - Pre-tokenize data to .pt files")
    print("  - Use faster storage (local SSD vs NFS)")
    print("  - Increase num_workers further")
else:
    print("✓ 4 workers is sufficient — data pipeline keeps up with GPU")
    print(f"  Data/compute ratio: {data_time_fast/compute_time:.1%} (target: <10%)")

# Part 4: pin_memory effect
print(f"\n=== pin_memory Effect ===\n")

for pin in [False, True]:
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=pin)
    it = iter(loader)
    next(it)

    transfer_times = []
    for i, (batch,) in enumerate(loader):
        if i >= 50: break
        t0 = time.perf_counter()
        batch = batch.to("cuda", non_blocking=pin)
        if not pin:
            torch.cuda.synchronize()
        transfer_times.append(time.perf_counter() - t0)

    avg_transfer = sum(transfer_times) / len(transfer_times) * 1000
    print(f"pin_memory={str(pin):>5}: {avg_transfer:.2f} ms/transfer "
          f"{'(async DMA)' if pin else '(synchronous)'}")
```

**Key results:**
- `num_workers=0`: data loading happens on the main thread, GPU stalls every step
- `num_workers=4`: data prefetched in background, usually eliminates the bottleneck for pre-tokenized data
- `pin_memory=True`: enables async CPU→GPU transfer via DMA, overlapping with compute. Saves 0.5-2ms per batch.
- Pre-tokenized `.pt` files are 10-100x faster to load than raw text + tokenization

**Scaling diagnostic for multi-GPU:**
If 4 GPUs gives 1.3x instead of 4x speedup, check:
1. **Data I/O:** 4 GPUs = 4x the read bandwidth needed. `iostat -x 1` shows if disk is saturated.
2. **Communication:** `torch.profiler` shows NCCL all-reduce time. If it's >30% of step time, gradient compression or FSDP can help.
3. **Load imbalance:** different GPUs finishing at different times due to uneven batches.

</details>

---

## Exercise 10: Profile-Guided Optimization Workflow

**Scenario:** You have a model and training pipeline that works but is slower than expected. You need a systematic approach to find and fix bottlenecks.

**Task:** Follow this workflow on the `small.yaml` configuration:

### Step 1: Establish baselines

```bash
# Training throughput
uv run python scripts/train.py --config configs/small.yaml --max-steps 20 --dtype bfloat16
# Note: tokens/sec from the log output

# Inference throughput
uv run python scripts/benchmark.py --config configs/small.yaml --dtype bfloat16

# FLOP efficiency
uv run python scripts/profile_model.py --config configs/small.yaml
```

### Step 2: Profile

```bash
# PyTorch profiler trace
# Open the output in chrome://tracing or TensorBoard
```

### Step 3: Identify the bottleneck category

| Symptom | Bottleneck | Fix |
|---------|-----------|-----|
| GPU util low, CPU util high | Data loading | More workers, pin_memory, pre-tokenize |
| GPU util high, MFU low | Memory bandwidth | torch.compile, Flash Attention, fused ops |
| GPU util varies wildly | Sync points | Remove .item() calls, async logging |
| OOM at reasonable batch size | Activation memory | Gradient checkpointing, BF16, FSDP |
| Multi-GPU scaling <0.8x linear | Communication | Overlap comm+compute, reduce gradient size |

### Step 4: Apply one fix at a time, re-measure

Never apply multiple optimizations simultaneously — you won't know which one helped. Keep a log:

```
Baseline:          1,200 tok/s, MFU 22%
+ torch.compile:   1,800 tok/s, MFU 34%  (+50%)
+ BF16:            3,100 tok/s, MFU 38%  (+72%)
+ Flash Attention:  3,900 tok/s, MFU 48%  (+26%)
```

<details>
<summary>Hints</summary>

- Always benchmark with warmup (exclude first 3-5 steps).
- Use `torch.cuda.synchronize()` before timing CUDA operations — otherwise you're timing kernel launch, not execution.
- Profile memory with `torch.cuda.memory_summary()` for a full breakdown of the allocator state.
- When in doubt about whether something is compute-bound or memory-bound, compute the arithmetic intensity: `FLOPs / bytes_loaded`. Compare to the GPU's compute-to-bandwidth ratio (A100: 312 TFLOPS / 2 TB/s = 156 FLOPs/byte). Operations below this ratio are memory-bound.
- The profiling modules in this repo (`profiling/flops.py`, `profiling/memory.py`, `profiling/nsight.py`) are designed to be used together for this workflow.

</details>

<details>
<summary>Tool to use</summary>

This exercise uses **all the profiling tools together**. Here's when to reach for each one:

| Question | Tool | What it tells you |
|----------|------|-------------------|
| How fast am I training? | `scripts/train.py` log output | Tokens/sec, loss curve |
| How fast is inference? | `benchmark.benchmark_generation` | Prefill/decode tok/s, TTFT, peak memory |
| Am I using the GPU efficiently? | `profiling.flops.compute_mfu` | MFU percentage — the single most important metric |
| Where does memory go? | `profiling.memory.MemoryTracker` | Per-component memory breakdown |
| Where does time go? | `profiling.nsight.profile_with_torch` | Chrome trace with CPU/GPU timeline |
| Is data loading slow? | `time.perf_counter()` on DataLoader | ms/batch vs ms/step |
| Are there GPU stalls? | Chrome trace from `profile_with_torch` | Gaps between CUDA kernels |
| What's the compute breakdown? | `profiling.flops.estimate_flops` | Attention vs FFN, per-layer vs total |
| Is it compute or memory bound? | Arithmetic intensity calculation | FLOPs / bytes → compare to GPU roofline |

**External tools:**
- `nvidia-smi dmon -s u -d 1` — real-time GPU utilization monitoring
- `htop` — CPU utilization across cores
- `iostat -x 1` — disk I/O utilization
- `nsys profile` — Nsight Systems for deep kernel-level timeline
- `chrome://tracing` or `perfetto.dev` — view PyTorch profiler traces

</details>

<details>
<summary>Solution</summary>

```python
"""
Complete profile-guided optimization workflow.
Run each section, record the result, apply one fix, repeat.
"""
import time
import torch
from ai_playground.models import Transformer, TransformerConfig
from ai_playground.profiling.flops import compute_mfu, estimate_flops
from ai_playground.profiling.memory import MemoryTracker
from ai_playground.inference.benchmark import benchmark_generation, print_benchmark

config = TransformerConfig.SMALL
results_log = []

def benchmark_training(model, label, use_amp=False, use_compile=False):
    """Measure training throughput and MFU."""
    if use_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, config.vocab_size, (16, 2048)).cuda()

    # Warmup (extra for torch.compile)
    warmup = 8 if use_compile else 3
    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 20
    for _ in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(x)
            out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    step_time = (time.perf_counter() - t0) / steps
    tokens_per_sec = 16 * 2048 / step_time
    mfu = compute_mfu(config, seq_len=2048, batch_size=16,
                       step_time_sec=step_time, gpu_name="A100_80GB")

    results_log.append({
        "label": label,
        "tokens_per_sec": tokens_per_sec,
        "mfu_percent": mfu["mfu_percent"],
        "step_time_ms": step_time * 1000,
    })

    return step_time

# === Run optimization sequence ===

# 1. Baseline: Eager FP32
model = Transformer(config).cuda()
benchmark_training(model, "Baseline (Eager FP32)")

# 2. + BF16 mixed precision
model = Transformer(config).cuda()
benchmark_training(model, "+ BF16", use_amp=True)

# 3. + torch.compile
model = Transformer(config).cuda()
benchmark_training(model, "+ torch.compile", use_compile=True)

# 4. + Both
model = Transformer(config).cuda()
benchmark_training(model, "+ compile + BF16", use_amp=True, use_compile=True)

# === Print optimization log ===
print(f"\n{'='*70}")
print(f"{'OPTIMIZATION LOG':^70}")
print(f"{'='*70}")
print(f"{'Configuration':<30} {'Tokens/sec':>12} {'MFU':>8} {'Step ms':>10} {'Speedup':>8}")
print(f"{'-'*70}")

baseline = results_log[0]["tokens_per_sec"]
for r in results_log:
    speedup = r["tokens_per_sec"] / baseline
    print(f"{r['label']:<30} {r['tokens_per_sec']:>10,.0f} {r['mfu_percent']:>7.1f}% "
          f"{r['step_time_ms']:>8.0f}ms {speedup:>7.2f}x")

# === Memory analysis at best config ===
print(f"\n{'='*70}")
print(f"{'MEMORY BREAKDOWN':^70}")
print(f"{'='*70}")

tracker = MemoryTracker()
model = Transformer(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
x = torch.randint(0, config.vocab_size, (16, 2048)).cuda()

tracker.snapshot("model_loaded")
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    model(x).sum().backward()
tracker.snapshot("after_backward")
optimizer.step()
tracker.snapshot("after_optimizer")

tracker.report()

print(f"\n{'='*70}")
print(f"{'WHAT TO TRY NEXT':^70}")
print(f"{'='*70}")
best = results_log[-1]
print(f"\nCurrent best: {best['mfu_percent']:.0f}% MFU")
if best['mfu_percent'] < 35:
    print("→ Try Flash Attention (install flash-attn package)")
    print("→ Check for graph breaks: torch._dynamo.explain(model, x)")
    print("→ Profile for CPU-GPU sync points in the training loop")
elif best['mfu_percent'] < 50:
    print("→ Try Flash Attention for another 10-20% improvement")
    print("→ Try Triton fused kernels for RMSNorm + residual")
    print("→ Consider CUDA graphs for kernel launch overhead")
else:
    print("→ You're in good shape! Focus on scaling (DDP/FSDP) next")
    print("→ Or try larger batch sizes to improve GPU utilization")
```

**Expected output on A100:**

```
======================================================================
                          OPTIMIZATION LOG
======================================================================
Configuration              Tokens/sec      MFU   Step ms  Speedup
----------------------------------------------------------------------
Baseline (Eager FP32)          38,400    22.0%     854ms    1.00x
+ BF16                         62,000    30.5%     529ms    1.61x
+ torch.compile                57,000    34.2%     575ms    1.48x
+ compile + BF16               95,000    44.8%     345ms    2.47x
```

**The workflow in summary:**

```
1. MEASURE  →  tokens/sec, MFU, memory usage
2. PROFILE  →  Chrome trace, nvidia-smi, memory tracker
3. DIAGNOSE →  compute-bound? memory-bound? data-bound? comm-bound?
4. FIX ONE  →  apply the fix matching the diagnosed bottleneck
5. MEASURE  →  verify improvement, record in log
6. REPEAT   →  until MFU is satisfactory or the next bottleneck is acceptable
```

**Common mistake:** Applying all optimizations at once. You won't know which helped, and some can interact negatively (e.g., `torch.compile` + custom CUDA kernels can cause graph breaks). Always change one variable at a time.

</details>
