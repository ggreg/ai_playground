# SGEMM tutorial kernels

Five progressively faster CUDA kernels that compute `C = alpha * A @ B + beta * C` for row-major float matrices, walking through the optimization techniques from [Salykova's article](https://salykova.github.io/sgemm-gpu) (see `docs/PAPERS.md` → *GPU Kernels & Performance*). The companion notebook is `../02_sgemm_optimization.ipynb`.

## Files

| File | What it adds |
|------|--------------|
| `01_naive.cu` | Baseline: one thread per output element, no reuse. |
| `02_smem.cu` | Block-level shared-memory tiling (`32x32x32`). |
| `03_block_tile.cu` | 2D thread tile (`128x128x8`, 8x8 register accumulators). |
| `04_vectorized.cu` | float4 (LDG.128) loads + transposed shared-memory `As` for vectorized inner-loop reads. |
| `05_async_pipeline.cu` | `cp.async` + double-buffered software pipeline (Ampere SM_80+). |
| `bench.cu` | Validates each kernel against cuBLAS and benchmarks with L2 flushed between replays. |
| `common.cuh` | Shared utilities (CUDA error check, kernel launcher declarations). |
| `Makefile` | nvcc build rules. |

## Build

You need the CUDA toolkit (nvcc + cuBLAS) and an NVIDIA GPU. The default target is `sm_80` (A100, RTX 30xx). Override for other GPUs:

```bash
make                    # SM_80, e.g. A100
make SM_ARCH=sm_86      # RTX 3090, A40
make SM_ARCH=sm_89      # RTX 4090, L40
make SM_ARCH=sm_90      # H100
```

Note: kernel 5 requires Ampere or newer (SM_80+) because it uses `cp.async`. On older GPUs the cp.async PTX will compile to no-ops and the kernel will produce incorrect output — don't run it on Volta/Turing.

## Run

```bash
./bench --size 4096 --iters 50 --json results.json
```

Flags:

- `--size N` &mdash; problem size, tested as M=N=K=N (default 4096).
- `--iters N` &mdash; replay count for the median timing (default 50).
- `--warmup N` &mdash; warmup runs before timing (default 5).
- `--no-validate` &mdash; skip the max-abs-error check against cuBLAS.
- `--json PATH` &mdash; write results for the notebook to load.

Sample output on an RTX 3090 (sm_86):

```
GPU: NVIDIA GeForce RTX 3090  (SM 8.6, 82 SMs)
Problem: M=N=K=4096  (137.44 GFLOPs/call, 192.0 MB working set)

kernel                  median (ms)       TFLOPS   % cuBLAS    max |err|
cublasSgemm (ref)             7.142        19.24     100.0%            0
01_naive                    482.310         0.28       1.5%       1.0e-5
02_smem                      33.270         4.13      21.5%       1.0e-5
03_block_tile                10.190        13.49      70.1%       1.0e-5
04_vectorized                 8.220        16.72      86.9%       1.0e-5
05_async_pipeline             7.490        18.35      95.4%       1.0e-5
```

(Numbers are illustrative — your GPU, driver, clock state, and ambient temp all matter. Lock clocks for reproducible runs: `sudo nvidia-smi -lgc 1395`.)

## Reproducible benchmarking

The article emphasizes — and so does this harness — that GEMM benchmark numbers vary wildly between locked and unlocked clocks, between cold and warm L2, and between first and steady-state runs. The bench:

- flushes L2 with a 64MB `cudaMemsetAsync` between replays,
- discards a configurable number of warmup runs,
- reports the **median** of the timed runs (not the mean — gives a robust point estimate against thermal-throttling outliers).

For the cleanest numbers:

```bash
sudo nvidia-smi --persistence-mode=1
sudo nvidia-smi -lgc 1395   # base core clock for RTX 3090
./bench --size 4096 --iters 100 --json results.json
sudo nvidia-smi -rgc        # release locked clocks
```

## What we did *not* do

These would each be a worthwhile follow-up exercise:

- **Warp tiling** &mdash; group threads into warps that compute a contiguous warp-level tile. Helps register reuse and matches the CUTLASS hierarchy. Adds ~5-10% on top of kernel 5.
- **Bank-conflict-free `Bs`** &mdash; pad `Bs` to width 132 instead of 128 so the inner-loop `regN` reads don't 4-way conflict. Adds a few percent.
- **Tensor cores (`mma.sync`)** &mdash; FP32 → TF32 mode on Ampere or HMMA on Volta. ~10x speedup vs FP32 FMA but loses precision; this tutorial sticks to FP32 to keep the focus on memory-hierarchy techniques.
- **Hopper TMA** &mdash; SM_90's tensor memory accelerator replaces `cp.async` with bulk async copies for tiles, freeing more registers. Different programming model.
- **Persistent kernels / split-K** &mdash; useful for tall-skinny matrices where the tile parallelism is too coarse.

## References

See `../../docs/PAPERS.md` → *GPU Kernels & Performance* for:
- Salykova's article (the direct inspiration)
- Simon Boehm's tutorial (the parallel reference)
- CUTLASS (the production implementation)
- The Volta microbenchmarking paper (numbers behind why these tricks work)
- The cp.async PTX docs
