# Virtual GPU — Design Spec

Status: **Phases 1–3 implemented** (`gpusim/spec.py`, `occupancy.py`, `timing.py`, `trace.py`,
`render.py`, chapter `01c_virtual_gpu.ipynb`, `tests/test_vgpu.py`); hardware verification and
the real-T4 validation table still pending (needs a Colab session), Phase 4 not started. Extends `src/ai_playground/gpusim/` (the pure-Python SIMT
simulator, see `notebooks/05_gpu_nvidia_tools/01b_simt_simulator.ipynb`) with a
**cycle-approximate timing model of a real GPU** and a **graphical view of the machine
executing a kernel**.

## Motivation

`gpusim` today answers *"what does my kernel do"* (semantics) and *"how much traffic does it
generate"* (transaction counts). It cannot answer *"why does it take this long"* — the question
every profiler answers with numbers. The virtual GPU answers it with a picture: SMs lighting
up, warps stalling on memory, bandwidth gauges saturating, and a time axis under all of it.

The learning payoffs, in order:

1. **Latency hiding is occupancy** — watch an SM's timeline fill its stall holes as resident
   warps go from 4 to 32.
2. **Bandwidth-bound vs compute-bound is visible** — a memory-bound kernel turns the whole die
   orange (stalled-on-memory) while the FLOP units idle.
3. **Occupancy math is mechanical** — registers/thread, shared/block, and warp slots each cap
   residency; the calculator shows which one binds.
4. **Profilers stop being magic** — our warp waterfall is baby-Nsight; after reading it,
   Nsight Compute's sections map onto concepts the reader has already animated.

## Target hardware: NVIDIA Tesla T4

Chosen because it is **Colab's free GPU** and every notebook in this repo carries a Colab
badge — readers can run the same kernel on the real card the simulator mimics and compare.
Model parameters (Turing TU104, compute capability 7.5):

| Parameter | Value | Notes |
|---|---|---|
| SMs | 40 | |
| FP32 lanes / SM | 64 | 2,560 CUDA cores total |
| Warp schedulers / SM | 4 | 1 instruction/scheduler/cycle |
| Max threads / SM | 1,024 | = 32 warps (Turing halved Volta's 2,048) |
| Max blocks / SM | 16 | |
| Registers / SM | 65,536 × 32-bit | 256 KB register file |
| Shared memory / SM | up to 64 KB | carved from 96 KB unified L1/shared |
| L2 | 4 MB | shared by all SMs |
| DRAM | 16 GB GDDR6, 256-bit | 320 GB/s theoretical; ~220–250 GB/s achievable |
| Boost clock | 1,590 MHz | 2,560 × 2 × 1.59 GHz ≈ 8.1 FP32 TFLOPS |
| FP16 tensor peak | 65 TFLOPS | matches `GPU_PEAK_TFLOPS["T4"]` in `profiling/flops.py` |

The spec ships as a dataclass so other cards are just different numbers:

```python
@dataclass(frozen=True)
class GPUSpec:
    name: str
    n_sm: int
    fp32_lanes_per_sm: int
    warp_schedulers_per_sm: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    registers_per_sm: int
    smem_per_sm: int            # bytes
    l2_bytes: int
    dram_bandwidth: float       # bytes/sec, theoretical
    boost_clock_hz: float
    # Timing model parameters (cycles) — see "Latencies" below
    lat_dram: int = 450
    lat_smem: int = 25
    lat_alu: int = 4

T4 = GPUSpec(name="T4", n_sm=40, ...)
```

**Phase-1 verification task:** confirm every number against `deviceQuery` and the CUDA
occupancy API on a real Colab T4, and against Jia et al.,
[*Dissecting the NVIDIA Turing T4 GPU via Microbenchmarking*](https://arxiv.org/abs/1903.07486)
(the definitive measured-latency source for this exact card). Numbers above are from the
Turing whitepaper and public spec sheets; treat them as unverified until this task closes.

## Fidelity model — what we simulate and what we refuse

Three layers; the first two exist:

1. **Functional** (`gpusim.core`) — indexing, shared memory, barriers. Exists.
2. **Transaction counting** (`gpusim.memory`) — 128 B segments per warp access, bank
   conflicts. Exists.
3. **Timing** (this spec) — event-driven, *cycle-approximate*:
   - Blocks are assigned to SMs up to the **occupancy limit** (min over the four caps:
     warp slots, blocks, registers, shared memory).
   - Each SM's schedulers issue **ready warps round-robin**, one instruction-equivalent
     per cycle per scheduler.
   - A global-memory access parks the warp for `lat_dram` cycles **and** consumes tokens
     from a chip-wide **bandwidth token bucket** (`dram_bandwidth / boost_clock` bytes per
     cycle) — this is how contention and the bandwidth wall emerge.
   - Transactions counted by the existing coalescing logic decide how many tokens an access
     costs: a coalesced warp read costs 1×128 B, a fully strided one 32×128 B.
   - Shared-memory access parks for `lat_smem × conflict_degree`.
   - `syncthreads` parks until the whole block arrives.
   - Arithmetic between memory ops is charged at a coarse per-access `lat_alu` (we do not
     count individual Python operations — see Refusals).

**Refusals** (each would multiply complexity without a proportional teaching payoff):
L1/L2 cache hit modeling and replacement policies, instruction-level parallelism and
dual-issue, DRAM row-buffer effects, ECC, clock boosting/throttling, tensor cores,
instruction fetch/decode. Research-grade simulation of those is
[GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution)
([Bakhoda et al., ISPASS 2009](https://ieeexplore.ieee.org/document/4919648)) and
[Accel-Sim](https://accel-sim.github.io/) ([Khairy et al., ISCA 2020](https://arxiv.org/abs/1811.08933))
territory — years of C++, validated per-benchmark. We are building a teaching instrument,
not a research instrument, and we say so on the tin.

**Honesty rule:** the simulator predicts *trends and ratios*, never absolute milliseconds.
Every notebook claim must be phrased comparatively ("tiled is predicted 4.2× faster;
the real T4 measures 3.7×") and validated per the Validation section.

## Architecture

```
gpusim/
├── core.py        # existing: functional SIMT execution
├── memory.py      # existing: access log, transactions, bank conflicts
├── viz.py         # existing: per-warp access plots
├── spec.py        # NEW  Ph1: GPUSpec dataclass + named specs (T4, ...)
├── occupancy.py   # NEW  Ph1: residency calculator (which cap binds, and why)
├── timing.py      # NEW  Ph2: event-driven engine  -> Trace
├── trace.py       # NEW  Ph2: Trace container; export to JSON + Chrome trace format
└── render.py      # NEW  Ph3: matplotlib floorplan + warp waterfall + gauges
```

- `timing.py` **reuses kernels unchanged**: the functional simulator already surfaces every
  memory access and barrier, which are exactly the timing model's event points. A timed run
  wraps the functional run: first execute functionally per block (correctness + access log),
  then replay the per-warp event streams through the event-driven SM model.
- `trace.py` defines one event schema — `(t_start, t_end, sm, warp_slot, block, state)` with
  `state ∈ {issue, stall_mem, stall_smem, stall_sync, done}` plus chip-level bandwidth
  samples — and exports both raw JSON and
  [Chrome trace format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
  so any trace opens in Perfetto / `chrome://tracing`. (The repo's PyTorch profiler wrapper
  already emits Chrome traces; this keeps one viewer for both.)

## Graphical representation

All views render from the Trace — the engine never draws.

**1. Die floorplan (hero visual).** Grid of 40 SM tiles; each tile shows its warp slots as
small squares colored by state (green = issuing, orange = stalled on DRAM, violet = waiting
at barrier, gray = done — house palette), DRAM controllers at the edges with live bandwidth
bars, L2 in the center, a global cycle counter. Rendered as matplotlib animation (embedded
video on the site); scrubbing = frame index.

```
┌──────────────────────── T4 · cycle 12,480 ────────────────────────┐
│ [SM00 ▦▦▦▣▣▣··] [SM01 ▦▦▦▦▣▣··] … [SM09 ▦▣▣▣▣▣··]   DRAM ▓▓▓▓▓░ │
│ [SM10 ▦▦▣▣▣▣··] …                                     DRAM ▓▓▓▓▓▓ │
│                        [   L2 · 4 MB   ]                          │
└───────────────────────────────────────────────────────────────────┘
```

**2. Warp waterfall.** One SM zoomed: warps as rows, cycles as x, colored bars per state.
The latency-hiding lesson lives here: at low occupancy the row gaps are visible; at 32
resident warps the gaps tile over.

**3. Gauges + roofline.** Achieved vs peak bandwidth, occupancy per cap, and the kernel's
dot on the T4 roofline, animated as the run progresses.

**4. (Stretch) Interactive HTML player.** The trace is JSON; a self-contained canvas+slider
page fits the site's no-external-dependency constraint. Prototype as an artifact before
committing to it. Until then, Perfetto covers interactive exploration for free.

## Validation

Run on a real Colab T4 (badge already on every module-05 notebook):

| Experiment | Simulator claim to validate |
|---|---|
| contiguous vs strided copy | bandwidth ratio (~10–30× measured in `01_cuda_basics` §7) |
| vector add, occupancy sweep (4→32 warps/SM) | throughput vs occupancy curve *shape* |
| naive vs tiled matmul | relative speedup ratio |
| bandwidth-bound kernel | achieved GB/s within ~2× of measured |

**Acceptance bar:** rank order always preserved; ratios within ~2×. If a config misses the
bar, the notebook shows the miss and explains it — a wrong-but-explained prediction teaches
more than a tuned one.

**Runnable harness:** `01c_virtual_gpu.ipynb` §7 contains the deviceQuery checks and the
first two trend experiments as CUDA-guarded cells — open the chapter on Colab (T4 runtime),
run them, and record the results here. Occupancy-API cross-check and the matmul trend are
still to be added to the harness.

## Phases

| Phase | Deliverable | Size | Definition of done |
|---|---|---|---|
| 1 | `spec.py` + `occupancy.py` — **done** (12 tests, hand-derived expectations) | ~150 lines | matches CUDA occupancy API on 10 configs (Colab verification still open); standalone useful |
| 2 | `timing.py` + `trace.py` — **done** (model-level tests: latency hiding, bandwidth wall, tiled < naive, determinism; Chrome export structurally verified) | ~400 lines | real-T4 validation table and a visual Perfetto check still open |
| 3 | `render.py` + notebook `01c_virtual_gpu.ipynb` — **done** (floorplan animation, waterfall, gauges; chapter on site) | ~250 lines + chapter | real-T4 comparison cells still open (Colab) |
| 4 (stretch) | interactive HTML trace player — **prototyped as a private artifact** (canvas die floorplan + DRAM gauge + per-SM waterfall, time scrubber, two embedded kernel traces, light/dark; verified in-browser) | — | promote to a site page only if the prototype earns it; Perfetto covers interactive needs meanwhile |

Each phase lands with tests (extend `tests/test_gpusim.py` or add `tests/test_vgpu.py`),
ruff-clean code, docstrings that explain the *why* per `CLAUDE.md`, and — for notebook
claims — numbers that were actually measured.

## Risks

- **Simulation speed.** Python event loop over ~10⁵ warp-events: fine at teaching sizes
  (n ≤ 64 matmuls, ≤ 10⁴ threads). Keep kernels toy-sized; never present simulated wall-time
  as a benchmark.
- **Over-modeling.** Every mechanism added must have a *visible* teaching payoff in one of
  the three views; otherwise it goes on the Refusals list. This is the project's main
  failure mode — the spec exists partly to say no later.
- **Spec drift.** The T4 numbers above are unverified against hardware until Phase 1's
  deviceQuery task; anything downstream (occupancy tests, validation ratios) blocks on that.

## References

- Jia, Maggioni, Smith, Scarpazza — [Dissecting the NVIDIA Turing T4 GPU via Microbenchmarking](https://arxiv.org/abs/1903.07486) (measured latencies for this exact card)
- [NVIDIA Turing architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
- Bakhoda et al. — GPGPU-Sim ([ISPASS 2009](https://ieeexplore.ieee.org/document/4919648)); Khairy et al. — [Accel-Sim](https://arxiv.org/abs/1811.08933) (what research-grade looks like; we are deliberately not this)
- Stephen Jones — [How CUDA Programming Works](https://www.youtube.com/watch?v=QQceTDjA4f4) (GTC 2022) — the mental model the floorplan view animates; see [VIDEOS.md](VIDEOS.md)
- This repo: `notebooks/05_gpu_nvidia_tools/01_cuda_basics.ipynb` (concepts), `01b_simt_simulator.ipynb` (functional layer), `src/ai_playground/profiling/flops.py` (roofline/MFU math to reuse)
