"""Hardware specs for the virtual GPU (docs/VIRTUAL_GPU_SPEC.md).

A GPU model is just numbers: how many SMs, how many warp slots, how big the
register file, how fast the DRAM. Everything downstream — the occupancy
calculator, the timing engine, the floorplan renderer — consumes a GPUSpec
and nothing else, so mimicking a different card means adding one constant.

The reference card is the Tesla T4 (Turing TU104, compute capability 7.5)
because it is Colab's free GPU: every notebook in this repo carries an
"Open in Colab" badge, so simulator predictions can be checked against the
real card they model from the same notebook.

VERIFICATION STATUS: values below come from the Turing/Ampere whitepapers
and public spec sheets, cross-checked against Jia et al.,
"Dissecting the NVIDIA Turing T4 GPU via Microbenchmarking"
(https://arxiv.org/abs/1903.07486) — but they have NOT yet been confirmed
with deviceQuery / the CUDA occupancy API on live hardware (the Phase-1
verification task in the spec). The latency fields are *model parameters*
in plausible measured ranges, not measurements.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    """Everything the simulator knows about a card.

    Occupancy-related fields mirror the CUDA occupancy calculator's inputs.
    Allocation granularities matter: registers are allocated in
    `reg_alloc_granularity`-sized chunks per warp and shared memory in
    `smem_alloc_granularity`-byte chunks per block, so a kernel using
    33 registers/thread really consumes 1,280 registers per warp, not 1,056.
    """

    name: str
    n_sm: int
    fp32_lanes_per_sm: int
    warp_schedulers_per_sm: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    registers_per_sm: int          # 32-bit registers
    max_registers_per_thread: int
    smem_per_sm: int               # bytes available to blocks (max carveout)
    max_smem_per_block: int        # bytes one block may request
    l2_bytes: int
    dram_bandwidth: float          # bytes/sec, theoretical peak
    boost_clock_hz: float
    warp_size: int = 32
    reg_alloc_granularity: int = 256   # registers, per warp
    smem_alloc_granularity: int = 256  # bytes, per block
    # Timing-model parameters (cycles) — used by Phase 2, documented in the spec
    lat_dram: int = 450
    lat_smem: int = 25
    lat_alu: int = 4

    @property
    def max_warps_per_sm(self) -> int:
        return self.max_threads_per_sm // self.warp_size

    @property
    def fp32_tflops(self) -> float:
        """Peak FP32: lanes x 2 (FMA = 2 FLOPs) x clock."""
        return self.n_sm * self.fp32_lanes_per_sm * 2 * self.boost_clock_hz / 1e12


# Tesla T4 — Colab's free GPU. 2,560 CUDA cores; 8.1 FP32 TFLOPS at boost.
# Turing halved Volta's thread residency: 1,024 threads (32 warps) per SM,
# which is why full occupancy on T4 is easier to hit than on V100/A100.
T4 = GPUSpec(
    name="T4",
    n_sm=40,
    fp32_lanes_per_sm=64,
    warp_schedulers_per_sm=4,
    max_threads_per_sm=1024,
    max_blocks_per_sm=16,
    registers_per_sm=65_536,
    max_registers_per_thread=255,
    smem_per_sm=64 * 1024,          # of the 96 KB unified L1/shared carveout
    max_smem_per_block=64 * 1024,
    l2_bytes=4 * 1024 * 1024,
    dram_bandwidth=320e9,           # 256-bit GDDR6 @ 10 Gbps; ~220-250 GB/s achievable
    boost_clock_hz=1.590e9,
)

# A100-SXM 40 GB — included for contrast: 2x the warp slots per SM, 2.5x the
# shared memory, ~5x the bandwidth. The same kernel's occupancy story changes
# card to card, which is the point of parameterizing.
A100_40GB = GPUSpec(
    name="A100_40GB",
    n_sm=108,
    fp32_lanes_per_sm=64,
    warp_schedulers_per_sm=4,
    max_threads_per_sm=2048,
    max_blocks_per_sm=32,
    registers_per_sm=65_536,
    max_registers_per_thread=255,
    smem_per_sm=164 * 1024,
    max_smem_per_block=163 * 1024,  # 1 KB reserved by the runtime
    l2_bytes=40 * 1024 * 1024,
    dram_bandwidth=1_555e9,
    boost_clock_hz=1.410e9,
    lat_dram=400,
    lat_smem=20,
)

SPECS = {s.name: s for s in (T4, A100_40GB)}
