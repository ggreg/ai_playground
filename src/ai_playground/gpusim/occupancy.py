"""Occupancy: how many warps actually fit on an SM, and which cap binds.

Occupancy = resident warps / warp slots. It matters because GPUs hide memory
latency by switching to another ready warp; with too few resident warps
there is nothing to switch to and the SM idles for the full ~450-cycle DRAM
round trip. It is *mechanical*: four independent caps each limit resident
blocks, the hardware takes the minimum, and the interesting question is
always *which* cap bound you — that tells you what to change.

The four caps, per SM:

1. **Warp slots** — resident warps <= max_warps_per_sm (32 on T4, 64 on A100).
2. **Block slots** — resident blocks <= max_blocks_per_sm (16 on T4).
3. **Registers** — each warp's registers round up to the allocation
   granularity (256), so 33 regs/thread costs ceil(33*32/256)*256 = 1,280
   registers per warp. One extra register per thread can drop a whole block.
4. **Shared memory** — each block's request rounds up to 256 B; a 48 KB
   tile on a 64 KB SM means one resident block, period.

Classic surprise this makes obvious: 96 threads/block on T4 gives 3 warps
per block, floor(32/3) = 10 blocks, 30/32 warps — you lose 6% to indivisible
block granularity before any resource pressure, and blocks of 3 warps also
burn block slots 5x faster than 512-thread blocks would.

Matches the CUDA occupancy calculator's model for CC 7.5/8.0. Hardware
verification on a live Colab T4 is the spec's Phase-1 close-out task.
"""

from dataclasses import dataclass

from .spec import GPUSpec


def _ceil_to(x: int, granularity: int) -> int:
    return -(-x // granularity) * granularity


@dataclass(frozen=True)
class Occupancy:
    """Result of the calculation, with the full why."""

    spec_name: str
    threads_per_block: int
    warps_per_block: int
    blocks_per_sm: int
    warps_per_sm: int
    occupancy: float          # resident warps / warp slots, 0.0-1.0
    limiter: str              # which cap bound: 'warps' | 'blocks' | 'registers' | 'shared'
    limits: dict[str, int]    # blocks allowed by each cap individually
    regs_per_warp_allocated: int
    smem_per_block_allocated: int

    def __repr__(self):
        caps = ", ".join(f"{k}={v}" for k, v in self.limits.items())
        return (
            f"<Occupancy {self.spec_name}: {self.occupancy:.0%} "
            f"({self.warps_per_sm} warps in {self.blocks_per_sm} blocks/SM), "
            f"limited by {self.limiter} [{caps}]>"
        )


def occupancy(
    spec: GPUSpec,
    threads_per_block: int,
    regs_per_thread: int = 32,
    smem_per_block: int = 0,
) -> Occupancy:
    """Compute achievable occupancy for a launch configuration.

    Raises ValueError for configurations that cannot launch at all
    (too many threads per block, registers, or shared memory) — the same
    configurations for which cudaLaunchKernel returns an error.
    """
    if threads_per_block < 1:
        raise ValueError("threads_per_block must be >= 1")
    if threads_per_block > spec.max_threads_per_sm:
        raise ValueError(
            f"{threads_per_block} threads/block exceeds {spec.name}'s "
            f"per-SM thread limit of {spec.max_threads_per_sm} — launch would fail"
        )
    if regs_per_thread > spec.max_registers_per_thread:
        raise ValueError(
            f"{regs_per_thread} registers/thread exceeds the architectural "
            f"limit of {spec.max_registers_per_thread}"
        )
    if smem_per_block > spec.max_smem_per_block:
        raise ValueError(
            f"{smem_per_block} B of shared memory exceeds {spec.name}'s "
            f"per-block limit of {spec.max_smem_per_block} B — launch would fail"
        )

    warps_per_block = -(-threads_per_block // spec.warp_size)
    regs_per_warp = _ceil_to(regs_per_thread * spec.warp_size, spec.reg_alloc_granularity)
    regs_per_block = regs_per_warp * warps_per_block
    smem_alloc = _ceil_to(smem_per_block, spec.smem_alloc_granularity) if smem_per_block else 0

    limits = {
        "warps": spec.max_warps_per_sm // warps_per_block,
        "blocks": spec.max_blocks_per_sm,
        "registers": spec.registers_per_sm // regs_per_block if regs_per_block else 10**9,
        "shared": spec.smem_per_sm // smem_alloc if smem_alloc else 10**9,
    }
    blocks = min(limits.values())
    # On ties, report the architectural cap (warps/blocks) rather than a
    # resource cap: "you're at the hardware limit" is the more useful message
    # when both are true.
    limiter = next(k for k in ("warps", "blocks", "shared", "registers") if limits[k] == blocks)

    warps = blocks * warps_per_block
    return Occupancy(
        spec_name=spec.name,
        threads_per_block=threads_per_block,
        warps_per_block=warps_per_block,
        blocks_per_sm=blocks,
        warps_per_sm=warps,
        occupancy=warps / spec.max_warps_per_sm,
        limiter=limiter,
        limits=limits,
        regs_per_warp_allocated=regs_per_warp,
        smem_per_block_allocated=smem_alloc,
    )


def occupancy_sweep(
    spec: GPUSpec,
    regs_per_thread: int = 32,
    smem_per_block: int = 0,
    block_sizes: tuple[int, ...] = (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
) -> list[Occupancy]:
    """Occupancy across block sizes — the occupancy calculator's classic chart."""
    return [
        occupancy(spec, t, regs_per_thread, smem_per_block)
        for t in block_sizes
        if t <= spec.max_threads_per_sm
    ]
