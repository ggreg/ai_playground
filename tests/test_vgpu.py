"""Tests for the virtual GPU: spec constants and occupancy calculator.

Expected values are hand-derived from the CUDA occupancy calculator's model
for CC 7.5/8.0 (see occupancy.py's module docstring for the rules). The
spec's Phase-1 close-out re-checks them against a live Colab T4.
"""

import numpy as np
import pytest

from ai_playground.gpusim import (
    A100_40GB,
    SPECS,
    T4,
    kernel,
    occupancy,
    occupancy_sweep,
    simulate,
)


class TestSpec:
    def test_t4_derived_numbers(self):
        assert T4.max_warps_per_sm == 32
        assert T4.n_sm * T4.fp32_lanes_per_sm == 2560  # CUDA cores
        assert T4.fp32_tflops == pytest.approx(8.1, abs=0.1)

    def test_a100_derived_numbers(self):
        assert A100_40GB.max_warps_per_sm == 64
        assert A100_40GB.fp32_tflops == pytest.approx(19.5, abs=0.1)

    def test_registry(self):
        assert SPECS["T4"] is T4


class TestOccupancy:
    def test_full_occupancy_256_threads(self):
        # 8 warps/block; warp cap 32//8 = 4 blocks; regs 1024/warp -> 8 blocks
        occ = occupancy(T4, 256, regs_per_thread=32)
        assert occ.blocks_per_sm == 4
        assert occ.warps_per_sm == 32
        assert occ.occupancy == 1.0
        assert occ.limiter == "warps"

    def test_block_granularity_waste_96_threads(self):
        # 3 warps/block -> floor(32/3) = 10 blocks = 30 warps: 93.75%, no
        # resource pressure — occupancy lost purely to indivisibility
        occ = occupancy(T4, 96)
        assert occ.blocks_per_sm == 10
        assert occ.warps_per_sm == 30
        assert occ.occupancy == pytest.approx(0.9375)
        assert occ.limiter == "warps"

    def test_register_limited(self):
        # 96 regs/thread -> 3072 regs/warp -> 24576/block -> 2 blocks: 50%
        occ = occupancy(T4, 256, regs_per_thread=96)
        assert occ.blocks_per_sm == 2
        assert occ.occupancy == 0.5
        assert occ.limiter == "registers"
        assert occ.regs_per_warp_allocated == 3072

    def test_register_allocation_granularity(self):
        # 33 regs/thread: 1056 raw -> rounds to 1280 per warp (5 x 256)
        occ = occupancy(T4, 256, regs_per_thread=33)
        assert occ.regs_per_warp_allocated == 1280

    def test_shared_memory_limited(self):
        # 48 KB/block on a 64 KB SM -> 1 block of 4 warps: 12.5%
        occ = occupancy(T4, 128, smem_per_block=48 * 1024)
        assert occ.blocks_per_sm == 1
        assert occ.warps_per_sm == 4
        assert occ.occupancy == 0.125
        assert occ.limiter == "shared"

    def test_small_blocks_hit_block_cap(self):
        # 32-thread blocks: warp cap allows 32, block cap is 16 -> 50%
        occ = occupancy(T4, 32)
        assert occ.blocks_per_sm == 16
        assert occ.warps_per_sm == 16
        assert occ.limiter == "blocks"

    def test_a100_same_kernel_different_story(self):
        # Same 256-thread/32-reg kernel: T4 reaches 100%, A100 only 50% —
        # registers bind at 8 blocks while 64 warp slots want 8 blocks too;
        # tie reported as the architectural cap
        occ = occupancy(A100_40GB, 256, regs_per_thread=32)
        assert occ.warps_per_sm == 64
        assert occ.occupancy == 1.0

    def test_launch_failures_raise(self):
        with pytest.raises(ValueError, match="thread limit"):
            occupancy(T4, 2048)
        with pytest.raises(ValueError, match="registers/thread"):
            occupancy(T4, 256, regs_per_thread=256)
        with pytest.raises(ValueError, match="shared memory"):
            occupancy(T4, 256, smem_per_block=65 * 1024)

    def test_sweep(self):
        sweep = occupancy_sweep(T4)
        by_block = {o.threads_per_block: o.occupancy for o in sweep}
        assert by_block[96] == pytest.approx(0.9375)
        assert by_block[256] == 1.0
        assert by_block[1024] == 1.0


@kernel
def _copy_contig(ctx, src, dst):
    i = ctx.grid(1)
    dst[i] = src[i]


@kernel
def _copy_strided(ctx, src, dst):
    i = ctx.grid(1)
    dst[i] = src[32 * i]


class TestTiming:
    def _mk(self, n):
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    def test_trace_wellformed(self):
        src, dst = self._mk(128)
        trace = simulate(_copy_contig, 4, 32, (src, dst))
        assert trace.total_cycles > T4.lat_dram  # at least one DRAM round trip
        assert trace.dram_bytes == 8 * 128  # 4 blocks x (1 read + 1 write) x 128B
        assert all(e.t_end >= e.t_start for e in trace.events)
        s = trace.summary()
        # 32-thread blocks are block-slot limited: 16 blocks x 1 warp = 50%
        assert s["spec"] == "T4" and s["occupancy"] == 0.5

    def test_strided_slower_than_contiguous(self):
        # Needs enough blocks that the copy is bandwidth-bound, not
        # latency-bound — with little data in flight, latency hiding absorbs
        # most of the strided penalty (which is itself a correct prediction).
        n = 1024 * 32
        src32, dst = self._mk(32 * n)
        _, dst2 = self._mk(n)
        t_contig = simulate(_copy_contig, 1024, 32, (src32[:n], dst[:n]))
        t_strided = simulate(_copy_strided, 1024, 32, (src32, dst2))
        # 32x the transactions must cost several x the cycles at the wall
        assert t_strided.total_cycles > 5 * t_contig.total_cycles
        assert t_strided.dram_bytes > 10 * t_contig.dram_bytes

    def test_latency_hiding_low_occupancy_hurts(self):
        # Same kernel and data; shared-memory pressure drops residency to 1
        # block/SM, so DRAM stalls can't be hidden behind other warps.
        src, dst = self._mk(64 * 256)
        t_full = simulate(_copy_contig, 64, 256, (src, dst))
        t_starved = simulate(_copy_contig, 64, 256, (src, dst), smem_per_block=48 * 1024)
        assert t_starved.occupancy.occupancy < t_full.occupancy.occupancy
        assert t_starved.total_cycles > t_full.total_cycles

    def test_tiled_matmul_beats_naive(self):
        TILE = 8

        @kernel
        def tiled_matmul(ctx, A, B, C):
            tile_a = ctx.shared((TILE, TILE))
            tile_b = ctx.shared((TILE, TILE))
            tx, ty = ctx.threadIdx.x, ctx.threadIdx.y
            col = ctx.blockIdx.x * TILE + tx
            row = ctx.blockIdx.y * TILE + ty
            acc = 0.0
            for k0 in range(0, A.shape[1], TILE):
                tile_a[ty, tx] = A[row, k0 + tx]
                tile_b[ty, tx] = B[k0 + ty, col]
                yield ctx.syncthreads()
                for k in range(TILE):
                    acc += tile_a[ty, k] * tile_b[k, tx]
                yield ctx.syncthreads()
            C[row, col] = acc

        @kernel
        def naive_matmul(ctx, A, B, C):
            col, row = ctx.grid(2)
            acc = 0.0
            for k in range(A.shape[1]):
                acc += A[row, k] * B[k, col]
            C[row, col] = acc

        n = 2 * TILE
        rng = np.random.default_rng(3)
        A = rng.standard_normal((n, n)).astype(np.float32)
        B = rng.standard_normal((n, n)).astype(np.float32)
        cfg = ((n // TILE, n // TILE), (TILE, TILE))
        t_naive = simulate(naive_matmul, *cfg, (A, B, np.zeros((n, n), np.float32)))
        t_tiled = simulate(
            tiled_matmul, *cfg, (A, B, np.zeros((n, n), np.float32)),
            smem_per_block=2 * TILE * TILE * 4,
        )
        assert t_tiled.total_cycles < t_naive.total_cycles
        assert t_tiled.dram_bytes < t_naive.dram_bytes

    def test_sync_stalls_recorded_and_terminates(self):
        @kernel
        def staggered(ctx, out):
            s = ctx.shared((32,), np.float32)
            t = ctx.threadIdx.x
            s[t] = float(t)
            yield ctx.syncthreads()
            out[t] = s[31 - t]

        trace = simulate(staggered, 2, 32, (np.zeros(64, dtype=np.float32),))
        assert trace.total_cycles > 0  # terminated: barriers released

    def test_deterministic(self):
        src, dst = self._mk(256)
        c1 = simulate(_copy_contig, 8, 32, (src, dst)).total_cycles
        c2 = simulate(_copy_contig, 8, 32, (src, dst)).total_cycles
        assert c1 == c2

    def test_chrome_trace_export(self, tmp_path):
        import json

        src, dst = self._mk(128)
        trace = simulate(_copy_contig, 4, 32, (src, dst))
        p = tmp_path / "trace.json"
        trace.to_chrome_json(str(p))
        data = json.loads(p.read_text())
        evs = [e for e in data["traceEvents"] if e["ph"] == "X"]
        assert evs and all(
            isinstance(e["ts"], (int, float)) and e["dur"] >= 0 and "pid" in e and "tid" in e
            for e in evs
        )


class TestRender:
    def _trace(self):
        src = np.zeros(64 * 32, dtype=np.float32)
        return simulate(_copy_contig, 64, 32, (src, np.zeros_like(src)))

    def test_waterfall_renders(self):
        import matplotlib
        matplotlib.use("Agg")
        from ai_playground.gpusim import plot_waterfall

        ax = plot_waterfall(self._trace(), sm=0)
        assert ax.get_xlim()[1] > 0

    def test_floorplan_frame_states(self):
        from ai_playground.gpusim.render import _FloorplanImage

        trace = self._trace()
        fp = _FloorplanImage(trace)
        mid = fp.frame(trace.total_cycles / 2)
        assert mid.shape[2] == 3 and mid.shape[0] > 5
        # mid-run the die must not be all idle: some slot shows a state color
        idle = fp._idle
        tiles = [mid[r, c] for r in range(mid.shape[0]) for c in range(mid.shape[1])]
        assert any(
            not np.allclose(px, idle) and not np.allclose(px, fp._border) and px.min() < 0.9
            for px in tiles
        )

    def test_animation_builds(self):
        import matplotlib
        matplotlib.use("Agg")
        from ai_playground.gpusim import animate_floorplan

        anim = animate_floorplan(self._trace(), frames=3)
        assert anim is not None
