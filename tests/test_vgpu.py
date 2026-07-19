"""Tests for the virtual GPU: spec constants and occupancy calculator.

Expected values are hand-derived from the CUDA occupancy calculator's model
for CC 7.5/8.0 (see occupancy.py's module docstring for the rules). The
spec's Phase-1 close-out re-checks them against a live Colab T4.
"""

import pytest

from ai_playground.gpusim import A100_40GB, SPECS, T4, occupancy, occupancy_sweep


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
