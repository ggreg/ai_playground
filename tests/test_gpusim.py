"""Tests for the pure-Python SIMT simulator."""

import numpy as np
import pytest

from ai_playground.gpusim import (
    BarrierDivergenceError,
    bank_conflict_report,
    coalescing_report,
    kernel,
    total_transactions,
)

TILE = 8


@kernel
def vector_add(ctx, a, b, c):
    i = ctx.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]


@kernel
def matrix_scale(ctx, x, out, alpha):
    col, row = ctx.grid(2)
    if row < x.shape[0] and col < x.shape[1]:
        out[row, col] = alpha * x[row, col]


@kernel
def tiled_matmul(ctx, A, B, C):
    """Classic shared-memory tiled matmul; assumes square, TILE-divisible."""
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


class TestExecution:
    def test_vector_add_matches_numpy(self):
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal(200), rng.standard_normal(200)
        c = np.zeros(200)
        vector_add[7, 32](a, b, c)  # 224 threads, guard trims the excess
        np.testing.assert_allclose(c, a + b)

    def test_2d_grid(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((13, 17))
        out = np.zeros_like(x)
        matrix_scale[(3, 2), (8, 8)](x, out, 2.5)
        np.testing.assert_allclose(out, 2.5 * x)

    def test_tiled_matmul_matches_numpy(self):
        rng = np.random.default_rng(2)
        n = 2 * TILE
        A = rng.standard_normal((n, n)).astype(np.float32)
        B = rng.standard_normal((n, n)).astype(np.float32)
        C = np.zeros((n, n), dtype=np.float32)
        tiled_matmul[(n // TILE, n // TILE), (TILE, TILE)](A, B, C)
        np.testing.assert_allclose(C, A @ B, rtol=1e-4)

    def test_launch_requires_config(self):
        with pytest.raises(TypeError, match="launch configuration"):
            vector_add(np.zeros(4), np.zeros(4), np.zeros(4))

    def test_block_size_limit(self):
        with pytest.raises(ValueError, match="1024"):
            vector_add[1, 2048]

    def test_slice_indexing_rejected(self):
        @kernel
        def bad(ctx, a):
            _ = a[0:2]

        with pytest.raises(TypeError, match="one element per\nthread|one element per thread"):
            bad[1, 1](np.zeros(4))


class TestBarriers:
    def test_early_exit_divergence_detected(self):
        @kernel
        def early_exit(ctx, a):
            if ctx.threadIdx.x == 0:
                return
            a[ctx.threadIdx.x] = 1.0
            yield ctx.syncthreads()

        with pytest.raises(BarrierDivergenceError, match="exited while"):
            early_exit[1, 4](np.zeros(4))

    def test_divergent_barrier_detected(self):
        @kernel
        def divergent(ctx, a):
            if ctx.threadIdx.x < 2:
                yield ctx.syncthreads()
            yield ctx.syncthreads()

        with pytest.raises(BarrierDivergenceError):
            divergent[1, 4](np.zeros(4))

    def test_shared_shape_disagreement(self):
        @kernel
        def bad_shared(ctx, a):
            _ = ctx.shared((ctx.threadIdx.x + 1,))
            yield ctx.syncthreads()

        with pytest.raises(ValueError, match="disagree"):
            bad_shared[1, 2](np.zeros(4))


class TestMemoryAnalysis:
    def test_contiguous_read_is_one_transaction(self):
        @kernel
        def read_contig(ctx, a, out):
            i = ctx.grid(1)
            out[i] = a[i]

        a = np.zeros(32, dtype=np.float32)
        report = read_contig[1, 32](a, np.zeros(32, dtype=np.float32))
        reads = [g for g in coalescing_report(report.log) if g.array == "a"]
        assert len(reads) == 1
        # 32 lanes x 4B = 128B = exactly one segment
        assert reads[0].transactions == 1 and reads[0].ideal == 1

    def test_strided_read_is_32_transactions(self):
        @kernel
        def read_strided(ctx, a, out):
            i = ctx.grid(1)
            out[i] = a[32 * i]  # 128B apart: every lane its own segment

        a = np.zeros(32 * 32, dtype=np.float32)
        report = read_strided[1, 32](a, np.zeros(32, dtype=np.float32))
        reads = [g for g in coalescing_report(report.log) if g.array == "a"]
        assert reads[0].transactions == 32 and reads[0].ideal == 1

    def test_total_transactions_summary(self):
        @kernel
        def copy(ctx, a, out):
            i = ctx.grid(1)
            out[i] = a[i]

        n = 128
        report = copy[4, 32](np.zeros(n, np.float32), np.zeros(n, np.float32))
        actual, ideal = total_transactions(coalescing_report(report.log))
        assert actual == ideal  # perfectly coalesced copy

    def test_bank_conflicts(self):
        @kernel
        def shared_patterns(ctx, out):
            s = ctx.shared((64,), np.float32)
            lane = ctx.threadIdx.x
            s[lane] = 0.0  # write: conflict-free
            out[lane] = s[2 * lane % 64]  # read stride 2: 2-way conflict
            yield ctx.syncthreads()

        report = shared_patterns[1, 32](np.zeros(32, np.float32))
        groups = bank_conflict_report(report.log)
        by_kind = {g.kind: g for g in groups}
        assert by_kind["w"].transactions == 1
        assert by_kind["r"].transactions == 2

    def test_broadcast_is_free(self):
        @kernel
        def broadcast(ctx, out):
            s = ctx.shared((4,), np.float32)
            s[ctx.threadIdx.x % 4] = 1.0
            yield ctx.syncthreads()
            out[ctx.threadIdx.x] = s[0]  # all lanes, same address

        report = broadcast[1, 32](np.zeros(32, np.float32))
        reads = [g for g in bank_conflict_report(report.log) if g.kind == "r"]
        assert reads[0].transactions == 1
