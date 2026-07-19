"""A pure-Python SIMT simulator: CUDA's execution model in ~200 lines.

Why simulate: the CUDA concepts that matter — grid/block/thread indexing,
shared memory, barriers, warps, coalescing — are *semantics*, and semantics
don't need a GPU. This module runs kernels written in numba.cuda style on
plain NumPy so they work anywhere Python does (CI, a laptop, Pyodide in the
browser), while logging every memory access so notebooks can visualize the
patterns that real hardware punishes or rewards.

The one hard problem is `syncthreads()`: a barrier requires every thread in
the block to arrive before any proceeds, so threads must be suspendable.
Real CUDA does this in hardware; numba's simulator uses one OS thread per
CUDA thread (unavailable in WebAssembly). Here, kernels that use barriers
are written as generators and `yield ctx.syncthreads()` — the scheduler
advances all threads to the barrier, checks they arrived at the *same* one,
then resumes them. The visible `yield` is a feature: it marks exactly where
execution suspends.

What this deliberately is not: a performance model. It simulates what
happens, not how fast. Timing a simulated kernel teaches nothing; counting
its memory transactions (see `memory.py`) teaches a lot.

Known simplification: within a scheduling round threads run sequentially to
completion (or to the next barrier), so data races between barriers are
invisible rather than nondeterministic. Well-synchronized kernels behave
identically to hardware; racy ones silently get one legal interleaving.
"""

import inspect
from typing import NamedTuple

import numpy as np

from .memory import AccessLog, GlobalArray, SharedArray

WARP_SIZE = 32
MAX_THREADS_PER_BLOCK = 1024  # CUDA hardware limit, enforced for realism


class Dim3(NamedTuple):
    """CUDA dim3: x varies fastest, matching hardware thread linearization."""

    x: int = 1
    y: int = 1
    z: int = 1


def _dim3(v: int | tuple[int, ...]) -> Dim3:
    if isinstance(v, int):
        return Dim3(v)
    return Dim3(*v)


class BarrierDivergenceError(RuntimeError):
    """Raised when threads of a block disagree about reaching a barrier.

    On real hardware this is undefined behavior (pre-Volta GPUs hang). The
    simulator turns it into a loud error because it is almost always a bug:
    a `syncthreads()` inside a divergent branch, or a thread returning early
    while its block-mates still expect it at the barrier.
    """


class _Barrier(NamedTuple):
    count: int


class _BlockState:
    """Per-block shared memory: allocations live exactly as long as the block."""

    def __init__(self, log: AccessLog):
        self.shared_allocs: list[SharedArray] = []
        self.log = log


class Ctx:
    """Per-thread view of the launch: indices, shared memory, and barriers.

    One explicit object instead of numba's module-level globals — every
    thread's identity is visible in the kernel signature.
    """

    def __init__(
        self,
        block: _BlockState,
        linear: int,
        threadIdx: Dim3,
        blockIdx: Dim3,
        blockDim: Dim3,
        gridDim: Dim3,
    ):
        self.threadIdx = threadIdx
        self.blockIdx = blockIdx
        self.blockDim = blockDim
        self.gridDim = gridDim
        self._block = block
        self._linear = linear  # linear id within the block; warp = linear // 32
        self._barriers = 0
        self._alloc_i = 0

    @property
    def warp(self) -> int:
        return self._linear // WARP_SIZE

    @property
    def lane(self) -> int:
        return self._linear % WARP_SIZE

    def grid(self, ndim: int) -> int | tuple[int, ...]:
        """Global thread index, numba-style: cuda.grid(1) -> x, grid(2) -> (x, y)."""
        gx = self.blockIdx.x * self.blockDim.x + self.threadIdx.x
        gy = self.blockIdx.y * self.blockDim.y + self.threadIdx.y
        gz = self.blockIdx.z * self.blockDim.z + self.threadIdx.z
        return (gx, (gx, gy), (gx, gy, gz))[ndim - 1] if ndim in (1, 2, 3) else _bad_ndim(ndim)

    def shared(self, shape: int | tuple[int, ...], dtype=np.float32) -> SharedArray:
        """Allocate (or join) a block-shared array.

        Every thread executes this call, but the block must end up with one
        array — so the first thread to reach the i-th allocation creates it
        and the rest join it, with shape/dtype checked for agreement.
        """
        i = self._alloc_i
        self._alloc_i += 1
        allocs = self._block.shared_allocs
        if i == len(allocs):
            data = np.zeros(shape, dtype=dtype)
            allocs.append(SharedArray(data, f"shared{i}", self._block.log))
        arr = allocs[i]
        if arr.data.shape != np.zeros(shape, dtype=dtype).shape or arr.data.dtype != dtype:
            raise ValueError(
                f"shared allocation #{i}: threads disagree on shape/dtype "
                f"({arr.data.shape}/{arr.data.dtype} vs {shape}/{dtype})"
            )
        return arr

    def syncthreads(self) -> _Barrier:
        """Barrier token — generator kernels must `yield ctx.syncthreads()`."""
        self._barriers += 1
        # Logged so the timing model (gpusim.timing) knows where warps wait
        self._block.log.record("sync", "", self._barriers, 0, "s")
        return _Barrier(self._barriers)


def _bad_ndim(ndim):
    raise ValueError(f"grid(ndim) takes 1, 2 or 3, got {ndim}")


class LaunchReport:
    """What a launch returns: the access log plus launch geometry.

    Real CUDA launches return nothing; returning the report is a deliberate
    break from fidelity because inspecting the access pattern is the point
    of simulating at all.
    """

    def __init__(self, log: AccessLog, grid: Dim3, block: Dim3):
        self.log = log
        self.grid = grid
        self.block = block

    def __repr__(self):
        n = len(self.log.accesses)
        threads = self.grid.x * self.grid.y * self.grid.z * (
            self.block.x * self.block.y * self.block.z
        )
        return f"<LaunchReport grid={tuple(self.grid)} block={tuple(self.block)} " \
               f"threads={threads} logged_accesses={n}>"


class _Launcher:
    def __init__(self, fn, grid: Dim3, block: Dim3):
        self.fn = fn
        self.grid = grid
        self.block = block
        threads = block.x * block.y * block.z
        if threads > MAX_THREADS_PER_BLOCK:
            raise ValueError(f"{threads} threads per block exceeds CUDA's limit of 1024")

    def __call__(self, *args) -> LaunchReport:
        log = AccessLog()
        params = list(inspect.signature(self.fn).parameters)[1:]  # drop ctx
        wrapped = [
            GlobalArray(a, params[i] if i < len(params) else f"arg{i}", log)
            if isinstance(a, np.ndarray)
            else a
            for i, a in enumerate(args)
        ]
        is_gen = inspect.isgeneratorfunction(self.fn)
        g, b = self.grid, self.block
        for bz in range(g.z):
            for by in range(g.y):
                for bx in range(g.x):
                    linear = bz * (g.y * g.x) + by * g.x + bx
                    self._run_block(Dim3(bx, by, bz), linear, wrapped, log, is_gen)
        return LaunchReport(log, g, b)

    def _make_ctxs(self, block_idx: Dim3, log: AccessLog) -> list[Ctx]:
        state = _BlockState(log)
        b = self.block
        return [
            Ctx(state, tz * (b.y * b.x) + ty * b.x + tx,
                Dim3(tx, ty, tz), block_idx, b, self.grid)
            for tz in range(b.z) for ty in range(b.y) for tx in range(b.x)
        ]

    def _run_block(
        self, block_idx: Dim3, block_linear: int, args: list, log: AccessLog, is_gen: bool
    ) -> None:
        ctxs = self._make_ctxs(block_idx, log)
        if not is_gen:
            for ctx in ctxs:
                log.set_thread(ctx._linear, ctx.warp, block_linear)
                self.fn(ctx, *args)
            return

        # Cooperative scheduling: advance every live thread to its next
        # barrier (or completion), then check the block agrees.
        gens = [(ctx, self.fn(ctx, *args)) for ctx in ctxs]
        alive = gens
        while alive:
            waiting, finished = [], []
            for ctx, gen in alive:
                log.set_thread(ctx._linear, ctx.warp, block_linear)
                try:
                    y = next(gen)
                except StopIteration:
                    finished.append(ctx)
                    continue
                if not isinstance(y, _Barrier):
                    raise TypeError(
                        f"kernel yielded {y!r}; generators may only "
                        "`yield ctx.syncthreads()`"
                    )
                waiting.append((ctx, gen, y))
            if waiting and finished:
                raise BarrierDivergenceError(
                    f"threads {[c._linear for c in finished]} exited while "
                    f"{len(waiting)} thread(s) wait at a barrier — on real "
                    "hardware this hangs or is undefined behavior"
                )
            counts = {y.count for _, _, y in waiting}
            if len(counts) > 1:
                raise BarrierDivergenceError(
                    f"threads reached different barriers (counts {sorted(counts)}) — "
                    "is a syncthreads() inside a divergent branch?"
                )
            alive = [(ctx, gen) for ctx, gen, _ in waiting]


class Kernel:
    """Wraps a kernel function; launch with CUDA-style `k[grid, block](args)`."""

    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

    def __getitem__(self, cfg) -> _Launcher:
        if not (isinstance(cfg, tuple) and len(cfg) == 2):
            raise TypeError("launch config is kernel[grid, block] — two entries")
        grid, block = cfg
        return _Launcher(self.fn, _dim3(grid), _dim3(block))

    def __call__(self, *args, **kwargs):
        raise TypeError(
            f"kernels need a launch configuration: "
            f"{self.__name__}[grid, block]({', '.join('…' for _ in args)})"
        )


def kernel(fn) -> Kernel:
    """Decorator: turn a function into a launchable simulated GPU kernel."""
    return Kernel(fn)
