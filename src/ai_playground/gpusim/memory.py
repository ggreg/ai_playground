"""Instrumented memory for the SIMT simulator: every access, logged.

This is the payoff of simulating instead of running: real hardware punishes
bad access patterns with silence and slowness, while the simulator can show
you the pattern itself. Two analyses matter for the concepts chapters:

- **Coalescing** (global memory): a warp's 32 loads are serviced in 128-byte
  transactions. If the lanes read 32 consecutive fp32 values, that is one
  transaction; a stride of 32 elements puts every lane in its own segment —
  32 transactions for the same useful bytes, i.e. ~1/32 of the effective
  bandwidth. `coalescing_report` counts exactly this.
- **Bank conflicts** (shared memory): shared memory has 32 banks of 4 bytes;
  lanes hitting distinct addresses in the same bank serialize. Reading
  `shared[lane]` is conflict-free, `shared[2 * lane]` (fp32) is 2-way.
  All lanes reading the *same* address broadcast for free.

Lockstep assumption: accesses are grouped across lanes by each thread's
per-thread sequence number (its k-th access lines up with its warp-mates'
k-th). True for the branch-free inner loops these analyses target; divergent
kernels shift lanes' sequences and the grouping degrades gracefully.

Only scalar element access is supported — one element per thread per access,
which is exactly the SIMT contract. Slicing raises with an explanation
rather than silently logging something misleading.
"""

from dataclasses import dataclass, field

import numpy as np

SEGMENT_BYTES = 128  # global-memory transaction size on every modern NVIDIA GPU
N_BANKS = 32
BANK_BYTES = 4


@dataclass(frozen=True)
class Access:
    block: int  # linear block id within the grid
    thread: int  # linear id within the block
    warp: int
    seq: int  # per-thread access counter — aligns lanes under lockstep
    space: str  # 'global' | 'shared' | 'sync'
    array: str
    flat: int  # flattened element index (barrier count for 'sync')
    itemsize: int
    kind: str  # 'r' | 'w' | 's'


@dataclass
class AccessLog:
    accesses: list[Access] = field(default_factory=list)
    _block: int = 0
    _thread: int = 0
    _warp: int = 0
    _seq: dict[tuple[int, int], int] = field(default_factory=dict)

    def set_thread(self, thread: int, warp: int, block: int = 0) -> None:
        self._thread, self._warp, self._block = thread, warp, block

    def record(self, space: str, array: str, flat: int, itemsize: int, kind: str) -> None:
        key = (self._block, self._thread)
        seq = self._seq.get(key, 0)
        self._seq[key] = seq + 1
        self.accesses.append(
            Access(self._block, self._thread, self._warp, seq, space, array, flat, itemsize, kind)
        )


class _InstrumentedArray:
    """NumPy-backed array that logs scalar accesses; base for global/shared."""

    space = "?"

    def __init__(self, data: np.ndarray, name: str, log: AccessLog):
        self.data = data
        self.name = name
        self._log = log

    # Mirror the ndarray surface kernels legitimately need
    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def _flat(self, idx) -> int:
        try:
            if self.data.ndim == 1:
                return int(np.ravel_multi_index((idx,), self.data.shape))
            return int(np.ravel_multi_index(idx, self.data.shape))
        except (TypeError, ValueError):
            raise TypeError(
                f"{self.name}[{idx!r}]: simulated kernels access one element per "
                "thread (ints only) — that is the SIMT contract. Slices and "
                "fancy indexing belong outside the kernel."
            ) from None

    def __getitem__(self, idx):
        flat = self._flat(idx)
        self._log.record(self.space, self.name, flat, self.data.itemsize, "r")
        return self.data[idx]

    def __setitem__(self, idx, value):
        flat = self._flat(idx)
        self._log.record(self.space, self.name, flat, self.data.itemsize, "w")
        self.data[idx] = value

    def __repr__(self):
        return f"<{type(self).__name__} {self.name} {self.data.shape} {self.data.dtype}>"


class GlobalArray(_InstrumentedArray):
    """Device-global memory: what kernel arguments become at launch."""

    space = "global"


class SharedArray(_InstrumentedArray):
    """Block-shared memory: allocated via ctx.shared(), lives per block."""

    space = "shared"


@dataclass(frozen=True)
class WarpAccessGroup:
    """One warp-level access: the lanes' simultaneous loads/stores."""

    warp: int
    seq: int
    array: str
    kind: str
    lanes: int
    transactions: int  # 128B segments touched ('global') or conflict degree ('shared')
    ideal: int


def _grouped(log: AccessLog, space: str):
    groups: dict[tuple, list[Access]] = {}
    for a in log.accesses:
        if a.space == space:
            groups.setdefault((a.block, a.warp, a.seq, a.array, a.kind), []).append(a)
    return groups


def coalescing_report(log: AccessLog) -> list[WarpAccessGroup]:
    """Count 128-byte transactions per warp-level global access.

    ideal = the minimum transactions the lanes' byte volume could need;
    transactions == ideal is perfectly coalesced, 32x ideal is worst-case.
    """
    out = []
    for (_block, warp, seq, array, kind), accs in sorted(_grouped(log, "global").items()):
        segments = {a.flat * a.itemsize // SEGMENT_BYTES for a in accs}
        nbytes = sum(a.itemsize for a in accs)
        ideal = max(1, -(-nbytes // SEGMENT_BYTES))  # ceil division
        out.append(WarpAccessGroup(warp, seq, array, kind, len(accs), len(segments), ideal))
    return out


def bank_conflict_report(log: AccessLog) -> list[WarpAccessGroup]:
    """Conflict degree per warp-level shared access (1 = conflict-free).

    Distinct addresses in the same bank serialize; identical addresses
    broadcast, so duplicates are collapsed before counting.
    """
    out = []
    for (_block, warp, seq, array, kind), accs in sorted(_grouped(log, "shared").items()):
        addrs = {a.flat * a.itemsize for a in accs}  # dedupe: same addr = broadcast
        per_bank: dict[int, int] = {}
        for addr in addrs:
            bank = (addr // BANK_BYTES) % N_BANKS
            per_bank[bank] = per_bank.get(bank, 0) + 1
        degree = max(per_bank.values(), default=1)
        out.append(WarpAccessGroup(warp, seq, array, kind, len(accs), degree, 1))
    return out


def total_transactions(report: list[WarpAccessGroup]) -> tuple[int, int]:
    """(actual, ideal) summed over a report — the headline coalescing number."""
    return sum(g.transactions for g in report), sum(g.ideal for g in report)
