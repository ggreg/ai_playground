"""Event-driven timing engine: from access log to cycle-approximate trace.

Pipeline: run the kernel *functionally* (core.py) to get correctness plus
the full access log, compress each warp's log into an op stream (one op per
lockstep warp access, priced by the transaction math in memory.py), then
replay the streams through a model of the chip:

- Blocks become resident on SMs up to the occupancy limit (occupancy.py);
  the rest queue and are assigned as blocks retire — this is how "waves"
  of blocks emerge on the floorplan view.
- Each SM issues ready warps in ready-order; issue throughput is
  1/warp_schedulers_per_sm cycles per op, so scheduler pressure exists but
  is rarely the bottleneck (as on real hardware).
- A DRAM op parks its warp for `lat_dram` cycles *and* occupies the
  chip-wide DRAM pipe for `bytes / bytes_per_cycle` — a single token bucket
  shared by all SMs. Coalescing sets `bytes`: a coalesced warp read costs
  one 128 B transaction, a fully strided one costs 32. Bandwidth contention
  and the memory wall emerge from this one queue.
- A shared-memory op parks for `lat_smem x conflict_degree`.
- `syncthreads` parks the warp until every warp of its block arrives.
- A coarse `lat_alu` per op stands in for the arithmetic between accesses.

Why latency hiding emerges rather than being programmed in: while one warp
is parked on DRAM, the SM's other resident warps keep issuing — so with 32
resident warps a ~450-cycle stall costs almost nothing, and with 4 it
stalls the whole SM. That is the occupancy lesson, produced by the model
instead of asserted by it.

Everything here is cycle-approximate (see docs/VIRTUAL_GPU_SPEC.md,
"Refusals"): no caches, no ILP, no DRAM row effects. Predictions are
trends and ratios, never milliseconds.
"""

import heapq
from dataclasses import dataclass, field

from .core import Kernel, _dim3
from .memory import SEGMENT_BYTES, AccessLog
from .occupancy import occupancy
from .spec import T4, GPUSpec
from .trace import EXEC, STALL_MEM, STALL_SMEM, STALL_SYNC, Trace, WarpEvent


@dataclass(frozen=True)
class _Op:
    kind: str  # 'dram' | 'smem' | 'sync'
    cost: int  # bytes for dram, conflict degree for smem, barrier id for sync


def _build_op_streams(log: AccessLog) -> dict[int, dict[int, list[_Op]]]:
    """Compress the access log into per-block, per-warp op streams.

    Lockstep assumption (as in memory.py): a warp's k-th access happens
    together across its lanes, so accesses grouped by (block, warp, seq)
    form one warp-level op. Ops are priced here so the engine never needs
    the raw log.
    """
    groups: dict[tuple[int, int], dict[int, list]] = {}
    for a in log.accesses:
        groups.setdefault((a.block, a.warp), {}).setdefault(a.seq, []).append(a)

    streams: dict[int, dict[int, list[_Op]]] = {}
    for (block, warp), by_seq in groups.items():
        ops = []
        for seq in sorted(by_seq):
            accs = by_seq[seq]
            space = accs[0].space
            if space == "global":
                segments = {a.flat * a.itemsize // SEGMENT_BYTES for a in accs}
                ops.append(_Op("dram", len(segments) * SEGMENT_BYTES))
            elif space == "shared":
                addrs = {a.flat * a.itemsize for a in accs}
                banks: dict[int, int] = {}
                for addr in addrs:
                    b = (addr // 4) % 32
                    banks[b] = banks.get(b, 0) + 1
                ops.append(_Op("smem", max(banks.values(), default=1)))
            else:  # sync
                ops.append(_Op("sync", accs[0].flat))
        streams.setdefault(block, {})[warp] = ops
    return streams


@dataclass
class _WarpState:
    block: int
    warp: int
    sm: int
    slot: int
    ops: list[_Op]
    i: int = 0


@dataclass
class _BlockSync:
    expected: int
    arrived: list = field(default_factory=list)  # (warp_state, arrival_cycle)


def simulate(
    kern: Kernel,
    grid,
    block,
    args: tuple,
    spec: GPUSpec = T4,
    regs_per_thread: int = 32,
    smem_per_block: int = 0,
) -> Trace:
    """Run a kernel functionally, then predict its timing on `spec`.

    regs_per_thread and smem_per_block feed the occupancy calculation —
    Python kernels have no register allocator, so state what the real
    kernel would use (nvcc's --ptxas-options=-v prints it).
    """
    b = _dim3(block)
    report = kern[grid, block](*args)  # functional pass: correctness + log
    occ = occupancy(spec, b.x * b.y * b.z, regs_per_thread, smem_per_block)
    streams = _build_op_streams(report.log)
    trace = Trace(spec=spec, occupancy=occ)

    bytes_per_cycle = spec.dram_bandwidth / spec.boost_clock_hz
    issue_interval = 1.0 / spec.warp_schedulers_per_sm

    # --- chip state ---
    dram_free = 0.0
    sm_issue_free = [0.0] * spec.n_sm
    sm_resident = [0] * spec.n_sm
    slot_free: list[list[int]] = [list(range(occ.blocks_per_sm)) for _ in range(spec.n_sm)]
    syncs: dict[int, _BlockSync] = {}
    ready: list[tuple[float, int, _WarpState]] = []  # (cycle, tiebreak, warp)
    tick = 0

    pending = sorted(streams)  # blocks not yet resident, in launch order

    def place_block(blk: int, sm: int, at: float) -> None:
        nonlocal tick
        slot = slot_free[sm].pop(0)
        warps = streams[blk]
        syncs[blk] = _BlockSync(expected=len(warps))
        sm_resident[sm] += 1
        for warp_id in sorted(warps):
            w = _WarpState(blk, warp_id, sm, slot * 64 + warp_id, warps[warp_id])
            tick += 1
            heapq.heappush(ready, (at, tick, w))

    # Initial wave: fill every SM to its residency limit, round-robin
    sm_rr = 0
    while pending and any(r < occ.blocks_per_sm for r in sm_resident):
        if sm_resident[sm_rr] < occ.blocks_per_sm:
            place_block(pending.pop(0), sm_rr, 0.0)
        sm_rr = (sm_rr + 1) % spec.n_sm

    def retire_warp(w: _WarpState, at: float) -> None:
        nonlocal tick
        sync = syncs[w.block]
        sync.expected -= 1
        if sync.expected == 0 and not sync.arrived:  # whole block done
            sm_resident[w.sm] -= 1
            slot_free[w.sm].append(w.slot // 64)
            if pending:
                place_block(pending.pop(0), w.sm, at)

    while ready:
        t, _, w = heapq.heappop(ready)
        if w.i >= len(w.ops):
            retire_warp(w, t)
            trace.total_cycles = max(trace.total_cycles, t)
            continue
        op = w.ops[w.i]

        issue = max(t, sm_issue_free[w.sm])
        sm_issue_free[w.sm] = issue + issue_interval

        if op.kind == "dram":
            start = max(issue, dram_free)  # queue on the shared DRAM pipe
            dram_free = start + op.cost / bytes_per_cycle
            done = dram_free + spec.lat_dram
            trace.dram_bytes += op.cost
            trace.events.append(WarpEvent(issue, done, w.sm, w.slot, w.block, w.warp, STALL_MEM))
        elif op.kind == "smem":
            done = issue + spec.lat_smem * op.cost
            trace.events.append(WarpEvent(issue, done, w.sm, w.slot, w.block, w.warp, STALL_SMEM))
        else:  # sync: park until the whole block arrives
            sync = syncs[w.block]
            sync.arrived.append((w, issue))
            if len(sync.arrived) == sync.expected:
                release = max(at for _, at in sync.arrived)
                for warp_state, arrived_at in sync.arrived:
                    if release > arrived_at:
                        trace.events.append(
                            WarpEvent(arrived_at, release, warp_state.sm, warp_state.slot,
                                      warp_state.block, warp_state.warp, STALL_SYNC)
                        )
                    warp_state.i += 1
                    tick += 1
                    heapq.heappush(ready, (release + spec.lat_alu, tick, warp_state))
                sync.arrived = []
            continue  # re-queued (or parked) above; skip the common tail

        w.i += 1
        next_ready = done + spec.lat_alu
        trace.events.append(WarpEvent(done, next_ready, w.sm, w.slot, w.block, w.warp, EXEC))
        tick += 1
        heapq.heappush(ready, (next_ready, tick, w))

    return trace
