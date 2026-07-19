"""Execution traces for the virtual GPU: one schema, every view renders from it.

A trace is a flat list of warp-state intervals plus launch metadata. The
timing engine (timing.py) produces it; renderers, notebooks, and exporters
only ever consume it — the engine never draws.

Chrome trace format export means we get a professional interactive viewer
for free: open the JSON in Perfetto (https://ui.perfetto.dev) or
chrome://tracing, with SMs as processes and warp slots as threads. The
repo's PyTorch profiler wrapper already emits Chrome traces, so readers
learn one viewer for both simulated and real timelines. Cycles are written
into the microsecond field one-to-one (1 cycle = 1 "us") — the viewer's
time axis therefore reads in cycles, which is the honest unit for a
cycle-approximate model.
"""

import json
from dataclasses import dataclass, field

from .occupancy import Occupancy
from .spec import GPUSpec

# Warp interval states
EXEC = "exec"  # issued and computing (includes the coarse ALU charge)
STALL_MEM = "stall_mem"  # waiting on DRAM (latency + bandwidth queueing)
STALL_SMEM = "stall_smem"  # waiting on shared memory (incl. bank conflicts)
STALL_SYNC = "stall_sync"  # waiting at a barrier for block-mates


@dataclass(frozen=True)
class WarpEvent:
    t_start: float  # cycles
    t_end: float
    sm: int
    slot: int  # resident-warp slot on that SM
    block: int  # linear block id (grid-wide)
    warp: int  # warp id within the block
    state: str


@dataclass
class Trace:
    spec: GPUSpec
    occupancy: Occupancy
    events: list[WarpEvent] = field(default_factory=list)
    total_cycles: float = 0.0
    dram_bytes: int = 0  # transaction bytes actually moved (coalescing-aware)

    @property
    def time_s(self) -> float:
        """Wall-clock estimate at boost clock. Cycle-approximate — use for
        ratios between runs, never as a benchmark number."""
        return self.total_cycles / self.spec.boost_clock_hz

    @property
    def achieved_bandwidth(self) -> float:
        """bytes/sec implied by the run — compare against spec.dram_bandwidth."""
        return self.dram_bytes / self.time_s if self.total_cycles else 0.0

    def summary(self) -> dict:
        stall = {}
        for e in self.events:
            stall[e.state] = stall.get(e.state, 0.0) + (e.t_end - e.t_start)
        return {
            "spec": self.spec.name,
            "total_cycles": round(self.total_cycles, 1),
            "occupancy": self.occupancy.occupancy,
            "occupancy_limiter": self.occupancy.limiter,
            "dram_bytes": self.dram_bytes,
            "achieved_GBps": round(self.achieved_bandwidth / 1e9, 1),
            "peak_GBps": round(self.spec.dram_bandwidth / 1e9, 1),
            "warp_cycles_by_state": {k: round(v, 1) for k, v in sorted(stall.items())},
        }

    def to_chrome(self) -> dict:
        """Chrome trace format: SMs as processes, warp slots as threads."""
        out = []
        for sm in sorted({e.sm for e in self.events}):
            out.append(
                {"ph": "M", "name": "process_name", "pid": sm,
                 "args": {"name": f"SM {sm:02d}"}}
            )
        for e in self.events:
            out.append(
                {
                    "name": e.state,
                    "cat": "warp",
                    "ph": "X",
                    "ts": e.t_start,  # cycles, written as "us" (see module docstring)
                    "dur": e.t_end - e.t_start,
                    "pid": e.sm,
                    "tid": e.slot,
                    "args": {"block": e.block, "warp": e.warp},
                }
            )
        return {"traceEvents": out, "displayTimeUnit": "ms",
                "otherData": {"spec": self.spec.name, "unit": "1 ts = 1 GPU cycle"}}

    def to_chrome_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_chrome(), f)
