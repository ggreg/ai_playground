"""Renderers for virtual-GPU traces: die floorplan, warp waterfall, animation.

Design rule (docs/VIRTUAL_GPU_SPEC.md): the engine never draws and the
renderers never simulate — everything here consumes a Trace.

The floorplan is built as a small RGB image (one pixel per warp slot,
bordered tiles per SM) rather than thousands of matplotlib patches: an
animation redraws the image with `set_data`, which keeps a 40-frame
die animation at seconds, not minutes, of render time.

Colors are the house plot palette by *state role*:
green = executing, orange = stalled on DRAM, yellow = stalled on shared
memory, violet = waiting at a barrier, light gray = no resident warp.
A memory-bound kernel therefore literally turns the die orange.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from .trace import EXEC, STALL_MEM, STALL_SMEM, STALL_SYNC, Trace

STATE_COLORS = {
    EXEC: "#008300",
    STALL_MEM: "#eb6834",
    STALL_SMEM: "#eda100",
    STALL_SYNC: "#4a3aa7",
}
IDLE_COLOR = "#e1e0d9"
BORDER_COLOR = "#b6b4ad"
BW_COLOR = "#eb6834"


def _events_by_slot(trace: Trace) -> dict[tuple[int, int], list]:
    out: dict[tuple[int, int], list] = {}
    for e in trace.events:
        out.setdefault((e.sm, e.slot), []).append(e)
    for evs in out.values():
        evs.sort(key=lambda e: e.t_start)
    return out


def plot_waterfall(trace: Trace, sm: int = 0, ax=None):
    """One SM zoomed in: warp slots as rows, cycles as x, state as color.

    This is where latency hiding is visible: at low occupancy each row is
    mostly orange (stalled) with nothing to fill the gaps; at high
    occupancy the green execution segments of other rows tile over them.
    """
    by_slot = _events_by_slot(trace)
    slots = sorted(s for (m, s) in by_slot if m == sm)
    if not slots:
        raise ValueError(f"no events on SM {sm}")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, min(7.0, 0.35 * len(slots) + 1.2)))
    for row, slot in enumerate(slots):
        for e in by_slot[(sm, slot)]:
            ax.broken_barh(
                [(e.t_start, e.t_end - e.t_start)], (row - 0.4, 0.8),
                facecolors=STATE_COLORS[e.state], linewidth=0,
            )
    ax.set_xlim(0, trace.total_cycles * 1.02)
    ax.set_ylim(-0.6, len(slots) - 0.4)
    ax.set_xlabel("cycle")
    ax.set_ylabel(f"resident warp (SM {sm})")
    ax.set_title(f"{trace.spec.name} warp timeline — SM {sm}")
    ax.legend(
        handles=[Patch(color=c, label=s) for s, c in STATE_COLORS.items()],
        loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8,
    )
    return ax


def _bw_utilization(trace: Trace, t0: float, t1: float) -> float:
    """Fraction of [t0, t1] the DRAM pipe was busy."""
    if t1 <= t0:
        return 0.0
    busy = sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in trace.dram_busy)
    return min(1.0, busy / (t1 - t0))


class _FloorplanImage:
    """Maps (trace, cycle) -> RGB array: bordered SM tiles + a DRAM gauge."""

    def __init__(self, trace: Trace, ncols: int = 8, slot_cols: int = 4):
        self.trace = trace
        self.by_slot = _events_by_slot(trace)
        self.slot_index: dict[tuple[int, int], int] = {}
        per_sm: dict[int, list[int]] = {}
        for sm, slot in self.by_slot:
            per_sm.setdefault(sm, []).append(slot)
        for sm, slots in per_sm.items():
            for i, slot in enumerate(sorted(slots)):
                self.slot_index[(sm, slot)] = i
        self.n_sm = trace.spec.n_sm
        self.ncols = ncols
        self.nrows = -(-self.n_sm // ncols)
        max_slots = max((len(s) for s in per_sm.values()), default=1)
        self.slot_cols = slot_cols
        self.slot_rows = max(1, -(-max_slots // slot_cols))
        self.tile_h = self.slot_rows + 1  # +1 border
        self.tile_w = self.slot_cols + 1
        self.h = self.nrows * self.tile_h + 1
        self.w = self.ncols * self.tile_w + 1 + 3  # +3: gap + 2-wide DRAM gauge
        self._idle = np.array(to_rgb(IDLE_COLOR))
        self._border = np.array(to_rgb(BORDER_COLOR))
        self._state_rgb = {s: np.array(to_rgb(c)) for s, c in STATE_COLORS.items()}

    def frame(self, t: float, window: float = 0.0) -> np.ndarray:
        img = np.ones((self.h, self.w, 3))
        img[:, : self.ncols * self.tile_w + 1] = self._border
        for sm in range(self.n_sm):
            r0 = (sm // self.ncols) * self.tile_h + 1
            c0 = (sm % self.ncols) * self.tile_w + 1
            img[r0 : r0 + self.slot_rows, c0 : c0 + self.slot_cols] = self._idle
        for (sm, slot), evs in self.by_slot.items():
            state = None
            for e in evs:
                if e.t_start <= t < e.t_end:
                    state = e.state
                    break
            if state is None:
                continue
            i = self.slot_index[(sm, slot)]
            r0 = (sm // self.ncols) * self.tile_h + 1 + i // self.slot_cols
            c0 = (sm % self.ncols) * self.tile_w + 1 + i % self.slot_cols
            img[r0, c0] = self._state_rgb[state]
        # DRAM gauge: rightmost columns fill bottom-up with utilization
        w = window or max(1.0, self.trace.total_cycles / 50)
        util = _bw_utilization(self.trace, t - w, t + w)
        filled = int(round(util * (self.h - 2)))
        img[:, -2:] = self._idle
        if filled:
            img[self.h - 1 - filled : self.h - 1, -2:] = np.array(to_rgb(BW_COLOR))
        return img


def plot_floorplan(trace: Trace, t: float, ax=None, ncols: int = 8):
    """Snapshot of the die at cycle t: every SM's warp slots plus DRAM gauge."""
    fp = _FloorplanImage(trace, ncols=ncols)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9 * fp.h / fp.w))
    ax.imshow(fp.frame(t), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{trace.spec.name} die @ cycle {t:,.0f} / {trace.total_cycles:,.0f} "
        f"(right bar: DRAM pipe utilization)"
    )
    ax.legend(
        handles=[Patch(color=c, label=s) for s, c in STATE_COLORS.items()]
        + [Patch(color=IDLE_COLOR, label="idle slot")],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8,
    )
    return ax


def animate_floorplan(trace: Trace, frames: int = 40, ncols: int = 8, figsize=(9, 5)):
    """Animated die: watch block waves sweep the chip and the DRAM gauge move.

    Returns a matplotlib FuncAnimation; in a notebook, display it with
    `IPython.display.HTML(anim.to_jshtml())` (self-contained, works on the
    static site). Image-based redraw keeps this fast — see module docstring.
    """
    fp = _FloorplanImage(trace, ncols=ncols)
    fig, ax = plt.subplots(figsize=figsize)
    times = np.linspace(0, trace.total_cycles, frames)
    im = ax.imshow(fp.frame(0.0), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")
    ax.legend(
        handles=[Patch(color=c, label=s) for s, c in STATE_COLORS.items()]
        + [Patch(color=IDLE_COLOR, label="idle slot")],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8,
    )

    def update(i):
        im.set_data(fp.frame(times[i]))
        title.set_text(
            f"{trace.spec.name} die @ cycle {times[i]:,.0f} / {trace.total_cycles:,.0f}"
        )
        return im, title

    anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    plt.close(fig)  # the animation owns the figure; avoid a stray static plot
    return anim
