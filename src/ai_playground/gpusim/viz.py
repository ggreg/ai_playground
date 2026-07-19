"""Visualization for simulated memory traffic.

One picture explains coalescing better than any prose: plot which byte
segments each lane of a warp touches. Coalesced access is a tight vertical
stripe (all 32 lanes inside one or two 128-byte segments); strided access is
a diagonal smear crossing 32 segments.
"""

import matplotlib.pyplot as plt
import numpy as np

from .memory import SEGMENT_BYTES, AccessLog


def plot_warp_accesses(
    log: AccessLog,
    array: str,
    warp: int = 0,
    kind: str = "r",
    max_seqs: int = 8,
    ax=None,
):
    """Scatter lanes (y) vs byte address (x) with 128-byte segment gridlines.

    Each color is one warp-level access (one `seq` step); the number of
    distinct vertical bands a color crosses is its transaction count.
    """
    accs = [
        a
        for a in log.accesses
        if a.space == "global" and a.array == array and a.warp == warp and a.kind == kind
    ]
    if not accs:
        raise ValueError(f"no logged '{kind}' accesses to {array!r} by warp {warp}")
    seqs = sorted({a.seq for a in accs})[:max_seqs]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    for i, seq in enumerate(seqs):
        group = [a for a in accs if a.seq == seq]
        xs = [a.flat * a.itemsize for a in group]
        ys = [a.thread % 32 for a in group]
        n_seg = len({x // SEGMENT_BYTES for x in xs})
        ax.scatter(xs, ys, s=14, label=f"access {seq}: {n_seg} txn", color=f"C{i}")

    lo = min(a.flat * a.itemsize for a in accs if a.seq in seqs)
    hi = max(a.flat * a.itemsize for a in accs if a.seq in seqs)
    for x in np.arange(lo // SEGMENT_BYTES * SEGMENT_BYTES, hi + SEGMENT_BYTES, SEGMENT_BYTES):
        ax.axvline(x, lw=0.5, alpha=0.5, zorder=0)
    ax.set_xlabel(f"byte address in {array} (gridlines: {SEGMENT_BYTES}B segments)")
    ax.set_ylabel(f"lane (warp {warp})")
    ax.set_title(f"{array}: memory transactions per warp access")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    return ax
