"""gpusim — a pure-Python SIMT simulator for learning CUDA semantics.

Runs numba.cuda-style kernels on NumPy with full memory-access logging.
No GPU, no compiler, no native dependencies: works in CI, on any laptop,
and under Pyodide in the browser. See core.py for the execution model and
memory.py for the coalescing / bank-conflict analyses.
"""

from .core import (
    WARP_SIZE,
    BarrierDivergenceError,
    Ctx,
    Dim3,
    Kernel,
    LaunchReport,
    kernel,
)
from .memory import (
    AccessLog,
    GlobalArray,
    SharedArray,
    WarpAccessGroup,
    bank_conflict_report,
    coalescing_report,
    total_transactions,
)
from .occupancy import Occupancy, occupancy, occupancy_sweep
from .spec import SPECS, A100_40GB, T4, GPUSpec
from .viz import plot_warp_accesses

__all__ = [
    "SPECS",
    "A100_40GB",
    "T4",
    "GPUSpec",
    "Occupancy",
    "occupancy",
    "occupancy_sweep",
    "WARP_SIZE",
    "BarrierDivergenceError",
    "Ctx",
    "Dim3",
    "Kernel",
    "LaunchReport",
    "kernel",
    "AccessLog",
    "GlobalArray",
    "SharedArray",
    "WarpAccessGroup",
    "bank_conflict_report",
    "coalescing_report",
    "total_transactions",
    "plot_warp_accesses",
]
