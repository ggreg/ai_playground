"""GPU memory tracking and analysis.

Understanding memory usage is critical for:
- Choosing batch sizes and sequence lengths
- Deciding between DDP vs FSDP
- Estimating KV cache requirements for serving
"""

from contextlib import contextmanager

import torch


class MemoryTracker:
    """Track GPU memory allocation over time.

    Usage:
        tracker = MemoryTracker()
        tracker.snapshot("before_forward")
        output = model(input)
        tracker.snapshot("after_forward")
        tracker.report()
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.snapshots: list[tuple[str, dict]] = []

    def snapshot(self, label: str):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        self.snapshots.append((label, {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }))

    def report(self):
        if not self.snapshots:
            print("No snapshots recorded (CUDA not available?)")
            return

        print(f"\n{'Label':<30} {'Allocated':>12} {'Reserved':>12} {'Peak':>12}")
        print("-" * 70)
        for label, stats in self.snapshots:
            print(
                f"{label:<30} "
                f"{stats['allocated_mb']:>10.1f}MB "
                f"{stats['reserved_mb']:>10.1f}MB "
                f"{stats['max_allocated_mb']:>10.1f}MB"
            )

    def delta(self, label_a: str, label_b: str) -> float:
        """Return memory difference (MB) between two snapshots."""
        a = next(s for l, s in self.snapshots if l == label_a)
        b = next(s for l, s in self.snapshots if l == label_b)
        return b["allocated_mb"] - a["allocated_mb"]


def print_memory_stats(device: str = "cuda"):
    """Print current GPU memory statistics."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    total = torch.cuda.get_device_properties(device).total_mem / 1024**2

    print(f"GPU Memory: {allocated:.1f}MB allocated / {reserved:.1f}MB reserved / "
          f"{max_allocated:.1f}MB peak / {total:.1f}MB total")
    print(f"Utilization: {allocated / total * 100:.1f}%")


@contextmanager
def track_memory(label: str = "operation"):
    """Context manager to measure memory delta of an operation."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()

    yield

    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    mb = 1024**2

    print(f"[{label}] Delta: {(after - before) / mb:+.1f}MB, "
          f"Peak: {peak / mb:.1f}MB, Current: {after / mb:.1f}MB")
