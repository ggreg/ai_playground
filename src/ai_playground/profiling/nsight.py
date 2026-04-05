"""Nsight Systems and Nsight Compute integration helpers.

Nsight Systems: timeline profiler — shows what the GPU is doing over time
    nsys profile -o trace python scripts/train.py --config configs/tiny.yaml

Nsight Compute: kernel-level profiler — deep dive into a single CUDA kernel
    ncu --set full python scripts/profile_model.py

These require the NVIDIA tools to be installed (part of CUDA Toolkit).
"""

from contextlib import contextmanager

import torch


@contextmanager
def nsight_range(name: str):
    """Mark a code region for Nsight Systems profiling.

    Shows up as a labeled range in the Nsight Systems timeline.
    Useful for identifying which part of training each GPU operation belongs to.

    Usage:
        with nsight_range("forward_pass"):
            output = model(input)
        with nsight_range("backward_pass"):
            loss.backward()
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def profile_with_torch(
    model: torch.nn.Module,
    input_fn,
    num_steps: int = 5,
    warmup_steps: int = 2,
    output_path: str = "profile_trace",
) -> torch.profiler.profile:
    """Profile model using PyTorch's built-in profiler.

    Generates a Chrome trace that can be viewed at chrome://tracing
    or in TensorBoard.

    Args:
        model: model to profile
        input_fn: callable that returns (input_tensor,) or (input, target)
        num_steps: number of profiled steps
        warmup_steps: number of warmup steps (not profiled)
        output_path: path prefix for trace output
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=warmup_steps,
            active=num_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(warmup_steps + num_steps):
            inputs = input_fn()
            if isinstance(inputs, tuple):
                output = model(inputs[0])
            else:
                output = model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()

    print(f"Profile saved to {output_path}/")
    print("\nTop CUDA kernels by time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    return prof
