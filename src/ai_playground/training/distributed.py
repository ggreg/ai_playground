"""Distributed training helpers for DDP and FSDP.

Usage with torchrun:
    torchrun --nproc_per_node=4 scripts/train.py --config configs/small.yaml --distributed ddp
"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed process group.

    Returns: (rank, local_rank, world_size)
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


def wrap_ddp(model: nn.Module, local_rank: int) -> DDP:
    """Wrap model with DistributedDataParallel."""
    return DDP(model.to(local_rank), device_ids=[local_rank])


def wrap_fsdp(model: nn.Module, local_rank: int) -> nn.Module:
    """Wrap model with Fully Sharded Data Parallel (FSDP).

    FSDP shards parameters, gradients, and optimizer states across GPUs.
    This enables training models that don't fit in a single GPU's memory.

    Equivalent to ZeRO Stage 3 in DeepSpeed.

    Papers:
    - PyTorch FSDP: https://arxiv.org/abs/2304.11277
    - ZeRO: https://arxiv.org/abs/1910.02054
    See also: docs/PAPERS.md § Distributed Training
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from functools import partial

    # Import the block class for auto-wrapping
    from ..models.transformer import TransformerBlock

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    return FSDP(
        model.to(local_rank),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=local_rank,
    )
