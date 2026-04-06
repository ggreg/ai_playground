"""Quantization utilities for inference optimization.

Implements basic quantization schemes for learning purposes.
For production, use bitsandbytes, GPTQ, or AWQ.

See also: docs/PAPERS.md § Quantization
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- LLM.int8(): https://arxiv.org/abs/2208.07339
"""

import torch
import torch.nn as nn


def quantize_tensor_absmax(x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float]:
    """Absmax quantization: simplest symmetric quantization scheme.

    Maps float values to int range [-2^(bits-1), 2^(bits-1)-1]
    using a single scale factor per tensor.

    Scale = max(|x|) / (2^(bits-1) - 1)
    x_quant = round(x / scale)
    x_dequant = x_quant * scale

    This is what INT8 inference typically uses.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().max().item() / qmax
    x_quant = (x / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return x_quant, scale


def dequantize_tensor(x_quant: torch.Tensor, scale: float) -> torch.Tensor:
    return x_quant.float() * scale


def quantize_model_weights(model: nn.Module, bits: int = 8) -> dict:
    """Quantize all linear layer weights and report compression stats.

    Returns dict with original_mb, quantized_mb, and compression_ratio.
    This is weight-only quantization (W8A16) — weights are INT8 but
    activations stay in FP16/FP32 during computation.
    """
    original_bytes = 0
    quantized_bytes = 0
    quantized_weights = {}

    for name, param in model.named_parameters():
        original_bytes += param.numel() * param.element_size()
        if param.dim() >= 2:  # Only quantize weight matrices
            q, scale = quantize_tensor_absmax(param.data, bits)
            quantized_weights[name] = (q, scale)
            quantized_bytes += q.numel() * q.element_size() + 4  # +4 for scale
        else:
            quantized_bytes += param.numel() * param.element_size()

    mb = 1024 * 1024
    return {
        "original_mb": original_bytes / mb,
        "quantized_mb": quantized_bytes / mb,
        "compression_ratio": original_bytes / quantized_bytes,
        "num_quantized_tensors": len(quantized_weights),
        "quantized_weights": quantized_weights,
    }
