"""FLOP counting and Model FLOP Utilization (MFU) for transformers.

MFU = achieved_flops / theoretical_peak_flops

This is the gold standard metric for how efficiently you're using
your GPU. State-of-the-art training achieves ~40-60% MFU.

References:
- Gopher: https://arxiv.org/abs/2112.11446
- Chinchilla scaling laws: https://arxiv.org/abs/2203.15556
See also: docs/PAPERS.md § Training Optimization
"""

from ..models.config import TransformerConfig

# Peak FLOPS for common GPUs (FP16/BF16 tensor core throughput)
GPU_PEAK_TFLOPS = {
    "A100_40GB": 312,
    "A100_80GB": 312,
    "H100_SXM": 989,
    "H100_PCIe": 756,
    "H200": 989,
    "L40S": 362,
    "A10G": 125,
    "T4": 65,
    "V100": 125,
    "RTX_4090": 330,
    "RTX_3090": 142,
}


def estimate_flops(config: TransformerConfig, seq_len: int, batch_size: int) -> dict:
    """Estimate FLOPs for a single forward+backward pass.

    Uses the approximation from the Chinchilla paper (Hoffmann et al., 2022):
    Forward pass ≈ 2 * N * T FLOPs (where N = params, T = tokens)
    Backward pass ≈ 2x forward
    Paper: https://arxiv.org/abs/2203.15556

    More detailed breakdown:
    - Attention QKV projection: 2 * B * T * (3 * d_model * d_model)  [for MHA]
    - Attention scores: 2 * B * T * T * d_model
    - Attention output projection: 2 * B * T * d_model * d_model
    - FFN (SwiGLU): 2 * B * T * (3 * d_model * d_ffn)
    - Per layer, then multiply by n_layers
    """
    B, T = batch_size, seq_len
    D = config.dim
    L = config.n_layers
    H = config.ffn_hidden_dim
    V = config.vocab_size
    kv_dim = config.kv_heads * config.head_dim

    # Per-layer FLOPs (forward pass)
    attn_qkvo = 2 * B * T * (D * D + 2 * D * kv_dim + D * D)  # Q, K, V, O projections
    attn_scores = 2 * B * T * T * D  # Q @ K^T and attn @ V
    ffn = 2 * B * T * 3 * D * H  # SwiGLU: gate, up, down

    per_layer = attn_qkvo + attn_scores + ffn
    all_layers = L * per_layer

    # Embedding + output projection
    embedding = 2 * B * T * V * D

    forward_flops = all_layers + embedding
    # Backward ≈ 2x forward
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops

    return {
        "forward_tflops": forward_flops / 1e12,
        "backward_tflops": backward_flops / 1e12,
        "total_tflops": total_flops / 1e12,
        "per_layer_tflops": per_layer / 1e12,
        "attention_fraction": (attn_qkvo + attn_scores) / per_layer,
        "ffn_fraction": ffn / per_layer,
    }


def compute_mfu(
    config: TransformerConfig,
    seq_len: int,
    batch_size: int,
    step_time_sec: float,
    gpu_name: str = "A100_80GB",
    n_gpus: int = 1,
) -> dict:
    """Compute Model FLOP Utilization.

    MFU tells you what fraction of the GPU's theoretical peak
    throughput you're actually achieving. This is THE metric
    for training efficiency.

    Common MFU values:
    - Naive PyTorch: ~20-30%
    - With Flash Attention + compile: ~40-50%
    - Highly optimized (Megatron-LM): ~50-60%
    """
    flops = estimate_flops(config, seq_len, batch_size)
    achieved_tflops_per_sec = flops["total_tflops"] / step_time_sec

    peak_tflops = GPU_PEAK_TFLOPS.get(gpu_name, 312)
    total_peak = peak_tflops * n_gpus

    mfu = achieved_tflops_per_sec / total_peak

    return {
        "achieved_tflops_per_sec": achieved_tflops_per_sec,
        "peak_tflops": total_peak,
        "mfu": mfu,
        "mfu_percent": mfu * 100,
        "gpu": gpu_name,
        "n_gpus": n_gpus,
    }
