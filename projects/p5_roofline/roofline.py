"""p5 — your decode roofline calculator. See README.md for the brief and rules (short
version: the repo's profiling package is the after-the-fact comparison, not an import).

Conventions, pinned so the tests are unambiguous:
- fp16/bf16 everywhere: 2 bytes per parameter and per K/V element.
- "decode" = one forward pass of ONE new token per sequence in the batch.
- "weights" = every parameter that participates in a matmul: attention projections,
  FFN, norms, output head — but NOT the token-embedding table (decode reads one row of
  it, which is negligible). That's `cfg.num_params(include_embeddings=False)`.
- Attention-score FLOPs (q @ cached K, att @ cached V) are excluded: at decode they are
  tiny next to the GEMMs. K/V cache BYTES are emphatically not excluded — reading the
  cache is where decode's bandwidth goes.
- K/V cache per token per layer = 2 (K and V) * kv_heads * head_dim elements.
"""

from ai_playground.models import TransformerConfig  # noqa: F401  (configs are given)

# ---------------------------------------------------------------- SCAFFOLDING (given) --
# Peak dense fp16/bf16 tensor-core throughput and memory bandwidth.
GPU_SPECS = {
    "T4": {"tflops": 65, "gb_per_sec": 300},
    "A100_80GB": {"tflops": 312, "gb_per_sec": 1935},
    "H100_SXM": {"tflops": 989, "gb_per_sec": 3350},
}

DTYPE_BYTES = 2

# ------------------------------------------------------------------- YOUR CODE (build) --


def decode_flops_per_token(cfg: TransformerConfig) -> float:
    """FLOPs to decode one token for one sequence: 2 FLOPs per matmul weight, summed
    over the GEMMs (QKV/O projections, SwiGLU FFN, output head) across all layers.
    Should land within a few % of 2 * cfg.num_params(include_embeddings=False)."""
    raise NotImplementedError


def decode_bytes_per_step(cfg: TransformerConfig, context_len: int, batch_size: int) -> float:
    """Bytes read from HBM for ONE decode step of a whole batch: all matmul weights
    once (shared across the batch) + each sequence's K/V cache at context_len."""
    raise NotImplementedError


def arithmetic_intensity(cfg: TransformerConfig, context_len: int, batch_size: int) -> float:
    """FLOPs per byte for one decode step of the whole batch."""
    raise NotImplementedError


def predict_decode_tokens_per_sec(
    cfg: TransformerConfig, gpu: dict, context_len: int, batch_size: int
) -> float:
    """Roofline prediction: whole-batch tokens/sec under
    min(compute ceiling, bandwidth ceiling) for the gpu spec dict."""
    raise NotImplementedError


def crossover_batch(cfg: TransformerConfig, gpu: dict, context_len: int) -> int | None:
    """Smallest batch size at which decode becomes compute-bound on this gpu (i.e.
    arithmetic intensity reaches the gpu's FLOPs-per-byte ridge point).

    Careful: as batch grows, intensity approaches flops_per_token / kv_bytes_per_seq —
    if that asymptote is below the ridge, NO batch is compute-bound: return None.
    (At real context lengths this is the common case, and the reason KV-cache bytes,
    not FLOPs, rule serving. Discovering that is part of the project.)"""
    raise NotImplementedError


if __name__ == "__main__":
    from ai_playground.models.config import TINY

    for name, gpu in GPU_SPECS.items():
        tps = predict_decode_tokens_per_sec(TINY, gpu, context_len=512, batch_size=8)
        print(f"{name}: TINY @ ctx 512, batch 8 -> {tps:,.0f} tok/s, "
              f"crossover batch {crossover_batch(TINY, gpu, 512)}")
