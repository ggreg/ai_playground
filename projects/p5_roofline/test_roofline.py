"""Acceptance tests for p5 — the decode roofline.

One anchor test (2 FLOPs per matmul weight vs cfg.num_params) plus internal-consistency
properties every correct roofline must satisfy — monotonicity, amortization, the min()
of the two ceilings, and the no-crossover punchline at real context lengths.
"""

from pathlib import Path

import pytest

import roofline
from ai_playground.models.config import TINY, TransformerConfig
from roofline import DTYPE_BYTES, GPU_SPECS

CTX = 512


def _kv_bytes_per_seq(cfg: TransformerConfig, ctx: int) -> float:
    return 2 * cfg.n_layers * cfg.kv_heads * cfg.head_dim * ctx * DTYPE_BYTES


def test_no_forbidden_imports():
    lines = (Path(__file__).parent / "roofline.py").read_text().splitlines()
    code = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    assert "ai_playground.profiling" not in code, "build it yourself — see README rules"


def test_flops_are_two_per_matmul_weight(attempt):
    for cfg in (TINY, TransformerConfig(dim=128, n_layers=4, n_heads=4, vocab_size=4096)):
        flops = attempt(roofline.decode_flops_per_token, cfg)
        anchor = 2 * cfg.num_params(include_embeddings=False)
        assert flops == pytest.approx(anchor, rel=0.05), (
            f"decode FLOPs/token {flops:.4e} vs 2 x matmul weights {anchor:.4e} — far "
            "below means a missing GEMM (the head is the big one); far above usually "
            "means the embedding table snuck in"
        )


def test_bytes_weights_plus_kv(attempt):
    b1 = attempt(roofline.decode_bytes_per_step, TINY, CTX, 1)
    b8 = attempt(roofline.decode_bytes_per_step, TINY, CTX, 8)
    weight_bytes = TINY.num_params(include_embeddings=False) * DTYPE_BYTES
    kv = _kv_bytes_per_seq(TINY, CTX)

    assert b1 == pytest.approx(weight_bytes + kv, rel=0.1), (
        f"batch-1 step: {b1:.4e} vs weights+KV {weight_bytes + kv:.4e} — remember the "
        "embedding table is NOT read in full (see the pinned conventions)"
    )
    assert (b8 - b1) / 7 == pytest.approx(kv, rel=0.1), (
        "each extra sequence should add ~its own KV cache; weights are read once for "
        "the whole batch, not per sequence"
    )
    assert attempt(roofline.decode_bytes_per_step, TINY, 2 * CTX, 8) > b8, (
        "longer context means more K/V bytes"
    )


def test_intensity_grows_with_batch(attempt):
    vals = [attempt(roofline.arithmetic_intensity, TINY, CTX, b) for b in (1, 4, 16, 64)]
    assert all(a < b for a, b in zip(vals, vals[1:])), (
        f"arithmetic intensity must grow with batch (weight reads amortize): {vals}"
    )


def test_prediction_is_min_of_both_ceilings(attempt):
    for gpu in GPU_SPECS.values():
        for batch in (1, 8, 64):
            flops = attempt(roofline.decode_flops_per_token, TINY) * batch
            bytes_ = attempt(roofline.decode_bytes_per_step, TINY, CTX, batch)
            compute_ceiling = batch * (gpu["tflops"] * 1e12) / flops
            bandwidth_ceiling = batch * (gpu["gb_per_sec"] * 1e9) / bytes_
            want = min(compute_ceiling, bandwidth_ceiling)
            got = attempt(roofline.predict_decode_tokens_per_sec, TINY, gpu, CTX, batch)
            assert got == pytest.approx(want, rel=0.05), (
                f"batch {batch}: prediction {got:.3e} != min(compute {compute_ceiling:.3e}, "
                f"bandwidth {bandwidth_ceiling:.3e}) from your own flops/bytes functions"
            )


def test_crossover_with_no_kv(attempt):
    # context 0 (hypothetical no-cache decode): weights amortize forever, so a
    # crossover batch must exist and sit exactly where intensity crosses the ridge.
    gpu = GPU_SPECS["T4"]
    ridge = (gpu["tflops"] * 1e12) / (gpu["gb_per_sec"] * 1e9)
    b = attempt(roofline.crossover_batch, TINY, gpu, 0)
    assert isinstance(b, int) and b > 1, f"expected a finite crossover batch > 1, got {b}"
    assert attempt(roofline.arithmetic_intensity, TINY, 0, b) >= ridge
    assert attempt(roofline.arithmetic_intensity, TINY, 0, b - 1) < ridge
    # Batch-1 decode on a T4 is emphatically bandwidth-bound (the M9 conclusion).
    assert attempt(roofline.arithmetic_intensity, TINY, CTX, 1) < ridge
    # More TFLOPS at similar bandwidth-ratio -> needs a bigger batch to saturate.
    assert attempt(roofline.crossover_batch, TINY, GPU_SPECS["H100_SXM"], 0) >= b


def test_no_crossover_at_real_context(attempt):
    # The punchline: at ctx 512 the KV asymptote (flops/token / kv_bytes/seq ~ 15
    # FLOPs/byte for TINY) is far below the T4 ridge (~217) — no batch size makes
    # decode compute-bound. KV bytes, not FLOPs, rule serving.
    assert attempt(roofline.crossover_batch, TINY, GPU_SPECS["T4"], CTX) is None
