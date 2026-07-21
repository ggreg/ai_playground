"""M12 (session S5.8) — your model's decode GEMM shapes.

Certifies: the reader enumerated the actual matrix multiplies one decode step of their
model performs (projections, FFN, output head) at batch 1/8/32 — the shape table that
makes M9's "batch-1 is bandwidth-bound" conclusion concrete. Timings are required only
on GPU; the shape table is the CPU-fallback minimum.
"""

from conftest import metric


def test_gemm_shapes_recorded(metrics):
    gemms = metric(metrics, "m12_gemms", "S5.8")
    assert isinstance(gemms, (dict, list)) and len(gemms) > 0, (
        "m12_gemms must be a non-empty shape table (dict or list) covering the "
        "attention projections, FFN, and output head."
    )
