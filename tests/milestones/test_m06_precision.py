"""M6 (session S2.5) — the precision measurement, on whatever hardware exists.

Certifies: the reader measured (GPU) or baseline-recorded (CPU) their model's training
throughput per precision and adopted one. The discipline being certified is recording
numbers before adopting an optimization — not owning a GPU.
"""

from conftest import metric


def test_precision_measured_and_adopted(metrics, reader_config_raw):
    m6 = metric(metrics, "m6_precision", "S2.5")
    assert isinstance(m6, dict) and m6, "m6_precision must be a non-empty measurement dict."
    numeric = [v for v in m6.values() if isinstance(v, (int, float))]
    assert numeric, (
        "m6_precision needs at least one measured number (tokens/sec fp32 at minimum; "
        "bf16 too if on GPU)."
    )
    assert reader_config_raw.get("precision") in ("fp32", "fp16", "bf16"), (
        'config.json must record the adopted precision ("precision": "bf16" or "fp32").'
    )
