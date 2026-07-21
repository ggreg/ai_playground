"""M9 (session S5.2) — your model's decode roofline.

Certifies: the reader computed their own model's decode arithmetic intensity at batch
1/8/32 and located the bandwidth->compute crossover on the T4 roofline — the single
mental model behind every serving optimization in the book.
"""

from conftest import metric

INTENSITY_KEYS = ("intensity_b1", "intensity_b8", "intensity_b32")


def test_roofline_recorded(metrics):
    roof = metric(metrics, "m9_roofline", "S5.2")
    for key in INTENSITY_KEYS:
        assert key in roof and isinstance(roof[key], (int, float)) and roof[key] > 0, (
            f"m9_roofline needs '{key}' (FLOPs/byte, > 0)."
        )
    assert isinstance(roof.get("crossover_batch"), (int, float)), (
        "m9_roofline needs 'crossover_batch' — the smallest batch where decode goes "
        "compute-bound."
    )
    # Intensity grows with batch: same weights stream from DRAM, more tokens use them.
    assert roof["intensity_b1"] < roof["intensity_b8"] < roof["intensity_b32"], (
        "Arithmetic intensity must increase with batch size — if yours doesn't, the "
        "bytes/token term is probably being multiplied by batch."
    )
    assert isinstance(roof.get("conclusion"), str) and roof["conclusion"].strip(), (
        "m9_roofline needs a one-line 'conclusion' (hint: batch-1 decode is deeply "
        "bandwidth-bound)."
    )
