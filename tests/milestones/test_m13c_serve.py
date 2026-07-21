"""M13c (session S4.5) — continuous batching under memory pressure.

Certifies: the reader's scheduler pushes a 16-request workload through continuous
batching with a block pool deliberately smaller than worst-case — every request
completes AND preemption fires at least once, so the engine demonstrably survives the
condition real serving systems live in.
"""

import pytest


def _serve():
    try:
        import serve_myllm
    except ImportError:
        pytest.skip("checkpoints/myllm/src/serve_myllm.py not found — run session S4.3 first.")
    if not hasattr(serve_myllm, "serve"):
        pytest.skip("serve_myllm.serve not written yet — run session S4.5.")
    return serve_myllm.serve


def test_serve_sixteen_requests_with_preemption(ws, reader_config):
    serve = _serve()
    vocab = reader_config.vocab_size
    requests = [[(i * 7 + j) % vocab for j in range(3 + i % 5)] for i in range(16)]
    # block_budget=None -> the tight default chosen in session S4.5 to force preemption.
    report = serve(requests, max_new=16, block_budget=None)
    assert isinstance(report, dict), "serve must return a report dict."
    assert report.get("completed") == 16, (
        f"All 16 requests must complete; report says {report.get('completed')!r}."
    )
    assert report.get("preemptions", 0) >= 1, (
        "No preemption fired — shrink the block pool below worst-case so the "
        "scheduler has to evict and recompute at least once."
    )
    outputs = report.get("outputs")
    assert isinstance(outputs, list) and len(outputs) == 16, (
        "report['outputs'] must hold all 16 completed token lists."
    )
