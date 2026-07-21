"""M13a (session S4.3) — your weights in the engine.

Certifies: the reader mapped every weight of their release checkpoint onto the
mini-vLLM engine's layer structure in an importable module
(checkpoints/myllm/src/serve_myllm.py) — the finale runs on code the tests can call,
not on cells only a notebook kernel has seen.
"""

import pytest

from conftest import WORKSPACE


def _serve_module():
    try:
        import serve_myllm
    except ImportError:
        pytest.skip(
            "checkpoints/myllm/src/serve_myllm.py not found — run session S4.3 first."
        )
    return serve_myllm


def test_load_model(ws):
    serve_myllm = _serve_module()
    if not hasattr(serve_myllm, "load_model"):
        pytest.skip("serve_myllm.load_model not written yet — run session S4.3.")
    model = serve_myllm.load_model(WORKSPACE)
    assert model is not None, "load_model must return the mapped model, not None."
