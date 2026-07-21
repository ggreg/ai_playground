"""M13b (session S4.4) — paged equals reference, on your weights.

Certifies: greedy decoding through the reader's paged KV cache produces exactly the
same tokens as a full-recompute reference, for 3 prompts x 20 tokens. This is the
chapter's toy-model correctness assert, earned on the reader's own model — the proof
that the paging machinery changes memory layout and nothing else.
"""

import pytest


def _fn(name, session):
    try:
        import serve_myllm
    except ImportError:
        pytest.skip("checkpoints/myllm/src/serve_myllm.py not found — run session S4.3 first.")
    if not hasattr(serve_myllm, name):
        pytest.skip(f"serve_myllm.{name} not written yet — run session {session}.")
    return getattr(serve_myllm, name)


def _prompts(vocab_size):
    raw = [
        [1, 2, 3, 4, 5],
        [7, 8, 9],
        [11, 12, 13, 14, 15, 16, 17],
    ]
    return [[t % vocab_size for t in p] for p in raw]


def test_paged_matches_reference(ws, reader_config):
    greedy_paged = _fn("greedy_paged", "S4.4")
    greedy_reference = _fn("greedy_reference", "S4.4")
    for prompt in _prompts(reader_config.vocab_size):
        paged = greedy_paged(prompt, 20)
        reference = greedy_reference(prompt, 20)
        assert list(paged) == list(reference), (
            f"Paged and reference tokens diverge for prompt {prompt}:\n"
            f"  paged:     {list(paged)}\n"
            f"  reference: {list(reference)}\n"
            "First divergence index is where to start debugging — usually RoPE "
            "positions or a block-table off-by-one."
        )
