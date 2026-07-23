"""Acceptance tests for p3 — KV-cached decoding.

The bar is the one real serving engines use: the cached path must be token-for-token
identical to full recompute. A forward-call counter keeps the cached path honest.
"""

import math
from pathlib import Path

import torch

import kv_generate
from given_model import MAX_SEQ, VOCAB, tiny_model

PROMPTS = [[1, 2, 3, 4], [5], [7, 8, 9, 10, 11, 12]]
N_NEW = 12


def test_no_forbidden_imports():
    lines = (Path(__file__).parent / "kv_generate.py").read_text().splitlines()
    code = "\n".join(ln for ln in lines if not ln.lstrip().startswith("#"))
    assert "ai_playground.inference" not in code, "build it yourself — see README rules"


def test_scaffold_model_is_deterministic():
    a, b = tiny_model(), tiny_model()
    ids = torch.randint(0, VOCAB, (2, 10), generator=torch.Generator().manual_seed(1))
    assert torch.equal(a(ids), b(ids))
    assert a(ids).shape == (2, 10, VOCAB)


def test_generate_full_basics(attempt):
    model = tiny_model()
    out = attempt(kv_generate.generate_full, model, PROMPTS[0], N_NEW)
    assert out[: len(PROMPTS[0])] == PROMPTS[0], "output must start with the prompt"
    assert len(out) == len(PROMPTS[0]) + N_NEW
    assert all(isinstance(t, int) and 0 <= t < VOCAB for t in out)
    assert out == attempt(kv_generate.generate_full, model, PROMPTS[0], N_NEW), (
        "greedy decoding must be deterministic"
    )


def test_cached_matches_full(attempt):
    model = tiny_model()
    for prompt in PROMPTS:
        assert len(prompt) + N_NEW <= MAX_SEQ
        full = attempt(kv_generate.generate_full, model, prompt, N_NEW)
        cached = attempt(kv_generate.generate_cached, model, prompt, N_NEW)
        assert cached == full, (
            f"prompt {prompt}: cached {cached[len(prompt):]} != full {full[len(prompt):]}. "
            "First divergent position is where to look — off-by-one in pos_emb indices "
            "and stale ln inputs are the classic causes."
        )


def test_cached_does_not_recompute(attempt):
    model = tiny_model()
    calls = 0
    original = model.forward

    def counting_forward(ids):
        nonlocal calls
        calls += 1
        return original(ids)

    model.forward = counting_forward
    attempt(kv_generate.generate_cached, model, PROMPTS[0], N_NEW)
    assert calls <= 1, (
        f"generate_cached called the full model forward {calls} times for {N_NEW} new "
        "tokens — one full-sequence call (prefill) is allowed, per-step recompute is not."
    )


def test_top_p_sampling(attempt):
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log()

    g = torch.Generator().manual_seed(0)
    draws = [attempt(kv_generate.top_p_sample, logits, 0.7, g) for _ in range(300)]
    assert set(draws) <= {0, 1}, (
        f"p=0.7 nucleus over probs {probs.tolist()} is tokens {{0, 1}}, but sampled "
        f"{sorted(set(draws))}"
    )
    assert set(draws) == {0, 1}, "both nucleus tokens should appear in 300 draws"
    frac0 = draws.count(0) / len(draws)
    assert math.isclose(frac0, 0.5 / 0.8, abs_tol=0.12), (
        f"token 0 should appear with renormalized prob {0.5 / 0.8:.3f}, got {frac0:.3f}"
    )

    g = torch.Generator().manual_seed(1)
    draws = [attempt(kv_generate.top_p_sample, logits, 1.0, g) for _ in range(500)]
    assert set(draws) == {0, 1, 2, 3}, "p=1.0 must be able to sample every token"
